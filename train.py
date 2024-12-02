import os
from pathlib import Path
import time
import torch
import wandb
import traceback

import torch.distributed as dist
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn import DataParallel
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from einops import rearrange

from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import math

from dataset import VLMDataset
from model import MultimodalCaptionGenerator
from config import VisionConfig, MultimodalConfig
from preprocess import get_tokenizer

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from contextlib import contextmanager
import torch._dynamo

@contextmanager
def no_compile():
    torch._dynamo.config.disable = True 
    try:
        yield
    finally:
        torch._dynamo.config.disable = False

def compute_cider(candidate, reference):
    
    def preprocess(sent):

        return sent.lower().strip().split()
    
    def get_ngrams(tokens, n):

        ngrams = Counter()

        for i in range(len(tokens) - n + 1): # creating n-grams

            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1

        return ngrams
    
    candidate = preprocess(candidate)
    reference = preprocess(reference)
    
    n_scores = []

    for n in range(1, 5):  

        cand_ngrams = get_ngrams(candidate, n)
        ref_ngrams = get_ngrams(reference, n)
        
        # calculating term frequency
        cand_len = sum(cand_ngrams.values()) or 1 
        ref_len = sum(ref_ngrams.values()) or 1
        

        # normalizing
        cand_tf = {k: v/cand_len for k, v in cand_ngrams.items()}
        ref_tf = {k: v/ref_len for k, v in ref_ngrams.items()}
        
        # since there is only 1 reference, IDF is log (1/2)
        all_ngrams = set(cand_ngrams.keys()) | set(ref_ngrams.keys())
        idfs = {ngram: math.log(1.0 / 2) for ngram in all_ngrams} 
        
        # computing vectors
        def to_vec(tf_dict):

            vec = np.zeros(len(idfs))

            for i, (ngram, idf) in enumerate(idfs.items()):

                if ngram in tf_dict:
                    vec[i] = tf_dict[ngram] * idf

            return vec
        
        cand_vec = to_vec(cand_tf)
        ref_vec = to_vec(ref_tf)
        
        # computing cosine similarity
        norm_cand = np.linalg.norm(cand_vec)
        norm_ref = np.linalg.norm(ref_vec)
        
        if norm_cand == 0 or norm_ref == 0:
            score = 0
        else:
            score = np.dot(cand_vec, ref_vec) / (norm_cand * norm_ref)
        
        n_scores.append(score * 10)  # same as the paper
    
    return np.mean(n_scores)

def validate(epoch, model, val_loader, tokenizer, device, num_img_tokens, config, caption_table, artifact_dir, base_temperature=0.7, temp_decay=0.95, min_temperature=0.3):

    with no_compile():
        
        model.eval()
        val_metrics = defaultdict(float)
        num_batches = len(val_loader)
        fixed_batch_idx = config.fixed_batch_idx
        fixed_sample_idx = config.fixed_sample_idx

        pbar = tqdm(total=num_batches, desc=f'Val Epoch {epoch}')
        
        with torch.no_grad():

            for batch_idx, batch in enumerate(val_loader):

                images = batch['image'].to(device)
                tokens = batch['tokens'].to(device)
                
                logits, _ = model(tokens, images)
                labels = tokens[:, 1:].contiguous()
                logits = logits[:, config.num_image_tokens:-1, :].contiguous()
                
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    label_smoothing=0.10,
                    ignore_index=config.padding_idx
                )
                
                val_metrics['loss'] += loss.item()
                current_val_loss = val_metrics['loss'] / (batch_idx + 1)

                pbar.update(1)
                pbar.set_postfix({
                    'val_loss': f'{current_val_loss:.6f}',
                    'gpu_mem': f'{get_gpu_memory():.2f}GB'
                })

                if batch_idx == fixed_batch_idx and epoch%5==0:

                    epoch_dir = os.path.join(artifact_dir, f'epoch_{epoch}')
                    Path(epoch_dir).mkdir(parents=True, exist_ok=True)

                    image = batch['image'][fixed_sample_idx:fixed_sample_idx+1].to(device)
                    label = batch['tokens'][fixed_sample_idx:fixed_sample_idx+1].to(device)
                    tokens = torch.tensor([[config.sos_token_id]]).to(device)
                    generated_tokens = [config.sos_token_id]
                    attention_maps = []
                    
                    print(f"Saving artifacts...")

                    for i in range(config.max_len):

                        temperature = max(base_temperature * (temp_decay ** i), min_temperature)

                        logits, attention_scores = model(tokens, image)
                        logits = logits[0, -1] / temperature
                        top_k = 50

                        if len(generated_tokens) > 1:

                            for prev_token in generated_tokens:
                                logits[prev_token] /= 3.0 # Adding penalty for repitition

                        top_k_logits, top_k_indices = torch.topk(logits, top_k)
                        probs = torch.softmax(top_k_logits, dim=-1)
                        next_token_idx = torch.multinomial(probs, num_samples=1)
                        next_token = top_k_indices[next_token_idx] # Instead of greedy sampling, we do top-k beam sampling
                        generated_tokens.append(next_token.item())
                        attention_maps.append(attention_scores[-1].cpu()) # We take the attention map of only the first element of the batch

                        if next_token.item() == config.eos_token_id:
                            break
                        
                        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
                    
                    actual_tokens = []
                    for token in label[0].cpu().tolist(): 
                        if token == config.eos_token_id:
                            break
                        if token not in [config.padding_idx, config.sos_token_id]:
                            actual_tokens.append(token)
                    
                    caption = tokenizer.decode(generated_tokens[1:-1])  # removing SOS and EOS
                    print(f"The generated Caption: {caption}")
                    act_caption = tokenizer.decode(actual_tokens)
                    caption_table.add_data(epoch, caption, act_caption)
                    wandb.log({
                        f'val/caption': caption_table
                    })

                    orig_image = batch['image'][fixed_sample_idx].cpu()
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    orig_image = orig_image * std + mean
                    orig_image = orig_image.permute(1, 2, 0).numpy()
                    
                    for idx, (token_id, attention_map) in enumerate(zip(generated_tokens[1:], attention_maps)):

                        if token_id == config.eos_token_id:
                            break
                        token_text = tokenizer.decode([token_id])
                        
                        num_heads = attention_map.size(0)
                        # if len(attention_maps)==num_heads:
                        for head in range(num_heads):
                            # we take the attention_score for the image tokens of the last token added
                            token_attention = attention_map[head][-1, :num_img_tokens] 
                            h = w = int(np.sqrt(num_img_tokens))
                            token_attention = rearrange(token_attention, '(h w) -> h w', h=h, w=w)

                            token_attention = token_attention.clone().detach().unsqueeze(0).unsqueeze(0)  # add batch and channel dims [1, 1, 14, 14]
                            # We upsample the attention map from 14x14 -> 224x224. Note that we have 14x14 patches and attention score for each patch
                            img_attention_map = F.interpolate(token_attention, size=(224, 224), mode='bilinear', align_corners=False)
                            img_attention_map = img_attention_map.squeeze().numpy() 
                            
                            plt.figure(figsize=(10, 10))
                            plt.imshow(orig_image)
                            plt.imshow(img_attention_map, alpha=0.5, cmap='viridis')
                            plt.title(f'{token_text} - Head {head}')
                            plt.axis('off')
                            safe_token_text = token_text.replace('/', '_').replace('\\', '_')
                            plt.savefig(
                                os.path.join(epoch_dir, f'{idx}_{safe_token_text}_head_{head}.png'),
                                bbox_inches='tight',
                                pad_inches=0
                            )
                            plt.close()
                        # else:
                        #     print(f"Attention Maps: {len(attention_maps)}, num_heads: {num_heads}")

    pbar.close()
    val_metrics['loss'] /= num_batches
    wandb.log({
        'val/loss': val_metrics['loss'],
        'epoch': epoch
    })

    return val_metrics


def get_gpu_memory():
    return torch.cuda.memory_allocated() / 1e9

def train_epoch(epoch, model, dataloader, val_loader, optimizer, scheduler, scaler, device, global_step, config, tokenizer, artifact_dir):

    model.train()
    epoch_metrics = defaultdict(float)
    epoch_start_time = time.time()
    num_batches = len(dataloader)
    start_mem = get_gpu_memory()
    
    pbar = tqdm(total=num_batches, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(dataloader):
        batch_start_time = time.time()
        
        images = batch['image'].to(device)
        tokens = batch['tokens'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            logits, _ = model(tokens, images)
            labels = tokens[:, 1:].contiguous()
            logits = logits[:, config.num_image_tokens:-1, :].contiguous()
            
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), # reshaping to [batch_sz*seq_len, vocab_size]
                labels.view(-1), # reshaping to [batch_sz*seq_len]
                label_smoothing=0.10,
                ignore_index=config.padding_idx
            )
        
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']

        batch_time = time.time() - batch_start_time
        current_mem = get_gpu_memory()
    
        pbar.update(1)
        pbar.set_postfix({
            'train_loss': f'{loss.item():.6f}', 
            'gpu_mem': f'{current_mem:.2f}GB',
            'learning_rate': f'{current_lr:.6f}'
        })

        epoch_metrics['loss'] += loss.item()
        epoch_metrics['batch_time'] += batch_time
        epoch_metrics['gpu_memory'] += current_mem
        epoch_metrics['learning_rate'] = current_lr
    pbar.close()

    val_metrics = None

    if val_loader:

        caption_table = wandb.Table(columns=["epoch", "generated_caption", "actual_caption"])

        val_metrics = validate(
                            epoch= epoch,
                            model= model,
                            val_loader= val_loader,
                            tokenizer= tokenizer,
                            device=device,
                            num_img_tokens= config.num_image_tokens,
                            config= config,
                            caption_table= caption_table,
                            artifact_dir= artifact_dir
                    )
 
    
    
    global_step += 1
    
    wandb.log({
            'batch/loss': loss.item(),
            'batch/learning_rate': current_lr,
            'batch/time_per_batch': batch_time,
            'batch/gpu_memory': current_mem,
            'global_step': global_step
    })
    
    epoch_metrics['loss'] /= num_batches
    epoch_metrics['batch_time'] /= num_batches
    epoch_metrics['gpu_memory'] /= num_batches
    epoch_metrics['epoch_time'] = time.time() - epoch_start_time
    epoch_metrics['peak_memory'] = torch.cuda.max_memory_allocated() / 1e9
    epoch_metrics['memory_increase'] = get_gpu_memory() - start_mem
    epoch_metrics['global_step'] = global_step
    
    return epoch_metrics, val_metrics

def load_checkpoint(run_dir, model, optimizer, scheduler, scaler):

    checkpoint_path = os.path.join(run_dir, 'checkpoints', 'latest_model.pt')
    
    if os.path.exists(checkpoint_path):

        print("Loading from the latest checkpoint ...")
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        best_loss = checkpoint['best_loss']
        print(f"Resuming from epoch {start_epoch}, global_step {global_step}, learning_rate: {scheduler.get_last_lr()[0]}")

        return start_epoch, global_step, best_loss
    
    return 0, 0, float('inf')

def setup_datasets(config, tokenizer):

    train_dataset = VLMDataset(
        image_dir=config.train_image_dir,
        h5_path=config.train_h5_path,
        tokenizer=tokenizer,
        split="train"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size= config.batch_size,
        shuffle= True,
        num_workers= config.num_workers,
        pin_memory= True,
        persistent_workers=True
    )
    
    if hasattr(config, 'val_image_dir') and hasattr(config, 'val_h5_path'):
        val_dataset = VLMDataset(
            image_dir= config.val_image_dir,
            h5_path= config.val_h5_path,
            tokenizer= tokenizer,
            split= "val"
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size= config.batch_size,
            shuffle= False,
            num_workers= config.num_workers,
            pin_memory= True,
            persistent_workers=True
        )
    else:
        val_loader = None
        
    return train_loader, val_loader

def setup_model(encoder_config, decoder_config, train_loader, device):
    
    model = MultimodalCaptionGenerator(encoder_config, decoder_config).to(device)
    model = DataParallel(model, device_ids=list(range(len(decoder_config.gpu_ids))))
    model = torch.compile(
                model,
                mode='default',
                fullgraph=True, 
                dynamic=True  
            )
    
    # Separating encoder and decoder params
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': decoder_config.learning_rate * 0.5},
        {'params': decoder_params, 'lr': decoder_config.learning_rate}
    ])

    # Linear warmup before cosine cycles
    num_training_steps = len(train_loader) * decoder_config.num_epochs
    num_warmup_steps = int(num_training_steps * decoder_config.warmup_ratio)
    
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, 
                  0.5 * (1 + math.cos(math.pi * (current_step - num_warmup_steps) / 
                  (num_training_steps - num_warmup_steps))))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    scaler = GradScaler()
    
    return model, optimizer, scheduler, scaler

def save_checkpoint(checkpoint, is_best, run_dir):

    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    latest_path = os.path.join(checkpoint_dir, 'latest_model.pt')
    torch.save(checkpoint, latest_path)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)

def main():

    encoder_config = VisionConfig()
    decoder_config = MultimodalConfig()
    
    try:
        
        print("Starting initialization...")
        os.environ["CUDA_VISIBLE_DEVICES"] = decoder_config.cuda_visible_devices
        device = torch.device("cuda:0")
        
        run_dir = f'runs/{decoder_config.project_name}'
        artifcats_dir = f'artifacts/{decoder_config.project_name}'
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(artifcats_dir, exist_ok=True)
        
        tokenizer = get_tokenizer()
        
        wandb.init(
            project= f"{decoder_config.project_name}",
            config={
                "learning_rate": decoder_config.learning_rate,
                "batch_size": decoder_config.batch_size,
                "num_epochs": decoder_config.num_epochs,
                "decoder_config": decoder_config.__dict__,
                "encoder_config": encoder_config.__dict__,
            },
            resume=True
        )
        
        train_loader, val_loader = setup_datasets(decoder_config, tokenizer)
        model, optimizer, scheduler, scaler = setup_model(encoder_config, decoder_config, train_loader, device)
        
        start_epoch, global_step, best_loss = load_checkpoint(
            run_dir,
            model,
            optimizer,
            scheduler,
            scaler
        )
        
        for epoch in range(start_epoch, decoder_config.num_epochs):

            metrics, val_metrics = train_epoch(
                epoch, model, train_loader, val_loader, optimizer, 
                scheduler, scaler, device, global_step, 
                decoder_config, tokenizer, artifcats_dir
            )
            
            global_step = metrics['global_step']
            
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict(),
                'best_loss': best_loss,
                "decoder_config": decoder_config.__dict__,
                "encoder_config": encoder_config.__dict__
            }
            
            is_best = val_metrics['loss'] < best_loss
            if is_best:
                best_loss = val_metrics['loss']

            
            save_checkpoint(checkpoint, is_best, run_dir)
            
            wandb.log({
                'train/loss': metrics['loss'],
                'val/loss': val_metrics['loss'],
                'train/epoch_time': metrics['epoch_time'],
                'train/gpu_memory': metrics['gpu_memory'],
                'val/gpu_memory': val_metrics['gpu_memory'],
                'epoch': epoch,
                'global_step': global_step
            })             

    except Exception as e:
        print(f"Error")
        traceback.print_exc()
        raise

if __name__== "__main__":
    main()