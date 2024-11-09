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

from dataset import VLMDataset
from model import MultimodalCaptionGenerator
from config import VisionConfig, MultimodalConfig
from preprocess import get_tokenizer

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def validate(epoch, model, val_loader, tokenizer, device, num_img_tokens, config, caption_table):

    model.eval()
    val_metrics = defaultdict(float)
    num_batches = len(val_loader)
    fixed_batch_idx = 2 
    fixed_sample_idx = 5

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
                ignore_index=config.padding_idx
            )
            
            val_metrics['loss'] += loss.item()
            current_val_loss = val_metrics['loss'] / (batch_idx + 1)

            pbar.update(1)
            pbar.set_postfix({
                'val_loss': f'{current_val_loss:.4f}',
                'gpu_mem': f'{get_gpu_memory():.2f}GB'
            })

            if batch_idx == fixed_batch_idx and epoch%10==0:

                artifact_dir = Path(f'artifacts/base_model_self_attn_sinPos_val_v3/epoch_{epoch}')
                artifact_dir.mkdir(parents=True, exist_ok=True)

                image = batch['image'][fixed_sample_idx:fixed_sample_idx+1].to(device)
                tokens = torch.tensor([[config.sos_token_id]]).to(device)
                generated_tokens = [config.sos_token_id]
                attention_maps = []
                
                print(f"Saving artifacts...")

                for _ in range(config.max_len):
                    logits, attention_scores = model(tokens, image)

                    next_token = logits[0, -1].argmax(dim=-1) # We follow a greedy approach where we get the token with the max prob
                    generated_tokens.append(next_token.item())

                    attention_maps.append(attention_scores[-1].cpu()) # We take the attention map of only the first element of the batch

                    if next_token.item() == config.eos_token_id:
                        break
                    
                    tokens = torch.cat([tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                
                caption = tokenizer.decode(generated_tokens[1:-1])  # removing SOS and EOS
                caption_table.add_data(epoch, caption)
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
                    for head in range(num_heads):
                        # we take the attention_score for the image tokens of the last token added
                        token_attention = attention_maps[head][-1][-1, :num_img_tokens] 
                        h = w = int(np.sqrt(num_img_tokens))
                        token_attention = rearrange(token_attention, '(h w) -> h w', h=h, w=w)

                        token_attention = torch.tensor(token_attention).unsqueeze(0).unsqueeze(0)  # add batch and channel dims [1, 1, 14, 14]
                        # We upsample the attention map from 14x14 -> 224x224. Note that we have 14x14 patches and attention score for each patch
                        attention_map = F.interpolate(token_attention, size=(224, 224), mode='bilinear', align_corners=False)
                        attention_map = attention_map.squeeze().numpy() 
                        
                        plt.figure(figsize=(10, 10))
                        plt.imshow(orig_image)
                        plt.imshow(attention_map, alpha=0.5, cmap='viridis')
                        plt.title(f'{token_text} - Head {head}')
                        plt.axis('off')
                        
                        plt.savefig(
                            artifact_dir / f'{idx}_{token_text}_head_{head}.png',
                            bbox_inches='tight',
                            pad_inches=0
                        )
                        plt.close()

    pbar.close()
    val_metrics['loss'] /= num_batches
    wandb.log({
        'val/loss': val_metrics['loss'],
        'epoch': epoch
    })

    return val_metrics


def get_gpu_memory():
    return torch.cuda.memory_allocated() / 1e9

def train_epoch(epoch, model, dataloader, val_loader, optimizer, scheduler, scaler, device, global_step, config, tokenizer):

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
            'train_loss': f'{loss.item():.4f}', 
            'gpu_mem': f'{current_mem:.2f}GB',
            'step': global_step
        })
    pbar.close()

    val_metrics = None

    if val_loader:

        caption_table = wandb.Table(columns=["epoch", "generated_caption"])

        val_metrics = validate(
                            epoch= epoch,
                            model= model,
                            val_loader= val_loader,
                            tokenizer= tokenizer,
                            device=device,
                            num_img_tokens= config.num_image_tokens,
                            config= config,
                            caption_table= caption_table
                    )
 
    epoch_metrics['loss'] += loss.item()
    epoch_metrics['batch_time'] += batch_time
    epoch_metrics['gpu_memory'] += current_mem
    epoch_metrics['learning_rate'] = current_lr
    
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
        print(f"Resuming from epoch {start_epoch}, global_step {global_step}")

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
    model = DataParallel(model)
    model = torch.compile(
                model,
                mode='dafult',
                fullgraph=True, 
                dynamic=True,   
                verbose=True     
            )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=decoder_config.learning_rate,
        weight_decay=decoder_config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=decoder_config.learning_rate,
        epochs=decoder_config.num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1  # 10% warmup
    )
    
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
    
    try:
        print("Starting initialization...")
        device = torch.device("cuda:0")
        
        run_dir = f'runs/base_model_self_attn_sinPos_val_v3'
        artifcats_dir = f'artifacts/base_model_self_attn_sinPos_val_v3'
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(artifcats_dir, exist_ok=True)
        
        encoder_config = VisionConfig()
        decoder_config = MultimodalConfig()
        tokenizer = get_tokenizer()
        
        wandb.init(
            project= "base_vlm_self_attn_sinPos_val_v2_1",
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
                decoder_config, tokenizer
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
                'global_step': global_step,
                'learning_rate': optimizer.param_groups[0]['lr']
            })             

    except Exception as e:
        print(f"Error")
        traceback.print_exc()
        raise

if __name__== "__main__":
    main()