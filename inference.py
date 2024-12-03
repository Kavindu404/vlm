import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from pathlib import Path
import torch.nn.functional as F
from einops import rearrange
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from train import compute_cider
from collections import defaultdict
import json
import random

from model import MultimodalCaptionGenerator, MultimodalVQAModel
from config import VisionConfig, MultimodalConfig
from dataset import VLMDataset, VQADataset
from preprocess import get_tokenizer, get_vqa_tokenizer
from contextlib import contextmanager
import torch._dynamo

@contextmanager
def no_compile():
    torch._dynamo.config.disable = True
    try:
        yield
    finally:
        torch._dynamo.config.disable = False

def load_model(model_path, model_type="caption"):
    """Load trained model"""
    encoder_config = VisionConfig()
    decoder_config = MultimodalConfig()
    
    if model_type == "caption":
        model = MultimodalCaptionGenerator(encoder_config, decoder_config)
    else:
        model = MultimodalVQAModel(encoder_config, decoder_config)
    
    checkpoint = torch.load(model_path)
    state_dict = checkpoint['model_state_dict']
    
    # Handle both DataParallel and compiled model prefixes
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '').replace('_orig_mod.', '')
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = torch.compile(
        model,
        mode='default',
        fullgraph=True,
        dynamic=True
    )
    model.eval()
    
    return model, decoder_config

def generate_text(model, image, question=None, tokenizer=None, config=None, 
                 base_temperature=0.7, temp_decay=0.95, min_temperature=0.3, top_k=50):
    """Generate text with temperature and top-k sampling"""
    with torch.no_grad():
        tokens = torch.tensor([[config.sos_token_id]]).cuda()
        generated_tokens = [config.sos_token_id]
        attention_maps = []
        
        max_len = config.max_question_len if question is not None else config.max_len
        
        for i in range(max_len):
            temperature = max(base_temperature * (temp_decay ** i), min_temperature)
            
            if question is not None:
                logits, attention_scores = model(question, tokens, image)
            else:
                logits, attention_scores = model(tokens, image)
            
            logits = logits[0, -1] / temperature
            
            # Apply repetition penalty
            if len(generated_tokens) > 1:
                for prev_token in generated_tokens:
                    logits[prev_token] /= 3.0
            
            # Top-k sampling
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            probs = torch.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[next_token_idx]
            
            generated_tokens.append(next_token.item())
            attention_maps.append(attention_scores[-1].cpu())
            
            if next_token.item() == config.eos_token_id:
                break
            
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
    
    return generated_tokens, attention_maps

def compute_caption_metrics(generated_text, true_text):
    """
    Compute metrics for image captioning:
    1. BLEU-1,2,3,4 scores
    2. CiDEr score
    """
    # BLEU scores with different n-grams
    bleu_1 = sentence_bleu([true_text.split()], generated_text.split(), weights=(1, 0, 0, 0))
    bleu_2 = sentence_bleu([true_text.split()], generated_text.split(), weights=(0.5, 0.5, 0, 0))
    bleu_3 = sentence_bleu([true_text.split()], generated_text.split(), weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = sentence_bleu([true_text.split()], generated_text.split(), weights=(0.25, 0.25, 0.25, 0.25))
    
    # CiDEr score
    cider_score = compute_cider(generated_text, true_text)
    
    return {
        'bleu_1': bleu_1,
        'bleu_2': bleu_2,
        'bleu_3': bleu_3,
        'bleu_4': bleu_4,
        'cider': cider_score
    }

def compute_vqa_metrics(generated_answer, true_answer):
    """
    Compute VQA-specific metrics:
    1. Exact Match
    2. Normalized Match (case-insensitive, removing articles)
    3. Token Overlap
    """
    def normalize_answer(answer):
        articles = set(['a', 'an', 'the'])
        answer = answer.lower().strip()
        words = answer.split()
        words = [w for w in words if w not in articles]
        return ' '.join(words)
    
    # Exact match
    exact_match = int(generated_answer == true_answer)
    
    # Normalized match
    norm_gen = normalize_answer(generated_answer)
    norm_true = normalize_answer(true_answer)
    normalized_match = int(norm_gen == norm_true)
    
    # Token overlap (F1-like score)
    gen_tokens = set(normalize_answer(generated_answer).split())
    true_tokens = set(normalize_answer(true_answer).split())
    if len(gen_tokens) == 0 and len(true_tokens) == 0:
        token_overlap = 1.0
    elif len(gen_tokens) == 0 or len(true_tokens) == 0:
        token_overlap = 0.0
    else:
        intersection = len(gen_tokens.intersection(true_tokens))
        precision = intersection / len(gen_tokens)
        recall = intersection / len(true_tokens)
        token_overlap = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'exact_match': exact_match,
        'normalized_match': normalized_match,
        'token_overlap': token_overlap
    }

def visualize_attention(image, attention_map, token_text, save_path, head_idx, num_img_tokens, save_mean=False):
    """Visualize attention for a specific token and head"""
    plt.figure(figsize=(10, 10))
    
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_display = image.cpu() * std + mean
    img_display = img_display.permute(1, 2, 0).numpy()
    
    # Process attention map
    if save_mean:
        # Average across all heads
        token_attention = attention_map.mean(0)[-1, :num_img_tokens]
    else:
        token_attention = attention_map[head_idx][-1, :num_img_tokens]
    
    h = w = int(np.sqrt(num_img_tokens))
    token_attention = rearrange(token_attention, '(h w) -> h w', h=h, w=w)
    
    # Upsample attention map
    token_attention = token_attention.unsqueeze(0).unsqueeze(0)
    attention_map = F.interpolate(token_attention, size=(224, 224), mode='bilinear', align_corners=False)
    attention_map = attention_map.squeeze().numpy()
    
    plt.imshow(img_display)
    plt.imshow(attention_map, alpha=0.5, cmap='viridis')
    title = f'Mean attention for token: {token_text}' if save_mean else f'Attention for token: {token_text} (Head {head_idx})'
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def evaluate_model(model, val_dataset, tokenizer, config, save_dir, num_samples=32, model_type="caption"):
    """Evaluate model on random samples"""
    os.makedirs(save_dir, exist_ok=True)
    results = []
    
    # Randomly sample indices
    total_samples = len(val_dataset)
    sample_indices = random.sample(range(total_samples), min(num_samples, total_samples))
    
    with no_compile():
        model.eval()
        for sample_idx in tqdm(sample_indices, desc="Generating samples"):
            sample = val_dataset[sample_idx]
            image = sample['image'].unsqueeze(0).cuda()
            
            if model_type == "caption":
                generated_tokens, attention_maps = generate_text(
                    model, image, None, tokenizer, config
                )
                question_tokens = [t for t in sample['caption'].tolist() 
                                if t not in [config.padding_idx, config.sos_token_id, config.eos_token_id]]
                true_text = tokenizer.decode(question_tokens)
                generated_text = tokenizer.decode(generated_tokens[1:-1])
                
                # Calculate metrics
                metrics = compute_caption_metrics(generated_text, true_text)
                
                result = {
                    'sample_id': sample_idx,
                    'true_text': true_text,
                    'generated_text': generated_text,
                    'metrics': metrics
                }
                
            else:  # VQA
                question = sample['questions'].unsqueeze(0).cuda()
                question_tokens = [t for t in sample['questions'].tolist() 
                                if t not in [config.padding_idx, config.q_start_token_id, config.q_end_token_id]]
                question_text = tokenizer.decode(question_tokens)
                
                generated_tokens, attention_maps = generate_text(
                    model, image, question, tokenizer, config
                )
                
                answer_tokens = [t for t in sample['answers'].tolist()
                                if t not in [config.padding_idx, config.sos_token_id, config.eos_token_id]]
                true_answer = tokenizer.decode(answer_tokens)
                generated_answer = tokenizer.decode(generated_tokens[1:-1])
                
                # Calculate metrics
                metrics = compute_vqa_metrics(generated_answer, true_answer)
                
                result = {
                    'sample_id': sample_idx,
                    'question': question_text,
                    'true_answer': true_answer,
                    'generated_answer': generated_answer,
                    'metrics': metrics
                }
            
            # Save attention visualizations
            sample_dir = os.path.join(save_dir, f'sample_{sample_idx}')
            os.makedirs(sample_dir, exist_ok=True)
            
            img_save_path = os.path.join(sample_dir, 'input_image.png')
            plt.figure(figsize=(10, 10))
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_display = sample['image'].cpu() * std + mean
            img_display = img_display.permute(1, 2, 0).numpy()
            plt.imshow(img_display)
            plt.axis('off')
            plt.savefig(img_save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            for token_idx, (token_id, attn_map) in enumerate(zip(generated_tokens[1:], attention_maps)):
                if token_id == config.eos_token_id:
                    break
                    
                token_text = tokenizer.decode([token_id])
                
                # Save individual head attention maps
                for head in range(config.num_heads):
                    save_path = os.path.join(sample_dir, f'token_{token_idx}_head_{head}.png')
                    visualize_attention(
                        sample['image'], 
                        attn_map, 
                        token_text, 
                        save_path, 
                        head,
                        config.num_image_tokens,
                        save_mean=False
                    )
                
                # Save mean attention map
                mean_save_path = os.path.join(sample_dir, f'token_{token_idx}_mean.png')
                visualize_attention(
                    sample['image'],
                    attn_map,
                    token_text,
                    mean_save_path,
                    0,  # head_idx is ignored when save_mean=True
                    config.num_image_tokens,
                    save_mean=True
                )
            
            results.append(result)
    
    # Before saving results in evaluate_model:
    if len(results) > 0:
        if model_type == "caption":
            aggregate_metrics = {
                'avg_bleu_1': np.mean([r['metrics']['bleu_1'] for r in results]),
                'avg_bleu_2': np.mean([r['metrics']['bleu_2'] for r in results]),
                'avg_bleu_3': np.mean([r['metrics']['bleu_3'] for r in results]),
                'avg_bleu_4': np.mean([r['metrics']['bleu_4'] for r in results]),
                'avg_cider': np.mean([r['metrics']['cider'] for r in results])
            }
        else:
            aggregate_metrics = {
                'avg_exact_match': np.mean([r['metrics']['exact_match'] for r in results]),
                'avg_normalized_match': np.mean([r['metrics']['normalized_match'] for r in results]),
                'avg_token_overlap': np.mean([r['metrics']['token_overlap'] for r in results])
            }
        
        final_results = {
            'individual_results': results,
            'aggregate_metrics': aggregate_metrics
        }
    else:
        final_results = {'individual_results': results}

    # Save results
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    return results


def main():

    # caption_model_path = "/unity/g1/kgalla/projects/vlm/runs/base_vlm_self_attn_sinPos_val_v32/checkpoints/best_model.pt"
    vqa_model_path = "/unity/g1/kgalla/projects/vlm/runs/base_vqa_self_attn_sinPos_gqa_v1/checkpoints/best_model.pt"
    output_dir = "inference_results"
    
    # caption_model, caption_config = load_model(caption_model_path, "caption")
    vqa_model, vqa_config = load_model(vqa_model_path, "vqa")
    
    # Set up tokenizers
    # caption_tokenizer = get_tokenizer()
    vqa_tokenizer = get_vqa_tokenizer()
    
    # Set up datasets
    # caption_val_dataset = VLMDataset(
    #     image_dir=caption_config.val_image_dir,
    #     h5_path=caption_config.val_h5_path,
    #     tokenizer=caption_tokenizer,
    #     split="val"
    # )
    
    vqa_val_dataset = VQADataset(
        image_dir=vqa_config.val_image_dir,
        h5_path=vqa_config.val_h5_path,
        tokenizer=vqa_tokenizer,
        split="val"
    )
    
    # Run evaluations
    # print("Evaluating Image Captioning Model...")
    # caption_results = evaluate_model(
    #     caption_model,
    #     caption_val_dataset,
    #     caption_tokenizer,
    #     caption_config,
    #     os.path.join(output_dir, 'caption_results'),
    #     model_type="caption"
    # )
    
    print("Evaluating VQA Model...")
    vqa_results = evaluate_model(
        vqa_model,
        vqa_val_dataset,
        vqa_tokenizer,
        vqa_config,
        os.path.join(output_dir, 'vqa_results'),
        model_type="vqa"
    )
    
    print("Evaluation Complete!")

if __name__ == "__main__":
    main()