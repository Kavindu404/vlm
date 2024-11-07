# Multimodal Vision-Language model for Image Caption Generation

This repository contains the implementation of a Vision-Language Model (VLM) for image captioning. This is the base model that uses standard multi-head self-attention with sinusoidal positional embeddings.

## Model Architecture

1. Vision Encoder (ViT)
  - Patch embedding layer
  - Learned positional embeddings
  - Transformer encoder blocks (self-attention + MLP)
  - Layer normalization

2. Decoder
  - Token embedding layer
  - Sinusoidal positional embeddings
  - Transformer decoder blocks:
    - Multi-head self-attention
    - Gated MLP
    - RMS normalization
  
3. Cross-Modal Integration
  - Linear projection from vision to text dimensions
  - Concatenated attention (image + text)
  - Output projection to vocabulary

### Base Model Components:
- Multi-head Self-Attention
- Sinusoidal Positional Embeddings
- Vision Transformer (ViT) for image encoding
- Autoregressive decoder for caption generation
- RMSNorm for layer normalization

### Other Variations (To be released soon):
- Flash Attention for efficient attention computation
- Rotary Positional Embeddings (RoPE)
- KV-Cache for faster inference

## Configuration

The model consists of two main configuration classes:

### Vision Config
Controls the vision transformer parameters:
- Patch size: 16x16
- Embedding dimension: 768
- Number of heads: 12
- Number of encoder layers: 12
- Image size: 224x224

### Multimodal Config
Controls the decoder and training parameters:
- Hidden size: 2048
- Number of heads: 8
- Number of decoder layers: 8
- Projection dimension: 2048
- Learning rate: 1e-4
- Batch size: 128
- Maximum sequence length: 196

## Training Your Own Model

To train your own model, modify the configurations in `config.py`. Key parameters you might want to adjust:

```python
# Vision parameters
self.patch_sz = 16
self.embed_dim = 768
self.num_encoder_layers = 12

# Decoder parameters
self.hidden_size = 2048
self.num_decoder_layers = 8
self.num_heads = 8

# Training parameters
self.learning_rate = 1e-4
self.batch_size = 128
self.num_epochs = 100000

Also update the data paths:

self.train_image_dir = "path/to/train/images"
self.train_h5_path = "path/to/train/captions.h5"
self.val_image_dir = "path/to/val/images"
self.val_h5_path = "path/to/val/captions.h5"
```

Note that, you can preprocess your data to create the .h5 file with preprocess.py 

## Dataset

The model is trained on the MSCOCO dataset. Preprocessing scripts are included to:

Tokenize captions: We used the Tiktoken(gpt2) tokenizer from OpenAI

Create H5 files for efficient loading

Handle special tokens (SOS, EOS, PAD)


