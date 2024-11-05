import torch
import torch.nn as nn
from einops import rearrange
import math


class RMSNorm(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.eps = config.eps
        self.weight = nn.Parameter(torch.zeros(config.hidden_size)) # dim here is the hidden_dim
    
    def _norm(self, x):
        # rsqrt is the reciprocal of the sqrt. so this gives us x/rms
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        out = self._norm(x.float())
        out = out * (1.0 + self.weight.float()) # This is the gamma parameter, one for each row

        return out.type_as(x)

class PosEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = self.config.hidden_size
        self.seq_len = self.config.seq_len

        pos_emb = torch.zeros(self.seq_len, self.hidden_size)
        # One position for each seq. The extra dim helps to broadcast
        positions = torch.arange(0, self.seq_len).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, self.hidden_size, 2).float() * (-math.log(10000.0)/self.hidden_size))
        pos_emb[:, 0::2] = torch.sin(positions * denominator)
        pos_emb[:, 1::2] = torch.cos(positions * denominator)
        pos_emb = pos_emb.unsqueeze(0)

        self.register_buffer('pos_emb', pos_emb)
    
    def forward(self, x):
        x = x + self.pos_emb[:, :x.shape[1], :].requires_grad_(False)
        return x


class DecoderMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = self.config.hidden_size
        self.inter_dim = self.config.inter_dim

        self.gate_proj = nn.Linear(self.hidden_size, self.inter_dim)
        self.up_proj = nn.Linear(self.hidden_size, self.inter_dim)
        self.down_proj = nn.Linear(self.inter_dim, self.hidden_size)
    
    def forward(self, x):
        # x here is of shape [bs, num_patches, hidden_size]
        y = self.gate_proj(x) # [bs, num_patches, inter_dim]
        y = nn.functional.gelu(y, approximate='tanh')
        u = self.up_proj(x) # [bs, num_patches, inter_dim]
        y = y * u # element-wise
        out = self.down_proj(y) # [bs, num_patches, hidden_size]

        return out

class DecoderAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5 

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
    
    def forward(self, x, mask=None):
        bs, seq_len, _ = x.size()

        wq = self.q_proj(x)
        wk = self.k_proj(x)
        wv = self.v_proj(x)

        wq = rearrange(wq, 'b s (h d) -> b h s d', h=self.num_heads, s=seq_len, d= self.head_dim)
        wv = rearrange(wv, 'b s (h d) -> b h s d', h=self.num_heads, s=seq_len, d= self.head_dim)
        wk = rearrange(wk, 'b s (h d) -> b h s d', h=self.num_heads, s=seq_len, d= self.head_dim)

        attention_w = (torch.matmul(wq, wk.transpose(2, 3)) * self.scale)

        if attention_w.size() != (bs, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f" Size mismatch for attention_w. Should be {(bs, self.num_heads, seq_len, seq_len)},"
                f" but got {attention_w.size()}"
            )
        if mask is not None:
            mask = mask.unsqueeze(1)
            attention_w.masked_fill_(mask==0, -1e4)

        # We can use this for visualization later
        attention_scores = nn.functional.softmax(attention_w, dim=-1, dtype=torch.float32).to(wq.dtype)

        attention_out = torch.matmul(attention_scores, wv) 

        if attention_out.size() != (bs, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f" Size mismatch for attention_out. Should be {(bs, self.num_heads, seq_len, self.head_dim)},"
                f" but got {attention_out.size()}"
            )
        attention_out = rearrange(attention_out, 'b h s d -> b s (h d)', h=self.num_heads, s=seq_len, d= self.head_dim)

        attention_out = self.o_proj(attention_out)

        return attention_out, attention_scores

class DecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inputRmsNorm = RMSNorm(config)
        self.outputRmsNorm = RMSNorm(config)
        self.multimodalAttention = DecoderAttention(config)
        self.mlp = DecoderMLP(config)
    
    def forward(self, x, attention_mask=None):
        residual = x

        x_norm = self.inputRmsNorm(x)
        x, attn_scores = self.multimodalAttention(x_norm, attention_mask)

        x = residual + x

        residual = x

        x_norm = self.outputRmsNorm(x)
        x = self.mlp(x_norm)
        x = x + residual

        return x, attn_scores

class MultimodalDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.padding_idx)
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_decoder_layers)]
        )
    
    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, x, mask=None):

        for layer in self.layers:
            x, attn_scores = layer(x, mask)

        return x, attn_scores

class MultimodalProjector(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.emb_dim = config.emb_dim
        self.proj_dim = config.proj_dim

        self.multimodal_proj_layer = nn.Linear(self.emb_dim, self.proj_dim, bias=True)
    
    def forward(self, x):

        return self.multimodal_proj_layer(x)
