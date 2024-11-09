import torch
import torch.nn as nn
from einops import rearrange

class ImageEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_sz = config.patch_sz
        self.embed_dim = config.embed_dim
        self.img_sz = config.img_sz

        self.patches = nn.Conv2d(
            in_channels = config.in_channels,
            out_channels = self.embed_dim,
            kernel_size= self.patch_sz,
            stride = self.patch_sz,
            padding= "valid"
        )

        self.num_patches = (self.img_sz // self.patch_sz)**2
        self.pos_embeds = nn.Embedding(self.num_patches, self.embed_dim)

        # We use register_buffer when we don't want to keep track of the gradients. Also, we don't save it in the state_dict when persistent=False
        self.register_buffer(
            "pos_ids",
            rearrange(torch.arange(self.num_patches), 'n -> 1 n'),
            persistent=False
        )
    
    def forward(self, x):

        _, _, h, w = x.shape
        patch_embeds = self.patches(x) # This is of shape (bs, embed_dim, num_patches_h, num_patches_w). note num_patches_h = num_patches_w = num_patches
        patch_embeds = rearrange(patch_embeds, 'b e h w -> b (h w) e')
        # creating the look-up table of shape (bs, num_patches, embed_dim). 
        # Note that this is like nn.Linear(x) but with nn.Embeddings, rather than multiplying, we create a look up table
        # Note that self.pos_embeds is a nn.Embeddings instance where you have to access the tensor by self.pos_embeds.weight
        # pos_embeds is a tensor after applying the Embedding layer which is of shape [bs, num_patches, embed_dim]
        pos_embeds = self.pos_embeds(self.pos_ids) 

        # Note: To visualize the attention later, we need to get the attention scores in this shape
        # But the attention scores should be with regards to the text tokens
        return patch_embeds + pos_embeds

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.embed_dim, config.inter_size)
        self.fc2 = nn.Linear(config.inter_size, config.embed_dim)
    
    def forward(self, x):
        return self.fc2(nn.functional.gelu(self.fc1(x), approximate="tanh"))

class VisionAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5 
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.output_dropout = nn.Dropout(config.dropout_prob)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        bs, num_patches, _ = x.size()

        wq = self.q_proj(x)
        wk = self.k_proj(x)
        wv = self.v_proj(x)

        # Note: so far the shape of wq, wk, wv is [bs, num_patches, embed_dims]
        # We want to decompose the embed_dim for multi-head attention
        # [bs, num_heads, num_patches, head_dim]

        wq = rearrange(wq, 'b p (h d) -> b h p d', h=self.num_heads, d=self.head_dim)
        wk = rearrange(wk, 'b p (h d) -> b h p d', h=self.num_heads, d=self.head_dim)
        wv = rearrange(wv, 'b p (h d) -> b h p d', h=self.num_heads, d=self.head_dim)
        # Note that I transposed the first and second dims cause then each head can work over all the patches

        attention_w = (torch.matmul(wq, wk.transpose(2, 3)) * self.scale) # [bs, num_heads, num_patches, num_patches]
        attention_w = self.attn_dropout(attention_w)

        if attention_w.size() != (bs, self.num_heads, num_patches, num_patches):
            raise ValueError(
                f" Size mismatch for attention_w. Should be {(bs, self.num_heads, num_patches, num_patches)},"
                f" but got {attention_w.size()}"
            )
        # We apply softmax row-wise
        attention_w = nn.functional.softmax(attention_w, dim=-1, dtype=torch.float32).to(wq.dtype)

        attention_out = torch.matmul(attention_w, wv) 
        # here the inner dims are the same so no need to transpose
        # we go back to [bs, num_heads, num_patches, head_dim]

        if attention_out.size() != (bs, self.num_heads, num_patches, self.head_dim):
            raise ValueError(
                f" Size mismatch for attention_out. Should be {(bs, self.num_heads, num_patches, self.head_dim)}, "
                f" but got {attention_out.size()}"
            )

        # Concating heads
        attention_out = rearrange(attention_out, 'b h p d -> b p (h d)')

        attention_out = self.o_proj(attention_out)
        attention_out = self.output_dropout(attention_out)

        return attention_out

class EncoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.mh_attention = VisionAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, config.layer_norm_eps)

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = self.mh_attention(x)
        x = x + residual

        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + residual

        # Note that all throughout this, the dim [bs, num_patches, embed_dim] persists 
        return x

class VisionEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(self.config.num_encoder_layers)]
        )
    
    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        
        return x
    
class VisionTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embeds = ImageEmbeddings(self.config)
        self.encoder = VisionEncoder(self.config)
        self.post_layer_norm = nn.LayerNorm(self.config.embed_dim, self.config.layer_norm_eps)
    
    def forward(self, x):

        # Note that x here is of shape [bs, channels, h, w]
        embeds = self.embeds(x)
        out = self.post_layer_norm(self.encoder(embeds))

        # Out is of shape [bs, num_patches, emb_dim]
        return out




