import torch
import torch.nn as nn
import math
import torchvision.transforms as transforms
from torchvision.io import decode_image
from torchvision import tv_tensors
from PIL import Image
# Constants
BATCH_SIZE = 16
IMAGE_SIZE = 800

PATCH_SIZE = 16
PATCH_NUM = (IMAGE_SIZE // PATCH_SIZE) ** 2
PATCH_EMBD = PATCH_SIZE ** 2 * 3  # Square patchs with 3 channels R,G,B

def get_patch_embedding(images: torch.Tensor, patch_size: int = PATCH_SIZE):
    """
    images: Tensor of shape [B, C, H, W]
    patch_size: patch size (e.g., 16)
    
    Returns: patches [B, num_patches, patch_dim]
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image size must be divisible by patch size"

    # Use unfold to extract non-overlapping patches
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # Shape: [B, C, num_patches_H, num_patches_W, patch_size, patch_size]

    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    # [B, C, num_patches, patch_size, patch_size]

    patches = patches.permute(0, 2, 1, 3, 4)  # [B, num_patches, C, patch_size, patch_size]
    patches = patches.contiguous().view(B, -1, C * patch_size * patch_size)
    # PATCH_EMBD = C*patch_size*patch_size
    
    # [B, num_patches, patch_dim]

    return patches

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int = PATCH_SIZE, in_channels: int = 3):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_dim = patch_size * patch_size * in_channels

    def forward(self, images: torch.Tensor):
        return get_patch_embedding(images, self.patch_size)
    
    def extra_repr(self) -> str:
        return f"patch_size={self.patch_size}, in_channels={self.in_channels}, patch_dim={self.patch_dim}"
    
class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor):
        B,T,D = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attention_scores : torch.Tensor = (q @ k.transpose(-2, -1)) / self.scale
        mask = torch.tril(torch.ones(T, T)).unsqueeze(0).unsqueeze(0).to(x.device)
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = attention_weights @ v
        attention_output = attention_output.transpose(1, 2).contiguous().reshape(B, T, D)
        return self.out_proj(attention_output)
    

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int = PATCH_EMBD, num_heads: int = 8, ff_hidden_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MaskedMultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor):
        attn_output = self.self_attention(x)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        return self.norm2(x + ff_output)

class Transformer(nn.Module):
    def __init__(self, num_layers: int = 6, d_model: int = PATCH_EMBD, num_heads: int = 8, ff_hidden_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size=PATCH_SIZE)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, ff_hidden_dim, dropout) for _ in range(num_layers)
        ])
        
    def forward(self, images: torch.Tensor):
        patches = self.patch_embedding(images)  # [B, num_patches, patch_dim]
        x = patches
        for block in self.decoder_blocks:
            x = block(x)
        return x