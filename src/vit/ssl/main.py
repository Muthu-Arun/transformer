import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from ImagePreprocessing import create_batch,get_images
# Vision Transformer (ViT) for Self-Supervised Learning (SSL)
# This code implements a Vision Transformer model for self-supervised learning tasks.
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


"""
def random_masking(patches: torch.Tensor, mask_ratio: float = 0.5):
    '''
    Randomly mask patches in the input tensor.
    
    Args:
        patches (torch.Tensor): Input tensor of shape [B, num_patches, patch_dim].
        mask_ratio (float): Ratio of patches to be masked.
        
    Returns:
        torch.Tensor: Masked patches with shape [B, num_patches, patch_dim].
        torch.Tensor: Mask indicating which patches are masked.
    '''
    B, num_patches, patch_dim = patches.shape
    num_masked = int(num_patches * mask_ratio)
    
    # Generate random indices for masking
    indices = torch.randperm(num_patches)[:num_masked]
    
    # Create a mask
    mask = torch.zeros((B, num_patches), dtype=torch.bool)
    mask[:, indices] = True
    
    # Apply the mask
    masked_patches = patches.clone()
    masked_patches[mask] = 0  # Set masked patches to zero
    
    return masked_patches, mask

    
"""

'''
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int = PATCH_SIZE, in_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_dim = patch_size * patch_size * in_channels

    def forward(self, images: torch.Tensor):
        return get_patch_embedding(images, self.patch_size)
    
    def extra_repr(self) -> str:
        return f"patch_size={self.patch_size}, in_channels={self.in_channels}, patch_dim={self.patch_dim}"

''' 

def seperate_mask_and_patches(input_tensor: torch.Tensor, mask_ratio: float = 0.5) -> tuple:
    """
    Separates masked patches from the input tensor based on a given mask ratio.
    
    Args:
        input_tensor (torch.Tensor): Input tensor of shape [B, num_patches, patch_dim].
        mask_ratio (float): Ratio of patches to be masked.
        
    Returns:
        tuple: A tuple containing:
            - masked_patches (torch.Tensor): Patches with masked values set to zero.
            - mask (torch.Tensor): Boolean mask indicating which patches are masked.
    """
    B, num_patches, patch_dim = input_tensor.shape
    num_masked = int(num_patches * mask_ratio)
    
    # Generate random indices for masking
    indices = torch.randperm(num_patches)[:num_masked]
    
    masked_patches = input_tensor.clone()
    masked_patches = masked_patches[:, indices, :]  # Select patches to be masked
    input_tensor = input_tensor[:, ~torch.tensor(indices, dtype=torch.bool), :]  # Remaining patches
    return input_tensor, masked_patches



def separate_mask_and_patches_by_gpt(input_tensor: torch.Tensor, mask_ratio: float = 0.5) -> tuple:
    """
    Separates masked patches from the input tensor based on a given mask ratio.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape [B, num_patches, patch_dim].
        mask_ratio (float): Ratio of patches to be masked.

    Returns:
        tuple: 
            - visible_patches (torch.Tensor): Patches **not masked**.
            - masked_patches (torch.Tensor): Patches **masked out**.
            - mask (torch.Tensor): Boolean mask of shape [B, num_patches], True where masked.
    """
    B, num_patches, patch_dim = input_tensor.shape
    num_masked = int(num_patches * mask_ratio)

    # For each batch element, generate mask indices independently
    masks = []
    visible_list = []
    masked_list = [] 
    


    for b in range(B):
        perm = torch.randperm(num_patches)
        masked_indices = perm[:num_masked]
        visible_indices = perm[num_masked:]

        mask = torch.zeros(num_patches, dtype=torch.bool)
        mask[masked_indices] = True

        masked_patches = input_tensor[b, masked_indices, :]
        visible_patches = input_tensor[b, visible_indices, :]

        masks.append(mask)
        masked_list.append(masked_patches)
        visible_list.append(visible_patches)

    # Stack along batch dimension
    mask = torch.stack(masks, dim=0)                    # [B, num_patches]
    masked_patches = torch.stack(masked_list, dim=0)   # [B, num_masked, patch_dim]
    visible_patches = torch.stack(visible_list, dim=0) # [B, num_visible, patch_dim]

    return visible_patches, masked_patches, mask

class SelfAttention(nn.Module):
    def __init__(self, d_model: int = PATCH_EMBD, num_heads: int = 8):
        super(SelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        B, N, _ = x.shape
        qkv : torch.Tensor = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, N, self.d_model)
        return self.out_proj(attn_output)
    

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int = PATCH_EMBD, num_heads: int = 8, ff_hidden_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = SelfAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        attn_output = self.self_attn(x)
        x = self.norm1(x + attn_output)  # Residual connection
        ffn_output = self.ffn(x)
        return self.norm2(x + ffn_output)  # Residual connection

class DecoderBlock(nn.Module):
    def __init__(self,d_model: int = PATCH_EMBD, num_heads: int = 8, ff_hidden_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = SelfAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        attn_output = self.self_attn(x)
        x = self.norm1(x + attn_output)  # Residual connection
        ffn_output = self.ffn(x)
        return self.norm2(x + ffn_output)  # Residual connection

class Transformer(nn.Module):
    def __init__(self, num_encoder_layers: int = 6, num_decoder_layers: int = 6, d_model: int = PATCH_EMBD, mask_ratio : float = 0.5,num_heads: int = 8, ff_hidden_dim: int = 2048, dropout: float = 0.1):
        super(Transformer, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderBlock(d_model, num_heads, ff_hidden_dim, dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, num_heads, ff_hidden_dim, dropout) for _ in range(num_decoder_layers)])
        self.output_layer = nn.Linear(d_model, PATCH_EMBD)
        self.mask_ratio = mask_ratio
        self.visible_patch_num = int(PATCH_NUM * (1 - mask_ratio))
        self.masked_patch_num = PATCH_NUM - self.visible_patch_num
        self.mask = self.masked_patches = self.visible_patches = torch.Tensor()
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))  # Mask token for masked patches
        

        self.encoder_pos_embedding = nn.Parameter(torch.randn(1, self.visible_patch_num, d_model))
        self.decoder_pos_embedding = nn.Parameter(torch.randn(1, PATCH_NUM, d_model))
    
    def forward(self, x: torch.Tensor):
        x = get_patch_embedding(x)  # [B, PATCH_NUM, d_model]

        # Masking
        self.visible_patches, self.masked_patches, self.mask = separate_mask_and_patches_by_gpt(x)
        B = x.size(0)
        
        # Encode only visible patches
        self.visible_patches = self.visible_patches + self.encoder_pos_embedding  # [B, N_vis, d_model]
        for layer in self.encoder_layers:
            self.visible_patches = layer(self.visible_patches)

        # Prepare decoder input: insert mask tokens in masked positions
        mask_tokens = self.mask_token.expand(B, self.masked_patch_num, x.size(-1))  # [B, N_mask, d_model]
        
        full_sequence = torch.empty(B, PATCH_NUM, x.size(-1), device=x.device)
        full_sequence[self.mask == 0] = self.visible_patches
        full_sequence[self.mask == 1] = mask_tokens

        # Add decoder pos embedding
        full_sequence = full_sequence + self.decoder_pos_embedding  # [B, PATCH_NUM, d_model]
        
        # Decode
        x = full_sequence
        for layer in self.decoder_layers:
            x = layer(x)

        return self.output_layer(x)  # [B, PATCH_NUM, PATCH_EMBD]

    
model = Transformer()
image_files = get_images("data")
input_tensor = create_batch(image_files,32,0)
test_input_tensor = torch.randn((32,3,800,800))
model(test_input_tensor)
# model.compile(
#     optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
#     loss_fn=nn.CrossEntropyLoss(),
#     metrics=['accuracy']
# )