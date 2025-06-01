import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection 
from torchvision.io import decode_image
import os
import math
BATCH_SIZE = 16
IMAGE_SIZE = 800

PATCH_SIZE = 16
PATCH_NUM = (IMAGE_SIZE // PATCH_SIZE) ** 2
PATCH_EMBD = PATCH_SIZE ** 2 * 3  # Square patchs with 3 channels R,G,B

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.RandomHorizontalFlip(0.2),
    transforms.ToTensor(),  # Converts to [C, H, W] and normalizes to [0,1]
    # transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # Optional
    
])

# Folder structure:
# root/train/class1/*.jpg
# root/train/class2/*.jpg
# root/val/class1/*.jpg


'''
train_dataset = datasets.ImageFolder(root='./data/train', transform=transform)
val_dataset = datasets.ImageFolder(root='./data/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
'''


# patch_dim : int = 0
# patch_size : int = 16
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


class Embedding(nn.Module):
    def __init__(self,patch_dim : int = PATCH_EMBD,patch_size : int = PATCH_SIZE):
        
        super().__init__()
        self.patch_dim = patch_dim
        self.patch_num = (IMAGE_SIZE // patch_size) ** 2                                #(800 / 16) ^ 2
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1,self.patch_num+1,self.patch_dim))
        self.cls_token = nn.Parameter(torch.randn(1,1,self.patch_dim))


    def forward(self,x : torch.Tensor):
        B = x.shape[0]

        x = get_patch_embedding(x,self.patch_size)
        clc_tokens = self.cls_token.expand(B,-1,-1)

        x = torch.cat([x,clc_tokens],dim=1)
        x = x + self.pos_embedding
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self,dmodel : int = PATCH_EMBD,num_heads = 8):
        
        assert dmodel % num_heads == 0
        super().__init__()
        self.num_heads = num_heads
        

        self.head_dim = int(dmodel/num_heads)
        self.qkv_projection = nn.Linear(dmodel,3*dmodel)
        self.out_projection = nn.Linear(dmodel,dmodel)
    
# Encoder Only
    def forward(self,x : torch.Tensor):
        B,T,D = x.size()
        qkv : torch.Tensor = self.qkv_projection(x)
        qkv = qkv.reshape(B,T,3,self.num_heads,self.head_dim)
        qkv = qkv.permute(2,0,3,1,4)
        q,k,v = qkv[0],qkv[1],qkv[2]
        attention_score = q @ k.transpose(-2,-1) / math.sqrt(self.head_dim)

        attention_weights = torch.softmax(attention_score,-1)

        attention_output = attention_weights @ v

        attention_output = attention_output.transpose(1,2).contiguous().reshape(B,T,D)

        return self.out_projection(attention_output)


class EncoderBlock(nn.Module):
    def __init__(self,dmodel : int = PATCH_EMBD,ff_hidden_dim : int = 2048, dropout : float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention()
        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)
        self.feed_forward = nn.Sequential(
            nn.Linear(dmodel,ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim,ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim,dmodel)

        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x : torch.Tensor):
        attention_out = self.self_attention(self.norm1(x))

        x = x + self.dropout(attention_out)

        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x
    
class ViT(nn.Module):
    def __init__(self,dmodel : int = PATCH_EMBD):
        super().__init__()
        self.embedding = Embedding()
        self.encoders = nn.Sequential(*[EncoderBlock() for _ in range(6)])
        self.layernorm = nn.LayerNorm(dmodel)
        self.vit_head = nn.Linear(dmodel,4)
    
    def forward(self, x : torch.Tensor):
        x = self.embedding(x)
        x = self.encoders(x)
        x = self.layernorm(x)
        logits = self.vit_head(x)
        return logits
    

model = ViT()