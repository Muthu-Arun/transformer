import PIL.Image
import torch
import torch.nn.functional as F
import PIL
import torchvision.transforms as transforms
import time
import os
def display_image_tensor(input: torch.Tensor) :
    to_pil = transforms.ToPILImage()
    image: PIL.Image.Image = to_pil(input.squeeze())
    image.show()
    
def save_image_tensor(path: str = "data/output/"):
    pass
    
    
def reconstruct_image(image: torch.Tensor, patch_size: int, image_size: int) -> torch.Tensor:
    B, num_patches, patch_dim = image.shape
    C: int = 3
    assert image_size % patch_size == 0

    recon_image = image.contiguous().view(B,num_patches,3,patch_size,patch_size)

    recon_image = recon_image.permute(0,2,1,3,4)
    
    recon_image = recon_image.contiguous().view(B,C,num_patches,patch_size,patch_size)

    recon_image = F.fold(recon_image,(B,C,800,800),(16,16))
    return recon_image
    
def reconstruct_image_gpt(patches: torch.Tensor, patch_size: int, image_size: int, channels: int = 3) -> torch.Tensor:
    """
    Reconstructs the image from flattened patch embeddings.

    Args:
        patches: Tensor of shape [B, num_patches, patch_dim] (patch_dim = C*P*P)
        patch_size: Size of each patch (P)
        image_size: Original image size (assumes square: H=W=image_size)
        channels: Number of channels in the image (e.g., 3 for RGB)

    Returns:
        Reconstructed image of shape [B, C, H, W]
    """
    B, num_patches, patch_dim = patches.shape
    assert patch_dim == channels * patch_size * patch_size
    assert image_size % patch_size == 0
    num_patches_per_row = image_size // patch_size

    # Prepare for F.fold
    # reshape patches to [B, patch_dim, num_patches]
    patches = patches.permute(0, 2, 1)

    # Use fold to reconstruct
    output = F.fold(
        patches,                             # [B, C*patch_area, num_patches]
        output_size=(image_size, image_size),
        kernel_size=(patch_size, patch_size),
        stride=(patch_size, patch_size)
    )

    return output


test_tensor = torch.randint(0,255,(3,800,800),dtype=torch.uint8)
test_recon = torch.randn(1,2500,768)
recon_image = reconstruct_image_gpt(test_recon,16,800)
print(recon_image.shape)
# display_image_tensor(test_tensor)
# reconstruct_image()