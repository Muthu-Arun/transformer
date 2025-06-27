import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.io import decode_image
import torch.nn as nn

'''
def preprocess_image(image_path: str, image_size: int = 800):
    """
    Preprocess an image for input into a Vision Transformer model.
    
    Args:
        image_path (str): Path to the image file.
        image_size (int): Desired size of the image (default is 800).
        
    Returns:
        Image.Image: Preprocessed image.
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")
    # Resize the image
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize to 800x800
        transforms.RandomHorizontalFlip(0.2),  # Random horizontal flip with a probability of 0.2
        transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
        # Optional normalization can be added here if needed
    ])
    image = transform(image)
    
    return image

'''
def preprocess_image_tensor(image : torch.Tensor, image_size: int = 800):
    """
    Preprocess an image for input into a Vision Transformer model.
    
    Args:
        image_path (str): Path to the image file.
        image_size (int): Desired size of the image (default is 800).
        
    Returns:
        Image.Image: Preprocessed image.
    """
    # Load the image
    # image = Image.open(image_path).convert("RGB")
    # Resize the image
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize to 800x800
        transforms.RandomHorizontalFlip(0.2),  # Random horizontal flip with a probability of 0.2
        transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
        # Optional normalization can be added here if needed
    ])
    image = transform(image)
    
    return image

img_tensor = decode_image("/home/arun/dev/transformer/data/blue.png")
# image = preprocess_image("/home/arun/dev/transformer/data/blue.png")
print(img_tensor)