import torch
import torchvision.transforms as transforms
from PIL import Image
# from torchvision.io import decode_image
import torch.nn as nn
import os

IMAGE_SIZE = 800
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize to 800x800
    # Random horizontal flip with a probability of 0.2
    transforms.RandomHorizontalFlip(0.2),
    transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
    # Optional normalization can be added here if needed
])
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


def get_images(root_dir: str) -> tuple:
    possible_files = os.listdir(root_dir)
    formats = ["jpg", "jpeg", "png"]
    files = []
    for file in possible_files:
        if file.endswith(tuple(formats)):
            files.append(root_dir+"/"+file)
    return tuple(files)


def preprocess_image_tensor(image: Image.Image, image_size: int = 800):
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
        # Random horizontal flip with a probability of 0.2
        transforms.RandomHorizontalFlip(0.2),
        transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
        # Optional normalization can be added here if needed
    ])
    image = transform(image)

    return image


def create_batch(image_files: tuple, batch_size: int, offset: int):
    images = []
    if len(image_files) >= batch_size+offset:
        for elem in range(batch_size):
            images.append(transform(Image.open(
                image_files[elem+offset]).convert("RGB")))
        return torch.stack(images)
    for elem in range(batch_size):
        if len(image_files) >= elem+offset+1:
            images.append(transform(Image.open(
                image_files[elem+offset]).convert("RGB")))
        else:
            break
    return torch.stack(images)


image_files = get_images("data")
image_tensors: torch.Tensor = create_batch(image_files, 32, 0)
print(image_tensors)
print("Shape : ", image_tensors.shape)

'''
img_tensor = decode_image("/home/arun/dev/transformer/data/blue.png")
# image = preprocess_image("/home/arun/dev/transformer/data/blue.png")
print(img_tensor)
'''
