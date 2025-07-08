import PIL.Image
import torch
import PIL
import torchvision.transforms as transforms
import time
def display_image_tensor(input: torch.Tensor) :
    to_pil = transforms.ToPILImage()
    image: PIL.Image.Image = to_pil(input)
    image.show()
    
def save_image_tensor(pth: str = "data/output/"):
    pass
    
    
    
    

test_tensor = torch.randint(100,255,(3,800,800),dtype=torch.uint8)
display_image_tensor(test_tensor)
    