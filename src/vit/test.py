import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
import queue
from .ImageLoader import ImageLoader
def read_image(imageFile : str):
    image = cv.imread(imageFile)
    return image
def reshape(image : cv.Mat):
    image = cv.resize(image,(800,800))
    return image

image = read_image("data/blue.png")
cv.imshow("window1",image)
cv.waitKey(0)
image = reshape(image)
cv.imshow("resized image",image)
cv.waitKey(0)
image_tensor = torch.tensor(image)
print(image_tensor)
cv.cvtColor(image,cv.COLOR_BGR2RGB,image)
# input()
image_tensor = torch.tensor(image)
print(image_tensor)
loader = ImageLoader(8)
