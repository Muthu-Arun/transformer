import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
import queue
import os
def load_image_file_names(root_dir : str):
    files = os.listdir(root_dir)
    
class ImageLoader:
     
    def __init__(self,root_folder : str,batch_size : int):
        self.batch_size = batch_size
        self.root_dir = root_folder
        self.image_file_train_queue = queue.Queue(load_image_file_names(root_folder+"train"))
        self.image_file_val_queue = queue.Queue(load_image_file_names(root_folder+"val"))
        self.train = 0
        self.val = 1
        self.images = queue.Queue()
        # self.image_queue = queue()
    def load_next_batch(self,toggle : int):
        assert toggle != self.train or toggle != self.val
        if toggle == self.train:
            q : queue.Queue= self.image_file_train_queue
        elif toggle == self.val:
            q : queue.Queue= self.image_file_val_queue
        
        for i in range(self.batch_size):
            if q.empty():
                break
            self.images.put(cv.imread(self.root_dir+q.get()))
            
            
