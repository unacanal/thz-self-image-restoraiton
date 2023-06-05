import PIL
import numpy as np
import random
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from source_target_transforms import *


class DataSampler:
    def __init__(self, img, target, crop_size):
        self.img = img
        self.target = target
        self.pairs = self.target, self.img

        self.transform = transforms.Compose([
            RandomRotationFromSequence([0, 90, 180, 270]),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomCrop(crop_size),
            ToTensor()]) 

    def generate_data(self):
        while True:
            gt, inp = self.pairs
            gt_tensor, inp_tensor = self.transform((gt, inp))
            gt_tensor = torch.unsqueeze(gt_tensor, 0)
            inp_tensor = torch.unsqueeze(inp_tensor, 0)
            yield gt_tensor, inp_tensor


class PairedDataSampler:
    """
    Args:
        inp_minmax: tuple (0.2, 0.3)
        gt_minmax: tuple (2.1, 2.2)
    """
    def __init__(self, name, inp_minmax, gt_minmax, crop_size):
        inp_min, inp_max = map(float, inp_minmax)
        gt_min, gt_max = map(float, gt_minmax)
        
        self.inp_paths = []
        for thz in np.arange(inp_min, round(inp_max + 0.1, 2), 0.1):
            thz = round(thz, 2)
            self.inp_paths.append(f'thz/{name}_{thz}THz.png')

        self.gt_paths = []
        for thz in np.arange(gt_min, round(gt_max + 0.1, 2), 0.1):
            thz = round(thz, 2)
            self.gt_paths.append(f'thz/{name}_{thz}THz.png')

        # When two paths length different
        if len(self.inp_paths) != len(self.gt_paths):
            inp_len = len(self.inp_paths)
            self.inp_paths = np.concatenate([([i]*len(self.gt_paths)) for i in self.inp_paths], axis=0)
            self.gt_paths = self.gt_paths * inp_len
        
        self.transform = transforms.Compose([
            RandomRotationFromSequence([0, 90, 180, 270]),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomCrop(crop_size),
            ToTensor()]) 

    def generate_data(self):
        while True:
            index = random.randrange(len(self.gt_paths))
            gt_path = self.gt_paths[index]
            inp_path = self.inp_paths[index]
            
            gt = PIL.Image.open(gt_path)
            inp = PIL.Image.open(inp_path)
            
            gt_tensor, inp_tensor = self.transform((gt, inp))
            gt_tensor = torch.unsqueeze(gt_tensor, 0)
            inp_tensor = torch.unsqueeze(inp_tensor, 0)
            yield gt_tensor, inp_tensor
