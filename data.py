import PIL
import numpy as np
import random
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from source_target_transforms import *


try:
    BICUBIC = PIL.Image.Resampling.BICUBIC
except AttributeError:  # Pillow < 9.1
    BICUBIC = PIL.Image.BICUBIC


class DataSampler:
    def __init__(self, img, target, crop_size, scale=1):
        self.target = target
        if scale > 1:
            # ZSSR: build a low-resolution (blurred) counterpart of the target by
            # bicubic down-/up-sampling, so the network learns to restore the
            # high-frequency details lost when downscaling by `scale`.
            w, h = target.size
            lr = target.resize((w // scale, h // scale), BICUBIC)
            self.img = lr.resize((w, h), BICUBIC)
        else:
            self.img = img
        self.pairs = self.target, self.img

        self.transform = transforms.Compose([
            RandomRotationFromSequence([0, 90, 180, 270]),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomCrop(crop_size),
            ToTensor(),
        ])
        # self.normalize = transforms.Normalize((0.5, ), (0.5, ))

    def generate_data(self):
        while True:
            gt, inp = self.pairs
            gt_tensor, inp_tensor = self.transform((gt, inp))
            # inp_tensor = self.normalize(inp_tensor)
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
            self.inp_paths.append(f'thz_new/{name}_{thz}THz.png')

        self.gt_paths = []
        for thz in np.arange(gt_min, round(gt_max + 0.1, 2), 0.1):
            thz = round(thz, 2)
            self.gt_paths.append(f'thz_new/{name}_{thz}THz.png')

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


class TripleDataSampler:
    def __init__(self, img, low_target, high_target, crop_size):
        self.pairs = low_target, high_target, img

        self.transform = transforms.Compose([
            RandomRotationFromSequenceV2([0, 90, 180, 270]),
            RandomHorizontalFlipV2(),
            RandomVerticalFlipV2(),
            RandomCropV2(crop_size),
            ToTensorV2()]) 

    def generate_data(self):
        while True:
            low_gt, high_gt, inp = self.pairs
            low_gt_tensor, high_gt_tensor, inp_tensor = self.transform((low_gt, high_gt, inp))
            
            low_gt_tensor = torch.unsqueeze(low_gt_tensor, 0)
            high_gt_tensor = torch.unsqueeze(high_gt_tensor, 0)
            inp_tensor = torch.unsqueeze(inp_tensor, 0)
            yield low_gt_tensor, high_gt_tensor, inp_tensor
            
            
if __name__ == '__main__':
    gt1 = PIL.Image.open("thz/resol1951_2_e_2.0THz.png")
    gt2 = PIL.Image.open("thz/resol1951_2_e_2.8THz.png")
    img = PIL.Image.open("thz/resol1951_2_e_2.3THz.png")
    img.save('img0.png')
    sampler = TripleDataSampler(img, gt1, gt2, 128)
    import cv2
    for x in sampler.generate_data():
        print(x.shape)
        gt1, gt2, img = x
        gt1 = gt1.numpy().squeeze(0).transpose((1, 2, 0))
        gt2 = gt2.numpy().squeeze(0).transpose((1, 2, 0))
        img = img.numpy().squeeze(0).transpose((1, 2, 0))
        cv2.imwrite('gt1.png', gt1)
        cv2.imwrite('gt2.png', gt2)
        cv2.imwrite('img.png', img)
        break
        