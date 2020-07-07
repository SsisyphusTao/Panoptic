import cv2 as cv
import numpy as np
import json
import h5py
from os.path import join
from tqdm import tqdm

from torch.utils.data import Dataset

class panopticDataset(Dataset):
    def __init__(self, f, aug):
        super().__init__()
        self.data = f
        self.Ids = list(self.data.keys())
        self.aug = aug
        np.random.shuffle(self.Ids)

    def __len__(self):
        return len(self.Ids)

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i
    
    def __getitem__(self, index):
        sample = self.data[self.Ids[index]]
        img = sample[:,:,:3]
        ann = sample[:,:,3]
        edge = sample[:,:,4]

        ann[np.where(ann>90)] = 0

        img, ann, edge = self.aug(img, ann, edge)
        return img, ann.astype(np.int), edge
    name = 'panoptic-sementic'

