import numpy as np
import h5py
from os.path import join

from torch.utils.data import Dataset

class panopticDataset(Dataset):
    def __init__(self, f, aug):
        super().__init__()
        self.f = f
        with h5py.File(f, 'r') as data:
            self.Ids = list(data.keys())
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
        with h5py.File(self.f, 'r') as data:
            sample = data[self.Ids[index]]
            img = sample[:,:,:3]
            ann = sample[:,:,3]
            edge = sample[:,:,4]

        img, ann, edge = self.aug(img, ann, edge)
        return img, ann, edge
    name = 'panoptic-sementic'

