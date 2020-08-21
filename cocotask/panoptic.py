import cv2 as cv
import numpy as np
import json
import h5py
from os.path import join
# from tqdm import tqdm
from os import listdir

from torch.utils.data import Dataset

class panopticDataset(Dataset):
    def __init__(self, aug):
        super().__init__()
        self.Ids = [x.replace('.jpg', '') for x in listdir('/ai/ailab/Share/TaoData/coco/panoptic/converted/imgs/')]
        self.aug = aug
        np.random.shuffle(self.Ids)

    def __len__(self):
        return len(self.Ids)
    
    def __getitem__(self, index):
        img = cv.cvtColor(cv.imread('/ai/ailab/Share/TaoData/coco/panoptic/converted/imgs/'+self.Ids[index]+'.jpg'), cv.COLOR_BGR2RGB)
        ann = cv.cvtColor(cv.imread('/ai/ailab/Share/TaoData/coco/panoptic/converted/segs/'+self.Ids[index]+'.png'), cv.COLOR_BGR2GRAY)

        img, ann, edge = self.aug(img, ann, None)
        return img, ann
    name = 'panoptic-sementic'

