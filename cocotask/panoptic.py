import cv2 as cv
import numpy as np
import json
import h5py
from os.path import join
from tqdm import tqdm

from torch.utils.data import Dataset

def serialize():
    root = '/root/data/coco/'
    # with open(join(root, 'panoptic/annotations/panoptic_val2017.json'), 'r') as f:
    #     cocoval = json.load(f)
    with open(join(root, 'panoptic/annotations/panoptic_train2017.json'), 'r') as f:
        cocotrain = json.load(f)

        # for i in cocoval['annotations']:
        #     img_name = i['file_name'].replace('.png', '.jpg')
        #     img = cv.cvtColor(cv.imread(join(root, 'val2017', img_name)), cv.COLOR_BGR2RGB)
        #     ann = cv.cvtColor(cv.imread(join(root, 'panoptic/annotations/converted_anns', i['file_name'])), cv.COLOR_BGR2RGB)
        #     data = np.concatenate([img,ann],-1)
        #     f.create_dataset(str(i['image_id']), data=data)
    l = len(cocotrain['annotations'])
    with tqdm(total=l) as bar:
        with h5py.File('/ai/ailab/Share/TaoData/panoptic.hdf5', 'w') as f:
            for i in cocotrain['annotations']:
                img_name = i['file_name'].replace('.png', '.jpg')
                img = cv.cvtColor(cv.imread(join(root, 'train2017', img_name)), cv.COLOR_BGR2RGB)
                ann = cv.cvtColor(cv.imread(join(root, 'panoptic/annotations/converted_imgs', i['file_name'])), cv.COLOR_BGR2GRAY)
                left_edge = np.zeros_like(ann, dtype=np.uint8)
                down_edge = np.zeros_like(ann, dtype=np.uint8)
                shape = np.shape(ann)
                for j in range(shape[0]):
                    for k in range(shape[1]):
                        if not ann[j][k] == ann[j][min(k+1, shape[1]-1)]:
                            left_edge[j][k] = 1
                            left_edge[j][min(k+1, shape[1]-1)] = 1
                        
                        if not ann[j][k] == ann[min(j+1, shape[0]-1)][k]:
                            down_edge[j][k] = 1
                            down_edge[j][min(k+1, shape[1]-1)] = 1

                new_label = np.stack([ann, left_edge+down_edge], -1)
                data = np.concatenate([img,new_label],-1)
                f.create_dataset('%012d'%int(i['image_id']), data=data)
                bar.update(1)

class panopticDataset(Dataset):
    def __init__(self, f, aug):
        super().__init__()
        self.data = f
        self.Ids = list(self.data.keys())
        self.aug = aug

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
        edge[np.where(edge==2)] = 1

        img, ann, edge = self.aug(img, ann, edge)
        return img, ann.astype(np.int), edge
    name = 'panoptic-sementic'

if __name__ == "__main__":
    serialize()