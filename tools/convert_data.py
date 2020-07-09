import cv2 as cv
import numpy as np
import json
import h5py
from os.path import join
from tqdm import tqdm

from torch.utils.data import Dataset

_valid_ids = [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 
      24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 
      37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 
      48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
      58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 
      72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 
      82, 84, 85, 86, 87, 88, 89, 90]
cat_ids = {v: i+1 for i, v in enumerate(_valid_ids)}
cat_ids.update({0: 0})
fc = np.vectorize(lambda x: cat_ids[x])
def serialize():
    root = '/ai/ailab/Share/TaoData/coco/'
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
        with h5py.File('/ai/ailab/Share/TaoData/sementic.h5', 'w') as f:
            for i in cocotrain['annotations']:
                img_name = i['file_name'].replace('.png', '.jpg')
                img = cv.cvtColor(cv.imread(join(root, 'images/train2017', img_name)), cv.COLOR_BGR2RGB)
                ann = cv.cvtColor(cv.imread(join(root, 'panoptic/converted_anns', i['file_name'])), cv.COLOR_BGR2GRAY)
                shape = np.shape(ann)
                
                ann[np.where(ann>90)] = 0
                ann = fc(ann).astype(np.uint8)
                seg = ann.copy()
                edge = np.zeros_like(ann, np.uint8)

                for n in range(10):
                    edge[np.where(edge>0)] += 1
                    neighbour=np.vstack((ann[1:],ann[-1]))
                    edge += (ann != (0 or neighbour)).astype(np.uint8)
                    neighbour=np.vstack((ann[0], ann[:-1]))
                    edge += (ann != (0 or neighbour)).astype(np.uint8)
                    neighbour=np.hstack((ann[:,0:1], ann[:,:-1]))
                    edge += (ann != (0 or neighbour)).astype(np.uint8)
                    neighbour=np.hstack((ann[:,1:], ann[:,-1:]))
                    edge += (ann != (0 or neighbour)).astype(np.uint8)
                    ann[np.where(edge>0)] = 0
                new_label = np.stack([seg, edge], -1)
                data = np.concatenate([img,new_label],-1)
                f.create_dataset('%012d'%int(i['image_id']), data=data)
                bar.update(1)

if __name__ == "__main__":
    serialize()