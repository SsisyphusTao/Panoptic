import torch
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from os.path import join
import cv2 as cv
import numpy as np
from os import listdir
from random import shuffle

import sys
sys.path.append("..")
from dali_augmentations import Augmentation

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

class SegInputIterator(object):
    def __init__(self, batch_size, device_id, num_gpus):
        self.images_dir = '/ai/ailab/Share/TaoData/coco/panoptic/converted/imgs/'
        self.anns_dir = '/ai/ailab/Share/TaoData/coco/panoptic/converted/segs/'
        self.batch_size = batch_size
        self.files = listdir(self.images_dir)
        # whole data set size
        self.data_set_len = len(self.files)
        # based on the device_id and total number of GPUs - world size
        # get proper shard
        self.files = self.files[self.data_set_len * device_id // num_gpus:
                                self.data_set_len * (device_id + 1) // num_gpus]
        self.n = len(self.files)

    def __iter__(self):
        self.i = 0
        # shuffle(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []

        if self.i >= self.n:
            raise StopIteration

        for _ in range(self.batch_size):
            jpeg_filename = self.files[self.i]
            label_filename = self.files[self.i].replace('jpg','png')
            f = open(join(self.images_dir , jpeg_filename), 'rb')
            batch.append(np.frombuffer(f.read(), dtype = np.uint8))
            f = open(join(self.anns_dir , label_filename), 'rb')
            label = np.frombuffer(f.read(), dtype = np.uint8)
            labels.append(label)
            self.i = (self.i + 1) % self.n
        return (batch, labels)

    @property
    def size(self,):
        return self.data_set_len

    next = __next__

class panopticPipeline(Pipeline):
    def __init__(self, external, batch_size, num_threads, device_id):
        super().__init__(batch_size, num_threads, device_id)

        self.input_imgs = ops.ExternalSource()
        self.input_anns = ops.ExternalSource()
        self.decode_imgs = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.decode_anns = ops.ImageDecoder(device="mixed", output_type=types.GRAY)
        self.augment = Augmentation()
        self.external_data = external
        self.iterator = iter(self.external_data)

    def define_graph(self):
        self.images_ = self.input_imgs()
        self.anns_ = self.input_anns()

        images = self.decode_imgs(self.images_).gpu()
        anns = self.decode_anns(self.anns_).gpu()

        images, anns = self.augment(images, anns)
        return (images, anns)
    def iter_setup(self):
        try:
            (images, labels) = self.iterator.next()
            self.feed_input(self.images_, images)
            self.feed_input(self.anns_, labels)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration

def dali_makes_miricle():
    it = SegInputIterator(4,0,1)
    pipe = panopticPipeline(it, 4, 1, 0)
    train_loader = DALIGenericIterator(
        pipe,
        ["images", "anns"],
        fill_last_batch=True)
    for nbatch, data in enumerate(train_loader):
        images = data[0]["images"]
        anns = data[0]["anns"]
        break
    print(images)
    edges = torch.zeros_like(anns)
    neighbour=torch.cat((anns[:,1:],anns[:,-1:]), 1)
    edges += (anns != (0 or neighbour)).type(torch.uint8)
    neighbour=torch.cat((anns[:,:1], anns[:,:-1]),1)
    edges += (anns != (0 or neighbour)).type(torch.uint8)
    neighbour=torch.cat((anns[:,:,:1], anns[:,:,:-1]), 2)
    edges += (anns != (0 or neighbour)).type(torch.uint8)
    neighbour=torch.cat((anns[:,:,1:], anns[:,:,-1:]), 2)
    edges += (anns != (0 or neighbour)).type(torch.uint8)

    ann =anns.cpu().numpy()[3]
    edge = edges.cpu().numpy()[3]
    img =cv.cvtColor(images.cpu().numpy()[3], cv.COLOR_RGB2BGR)
    print(list(map(np.shape, [img, ann, edge])))
    edge = cv.cvtColor(edge, cv.COLOR_GRAY2BGR)
    ann = cv.cvtColor(ann, cv.COLOR_GRAY2BGR)

    edge[np.where(edge>0)] = 255
    show = cv.hconcat([img,ann,edge])
    cv.imwrite('out.jpg', show)

if __name__ == "__main__":
    dali_makes_miricle()