import torch
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from os.path import join
import numpy as np
from os import listdir

def create_edge(anns):
    edges = torch.zeros_like(anns)
    neighbour=torch.cat((anns[:,1:],anns[:,-1:]), 1)
    edges += (anns != (0 or neighbour)).type(torch.uint8)
    neighbour=torch.cat((anns[:,:1], anns[:,:-1]),1)
    edges += (anns != (0 or neighbour)).type(torch.uint8)
    neighbour=torch.cat((anns[:,:,:1], anns[:,:,:-1]), 2)
    edges += (anns != (0 or neighbour)).type(torch.uint8)
    neighbour=torch.cat((anns[:,:,1:], anns[:,:,-1:]), 2)
    edges += (anns != (0 or neighbour)).type(torch.uint8)
    return edges

class panopticInputIterator(object):
    def __init__(self, batch_size):
        try:
            shard_id = torch.distributed.get_rank()
            num_shards = torch.distributed.get_world_size()
        except AssertionError:
            shard_id = 0
            num_shards = 1

        self.images_dir = '/ai/ailab/Share/TaoData/coco/panoptic/converted/imgs/'
        self.anns_dir = '/ai/ailab/Share/TaoData/coco/panoptic/converted/segs/'
        self.batch_size = batch_size
        self.files = listdir(self.images_dir)
        self.data_set_len = len(self.files)
        self.files = self.files[self.data_set_len * shard_id // num_shards:
                                self.data_set_len * (shard_id + 1) // num_shards]
        self.n = len(self.files)

    def __iter__(self):
        self.i = 0
        np.random.shuffle(self.files)
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
    def __init__(self, external, augment, batch_size, num_threads, device_id):
        super().__init__(batch_size, num_threads, device_id)

        self.input_imgs = ops.ExternalSource()
        self.input_anns = ops.ExternalSource()
        self.decode_imgs = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.decode_anns = ops.ImageDecoder(device="mixed", output_type=types.GRAY)
        self.augment = augment
        self.external_data = external
        self.iterator = iter(self.external_data)

    def define_graph(self):
        self.images = self.input_imgs()
        self.anns = self.input_anns()

        images = self.decode_imgs(self.images)
        anns = self.decode_anns(self.anns)

        images, anns = self.augment(images, anns)
        return (images, anns)

    def iter_setup(self):
        try:
            (images, labels) = self.iterator.next()
            self.feed_input(self.images, images)
            self.feed_input(self.anns, labels)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration