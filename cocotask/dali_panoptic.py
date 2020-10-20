import torch
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from os.path import join
import numpy as np
import cv2 as cv
from os import listdir

def view_sample(**kwargs):
    out = []
    for i in kwargs:
        if kwargs[i].dim() == 4 and kwargs[i].size()[1] == 3:
            img = kwargs[i][0].permute(1,2,0).detach().cpu().numpy()
            out.append(img)
        if kwargs[i].dim() == 4 and kwargs[i].size()[-1] == 3:
            img = kwargs[i][0].detach().cpu().numpy()
            out.append(img)
        elif kwargs[i].dim() == 4 and kwargs[i].size()[-1] == 1:
            img = cv.cvtColor(kwargs[i][0].detach().cpu().numpy(), cv.COLOR_GRAY2BGR).astype(np.uint8)
            out.append(img)
        elif kwargs[i].dim() == 3:
            img = cv.cvtColor(kwargs[i][0].detach().cpu().numpy(), cv.COLOR_GRAY2BGR).astype(np.uint8)
            out.append(img)
    cv.imwrite('images/debug.jpg', cv.hconcat(out))
    raise RuntimeError

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
    return edges.squeeze()

def grad_preprocess(batch):
    reverse = lambda x : x[...,0]*100+x[...,1]*10+x[...,2]
    gx = reverse(batch['gx'].type(torch.float))
    gy = reverse(batch['gy'].type(torch.float))

    s = batch['s'].unsqueeze(-1).expand_as(gx).cuda()
    x = batch['x'][:,1].unsqueeze(-1).unsqueeze(-1).expand_as(gx)
    y = batch['y'][:,0].unsqueeze(-1).unsqueeze(-1).expand_as(gy)

    gx = torch.where(gx>0, gx/s-x, gx)
    gy = torch.where(gy>0, gy/s-y, gy)

    c1 = batch['c1'].unsqueeze(-1).expand_as(gx).cuda()
    c2 = batch['c2'].unsqueeze(-1).expand_as(gy).cuda()

    gx = torch.where(gx*c1>0, 511-gx, gx).clamp_min(0)
    gy = torch.where(gy*c2>0, 511-gy, gy).clamp_min(0)

    gx = gx.where(gx<512, torch.zeros_like(gx))
    gy = gy.where(gy<512, torch.zeros_like(gy))

    x = torch.from_numpy(np.expand_dims(np.array([x for x in range(512)]), 0).repeat(512, 0)).cuda()
    y = torch.from_numpy(np.expand_dims(np.array([x for x in range(512)]), 1).repeat(512, 1)).cuda()

    index = torch.stack([gx,gy], 1).permute(0,2,3,1)
    index = torch.cat([index, batch['anns']], -1).flatten(1,2).type(torch.long)
    index = index.unique(dim=1)
    dim = torch.tensor([[x] for x in range(index.size()[0])], dtype=torch.long).cuda()
    dim = dim.expand(-1, index.size()[1])
    ann = torch.zeros(batch['anns'].size()[0],32,32, dtype=torch.long).cuda()
    ann.index_put_((dim,index[...,1]//16,  \
                        index[...,0]//16), \
                        index[...,-1])
    return torch.where(gx>0, gx-x, gx), torch.where(gy>0, gy-y, gy), ann

class panopticInputIterator(object):
    def __init__(self, batch_size):
        try:
            self.shard_id = torch.distributed.get_rank()
            self.num_shards = torch.distributed.get_world_size()
        except AssertionError:
            self.shard_id = 0
            self.num_shards = 1

        self.images_dir = '/ai/ailab/Share/TaoData/coco/panoptic/converted/imgs/'
        self.anns_dir = '/ai/ailab/Share/TaoData/coco/panoptic/converted/seg/'
        self.gx_dir = '/ai/ailab/Share/TaoData/coco/panoptic/converted/gx/'
        self.gy_dir = '/ai/ailab/Share/TaoData/coco/panoptic/converted/gy/'
        self.batch_size = batch_size
        self.files_folder = listdir(self.images_dir)
        self.data_set_len = len(self.files_folder)
        self.seed = 0
        self.shuffle()
        self.n = len(self.files)

    def shuffle(self):
        np.random.seed(self.seed)
        np.random.shuffle(self.files_folder)
        self.files = self.files_folder[self.data_set_len * self.shard_id // self.num_shards:
                        self.data_set_len * (self.shard_id + 1) // self.num_shards]
        self.seed += 1

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        batch = []
        labels = []
        gx = []
        gy = []

        if self.i >= self.n:
            raise StopIteration

        for _ in range(self.batch_size):
            jpeg_filename = self.files[self.i]
            label_filename = self.files[self.i].replace('jpg','png')
            with open(join(self.images_dir , jpeg_filename), 'rb') as f:
                batch.append(np.frombuffer(f.read(), dtype = np.uint8))
            with open(join(self.anns_dir , label_filename), 'rb') as f:
                labels.append(np.frombuffer(f.read(), dtype = np.uint8))
            with open(join(self.gx_dir , label_filename), 'rb') as f:
                gx.append(np.frombuffer(f.read(), dtype = np.uint8))
            with open(join(self.gy_dir , label_filename), 'rb') as f:
                gy.append(np.frombuffer(f.read(), dtype = np.uint8))
            self.i = (self.i + 1) % self.n
        return (batch, labels, gx, gy)

    @property
    def size(self,):
        return self.data_set_len

    next = __next__

class panopticPipeline(Pipeline):
    def __init__(self, external, augment, batch_size, num_threads, device_id):
        super().__init__(batch_size, num_threads, device_id)

        self.input_imgs = ops.ExternalSource()
        self.input_anns = ops.ExternalSource()
        self.input_gx = ops.ExternalSource()
        self.input_gy = ops.ExternalSource()
        self.decode_rgb = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.decode_gray = ops.ImageDecoder(device="mixed", output_type=types.GRAY)
        self.augment = augment
        self.external_data = external
        self.iterator = iter(self.external_data)

    def define_graph(self):
        self.images = self.input_imgs()
        self.anns = self.input_anns()
        self.gx = self.input_gx()
        self.gy = self.input_gy()

        images, gx, gy = self.decode_rgb([self.images,self.gx,self.gy])
        anns = self.decode_gray(self.anns)

        images, anns, gx, gy, x, y, s, c1, c2 = self.augment(images, anns, gx, gy)
        return (images, anns, gx, gy, x, y, s, c1, c2)

    def iter_setup(self):
        try:
            (images, labels, gx, gy) = self.iterator.next()
            self.feed_input(self.images, images)
            self.feed_input(self.anns, labels)
            self.feed_input(self.gx, gx)
            self.feed_input(self.gy, gy)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration