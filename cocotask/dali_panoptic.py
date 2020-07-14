import torch
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from os.path import join

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

class panopticPipeline(Pipeline):
    def __init__(self, root, augment, batch_size, num_threads, device_id):
        super().__init__(batch_size, num_threads, device_id)

        self.input_imgs = ops.FileReader(file_root = join(root, 'imgs'), random_shuffle=True, seed=0)
        self.input_anns = ops.FileReader(file_root = join(root, 'anns'), random_shuffle=True, seed=0)
        self.decode_imgs = ops.ImageDecoder(device="cpu", output_type=types.RGB)
        self.decode_anns = ops.ImageDecoder(device="cpu", output_type=types.GRAY)
        self.augment = augment

    def define_graph(self):
        images_, _ = self.input_imgs()
        anns_, _ = self.input_anns()

        images = self.decode_imgs(images_).gpu()
        anns = self.decode_anns(anns_).gpu()

        images, anns = self.augment(images, anns)
        return (images, anns)