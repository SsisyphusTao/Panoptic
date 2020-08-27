import torch
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

class RandomColorTwist(object):
    def __init__(self, alpha=[0.5, 1.5], delta=[0.875, 1.125], gamma=[-0.5, 0.5]):
        self.contrast = ops.Uniform(range=alpha)
        self.brightness = ops.Uniform(range=delta)
        self.saturation = ops.Uniform(range=alpha)
        self.hue = ops.Uniform(range=gamma)
        self.ct = ops.ColorTwist(device="gpu")
        self.toss_a_coin = ops.CoinFlip(probability=0.75)

    # expects float image
    def __call__(self, images):
        if self.toss_a_coin():
            images = self.ct(images,
                            brightness=self.brightness(),
                            contrast=self.contrast(),
                            hue=self.hue(),
                            saturation=self.saturation())
        return images

class RandomFlip(object):
    def __init__(self):
        self.flip = ops.Flip(device="gpu")
        self.toss_a_coin = ops.CoinFlip(probability=0.5)

    def __call__(self, images, anns=None):
        coin1 = self.toss_a_coin()
        coin2 = self.toss_a_coin()
        images = self.flip(images, horizontal=coin1, vertical=coin2)
        anns = self.flip(anns, horizontal=coin1, vertical=coin2)
        return images, anns

class RandomPad(object):
    def __init__(self, size, fill_value):
        self.rz_img = ops.Resize(device = "gpu")
        self.rz_ann = ops.Resize(device = "gpu", interp_type=types.DALIInterpType.INTERP_NN)
        self.pt_img = ops.Paste(device = "gpu", ratio = 1, min_canvas_size = size, fill_value=fill_value)
        self.pt_ann = ops.Paste(device = "gpu", ratio = 1, min_canvas_size = size, fill_value=0)
        self.cp_img = ops.Crop(device="gpu", crop_h=size, crop_w=size, fill_values=fill_value, out_of_bounds_policy='pad')
        self.cp_ann = ops.Crop(device="gpu", crop_h=size, crop_w=size, fill_values=0, out_of_bounds_policy='pad')
        self.pos = ops.Uniform(range=[0.3, 0.7])
        self.ratio = ops.Uniform(range=[0.5*size, 1.5*size])

    def __call__(self, images, anns):
        x = self.pos()
        y = self.pos()
        r = self.ratio()
        images = self.cp_img(self.pt_img(self.rz_img(images, resize_longer=r)), crop_pos_x=x, crop_pos_y=y) 
        anns = self.cp_ann(self.pt_ann(self.rz_ann(anns, resize_longer=r)), crop_pos_x=x, crop_pos_y=y)
        
        return images, anns

class Normalize(object):
    def __init__(self, size, mean=None, std=None):
        self.nl = ops.CropMirrorNormalize(
            device="gpu",
            crop=(size, size),
            mean=mean,
            std=std,
            mirror=0,
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            pad_output=False)
        
    def __call__(self, images):
        return self.nl(images)

class Augmentation(object):
    def __init__(self, size=512, mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255]):
        self.toss_a_coin = ops.CoinFlip(probability=0.5)
        self.randomct = RandomColorTwist()
        self.randompad = RandomPad(size, mean)
        self.normalize = Normalize(size, mean, std)
        self.flip = RandomFlip()

    def __call__(self, imgs, anns):
        imgs = self.randomct(imgs)
        imgs, anns = self.randompad(imgs, anns)
        imgs, anns = self.flip(imgs, anns)
        imgs = self.normalize(imgs)
        return imgs, anns