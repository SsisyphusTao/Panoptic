import torch
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

class RandomColorTwist(object):
    def __init__(self, alpha=[0.5, 1.5], delta=[0.875, 1.125], gamma=[-0.5, 0.5]):
        self.contrast = ops.Uniform(range=gamma)
        self.brightness = ops.Uniform(range=[-0.125, 0.125])
        self.saturation = ops.Uniform(range=gamma)
        self.hue = ops.Uniform(range=gamma)
        self.ct = ops.ColorTwist(device="gpu")
        self.toss_a_coin = ops.CoinFlip(probability=0.5)

    # expects float image
    def __call__(self, images):
        images = self.ct(images,
                        brightness=self.brightness()*self.toss_a_coin()+1,
                        contrast=self.contrast()*self.toss_a_coin()+1,
                        hue=self.hue()*self.toss_a_coin(),
                        saturation=self.saturation()*self.toss_a_coin()+1)
        return images

class RandomFlip(object):
    def __init__(self):
        self.flip = ops.Flip(device="gpu")
        self.toss_a_coin = ops.CoinFlip(probability=0.5)

    def __call__(self, images, anns=None, gx=None, gy=None):
        coin1 = self.toss_a_coin()
        coin2 = self.toss_a_coin()
        images, anns, gx, gy = self.flip([images, anns, gx, gy], horizontal=coin1, vertical=coin2)
        return images, anns, gx, gy, coin1, coin2

class RandomPad(object):
    def __init__(self, size, fill_value):
        self.rz_img = ops.Resize(device = "gpu", resize_longer = size)
        self.rz_ann = ops.Resize(device = "gpu", resize_longer = size, interp_type=types.DALIInterpType.INTERP_NN)

        self.cp_img = ops.Crop(device="gpu", fill_values=fill_value, out_of_bounds_policy='pad')
        self.cp_ann = ops.Crop(device="gpu", fill_values=0, out_of_bounds_policy='pad')

        self.size = ops.Constant(fdata=size)
        self.pos = ops.Uniform(range=[0, 1])
        self.scale = ops.Uniform(range=[0.667, 1.5])
        self.shape = ops.Shapes(device="gpu")

    def __call__(self, images, anns, gx, gy):
        x = self.pos()
        y = self.pos()
        s = self.scale()
        window = self.size() * s
        shape = (self.shape(images) - window)/s

        images = self.cp_img(images, crop_w=window, crop_h=window, crop_pos_x=x, crop_pos_y=y)
        anns, gx, gy = self.cp_ann([anns, gx, gy], crop_w=window, crop_h=window, crop_pos_x=x, crop_pos_y=y)

        images = self.rz_img(images)
        anns, gx, gy = self.rz_ann([anns, gx, gy])

        return images, anns, gx, gy, x*shape, y*shape, s

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
    def __init__(self, size=512, mean=[0.47026115*255, 0.40789654*255, 0.44719302*255], std=[0.27809835*255, 0.28863828*255, 0.27408164*255]):
        self.randomct = RandomColorTwist()
        self.randompad = RandomPad(size, mean)
        self.normalize = Normalize(size, mean, std)
        self.flip = RandomFlip()

    def __call__(self, imgs, anns, gx, gy):
        imgs = self.randomct(imgs)
        imgs, anns, gx, gy, x, y, s = self.randompad(imgs, anns, gx, gy)
        imgs, anns, gx, gy, c1, c2 = self.flip(imgs, anns, gx, gy)
        imgs = self.normalize(imgs)
        return imgs, anns, gx, gy, x, y, s, c1, c2