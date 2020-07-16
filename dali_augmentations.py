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
        images = self.flip(images, horizontal=coin1)
        anns = self.flip(anns, horizontal=coin1)
        return images, anns

class RandomResizedCrop(object):
    def __init__(self, size=512):
        self.size = size
        # self.rt_img = ops.Rotate(device="gpu", fill_value=0, keep_size=True)
        # self.rt_ann = ops.Rotate(device="gpu", fill_value=0, keep_size=True, interp_type=types.DALIInterpType.INTERP_NN)
        self.cp = ops.Crop(device="gpu", crop=[self.size, self.size], output_dtype=types.UINT8)
        self.rz_img = ops.Resize(device="gpu")
        self.rz_ann = ops.Resize(device="gpu", interp_type=types.DALIInterpType.INTERP_NN)

        self.toss_a_coin = ops.CoinFlip(probability=0.5)
        self.angle = ops.Uniform(range=[0, 90])
        self.ratio = ops.Uniform(range=[1, 1.5])
        self.pos = ops.Uniform(range=[0.2,0.8])

    def __call__(self, images, anns):
        # if self.toss_a_coin():
        #     angle = self.angle()
        #     images= self.rt_img(images, angle=angle)
        #     anns = self.rt_ann(anns, angle=angle)

        r = self.ratio()
        x = self.pos()
        y = self.pos()

        images = self.rz_img(images, resize_shorter=r * self.size)
        anns = self.rz_ann(anns, resize_shorter=r * self.size)

        images = self.cp(images, crop_pos_x=x, crop_pos_y=y)
        anns = self.cp(anns, crop_pos_x=x, crop_pos_y=y)

        return images, anns

class RandomPad(object):
    def __init__(self, size, fill_value):
        self.rz_img = ops.Resize(device = "gpu", resize_longer = size)
        self.rz_ann = ops.Resize(device = "gpu", resize_longer = size, interp_type=types.DALIInterpType.INTERP_NN)
        self.pt_img = ops.Paste(device = "gpu", ratio = 1, min_canvas_size = size, fill_value=fill_value)
        self.pt_ann = ops.Paste(device = "gpu", ratio = 1, min_canvas_size = size, fill_value=0)
        self.pos = ops.Uniform(range=[0.05, 0.95])

    def __call__(self, images, anns):
        x = y = self.pos()
        return self.pt_img(self.rz_img(images), paste_x=x, paste_y=y), self.pt_ann(self.rz_ann(anns), paste_x=x, paste_y=y)

class Normalize(object):
    def __init__(self, size, mean=None, std=None):
        self.nl = ops.CropMirrorNormalize(
            device="gpu",
            crop=(size, size),
            mean=mean,
            std=std,
            mirror=0,
            output_dtype=types.FLOAT,
            output_layout=types.NCHW,
            pad_output=False)
        
    def __call__(self, images):
        return self.nl(images)

class Augmentation(object):
    def __init__(self, size=512, mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255]):
        # self.toss_a_coin = ops.CoinFlip(probability=0.75)
        self.randomct = RandomColorTwist()
        self.randomrrc = RandomResizedCrop(size)
        self.randompad = RandomPad(size, mean)
        self.normalize = Normalize(size, mean, std)
        self.flip = RandomFlip()
        self.cast = ops.Cast(device="gpu", dtype=types.FLOAT)
        self.resize = ops.Resize(device = "gpu", resize_longer = size/4, interp_type=types.DALIInterpType.INTERP_NN)

    def __call__(self, imgs, anns):
        imgs = self.randomct(imgs)
        # if self.toss_a_coin():
        #     imgs, anns = self.randomrrc(imgs, anns)
        # else:
        imgs, anns = self.randompad(imgs, anns)
        imgs, anns = self.flip(imgs, anns)

        imgs = self.normalize(imgs)
        return imgs, self.resize(anns), anns