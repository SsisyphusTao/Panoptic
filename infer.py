# %%
import torch
import cv2 as cv
import numpy as np
from sys import argv
from nets import get_pose_net
from PIL import Image

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result
def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv.getAffineTransform(np.float32(src), np.float32(dst))

    return trans
def pre_process(image, meta=None):
    height, width = image.shape[0:2]

    inp_height, inp_width = [224,224]
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv.resize(image, (width, height))
    inp_image = cv.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv.INTER_LINEAR)
    inp_image = ((inp_image / 255. - [0.5]*3)).astype(np.float32)
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    images = torch.from_numpy(images)
    return images

model = 'checkpoints/ctnet_dla_008_968.pth'
imgpath = 'sheep-on-green-grass.jpg'

heads = {'cls': 81,
        'edge': 1}
net = get_pose_net(34, heads).cuda()
missing, unexpected = net.load_state_dict({k.replace('module.',''):v 
for k,v in torch.load(model).items()})
if missing:
    print('Missing:', missing)
if unexpected:
    print('Unexpected:', unexpected)
net.eval()

img = cv.imread(imgpath)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = pre_process(img)
# img = cv.resize(img, (224,224))

with torch.no_grad():
    # img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).type(torch.float).cuda()
    output = net(img.cuda())
#%%
# a = torch.softmax(output['cls'],1).squeeze().cpu().numpy()
a = output['cls'].squeeze().cpu().numpy()
b = output['edge'].sigmoid().squeeze().cpu().numpy()

b[np.where(b<0.3)] = 0
b[np.where(b>0)] = 255
c = a[19]

# %%
display(Image.fromarray(b.astype(np.uint8)))
display(Image.fromarray(c.astype(np.uint8)))

# %%
