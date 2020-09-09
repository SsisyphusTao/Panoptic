import torch
import cv2 as cv
import numpy as np
from sys import argv
from nets import get_pose_net
from PIL import Image
import json
from cocotask import class_name

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

    inp_height, inp_width = [512,512]
    c = np.array([width / 2., height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv.resize(image, (width, height))
    inp_image = cv.warpAffine(
      resized_image, trans_input, (inp_width, inp_height),
      flags=cv.INTER_LINEAR)

    mean=np.array([0.47026115, 0.40789654, 0.44719302],
                   dtype=np.float32)
    std=np.array([0.27809835, 0.28863828, 0.27408164],
                   dtype=np.float32)

    inp_image = (inp_image.astype(np.float32) / 255. - mean) / std
    images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
    images = torch.from_numpy(images)
    return images

def visualize(c):
    with open('/ai/ailab/Share/TaoData/coco/panoptic/annotations/panoptic_coco_categories.json') as f:
        color = json.load(f)
    text = {}
    for i, x in enumerate(c):
        for j, y in enumerate(x):
            if c[i][j]:
                text.update({class_name[c[i][j]]: color[c[i][j]-1]['color']})
                c[i][j] = color[c[i][j]-1]['color']
            else:
                c[i][j] = [0, 0, 0]
    img = np.array(c, dtype=np.uint8)
    return img, text

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

model = 'checkpoints/dla_instance_040_847.pth'
imgpath = 'images/image1.jpg'

net = get_pose_net(34, {'hm': 80, 'grad': 2}).cuda()
missing, unexpected = net.load_state_dict(torch.load(model))
net.eval()

img = bg = cv.imread(imgpath)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = pre_process(img)

with torch.no_grad():
    output = net(img.cuda())
pred = output['hm'].squeeze().sigmoid().cpu()
grad = output['grad'].squeeze().tanh().cpu()
background = torch.ones(1,128,128) * 0.2
pred = torch.cat([background, pred])
# pred = torch.softmax(pred, 0)
pred = torch.argmax(pred, 0)
# clamp= lambda x : min(max(int(x), 127),0)
# gx = torch.from_numpy(np.expand_dims(np.array([x for x in range(128)]), 0).repeat(128, 0)) + grad[0]*128
# gy = torch.from_numpy(np.expand_dims(np.array([x for x in range(128)]), 1).repeat(128, 1)) + grad[1]*128
# pred = gx.map_(gy.cpu(), lambda x,y:pred[clamp(x)][clamp(y)])

s = (grad[0].pow(2)+grad[1].pow(2)).sqrt().cpu().numpy()*128*2
seg, text = visualize(pred.numpy().astype(np.int).tolist())
seg = cv.resize(seg, (512,512), interpolation=cv.INTER_NEAREST)
print(text)
# show = seg

# height, width = bg.shape[0:2]

# inp_height, inp_width = [512,512]
# c = np.array([width / 2., height / 2.], dtype=np.float32)
# s = max(height, width) * 1.0

# trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height], inv=1)
# mask = cv.warpAffine(
#     show, trans_input, (width, height),
#     flags=cv.INTER_NEAREST)

for i, t in enumerate(tuple(text)):
    cv.putText(seg, t, (10,30*(i+1)), 0, 1, tuple(text[t]), 2)

cv.imwrite('images/out.jpg', s)
cv.imwrite('images/mid.jpg', seg)