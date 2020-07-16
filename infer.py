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

    mean=np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32)
    std=np.array([0.28863828, 0.27408164, 0.27809835],
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
    for i, t in enumerate(tuple(text)):
        cv.putText(img, t, (i*50,20), 0, 0.5, tuple(text[t]), 1)
    return img

model = 'checkpoints/ctnet_dla_018_1047.pth'
imgpath = 'image3.jpg'

heads = {'cls': 81,
         'edge': 16}
net = get_pose_net(34, heads).cuda()
missing, unexpected = net.load_state_dict(torch.load(model))
if missing:
    print('Missing:', missing)
if unexpected:
    print('Unexpected:', unexpected)
net.eval()

img = cv.imread(imgpath)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = pre_process(img)

with torch.no_grad():
    output = net(img.cuda())
a = torch.softmax(output['cls'],1).squeeze()
a = torch.argmax(a, 0)
a = a.cpu().numpy().tolist()
b = output['edge'].cpu().sigmoid().squeeze()
b = b.reshape(4,4,128,128).permute(2,0,3,1).reshape(512,512).numpy()
b[np.where(b<0.3)] = 0
b[np.where(b>0)] = 255

edge = cv.cvtColor(b.astype(np.uint8), cv.COLOR_GRAY2BGR)
seg = visualize(a)

# show = cv.hconcat([edge, seg])
cv.imwrite('out.jpg', edge)