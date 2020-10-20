import torch
import cv2 as cv
import numpy as np
from sys import argv
from nets import get_pose_net
from PIL import Image
import json
from cocotask import class_name



def pre_process(image):
    height, width = image.shape[0:2]
    mean=np.array([0.47026115, 0.40789654, 0.44719302],
                dtype=np.float32)
    std=np.array([0.27809835, 0.28863828, 0.27408164],
                    dtype=np.float32)
    
    image = (image.astype(np.float32) / 255. - mean) / std
    c = max(height, width)
    image = torch.from_numpy(image)
    canvas = torch.zeros(c,c,3)
    canvas[(c-height)//2:c//2+height//2,(c-width)//2:c//2+width//2,:] = image[:,:,:]
    canvas = canvas.permute(2,0,1).unsqueeze(0)
    image = torch.nn.functional.interpolate(canvas, (512,512), mode='bilinear', align_corners=True)
    return image

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep
def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat
def _topk(scores, K=10):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds.true_divide(width)).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind.true_divide(K)).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

model = 'checkpoints/dla_instance_038_9234.pth'
imgpath = 'images/iceland_sheep.jpg'

net = get_pose_net(34, {'hm': 80, 'grad': 2, 'mask':1}).cuda()
missing, unexpected = net.load_state_dict(torch.load(model))
net.eval()

img = bg = cv.imread(imgpath)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = pre_process(img)

with torch.no_grad():
    output = net(img.cuda())
pred = output['hm'].sigmoid()
grad = output['grad']
mask = output['mask']

m = mask.sigmoid().ge(0.5).type_as(pred)
pred = pred * m.expand_as(pred)
sc,_,_,xs,ys = _topk(_nms(pred))

# grad = torch.nn.functional.interpolate(grad, size=[512,512], mode='bilinear', align_corners=True)[0]
# mask = torch.nn.functional.interpolate(mask, size=[512,512], mode='bilinear', align_corners=True)[0].sigmoid()


# pred = pred.where(mask>0.5, torch.zeros_like(pred))

# gx = torch.from_numpy(np.expand_dims(np.array([x for x in range(512)]), 0).repeat(512, 0)).cuda() + grad[0]
# gy = torch.from_numpy(np.expand_dims(np.array([x for x in range(512)]), 1).repeat(512, 1)).cuda() + grad[1]
# gx = (gx//4).clamp(0,127).type(torch.long).cpu()
# gy = (gy//4).clamp(0,127).type(torch.long).cpu()
# seg = gx.map_(gy, lambda x,y: pred[y][x]).type(torch.long)

print(sc)