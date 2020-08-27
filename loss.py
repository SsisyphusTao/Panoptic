from nets import get_pose_net
import torch
from torch.utils import data
from cocotask import panopticDataset, collate
from augmentations import Augmentation

def focalloss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.gt(0).float()
  neg_inds = gt.lt(1).float()

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

  num_pos  = pos_inds.sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if not num_pos:
    return -neg_loss
  else:
    return -(pos_loss + neg_loss) / num_pos

class NetwithLoss(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.criter_for_cls = torch.nn.CrossEntropyLoss(reduction='none')
        self.criter_for_edge = focalloss
        self.net = net
    @torch.cuda.amp.autocast()
    def forward(self, images, anns, edges):
        # import cv2 as cv
        # import numpy as np
        # img = images[0].permute(1,2,0).detach().cpu().numpy()
        # ann = cv.cvtColor(anns[0].detach().cpu().numpy(), cv.COLOR_GRAY2BGR).astype(img.dtype)
        # cv.imwrite('img%d.jpg'%r, cv.hconcat([img*128+128, ann]))
        preds = self.net(images)
        segs = torch.nn.functional.interpolate(preds['cls'], size=images.size()[2:], mode='bilinear', align_corners=True)
        w = preds['edge'].reshape(-1,4,4,128,128).permute(0,3,1,4,2).flatten(3,4).flatten(1,2).sigmoid()

        loss_cls = self.criter_for_cls(segs, anns.squeeze().type(torch.long)) * (1-edges.unsqueeze(1))
        if edges.gt(0).float().sum():
          loss_edge = self.criter_for_edge(w, edges)
          return loss_cls + loss_edge * 0.1
        else:
          return loss_cls