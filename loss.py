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
  neg_inds = gt.eq(0).float()

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
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.criter_for_edge = focalloss
        self.net = net

    def forward(self, imgs, anns, edges):
        preds = self.net(imgs)
        segs = torch.nn.functional.interpolate(preds['cls'], size=imgs.size()[2:], mode='bilinear', align_corners=True)
        w = preds['edge'].reshape(-1,4,4,128,128).permute(0,3,1,4,2).flatten(3,4).flatten(1,2).sigmoid()
        # edges = edges.reshape(-1,128,4,128,4).permute(0,2,4,1,3).flatten(1,2)

        loss_cls = self.criterion(segs, anns.type(torch.long))
        pt = torch.exp(-loss_cls)

        loss_cls = loss_cls * ((1 - pt) ** 2) * (w.detach()+edges)
        loss_edge = self.criter_for_edge(w, edges)
        return loss_cls.mean() + loss_edge, loss_cls, loss_edge
