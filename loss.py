from nets import get_pose_net
import torch
from torch.utils import data
from cocotask import panopticDataset, collate
from augmentations import Augmentation
import h5py

def focalloss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds1 = gt.eq(1).float()
  pos_inds2 = gt.eq(2).float()
  neg_inds = gt.lt(1).float()

  pos_loss1 = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds1
  pos_loss2 = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds2 * 1.5
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

  num_pos  = pos_inds1.sum() + pos_inds2.sum()
  pos_loss = pos_loss1.sum() + pos_loss2.sum()
  neg_loss = neg_loss.sum()

  if not num_pos:
    return -neg_loss
  else:
    return -(pos_loss + neg_loss) / num_pos

class NetwithLoss(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.criter_for_cls = torch.nn.CrossEntropyLoss()
        self.criter_for_edge = focalloss
        self.net = net
    def forward(self, imgs, anns, edges):
        preds = self.net(imgs)
        loss_cls = self.criter_for_cls(preds['cls'], anns.type(torch.long))
        loss_edge = self.criter_for_edge(preds['edge'].sigmoid_(), edges)
        return loss_cls + loss_edge
