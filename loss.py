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
        edges = edges.reshape(-1,128,4,128,4).permute(0,2,4,1,3).flatten(1,2)

        w0 = edges.sum(1).eq(0).float()
        w1 = edges.sum(1).gt(0).float()
        loss_cls = self.criterion(preds['cls'], anns.type(torch.long))
        loss_cls = loss_cls * w0 + loss_cls * w1 * 0.1
        loss_edge = self.criter_for_edge(preds['edge'].sigmoid_(), edges)
        return loss_cls.mean() + loss_edge, loss_cls, loss_edge
