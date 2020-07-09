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
  pos_inds = gt.ge(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds * neg_weights

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
        self.mean = torch.tensor([x for x in range(25)]).type(torch.float).mean().item()
        self.std = torch.tensor([x for x in range(25)]).type(torch.float).std().item()
    def forward(self, imgs, anns, edges):
        preds = self.net(imgs)
        anns = anns.type(torch.long)
        edges = edges.type(torch.float)
        edges = (edges - self.mean) / self.std / 2 + 1

        loss_cls = self.criterion(preds['cls'], anns) * edges
        loss_edge = self.criter_for_edge(preds['edge'].sigmoid_(), edges)
        return loss_cls.mean() + loss_edge * 0.1, loss_cls, loss_edge
