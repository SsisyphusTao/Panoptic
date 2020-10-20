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
  pos_inds = gt.eq(1).type_as(pred)
  neg_inds = gt.eq(0).type_as(pred)

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
        self._sigmoid = lambda x: torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        self.onehot = lambda x: torch.nn.functional.one_hot(x, 81).permute(0,3,1,2)[:,1:,:,:]
        self.criter_for_cls = focalloss
        self.criter_for_grad = torch.nn.SmoothL1Loss(reduction='sum')
        self.net = net
        self.net.train()
    @torch.cuda.amp.autocast()
    def forward(self, batch, anns, gx, gy):
        preds = self.net(batch['images'])
        
        grad = torch.nn.functional.interpolate(preds['grad'], size=[512,512], mode='bilinear', align_corners=True)
        mask = batch['anns'].squeeze().gt(0).type_as(grad)
        mask[mask==0] = -1
        loss_grad = self.criter_for_grad(grad, torch.stack([gx, gy, mask], 1))
        mp = batch['anns'].gt(0).type(torch.long).sum() * 3

        loss_cls = self.criter_for_cls(self._sigmoid(preds['hm']), self.onehot(anns))
        return loss_cls, 0.1 * (loss_grad / mp if mp else 1)