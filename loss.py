from nets import get_pose_net
import torch
from torch.utils import data
from cocotask import panopticDataset, collate
from augmentations import Augmentation

def focalloss(pred, gt, w):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).type_as(pred)
  neg_inds = gt.eq(0).type_as(pred)

  neg_weights = torch.pow(w, 4)

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds * neg_weights

  num_pos  = pos_inds.sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if not num_pos:
    return -neg_loss
  else:
    return -(pos_loss + neg_loss) / num_pos

class RegL1Loss(torch.nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, pred, target, mask):
    mask = mask.permute(0,3,1,2).expand_as(pred)
    loss = torch.nn.functional.l1_loss(pred * mask, target * mask, reduction='sum') / (mask.sum() + 1e-4)
    return loss

class NetwithLoss(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self._sigmoid = lambda x: torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        self.onehot = lambda x: torch.nn.functional.one_hot(x, 81).permute(0,3,1,2)[:,1:,:,:]
        self.criter_for_cls = focalloss
        self.criter_for_grad = torch.nn.SmoothL1Loss()
        self.net = net
        self.net.train()
    @torch.cuda.amp.autocast()
    def forward(self, images, anns, gx, gy, mask):
        preds = self.net(images)

        grad = torch.nn.functional.interpolate(preds['grad'], size=[512,512], mode='bilinear', align_corners=True)
        loss_grad = self.criter_for_grad(grad, torch.stack([gx, gy], 1))

        anns = self.onehot(anns)
        mask = self.onehot(mask.squeeze().type(torch.long))

        w = (gx.pow(2)+gy.pow(2)).sqrt()/511
        w = torch.nn.functional.interpolate(w.unsqueeze(1), size=[128,128], mode='bilinear', align_corners=True).expand_as(anns)
        w = w.where(mask>0, torch.ones_like(w))
        loss_cls = self.criter_for_cls(self._sigmoid(preds['hm']), anns, w)
        return loss_cls, loss_grad*0.1

# grad_x = preds['grad'][:16].reshape(-1,4,4,128,128).permute(0,3,1,4,2).flatten(3,4).flatten(1,2)
# grad_y = preds['grad'][16:].reshape(-1,4,4,128,128).permute(0,3,1,4,2).flatten(3,4).flatten(1,2)