#%%
from nets import get_pose_net
import torch
from torch.utils import data
from cocotask import panopticDataset, collate
from augmentations import Augmentation
import h5py

heads = {'cls': 81,
         'edge': 1}
net = get_pose_net(34, heads)
# print(net)
f = h5py.File('/ai/ailab/Share/TaoData/panoptic.hdf5', 'r')
dataset = panopticDataset(f, Augmentation())
data_loader = data.DataLoader(dataset, 1,
                                  num_workers=0,
                                  shuffle=True, collate_fn=collate,
                                  pin_memory=False)

# %%
def focalloss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

  num_pos  = pos_inds.sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()
  print(pos_loss, neg_loss, num_pos)
  if num_pos == 0:
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
        loss_cls = self.criter_for_cls(preds['cls'], anns)
        loss_edge = self.criter_for_edge(preds['edge'].sigmoid_(), edges)
        print(loss_cls.item(), loss_edge.item())
        return loss_cls + loss_edge

#%%
getloss = NetwithLoss(net)
for x in data_loader:

    loss = getloss(x[0], *x[1:])
    print(loss.item())
    break

# %%
