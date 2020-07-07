from cocotask import panopticDataset, collate
from augmentations import Augmentation
from loss import NetwithLoss
import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as data
from nets import get_pose_net
import argparse
import time
import h5py

parser = argparse.ArgumentParser(
    description='CenterNet task')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--datafile', default='/ai/ailab/Share/TaoData/panoptic.hdf5',
                    help='Path of training set')
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--epochs', default=70, type=int,
                    help='the number of training epochs')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=1, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1.25e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--save_folder', default='checkpoints/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# @profile
def train_one_epoch(loader, getloss, optimizer, epoch):
    loss_amount = 0
    t0 = time.clock()
    # load train data
    for iteration, batch in enumerate(loader): 
        batch[0] = batch[0].cuda(non_blocking=True)
        batch[1] = batch[1].cuda(non_blocking=True)  
        batch[2] = batch[2].cuda(non_blocking=True)  
        # forward & backprop
        optimizer.zero_grad()
        loss = getloss(*batch).mean()
        loss.backward()
        optimizer.step()
        t1 = time.clock()
        loss_amount += loss.item()
        if iteration % 10 == 0 and not iteration == 0:
            print('Loss: %.6f | iter: %03d | timer: %.4f sec. | epoch: %d' %
                    (loss_amount/iteration, iteration, t1-t0, epoch))
        t0 = t1
    print('Loss: %.6f -------------------------------------------------------------------------------' % (loss_amount/iteration))
    return '_%d' % (loss_amount/iteration*1000)

def train():
    start_time = time.clock()
    f = h5py.File('/ai/ailab/Share/TaoData/panoptic.hdf5', 'r')
    dataset = panopticDataset(f, Augmentation())
    heads = {'cls': 81,
            'edge': 1}
    net = get_pose_net(34, heads)
    if args.resume:
        missing, unexpected = net.load_state_dict({k.replace('module.',''):v 
        for k,v in torch.load(args.resume).items()})
        if missing:
            print('Missing:', missing)
        if unexpected:
            print('Unexpected:', unexpected)
    net.train()
    # net = nn.DataParallel(net.cuda(), device_ids=[0,1,2,3])
    torch.backends.cudnn.benchmark = True

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
    #                       weight_decay=5e-4)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.lr
    adjust_learning_rate = optim.lr_scheduler.MultiStepLR(optimizer, [35, 75], 0.1, args.start_iter)
    # adjust_learning_rate = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.start_iter)
    getloss = nn.DataParallel(NetwithLoss(net).cuda(), device_ids=[0,1,2,3])

    print('Loading the dataset...')
    f = h5py.File('/ai/ailab/Share/TaoData/panoptic.hdf5', 'r')
    dataset = panopticDataset(f, Augmentation())
    data_loader = data.DataLoader(dataset, args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=True, collate_fn=collate,
                                pin_memory=True)
    print('Training CenterNet on:', dataset.name)
    print('Using the specified args:')
    print(args)

    # create batch iterator
    for iteration in range(args.start_iter + 1, args.epochs):
        loss = train_one_epoch(data_loader, getloss, optimizer, iteration)
        adjust_learning_rate.step()
        if (not (iteration-args.start_iter) == 0):
            print('Saving state, iter:', iteration)
            torch.save(net.state_dict(), args.save_folder + 'ctnet_dla_' +
                       '%03d'%iteration + loss + '.pth')
    torch.save(net.state_dict(),
                args.save_folder + 'ctnet_dla_end' + loss + '.pth')
    end_time=time.clock()
    print('Time spend: %d' % (time.gmtime(end_time).tm_yday-time.gmtime(start_time).tm_yday), time.strftime('%H:%M:%S', time.gmtime(end_time-start_time)))

if __name__ == '__main__':
    train()
