from cocotask.dali_panoptic import panopticInputIterator, panopticPipeline, create_edge
from dali_augmentations import Augmentation
from nets import get_pose_net
from loss import NetwithLoss

import torch
from torch import nn
import torch.optim as optim
import torch.utils.data as data
from nvidia.dali.plugin.pytorch import DALIGenericIterator

import argparse
import time
import os

parser = argparse.ArgumentParser(
    description='CenterNet task')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--datafile', default=None,
                    help='Path of training set')
parser.add_argument('--batch_size', default=20, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--epochs', default=70, type=int,
                    help='the number of training epochs')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1.25e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--save_folder', default='checkpoints/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--local_rank', default=0, type=int,
    help='Used for multi-process training. Can either be manually set ' +
        'or automatically set by using \'python -m multiproc\'.')
args = parser.parse_args()
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# @profile
def train_one_epoch(loader, getloss, optimizer, epoch):
    loss_amount = 0
    t0 = time.clock()
    # load train data
    for iteration, batch in enumerate(loader):
        images = batch[0]["images"]
        anns = batch[0]["anns"].squeeze()
        edges = create_edge(batch[0]["edges"].squeeze())
        # forward & backprop
        optimizer.zero_grad()
        loss, c, e = getloss(images, anns, edges)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        t1 = time.clock()
        loss_amount += loss.item()
        if iteration % 10 == 0 and not iteration == 0 and images.device.index == 0:
            print('Loss: %.6f, cls: %.6f, edge: %.6f | iter: %03d | timer: %.4f sec. | epoch: %d' %
                    (loss_amount/iteration, c.mean().item(), e.mean().item(), iteration, t1-t0, epoch))
        t0 = t1
    print('Device:%d  Loss: %.6f' % (images.device.index, (loss_amount/iteration)))
    return '_%d' % (loss_amount/iteration*1000)

def train():
    torch.backends.cudnn.benchmark = True
    _distributed = False
    if 'WORLD_SIZE' in os.environ:
        _distributed = int(os.environ['WORLD_SIZE']) > 1

    if _distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        N_gpu = torch.distributed.get_world_size()
    else:
        N_gpu = 1

    start_time = time.clock()
    heads = {'cls': 81,
             'edge': 16}
    net = get_pose_net(34, heads)
    if args.resume:
        missing, unexpected = net.load_state_dict({k.replace('module.',''):v 
        for k,v in torch.load(args.resume).items()}, strict=False)
        if missing:
            print('Missing:', missing)
        if unexpected:
            print('Unexpected:', unexpected)
    net.train()

    # optimizer = optim.Adam(net.parameters(), lr=args.lr)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4)

    # args.lr = args.lr * N_gpu * (args.batch_size / 32)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.lr
    # adjust_learning_rate = optim.lr_scheduler.MultiStepLR(optimizer, [35, 75], 0.1, args.start_iter)
    adjust_learning_rate = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.start_iter)
    getloss = nn.parallel.DistributedDataParallel(NetwithLoss(net), device_ids=[args.local_rank], find_unused_parameters=True)

    if not args.local_rank:
        print('Loading the dataset...')
    external = panopticInputIterator(args.batch_size)
    pipe = panopticPipeline(external, Augmentation(), args.batch_size, args.num_workers, args.local_rank)
    data_loader = DALIGenericIterator(pipe,
                                      ["images", "anns", "edges"],
                                      fill_last_batch = False,
                                      size = external.size // N_gpu + 1)
    if not args.local_rank:
        print('Training CenterNet on:', 'dali-panoptic no.%d' % args.local_rank)
        print('Using the specified args:')
        print(args)

    # create batch iterator
    for iteration in range(args.start_iter + 1, args.epochs):
        loss = train_one_epoch(data_loader, getloss, optimizer, iteration)
        data_loader.reset()
        adjust_learning_rate.step()
        if (not (iteration-args.start_iter) == 0) and not args.local_rank:
            torch.save(net.state_dict(), args.save_folder + 'ctnet_dla_' +
                       '%03d'%iteration + loss + '.pth')
    if not args.local_rank:
        torch.save(net.state_dict(),
                    args.save_folder + 'ctnet_dla_end' + loss + '.pth')
    end_time=time.clock()

if __name__ == '__main__':
    train()
