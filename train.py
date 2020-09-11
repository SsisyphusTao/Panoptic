from cocotask.dali_panoptic import panopticInputIterator, panopticPipeline, grad_preprocess, create_edge
from dali_augmentations import Augmentation as DALIAugmentation
from augmentations import Augmentation
from cocotask import panopticDataset, collate
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
parser.add_argument('--batch_size', default=128, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--epochs', default=140, type=int,
                    help='the number of training epochs')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=32, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1.25e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--save_folder', default='checkpoints/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--local_rank', default=0, type=int,
    help='Used for multi-process training. Can either be manually set ' +
        'or automatically set by using \'python -m multiproc\'.')
args = parser.parse_args()
scaler = torch.cuda.amp.GradScaler()

def train_one_epoch(loader, getloss, optimizer, epoch):
    loss_amount = 0
    t0 = time.clock()
    # load train data
    for iteration, batch in enumerate(loader):
        # forward & backprop
        optimizer.zero_grad()
        gx, gy, ann = grad_preprocess(batch[0])
        loss_c, loss_g = getloss(batch[0]['images'], ann, gx, gy, batch[0]['anns'])
        loss = loss_c + loss_g
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        t1 = time.clock()
        loss_amount += loss.item()
        scaler.update()
        if iteration % 10 == 0 and not iteration == 0 and not args.local_rank:
            print('Loss: %.4f, cls: %.4f, grad: %.4f | iter: %03d | timer: %.4f sec. | epoch: %d' %
                    (loss_amount/iteration, loss_c.item(), loss_g.item(), iteration, t1-t0, epoch))
        t0 = t1
    print('Device:%d  Loss: %.6f' % (args.local_rank, (loss_amount/iteration)))
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
    net = get_pose_net(34, {'hm': 80, 'grad':2})
    if args.resume:
        missing, unexpected = net.load_state_dict(torch.load(args.resume, map_location='cpu'), strict=False)

    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
    #                       weight_decay=5e-4)
    optimizer = torch.optim.Adam(net.parameters(), args.lr)
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = args.lr
    adjust_learning_rate = optim.lr_scheduler.MultiStepLR(optimizer, [90, 120], 0.1, args.start_iter)
    # adjust_learning_rate = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.start_iter)

    if not args.local_rank:
        print('Loading the dataset....', end='')
    if _distributed:
        getloss = nn.parallel.DistributedDataParallel(NetwithLoss(net).cuda(), device_ids=[args.local_rank], find_unused_parameters=True)
        external = panopticInputIterator(args.batch_size)
        pipe = panopticPipeline(external, DALIAugmentation(512), args.batch_size, args.num_workers, args.local_rank)
        data_loader = DALIGenericIterator(pipe,
                                          ["images", "anns", "gx", "gy", "x", "y", "s", "c1", "c2"],
                                          fill_last_batch = False, auto_reset=True,
                                          size = external.size // N_gpu + 1)
    else:
        getloss = nn.DataParallel(NetwithLoss(net).cuda(), device_ids=[0,1,2,3,4,5,6,7])
        dataset = panopticDataset(Augmentation(512))
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False, collate_fn=collate,
                                  pin_memory=True, sampler=sampler)
    
    if not args.local_rank:
        print('Finished!')

    if not args.local_rank:
        print('Training CenterNet on:', 'dali-panoptic no.%d' % args.local_rank)
        print('Using the specified args:')
        print(args)
    torch.cuda.empty_cache()
    # create batch iterator
    for iteration in range(args.start_iter + 1, args.epochs):
        loss = train_one_epoch(data_loader, getloss, optimizer, iteration)
        external.shuffle()
        adjust_learning_rate.step()
        if (not (iteration-args.start_iter) == 0):
            torch.distributed.barrier()
            if not args.local_rank:
                torch.save(net.state_dict(), args.save_folder + 'dla_instance_' +
                        '%03d'%iteration + loss + '.pth')
                print('Save model %03d'%iteration+loss+'.pth')

if __name__ == '__main__':
    train()
