#! /bin/bash
rm log.txt
time \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
            --nproc_per_node=4 train.py \
            --batch_size=20 \
            --lr=1.25e-4 \
            --resume checkpoints/ctnet_dla_temp.pth >> log.txt