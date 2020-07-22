#! /bin/bash
rm log.txt
time \
CUDA_VISIBLE_DEVICES=1,2,3 \
python -m torch.distributed.launch \
            --nproc_per_node=3 train.py \
            --batch_size=20 \
            --lr=1e-3 \
            --start_iter=1 \
            --resume=checkpoints/ctnet_dla_001_865.pth >> log.txt