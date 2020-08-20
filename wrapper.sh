#! /bin/bash
rm log.txt
time \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch \
            --nproc_per_node=8 train.py \
            --batch_size=16 \
            --lr=1e-3 \
            >> log.txt