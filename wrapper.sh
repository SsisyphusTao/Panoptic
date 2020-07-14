#! /bin/bash
time python -m torch.distributed.launch \
            --nproc_per_node=4 train.py \
            --lr 1e-3