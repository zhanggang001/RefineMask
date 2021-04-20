#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PROJ_ROOT=$HOME/mmdetection

export PYTHONPATH=$PROJ_ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $PROJ_ROOT/tools/train.py $CONFIG --launcher pytorch ${@:3}

