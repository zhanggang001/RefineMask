#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

PROJ_ROOT=$HOME/mmdetection

export PYTHONPATH=$PROJ_ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $PROJ_ROOT/tools/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
