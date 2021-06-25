#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

PROJ_ROOT=`pwd`
export PYTHONPATH=$PROJ_ROOT:$PYTHONPATH
export OMP_NUM_THREADS=1

WORK_DIR=$3
CHECKPOINT=$WORK_DIR/latest.pth
OUT_FILE=$WORK_DIR/result.pkl
TMP_DIR=$WORK_DIR/tmpdir
PY_ARGS="--eval bbox segm --out $OUT_FILE --tmpdir $TMP_DIR"

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $PROJ_ROOT/tools/test.py $CONFIG $CHECKPOINT --launcher pytorch ${PY_ARGS}
