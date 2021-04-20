#!/usr/bin/env bash
set -x

# Environment
PROJ_DIR=$HOME/mmdetection # !!! Change $HOME to your workplace directory
export PYTHONPATH=$PROJ_DIR:$PYTHONPATH

# SRUN Args
PARTITION=$1
GPUS=$2
GPUS_PER_NODE=$(($2<8?$2:8))
CPUS_PER_TASK=5
JOB_NAME=lvis
SRUN_ARGS=""

# PYTHON Args
OUTPUT_DIR=`pwd`
WORK_DIR=$OUTPUT_DIR/output
CONFIG=$OUTPUT_DIR/config.py
CHECKPOINT=$WORK_DIR/latest.pth
OUT_FILE=$OUTPUT_DIR/result.pkl
TMP_DIR=$OUTPUT_DIR/tmpdir
PY_ARGS="--eval bbox segm --out $OUT_FILE --tmpdir $TMP_DIR"

# Command
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u $PROJ_DIR/tools/test.py ${CONFIG} ${CHECKPOINT} --launcher="slurm" ${PY_ARGS}
