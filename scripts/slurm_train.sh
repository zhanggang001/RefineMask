#!/usr/bin/env bash
set -x

# Environment
PROJ_DIR=$HOME/mmdetection
export PYTHONPATH=$PROJ_DIR:$PYTHONPATH

# SRUN Args
PARTITION=$1
GPUS=$2
GPUS_PER_NODE=$(($2<8?$2:8))
CPUS_PER_TASK=5
SRUN_ARGS=""
PY_ARGS=""
JOB_NAME=lvis

# PYTHON Args
OUTPUT_DIR=`pwd`
CONFIG=$OUTPUT_DIR/config.py
WORK_DIR=$OUTPUT_DIR/output

# Command
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u $PROJ_DIR/tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --launcher="slurm" ${PY_ARGS}
