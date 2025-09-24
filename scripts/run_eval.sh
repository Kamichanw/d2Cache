#!/bin/bash
# This script runs evaluation for a dataset

DATASET="math-500"
MODEL="llada-inst"
OUTPUT_DIR="./outputs/${MODEL}-${DATASET}"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=${DATASET} \
    batch_size=1 \
    seed=1234 \
    cache=heat \
    generation=dyna \
    model=${MODEL} \
    hydra.run.dir=${OUTPUT_DIR}