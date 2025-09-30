#!/bin/bash
# This script runs evaluation for a dataset

DATASET="math-500"
MODEL="llada-inst"
OUTPUT_DIR="./outputs/${MODEL}-${DATASET}"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# 1. Test vanilla decoding on GSM8K with LLaDA-7B-Instruct, run:

accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    generation=vanilla \
    model=llada-inst 

# 2. Test Fast-dLLM on HumanEval with LLaDA-7B-Base, run:

accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=humaneval \
    batch_size=1 \
    seed=1234 \
    cache=prefix \
    generation=vanilla \
    model=llada-base

# 3. Test dLLM-Cache on MATH with Dream-v0-Instruct-7B, run:

accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=math-500 \
    batch_size=1 \
    seed=1234 \
    cache=dllm \
    generation=vanilla \
    model=dream-inst

# 4. Test d2Cache on MBPP with Dream-v0-Base-7B, run:

accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=mbpp \
    batch_size=1 \
    seed=1234 \
    cache=d2cache \
    generation=dyna \
    model=dream-base
