#!/bin/bash
# This script runs evaluation for a dataset

# # 1. Test vanilla decoding on GSM8K with LLaDA-7B-Instruct, run:
# accelerate launch \
#     --num_machines 1 \
#     --num_processes 4 \
#     eval.py \
#     dataset.name=gsm8k \
#     batch_size=1 \
#     seed=1234 \
#     generation=vanilla \
#     model=llada-inst 

# # 2. Test Fast-dLLM on HumanEval with LLaDA-7B-Base, run:
# accelerate launch \
#     --num_machines 1 \
#     --num_processes 4 \
#     eval.py \
#     dataset.name=humaneval \
#     batch_size=1 \
#     seed=1234 \
#     cache=prefix \
#     generation=vanilla \
#     model=llada-base

# # 3. Test dLLM-Cache on MATH with Dream-v0-Instruct-7B, run:
# accelerate launch \
#     --num_machines 1 \
#     --num_processes 4 \
#     eval.py \
#     dataset.name=math-500 \
#     batch_size=1 \
#     seed=1234 \
#     cache=dllm \
#     generation=vanilla \
#     model=dream-inst

# 4.1 Test d2Cache on MBPP with Dream-v0-Base-7B, run:
# certainty prior guided decoding is enabled by default when using d2Cache
# accelerate launch \
#     --num_machines 1 \
#     --num_processes 4 \
#     eval.py \
#     dataset.name=humaneval \
#     batch_size=1 \
#     seed=1234 \
#     cache=d2cache \
#     generation=vanilla \
#     model=llada-inst

# # 4.2 d2Cache is also compatible with semi-ar decoding and parallel decoding, run:
# # explicitly set sigma to 0 to disable certainty prior guided decoding
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=gpqa_main_generative_n_shot \
    batch_size=1 \
    seed=1234 \
    dataset.size=4 \
    cache=d2cache \
    generation=vanilla \
    generation.threshold=0.9 \
    generation.block_length=32 \
    generation.sigma=0.0 \
    model=dream-inst