# An example to evaluate parallel decoding on LLaDA-8B-Instruct/HumanEval
# More methods please refer to docs

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    eval.py \
    dataset.name=humaneval \
    batch_size=1 \
    seed=1234 \
    generation=vanilla \
    generation.steps=512 \
    generation.gen_length=512 \
    generation.block_length=32 \
    generation.threshold=0.9 \
    model=llada-inst 

