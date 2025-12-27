# An example to evaluate parallel decoding on LLaDA-8B-Instruct/HumanEval
# More methods please refer to docs

accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=humaneval \
    dataset.n_shot=0 \
    batch_size=1 \
    seed=1234 \
    generation=wino_official \
    generation.gen_length=256 \
    generation.block_length=128 \
    model=llada-inst 

