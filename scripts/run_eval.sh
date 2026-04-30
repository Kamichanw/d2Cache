# An example to evaluate parallel decoding on LLaDA-8B-Instruct/HumanEval
# More methods please refer to docs

accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    eval.py \
    dataset.name=humaneval \
    dataset.size=10 \
    batch_size=1 \
    seed=1234 \
    generation=vanilla \
    model=llada-inst 
