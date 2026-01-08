# An example to evaluate parallel decoding on LLaDA-8B-Instruct/HumanEval
# More methods please refer to docs

accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=humaneval \
    dataset.size=4 \
    cache=dllm \
    batch_size=1 \
    seed=1234 \
    cache=dllm \
    attn_implementation="eager" \
    generation.stop_until_eot=true \
    generation=vanilla \
    generation.steps=512 \
    generation.gen_length=512 \
    generation.block_length=32 \
    generation.threshold=0.9 \
    model=llada-inst 
