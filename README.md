# d2Cache

# Setup
```bash
# Create and activate the environment
conda create -n d2cache python=3.11 -y
conda activate d2cache

# Install dependencies
pip install -r requirements/common.txt

# Prepare dotenv file, and set model path manually 
cp .env.example .env
```


# Evaluation
Available models:
- llada-base: GSAI-ML/LLaDA-8B-Base
- llada-inst: GSAI-ML/LLaDA-8B-Instruct
- dream-base: Dream-org/Dream-v0-Base-7B
- dream-inst: Dream-org/Dream-v0-Instruct-7B

Available datasets:
- gsm8k
- humaneval
- math-500
- mbpp

Available cache methods:
- no cache (default)
- prefix (Fast-dLLM)
- dllm (dLLM-Cache)
- d2cache (d2Cache)


Additional general arguments can be specified in `configs/geneation/*.yaml` or `configs/gen_args.py`.

1. Test vanilla decoding on GSM8K with LLaDA-7B-Instruct, run:
```bash
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    generation=vanilla \
    model=llada-inst 
```
2. Test Fast-dLLM on HumanEval with LLaDA-7B-Base, run:
```bash
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
```
3. Test dLLM-Cache on MATH with Dream-v0-Instruct-7B, run:
```bash
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
```
4. Test d2Cache on MBPP with Dream-v0-Base-7B, run:
```bash
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
```