export CATTINO_HOME="./outputs"
datasets="[humaneval,mbpp,gsm8k,math-500]"
models="[llada-inst,dream-inst]"
meow set override-exist-tasks "allow"
meow set visible-devices [0,1,2,3]
# baseline

meow create --min-devices 4 --requires-memory-per-device 20000 -m --task-name "exp3_model_cache_compare/{model}/{dataset.name}/{cache}" \
    "accelerate launch --num_machines 1 --num_processes 4 eval.py" -- \
    dataset.name=$datasets \
    dataset.size=100 \
    batch_size=1 \
    seed=1234 \
    cache=d2cache \
    generation=vanilla \
    model=$models \
    generation.gen_length=1024 \
    generation.block_length=1024 \
    generation.steps=1024 \
    hydra.run.dir='${run_dir}/${fullname}'

meow create --min-devices 4 --requires-memory-per-device 20000 -m --task-name "exp3_model_cache_compare/{model}/{dataset.name}/{cache}" \
    "accelerate launch --num_machines 1 --num_processes 4 eval.py" -- \
    dataset.name=$datasets \
    dataset.size=100 \
    batch_size=1 \
    seed=1234 \
    cache=[dllm,prefix] \
    generation=vanilla \
    model=$models \
    generation.gen_length=1024 \
    generation.block_length=32 \
    generation.steps=1024 \
    hydra.run.dir='${run_dir}/${fullname}'

meow create --min-devices 4 --requires-memory-per-device 20000 -m --task-name "exp3_model_cache_compare/{model}/{dataset.name}/no_cache" \
    "accelerate launch --num_machines 1 --num_processes 4 eval.py" -- \
    dataset.name=$datasets \
    dataset.size=100 \
    batch_size=1 \
    seed=1234 \
    generation=vanilla \
    model=$models \
    generation.gen_length=1024 \
    generation.block_length=32 \
    generation.steps=1024 \
    hydra.run.dir='${run_dir}/${fullname}'