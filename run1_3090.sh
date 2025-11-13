export CATTINO_HOME="./outputs"
datasets="[humaneval]"
models="[dream-inst]"
meow set override-exist-tasks "allow"
meow set visible-devices [0,1,2,3]
# baseline

meow create --min-devices 4 --requires-memory-per-device 20000 -m --task-name "exp1_pk_search/len{generation.gen_length}/p{cache.rollout_p}_k{cache.current_k}" \
    "accelerate launch --num_machines 1 --num_processes 4 eval.py" -- \
    dataset.name=$datasets \
    dataset.size=100 \
    batch_size=1 \
    seed=1234 \
    cache=d2cache \
    cache.rollout_p=[0.05,0.1,0.2,0.4] \
    cache.current_k=[16,24,32,40] \
    generation=vanilla \
    model=$models \
    generation.gen_length=256 \
    generation.block_length=256 \
    generation.steps=256 \
    hydra.run.dir='${run_dir}/${fullname}'

meow create --min-devices 4 --requires-memory-per-device 20000 -m --task-name "exp1_pk_search/len{generation.gen_length}/p{cache.rollout_p}_k{cache.current_k}" \
    "accelerate launch --num_machines 1 --num_processes 4 eval.py" -- \
    dataset.name=$datasets \
    dataset.size=100 \
    batch_size=1 \
    seed=1234 \
    cache=d2cache \
    cache.rollout_p=[0.05,0.1,0.2,0.4] \
    cache.current_k=[16,24,32,40] \
    generation=vanilla \
    model=$models \
    generation.gen_length=1024 \
    generation.block_length=1024 \
    generation.steps=1024 \
    hydra.run.dir='${run_dir}/${fullname}'

meow create --min-devices 4 --requires-memory-per-device 20000 -m --task-name "exp2_sigma_search/len{generation.gen_length}/sigma{cache.sigma}" \
    "accelerate launch --num_machines 1 --num_processes 4 eval.py" -- \
    dataset.name=$datasets \
    dataset.size=100 \
    batch_size=1 \
    seed=1234 \
    cache=d2cache \
    cache.sigma=[1.0,10.0,20.0,40.0,80.0] \
    generation=vanilla \
    model=$models \
    generation.gen_length=256 \
    generation.block_length=256 \
    generation.steps=256 \
    hydra.run.dir='${run_dir}/${fullname}'


meow create --min-devices 4 --requires-memory-per-device 20000 -m --task-name "exp2_sigma_search/len{generation.gen_length}/sigma{cache.sigma}" \
    "accelerate launch --num_machines 1 --num_processes 4 eval.py" -- \
    dataset.name=$datasets \
    dataset.size=100 \
    batch_size=1 \
    seed=1234 \
    cache=d2cache \
    cache.sigma=[1.0,10.0,20.0,40.0,80.0] \
    generation=vanilla \
    model=$models \
    generation.gen_length=1024 \
    generation.block_length=1024 \
    generation.steps=1024 \
    hydra.run.dir='${run_dir}/${fullname}'