import subprocess
import time
import os
from collections import deque
import argparse
import shlex

# --- 任务生成函数 (已按要求修改) ---

def generate_tasks(gpus_per_task, run_dir):
    """根据新的实验设计, 显式生成所有独立的命令行任务"""
    tasks = deque()
    base_command = f"accelerate launch --num_machines 1 --num_processes {gpus_per_task} eval.py"
    
    # --- 非常通用的参数，所有实验共享 ---
    common_args = "batch_size=1 seed=1234 generation=vanilla"

    # ==========================================================================
    # 实验一: 网格搜索 cache.rollout_p 和 cache.current_k
    # ==========================================================================
    print("--- 正在生成 [实验一: 网格搜索 rollout_p 和 current_k] 任务 ---")
    # 实验一固定模型、缓存和数据集
    exp1_base_args = f"{common_args} model=dream-inst cache=d2cache dataset.name=humaneval"
    gen_lengths_exp1 = [256, 1024]
    rollout_ps = [0.05, 0.1, 0.2, 0.4]
    current_ks = [16, 24, 32, 40]

    for length in gen_lengths_exp1:
        for p in rollout_ps:
            for k in current_ks:
                task_name = f"exp1_pk_search/len{length}/p{p}_k{k}"
                output_dir = os.path.join(run_dir, task_name)
                stderr_log_path = os.path.join(output_dir, "stderr.log")
                hydra_run_dir = f"hydra.run.dir={output_dir}"

                task_specific_args = (
                    f"generation.gen_length={length} "
                    f"cache.rollout_p={p} "
                    f"cache.current_k={k}"
                )
                
                args = f"{exp1_base_args} {task_specific_args} {hydra_run_dir}"
                full_command = f"{base_command} -- {args}"
                tasks.append((full_command, stderr_log_path))

    # ==========================================================================
    # 实验二: 网格搜索 cache.sigma
    # ==========================================================================
    print("--- 正在生成 [实验二: 网格搜索 sigma] 任务 ---")
    # 实验二也固定模型、缓存和数据集
    exp2_base_args = f"{common_args} model=dream-inst cache=d2cache dataset.name=humaneval"
    gen_lengths_exp2 = [256, 1024]
    sigmas = [1.0, 10.0, 20.0, 40.0, 80.0]
    fixed_rollout_p = 0.1
    fixed_current_k = 32

    for length in gen_lengths_exp2:
        for sigma in sigmas:
            task_name = f"exp2_sigma_search/len{length}/sigma{sigma}"
            output_dir = os.path.join(run_dir, task_name)
            stderr_log_path = os.path.join(output_dir, "stderr.log")
            hydra_run_dir = f"hydra.run.dir={output_dir}"

            task_specific_args = (
                f"generation.gen_length={length} "
                f"cache.rollout_p={fixed_rollout_p} "
                f"cache.current_k={fixed_current_k} "
                f"cache.sigma={sigma}"
            )
            
            args = f"{exp2_base_args} {task_specific_args} {hydra_run_dir}"
            full_command = f"{base_command} -- {args}"
            tasks.append((full_command, stderr_log_path))
            
    # ==========================================================================
    # 实验三: 对比不同模型、数据集和缓存策略 (已更新)
    # ==========================================================================
    print("--- 正在生成 [实验三: 对比模型、数据集和缓存策略] 任务 ---")
    exp3_models = ["dream-inst", "llada-inst"]
    exp3_caches = ["no_cache", "dllm", "prefix", "d2cache"]
    exp3_datasets = ["humaneval", "mbpp", "gsm8k", "math-500"] # 新增的数据集列表
    exp3_gen_length = 1024

    for model in exp3_models:
        for dataset in exp3_datasets: # 新增数据集循环
            for cache in exp3_caches:
                # 更新任务命名以包含数据集
                task_name = f"exp3_model_cache_compare/{model}/{dataset}/{cache}"
                output_dir = os.path.join(run_dir, task_name)
                stderr_log_path = os.path.join(output_dir, "stderr.log")
                hydra_run_dir = f"hydra.run.dir={output_dir}"

                # 特殊处理 no_cache 的情况
                if cache == "no_cache":
                    cache_arg = "" # 当是 no_cache 时, 不添加 cache= 参数
                else:
                    cache_arg = f"cache={cache}"

                # 组合参数，现在包含动态的数据集名称
                args = (
                    f"{common_args} "
                    f"model={model} "
                    f"dataset.name={dataset} "
                    f"{cache_arg} "
                    f"generation.gen_length={exp3_gen_length} "
                    f"{hydra_run_dir}"
                )
                
                # 清理因 cache_arg 为空可能产生的多余空格
                args = ' '.join(args.split())

                full_command = f"{base_command} -- {args}"
                tasks.append((full_command, stderr_log_path))

    return tasks

# --- 主调度逻辑 (保持不变) ---

def main(available_gpu_ids):
    """主函数，用于调度和管理任务"""

    gpus_per_task = len(available_gpu_ids)
    if gpus_per_task == 0:
        print("错误：未指定任何 GPU。请使用 --gpus 参数提供 GPU ID。")
        return
    print(f"检测到 {gpus_per_task} 个指定的 GPU。每个任务将使用所有这 {gpus_per_task} 个 GPU。")
    
    timestamp = time.strftime("%Y-%m-%d")
    run_dir = f"outputs/{timestamp}"
    print(f"所有实验输出将保存在基目录: {run_dir}")

    task_queue = generate_tasks(gpus_per_task, run_dir)
    total_tasks = len(task_queue)
    print(f"\n成功生成 {total_tasks} 个独立任务。")

    gpu_slot = available_gpu_ids
    slot_is_available = True
    running_process_info = None
    completed_tasks = 0

    while task_queue or running_process_info:
        if running_process_info:
            process, stderr_log_file = running_process_info
            if process.poll() is not None:
                exit_code = process.returncode
                stderr_log_file.close()

                print(f"✅ 任务 (PID: {process.pid}) 已完成，退出码: {exit_code}。释放 GPU 插槽: {gpu_slot}")
                print(f"   - Stderr 日志已保存至: {stderr_log_file.name}")
                slot_is_available = True
                if exit_code == 0:
                    completed_tasks += 1
                else:
                    print(f"❌ 警告：任务 (PID: {process.pid}) 异常退出！请检查日志文件。")
                running_process_info = None

        if slot_is_available and task_queue:
            command_to_run, stderr_log_path = task_queue.popleft()

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_slot))

            print("-" * 60)
            print(f"🚀 准备启动新任务 ({total_tasks - len(task_queue)}/{total_tasks}):")
            print(f"   - 命令: {command_to_run}")
            print(f"   - 使用 GPU: {gpu_slot} (CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']})")
            print(f"   - Stderr 将输出到: {stderr_log_path}")

            os.makedirs(os.path.dirname(stderr_log_path), exist_ok=True)
            stderr_log_file = open(stderr_log_path, 'w')
            
            command_list = shlex.split(command_to_run)
            process = subprocess.Popen(
                command_list, 
                shell=False,
                env=env, 
                stderr=stderr_log_file
            )
            
            running_process_info = (process, stderr_log_file)
            slot_is_available = False

            print(f"   - 任务已启动，PID: {process.pid}")

        time.sleep(10)

    print("\n" + "=" * 60)
    print("🎉 所有任务已执行完毕！")
    print(f"总计成功完成 {completed_tasks}/{total_tasks} 个任务。")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU 任务管理器")
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help="指定用于任务调度的 GPU ID 列表，以逗号分隔。例如 '0,1,2,3'"
    )
    args = parser.parse_args()

    try:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(',') if g.strip()]
    except ValueError:
        print("错误：--gpus 参数格式不正确。请输入以逗号分隔的数字，例如 '0,1,2,3'")
        exit(1)

    main(gpu_ids)