## Large Language Diffusion Models
Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, Chongxuan Li
### Abstract
The capabilities of large language models (LLMs) are widely regarded as relying on autoregressive models (ARMs). We challenge this notion by introducing LLaDA, a diffusion model trained from scratch under the pre-training and supervised fine-tuning (SFT) paradigm. LLaDA employs a forward data masking process and a reverse generation process, parameterized by a Transformer to predict masked tokens. It provides a principled generative approach for probabilistic inference by optimizing a likelihood lower bound. Across extensive benchmarks on general tasks, math, code, and so on, LLaDA demonstrates strong scalability and performs comparably to our self-constructed ARM baselines. Remarkably, LLaDA 8B is competitive with strong LLMs like LLaMA3 8B in in-context learning and, after SFT, exhibits impressive instruction-following abilities in case studies such as multi-turn dialogue. Moreover, LLaDA addresses the reversal curse, surpassing GPT-4o in a reversal poem completion task. Our findings show the promise of diffusion models for language modeling at scale and challenge the common assumption that core LLM capabilities discussed above inherently depend on ARMs. Project page and codes: this https URL.

### Example Usage
```bash
# Test vanilla decoding on GSM8K with LLaDA-7B-Instruct, run:
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    generation=vanilla \
    generation.steps=256 \
    generation.gen_length=256 \
    generation.block_length=32 \
    model=llada-inst 
```

## Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding
Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, Enze Xie
### Abstract
Diffusion-based large language models (Diffusion LLMs) have shown promise for non-autoregressive text generation with parallel decoding capabilities. However, the practical inference speed of open-sourced Diffusion LLMs often lags behind autoregressive models due to the lack of Key-Value (KV) Cache and quality degradation when decoding multiple tokens simultaneously. To bridge this gap, we introduce a novel block-wise approximate KV Cache mechanism tailored for bidirectional diffusion models, enabling cache reuse with negligible performance drop. Additionally, we identify the root cause of generation quality degradation in parallel decoding as the disruption of token dependencies under the conditional independence assumption. To address this, we propose a confidence-aware parallel decoding strategy that selectively decodes tokens exceeding a confidence threshold, mitigating dependency violations and maintaining generation quality. Experimental results on LLaDA and Dream models across multiple LLM benchmarks demonstrate up to 27.6 throughput improvement with minimal accuracy loss, closing the performance gap with autoregressive models and paving the way for practical deployment of Diffusion LLMs.

### Example usages
Set `generation.threshold` to retain tokens whose confidence above a fixed threshold:
```bash
# Test parallel decoding on GSM8K with LLaDA-7B-Instruct, run:
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    generation=vanilla \
    generation.steps=256 \
    generation.gen_length=256 \
    generation.block_length=32 \
    generation.threshold=0.9 \
    model=llada-inst 
```

The factor based decoding proposed in Sec 3.3 is als implemented, to use this, add `generation.factor=1` to the command.

## PC-Sampler: Position-Aware Calibration of Decoding Bias in Masked Diffusion Models
Pengcheng Huang, Shuhao Liu, Zhenghao Liu, Yukun Yan, Shuo Wang, Zulong Chen, Tong Xiao
### Abstract
Recent advances in masked diffusion models (MDMs) have established them as powerful non-autoregressive alternatives for sequence generation. Nevertheless, our preliminary experiments reveal that the generation quality of MDMs is still highly sensitive to the choice of decoding strategy. In particular, widely adopted uncertainty-based samplers suffer from two key limitations: a lack of global trajectory control and a pronounced bias toward trivial tokens in the early stages of decoding. These shortcomings restrict the full potential of MDMs. In this work, we introduce Position-Aware Confidence-Calibrated Sampling (PC-Sampler), a novel decoding strategy that unifies global trajectory planning with content-aware informativeness maximization. PC-Sampler incorporates a position-aware weighting mechanism to regulate the decoding path and a calibrated confidence score to suppress the premature selection of trivial tokens. Extensive experiments on three advanced MDMs across seven challenging benchmarks-including logical reasoning and planning tasks-demonstrate that PC-Sampler consistently outperforms existing MDM decoding strategies by more than 10% on average, significantly narrowing the performance gap with state-of-the-art autoregressive models. All codes are available at this https URL.

### Example Usage
```bash
# Test PC sampler on GSM8K with LLaDA-7B-Instruct, run:
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    generation=pc_sampler \
    generation.steps=256 \
    generation.gen_length=256 \
    generation.block_length=32 \
    generation.debias=true \
    model=llada-inst 
```

> [!NOTE]
> Global Trajectory Control has not been implemented yet.

## d2Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching
Yuchu Jiang, Yue Cai, Xiangzhong Luo, Jiale Fu, Jiarui Wang, Chonghan Liu, Xu Yang
### Abstract
Diffusion-based large language models (dLLMs), despite their promising performance, still suffer from inferior inference efficiency. This is because dLLMs rely on bidirectional attention and cannot directly benefit from the standard key-value (KV) cache as autoregressive models (ARMs) do. To tackle this issue, we introduce *Dual aDaptive Cache* (dCache), which is a training-free approximate KV cache framework for accelerating dLLM inference. dCache features a two-stage fine-grained selection strategy to identify tokens and adaptively update their KV states at each decoding step, while caching the KV states of the remaining tokens for reuse. Furthermore, dCache naturally offers a more reliable decoding alternative, which can enable quasi left-to-right generation and mitigate premature overconfidence in tokens at the end of the sequence. Extensive experimental results on two representative dLLMs (*i.e.*, LLaDA and Dream) demonstrate that dCache not only achieves substantial inference speedups, but also yields consistent improvements in generation quality. The code is available at this https URL.

### Example Usage
The sigma in Eq. (3) is used to control the decoding order: a smaller makes the model behave more autoregressively, whereas a larger σ pushes it toward more non-autoregressive decoding.
```bash
# Test certainty prior-guided decoding on GSM8K with LLaDA-7B-Instruct, run:
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    generation=vanilla \
    generation.steps=256 \
    generation.gen_length=256 \
    generation.sigma=10 \
    model=llada-inst 
```

## Beyond Fixed: Training-Free Variable-Length Denoising for Diffusion Large Language Models
Jinsong Li, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Jiaqi Wang, Dahua Lin

### Abstract
Diffusion Large Language Models (DLLMs) are emerging as a powerful alternative to the dominant Autoregressive Large Language Models, offering efficient parallel generation and capable global context modeling. However, the practical application of DLLMs is hindered by a critical architectural constraint: the need for a statically predefined generation length. This static length allocation leads to a problematic trade-off: insufficient lengths cripple performance on complex tasks, while excessive lengths incur significant computational overhead and sometimes result in performance degradation. While the inference framework is rigid, we observe that the model itself possesses internal signals that correlate with the optimal response length for a given task. To bridge this gap, we leverage these latent signals and introduce DAEDAL, a novel training-free denoising strategy that enables Dynamic Adaptive Length Expansion for Diffusion Large Language Models. DAEDAL operates in two phases: 1) Before the denoising process, DAEDAL starts from a short initial length and iteratively expands it to a coarse task-appropriate length, guided by a sequence completion metric. 2) During the denoising process, DAEDAL dynamically intervenes by pinpointing and expanding insufficient generation regions through mask token insertion, ensuring the final output is fully developed. Extensive experiments on DLLMs demonstrate that DAEDAL achieves performance comparable, and in some cases superior, to meticulously tuned fixed-length baselines, while simultaneously enhancing computational efficiency by achieving a higher effective token ratio. By resolving the static length constraint, DAEDAL unlocks new potential for DLLMs, bridging a critical gap with their Autoregressive counterparts and paving the way for more efficient and capable generation.

### Example Usage
Here are full description of hyper-parameters and their default values:
```yaml
# our param name           vs   official param name
# initial_gen_length        --   initial_gen_length
# initial_eot_expand_thres  --   eos_confidence_threshold
# decode_eot_expand_thres  --   expand_eos_confidence_threshold
# low_conf_expand_thres    --   low_conf_threshold
# threshold (in vanilla)   --   high_conf_threshold
# max_gen_length           --   max_gen_length

initial_gen_length: 64

# at [Stage-1] Initial Length Adjustment, if the average confidence of the last `num_check_last_eot` tokens
# is lower than `initial_eot_expand_thres`, we expand the generation length
initial_eot_expand_thres: 0.5

# at [Stage-2] Iterative Denoising and Mask Insertion, if the average confidence of the last `num_check_last_eot` tokens
# is lower than `decode_eot_expand_thres`, we expand the generation length
decode_eot_expand_thres: 0.9
# at Stage 2, if we decide to expand, we replace select the lowest-confidence tokens whose confidence is below this threshold
# as the expansion point
low_conf_expand_thres: 0.1

# # of last EOT tokens to check
num_check_last_eot: 32
# replace the expansion point to expansion_factor masked tokens
expansion_factor: 8
max_gen_length: 2048
```
```bash
# Test DAEDAL decoding on GSM8K with LLaDA-7B-Instruct, run:
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=humaneval \
    batch_size=8 \
    seed=1234 \
    generation=daedal \
    generation.initial_gen_length=128 \
    model=llada-inst 
```

> [!NOTE]
> DAEDAL is not compatible to all current KV cache methods.