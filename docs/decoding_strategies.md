## Large Language Diffusion Models
Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, Chongxuan Li

<details>
<summary><strong>Abstract</strong></summary>

The capabilities of large language models (LLMs) are widely regarded as relying on autoregressive models (ARMs). We challenge this notion by introducing LLaDA, a diffusion model trained from scratch under the pre-training and supervised fine-tuning (SFT) paradigm. LLaDA employs a forward data masking process and a reverse generation process, parameterized by a Transformer to predict masked tokens. It provides a principled generative approach for probabilistic inference by optimizing a likelihood lower bound. Across extensive benchmarks on general tasks, math, code, and so on, LLaDA demonstrates strong scalability and performs comparably to our self-constructed ARM baselines. Remarkably, LLaDA 8B is competitive with strong LLMs like LLaMA3 8B in in-context learning and, after SFT, exhibits impressive instruction-following abilities in case studies such as multi-turn dialogue. Moreover, LLaDA addresses the reversal curse, surpassing GPT-4o in a reversal poem completion task. Our findings show the promise of diffusion models for language modeling at scale and challenge the common assumption that core LLM capabilities discussed above inherently depend on ARMs. Project page and codes: this https URL.

</details>

### Hyper-parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `generation.steps` | `int` | Total number of denoising steps. |
| `generation.gen_length` | `int` | The total length of the sequence to generate. |
| `generation.block_length` | `int` | The size of the block for semi-autoregressive decoding. |

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

<details>
<summary><strong>Abstract</strong></summary>

Diffusion-based large language models (Diffusion LLMs) have shown promise for non-autoregressive text generation with parallel decoding capabilities. However, the practical inference speed of open-sourced Diffusion LLMs often lags behind autoregressive models due to the lack of Key-Value (KV) Cache and quality degradation when decoding multiple tokens simultaneously. To bridge this gap, we introduce a novel block-wise approximate KV Cache mechanism tailored for bidirectional diffusion models, enabling cache reuse with negligible performance drop. Additionally, we identify the root cause of generation quality degradation in parallel decoding as the disruption of token dependencies under the conditional independence assumption. To address this, we propose a confidence-aware parallel decoding strategy that selectively decodes tokens exceeding a confidence threshold, mitigating dependency violations and maintaining generation quality. Experimental results on LLaDA and Dream models across multiple LLM benchmarks demonstrate up to 27.6 throughput improvement with minimal accuracy loss, closing the performance gap with autoregressive models and paving the way for practical deployment of Diffusion LLMs.

</details>

### Hyper-parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `generation.threshold` | `float` | Confidence threshold for retaining tokens in parallel decoding. |
| `generation.factor` | `float` | Speed-up factor for adaptive step allocation (alternative to threshold). |

### Example Usage

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

The factor based decoding proposed in Sec 3.3 is also implemented, to use this, add `generation.factor=1` to the command.


## PC-Sampler: Position-Aware Calibration of Decoding Bias in Masked Diffusion Models
Pengcheng Huang, Shuhao Liu, Zhenghao Liu, Yukun Yan, Shuo Wang, Zulong Chen, Tong Xiao

<details>
<summary><strong>Abstract</strong></summary>

Recent advances in masked diffusion models (MDMs) have established them as powerful non-autoregressive alternatives for sequence generation. Nevertheless, our preliminary experiments reveal that the generation quality of MDMs is still highly sensitive to the choice of decoding strategy. In particular, widely adopted uncertainty-based samplers suffer from two key limitations: a lack of global trajectory control and a pronounced bias toward trivial tokens in the early stages of decoding. These shortcomings restrict the full potential of MDMs. In this work, we introduce Position-Aware Confidence-Calibrated Sampling (PC-Sampler), a novel decoding strategy that unifies global trajectory planning with content-aware informativeness maximization. PC-Sampler incorporates a position-aware weighting mechanism to regulate the decoding path and a calibrated confidence score to suppress the premature selection of trivial tokens. Extensive experiments on three advanced MDMs across seven challenging benchmarks-including logical reasoning and planning tasks-demonstrate that PC-Sampler consistently outperforms existing MDM decoding strategies by more than 10% on average, significantly narrowing the performance gap with state-of-the-art autoregressive models. All codes are available at this https URL.

</details>

### Hyper-parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `generation.debias` | `bool` | Whether to enable position-aware bias calibration. |

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

<details>
<summary><strong>Abstract</strong></summary>

Diffusion-based large language models (dLLMs), despite their promising performance, still suffer from inferior inference efficiency. This is because dLLMs rely on bidirectional attention and cannot directly benefit from the standard key-value (KV) cache as autoregressive models (ARMs) do. To tackle this issue, we introduce *Dual aDaptive Cache* (dCache), which is a training-free approximate KV cache framework for accelerating dLLM inference. dCache features a two-stage fine-grained selection strategy to identify tokens and adaptively update their KV states at each decoding step, while caching the KV states of the remaining tokens for reuse. Furthermore, dCache naturally offers a more reliable decoding alternative, which can enable quasi left-to-right generation and mitigate premature overconfidence in tokens at the end of the sequence. Extensive experimental results on two representative dLLMs (*i.e.*, LLaDA and Dream) demonstrate that dCache not only achieves substantial inference speedups, but also yields consistent improvements in generation quality. The code is available at this https URL.

</details>

### Hyper-parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `generation.sigma` | `float` | Controls the decoding order (smaller $\sigma$ $\to$ more autoregressive). |

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

<details>
<summary><strong>Abstract</strong></summary>

Diffusion Large Language Models (DLLMs) are emerging as a powerful alternative to the dominant Autoregressive Large Language Models, offering efficient parallel generation and capable global context modeling. However, the practical application of DLLMs is hindered by a critical architectural constraint: the need for a statically predefined generation length. This static length allocation leads to a problematic trade-off: insufficient lengths cripple performance on complex tasks, while excessive lengths incur significant computational overhead and sometimes result in performance degradation. While the inference framework is rigid, we observe that the model itself possesses internal signals that correlate with the optimal response length for a given task. To bridge this gap, we leverage these latent signals and introduce DAEDAL, a novel training-free denoising strategy that enables Dynamic Adaptive Length Expansion for Diffusion Large Language Models. DAEDAL operates in two phases: 1) Before the denoising process, DAEDAL starts from a short initial length and iteratively expands it to a coarse task-appropriate length, guided by a sequence completion metric. 2) During the denoising process, DAEDAL dynamically intervenes by pinpointing and expanding insufficient generation regions through mask token insertion, ensuring the final output is fully developed. Extensive experiments on DLLMs demonstrate that DAEDAL achieves performance comparable, and in some cases superior, to meticulously tuned fixed-length baselines, while simultaneously enhancing computational efficiency by achieving a higher effective token ratio. By resolving the static length constraint, DAEDAL unlocks new potential for DLLMs, bridging a critical gap with their Autoregressive counterparts and paving the way for more efficient and capable generation.

</details>

### Hyper-parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `initial_gen_length` | `int` | `64` | Initial generation length. |
| `initial_eot_expand_thres` | `float` | `0.5` | Threshold for initial length expansion (Stage 1). |
| `decode_eot_expand_thres` | `float` | `0.9` | Threshold for dynamic length expansion (Stage 2). |
| `low_conf_expand_thres` | `float` | `0.1` | Confidence threshold for selecting expansion points. |
| `num_check_last_eot` | `int` | `32` | Number of last tokens to check for EOT confidence. |
| `expansion_factor` | `int` | `8` | Number of mask tokens to insert during expansion. |
| `max_gen_length` | `int` | `2048` | Maximum allowed generation length. |

### Example Usage

```bash
# Test DAEDAL decoding on GSM8K with LLaDA-7B-Instruct, run:
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=humaneval_instruct \
    batch_size=8 \
    seed=1234 \
    generation=daedal \
    generation.initial_gen_length=128 \
    model=llada-inst 
```

> [!NOTE]
> DAEDAL is not compatible to all current KV cache methods.


## KLASS: KL-Guided Fast Inference in Masked Diffusion Models
Seo Hyun Kim, Sunwoo Hong, Hojung Jung, Youngrok Park, Se-Young Yun

<details>
<summary><strong>Abstract</strong></summary>

Masked diffusion models have demonstrated competitive results on various tasks including language generation. However, due to its iterative refinement process, the inference is often bottlenecked by slow and static sampling speed. To overcome this problem, we introduce `KL-Adaptive Stability Sampling' (KLASS), a fast yet effective sampling method that exploits token-level KL divergence to identify stable, high-confidence predictions. By unmasking multiple tokens in each iteration without any additional model training, our approach speeds up generation significantly while maintaining sample quality. On reasoning benchmarks, KLASS achieves up to 2.78x wall-clock speedups while improving performance over standard greedy decoding, attaining state-of-the-art results among diffusion-based samplers. We further validate KLASS across diverse domains, including text, image, and molecular generation, showing its effectiveness as a broadly applicable sampler across different models.

</details>

### Hyper-parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `generation.kl_threshold` | `float` | Threshold for KL divergence to determine stability. |
| `generation.kl_history_length` | `int` | Length of history to consider for KL calculation. |

### Example Usage

```bash
# Test KLASS on GSM8K with LLaDA-7B-Instruct, run:
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    generation=klass \
    generation.block_length=64 \
    generation.kl_threshold=0.01 \
    generation.kl_history_length=2 \
    model=llada-inst 
```


## Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking
Heli Ben-Hamu, Itai Gat, Daniel Severo, Niklas Nolte, Brian Karrer

<details>
<summary><strong>Abstract</strong></summary>
Recent masked diffusion models (MDMs) have shown competitive performance compared to autoregressive models (ARMs) for language modeling. While most literature has focused on performance enhancing sampling procedures, efficient sampling from MDMs has been scarcely explored. We make the observation that often a given sequence of partially masked tokens determines the values of multiple unknown tokens deterministically, meaning that a single prediction of a masked model holds additional information unused by standard sampling procedures. Based on this observation, we introduce EB-Sampler, a simple drop-in replacement for existing samplers, utilizing an Entropy Bounded unmasking procedure that dynamically unmasks multiple tokens in one function evaluation with predefined approximate error tolerance. We formulate the EB-Sampler as part of a broad family of adaptive samplers for which we provide an error analysis that motivates our algorithmic choices. EB-Sampler accelerates sampling from current state of the art MDMs by roughly 2-3x on standard coding and math reasoning benchmarks without loss in performance. We also validate the same procedure works well on smaller reasoning tasks including maze navigation and Sudoku, tasks ARMs often struggle with.
</details>

### Hyper-parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `generation.gamma` | `float` | Threshold described in Eq.(2), influencing the number of steps where $\gamma=0$ will unmask one token each step and $\gamma=\infty$ will unmask all tokens at once. |

### Example Usage

```bash
# Test EB-Sampler on GSM8K with LLaDA-7B-Instruct, run:
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    generation=eb_sampler \
    generation.block_length=64 \
    generation.gamma=0.001 \
    model=llada-inst 
```

## Wide-In, Narrow-Out: Revokable Decoding for Efficient and Effective DLLMs
Feng Hong, Geng Yu, Yushi Ye, Haicheng Huang, Huangjie Zheng, Ya Zhang, Yanfeng Wang, Jiangchao Yao
<details>
<summary><strong>Abstract</strong></summary>
Diffusion Large Language Models (DLLMs) have emerged as a compelling alternative to Autoregressive models, designed for fast parallel generation. However, existing DLLMs are plagued by a severe quality-speed trade-off, where faster parallel decoding leads to significant performance degradation. We attribute this to the irreversibility of standard decoding in DLLMs, which is easily polarized into the wrong decoding direction along with early error context accumulation. To resolve this, we introduce Wide-In, Narrow-Out (WINO), a training-free decoding algorithm that enables revokable decoding in DLLMs. WINO employs a parallel draft-and-verify mechanism, aggressively drafting multiple tokens while simultaneously using the model's bidirectional context to verify and re-mask suspicious ones for refinement. Verified in open-source DLLMs like LLaDA and MMaDA, WINO is shown to decisively improve the quality-speed trade-off. For instance, on the GSM8K math benchmark, it accelerates inference by 6x while improving accuracy by 2.58%; on Flickr30K captioning, it achieves a 10x speedup with higher performance. More comprehensive experiments are conducted to demonstrate the superiority and provide an in-depth understanding of WINO.
</details>

### Hyper-parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `generation.wide_in_thres` | `float` | Threshold used in draft stage, which allows more possible tokens to be decoded at each step. |
| `generation.narrow_out_thres` | `float` | Confidence threshold for verification. |

### Example Usage

```bash
# Test WINO on GSM8K with LLaDA-7B-Instruct, run:
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=gsm8k \
    batch_size=1 \
    seed=1234 \
    generation=wino \
    generation.block_length=128 \
    generation.wide_in_thres=0.6 \
    generation.narrow_out_thres=0.9 \
    model=llada-inst 
```

> [!NOTE]
> WINO could theoretically be used in conjunction with `PrefixCache` or `DualCache`, but this is not currently implemented.