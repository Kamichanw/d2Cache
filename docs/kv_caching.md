## Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding
Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, Enze Xie

<details>
<summary><strong>Abstract</strong></summary>

Diffusion-based large language models (Diffusion LLMs) have shown promise for non-autoregressive text generation with parallel decoding capabilities. However, the practical inference speed of open-sourced Diffusion LLMs often lags behind autoregressive models due to the lack of Key-Value (KV) Cache and quality degradation when decoding multiple tokens simultaneously. To bridge this gap, we introduce a novel block-wise approximate KV Cache mechanism tailored for bidirectional diffusion models, enabling cache reuse with negligible performance drop. Additionally, we identify the root cause of generation quality degradation in parallel decoding as the disruption of token dependencies under the conditional independence assumption. To address this, we propose a confidence-aware parallel decoding strategy that selectively decodes tokens exceeding a confidence threshold, mitigating dependency violations and maintaining generation quality. Experimental results on LLaDA and Dream models across multiple LLM benchmarks demonstrate up to 27.6 throughput improvement with minimal accuracy loss, closing the performance gap with autoregressive models and paving the way for practical deployment of Diffusion LLMs.

</details>

### Hyper-parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `cache.use_dual` | `bool` | Enable dual caching mechanism (caches tokens after current decoding block). |

### Example Usage

```bash
# Test PrefixCache on HumanEval with LLaDA-7B-Base, run:
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=humaneval \
    batch_size=1 \
    seed=1234 \
    cache=prefix \
    generation=vanilla \
    generation.block_length=32 \
    model=llada-base
# Note that generation.block_length must be set to use semi ar decoding.
```

## dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching
Zhiyuan Liu, Yicun Yang, Yaojie Zhang, Junjie Chen, Chang Zou, Qingyuan Wei, Shaobo Wang, Linfeng Zhang

<details>
<summary><strong>Abstract</strong></summary>

Autoregressive Models (ARMs) have long dominated the landscape of Large Language Models. Recently, a new paradigm has emerged in the form of diffusion-based Large Language Models (dLLMs), which generate text by iteratively denoising masked segments. This approach has shown significant advantages and potential. However, dLLMs suffer from high inference latency. Traditional ARM acceleration techniques, such as Key-Value caching, are incompatible with dLLMs due to their bidirectional attention mechanism. To address this specific challenge, our work begins with a key observation that dLLM inference involves a static prompt and a partially dynamic response, where most tokens remain stable across adjacent denoising steps. Based on this, we propose dLLM-Cache, a training-free adaptive caching framework that combines long-interval prompt caching with partial response updates guided by feature similarity. This design enables efficient reuse of intermediate computations without compromising model performance. Extensive experiments on representative dLLMs, including LLaDA 8B and Dream 7B, show that dLLM-Cache achieves up to 9.1 x speedup over standard inference without compromising output quality. Notably, our method brings dLLM inference latency close to that of ARMs under many settings. Codes are provided in the supplementary material and will be released publicly on GitHub.

</details>

### Hyper-parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cache.kr` | `int` | `2` | Update interval for response part. |
| `cache.kp` | `int` | `50` | Update interval for prompt part. |
| `cache.rou` | `float` | `0.25` | Adaptive update ratio for response tokens. |

### Example Usage

To use `dLLMCache`:
```bash
# Test dLLM-Cache on MATH with Dream-v0-Instruct-7B, run:
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=math-500 \
    batch_size=1 \
    seed=1234 \
    cache=dllm \
    cache.rou=0.25 \
    cache.kp=50 \
    cache.kr=1 \
    generation=vanilla \
    model=dream-inst
```

## d2Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching
Yuchu Jiang, Yue Cai, Xiangzhong Luo, Jiale Fu, Jiarui Wang, Chonghan Liu, Xu Yang

<details>
<summary><strong>Abstract</strong></summary>

Diffusion-based large language models (dLLMs), despite their promising performance, still suffer from inferior inference efficiency. This is because dLLMs rely on bidirectional attention and cannot directly benefit from the standard key-value (KV) cache as autoregressive models (ARMs) do. To tackle this issue, we introduce *Dual aDaptive Cache* (dCache), which is a training-free approximate KV cache framework for accelerating dLLM inference. dCache features a two-stage fine-grained selection strategy to identify tokens and adaptively update their KV states at each decoding step, while caching the KV states of the remaining tokens for reuse. Furthermore, dCache naturally offers a more reliable decoding alternative, which can enable quasi left-to-right generation and mitigate premature overconfidence in tokens at the end of the sequence. Extensive experimental results on two representative dLLMs (*i.e.*, LLaDA and Dream) demonstrate that dCache not only achieves substantial inference speedups, but also yields consistent improvements in generation quality. The code is available at this https URL.

</details>

### Hyper-parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `cache.rollout_p` | `float` | `0.1` | Top-p ratio for attention rollout selection. |
| `cache.current_k` | `int` | `32` | Number of tokens to update in masked tokens. |
| `cache.sigma` | `float` | `10.0` | Sigma for certainty density calculation. |
| `cache.inflate_w` | `int` | `4` | Window size for mask inflation. |

### Example Usage

```bash
# Test d2Cache on MBPP with Dream-v0-Base-7B, run:
# certainty prior guided decoding is enabled by default when using d2Cache
accelerate launch \
    --num_machines 1 \
    --num_processes 4 \
    eval.py \
    dataset.name=humaneval \
    batch_size=1 \
    seed=1234 \
    attn_implementation=eager \
    cache=d2cache \
    cache.rollour_p=0.1 \
    cache.current_k=32 \
    cache.sigma=10 \
    generation=vanilla \
    model=llada-inst
```
When using parallel decoding, please disable certainty prior decoding explicitly, and set `inflate_w` to a proper value.
```bash
# d2Cache is also compatible with semi-ar decoding and parallel decoding, run:
# explicitly set generation.sigma to 0 to disable certainty prior guided decoding
accelerate launch \
    --num_machines 1 \
    --num_processes 1 \
    eval.py \
    dataset.name=humaneval \
    batch_size=1 \
    seed=1234 \
    attn_implementation=eager \
    cache=d2cache \
    cache.rollour_p=0.1 \
    cache.current_k=32 \
    cache.sigma=10 \
    cache.inflate_w=4 \
    generation.sigma=0
```


> [!NOTE]
> The `attn_implementation` parameter should be set to `eager` to obtain attention weights required by d2Cache.
