<h1 align="center">	Research dLLM </h1>
<p align="center">
Codebase for Diffusion Language Models Research
</p>
<p align="center">
<img src="./assets/logo.png" width="400">
</p>

<div align="center">
<p align="center">
      <a href="https://github.com/Kamichanw/d2Cache/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/Kamichanw/d2Cache?color=ccf" />
      </a>
      <a href="https://github.com/Kamichanw/d2Cache/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/Kamichanw/d2Cache?color=0088ff" />
      </a>
      <a href="https://github.com/Kamichanw/d2Cache/discussions">
        <img alt="Issues" src="https://img.shields.io/github/discussions/Kamichanw/d2Cache?color=0088ff" />
      </a>
      <a href="https://github.com/Kamichanw/d2Cache/pulls">
        <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/Kamichanw/d2Cache?color=0088ff" />
      </a>
</p>
</div>

<hr>


**Research dLLM** is a research-focused library for Diffusion Language Models (dLLMs), providing a comprehensive collection of baseline methods (primarily KV caching and decoding strategies) for reproducible experiments.

## Why Use This Codebase?

- **Unified Evaluation Framework**: Research dLLM provides a standardized testing environment that allows users to seamlessly switch between different baseline methods. 
- **Clean and Well-Documented Code**: Research dLLM is written with a strong emphasis on clarity and readability. 
- **Active and Ongoing Maintenance**: Research dLLM is actively maintained and continuously updated. More out-of-the-box baselines will be included in the future.

 # News
[25/9/30] We released [code reading guides](./docs/code_reading_guides.md), hoping this can help you to grasp our work :)

[25/10/13] Now, batch inference is supported!

[25/12/06] 🚀 We are looking for help to reproduce our experimental results using A100 GPUs. Please contact us if you can assist with the benchmarking.

# Supported Methods
The corresponding usages can be found [here](./docs/kv_caching.md).
## KV Caching

| Method | Paper | Original Code Repo |
|:---|:---:|:---:|
| PrefixCache / DualCache | <a href="https://arxiv.org/abs/2505.22618"> <img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2505.22618-red"></a> | <a href="https://github.com/NVLabs/Fast-dLLM"><img alt="Static Badge" src="https://img.shields.io/github/stars/NVLabs/Fast-dLLM?style=social&label=Stars"></a> |
| dLLM Cache | <a href="https://arxiv.org/abs/2506.06295"> <img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2506.06295-red"></a> | <a href="https://github.com/maomaocun/dLLM-Cache"><img alt="Static Badge" src="https://img.shields.io/github/stars/maomaocun/dLLM-Cache?style=social&label=Stars"></a> |
| d2Cache | <a href="https://arxiv.org/abs/2509.23094"> <img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2509.23094-red"></a> | This Repo |

## Decoding Strategies
The corresponding usages can be found [here](./docs/decoding_strategies.md).

| Method | Paper | Original Code Repo |
|:---|:---:|:---:|
| Vanilla / Semi-AR | <a href="https://arxiv.org/abs/2502.09992"><img alt="Static Badge" src="https://img.shields.io/badge/NIPS Oral-2502.09992-purple"></a> | <a href="https://github.com/ML-GSAI/LLaDA"><img alt="Static Badge" src="https://img.shields.io/github/stars/ML-GSAI/LLaDA?style=social&label=Stars"></a> |
| Parallel | <a href="https://arxiv.org/abs/2505.22618"><img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2505.22618-red"></a> | <a href="https://github.com/NVlabs/Fast-dLLM"><img alt="Static Badge" src="https://img.shields.io/github/stars/NVlabs/Fast-dLLM?style=social&label=Stars"></a> |
| PC-Sampler | <a href="https://arxiv.org/abs/2508.13021"><img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2508.13021-red"></a> | <a href="https://github.com/NEUIR/PC-Sampler"><img alt="Static Badge" src="https://img.shields.io/github/stars/NEUIR/PC-Sampler?style=social&label=Stars"></a> |
| Certainty Prior Decoding | <a href="https://arxiv.org/abs/2509.23094"><img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2509.23094-red"></a> | This Repo|
|DAEDAL |<a href="https://arxiv.org/abs/2508.00819"><img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2508.00819-red"></a> | <a href="https://github.com/Li-Jinsong/DAEDAL"><img alt="Static Badge" src="https://img.shields.io/github/stars/Li-Jinsong/DAEDAL?style=social&label=Stars"></a> |
| KLASS | <a href="https://arxiv.org/abs/2511.05664"><img alt="Static Badge" src="https://img.shields.io/badge/NIPS Spotlight-2511.05664-purple"></a> | <a href="https://github.com/shkim0116/KLASS"><img alt="Static Badge" src="https://img.shields.io/github/stars/shkim0116/KLASS?style=social&label=Stars"></a> |
| EB-Sampler | <a href="https://arxiv.org/abs/2505.24857"><img alt="Static Badge" src="https://img.shields.io/badge/NIPS-2511.05664-purple"></a> | Not release |
| WINO | <a href="https://arxiv.org/abs/2507.18578"><img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2511.05664-red"></a> | <a href="https://github.com/shkim0116/KLASS"><img alt="Static Badge" src="https://img.shields.io/github/stars/Feng-Hong/WINO-DLLM?style=social&label=Stars"> |
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

Please check docs/ for detailed instructions on how to run evaluations with different methods.
A quick example is shown in [srcipts/run_eval.sh](./scripts/run_eval.sh).

Available models:
- llada-base: GSAI-ML/LLaDA-8B-Base
- llada-inst: GSAI-ML/LLaDA-8B-Instruct
- dream-base: Dream-org/Dream-v0-Base-7B
- dream-inst: Dream-org/Dream-v0-Instruct-7B

Available datasets:
- gsm8k
- humaneval / humaneval_instruct
- math-500
- mbpp / mbpp_instruct
- ... (all tasks specified in `lm-eval` are available)

> [!IMPORTANT]
> To evaluate humaneval dataset on Dream-v0-Instruct-7B, please use its corresponding instruct variants, i.e., `humaneval_instruct`.

Additional general arguments can be specified in `configs/generation/*.yaml` or `configs/gen_args.py`.

# Starchart

[![Star History Chart](https://api.star-history.com/svg?repos=Kamichanw/d2Cache&type=Date)](https://star-history.com/#Kamichanw/d2Cache&Date)


# Citation
If you find d²Cache or this repository useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{jiang2025d2cache,
  title={d $\^{} 2$ Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching},
  author={Jiang, Yuchu and Cai, Yue and Luo, Xiangzhong and Fu, Jiale and Wang, Jiarui and Liu, Chonghan and Yang, Xu},
  journal={arXiv preprint arXiv:2509.23094},
  year={2025}
}
```
# Acknowledgment
We would like to thank the authors of all models and baseline methods for their excellent work and open-source contributions.

# License
This project is licensed under the Apache 2.0 License. See the [LICENSE](./LICENSE) file for details.