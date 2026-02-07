import math
import torch
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache

from src.frame import DecodeRecord, Frame, FrameDelta
from src.utils import register


def _top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[..., -1, None]
    return torch.where(
        logits < min_values, torch.full_like(logits, float("-inf")), logits
    )


def _top_p_logits(logits: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(
        torch.full_like(logits, False, dtype=torch.bool), -1, sorted_indices, sorted_mask
    )
    return logits.masked_fill(mask_indices, float("-inf"))


def _sample_tokens(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        logits: (B, L, V)
    Returns:
        tokens: (B, L)
        token_probs: (B, L) probability of selected tokens
        probs: (B, L, V) full distribution (float32)
    """
    if temperature == 0.0:
        tokens = torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits.to(torch.float32), dim=-1)
        token_probs = torch.gather(probs, -1, tokens.unsqueeze(-1)).squeeze(-1)
        return tokens, token_probs, probs

    scaled = logits / temperature
    if top_k > 0:
        scaled = _top_k_logits(scaled, top_k)
    if top_p < 1.0:
        scaled = _top_p_logits(scaled, top_p)

    probs = torch.softmax(scaled.to(torch.float32), dim=-1)
    tokens = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(
        probs.size(0), probs.size(1)
    )
    token_probs = torch.gather(probs, -1, tokens.unsqueeze(-1)).squeeze(-1)
    return tokens, token_probs, probs


def _get_num_transfer_tokens(block_length: int, denoising_steps: int) -> torch.Tensor:
    base = block_length // denoising_steps
    remainder = block_length % denoising_steps
    num_transfer_tokens = torch.full((denoising_steps,), base, dtype=torch.long)
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens


@torch.no_grad()
def _block_diffusion_generate_one(
    model,
    *,
    prompt_ids: torch.Tensor,
    mask_id: int,
    eos_token_id: int,
    gen_length: int,
    block_length: int,
    denoising_steps: int,
    temperature: float,
    top_k: int,
    top_p: float,
    remasking_strategy: str,
    confidence_threshold: float,
    eb_threshold: float | None,
    stop_until_eot: bool,
) -> torch.Tensor:
    device = prompt_ids.device
    prompt_length = int(prompt_ids.numel())
    num_blocks = math.ceil((prompt_length + gen_length) / block_length)
    total_length = num_blocks * block_length

    block_mask = torch.tril(
        torch.ones((num_blocks, num_blocks), device=device, dtype=torch.float32)
    )
    # (1, total_length, total_length)
    attention_mask = (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
        .unsqueeze(0)
    )
    position_ids = torch.arange(total_length, device=device).unsqueeze(0)

    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = prompt_ids.unsqueeze(0)

    past_key_values = DynamicCache()

    prefill_blocks = prompt_length // block_length
    prefill_length = prefill_blocks * block_length

    if prefill_length > 0:
        cur_x = x[:, :prefill_length]
        cur_attn_mask = attention_mask[:, :prefill_length, :prefill_length]
        cur_position_ids = position_ids[:, :prefill_length]
        model(
            cur_x,
            attention_mask=cur_attn_mask,
            position_ids=cur_position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=True,
        )

    if denoising_steps <= 0:
        raise ValueError(f"{denoising_steps=} must be > 0 for SDAR decoding.")
    num_transfer_tokens = _get_num_transfer_tokens(block_length, denoising_steps).to(
        device
    )

    for num_block in range(prefill_blocks, num_blocks):
        block_start = num_block * block_length
        block_end = (num_block + 1) * block_length

        cur_x = x[:, block_start:block_end].clone()
        cur_attn_mask = attention_mask[:, block_start:block_end, :block_end]
        cur_position_ids = position_ids[:, block_start:block_end]

        for step in range(denoising_steps + 1):
            mask_index = cur_x == mask_id
            if int(mask_index.sum()) == 0:
                model(
                    cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    store_kv=True,
                )
                break

            logits = model(
                cur_x,
                attention_mask=cur_attn_mask,
                position_ids=cur_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=False,
            ).logits

            sampled, sampled_p, probs = _sample_tokens(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            if step >= denoising_steps:
                raise RuntimeError(
                    "SDAR decoding did not finish within denoising_steps; "
                    "consider increasing `steps` or reducing `gen_length`."
                )
            k = int(num_transfer_tokens[step].item())

            if remasking_strategy == "sequential":
                transfer_index = torch.zeros_like(sampled, dtype=torch.bool)
                for j in range(cur_x.size(0)):
                    if not mask_index[j].any():
                        continue
                    first = int(mask_index[j].nonzero(as_tuple=True)[0].min().item())
                    transfer_index[j, first : first + k] = True

            elif remasking_strategy == "low_confidence_static":
                confidence = torch.where(mask_index, sampled_p, -torch.inf)
                _, idx = torch.topk(confidence, k, dim=1)
                transfer_index = torch.zeros_like(sampled, dtype=torch.bool).scatter_(
                    1, idx, True
                )

            elif remasking_strategy == "low_confidence_dynamic":
                confidence = torch.where(mask_index, sampled_p, -torch.inf)
                transfer_index = torch.zeros_like(sampled, dtype=torch.bool)
                for j in range(confidence.size(0)):
                    high_conf_mask = confidence[j] > confidence_threshold
                    if int(high_conf_mask.sum()) >= k:
                        transfer_index[j] = high_conf_mask
                    else:
                        _, idx = torch.topk(confidence[j], k)
                        transfer_index[j, idx] = True

            elif remasking_strategy == "entropy_bounded":
                if eb_threshold is None:
                    raise ValueError(
                        "eb_threshold must be provided when remasking_strategy='entropy_bounded'."
                    )
                eps = 1e-12
                entropies = -(
                    probs.clamp_min(eps) * probs.clamp_min(eps).log()
                ).sum(dim=-1)
                entropies = torch.where(mask_index, entropies, torch.inf)
                ent_sorted, order = torch.sort(entropies, dim=1, descending=False)
                cumsum = torch.cumsum(ent_sorted, dim=1)
                transfer_index = torch.zeros_like(sampled, dtype=torch.bool)
                for j in range(cur_x.size(0)):
                    t = torch.tensor(eb_threshold, device=device, dtype=cumsum.dtype)
                    kk = int(torch.searchsorted(cumsum[j], t, right=False).item())
                    kk = max(1, min(kk, int(mask_index[j].sum().item())))
                    selected = order[j, :kk]
                    transfer_index[j, selected] = True

            else:
                raise ValueError(f"Unknown remasking_strategy: {remasking_strategy!r}")

            cur_x[transfer_index] = sampled[transfer_index]

        x[:, block_start:block_end] = cur_x

        if stop_until_eot and (x[:, prompt_length:] == eos_token_id).any():
            break

    out = x[:, prompt_length : prompt_length + gen_length].clone()
    out[out == mask_id] = eos_token_id
    return out.squeeze(0)


@register.gen_strategy("sdar")
@torch.no_grad()
def sdar_generate(
    model,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    gen_length: int,
    block_length: int,
    steps: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    remasking_strategy: str = "low_confidence_dynamic",
    confidence_threshold: float = 0.85,
    eb_threshold: float | None = 0.35,
    stop_until_eot: bool = True,
    mask_token_id: int | None = None,
    eot_token_id: int | None = None,
    **_: object,
) -> DecodeRecord:
    """
    SDAR block-diffusion decoding (ported from SDAR repo `generate.py`).
    Notes:
    - Supports left-padded `input_ids` using `attention_mask` to crop the prompt.
    - `steps` is the *total* denoising steps across blocks; per-block denoising steps are derived as
      `denoising_steps = steps // (gen_length // block_length)` (matching the constraints in `configs/gen_args.py`).
    """
    if input_ids.dim() != 2:
        raise ValueError(f"Expected input_ids shape (B, L), got {tuple(input_ids.shape)}")
    batch_size = input_ids.size(0)

    if gen_length <= 0 or block_length <= 0 or steps <= 0:
        raise ValueError(f"Invalid lengths: {gen_length=}, {block_length=}, {steps=}")
    if gen_length % block_length != 0:
        raise ValueError(f"{gen_length=} must be divisible by {block_length=}")
    num_gen_blocks = gen_length // block_length
    if steps % num_gen_blocks != 0:
        raise ValueError(f"{steps=} must be divisible by num_gen_blocks={num_gen_blocks}")
    denoising_steps = steps // num_gen_blocks

    mask_id = int(mask_token_id if mask_token_id is not None else getattr(model.config, "mask_token_id"))
    eos_id = int(eot_token_id if eot_token_id is not None else getattr(model.config, "eos_token_id"))

    outputs: list[torch.Tensor] = []
    for i in range(batch_size):
        if attention_mask is not None:
            prompt_len = int(attention_mask[i].sum().item())
            prompt_ids = input_ids[i, -prompt_len:]
        else:
            prompt_ids = input_ids[i]

        outputs.append(
            _block_diffusion_generate_one(
                model,
                prompt_ids=prompt_ids,
                mask_id=mask_id,
                eos_token_id=eos_id,
                gen_length=gen_length,
                block_length=block_length,
                denoising_steps=denoising_steps,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                remasking_strategy=remasking_strategy,
                confidence_threshold=confidence_threshold,
                eb_threshold=eb_threshold,
                stop_until_eot=stop_until_eot,
            )
        )

    decoded_tokens = torch.stack(outputs, dim=0)
    initial_frame = Frame.create_initial_frame(
        input_ids, gen_length=gen_length, mask_token_id=mask_id
    )
    transfer_index = tuple(
        torch.arange(gen_length, device=input_ids.device) for _ in range(batch_size)
    )
    delta = FrameDelta(transfer_index=transfer_index, decoded_tokens=decoded_tokens)
    record = DecodeRecord(initial_frame=initial_frame, deltas=[delta], block_length=block_length)
    return record

