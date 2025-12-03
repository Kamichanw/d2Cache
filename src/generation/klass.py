import os
import torch
import torch.nn.functional as F

from typing import Type

from src.cache import dCache
from src.frame import Frame, FrameDelta, DecodeRecord, Intermediate
from src.generation.utils import prepare_logits_for_generation, sample_tokens
from src.generation.vanilla import get_num_transfer_tokens, confidence_unmasking
from src.utils import register

@register.gen_strategy("klass")
def klass_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    alg: str = "klass", # klass specific
    steps: int = 128,
    block_length: int = 32,
    gen_length: int = 128,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    mask_token_id: int | None = None,
    pad_token_id: int | None = None,
    # KLASS specific parameters
    conf_threshold: float = 0.9,
    kl_threshold: float = 0.01,
    kl_history_length: int = 2,
    # Common args
    output_hidden_states: bool = False,
    output_probs: bool = False,
    cache_cls: Type[dCache] | None = None,
) -> DecodeRecord:
    """
    KLASS generation strategy: KL-Adaptive Stability Sampling.
    """
    
    if mask_token_id is None and os.environ.get("MASK_TOKEN_ID", None) is None:
        raise ValueError(
            "mask_token_id must be provided either as an argument or an environment variable."
        )
    mask_token_id = mask_token_id or int(os.environ.get("MASK_TOKEN_ID"))

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    initial_frame = Frame.create_initial_frame(
        input_ids,
        gen_length=gen_length,
        mask_token_id=mask_token_id,
    ).to(device=model.device, dtype=model.dtype)

    if attention_mask is None and pad_token_id is not None:
        attention_mask = (input_ids != pad_token_id).long()

    if attention_mask is not None:
        attention_mask = F.pad(attention_mask, (0, gen_length), value=1).to(
            model.device
        )

    cache = cache_cls(model.config) if cache_cls is not None else None
    frame = initial_frame

    # KLASS State Initialization
    batch_size = input_ids.size(0)
    
    kl_history = None 
    p_prev = None

    deltas = []

    for block_idx in range(num_blocks):
        block_mask = torch.zeros(
            (input_ids.size(0), gen_length),
            dtype=torch.bool,
            device=model.device,
        )
        block_mask[
            :,
            block_idx * block_length : (block_idx + 1) * block_length,
        ] = True

        # Precompute fallback transfer counts (linear schedule)
        num_transfer_tokens_schedule = get_num_transfer_tokens(block_mask, steps_per_block)
        
        start_frame = frame.clone()
        if cache is not None:
            cache.on_block_start(block_mask, frame)
            
        block_deltas = []
        
        for i in range(steps_per_block):
            if cache is not None:
                cache.on_step_start(block_mask, frame)
            
            # --- KLASS Step Logic Start ---
            
            # 1. Prepare Inputs
            remaining_mask = frame.generated_tokens == mask_token_id
            can_generate = (block_mask & remaining_mask).any(dim=-1)
            
            if not torch.any(can_generate):
                break
                
            if cache is not None:
                cache.active_seq_mask = can_generate

            x = torch.cat([frame.prompts, frame.generated_tokens], dim=-1)[can_generate]
            current_attention_mask = (
                attention_mask[can_generate] if attention_mask is not None else None
            )

            # 2. Model Forward
            outputs = model(
                x,
                attention_mask=current_attention_mask,
                output_hidden_states=output_hidden_states,
                past_key_values=cache,
                use_cache=cache is not None,
            )
            
            # 3. Process Logits
            logits = prepare_logits_for_generation(model, outputs.logits)
            prompt_length = frame.prompts.shape[1]
            
            # Apply dream-specific mask correction if needed
            valid_mask = None
            if cache is not None and cache.active_q_mask is not None:
                if model.config.model_type.lower() == "dream":
                    valid_mask = cache.active_q_mask[:, prompt_length - 1 : -1]
                else:
                    valid_mask = cache.active_q_mask[:, prompt_length:]
                # Here we logicallly apply valid mask but logits are already sliced
                
            logits = logits[:, prompt_length:] # (B_active, gen_length, V)
            
            # 4. KLASS Core Logic: P_curr, Confidence, KL Calculation
            # We use float64 for precision as per KLASS implementation
            p_curr = F.softmax(logits.to(torch.float64), dim=-1)
            
            # Initialize history if needed
            if p_prev is None:
                # Shape: (B, gen_length, V) - expanding to full batch
                p_prev = torch.zeros(
                    (batch_size, gen_length, logits.size(-1)), 
                    dtype=torch.float64, 
                    device=logits.device
                )
                kl_history = torch.zeros(
                    (batch_size, gen_length, kl_history_length), 
                    dtype=torch.float64, 
                    device=logits.device
                )

            # Update active parts of history
            active_indices = torch.nonzero(can_generate).view(-1)
            
            if active_indices.numel() == 0:
                continue

            # Sample tokens (get x0 and confidence)
            # KLASS uses top-1 confidence by default for 'klass' alg
            confidence_active, x0_active, _ = sample_tokens(
                logits, # Pass original logits (float32/16) to sample_tokens
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                alg="default" if alg == "klass" else alg 
            )
            
            # Compute KL divergence
            eps = 1e-12
            p_prev_active = p_prev[active_indices]
            
            # kl_current_prev: (B_active, gen_length)
            kl_current_prev = (p_curr * (torch.log(p_curr + eps) - torch.log(p_prev_active + eps))).sum(dim=-1)
            
            # Update History buffers
            # Roll and update kl_history
            current_kl_hist = kl_history[active_indices]
            rolled_hist = torch.roll(current_kl_hist, shifts=-1, dims=-1)
            rolled_hist[..., -1] = kl_current_prev
            
            if valid_mask is not None:
                # Only update history for valid positions
                # valid_mask: (B_active, gen_length)
                kl_history[active_indices] = torch.where(
                    valid_mask.unsqueeze(-1), 
                    rolled_hist, 
                    current_kl_hist
                )
                p_prev[active_indices] = torch.where(
                    valid_mask.unsqueeze(-1), 
                    p_curr, 
                    p_prev_active
                )
            else:
                kl_history[active_indices] = rolled_hist
                p_prev[active_indices] = p_curr
            
            # 5. Determine Selection Masks (Stable & Confident)
            
            if i >= kl_history_length - 1:
                stable_mask_active = torch.all(kl_history[active_indices] < kl_threshold, dim=-1)
            else:
                stable_mask_active = torch.zeros_like(confidence_active, dtype=torch.bool)
            
            # conf_mask_active is handled inside confidence_unmasking via 'threshold'
            
            # Only consider tokens in current block and currently masked
            current_block_mask_active = block_mask[active_indices]
            current_remaining_mask_active = remaining_mask[active_indices]
            
            # Transfer mask: In-Block AND Masked (Base mask for all operations)
            base_mask = current_block_mask_active & current_remaining_mask_active
            if valid_mask is not None:
                base_mask = base_mask & valid_mask

            # Mask 1: Stable AND Base (For KLASS strategy)
            stable_transfer_mask = stable_mask_active & base_mask
            
            # Mask 2: Base only (For Fallback strategy)
            fallback_transfer_mask = base_mask

            # 6. Apply Unmasking Strategy
            
            # We need fallback counts for the active batch
            fallback_counts = num_transfer_tokens_schedule[active_indices, i] # (B_active,)

            # Call 1: KLASS Selection (Stable & High Confidence)
            # num_transfer_tokens=0 means "select only based on threshold", no forced top-k filling
            ta = confidence_unmasking(
                scores=confidence_active,
                transfer_index_mask=stable_transfer_mask,
                num_transfer_tokens=torch.zeros_like(fallback_counts),
                threshold=conf_threshold
            )
            
            # Call 2: Fallback Selection (Top-K Confidence, ignoring stability)
            # threshold=None means "select purely based on num_transfer_tokens (Top-K)"
            tb = confidence_unmasking(
                scores=confidence_active,
                transfer_index_mask=fallback_transfer_mask,
                num_transfer_tokens=fallback_counts,
                threshold=None
            )
            
            # Merge results: Use KLASS selection if available, otherwise fallback to Top-K
            transfer_index = tuple(a if a.numel() > 0 else b for a, b in zip(ta, tb))
            
            # 7. Create Delta
            
            full_transfer_index = []
            active_ptr = 0
            for is_active in can_generate:
                if is_active:
                    full_transfer_index.append(transfer_index[active_ptr])
                    active_ptr += 1
                else:
                    full_transfer_index.append(torch.tensor([], dtype=torch.long, device=model.device))
            
            full_transfer_index = tuple(full_transfer_index)
            
            delta = FrameDelta(
                transfer_index=full_transfer_index,
                decoded_tokens=x0_active,
                confidence=confidence_active,
                probs=None,
                intermediate=Intermediate(
                    hidden_states=tuple((i, hs) for i, hs in enumerate(outputs.hidden_states)) if output_hidden_states else tuple()
                ),
            )
            
            # Unbatch if necessary
            delta = delta.unbatch() if not frame.is_batched else delta
            
            if cache is not None:
                cache.on_step_end(block_mask, frame, delta)

            block_deltas.append(delta.to("cpu"))
            frame = frame.apply_delta(delta)

            # --- KLASS Step Logic End ---

        if cache is not None:
            cache.on_block_end(block_mask, start_frame, block_deltas)

        deltas.extend(block_deltas)

    return DecodeRecord(
        initial_frame=initial_frame.to("cpu"),
        deltas=deltas,
        block_length=block_length,
    )

