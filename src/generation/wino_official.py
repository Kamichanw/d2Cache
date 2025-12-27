import os
import torch
import torch.nn.functional as F
import numpy as np

from src.utils import register
from src.frame import Frame, DecodeRecord, FrameDelta, Intermediate

@torch.no_grad()
def decoding_wino(model, prompt, gen_length=128, block_length=128, temperature=0., mask_id=126336, threshold=0.6, threshold_back=0.9):

    device = model.device
    # Ensure prompt is 2D [1, seq_len]
    if prompt.dim() == 1:
        prompt = prompt.unsqueeze(0)
    
    # decoding_wino in compare.py hardcodes batch size 1 logic in some places (e.g. unsqueeze(0) for pos ids)
    # We will assume batch_size=1 for now as per the script.
    
    x_block = torch.full((1, prompt.shape[1] + gen_length + block_length), mask_id, dtype=torch.long).to(model.device)
    x_block[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x_block != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    step = 0
    

    for num_block in range(num_blocks):
        block_step = 0
        mask_index_block = (x_block == mask_id) # b, l
        mask_index_block[:, prompt.shape[1] + (num_block + 1) * block_length:] = False
        
        unmask_index_block = torch.full_like(mask_index_block, False)
        unmask_index_block[:,  -block_length:] = ~mask_index_block[:, prompt.shape[1] + num_block* block_length: prompt.shape[1] + (num_block + 1) * block_length]
        position_ids = torch.cat(
            [
                torch.arange(prompt.shape[1] + gen_length, device=device),
                torch.arange(
                    prompt.shape[1] + num_block * block_length,
                    prompt.shape[1] + (num_block + 1) * block_length,
                    device=device,
                ),
            ]
        ).unsqueeze(0)
        attention_mask = torch.ones(1, 1, x_block.shape[1], x_block.shape[1], dtype=torch.bool).to(device)
        attention_mask[:, :, :, -block_length:] = False
        attention_mask[:, :, -block_length:, -block_length:] = torch.ones(block_length, block_length, dtype=torch.bool).to(device)
        attention_mask[:, :, -block_length:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] = ~torch.eye(block_length, dtype=torch.bool).to(device)
        last_accept = 30
        while mask_index_block.any():
            max_accept = min(max(int(mask_index_block.sum() * 0.7), 5), 20)
            logits = model(x_block, attention_mask=attention_mask, position_ids=position_ids).logits # b, l, vocab_size
            x0 = torch.argmax(logits, dim=-1) # b, l
            unmask_index_block_shift_left = torch.zeros_like(unmask_index_block)
            unmask_index_block_shift_left[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] = unmask_index_block[:, -block_length:]
            x0[unmask_index_block] = x_block[unmask_index_block_shift_left]

            p = F.softmax(logits.to(torch.float64), dim=-1) # b, l, vocab_size
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            x0 = torch.where(mask_index_block, x0, x_block) # replace the masked tokens with the predicted tokens
            confidence = torch.where(mask_index_block, x0_p, -np.inf) # keep the confidence of the masked tokens
            confidence_back = torch.where(unmask_index_block, x0_p, np.inf)
            

            transfer_index = confidence > threshold
            if transfer_index.sum() > max_accept:
                # get top max_accept tokens
                _, indices = torch.topk(confidence, k=max_accept, largest=True)
                transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
                transfer_index.view(-1)[indices] = True
            
            # always transfer the max confidence token
            else:
                if not transfer_index.any():
                    max_confidence_index = torch.argmax(confidence)
                    transfer_index.view(-1)[max_confidence_index] = True
            x_block[transfer_index] = x0[transfer_index]
            
            num_accept = transfer_index.sum()
            
            if num_accept > 1:
                remask_index = confidence_back < threshold_back
                if remask_index.sum() >= last_accept:
                    num_remask = last_accept - 1
                    confidence_flat = confidence_back.view(-1)
                    temp_mask = torch.zeros_like(confidence_flat, dtype=torch.bool)
                    _, indices = torch.topk(confidence_flat, k=num_remask, largest=False)
                    temp_mask[indices] = True
                    remask_index = temp_mask.view(confidence_back.shape)
            else:
                remask_index = torch.zeros_like(transfer_index)
            
            remask_index_shift = torch.zeros_like(remask_index)
            remask_index_shift[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] = remask_index[:, -block_length:]
            x_block[remask_index_shift] = mask_id
            mask_index_block[transfer_index] = False
            mask_index_block[remask_index_shift] = True
            block_step += 1
            transfer_index_shift = torch.zeros_like(transfer_index)
            transfer_index_shift[:, -block_length:] = transfer_index[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length]
            unmask_index_block[transfer_index_shift] = True
            unmask_index_block[remask_index] = False
            last_accept = num_accept

        step += block_step

    return x_block[:, :prompt.shape[1] + gen_length], step


@register.gen_strategy("wino_official")
def wino_official_generate(
    model,
    input_ids: torch.Tensor,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    mask_token_id: int | None = None,
    # wino specific args from compare.py signature
    wide_in_thres: float = 0.6,
    narrow_out_thres: float = 0.9,
    **kwargs
) -> DecodeRecord:
    
    mask_token_id = mask_token_id or int(os.environ.get("MASK_TOKEN_ID", 126336))
    
    # Create initial frame
    initial_frame = Frame.create_initial_frame(
        input_ids,
        gen_length=gen_length,
        mask_token_id=mask_token_id,
    ).to(device=model.device, dtype=model.dtype)

    # Run decoding_wino
    # Note: decoding_wino expects single batch.
    # input_ids is (B, L)
    batch_size = input_ids.shape[0]
    if batch_size != 1:
        raise ValueError("wino_official strategy only supports batch_size=1")

    # Call the official implementation
    final_tokens, steps = decoding_wino(
        model, 
        input_ids, 
        gen_length=gen_length, 
        block_length=block_length, 
        temperature=temperature,
        mask_id=mask_token_id,
        threshold=wide_in_thres, # map wide_in_thres to threshold
        threshold_back=narrow_out_thres # map narrow_out_thres to threshold_back
    )

    # final_tokens is (1, prompt_len + gen_len)
    prompt_len = input_ids.shape[1]
    generated_part = final_tokens[:, prompt_len:]  # (1, gen_length)
    
    # Create FrameDelta to transfer all generated tokens
    # transfer_index: tuple of tensors, one per batch item
    # Each tensor contains indices in generated_tokens (0 to gen_length-1) that we want to transfer
    indices = torch.arange(gen_length, device=model.device, dtype=torch.long)
    
    # decoded_tokens: (active_batch_size, gen_length) - contains the actual token values
    # For batched delta (transfer_index is tuple), decoded_tokens should have shape (1, gen_length)
    delta = FrameDelta(
        transfer_index=(indices,),  # Tuple for batch, indicates we transfer all tokens
        decoded_tokens=generated_part,  # (1, gen_length) - the generated tokens
        confidence=torch.ones_like(generated_part, dtype=torch.float32), 
        extra={"step": steps}
    ).to(model.dtype)  # Convert to model dtype to match initial_frame dtype

    record = DecodeRecord(
        initial_frame=initial_frame.to("cpu"),
        deltas=[delta.to("cpu")],
        block_length=block_length,
    )
    
    return record

