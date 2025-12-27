import os
import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from src.llada.modeling_llada import LLaDAModelLM
from src.generation.wino import wino_generate
from src.generation.utils import decode_final_frame
import time

    
@torch.no_grad()
def decoding_wino(model, prompt, gen_length=128, block_length=128, temperature=0., mask_id=126336, threshold=0.6, threshold_back=0.9):

    device = model.device
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

def main():
    device = 'cuda:2'
    model_path = "GSAI-ML/LLaDA-8B-Instruct"
    gen_length = 256
    block_length = 128
    steps = 256
    mask_id = 126336
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = LLaDAModelLM.from_pretrained(model_path,torch_dtype=torch.bfloat16).to(device).eval()
    prompt = "<|startoftext|><|start_header_id|>user<|end_header_id|>\n\nfrom typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    # out, step = decoding_default(model, input_ids, steps=steps, gen_length=gen_length, block_length=block_length, temperature=0., cfg_scale=0., remasking='low_confidence')
    # print(f'Default: {tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]}')
    # print(f'\nSteps: {step}\n\n')

    out, step = decoding_wino(model, input_ids, gen_length=gen_length, block_length=block_length, temperature=0., threshold=0.6, threshold_back=0.9)
    print(f'WINO (reference): {tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]}')
    print(f'\nSteps: {step}\n')

    # Repo wino
    os.environ["MASK_TOKEN_ID"] = str(mask_id)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    os.environ["PAD_TOKEN_ID"] = str(pad_token_id)
    record = wino_generate(
        model,
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        gen_length=gen_length,
        block_length=block_length,
        temperature=0.0,
        top_k=None,
        top_p=None,
        sigma=None,
        mask_token_id=mask_id,
        pad_token_id=pad_token_id,
    )
    final_frame = record[-1]
    print(f"Repo wino: {decode_final_frame(tokenizer, final_frame, skip_special_tokens=True)[0]}")
    print(f'\nSteps: {len(record)}\n')

if __name__ == '__main__':
    main()