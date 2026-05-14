# Customization Guide

## 1. Implementing a Custom Cache Mechanism

To introduce a new caching mechanism, one must derive a new class from the abstract base class `dCache`. This architecture allows for precise control over the model's internal states during inference, enabling techniques such as KV-caching, attention masking, or state compression.

### Base Class Definition
The `dCache` class, located in `src/cache/base.py`, serves as the foundational interface. It utilizes Python context managers to intercept and modify the computation flow within the model's layers.

### Key Methods to Override

Subclasses should override the following methods to implement specific caching behaviors:

*   **`__init__(self, model_config, ...)`**: 
    Initializes the cache instance. This is where internal storage structures (e.g., lists for Key/Value tensors) should be instantiated.

*   **`model_forward(self, x, position_ids=None, attention_mask=None)`**:
    A context manager invoked around the entire model forward pass. It yields a `ModelForwardContext`, whose `input_embeds`, `position_ids`, and `attention_mask` fields are consumed by the model before layer-wise computation begins. Cache implementations that slice the sequence should restore `ctx.logits` before the context exits.

*   **`attention(self, layer_idx, ...)`**: 
    This is the most critical context manager for caching implementations. It intercepts the attention computation at each layer.
    *   **Input**: Receives the layer index, normalized input tensor `hidden_states`, and projection layers (`q_proj`, `k_proj`, `v_proj`).
    *   **Operation**: The implementation should compute or retrieve the Query, Key, and Value states. It must yield an `AttentionContext` object.
    *   **Usage**: This method is typically used to store computed K/V pairs for future reuse or to modify the attention mask to enforce specific sparsity patterns. Layer implementations should assign `ctx.o`, and may assign `ctx.attn_weights` when attention weights are needed.

*   **`ffn(self, layer_idx, hidden_states)`**: 
    A context manager for the Feed-Forward Network (FFN) layers. It yields an `FFNContext`; model code should read `ctx.hidden_states`, compute the FFN output, and assign `ctx.ffn_out`.

*   **`cache_kwargs`**:
    A property read by model attention modules when they call `past_key_values.update(...)`. The base implementation provides `active_rows` and, when `active_q_mask` is set, `cache_position`. Subclasses can override it to pass cache-specific metadata to their layer caches.

*   **Lifecycle Hooks**:
    *   `on_block_start(self, block_mask, frame)`: Executed before decoding a block. Useful for initializing block-local state.
    *   `on_step_start(self, block_mask, frame)`: Executed at the beginning of each generation step. Useful for preparing masks based on the current generation status.
    *   `on_step_end(self, block_mask, frame, delta)`: Executed after the generation step. This is where the cache should be updated with the newly generated information (e.g., updating confidence scores or density metrics).
    *   `on_block_end(self, block_mask, frame, deltas)`: Executed after a block finishes. Block-level caches can use this hook to prepare state for the next block.

### Configuration
To use your custom cache, create a new YAML configuration file in `configs/cache/` (e.g., `configs/cache/my_cache.yaml`).
```yaml
_target_: src.cache.MyCache
# ... specific parameters for __init__
```


## 2. Developing a New Decoding Strategy

The framework supports modular decoding strategies, allowing researchers to implement custom generation algorithms (e.g., autoregressive, non-autoregressive, or iterative refinement) without modifying the core model code.

### Registration Mechanism
New strategies are registered using the `@register` decorator. This allows the strategy to be selected via the configuration file or command-line arguments using its registered name.


### Implementation Workflow
A decoding strategy is implemented as a function that accepts the model, input identifiers, and strategy-specific hyperparameters. It must return a `DecodeRecord` object containing the generation history. The workflow typically involves the following steps:

1.  **Frame Initialization**: 
    Create an initial `Frame` object. This object encapsulates the state of the generation, including the prompts and the current sequence of tokens (initially masked).
    ```python
    frame = Frame.create_initial_frame(input_ids, num_new_tokens=..., mask_token_id=...)
    ```

2.  **Cache Initialization**: 
    Instantiate the appropriate `dCache` subclass if caching is required.

3.  **Iterative Generation Loop**: 
    Implement the core loop that drives the generation process. This typically involves:
    *   **Step Execution**: Calling `generate_step` (or a custom step function) to perform a forward pass and sample new tokens.
    *   **Delta Application**: The step function returns a `FrameDelta`, which contains the changes (new tokens, confidence scores) to be applied to the frame. Both `Frame` and `FrameDelta` may describe either a single sequence or a batch of sequences, so shape conventions should stay consistent with the current decoding mode.
    *   **State Update**: Apply the delta to the frame using `frame.apply_delta(delta)`.

4.  **Result Compilation**: 
    Collect all `FrameDelta` objects generated during the process and return them wrapped in a `DecodeRecord`.

### Example Structure

```python
@register("iterative_refinement")
def iterative_refinement_generate(model, input_ids, num_transfer_tokens=1, ...):
    # 1. Initialize
    frame = Frame.create_initial_frame(input_ids, ...)
    deltas = []
    
    # 2. Loop
    while True:
        # 3. Generate Step
        delta = generate_step(model, frame, ...)
        if delta is None:
            break
        
        # 4. Update
        frame = frame.apply_delta(delta)
        deltas.append(delta)
        
    # 5. Return
    return DecodeRecord(initial_frame=..., deltas=deltas, ...)
```

### Configuration
To use your custom strategy, create a new YAML configuration file in `configs/generation/` (e.g., `configs/generation/my_strategy.yaml`).
```yaml
name: my_strategy  # Matches the name used in @register
num_transfer_tokens: 1
# ... other parameters passed to the function
```

If your strategy requires complex validation or dynamic default values, provide a `gen_args_script` and implement the corresponding schema there. The default `configs/gen_args.py` can be used as a reference.

To use the new configuration:
```bash
python eval.py generation=my_strategy ...
```
