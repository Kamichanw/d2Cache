# Customization Guide

## 1. Implementing a Custom Cache Mechanism

To introduce a new caching mechanism, one must derive a new class from the abstract base class `dCache`. This architecture allows for precise control over the model's internal states during inference, enabling techniques such as KV-caching, attention masking, or state compression.

### Base Class Definition
The `dCache` class, located in `src/cache/base.py`, serves as the foundational interface. It utilizes Python context managers to intercept and modify the computation flow within the model's layers.

### Key Methods to Override

Subclasses should override the following methods to implement specific caching behaviors:

*   **`__init__(self, model_config, ...)`**: 
    Initializes the cache instance. This is where internal storage structures (e.g., lists for Key/Value tensors) should be instantiated.

*   **`model_forward(self, x: torch.Tensor)`**: 
    A context manager invoked wrapping the entire model forward pass. It allows for the modification of input embeddings or the preparation of global masks before layer-wise computations begin.

*   **`attention(self, layer_idx, ...)`**: 
    This is the most critical context manager for caching implementations. It intercepts the attention computation at each layer.
    *   **Input**: Receives the layer index, input tensor `x`, and projection layers (`q_proj`, `k_proj`, `v_proj`).
    *   **Operation**: The implementation should compute or retrieve the Query, Key, and Value states. It must yield an `AttentionContext` object.
    *   **Usage**: This method is typically used to store computed K/V pairs for future reuse or to modify the attention mask to enforce specific sparsity patterns.

*   **`ffn(self, layer_idx, x)`**: 
    A context manager for the Feed-Forward Network (FFN) layers, allowing for the inspection or modification of FFN inputs and outputs.

*   **Lifecycle Hooks**:
    *   `on_step_start(self, block_mask, frame)`: Executed at the beginning of each generation step. Useful for preparing masks based on the current generation status.
    *   `on_step_end(self, block_mask, frame, delta)`: Executed after the generation step. This is where the cache should be updated with the newly generated information (e.g., updating confidence scores or density metrics).

### Configuration
To use your custom cache, create a new YAML configuration file in `configs/cache/` (e.g., `configs/cache/my_cache.yaml`).
```yaml
_target_: src.cache.MyCache
# ... specific parameters for __init__
```


## 2. Developing a New Decoding Strategy

The framework supports modular decoding strategies, allowing researchers to implement custom generation algorithms (e.g., autoregressive, non-autoregressive, or iterative refinement) without modifying the core model code.

### Registration Mechanism
New strategies are registered using the `@register.gen_strategy` decorator. This allows the strategy to be selected via the configuration file or command-line arguments using its registered name.


### Implementation Workflow
A decoding strategy is implemented as a function that accepts the model, input identifiers, and strategy-specific hyperparameters. It must return a `DecodeRecord` object containing the generation history. The workflow typically involves the following steps:

1.  **Frame Initialization**: 
    Create an initial `Frame` object. This object encapsulates the state of the generation, including the prompts and the current sequence of tokens (initially masked).
    ```python
    frame = Frame.create_initial_frame(input_ids, gen_length=..., mask_token_id=...)
    ```

2.  **Cache Initialization**: 
    Instantiate the appropriate `dCache` subclass if caching is required.

3.  **Iterative Generation Loop**: 
    Implement the core loop that drives the generation process. This typically involves:
    *   **Step Execution**: Calling `generate_step` (or a custom step function) to perform a forward pass and sample new tokens.
    *   **Delta Application**: The step function returns a `FrameDelta`, which contains the changes (new tokens, confidence scores) to be applied to the frame.
    *   **State Update**: Apply the delta to the frame using `frame.apply_delta(delta)`.

4.  **Result Compilation**: 
    Collect all `FrameDelta` objects generated during the process and return them wrapped in a `DecodeRecord`.

### Example Structure

```python
@register.gen_strategy("iterative_refinement")
def iterative_refinement_generate(model, input_ids, steps=5, ...):
    # 1. Initialize
    frame = Frame.create_initial_frame(input_ids, ...)
    deltas = []
    
    # 2. Loop
    for i in range(steps):
        # 3. Generate Step
        delta = generate_step(model, frame, ...)
        
        # 4. Update
        frame = frame.apply_delta(delta)
        deltas.append(delta)
        
    # 5. Return
    return DecodeRecord(initial_frame=..., deltas=deltas, ...)
```

### Configuration
To use your custom strategy, create a new YAML configuration file in `configs/generation/` (e.g., `configs/generation/my_strategy.yaml`).
```yaml
name: my_strategy  # Matches the name used in @register.gen_strategy
steps: 10
# ... other parameters passed to the function
```

If your strategy requires complex validation or dynamic default values, update `configs/gen_args.py`. The `GenerationArgs` Pydantic model defines the schema and validation rules.

To use the new configuration:
```bash
python eval.py generation=my_strategy ...
```

