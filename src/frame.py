import os
import torch

from typing import overload
from pydantic import BaseModel, Field, model_validator
from collections.abc import Sequence
from functools import cached_property

from .utils import apply_fn, tensor_insert, tensor_delete

PLACEHOLDER_STEP = (
    -1
)  # Represents a placeholder for a token that has not been decoded yet.
INVALID_TOKEN_ID = -100  # Represents an invalid token ID, used for ignored tokens.


class Base(BaseModel):

    @property
    def is_batched(self) -> bool:
        raise NotImplementedError(
            "The derived class must implement this proeprty to check whether the current object is a batch."
        )

    def to(
        self, device: str | torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """
        Converts the model to a specific device and dtype. The `dtype` is only used for floating point tensors.
        """
        if isinstance(device, torch.dtype):
            dtype, device = device, dtype  # type: ignore

        def to(t):
            if isinstance(t, torch.Tensor) or hasattr(t, "to"):
                return t.to(
                    device=device, dtype=dtype if t.dtype.is_floating_point else t.dtype
                )
            return t

        return type(self).model_validate(apply_fn(self.model_dump(), to))

    def clone(self):
        def clone(t):
            if isinstance(t, torch.Tensor) or hasattr(t, "clone"):
                return t.clone()
            return t

        return type(self).model_validate(apply_fn(self.model_dump(), clone))

    def as_batch(self):
        """
        Converts the model to a batch format, if it only contains a single sequence.
        Otherwise, it returns the model as is.
        """

        if self.is_batched:
            return self

        def as_batch(t):
            if hasattr(t, "as_batch"):
                return t.as_batch()
            elif isinstance(t, torch.Tensor) and t.dim() == 1:
                return t.unsqueeze(0)
            return t

        return type(self).model_validate(apply_fn(self.model_dump(), as_batch))

    def unbatch(self):
        """
        Converts the delta to a single sequence format, if it is a batch that contains only one sequence.
        Otherwise, it returns the delta as is.
        """
        if self.is_batched:
            return self[0]
        return self

    def __getitem__(self, index):
        return type(self).model_validate(
            apply_fn(
                self.model_dump(),
                lambda t: t[index] if hasattr(t, "__getitem__") else t,
            )
        )


class Intermediate(Base):
    class Config:
        arbitrary_types_allowed = True

    # ((layer_idx, values: (batch_size, seq_len, hidden_dim)), ...)
    hidden_states: tuple[tuple[int, torch.Tensor], ...] = Field(
        default_factory=tuple, description="The hidden states of tokens from layers."
    )
    key_states: tuple[tuple[int, torch.Tensor], ...] = Field(
        default_factory=tuple,
        description="The key states of tokens after `w_k` projection.",
    )
    value_states: tuple[tuple[int, torch.Tensor], ...] = Field(
        default_factory=tuple,
        description="The value states of tokens after `w_v` projection.",
    )
    query_states: tuple[tuple[int, torch.Tensor], ...] = Field(
        default_factory=tuple,
        description="The query states of tokens after `w_q` projection.",
    )
    attention_out: tuple[tuple[int, torch.Tensor], ...] = Field(
        default_factory=tuple,
        description="The attention outputs of tokens after `w_o` projection and before adding residual.",
    )

    @property
    def is_batched(self) -> bool:
        def is_batch(v):
            return v[0][1].dim() == 3 if v else False

        return any(is_batch(t) for t in self.model_dump().values())

    @model_validator(mode="after")
    def check_member_shapes(self) -> "Intermediate":
        dummy_tensor = next((t[0][1] for t in self.model_dump().values() if t), None)
        if dummy_tensor is not None and not all(
            t[0][1].dim() == dummy_tensor.dim() for t in self.model_dump().values() if t
        ):
            raise ValueError(
                "All tensors in the intermediate states must have the same number of dimensions."
            )
        return self


class FrameDelta(Base):
    """
    A delta of a Frame object, tracking the changes made to the first frame during the decoding process.
    If any sequence has finished, the corresponding `transfer_index` will be an empty tensor, and other
    tensors will be shape of `[batch_size - num_finished_sequences, ...]`.
    """

    class Config:
        arbitrary_types_allowed = True

    transfer_index: torch.Tensor | tuple[torch.Tensor, ...] = Field(
        description="The indices of the transferred tokens within the decoded_tokens.",
    )
    transfer_src_index: torch.Tensor | tuple[torch.Tensor, ...] | None = Field(
        default=None,
        description=(
            "The indices of the source tokens within the decoded_tokens."
            " If it is None, transfer tokens at indices specified by `transfer_index`."
        ),
    )
    insert_index: torch.Tensor | None = Field(
        default=None,
        description="The target indices to insert in the transferred generated tokens.",
    )
    insert_src_index: torch.Tensor | None = Field(
        default=None,
        description=(
            "The source indices of inserted tokens within the decoded_tokens."
            " It can't be None if `insert_index` is specified."
        ),
    )
    delete_index: torch.Tensor | None = Field(
        default=None,
        description="The indices of deleted tokens in the inserted decoded_tokens",
    )
    decoded_tokens: torch.Tensor = Field(
        description=(
            "The decoded tokens in each sequence with shape [batch_size, gen_length]. It can be expanded to "
            "a large size compared to previous step, but deletion should only be specified by `delete_index`."
        ),
    )
    confidence: torch.Tensor | None = Field(
        default=None, description="The confidence scores for all decoded tokens."
    )
    probs: torch.Tensor | None = Field(
        default=None,
        description="The probability distributions of the decoded tokens over vocabulary.",
    )
    intermediate: Intermediate = Field(
        default_factory=Intermediate,
        description="The intermediate states (hidden/key/value/query) of tokens from layers.",
    )
    extra: dict = Field(
        default_factory=dict,
        description="A dict containing any additional variables you want to add to the delta.",
    )

    @model_validator(mode="after")
    def check_shapes(self) -> "FrameDelta":
        if self.insert_index is not None and self.insert_src_index is None:
            raise ValueError(
                "insert_src_index is required when insert_index is provided."
            )
        if (
            self.insert_index is not None
            and self.insert_src_index is not None
            and self.insert_index.shape != self.insert_src_index.shape
        ):
            raise ValueError(
                "insert_index and insert_src_index must have the same shape."
            )

        if self.is_batched:
            batch_size = len(self.transfer_index)
            num_finished_sequences = sum(t.numel() == 0 for t in self.transfer_index)
            active_batch_size = batch_size - num_finished_sequences

            if self.decoded_tokens.size(0) != active_batch_size:
                raise ValueError(
                    f"decoded_tokens batch size ({self.decoded_tokens.size(0)}) must match active sequence count ({active_batch_size})."
                )

            index_fields = {
                "insert_index": self.insert_index,
                "insert_src_index": self.insert_src_index,
                "delete_index": self.delete_index,
            }
            for name, field in index_fields.items():
                if field is not None:
                    if field.dim() != 2:
                        raise ValueError(
                            f"'{name}' must be a 2D tensor for a batched delta."
                        )
                    if field.size(0) != active_batch_size:
                        raise ValueError(
                            f"The first dimension of '{name}' ({field.size(0)}) must match active sequence count ({active_batch_size})."
                        )
        else:
            if self.insert_index is not None and self.insert_index.dim() != 1:
                raise ValueError("insert_index must be 1D for a non-batched delta.")
            if self.delete_index is not None and self.delete_index.dim() != 1:
                raise ValueError("delete_index must be 1D for a non-batched delta.")
        return self

    @property
    def is_batched(self) -> bool:
        return isinstance(self.transfer_index, tuple)

    @property
    def transferred_tokens(self) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Fetch the new transferred tokens, which may contains empty tensors, based on the transfer_index.
        If transfer_index is a tuple, it returns a tuple of size [batch_size,] containing tensors.
        Otherwise, it returns a single tensor.
        """
        transfer_src_index = (
            self.transfer_src_index
            if self.transfer_src_index is not None
            else self.transfer_index
        )
        if self.is_batched:
            decoded_tokens_iter = iter(self.decoded_tokens)
            return tuple(
                (
                    next(decoded_tokens_iter)[index]
                    if index.numel() > 0
                    else torch.tensor([], dtype=torch.long, device=index.device)
                )
                for index in transfer_src_index
            )
        return self.decoded_tokens[transfer_src_index]

    @property
    def inserted_tokens(self) -> torch.Tensor:
        """
        Fetch the new inserted tokens, if both `insert_index` and `insert_src_index` is not None.
        Note that it always returns a tensor gathered from the decoded tokens.
        """
        if self.insert_index is None or self.insert_src_index is None:
            raise RuntimeError(
                "Insert index and insert source index must be specified to get inserted tokens."
            )

        if self.is_batched:
            return torch.gather(self.decoded_tokens, 1, self.insert_src_index)
        return self.decoded_tokens[self.insert_src_index]

    def __getitem__(self, index: int | slice | tuple) -> "FrameDelta":
        """
        Allows indexing into the Frame object to access its attributes or a sequence.
        """
        if not self.is_batched:
            return super().__getitem__(index)

        is_active_mask = torch.tensor(
            [t.numel() > 0 for t in self.transfer_index], dtype=torch.bool
        )
        full_to_active_map = (torch.cumsum(is_active_mask, 0) - 1).tolist()

        if isinstance(index, int):
            # Handle integer indexing: un-batch the element.
            if index < 0:
                index += len(self.transfer_index)

            transfer_index = self.transfer_index[index]
            transfer_src_index = (
                self.transfer_src_index[index] if self.transfer_src_index else None
            )

            if is_active_mask[index]:
                active_idx = full_to_active_map[index]
                decoded_tokens = self.decoded_tokens[active_idx]
                confidence = (
                    self.confidence[active_idx] if self.confidence is not None else None
                )
                probs = self.probs[active_idx] if self.probs is not None else None
                insert_index = (
                    self.insert_index[active_idx]
                    if self.insert_index is not None
                    else None
                )
                insert_src_index = (
                    self.insert_src_index[active_idx]
                    if self.insert_src_index is not None
                    else None
                )
                delete_index = (
                    self.delete_index[active_idx]
                    if self.delete_index is not None
                    else None
                )
                intermediate = self.intermediate[active_idx]
            else:  # The indexed sequence is finished/inactive
                decoded_tokens = torch.tensor(
                    [],
                    dtype=self.decoded_tokens.dtype,
                    device=self.decoded_tokens.device,
                )
                confidence, probs, insert_index, insert_src_index, delete_index = (
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                intermediate = Intermediate()

        elif isinstance(index, slice):
            # Handle slice indexing: return a new batched delta.
            transfer_index = self.transfer_index[index]
            transfer_src_index = (
                self.transfer_src_index[index] if self.transfer_src_index else None
            )

            slice_indices = range(*index.indices(len(self.transfer_index)))
            active_indices_to_take = [
                full_to_active_map[i] for i in slice_indices if is_active_mask[i]
            ]

            def slice_active_tensor(tensor):
                if tensor is None:
                    return None
                if not active_indices_to_take:
                    return torch.empty(
                        0, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device
                    )
                return tensor[active_indices_to_take]

            decoded_tokens = slice_active_tensor(self.decoded_tokens)
            confidence = slice_active_tensor(self.confidence)
            probs = slice_active_tensor(self.probs)
            insert_index = slice_active_tensor(self.insert_index)
            insert_src_index = slice_active_tensor(self.insert_src_index)
            delete_index = slice_active_tensor(self.delete_index)
            intermediate = (
                self.intermediate[active_indices_to_take]
                if active_indices_to_take
                else Intermediate()
            )

        elif isinstance(index, tuple):
            batch_idx, *rest = index
            return self[batch_idx][tuple(rest)]

        else:
            raise TypeError(
                f"Unsupported index type for batched FrameDelta: {type(index)}"
            )

        return FrameDelta(
            transfer_index=transfer_index,
            transfer_src_index=transfer_src_index,
            decoded_tokens=decoded_tokens,  # type: ignore
            confidence=confidence,
            probs=probs,
            insert_index=insert_index,
            insert_src_index=insert_src_index,
            delete_index=delete_index,
            intermediate=intermediate,
            extra=self.extra,
        )


class Frame(Base):
    """
    A specific decoding step of LLaDA. It tracks the prompt, generated tokens, confidence scores, and steps when each token was decoded.
    The shape of all tensors can be (batch_size, sequence_length) or (sequence_length,). Different sequences can generate different numbers of tokens
    in one step.
    """

    class Config:
        arbitrary_types_allowed = True

    prompts: torch.Tensor = Field(frozen=True, description="The input prompt tensor.")
    generated_tokens: torch.Tensor = Field(
        frozen=True,
        description="The generated tokens, including mask tokens that have not been decoded yet.",
    )
    confidence: torch.Tensor | None = Field(
        default=None, description="The confidence scores for all generated tokens."
    )
    steps: torch.Tensor = Field(
        frozen=True,
        description=f"The token at the corresponding position is decoded at which step. For mask token, step is {PLACEHOLDER_STEP}.",
    )

    @model_validator(mode="after")
    def check_member_shapes(self) -> "Frame":
        dims = set(
            t.dim() for t in self.model_dump().values() if isinstance(t, torch.Tensor)
        )
        shapes = set(
            t.shape
            for t in self.model_dump().values()
            if isinstance(t, torch.Tensor) and t is not self.prompts
            # exclude prompts, as it can have different shapes
        )
        if len(dims) > 1:
            raise ValueError(
                "All tensors in the frame must have the same number of dimensions."
            )
        if len(shapes) > 1:
            raise ValueError(
                "The shapes of generated_tokens, confidence, and steps must match."
            )

        return self

    @classmethod
    def create_initial_frame(
        cls, prompts: torch.Tensor, gen_length: int, mask_token_id: int | None = None
    ) -> "Frame":
        try:
            mask_token_id = mask_token_id or int(os.environ["MASK_TOKEN_ID"])
        except KeyError:
            raise ValueError(
                "mask_token_id must be provided either as an argument or an environment variable."
            )
        is_batched = prompts.dim() > 1
        frame = cls(
            prompts=prompts.squeeze(0) if not is_batched else prompts,
            generated_tokens=torch.full(
                (prompts.size(0), gen_length),
                mask_token_id,
                dtype=torch.long,
                device=prompts.device,
            ),
            confidence=torch.full(
                (prompts.size(0), gen_length),
                -torch.inf,
                dtype=torch.float32,
                device=prompts.device,
            ),
            steps=torch.full(
                (prompts.size(0), gen_length),
                PLACEHOLDER_STEP,
                dtype=torch.long,
                device=prompts.device,
            ),
        )

        return frame.unbatch() if not is_batched else frame

    def apply_delta(
        self, delta: FrameDelta, mask_token_id: int | None = None
    ) -> "Frame":
        """
        Applies a delta to the current frame and returns a new frame.
        """

        delta = delta.as_batch()
        batch_size, device = len(delta.transfer_index), delta.decoded_tokens.device
        try:
            mask_token_id = mask_token_id or int(os.environ["MASK_TOKEN_ID"])
        except KeyError:
            raise ValueError(
                "mask_token_id must be provided either as an argument or an environment variable."
            )

        # 1. apply transferring
        new_frame = self.clone().as_batch().to(device=device)

        lengths = torch.tensor([t.numel() for t in delta.transfer_index], device=device)
        # consider transfer_index = [[10, 12], [5], [8, 9, 11]],
        # then lengths = [[2], [1], [3]], the base row indices are [0, 1, 2].
        # we finally get row_indices = [0, 0, 1, 2, 2, 2], 0 repeats 2 times, 1 repeats 1 time, etc.
        row_indices = torch.repeat_interleave(
            torch.arange(batch_size, device=device), repeats=lengths
        )
        col_indices = torch.cat(delta.transfer_index)  # type: ignore
        # similarly, we get new steps for each row index.
        next_steps = torch.repeat_interleave(new_frame.current_steps, repeats=lengths)  # type: ignore

        # overwrite steps and generated_tokens specified by transfer_index
        new_frame.steps[row_indices, col_indices] = next_steps + 1
        new_frame.generated_tokens[row_indices, col_indices] = torch.cat(delta.transferred_tokens)  # type: ignore

        if delta.confidence is not None and new_frame.confidence is not None:
            active_row_indices = torch.repeat_interleave(
                torch.arange(delta.decoded_tokens.size(0), device=device),
                repeats=lengths[lengths > 0],
            )
            transfer_src_index = (
                delta.transfer_src_index
                if delta.transfer_src_index is not None
                else delta.transfer_index
            )
            active_src_indices = torch.cat(
                [t for t in transfer_src_index if t.numel() > 0]
            )
            new_frame.confidence[row_indices, col_indices] = delta.confidence[
                active_row_indices, active_src_indices
            ]

        # 2. perform insertion
        active_mask = torch.tensor(
            [t.numel() > 0 for t in delta.transfer_index], device=device
        )
        if delta.insert_index is not None or delta.insert_src_index is not None:
            if (
                delta.insert_index is None
                or delta.insert_src_index is None
                or delta.insert_index.shape != delta.insert_src_index.shape  # type: ignore
            ):
                raise RuntimeError(
                    "The insert index and insert source index must both be specified and have the same shape."
                )
            B, K = delta.insert_index.shape

            # upsample delta data to full batch size
            expand_active_mask = active_mask.unsqueeze(1).expand(-1, K)
            full_insert_index = torch.zeros(
                B, K, dtype=torch.long, device=device
            ).masked_scatter_(expand_active_mask, delta.insert_index)
            full_insert_tokens = torch.full(
                (B, K), mask_token_id, dtype=torch.long, device=device
            ).masked_scatter_(expand_active_mask, delta.inserted_tokens)
            full_insert_steps = torch.full(
                (B, K), PLACEHOLDER_STEP, dtype=torch.long, device=device
            ).masked_scatter_(
                expand_active_mask,
                (self.current_steps[active_mask, None] + 1).expand(-1, K),  # type: ignore
            )
            full_insert_conf = None
            if delta.confidence is not None:
                full_insert_conf = torch.zeros(
                    B, K, dtype=delta.confidence.dtype, device=device
                ).masked_scatter_(
                    expand_active_mask,
                    torch.gather(delta.confidence, 1, delta.insert_src_index),
                )

            new_frame = Frame(
                prompts=new_frame.prompts,
                generated_tokens=tensor_insert(
                    new_frame.generated_tokens,
                    full_insert_index,
                    full_insert_tokens,
                ),
                steps=tensor_insert(
                    new_frame.steps,
                    full_insert_index,
                    full_insert_steps,
                ),
                confidence=(
                    tensor_insert(
                        new_frame.confidence,
                        full_insert_index,
                        full_insert_conf,  # type: ignore
                    )
                    if new_frame.confidence is not None
                    else None
                ),
            )

        # 3. perform deletion
        if delta.delete_index is not None:
            new_frame = Frame(
                prompts=new_frame.prompts,
                generated_tokens=tensor_delete(
                    new_frame.generated_tokens, delta.delete_index
                ),
                steps=tensor_delete(new_frame.steps, delta.delete_index),
                confidence=(
                    tensor_delete(new_frame.confidence, delta.delete_index)
                    if new_frame.confidence is not None
                    else None
                ),
            )

        return new_frame.unbatch() if not self.is_batched else new_frame

    @property
    def is_batched(self) -> bool:
        return self.prompts.dim() > 1

    @cached_property
    def current_steps(self) -> int | torch.Tensor:
        """
        Returns the current step of the frame. Returns -1 if it is the initial frame.
        If this frame has multiple sequences, returns a tensor with shape (batch_size,).
        Otherwise, returns an integer.
        """
        max_steps = self.steps.max(dim=-1).values
        if self.prompts.dim() == 1:
            return int(torch.unique(max_steps).item())
        return max_steps

    def __getitem__(self, index: int | slice | tuple) -> "Frame":
        """
        Allows indexing into the Frame object to access its attributes or a sequence.
        """
        return Frame(
            prompts=self.prompts[index],
            generated_tokens=self.generated_tokens[index],
            confidence=(
                self.confidence[index] if self.confidence is not None else None
            ),
            steps=self.steps[index],
        )


class DecodeRecord(Base, Sequence):
    """
    A record of the decoding process, containing a list of frames. To get a specific frame, use the index.
    The first frame is the initial frame, and subsequent frames are generated by applying deltas sequentially.
    """

    class Config:
        arbitrary_types_allowed = True

    initial_frame: Frame = Field(
        ..., description="The initial frame of the decoding process."
    )
    deltas: list[FrameDelta] = Field(
        default_factory=list,
        description="A list of deltas applied to the initial frame.",
    )
    block_length: int | None = Field(
        default=None,
        description="Less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.",
    )

    def append(self, delta: FrameDelta) -> None:
        """
        Appends a new delta to the record.
        The new delta's step must be adjacent to the last delta's step.
        """
        self.deltas.append(delta)

    @property
    def frames(self) -> list[Frame]:
        """
        Returns a list of frames in the record, including the initial frame and all frames generated by applying deltas.
        """
        frames = [self.initial_frame]
        for delta in self.deltas:
            frames.append(frames[-1].apply_delta(delta))
        return frames

    @property
    def num_steps(self) -> int:
        """
        Returns the number of steps in generation.
        """
        return len(self.deltas)

    @property
    def gen_length(self) -> int:
        """
        Returns the length of the generated sequence.
        """
        return self.initial_frame.generated_tokens.size(-1)

    def __len__(self) -> int:
        return 1 + self.num_steps  # +1 for the initial frame

    @overload
    def __getitem__(self, index: int) -> Frame: ...

    @overload
    def __getitem__(self, index: slice | tuple) -> list[Frame]: ...

    def __getitem__(self, index):
        """
        Returns the frame at the specified index.
        The first frame is the initial frame, and subsequent frames are generated by applying deltas.
        """
        if isinstance(index, int):
            if index < 0:
                index += len(self)
            frame = self.initial_frame
            for delta in self.deltas[:index]:
                frame = frame.apply_delta(delta)
            return frame
        elif isinstance(index, slice):
            frames = []
            for i in range(*index.indices(len(self))):
                frames.append(self[i])
            return frames
        elif isinstance(index, tuple):
            batch_idx, *rest = index
            frames = self[batch_idx]
            if isinstance(frames, list):
                return [frame[tuple(rest)] for frame in frames]
            else:
                return frames[tuple(rest)]
        else:
            raise TypeError(
                f"Indexing must be done with an integer or a slice, but got {type(index)}."
            )

    def __repr__(self) -> str:
        return f"DecodeRecord(gen_length={self.gen_length}, block_length={self.block_length}, len={len(self)})"