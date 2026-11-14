"""`Value<w>` and tensors: a dtype's width, the `Type` of a tensor, a type's shape and its maximal tensor chunks, and
`Alloc`, a typed allocation no rule has written yet. Width only; numerical interpretation belongs to functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from verity.circuits.types import Array, RecordType, Type, Value
from verity.circuits.types import Tuple as TupleT

from verity_vllm.program.frontend.errors import FrontendError

WIDTH_OF_DTYPE = {torch.bool: 1, torch.uint8: 8, torch.int8: 8, torch.int16: 16, torch.uint16: 16, torch.bfloat16: 16,
                  torch.float16: 16, torch.int32: 32, torch.uint32: 32, torch.float32: 32, torch.int64: 64,
                  torch.uint64: 64, torch.float64: 64,
                  # the OCP FP8 words of a block-scaled checkpoint (e4m3 weights/activations, UE8M0 scale bytes): 8-bit
                  # VALUES; their arithmetic is the registered Definitions' (registry.fp8, .marlin_moe)
                  torch.float8_e4m3fn: 8, torch.float8_e5m2: 8, torch.float8_e8m0fnu: 8}


@dataclass(eq=False)
class Alloc:
    """An uninitialised allocation (`empty`, `empty_like`, `new_empty`): the write target of an in-place rule.
    Reading it before a rule writes it is refused."""
    type: Type
    node: Any


def width_of(dtype: torch.dtype, where: str | None = None) -> int:
    try:
        return WIDTH_OF_DTYPE[dtype]
    except KeyError:
        raise FrontendError("E_WIDTH", f"dtype {dtype} carries no Value<w> width", where) from None


def type_of_tensor(t: torch.Tensor) -> Type:
    T: Type = Value(width_of(t.dtype))
    for n in reversed(tuple(t.shape)):
        T = Array(int(n), T)
    return T


def is_tensor_type(T: Type) -> bool:
    return isinstance(T, Value) or (isinstance(T, Array) and is_tensor_type(T.elem))


def shape_of(T: Type) -> tuple[tuple[int, ...], int]:
    shape: list[int] = []
    while isinstance(T, Array):
        shape.append(T.n)
        T = T.elem
    assert isinstance(T, Value)
    return tuple(shape), T.w


def flat_types(T: Type) -> list[Type]:
    """Maximal tensor chunks of a type, in leaf order."""
    if is_tensor_type(T):
        return [T]
    if isinstance(T, TupleT):
        return [c for f in T.fields for c in flat_types(f)]
    if isinstance(T, RecordType):
        return [c for _, f in T.fields for c in flat_types(f)]
    assert isinstance(T, Array)
    return [c for _ in range(T.n) for c in flat_types(T.elem)]


__all__ = ["WIDTH_OF_DTYPE", "Alloc", "width_of", "type_of_tensor", "is_tensor_type", "shape_of", "flat_types"]
