"""`Value<w>` and tensors: a dtype's width, the `Type` of a tensor or pytree, its maximal tensor chunks, and a collection of
references split into and joined from those chunks. Width only; numerical interpretation belongs to functions."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from verity.ml.circuits.refs import Coll, Refs, concat
from verity.ml.circuits.types import Array, Record, Tuple as TupleT, Type, Value
from verity_vllm.program.frontend.errors import FrontendError

WIDTH_OF_DTYPE = {torch.bool: 1, torch.uint8: 8, torch.int8: 8, torch.int16: 16, torch.uint16: 16, torch.bfloat16: 16,
                  torch.float16: 16, torch.int32: 32, torch.uint32: 32, torch.float32: 32, torch.int64: 64,
                  torch.uint64: 64, torch.float64: 64,
                  # the OCP FP8 words of a block-scaled checkpoint (e4m3 weights/activations, UE8M0 scale bytes): 8-bit
                  # VALUES, carried as uint8 (`CARRIER[8]`); their arithmetic is the registered Definitions' (registry.fp8, .marlin_moe)
                  torch.float8_e4m3fn: 8, torch.float8_e5m2: 8, torch.float8_e8m0fnu: 8}
CARRIER = {1: torch.bool, 8: torch.uint8, 16: torch.bfloat16, 32: torch.float32, 64: torch.int64}


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


def type_of(obj: Any, where: str | None = None) -> Type:
    """Tensor -> Array<..., Value<w>>; tuple -> Tuple; dict -> Record; list -> Array<n, T> (homogeneous)."""
    if isinstance(obj, torch.Tensor):
        return type_of_tensor(obj)
    if isinstance(obj, tuple):
        return TupleT(*[type_of(o, where) for o in obj])
    if isinstance(obj, dict):
        return Record(*[(k, type_of(v, where)) for k, v in obj.items()])
    if isinstance(obj, list):
        if not obj:
            raise FrontendError("E_SHAPE", "an empty list has no element type", where)
        ts = [type_of(o, where) for o in obj]
        if any(t != ts[0] for t in ts):
            raise FrontendError("E_SHAPE", f"list elements differ in type: {ts[0]!r} vs {[t for t in ts if t != ts[0]][0]!r}", where)
        return Array(len(ts), ts[0])
    raise FrontendError("E_UNREGISTERED_OP", f"a {type(obj).__name__} is not a value (values are tensors/tuples/dicts/lists)", where)


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
    """Maximal tensor chunks of a type, in leaf order (== pytree flatten order of `tensors_of(T)`)."""
    if is_tensor_type(T):
        return [T]
    if isinstance(T, TupleT):
        return [c for f in T.fields for c in flat_types(f)]
    if isinstance(T, Record):
        return [c for _, f in T.fields for c in flat_types(f)]
    assert isinstance(T, Array)
    return [c for _ in range(T.n) for c in flat_types(T.elem)]


def tensors_of(T: Type, device: str = "meta") -> Any:
    if is_tensor_type(T):
        shape, w = shape_of(T)
        return torch.empty(shape, dtype=CARRIER[w], device=device)
    if isinstance(T, TupleT):
        return tuple(tensors_of(f, device) for f in T.fields)
    if isinstance(T, Record):
        return {k: tensors_of(f, device) for k, f in T.fields}
    assert isinstance(T, Array)
    return [tensors_of(T.elem, device) for _ in range(T.n)]


def unflatten(T: Type, flat: list[torch.Tensor]) -> Any:
    it = iter(flat)

    def go(t: Type):
        if is_tensor_type(t):
            return next(it)
        if isinstance(t, TupleT):
            return tuple(go(f) for f in t.fields)
        if isinstance(t, Record):
            return {k: go(f) for k, f in t.fields}
        return [go(t.elem) for _ in range(t.n)]

    out = go(T)
    assert next(it, None) is None
    return out


def split_coll(c: Coll) -> list[Coll]:
    """Mirror of `flat_types` on a collection of references."""
    T = c.type
    if is_tensor_type(T):
        return [c]
    if isinstance(T, (TupleT, Record)):
        return [p for i in range(len(T.fields)) for p in split_coll(c[i])]
    return [p for i in range(T.n) for p in split_coll(c[i])]


def join_coll(T: Type, flats: Sequence[Coll]) -> Coll:
    """Assemble a collection of type `T` from its maximal tensor chunks, nesting the reference runs exactly as
    TA1's `tuple_of`/`record_of`/`array_of` do (a `concat` per structural level), so that a structured return
    such as `((nb, ni, nxt), ni)` encodes as `tuple_of(tuple_of(nb, ni, nxt), ni)` and digests identically."""
    it = iter(flats)

    def go(t: Type) -> Refs:
        if is_tensor_type(t):
            return next(it).refs
        if isinstance(t, TupleT):
            return concat([go(f) for f in t.fields])
        if isinstance(t, Record):
            return concat([go(f) for _, f in t.fields])
        assert isinstance(t, Array)
        return concat([go(t.elem) for _ in range(t.n)])

    refs = go(T)
    if next(it, None) is not None:
        raise FrontendError("E_SHAPE", f"more tensor chunks than {T!r} has leaves", "join")
    return Coll(T, refs)
