"""Static torch views as reference rearrangements (no gates): `view`, `slice`, `select`, `permute`, `expand` and `stack` over a
collection of references, kept O(1) in the descriptor where the references are one affine run."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from verity.ml.circuits.refs import Affine, Coll, Concat, Explicit, Ref, Refs, Strided, array_of, concat
from verity.ml.circuits.types import Array, Type, Value
from verity_vllm.program.frontend.errors import FrontendError
from verity_vllm.program.frontend.value_types import shape_of

_EXPLICIT_LIMIT = 1 << 16  # leaves an `Explicit` fallback may enumerate (permute of a non-affine base)
_VIEW_PARTS_LIMIT = 1 << 14  # `Concat` parts a multi-level affine view may expand to


def _dim_len(T: Type, dim: int) -> int:
    for _ in range(dim):
        T = T.elem
    return T.n


def _norm_dim(dim: int, T: Type) -> int:
    nd = len(shape_of(T)[0])
    return dim + nd if dim < 0 else dim


def _row_major(shape: Sequence[int]) -> list[int]:
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _tensor_type(shape: Sequence[int], w: int) -> Type:
    T: Type = Value(w)
    for n in reversed(tuple(shape)):
        T = Array(n, T)
    return T


# -- the reference builder (§8.5) --------------------------------------------------------------------------
# Every torch view is an affine index map `leaf(i_0..i_r) = offset + sum_k i_k * stride_k` over the leaves of
# its input. When the input's references are one `Affine` run (a parameter or a node output) the view's
# references are that map over the run: size-1 axes drop, row-major-contiguous axes merge, and the result is
# an `Affine` (one axis, stride 1), a `Strided` (<= two axes — `base + (j // inner) * outer_stride + (j % inner)
# * inner_stride`, zero strides denoting broadcasts) or a `Concat` over the leading axis of such pieces. This
# is how `expand`, inner-axis `select`/`slice`, `permute`/`transpose` and a `stack` of one repeated collection
# stay O(1) in the descriptor. `[z, z, z]` (one reference repeated) compacts to a zero-stride `Strided`;
# `[z0, z1, z2]` (three produced values) never does — the sequences differ leaf by leaf.


def _strided_refs(space: str, idx: int, base: int, shape: Sequence[int], strides: Sequence[int], where: str) -> Refs:
    count = math.prod(shape)
    if count == 0:
        return Explicit(())
    dims: list[tuple[int, int]] = []
    for n, st in zip(shape, strides):
        if n == 1:
            continue
        if dims and dims[-1][1] == n * st:  # outer axis is the contiguous continuation of this one (incl. 0/0)
            dims[-1] = (dims[-1][0] * n, st)
        else:
            dims.append((n, st))
    if not dims:
        return Affine(space, idx, base, 1)
    if len(dims) == 1:
        n, st = dims[0]
        return Affine(space, idx, base, n) if st == 1 else Strided(space, idx, base, 1, st, 0, n)
    if len(dims) == 2:
        (n0, s0), (n1, s1) = dims
        return Strided(space, idx, base, n1, s0, s1, n0 * n1)
    n0, s0 = dims[0]
    if n0 > _VIEW_PARTS_LIMIT:
        raise FrontendError("E_UNSUPPORTED_VIEW", f"a {len(dims)}-level view would need {n0} parts (limit {_VIEW_PARTS_LIMIT})", where)
    rest_shape, rest_strides = [n for n, _ in dims[1:]], [st for _, st in dims[1:]]
    return Concat([_strided_refs(space, idx, base + i * s0, rest_shape, rest_strides, where) for i in range(n0)])


def _constant_run(r: Refs) -> Ref | None:
    """The one reference a zero-stride run repeats, if `r` is such a run."""
    if isinstance(r, Affine) and r.count == 1:
        return r[0]
    if isinstance(r, Strided) and r.outer_stride == 0 and r.inner_stride == 0:
        return (r.space, r.idx, r.base)
    return None


def _repeat(ref: Ref, n: int) -> Refs:
    space, idx, leaf = ref
    return Affine(space, idx, leaf, 1) if n == 1 else Strided(space, idx, leaf, 1, 0, 0, n)


def _view(x: Coll, shape: Sequence[int], strides: Sequence[int], offset: int, where: str, what: str) -> Coll:
    """`x` viewed through the affine index map (as_strided semantics on x's own leaf order)."""
    w = shape_of(x.type)[1]
    out_t = _tensor_type(shape, w)
    r = x.refs
    if isinstance(r, Affine):
        return Coll(out_t, _strided_refs(r.space, r.idx, r.base + offset, shape, strides, where))
    c = _constant_run(r)
    if c is not None:  # a broadcast of one reference stays one reference
        return Coll(out_t, _repeat(c, math.prod(shape)))
    count = math.prod(shape)
    if count > _EXPLICIT_LIMIT:
        raise FrontendError("E_UNSUPPORTED_VIEW", f"{what} of {count} leaves over a composite reference sequence has no O(1) form "
                                                  f"(limit {_EXPLICIT_LIMIT}); apply the view to the producer's output or a parameter", where)
    items = []
    for j in range(count):
        rem, leaf = j, offset
        for n, st in zip(reversed(shape), reversed(strides)):
            rem, q = divmod(rem, n)
            leaf += q * st
        items.append(r[leaf])
    return Coll(out_t, Explicit(tuple(items)))


def _slice(x: Coll, dim: int, lo, hi, step: int, where: str) -> Coll:
    if step != 1:
        raise FrontendError("E_UNSUPPORTED_VIEW", "only unit-stride slices", where)
    n = _dim_len(x.type, dim)
    lo = 0 if lo is None else (lo + n if lo < 0 else min(lo, n))
    hi = n if hi is None else (hi + n if hi < 0 else min(hi, n))
    hi = max(hi, lo)
    if lo == 0 and hi == n:
        return x  # the whole axis: nothing to rearrange
    if dim == 0:
        return x[lo:hi]
    shape, _ = shape_of(x.type)
    if isinstance(x.refs, Affine) or _constant_run(x.refs) is not None:
        strides = _row_major(shape)
        new_shape = list(shape)
        new_shape[dim] = hi - lo
        return _view(x, new_shape, strides, lo * strides[dim], where, "slice")
    return array_of([_slice(x[i], dim - 1, lo, hi, 1, where) for i in range(x.type.n)])


def _select(x: Coll, dim: int, i: int, where: str) -> Coll:
    if dim == 0:
        return x[i]
    shape, _ = shape_of(x.type)
    if isinstance(x.refs, Affine) or _constant_run(x.refs) is not None:
        strides = _row_major(shape)
        return _view(x, [n for k, n in enumerate(shape) if k != dim], [st for k, st in enumerate(strides) if k != dim],
                     i * strides[dim], where, "select")
    return x.take(dim, i)


def _permute(x: Coll, dims: Sequence[int], where: str) -> Coll:
    shape, w = shape_of(x.type)
    dims = tuple(d + len(shape) if d < 0 else d for d in dims)
    if dims == tuple(range(len(shape))):
        return x
    strides, k = _row_major(shape), max(i + 1 for i, d in enumerate(dims) if d != i)  # axes k.. stay in place
    # whole rows of axes k.., reordered; past the Explicit limit only, so the Programs that derive today keep their bytes
    if (x.type.leaves > _EXPLICIT_LIMIT and k < len(shape) and (lead := math.prod(shape[:k])) <= _VIEW_PARTS_LIMIT
            and _constant_run(x.refs) is None and not isinstance(x.refs, Affine)):
        L, rows = x.type.leaves // lead, torch.arange(lead, device="cpu").reshape(shape[:k]).permute(dims[:k]).reshape(-1).tolist()
        return Coll(_tensor_type([shape[d] for d in dims], w), concat([x.refs.slice(r * L, (r + 1) * L) for r in rows]))
    return _view(x, [shape[d] for d in dims], [strides[d] for d in dims], 0, where, f"permute {dims}")


def _expand(x: Coll, sizes: Sequence[int], where: str) -> Coll:
    shape, w = shape_of(x.type)
    sizes = list(sizes)
    if len(sizes) < len(shape):
        raise FrontendError("E_SHAPE", f"expand to {sizes} has fewer axes than {shape}", where)
    lead = len(sizes) - len(shape)
    src = [1] * lead + list(shape)
    src_strides = [0] * lead + _row_major(shape)
    out_shape, out_strides = [], []
    for n, m, st in zip(src, sizes, src_strides):
        m = n if m == -1 else m
        if m == n:
            out_shape.append(n)
            out_strides.append(st)
        elif n == 1:
            out_shape.append(m)
            out_strides.append(0)  # a broadcast axis reads the same leaves: zero stride
        else:
            raise FrontendError("E_SHAPE", f"cannot expand axis of size {n} to {m}", where)
    if out_shape == list(shape):
        return x
    return _view(x, out_shape, out_strides, 0, where, "expand")


def _stack(parts: Sequence[Coll], where: str) -> Coll:
    """`stack` along the leading axis: `array_of` — unless every part is the *same* reference sequence over one
    run, which is a broadcast (zero outer stride), not a list of distinct values."""
    if not parts:
        raise FrontendError("E_SHAPE", "stack of nothing", where)
    first = parts[0]
    if len(parts) > 1 and all(p.type == first.type for p in parts):
        r0 = first.refs
        if all(p.refs is r0 or p.refs.to_json() == r0.to_json() for p in parts[1:]):
            n, L = len(parts), first.type.leaves
            c = _constant_run(r0)
            if c is not None:
                return Coll(Array(n, first.type), _repeat(c, n * L))
            if isinstance(r0, Affine):
                return Coll(Array(n, first.type), _strided_refs(r0.space, r0.idx, r0.base, (n, L), (0, 1), where))
    return array_of(list(parts))
