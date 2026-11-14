"""`strided_views`: a static torch view is a rearrangement of references, O(1) in the descriptor where the references are one
affine run, and refused rather than enumerated past a limit. Needs torch; skips cleanly without it."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
from verity.circuits.refs import (
    Affine,
    Coll,
    Concat,
    Explicit,
    Strided,
    array_of,
)
from verity.circuits.types import Value

from verity_vllm.program.frontend import strided_views as sv
from verity_vllm.program.frontend.errors import FrontendError

V16 = Value(16)
WHERE = "test"


def _param(idx: int, *shape: int) -> Coll:
    """An input of `shape`: one affine run of references."""
    t = sv._tensor_type(shape, 16)
    return Coll(t, Affine("in", idx, 0, t.leaves))


def _value(idx: int) -> Coll:
    """One produced value: a single reference."""
    return Coll(V16, Affine("node", idx, 0, 1))


def _leaves(c: Coll) -> list[int]:
    return [c.refs[j][2] for j in range(c.type.leaves)]


def _swapped_rows(x: Coll) -> Coll:
    """`cat([x[1:2], x[0:1]])`: a composite reference sequence, with no single affine run underneath."""
    return array_of([x[1], x[0]])


def test_a_broadcast_is_a_zero_stride_run_and_distinct_values_are_not_merged():
    z = _value(0)
    rep = sv._expand(z, [3], WHERE).refs  # one reference broadcast
    assert isinstance(rep, Strided) and rep.outer_stride == 0 and rep.inner_stride == 0 and rep.count == 3
    assert [rep[j] for j in range(3)] == [rep[0]] * 3
    assert sv._stack([z, z, z], WHERE).refs.to_json() == rep.to_json()  # a stack of one repeated reference is the broadcast
    three = sv._stack([_value(k) for k in range(3)], WHERE).refs  # three produced values
    assert not isinstance(three, Strided) and len({three[j] for j in range(3)}) == 3  # never merged


def test_a_view_of_a_broadcast_is_the_same_one_reference():
    rep = sv._expand(_value(0), [2, 3], WHERE)  # one reference, six times
    views = ((sv._select(rep, 1, 0, WHERE), 2), (sv._slice(rep, 1, 1, 3, 1, WHERE), 4), (sv._permute(rep, (1, 0), WHERE), 6))
    for view, count in views:
        r = view.refs
        assert isinstance(r, Strided) and r.outer_stride == 0 and r.inner_stride == 0 and r.count == count
        assert [r[k] for k in range(count)] == [rep.refs[0]] * count


def test_an_inner_axis_select_is_a_strided_view():
    kc, v = _param(0, 7, 3, 8), _param(1, 8, 8)
    head = sv._select(kc, 1, 1, WHERE)  # [7][8]: one KV head out of [T][KVH][D]
    h0, h6 = head[0], head[6]
    assert not isinstance(h0.refs, (Explicit, Concat)) and _leaves(h0) == [8 + j for j in range(8)]
    assert not isinstance(h6.refs, (Explicit, Concat)) and _leaves(h6) == [6 * 24 + 8 + j for j in range(8)]
    col = sv._select(v, 1, 0, WHERE)  # [8]: column 0 of v, as a view of the input
    assert isinstance(col.refs, Strided) and _leaves(col) == [j * 8 for j in range(8)]


def test_a_permute_of_a_composite_sequence_is_explicit_under_a_limit_and_refused_past_it(monkeypatch):
    y = _swapped_rows(_param(0, 2, 3, 4))
    ret = sv._permute(y, (2, 1, 0), WHERE).refs
    assert isinstance(ret, Explicit) and ret.count == 24
    monkeypatch.setattr(sv, "_EXPLICIT_LIMIT", 8)
    with pytest.raises(FrontendError) as ei:
        sv._permute(y, (2, 1, 0), WHERE)
    assert ei.value.code == "E_UNSUPPORTED_VIEW"


def test_a_permute_of_the_leading_axes_past_the_explicit_limit_is_a_concat_of_rows(monkeypatch):
    """DeepSeek-V4's `fused_inv_rope_fp8_quant` output [G, T, d] read as [T, G, d]: past the Explicit limit, a permute that keeps the
    trailing axes in place reorders whole rows (one part each) where the leaf-by-leaf fallback refuses; under the Build's default
    meta device."""
    monkeypatch.setattr(sv, "_EXPLICIT_LIMIT", 8)
    with torch.device("meta"):
        ret = sv._permute(_swapped_rows(_param(0, 2, 3, 4)), (1, 0, 2), WHERE)
    assert isinstance(ret.refs, Concat) and len(ret.refs.parts) == 6
    assert _leaves(ret) == [(1 - a) * 12 + b * 4 + c for b in range(3) for a in range(2) for c in range(4)]


def test_a_three_level_view_is_a_concat_of_strided_pieces():
    ret = sv._permute(_param(0, 2, 3, 4), (2, 1, 0), WHERE)  # out[k][j][i] = x[i][j][k]
    assert isinstance(ret.refs, Concat) and len(ret.refs.parts) == 4 and all(isinstance(p, Strided) for p in ret.refs.parts)
    assert _leaves(ret) == [k + 4 * j + 12 * i for k in range(4) for j in range(3) for i in range(2)]
