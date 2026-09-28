"""[ov-sampling] SHARED-CODE tests for fold.py: (1) n>1 child request ids (`{index}_{parent}`, header `workload.children`)
bind to the parent's prompt parameter and the served sequences are per child; (2) the sampled-row fallback matches the
step's instance by its output element (family-agnostic), not only `TokenSelect_v*` by row."""

from __future__ import annotations

import dataclasses
import types

import pytest

from verity_vllm.observe.fold import fold as fold_mod
from verity_vllm.observe.events import RunHeader, StepLink, StepStart, ViewDesc
from verity_vllm.observe.fold.fold import FoldError, fold
from tests.observe.test_fold_synthetic import REQ, toy_log, toy_profile  # noqa: F401  (memory fixture lives there)
from tests.observe.test_fold_synthetic import memory  # noqa: F401
from verity.ml.circuits.refs import Affine
from verity.ml.circuits.types import Array, Value


def _rename(x, old: str, new: str):
    if isinstance(x, str):
        return new if x == old else x
    if isinstance(x, tuple):
        return tuple(_rename(v, old, new) for v in x)
    if isinstance(x, list):
        return [_rename(v, old, new) for v in x]
    if isinstance(x, dict):
        return {_rename(k, old, new): _rename(v, old, new) for k, v in x.items()}
    return x


def n2_log(child_of_unknown_parent: bool = False):
    """The toy log with the runner rows carrying vLLM child ids: step 0 and 1 belong to child `0_r0`; `1_r0` produced nothing."""
    evs = toy_log()
    hdr = evs[0]
    children = [{"request_id": "0_r0", "parent": REQ, "index": 0, "seed": 7}, {"request_id": "1_r0", "parent": REQ, "index": 1, "seed": 8}]
    if child_of_unknown_parent:
        children.append({"request_id": "0_zz", "parent": "zz", "index": 0, "seed": 9})
    wl = dict(hdr.workload)
    wl["requests"] = [dict(wl["requests"][0], n=2)]
    wl["children"] = children
    evs[0] = dataclasses.replace(hdr, workload=wl)
    out = []
    for e in evs:
        if isinstance(e, (StepStart, StepLink)):
            e = type(e)(*[_rename(getattr(e, f.name), REQ, "0_r0") for f in dataclasses.fields(e)])
        out.append(e)
    return out


def test_children_bind_to_the_parent_prompt_and_return_per_child(memory):
    with pytest.raises(FoldError, match="'1_r0' produced no sampled tokens"):      # strict: a silent child is an error
        fold(n2_log(), toy_profile(), strict=True)
    r = fold(n2_log(), toy_profile(), strict=False)
    assert not r.errors and r.program is not None
    assert [n for n, _ in r.params] == ["prompt", "weights"]          # ONE prompt root for the parent, not one per child
    assert r.params[0][1] == Array(2, Value(32))
    assert r.sampled == {"0_r0": [(0, 9), (1, 3)]}
    assert repr(r.body.ret_type) == repr((Array(2, Value(32)), Array(0, Value(32))))  # outputs per child, in child order
    assert any("'1_r0' produced no sampled tokens" in n for n in r.notes)


def test_child_of_unknown_parent_is_a_header_error(memory):
    with pytest.raises(FoldError, match="unknown parent 'zz'"):
        fold(n2_log(child_of_unknown_parent=True), toy_profile(), strict=True)


def test_plain_requests_are_unchanged(memory):
    r = fold(toy_log(), toy_profile(), strict=True)
    assert r.body.ret_type == Array(2, Value(32)) and r.sampled == {REQ: [(0, 9), (1, 3)]}


# --- (2) family-agnostic sampled-row fallback --------------------------------------------------------------------------


def _inst(step, node, spec, row, out_view):
    return {"step": step, "node": node, "spec": spec, "row": row, "output_views": [out_view]}


def test_fallback_matches_the_instance_by_output_element():
    sampled = ViewDesc(900, 0, (2,), (1,), "int64", "cuda:0")
    el0, el1 = (ViewDesc(900, i, (), (), "int64", "cuda:0") for i in (0, 1))
    ins = [_inst(3, 40, "GumbelArgmaxBf16_v1{V=16,SEED=7,POS=5}", 0, el0), _inst(3, 41, "GumbelArgmaxBf16_v1{V=16,SEED=8,POS=5}", 1, el1)]
    f = types.SimpleNamespace(instances=ins)
    ts = fold_mod._Fold._token_select_output
    assert repr(ts(f, 3, 0, el0)) == repr(Affine("n", 40, 0, 1)) and repr(ts(f, 3, 1, el1)) == repr(Affine("n", 41, 0, 1))
    assert ts(f, 3, 0, None) is None                                          # not a TokenSelect: no by-row guess
    assert ts(f, 3, 0, ViewDesc(901, 0, (), (), "int64", "cuda:0")) is None    # another allocation
    assert ts(f, 2, 0, el0) is None                                           # another step
    # the registered greedy recognizer's TokenSelect is still found by row when the view is unavailable
    f2 = types.SimpleNamespace(instances=[_inst(3, 50, "TokenSelect_v1{V=16}", 1, None)])
    assert repr(ts(f2, 3, 1, None)) == repr(Affine("n", 50, 0, 1)) and ts(f2, 3, 0, None) is None
    assert sampled.numel == 2
