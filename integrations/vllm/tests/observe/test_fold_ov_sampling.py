"""Tests for fold.py: (1) n>1 child request ids (`{index}_{parent}`, header `workload.children`)
bind to the parent's prompt parameter and the served sequences are per child; (2) the sampled-row fallback matches the
step's instance by its output element (family-agnostic), not only `TokenSelect_v*` by row."""

from __future__ import annotations

import dataclasses
import types

import pytest

from verity_vllm.observe.fold import fold as fold_mod
from verity_vllm.observe.events import RunHeader, StepLink, StepStart, ViewDesc
from verity_vllm.observe.fold.fold import FoldError, fold
from tests.observe.test_fold_synthetic import REQ, toy_log, toy_profile  # memory fixture lives there
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


def _inst(step, node, spec, row, out_view, rows=1, request=None, position=None):
    return {"step": step, "node": node, "spec": spec, "row": row, "rows": rows, "request": request, "position": position,
            "output_views": [out_view]}


def _fold(instances, logits_rows=None):
    start = None if logits_rows is None else types.SimpleNamespace(logits_rows=logits_rows)
    return types.SimpleNamespace(instances=instances, cur_start=start)


def _node(k):
    return repr(Affine("n", k, 0, 1))


def test_fallback_matches_the_instance_by_output_element():
    sampled = ViewDesc(900, 0, (2,), (1,), "int64", "cuda:0")
    el0, el1 = (ViewDesc(900, i, (), (), "int64", "cuda:0") for i in (0, 1))
    ins = [_inst(3, 40, "GumbelArgmaxBf16_v1{V=16,SEED=7,POS=5}", 0, el0), _inst(3, 41, "GumbelArgmaxBf16_v1{V=16,SEED=8,POS=5}", 1, el1)]
    f = _fold(ins)
    ts = fold_mod._Fold._token_select_output
    assert repr(ts(f, 3, 0, el0)) == _node(40) and repr(ts(f, 3, 1, el1)) == _node(41)
    assert ts(f, 3, 0, None) is None                                          # not a TokenSelect: no by-row guess
    assert ts(f, 3, 0, ViewDesc(901, 0, (), (), "int64", "cuda:0")) is None    # another allocation
    assert ts(f, 2, 0, el0) is None                                           # another step
    # the registered greedy recognizer's TokenSelect (one Compute over the logits rows) is still found by row when the view is unavailable
    rows = [("a", 7), ("b", 4)]
    greedy = [_inst(3, 50, "TokenSelect_v1{V=16}", 0, None, rows=2, request="a", position=7),
              _inst(3, 51, "TokenSelect_v1{V=16}", 1, None, rows=2, request="b", position=4)]
    assert repr(ts(_fold(greedy, rows), 3, 1, None)) == _node(51) and repr(ts(_fold(greedy, rows), 3, 0, None)) == _node(50)
    unattributed = [_inst(3, 52, "TokenSelect_v1{V=16}", 1, None, rows=2)]
    assert repr(ts(_fold(unattributed, rows), 3, 1, None)) == _node(52) and ts(_fold(unattributed, rows), 3, 0, None) is None
    assert ts(_fold(unattributed, rows[:1]), 3, 1, None) is None              # a select not spanning the logits rows names none of them
    assert sampled.numel == 2


def test_fallback_takes_each_logits_row_s_own_gumbel_select_when_the_sampled_ids_are_freed():
    """A Gumbel select is one Compute per logits row, each its own row 0, emitted in logits-row order.  When the sampled ids' buffer
    is freed before SampleEnd (a rank's observer may order the free first), logits row 0 must get the select whose output element
    it is, not the step's last select: that one is another request's, and taking it feeds request a's next step request b's token."""
    spec = "GumbelTopPTokenSelectSharedGreedy_v1{V=154880}"
    el0, el1 = (ViewDesc(47277, i, (1,), (1,), "int64", "cuda:1") for i in (0, 1))
    rows = [("a", 248), ("b", 90)]
    ins = [_inst(13, 58, spec, 0, el0, request="a", position=248), _inst(13, 63, spec, 0, el1, request="b", position=90)]
    ts = fold_mod._Fold._token_select_output
    f = _fold(ins, rows)
    assert repr(ts(f, 13, 0, el0)) == _node(58) and repr(ts(f, 13, 1, el1)) == _node(63)
    assert repr(ts(f, 13, 0, None)) == _node(58) and repr(ts(f, 13, 1, None)) == _node(63)    # no view: by the row's (request, position)
    assert ts(f, 13, 2, None) is None
    assert ts(_fold([ins[1]], rows), 13, 0, None) is None                     # row 0's select missing: none, not row 1's
