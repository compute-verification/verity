"""The attention source on CPU (`verity_vllm/acquire/sources/attention_source.py`) against the GLM-5.3 capture map: the stand-in model
of `tests/acquire/glm53_model.py` runs the pinned vLLM's `DeepseekV32Attention.forward` call structure (`fused_norm_rope`'s cache
writes in vLLM's layouts, `_sparse_indexer_and_attn`, `forward_impl`'s split of a mixed step) on a dense prefill, a decode, a mixed
and a sparse prefill step, and a real `NativeHostCommitter`, whose `_acquire` records a copy on CPU, commits what the source binds.

It catches a name the map lists that the source does not bind on a step's routes, or binds off them or off the target; a value
taken from the wrong tensor, rows or moment (a cache row at the wrong slot or outside the indexer cache's block layout, the other
path's output rows, the shared top-k buffer after the next layer cleared it); and a step on the target that drops a target-only tap,
or a step off it that commits one."""
from __future__ import annotations

import copy
import types
from collections import Counter
from typing import Any

import pytest
import torch

from tests.acquire import glm53_model as G
from tests.acquire.glm53_model import CMAP, INDEXER, L, SPARSE, STEPS, glm_model, metadata, run_step
from tests.acquire.test_route_source import Run, map_names
from verity_vllm.acquire.sources import attention_source as AS
from verity_vllm.acquire.sources import route_source as RS

NORM_ROPE = G.fused_norm_rope
OTHERS = ("module_hook", "sampler", "allreduce_norm")
BY_ARGUMENT = {tid: arg for arg, tid in AS.ARGUMENTS.items()}


class AttnRun(Run):
    """`Run` with the attention source attached, recording each step's bound copies, the committer's acquire/flush order and what
    every layer computed."""

    def __init__(self, target: bool = True) -> None:
        super().__init__(target)
        self.att = AS.AttentionSource(self.com, self.src)
        self.src.attention = self.att
        self.att.install()
        self.rec: dict[int, list[tuple[str, torch.Tensor]]] = {}
        self.log: list[tuple[str, str]] = []
        self.seen: dict[int, dict[int, dict[str, Any]]] = {}
        self.com._acquire = self._record
        self.com.flush_after_plane = self._flush

    def _record(self, name: str, t: torch.Tensor, job: Any) -> None:
        self.rec.setdefault(int(job.step), []).append((name, t.clone()))
        self.log.append(("acquire", name))

    def _flush(self, what: str) -> bool:
        self.log.append(("flush", what))
        return False

    def step(self, k: int, md: Any, skip: tuple[int, ...] = ()) -> None:
        super().step(k, md, skip)
        self.seen[k] = {i: dict(self.src.layers[i].seen) for i in range(L) if i not in skip}

    def names(self, k: int) -> list[str]:
        return [n for n, _t in self.rec.get(k, [])]

    def close(self) -> None:
        self.att.uninstall()
        self.src.uninstall()


@pytest.fixture(scope="module")
def runs() -> dict[bool, AttnRun]:
    out = {}
    for target in (True, False):
        r = AttnRun(target)
        try:
            for k, (md, _routes, _phase) in enumerate(STEPS):
                r.step(k, md)
        finally:
            r.close()
        out[target] = r
    return out


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def expected(tid: str, seen: dict[str, Any]) -> torch.Tensor:
    if tid in BY_ARGUMENT:
        return seen[BY_ARGUMENT[tid]]
    if tid == "mla_kv_cache":
        return torch.cat((seen["kv_c"], seen["k_pe"]), dim=1)
    if tid == "indexer.k_cache":
        return torch.stack([G.index_key(k)[0] for k in seen["index_k"]])
    if tid == "indexer.k_scale":
        return torch.stack([G.index_key(k)[1].view(torch.float32) for k in seen["index_k"]])
    return {"indexer.topk": lambda: seen["topk"], "v_absorb": lambda: seen["output"][: seen["mqa_rows"]],
            "attn": lambda: seen["attn_mqa"], "attn.mha": lambda: seen["attn_mha"]}[tid]()


@pytest.mark.parametrize("target", [True, False], ids=["on-target", "off-target"])
def test_each_step_binds_once_each_name_capture_map_ids_lists(runs, capsys, target):
    r = runs[target]
    for k, (_md, routes, phase) in enumerate(STEPS):
        got = r.names(k)
        assert len(got) == len(set(got)), f"step {k} binds a name twice"
        assert sorted(got) == sorted(map_names(capsys, routes, phase, binders=(AS.BINDER,), off_target=not target)), f"step {k}"
    assert r.att.stats["unpaired"] == 0 and r.att.stats["layout_refused"] == 0 and r.att.stats["mha_unbound"] == 0


def test_each_bound_value_is_the_one_the_layer_computed(runs):
    r = runs[True]
    by_name = {n: (tid, i) for tid, layers in r.att.names.items() for i, n in layers.items()}
    checked: Counter = Counter()
    for k in range(len(STEPS)):
        for name, t in r.rec[k]:
            tid, i = by_name[name]
            assert same(t, expected(tid, r.seen[k][i])), f"step {k}: {name}"
            checked[tid] += 1
    assert set(checked) == AS.TAPS
    mixed = {r.att.names[tid][0]: tid for tid in ("v_absorb", "attn", "attn.mha")}
    rows = {mixed[n]: int(t.shape[0]) for n, t in r.rec[2] if n in mixed}
    assert rows == {"v_absorb": 1, "attn": 1, "attn.mha": 3}, "a mixed step's decode row goes through MQA, its prefill rows through MHA"


def test_the_shared_top_k_buffer_is_flushed_as_soon_as_it_is_bound(runs):
    log = runs[True].log
    topk = [j for j, (what, name) in enumerate(log) if what == "acquire" and name.endswith("self_attn/topk")]
    assert len(topk) == 3 * len(INDEXER), "the decode, mixed and sparse prefill steps, on every indexer layer"
    assert all(log[j + 1] == ("flush", AS.TOPK) for j in topk)


def full_steps(r: AttnRun, capsys, target: bool) -> dict[int, list[str]]:
    return {k: r.names(k) + map_names(capsys, routes, phase, binders=OTHERS, off_target=not target)
            for k, (_md, routes, phase) in enumerate(STEPS)}


def routes_bad(violation: dict[str, Any], names: set[str], counts: tuple[int, int]) -> bool:
    """`violation` reports exactly `names` against the map, each `counts` (committed, the map's): the first 8 by name, and how many."""
    want = {n: counts for n in names}
    return violation["routes_bad"] == dict(sorted(want.items())[:8]) and violation["routes_bad_n"] == len(want)


def test_identity_check_holds_a_step_to_the_target_only_taps_on_the_target_only(runs, capsys):
    q8 = f"model.layers.{min(INDEXER)}.self_attn/index_q"
    on, off = runs[True], runs[False]
    steps = full_steps(on, capsys, True)
    got = on.check(steps, sources=[on.src, on.att])
    assert got["ok"] is True, got["violations"][:2]
    v = on.check({**steps, 1: [n for n in steps[1] if n != q8]}, sources=[on.src, on.att])["violations"]
    assert [x["step"] for x in v] == [1] and routes_bad(v[0], {q8}, (0, 1)), "on the target a step without it fails"
    steps = full_steps(off, capsys, False)
    got = off.check(steps, sources=[off.src, off.att])
    assert got["ok"] is True, got["violations"][:2]
    v = off.check({**steps, 1: steps[1] + [q8]}, sources=[off.src, off.att])["violations"]
    assert [x["step"] for x in v] == [1] and routes_bad(v[0], {q8}, (1, 0)), "off the target a step that commits it fails"


def test_a_cache_write_the_source_did_not_see_binds_no_cache_rows(capsys):
    r = AttnRun()
    wrapped, G.fused_norm_rope = G.fused_norm_rope, NORM_ROPE
    try:
        r.step(0, STEPS[1][0])
    finally:
        G.fused_norm_rope = wrapped
        r.close()
    rows = {r.att.names["mla_kv_cache"][i] for i in range(L)} | {r.att.names[t][i] for t in ("indexer.k_cache", "indexer.k_scale") for i in INDEXER}
    assert set(map_names(capsys, SPARSE, "decode", binders=(AS.BINDER,))) - set(r.names(0)) == rows
    assert r.att.stats["unpaired"] == L and G.fused_norm_rope is NORM_ROPE
    v = r.check({0: r.names(0) + map_names(capsys, SPARSE, "decode", binders=OTHERS)}, sources=[r.src, r.att])["violations"]
    assert routes_bad(v[0], rows, (0, 1))


def test_a_prefill_over_a_chunked_context_leaves_its_mha_output_unbound(capsys):
    md = metadata(prefill=True, dense=True)
    md.prefill.chunked_context = types.SimpleNamespace()
    r = AttnRun()
    try:
        r.step(0, md)
    finally:
        r.close()
    mha = {r.att.names["attn.mha"][i] for i in range(L)}
    assert not mha & set(r.names(0)) and r.att.stats["mha_unbound"] == L
    v = r.check({0: r.names(0) + map_names(capsys, ("dense-mha",), "prefill", binders=OTHERS)}, sources=[r.src, r.att])["violations"]
    assert routes_bad(v[0], mha, (0, 1)), "the step fails closed"


def test_the_arena_estimate_covers_what_each_step_binds(runs):
    for r in runs.values():
        est = r.att.extra_step_bytes({}, None)
        for k, (md, _routes, _phase) in enumerate(STEPS):
            assert sum(t.numel() * t.element_size() for _n, t in r.rec[k]) <= est["per_token_bytes"] * md.num_actual_tokens
            assert len(r.rec[k]) <= est["tensors_per_step"]
    assert runs[True].att.extra_step_bytes({}, None)["per_token_bytes"] > runs[False].att.extra_step_bytes({}, None)["per_token_bytes"]


def test_install_wraps_each_call_and_uninstall_restores_them():
    model = glm_model()
    src = RS.RouteSource(None, model, CMAP, target=True)
    att = AS.AttentionSource(None, src)
    att.install()
    a0 = src.layers[0]
    assert AS.ATTEND in vars(a0) and AS.MQA in vars(a0.impl) and AS.MHA in vars(a0.impl) and G.fused_norm_rope is not NORM_ROPE
    run_step(model, STEPS[2][0], 0)
    assert att.stats["calls"] == 0 and att.stats["bound"] == 0, "nothing is bound outside a committer step"
    att.uninstall()
    for a in src.layers.values():
        assert AS.ATTEND not in vars(a) and AS.MQA not in vars(a.impl) and AS.MHA not in vars(a.impl)
    assert G.fused_norm_rope is NORM_ROPE


def test_attach_puts_the_attention_source_beside_the_route_source():
    class Com:
        def __init__(self) -> None:
            self.sources: list[Any] = []

        def attach_source(self, s: Any) -> None:
            self.sources.append(s)

    com = Com()
    src = RS.attach(com, glm_model(), rank=3)
    assert [type(s) for s in com.sources] == [RS.RouteSource, AS.AttentionSource] and src.attention is com.sources[1]
    assert src.stats["attention"]["layers"] == L


def test_a_layer_or_a_map_the_source_cannot_bind_is_refused():
    class Bare(torch.nn.Module):
        def _use_sparse_mha(self, attn_metadata: Any) -> bool:
            return False

    with pytest.raises(ValueError, match=f"has no {AS.ATTEND}"):
        AS.AttentionSource(None, RS.RouteSource(None, glm_model(attention=Bare), CMAP))
    cmap = copy.deepcopy(CMAP)
    tap = next(t for t in cmap["taps"] if t["id"] == "attn")
    cmap["taps"].append({**copy.deepcopy(tap), "id": "attn.lse", "commit": {**tap["commit"], "name": "model.layers.{i}.self_attn/lse"}})
    with pytest.raises(ValueError, match="binds no tap"):
        AS.AttentionSource(None, RS.RouteSource(None, glm_model(), cmap))
