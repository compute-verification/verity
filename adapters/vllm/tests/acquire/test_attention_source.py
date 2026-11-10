"""The attention source on CPU (`verity_vllm/acquire/sources/attention_source.py`) against the GLM-5.3 capture map: the stand-in model
of `tests/acquire/glm53_model.py` runs the pinned vLLM's `DeepseekV32Attention.forward` call structure (`fused_norm_rope`'s cache
writes in vLLM's layouts, `_sparse_indexer_and_attn`, `forward_impl`'s split of a mixed step) on a dense prefill, a decode, a mixed
and a sparse prefill step, and a real `NativeHostCommitter`, whose `_acquire` records a copy on CPU, commits what the source binds.

It catches a name the map lists that the source does not bind on a step's routes, or binds off them or off the target; a value
taken from the wrong tensor, rows or moment (a cache row at the wrong slot or outside the indexer cache's block layout, the other
path's output rows, the shared top-k buffer after the next layer cleared it); a step on the target that drops a target-only tap,
or a step off it that commits one; and a sparse MLA call boundary bound at a step whose `attn` tensor is not its value's shape, or
bound to another step's tensor or a layer's other output, or claimed without the attention source attached."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

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
from verity_vllm.acquire.sources import call_boundary_source as CB
from verity_vllm.acquire.sources import route_source as RS
from verity_vllm.commit import binding as B
from verity_vllm.query.manifest.bind import Binder, source_ports
from verity_vllm.commit.committer.native_host import GPU_CHUNK_BYTES, TensorMeta

NORM_ROPE = G.fused_norm_rope
OTHERS = ("module_hook", "sampler", "allreduce_norm")
BY_ARGUMENT = {tid: arg for arg, tid in AS.ARGUMENTS.items()}


class AttnRun(Run):
    """`Run` with the attention source attached, recording each step's bound copies, the committer's acquire/flush order and what
    every layer computed."""

    def __init__(self, target: bool = True, identities: tuple[dict[str, Any], ...] = ()) -> None:
        super().__init__(target)
        self.att = AS.AttentionSource(self.com, self.src, identities)
        self.src.attention = self.att
        self.att.install()
        self.rec: dict[int, list[tuple[str, torch.Tensor]]] = {}
        self.unstaged: list[str] = []
        self.log: list[tuple[str, str]] = []
        self.seen: dict[int, dict[int, dict[str, Any]]] = {}
        self.com._acquire = self._record
        self.com.flush_after_plane = self._flush

    def _record(self, name: str, t: torch.Tensor, job: Any) -> None:
        self.rec.setdefault(int(job.step), []).append((name, t.clone()))
        if not AS.pitched_2d(t):
            self.unstaged.append(name)
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


@pytest.mark.slow
@pytest.mark.parametrize("target", [True, False], ids=["on-target", "off-target"])
def test_each_step_binds_once_each_name_capture_map_ids_lists(runs, capsys, target):
    r = runs[target]
    for k, (_md, routes, phase) in enumerate(STEPS):
        got = r.names(k)
        assert len(got) == len(set(got)), f"step {k} binds a name twice"
        assert sorted(got) == sorted(map_names(capsys, routes, phase, binders=(AS.BINDER,), off_target=not target)), f"step {k}"
    assert r.att.stats["unpaired"] == 0 and r.att.stats["layout_refused"] == 0 and r.att.stats["mha_unbound"] == 0


@pytest.mark.slow
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


@pytest.mark.slow
def test_the_shared_top_k_buffer_is_flushed_as_soon_as_it_is_bound(runs):
    log = runs[True].log
    topk = [j for j, (what, name) in enumerate(log) if what == "acquire" and name.endswith("self_attn/topk")]
    assert len(topk) == 3 * len(INDEXER), "the decode, mixed and sparse prefill steps, on every indexer layer"
    assert all(log[j + 1] == ("flush", AS.TOPK) for j in topk)


def test_pitched_2d_is_the_native_collectors_staging_rule():
    base = torch.zeros(5, 576)
    assert AS.pitched_2d(base) and AS.pitched_2d(base[:, 512:]), "contiguous, and rows of 64 at a pitch of 576"
    assert AS.pitched_2d(base[:, 512:].unsqueeze(1)), "a size-1 dim carries no layout"
    absorbed = torch.zeros(8, 5, 512).transpose(0, 1)
    assert not AS.pitched_2d(absorbed), "a transposed bmm output [T, H, L], strides (L, T * L, 1)"
    assert AS.pitched_2d(absorbed[:1]), "one row of it is contiguous"
    assert AS.pitched_2d(base[:, ::2]), "every other word: rows of one word at a pitch of two"
    assert not AS.pitched_2d(base[::2, ::2]), "every other word of every other row"


@pytest.mark.slow
def test_every_value_handed_to_the_committer_is_one_the_native_collector_stages(runs):
    """The native collector stages a contiguous tensor or equally spaced contiguous rows and raises on anything else ("layout is
    not 2-D pitched", every rank on 8x H200). The W_UK_T product of a step of more than one token is a
    transposed bmm output, so the source hands the committer a contiguous copy of it, word-equal to it
    (`test_each_bound_value_is_the_one_the_layer_computed`)."""
    for r in runs.values():
        assert r.unstaged == []
        assert "q_nope_absorb" in r.att.stats["copied_by_tap"]
    multi = sum(1 for md, _routes, _phase in STEPS if md.num_actual_tokens > 1)
    assert runs[True].att.stats["copied_by_tap"]["q_nope_absorb"] == multi * L


def full_steps(r: AttnRun, capsys, target: bool) -> dict[int, list[str]]:
    return {k: r.names(k) + map_names(capsys, routes, phase, binders=OTHERS, off_target=not target)
            for k, (_md, routes, phase) in enumerate(STEPS)}


def routes_bad(violation: dict[str, Any], names: set[str], counts: tuple[int, int]) -> bool:
    """`violation` reports exactly `names` against the map, each `counts` (committed, the map's): the first 8 by name, and how many."""
    want = {n: counts for n in names}
    return violation["routes_bad"] == dict(sorted(want.items())[:8]) and violation["routes_bad_n"] == len(want)


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_the_arena_estimate_covers_what_each_step_binds(runs):
    for r in runs.values():
        est = r.att.extra_step_bytes({}, None)
        for k, (md, _routes, _phase) in enumerate(STEPS):
            assert sum(t.numel() * t.element_size() for _n, t in r.rec[k]) <= est["per_token_bytes"] * md.num_actual_tokens
            assert len(r.rec[k]) <= est["tensors_per_step"]
    assert runs[True].att.extra_step_bytes({}, None)["per_token_bytes"] > runs[False].att.extra_step_bytes({}, None)["per_token_bytes"]


@pytest.mark.slow
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


WIDTH = G.NHL * G.KVL
PD = "f" * 64


def boundary(step: int, layer: int, rows: int, **kw: Any) -> dict[str, Any]:
    """A sparse MLA call boundary of rank 3 as the GLM-5.3 TP manifest spells it: fx node `dsa_indexer_attn_<step * L + layer>`,
    [rows, NHL * KVL] bf16 words."""
    k = step * L + layer
    return {"family": "call_boundaries", "request_id": "r0", "step": step, "engine_step": step, "invocation": 0, "rank": 3,
            "op_path": f"model.layers.{layer}.self_attn/dsa_indexer_attn" + (f"_{k}" if k else ""), "output_member": "out",
            "element_range": [0, rows * WIDTH], "rows": rows, "width": WIDTH, "dtype": "bf16", "spec": "SparseMlaSplit_v1{T=1,NH=8}", **kw}


#: every layer's sparse MLA at the decode (1 row) and the sparse prefill (3 rows); layer 0's also at the dense prefill, which runs no
#: MQA, and at the mixed step as the Program would state it (4 rows), where MQA attends its 1 decode row only
BOUNDARIES = tuple([boundary(1, i, 1) for i in range(L)] + [boundary(3, i, 3) for i in range(L)] + [boundary(0, 0, 3), boundary(2, 0, 4)])


@pytest.fixture(scope="module")
def bruns() -> dict[bool, AttnRun]:
    out = {}
    for target in (True, False):
        r = AttnRun(target, BOUNDARIES)
        try:
            for k, (md, _routes, _phase) in enumerate(STEPS):
                r.step(k, md)
        finally:
            r.close()
        out[target] = r
    return out


def test_the_source_claims_the_attention_bodys_call_boundaries_its_taps_bind_and_nothing_else():
    """By the Call's Definition and the layer's attention module: the fx node's name does not enter."""
    b = boundary(1, 2, 1)
    assert AS.claims(b) == AS.BINDER and CB.claimed(b) == AS.BINDER and not CB.manifest_sites({"identities": [b]})
    assert AS.call_tap(b) == (AS.ATTN_MQA, 2) and AS.call_tap(boundary(0, 0, 3)) == (AS.ATTN_MQA, 0)
    for other in ({"spec": "SparseMla_v1{T=1}"}, {"op_path": "model.layers.2.self_attn/dsa_norm_rope_9"}):
        assert AS.call_tap({**b, **other}) == (AS.ATTN_MQA, 2), other
    assert AS.call_tap({**b, "spec": "BatchedGemm_v1{B=8}", "op_path": "model.layers.2.self_attn/bmm_3"}) == (AS.ARGUMENTS["ql_nope"], 2)
    for other in ({"spec": "DsaMqaLogits_v3{N=2}"}, {"output_member": "lse"}, {"dtype": "f32"}, {"family": "instance_outputs"},
                  {"op_path": "model.layers.2.mlp/dsa_indexer_attn_9"}, {"op_path": "model.layers.2.self_attn"}):
        assert AS.claims({**b, **other}) is None, other


@pytest.mark.slow
def test_each_sparse_mla_call_boundary_holds_the_attn_tensor_of_its_layer_and_step(bruns):
    on = bruns[True]
    labels = {n.rsplit("/", 1)[1] for n in on.att.names[AS.ATTN_MQA].values()}
    assert len(labels) == 1 and all(on.att.ports()[x] == AS.TAP_PORTS[AS.ATTN_MQA] for x in labels)
    assert on.att.stats["call_boundaries_held"] == 2 * L
    assert len(on.com.errors) == 1 and on.com.errors[0].startswith(f"attention: layer 0 at engine step 2: tap {AS.ATTN_MQA} is (1, ")
    off = bruns[False]
    assert off.att.stats["call_boundaries_held"] == 0 and off.com.errors == [], "off the target the attn tap is not bound, so neither are they"


@pytest.mark.slow
def test_the_binding_map_covers_each_held_call_boundary_exactly_and_names_the_rest(bruns):
    on = bruns[True]
    layouts = {}
    for k, recs in on.rec.items():
        layouts[k], off = [], 0
        for j, (name, t) in enumerate(recs):
            nb = t.numel() * t.element_size()
            m = TensorMeta(j, name, "bfloat16", tuple(t.shape), nb, GPU_CHUNK_BYTES, off // GPU_CHUNK_BYTES, 1)
            m.stream_off, off = off, off + nb
            layouts[k].append(m)
    steps = [types.SimpleNamespace(root=bytes([k]) * 32) for k in range(len(STEPS))]
    bmap = B.build_binding_map(layouts, steps, b"\0" * 32, PD, rank=3, binder=Binder(BOUNDARIES, rank=3, ports=source_ports([on.att])))
    pop = [B.identity(PD, r["request_id"], r["engine_step"], 0, r["op_path"], "out", *r["element_range"], 3) for r in BOUNDARIES]
    cov = B.coverage_check(pop, bmap)
    assert (cov["expected_n"], cov["bound_n"], cov["ambiguous_n"]) == (len(BOUNDARIES), 2 * L, 0)
    assert sorted((m["step"], m["op_path"]) for m in cov["missing"]) == [(0, boundary(0, 0, 3)["op_path"]), (2, boundary(2, 0, 4)["op_path"])]
    attn = on.att.names[AS.ATTN_MQA]
    tapped = [e for e in bmap["entries"] if e.get("observation") == "port"]
    assert len(tapped) == 2 * L + 1, "every attn tensor at a step whose layer has its call boundary, the mixed step's among them"
    for e in tapped:
        i, st, rows = AS.call_tap({**e["identity"], "family": CB.FAMILY, "dtype": "bf16", "spec": "SparseMlaSplit_v1"})[1], e["identity"]["step"], e["encoding"]["shape"][0]
        assert e["name"] == attn[i] and e["encoding"]["shape"] == [rows, G.NHL, G.KVL]
        assert e["identity"]["op_path"] == next(r["op_path"] for r in BOUNDARIES if r["step"] == st and AS.call_tap(r)[1] == i)
    mixed = next(e for e in tapped if e["identity"]["step"] == 2)
    assert mixed["identity"]["element_range"] == [0, WIDTH] != boundary(2, 0, 4)["element_range"], "bound, and short of the 4 rows it must hold"


def test_a_step_batching_requests_at_different_steps_of_their_own_holds_and_maps_each_requests_call_boundary():
    """[GLM-5.3 B8 mixed arrivals] Each request's Program numbers its fx nodes over its own steps, so engine step 9 spells r0's sparse
    MLA (its step 9, 1 row) `dsa_indexer_attn_<9 * L>` and r1's (its step 0, 3 rows) `dsa_indexer_attn`: the step's one attn tensor
    holds both, and the binding map writes each request's rows under its own spelling."""
    rows = (boundary(9, 0, 1), {**boundary(0, 0, 3), "request_id": "r1", "engine_step": 9})
    src = object.__new__(AS.AttentionSource)
    src.c, src.stats, src.boundaries = types.SimpleNamespace(errors=[]), {"call_boundaries_held": 0}, {}
    for r in rows:
        src.boundaries.setdefault((9, *AS.call_tap(r)), []).append(r)
    job = types.SimpleNamespace(step=9)
    src._hold_boundary(AS.ATTN_MQA, 0, torch.zeros(4, G.NHL, G.KVL, dtype=torch.bfloat16), job)
    assert (src.c.errors, src.stats["call_boundaries_held"]) == ([], 2)
    src._hold_boundary(AS.ATTN_MQA, 0, torch.zeros(3, G.NHL, G.KVL, dtype=torch.bfloat16), job)
    assert len(src.c.errors) == 1, "short of the 4 rows the two requests state"
    src.boundaries[(9, AS.ATTN_MQA, 0)].append({**rows[0], "op_path": rows[1]["op_path"]})
    src._hold_boundary(AS.ATTN_MQA, 0, torch.zeros(5, G.NHL, G.KVL, dtype=torch.bfloat16), job)
    assert len(src.c.errors) == 2, "two boundaries of one request at one layer and step"

    name = "model.layers.0.self_attn/attn"
    m = TensorMeta(0, name, "bfloat16", (4, G.NHL, G.KVL), 4 * WIDTH * 2, GPU_CHUNK_BYTES, 0, 1)
    m.stream_off = 0
    steps = [types.SimpleNamespace(root=bytes([k]) * 32) for k in range(10)]
    binder = Binder(rows, rank=3, ports={"attn": AS.TAP_PORTS[AS.ATTN_MQA]})
    bmap = B.build_binding_map({9: [m]}, steps, b"\0" * 32, PD, rank=3, request_rows={9: [("r0", 0, 1), ("r1", 1, 4)]}, binder=binder)
    got = sorted((e["identity"]["request_id"], e["identity"]["op_path"], e["identity"]["element_range"], e["element_offset"]) for e in bmap["entries"])
    assert got == [("r0", rows[0]["op_path"], [0, WIDTH], 0), ("r1", rows[1]["op_path"], [0, 3 * WIDTH], WIDTH)]
    pop = [B.identity(PD, r["request_id"], r["engine_step"], 0, r["op_path"], "out", *r["element_range"], 3) for r in rows]
    cov = B.coverage_check(pop, bmap)
    assert (cov["expected_n"], cov["bound_n"], cov["ambiguous_n"], cov["missing"]) == (2, 2, 0, [])


def test_a_non_contiguous_tap_is_staged_as_a_contiguous_copy_held_for_the_step():
    """[GLM-5.3] ql_nope is bmm(q_nope^T, W_UK_T).transpose(0, 1): [M, NH, L] over strides no 2-D pitch describes, which the native
    committer refuses ("layout is not 2-D pitched"); the source stages its contiguous copy, the same values in row-major order."""
    import torch
    from verity_vllm.acquire.sources import attention_source as AS
    got = []
    src = object.__new__(AS.AttentionSource)
    src.c = type("C", (), {"_acquire": lambda self, name, t, job: got.append((name, t))})()
    src.names, src.stats, src._copies = {"ql": {0: "model.layers.0.self_attn/ql"}}, {"bound": 0, "by_tap": {}, "copied_by_tap": {}}, []
    x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(3, 2, 4).transpose(0, 1)
    assert not x.is_contiguous()
    src._bind("ql", 0, x, job=None)
    (name, t), = got
    assert name == "model.layers.0.self_attn/ql" and t.is_contiguous() and torch.equal(t, x) and src._copies == [t]
    src.on_step_begin = AS.AttentionSource.on_step_begin.__get__(src)
    src._pending = src._active = None
    src.on_step_begin()
    assert src._copies == []
