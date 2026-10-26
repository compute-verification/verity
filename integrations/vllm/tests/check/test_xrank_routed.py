"""[TP-12] the cross-rank pass at world 8: each all-reduce pick is evaluated with the routed Definition its call binds (Multimem, Custom,
Mnnvl, the fused Mnnvl norm), equal to the Definition on every word at the picked rows; a flipped partial, a spec the ranks disagree on
or the rank Programs do not bind, and an uncovered call are each named, never a pass."""
from __future__ import annotations

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity.ml.circuits import evaluation as E  # noqa: E402
from verity.ml.circuits.defs import bind  # noqa: E402
from verity_vllm.check.match import xrank_collectives as XC  # noqa: E402
from verity_vllm.pipeline.tp import xrank_dump as XD  # noqa: E402
from verity.ml.definitions.registry import tp_allreduce as TA  # noqa: E402

W = 8
M, NC = 6, 24                       # 144 words: multimem's 32-word owner slices and custom's 2-stage packs cross the rows
ROWS = [1, 4, 5]
TREES = tuple(TA.chain(W, [(k + j) % W for j in range(W)]) for k in range(W))   # a different tree per owner


def _bf16(rng, shape, scale: float = 1.0) -> np.ndarray:
    return ((rng.standard_normal(shape) * scale).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _t(words: np.ndarray):
    return torch.from_numpy(np.ascontiguousarray(words).view(np.int16)).view(torch.bfloat16)


def _definition(defn, statics, parts_flat) -> np.ndarray:
    out = E.evaluate_batch(bind(defn, **statics), parts_flat[None])
    parts = out if isinstance(out, (tuple, list)) else [out]
    return np.concatenate([np.asarray(a, np.uint64).reshape(1, -1) for a in parts], axis=1)[0].astype(np.uint16)


FLAT = [("AllReduceMultimem_v1", TA.AllReduceMultimem, {"WORLD": W, "SIZE": M * NC, "TREES": TREES, "FTZ": False}),
        ("AllReduceCustom_v1", TA.AllReduceCustom, {"WORLD": W, "SIZE": M * NC, "STAGE": 2}),
        ("AllReduceMnnvl_v1", TA.AllReduceMnnvl, {"WORLD": W, "N": M * NC, "SHOT": 1}),
        ("AllReduceMnnvl_v1", TA.AllReduceMnnvl, {"WORLD": W, "N": M * NC, "SHOT": 2})]


W4 = 4
NCCL = [(XC.NCCL_TREE_FAMILY, TA.AllReduceNcclTree, {"WORLD": W4, "SIZE": M * NC, "CHAIN": chain}) for chain in ((0, 1, 2, 3), (2, 0, 3, 1))]


@pytest.mark.parametrize("family,defn,statics", FLAT + NCCL, ids=lambda x: x if isinstance(x, str) else "")
def test_routed_expected_is_the_definition_at_the_picked_rows(family, defn, statics) -> None:
    world = statics["WORLD"]
    parts = _bf16(np.random.default_rng(len(family) + statics.get("SHOT", 0)), (world, M, NC))
    want = _definition(defn, statics, parts.reshape(world, -1)).reshape(M, NC)
    (got,) = XC.routed_expected(family, statics, parts[:, ROWS], ROWS)
    assert (got == want[ROWS]).all()
    if family not in ("AllReduceMnnvl_v1", XC.NCCL_TREE_FAMILY):   # the owner routes read row * N + col of the call's M * N words
        with pytest.raises(ValueError, match="past SIZE"):
            XC.routed_expected(family, statics, np.zeros((W, 1, NC), np.uint16), [M])


def _fused(shot: int, fma: bool):
    rng = np.random.default_rng(shot * 2 + fma)
    N = 256
    parts, res, w = _bf16(rng, (W, M, N), 0.5), _bf16(rng, (M, N), 2.0), _bf16(rng, (N,), 0.1)
    statics = {"WORLD": W, "N": N, "SHOT": shot, "CLUSTER": 1, "EPS": 1e-5, "FMA": fma}
    return parts, res, w, statics


@pytest.mark.parametrize("shot,fma", [(1, True), (2, False)])
def test_the_fused_norm_is_the_definition_per_token(shot, fma) -> None:
    parts, res, w, statics = _fused(shot, fma)
    d = bind(TA.AllReduceRmsNormMnnvl, **statics)
    normed, r = XC.routed_expected(XC.FUSED_FAMILY, statics, parts[:, ROWS], ROWS, res[ROWS], w)
    for j, m in enumerate(ROWS):
        got = np.array(E.evaluate(d, parts[:, m].reshape(-1).tolist(), res[m].tolist(), w.tolist()), np.uint16)
        assert (got[:256] == normed[j]).all() and (got[256:] == r[j]).all()
    with pytest.raises(ValueError, match="residual"):
        XC.routed_expected(XC.FUSED_FAMILY, statics, parts[:, ROWS], ROWS)


def test_spec_key_reads_nested_statics_and_ignores_their_order() -> None:
    spec = bind(TA.AllReduceMultimem, WORLD=W, SIZE=M * NC, TREES=TREES, FTZ=False).id
    fam, items = XC.spec_key(spec)
    assert fam == "AllReduceMultimem_v1" and dict(items)["TREES"] == TREES and dict(items)["SIZE"] == M * NC
    assert XC.spec_key("AllReduceMnnvl_v1{SHOT=1,WORLD=8,N=96}") == XC.spec_key(bind(TA.AllReduceMnnvl, WORLD=8, N=96, SHOT=1).id)
    assert XC.spec_key("AllReduceMnnvl_v1{WORLD=8,N=96,SHOT=2}") != XC.spec_key("AllReduceMnnvl_v1{WORLD=8,N=96,SHOT=1}")


def test_spec_key_of_the_glm53_fused_norm_as_its_program_spells_it() -> None:
    """The GLM-5.3 Program's fused norm spec spells EPS as a JSON object: its key is hashable, so it can be a member of `known`."""
    eps = float.fromhex("0x1.4f8b588e368f1p-17")
    program = 'AllReduceRmsNormMnnvl_v1{WORLD=8,N=6144,SHOT=1,CLUSTER=8,EPS={"f64":"0x1.4f8b588e368f1p-17"},FMA=true}'
    route = bind(TA.AllReduceRmsNormMnnvl, WORLD=8, N=6144, SHOT=1, CLUSTER=8, EPS=eps, FMA=True).id
    assert XC.spec_key(route) in {XC.spec_key(program)}
    assert XC.spec_key(route.replace("SHOT=1", "SHOT=2")) not in {XC.spec_key(program)}


# --- check_picks over eight ranks' dumps -----------------------------------------------------------------------------------------

def _dumps(family, statics, *, fused: bool = False, flip: int | None = None, spec_of=None, binds=lambda r: True, drop=None, route_why=None,
           calls=None):
    world = W if statics is None else statics["WORLD"]
    if fused:
        parts, res, w, statics = _fused(1, True)
        defn, N = TA.AllReduceRmsNormMnnvl, 256
        out = XC.routed_expected(XC.FUSED_FAMILY, statics, parts[:, ROWS], ROWS, res[ROWS], w)
    else:
        parts = _bf16(np.random.default_rng(3), (world, M, NC))
        defn, N = {"AllReduceMultimem_v1": TA.AllReduceMultimem, "AllReduceCustom_v1": TA.AllReduceCustom,
                   "AllReduceMnnvl_v1": TA.AllReduceMnnvl, XC.NCCL_TREE_FAMILY: TA.AllReduceNcclTree}[family], NC
        out = (_definition(defn, statics, parts.reshape(world, -1)).reshape(M, N)[ROWS],)
    spec = bind(defn, **statics).id
    site = "model.layers.2.post_attention_layernorm"
    dumps = {}
    for r in range(world):
        part = parts[r, ROWS].copy()
        if flip == r:
            part[0] ^= 0x8000                         # a partial row that is not the collective's input
        p = {"site": site, "step": 3, "kind": "all_reduce", "family": family, "stratum": f"{site}|all_reduce|decode", "rows": ROWS,
             "rows_total": M, "spec": (spec_of(r, spec) if spec_of else spec), "statics": statics, "fused": fused, "program_binds": binds(r),
             "part": _t(part), "recv": _t(out[0]), "why": None}
        if fused:
            p.update(recv1=_t(out[1]), res=_t(res[ROWS]), w=_t(w))
        if calls is not None:
            p["calls"] = calls(r) if callable(calls) else calls
        if drop and drop[0] == r:
            p[drop[1]] = None
        if route_why is not None:
            p = {k: v for k, v in p.items() if k not in ("spec", "statics", "fused", "program_binds")} | {"route_why": route_why, "family": "AllReduce2_v1"}
        dumps[r] = {"picks": [p]}
    return dumps


@pytest.mark.parametrize("family,defn,statics", FLAT + NCCL[:1], ids=lambda x: x if isinstance(x, str) else "")
def test_an_honest_routed_pick_is_equal_and_a_flipped_partial_is_a_named_mismatch(family, defn, statics) -> None:
    world = statics["WORLD"]
    r = XC.check_picks(_dumps(family, statics), world=world)
    assert r["result"] is True and r["equal"] == r["picks"] == 1 and r["by_family"][family]["equal"] == 1
    assert r["by_stratum"]["model.layers.2.post_attention_layernorm|all_reduce|decode"]["families"] == [family]
    bad = XC.check_picks(_dumps(family, statics, flip=world - 3), world=world)
    m = bad["mismatches"][0]
    assert bad["result"] is False and bad["mismatch_n"] == 1 and m["family"] == family
    assert not all(m["received_equals_definition_per_rank"].values()) and m["received_equal_across_ranks"] is True


def test_the_fused_norm_pick_checks_both_returns_from_the_residual_and_weight() -> None:
    assert XC.check_picks(_dumps(XC.FUSED_FAMILY, None, fused=True), world=W)["result"] is True
    d = _dumps(XC.FUSED_FAMILY, None, fused=True)
    d[2]["picks"][0]["recv1"] = _t(np.zeros((len(ROWS), 256), np.uint16))
    bad = XC.check_picks(d, world=W)
    assert bad["result"] is False and bad["mismatches"][0]["received_equals_definition_per_rank"]["2"] is False
    miss = XC.check_picks(_dumps(XC.FUSED_FAMILY, None, fused=True, drop=(4, "res")), world=W)
    assert miss["result"] is False and miss["missing_n"] == 1 and "rank 4: res" in miss["missing"][0]["why"]


def test_a_spec_the_ranks_disagree_on_or_the_programs_do_not_bind_is_not_evaluable_by_name() -> None:
    fam, _d, st = FLAT[2]
    other = lambda r, s: s.replace("SHOT=1", "SHOT=2") if r == 6 else s  # noqa: E731
    r = XC.check_picks(_dumps(fam, st, spec_of=other), world=W)
    assert r["result"] is False and "different Definitions" in r["missing"][0]["why"]
    r = XC.check_picks(_dumps(fam, st, binds=lambda k: k != 3), world=W)
    assert r["result"] is False and "rank 3: the rank Program binds no AllReduceMnnvl_v1" in r["missing"][0]["why"]
    r = XC.check_picks(_dumps(fam, st, binds=lambda k: None), world=W)
    assert r["result"] is False and "routed specs were not read" in r["missing"][0]["why"]


CALLS = [(0, 2), (2, 4)]                  # two requests' runs of the launch's M rows; ROWS picks from both
OFFSET_FREE = [("AllReduceMultimem_v1", TA.AllReduceMultimem, lambda n: {"WORLD": W, "SIZE": n * NC, "TREES": (TREES[0],) * W, "FTZ": False}),
               ("AllReduceMnnvl_v1", TA.AllReduceMnnvl, lambda n: {"WORLD": W, "N": n * NC, "SHOT": 2})]


def _calls(defn, statics_of):
    return [{"lo": lo, "rows": n, "spec": bind(defn, **statics_of(n)).id, "statics": statics_of(n)} for lo, n in CALLS]


@pytest.mark.parametrize("family,defn,statics_of", OFFSET_FREE, ids=lambda x: x if isinstance(x, str) else "")
def test_a_launch_over_several_requests_is_each_requests_call_of_the_definition(family, defn, statics_of) -> None:
    parts = _bf16(np.random.default_rng(11), (W, M, NC))
    calls = _calls(defn, statics_of)
    want = np.concatenate([_definition(defn, c["statics"], parts[:, c["lo"]:c["lo"] + c["rows"]].reshape(W, -1)).reshape(c["rows"], NC)
                           for c in calls])
    (got,) = XC.calls_expected(family, statics_of(M), calls, parts[:, ROWS], ROWS)
    assert (got == want[ROWS]).all()
    with pytest.raises(ValueError, match="in no request's call"):
        XC.calls_expected(family, statics_of(M), calls[:1], parts[:, ROWS], ROWS)


def test_a_pick_over_several_requests_is_checked_per_call_and_every_call_must_be_bound() -> None:
    fam, defn, statics_of = OFFSET_FREE[0]
    calls = _calls(defn, statics_of)
    r = XC.check_picks(_dumps(fam, statics_of(M), calls=calls), world=W)
    assert r["result"] is True and r["equal"] == 1
    bad = XC.check_picks(_dumps(fam, statics_of(M), calls=calls, flip=2), world=W)
    assert bad["result"] is False and bad["mismatches"][0]["request_calls"] == CALLS
    split = XC.check_picks(_dumps(fam, statics_of(M), calls=lambda k: calls[:1] if k == 5 else calls), world=W)
    assert split["result"] is False and "different Definitions" in split["missing"][0]["why"]
    d = _dumps(fam, statics_of(M), calls=calls, binds=lambda k: k != 3)
    d[3]["picks"][0]["unbound_spec"] = calls[1]["spec"]
    r = XC.check_picks(d, world=W)
    assert r["result"] is False and f"rank 3: the rank Program binds no {calls[1]['spec'][:120]}" in r["missing"][0]["why"]


def test_an_uncovered_call_is_not_evaluated_by_the_two_rank_chain() -> None:
    fam, _d, st = FLAT[0]
    r = XC.check_picks(_dumps(fam, st, route_why="tp_allreduce.covered() has no entry"), world=W)
    assert r["result"] is False and r["equal"] == 0 and r["missing_n"] == 1
    assert r["missing"][0]["why"].startswith("no routed Definition for the call: rank 0: tp_allreduce.covered()")


# --- the rank side ---------------------------------------------------------------------------------------------------------------

def test_route_of_names_the_definition_the_switches_take_at_glm53() -> None:
    s, N = TA.settings_of(False, {}), 6144
    norm = types.SimpleNamespace(variance_epsilon=1e-5)
    fused = XD.route_of(s, 42, N, W, norm)
    assert fused["fused"] is True and fused["family"] == XC.FUSED_FAMILY
    assert fused["statics"] == {"WORLD": 8, "N": N, "SHOT": 2, "CLUSTER": 2, "EPS": 1e-5, "FMA": True}
    assert XD.route_of(s, 43, N, W, norm)["family"] == "AllReduceMultimem_v1"
    dec = XD.route_of(s, 4, N, W)
    assert dec["family"] == "AllReduceMnnvl_v1" and dec["statics"] == {"WORLD": 8, "N": 4 * N, "SHOT": 1} and dec["fused"] is False
    pre = XD.route_of(s, 256, N, W)
    assert XC.spec_key(pre["spec"]) == XC.spec_key(bind(TA.AllReduceMultimem, WORLD=8, SIZE=256 * N, TREES=(TA.GLM53_H200_TREE,) * 8,
                                                        FTZ=False).id)
    assert XD.route_of(s, 4, N, 2) is None
    assert "no entry" in XD.route_of(TA.settings_of(False, {"VLLM_ALLREDUCE_USE_SYMM_MEM": "0"}), 4096, N, W)["why"]
    assert "no captured setting" in XD.route_of(TA.settings_of(True, {"VLLM_ALLREDUCE_USE_FLASHINFER": "0"}), 4, N, W)["why"]


def test_route_of_over_several_requests_lists_the_builds_one_call_per_request() -> None:
    """The Build binds a launch over several requests as one call per request on the launch's route (`route_binding(.., rows)`); the
    fused norm is per token, one spec whatever the requests."""
    s, N = TA.settings_of(False, {}), 6144
    pre = XD.route_of(s, 256, N, W, spans=[(0, 200), (200, 56)])
    assert pre["family"] == "AllReduceMultimem_v1" and pre["statics"]["SIZE"] == 256 * N
    assert [(c["lo"], c["rows"], c["statics"]["SIZE"]) for c in pre["calls"]] == [(0, 200, 200 * N), (200, 56, 56 * N)]
    assert XC.spec_key(pre["calls"][1]["spec"]) == XC.spec_key(bind(TA.AllReduceMultimem, WORLD=8, SIZE=56 * N, TREES=(TA.GLM53_H200_TREE,) * 8,
                                                                    FTZ=False).id)
    dec = XD.route_of(s, 3, N, W, spans=[(0, 1), (1, 2)])
    assert [c["statics"] for c in dec["calls"]] == [{"WORLD": 8, "N": N, "SHOT": 1}, {"WORLD": 8, "N": 2 * N, "SHOT": 1}]
    assert "calls" not in XD.route_of(s, 256, N, W, spans=[(0, 256)])
    assert "calls" not in XD.route_of(s, 4, N, W, types.SimpleNamespace(variance_epsilon=1e-5), spans=[(0, 1), (1, 3)])


def test_a_route_that_reduces_by_offset_over_several_requests_is_named(monkeypatch) -> None:
    custom = next(c for c in TA.CAPTURED if c.definition == "AllReduceCustom_v1" and c.statics["STAGE"] == 2)
    monkeypatch.setattr(TA, "covered", lambda *a, **k: custom)
    s = TA.settings_of(False, {})
    assert "calls" not in XD.route_of(s, 64, 6144, W, spans=[(0, 64)])
    assert "reduces an element by its offset" in XD.route_of(s, 64, 6144, W, spans=[(0, 60), (60, 4)])["why"]


def test_call_spans_are_the_collectors_request_rows_in_launch_order() -> None:
    rows = {7: [("req-b", 0, 3), ("req-a", 3, 4)], 8: [("a", 0, 2), ("b", 3, 4)]}
    assert XD.call_spans(rows, 7, 4, 2) == ([(0, 3), (3, 1)], None)
    assert XD.call_spans(rows, 9, 1, 1) == (None, None)
    assert "no request rows recorded at step 9" in XD.call_spans(rows, 9, 2, 2)[1]
    assert "do not tile its 4 rows" in XD.call_spans(rows, 8, 4, 2)[1]
    assert "do not tile its 5 rows" in XD.call_spans(rows, 7, 5, 2)[1]


def test_the_programs_must_bind_every_request_call() -> None:
    r = XD.route_of(TA.settings_of(False, {}), 3, 6144, W, spans=[(0, 1), (1, 2)])
    known = {XC.spec_key(c["spec"]) for c in r["calls"]}
    assert XC.spec_key(r["spec"]) not in known                         # the launch-wide call is not what the Programs bind
    f = XD._routed({"site": "model.embed_tokens"}, r, None, None, None, known)
    assert f["program_binds"] is True and f["calls"] == r["calls"] and "unbound_spec" not in f
    g = XD._routed({"site": "model.embed_tokens"}, r, None, None, None, known - {XC.spec_key(r["calls"][1]["spec"])})
    assert g["program_binds"] is False and g["unbound_spec"] == r["calls"][1]["spec"]


def test_a_routed_family_no_phase_stratum_drew_gets_its_own_stratum() -> None:
    """Mixed arrivals leave prefill-sized launches among the decode-phase steps; a family a site takes that its phase strata missed
    (the fused norm at a few layers of the GLM-5.3 B8 Commit) is drawn on its own, and the phase strata draw what they drew."""
    site, sizes = "model.layers.38.input_layernorm", {0: 256, 1: 120, 2: 100, 3: 90, 4: 3, 5: 80}
    layouts = {s: [(f"{site}/part_rank0", (m, NC))] for s, m in sizes.items()}

    def fam(_site, kind, sz):
        return "F" if sz[0] <= 42 else "U"

    extra = 0
    for k in range(32):
        plain = XC.pick_strata(layouts, 0, bytes([k]), world=W)
        routed = XC.pick_strata(layouts, 0, bytes([k]), world=W, call_family=fam)
        assert routed["picks"][:len(plain["picks"])] == plain["picks"]
        assert {fam(site, "all_reduce", (sizes[p["step"]], NC)) for p in routed["picks"]} == {"F", "U"}
        new = [key for key in routed["strata"] if key not in plain["strata"]]
        assert new in ([], [f"{site}|all_reduce|route=F"])
        extra += bool(new)
    assert extra


def test_the_rank_dump_binds_each_request_call_and_draws_every_routed_family(tmp_path) -> None:
    N, site = 6144, "model.layers.0.input_layernorm"
    s = TA.settings_of(False, {})
    sizes = {0: 256, 1: 100, 2: 90, 3: 80, 4: 3}
    step_rows = {k: [("a", 0, 200 if k == 0 else 1), ("b", 200 if k == 0 else 1, m)] for k, m in sizes.items()}
    norm = types.SimpleNamespace(variance_epsilon=1e-5)
    specs = {c["spec"] for k, m in sizes.items() if m > 42 for c in XD.route_of(s, m, N, W, spans=XD.call_spans(step_rows, k, m, 2)[0])["calls"]}
    specs.add(XD.route_of(s, 3, N, W, norm)["spec"])
    com = types.SimpleNamespace(_layouts={k: [types.SimpleNamespace(name=f"{site}/part_rank0", shape=(m, N), dtype="torch.bfloat16")]
                                          for k, m in sizes.items()}, step_rows=step_rows)
    st = {"com": com, "rank": 0, "world": W, "rc": None, "n_requests": 2, "xrank_program_specs": sorted(specs),
          "eng": types.SimpleNamespace(model=types.SimpleNamespace(named_modules=lambda: [(site, norm)])),
          "tps": types.SimpleNamespace(far=types.SimpleNamespace(order=[site], paths=()))}
    for k in range(8):
        out = XD.dump(st, settings=s, out_dir=str(tmp_path), seed_hex=f"{k:016x}", pair=0)
        picks = torch.load(out["file"], weights_only=False)["picks"]
        assert all(p["program_binds"] is True for p in picks) and set(out["routed_specs"]) <= specs
        assert {p["family"] for p in picks} == {"AllReduceMultimem_v1", XC.FUSED_FAMILY}
        assert all(len(p.get("calls") or ()) == (2 if p["family"] == "AllReduceMultimem_v1" else 0) for p in picks)
    del step_rows[0]
    picks = torch.load(XD.dump(st, settings=s, out_dir=str(tmp_path), seed_hex="00" * 8, pair=0)["file"], weights_only=False)["picks"]
    assert "no request rows recorded at step 0" in next(p for p in picks if p["step"] == 0)["route_why"]


def test_route_of_names_nccl_s_tree_at_tp4_on_the_rtx_pro_6000_over_shm() -> None:
    """Batch-invariant pynccl over SHM on 4 RTX PRO 6000 is the tree down (0, 1, 2, 3); another GPU, P2P on or a world no capture
    covers keeps the AllReduce_v2 chain (None), and world 8 under batch invariance is named uncovered."""
    s, N = TA.settings_of(True, {"VLLM_BATCH_INVARIANT": "1", "NCCL_P2P_DISABLE": "1"}), 8192
    for M in (1, 256):
        r = XD.route_of(s, M, N, W4, device=TA.RTX_PRO_6000)
        assert r == {"spec": f"AllReduceNcclTree_v1{{WORLD=4,SIZE={M * N},CHAIN=(0,1,2,3)}}", "family": XC.NCCL_TREE_FAMILY,
                     "statics": {"WORLD": 4, "SIZE": M * N, "CHAIN": (0, 1, 2, 3)}, "fused": False}, r
    assert XD.route_of(s, 1, N, W4, device=((12, 0), 188))["family"] == XC.NCCL_TREE_FAMILY
    assert XD.route_of(s, 1, N, W4, types.SimpleNamespace(variance_epsilon=1e-5), device=TA.RTX_PRO_6000)["fused"] is False
    p2p = TA.settings_of(True, {"VLLM_BATCH_INVARIANT": "1"})
    for args in ((s, 1, N, W4), (s, 1, N, W4, None, ("9.0", 132)), (p2p, 1, N, W4, None, TA.RTX_PRO_6000), (s, 1, 4096, W4, None, TA.RTX_PRO_6000),
                 (s, 257, N, W4, None, TA.RTX_PRO_6000), (s, 1, N, 3, None, TA.RTX_PRO_6000)):
        assert XD.route_of(*args) is None, args
    assert "world=8" in XD.route_of(s, 1, N, W, device=TA.RTX_PRO_6000)["why"]


def test_the_residual_is_the_previous_calls_new_residual_and_the_embedding_first() -> None:
    order = ["model.layers.0.post_attention_layernorm", "model.layers.1.input_layernorm", "model.norm"]
    assert XD.residual_member(order, order[0], "model.embed_tokens") == "model.embed_tokens"
    assert XD.residual_member(order, order[2], "model.embed_tokens") == "model.layers.1.input_layernorm/1"
    assert XD.residual_member(order, "model.layers.9.input_layernorm", "e") is None and XD.residual_member(None, order[0], "e") is None


def test_a_call_that_took_the_other_path_than_its_route_is_named() -> None:
    order = ["a", "b"]
    fused, unfused = {"fused": True}, {"fused": False}
    assert XD.path_disagrees(fused, "FF", order, "b") is None and XD.path_disagrees(unfused, "NN", order, "a") is None
    assert "took path N" in XD.path_disagrees(fused, "FN", order, "b")["why"]
    assert "took path F" in XD.path_disagrees(unfused, "FN", order, "a")["why"]
    assert XD.path_disagrees(fused, None, order, "a") is None and XD.path_disagrees(fused, "F", order, "b") is None


def test_the_picks_routed_members() -> None:
    norm, lin = types.SimpleNamespace(variance_epsilon=1e-5), object()
    s, order = TA.settings_of(False, {}), ["model.layers.0.post_attention_layernorm", "model.layers.1.input_layernorm"]
    fr = XD.route_of(s, 4, 6144, W, norm)
    known = {XC.spec_key(fr["spec"])}
    pk = {"site": order[1], "recv_candidates": [f"{order[1]}/out"]}
    f = XD._routed(pk, fr, norm, order, "model.embed_tokens", known)
    assert f["recv_candidates"] == [f"{order[1]}/0"] and f["recv1_member"] == f"{order[1]}/1" and f["res_member"] == f"{order[0]}/1"
    assert f["program_binds"] is True
    ur = XD.route_of(s, 256, 6144, W, norm)
    u = XD._routed({"site": order[1]}, ur, norm, order, None, known)
    assert u["recv_candidates"] == [f"{order[1]}/all_reduce"] and u["program_binds"] is False
    o = XD._routed({"site": "model.layers.0.self_attn.o_proj"}, ur, lin, order, None, None)
    assert "recv_candidates" not in o and o["program_binds"] is None


def test_program_specs_are_the_routed_rows() -> None:
    progs = {"a": {"rows": [{"spec": "AllReduceMnnvl_v1{WORLD=8,N=96,SHOT=1}"}, {"spec": "Linear_v1{K=2}"}]},
             "b": {"rows": [{"spec": "AllReduceRmsNormMnnvl_v1{WORLD=8,N=256,SHOT=1,CLUSTER=1,EPS=1e-05,FMA=True}"}]}}
    assert XD.program_specs(progs) == ["AllReduceMnnvl_v1{WORLD=8,N=96,SHOT=1}",
                                       "AllReduceRmsNormMnnvl_v1{WORLD=8,N=256,SHOT=1,CLUSTER=1,EPS=1e-05,FMA=True}"]


def test_coverage_holds_a_program_site_to_a_stratum_of_its_family() -> None:
    res = {"result": True, "by_stratum": {"model.layers.0.post_attention_layernorm|all_reduce|decode": {"families": [XC.FUSED_FAMILY]}}}
    meta = [{"rank": 0, "unreplayable_sites": {XC.FUSED_FAMILY: ["model.layers.0"], "AllReduceMultimem_v1": ["model.layers.0"]}}]
    cov = XC.coverage_of_unreplayable(res, meta)
    assert cov["families"][XC.FUSED_FAMILY]["missing"] == [] and cov["families"]["AllReduceMultimem_v1"]["missing"] == ["model.layers.0"]
    assert cov["complete"] is False and res["result"] is False
