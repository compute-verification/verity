"""The VU exporter (`verity_vllm.pipeline.vu_export`) on the replay's honest test Programs: the sets it writes are grouped by
subcircuit, content-addressed and re-evaluate under their relation; a VU that does not replay bit-exactly is never exported."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

import numpy as np
import pytest
from verity.commitments.randomness import derive

from tests.check.test_sampled_replay import (
    H,
    Honest,
    I,
    V,
    _bf16,
    _program,
    _programs,
    _weights,
)
from verity_vllm.check.replay import evaluate as EV
from verity_vllm.pipeline import vu_export as VX
from verity_vllm.pipeline import vu_store as VS

ROOT = bytes(range(32))


def _world(**kw):
    rng = np.random.default_rng(29)
    inst, woff = _program(**kw)
    weights = _bf16(rng, max(hi for _lo, hi in woff.values()), 0.5)
    for f in ("model.model.layers.0.input_layernorm.weight", "model.model.norm.weight"):
        weights[woff[f][0]:woff[f][1]] = 0x3F80
    if "model.model.layers.0.self_attn.rotary_emb.cos_sin_cache" in woff:
        lo, hi = woff["model.model.layers.0.self_attn.rotary_emb.cos_sin_cache"]
        weights[lo:hi] = (np.asarray(rng.uniform(-1, 1, hi - lo), dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    return Honest(inst, woff, weights, {"r0": [1, 2], "r1": [5, 3]})


@pytest.fixture(scope="module")
def attn_world():
    return _world(attn=True)


def _export(world, out, *, limits=None, mutate=None, root=ROOT):
    bmap, committed = world.bmap_and_reader(mutate=mutate)
    return VS.export_vus(_programs(world), bmap, committed, _weights(world), out_dir=out, run_root=root,
                         provenance={"row": "test", "model": "tiny"}, limits=limits or VX.Limits(per_family=64, workers=1))


def test_sets_are_grouped_by_subcircuit_and_reevaluate(attn_world, tmp_path):
    s = _export(attn_world, tmp_path)
    names = {x["set"] for x in s["sets"]}
    assert {f"gemm-coordinate-ampere-bf16-k{H}", f"gemm-coordinate-ampere-bf16-k{I}", "attention-head-fa2-d16-bn16", "rope-head-d16",
            f"silumul-v1-i{I}", f"tokenselect-v1-v{V}"} <= names, names
    assert not any(n.startswith("embedding") for n in names)
    assert any(n.startswith(f"rmsnormtriton-v1-n{H}-") for n in names) and any(n.startswith(f"rmsnormfusedcuda-v2-n{H}-") for n in names)
    for fam, c in s["by_family"].items():
        assert c["drawn"] == c["replayed_equal"] > 0 and not any(k.startswith("not_exported") for k in c), (fam, c)
    assert "Embedding_v1" not in s["by_family"] and s["population"]["pool_by_family"]["Embedding_v1"] > 0
    assert s["schema"] == "vllm-vu-export/v1" and list(s["population"]["not_exported_families"]) == ["Embedding_v1"]
    v = VX.verify(tmp_path)
    assert v["ok"] and all(r["n"] > 0 for r in v["sets"]), v
    man = json.loads((tmp_path / "sets" / f"gemm-coordinate-ampere-bf16-k{H}" / "manifest.json").read_text())
    assert man["schema"] == VX.SET_SCHEMA and man["relation"]["subcircuit"] == f"GemmCoordinate<{H}>"
    assert man["ports"]["x"]["words"] == H and man["ports"]["w"]["words"] == H and man["ports"]["y"]["words"] == 1
    assert man["provenance"]["run_root"] == ROOT.hex() and man["provenance"]["row"] == "test"
    att = json.loads((tmp_path / "sets" / "attention-head-fa2-d16-bn16" / "manifest.json").read_text())
    assert att["ports"]["k"]["words"] is None and att["ports"]["k"]["offsets"] == "k.offsets.u64"   # T varies with the position


def test_the_export_is_fixed_by_the_root_and_content_addressed(attn_world, tmp_path):
    a = _export(attn_world, tmp_path / "a")
    b = _export(attn_world, tmp_path / "b")
    assert [(x["set"], x["content_digest"]) for x in a["sets"]] == [(x["set"], x["content_digest"]) for x in b["sets"]]
    c = _export(attn_world, tmp_path / "c", limits=VX.Limits(per_family=2, gemm_coords_per_row=2), root=bytes(32))
    assert {x["set"]: x["content_digest"] for x in a["sets"]} != {x["set"]: x["content_digest"] for x in c["sets"]}


def test_a_gemm_coordinate_is_the_row_word(attn_world, tmp_path):
    _export(attn_world, tmp_path, limits=VX.Limits(per_family=64, workers=1, gemm_coords_per_row=1000))
    d = tmp_path / "sets" / f"gemm-coordinate-ampere-bf16-k{H}"
    man = json.loads((d / "manifest.json").read_text())
    x: np.ndarray = np.frombuffer((d / "x.u16").read_bytes(), "<u2").reshape(man["n"], H)
    w: np.ndarray = np.frombuffer((d / "w.u16").read_bytes(), "<u2").reshape(man["n"], H)
    y: np.ndarray = np.frombuffer((d / "y.u16").read_bytes(), "<u2")
    idx = json.loads((d / "index.json").read_text())["rows"]
    rows: dict[tuple, dict] = {}
    for (_i, vu, *_ref, source, sub), xi, wi, yi in zip(idx, x, w, y):
        r = rows.setdefault((vu, source, sub["N"]), {"x": xi, "c": {}})
        assert np.array_equal(r["x"], xi)                     # one activation row per Gemm row
        r["c"][sub["n"]] = (wi, yi)
    assert rows and all(len(r["c"]) == N for (_v, _s, N), r in rows.items())
    for (_v, _s, N), r in rows.items():                       # every row re-derived by the registered Gemm kernel from its coordinates
        W = np.stack([r["c"][j][0] for j in range(N)])
        got = EV.evaluate("Gemm_v1", {"K": H, "N": N}, [[np.asarray(r["x"], dtype=np.uint16)], []], [None, W.reshape(-1)])["out"]
        assert got.tolist() == [int(r["c"][j][1]) for j in range(N)]


def test_a_vu_that_does_not_replay_is_not_exported(attn_world, tmp_path):
    def flip(store, entries):
        for (step, name), b in list(store.items()):
            if "down_proj" in name:
                a = bytearray(b)
                a[0] ^= 0x01
                store[(step, name)] = bytes(a)

    s = _export(attn_world, tmp_path, mutate=flip)
    g = s["by_family"]["Gemm_v1"]
    assert g.get("not_exported_mismatch", 0) > 0 and g["replayed_equal"] < g["drawn"]
    lines = [json.loads(x) for x in (tmp_path / "vus.jsonl").read_text().splitlines()]
    assert all(not e.get("stored") for e in lines if e["result"] is not True)
    assert VX.verify(tmp_path)["ok"]


def test_a_tampered_set_fails_verify(attn_world, tmp_path):
    _export(attn_world, tmp_path)
    f = tmp_path / "sets" / "rope-head-d16" / "out.u16"
    b = bytearray(f.read_bytes())
    b[0] ^= 1
    f.write_bytes(bytes(b))
    res = {r["set"]: r for r in VX.verify(tmp_path)["sets"]}
    assert not res["rope-head-d16"]["ok"] and "sha256" in res["rope-head-d16"]["why"]


def test_decomposition_mismatch_is_raised_not_written():
    rng = np.random.default_rng(3)
    x, w = _bf16(rng, H), _bf16(rng, 4 * H)
    out = EV.evaluate("Gemm_v1", {"K": H, "N": 4}, [[x], []], [None, w])
    bad = {"0": out["0"].copy()}
    bad["0"][:] ^= 1
    with pytest.raises(VX.DecompositionMismatch):
        VX.decompose("Gemm_v1", {"K": H, "N": 4}, [x, w], bad, lambda n, k: list(range(min(n, k))), VX.Limits(gemm_coords_per_row=4))
    assert len(VX.decompose("Gemm_v1", {"K": H, "N": 4}, [x, w], out, lambda n, k: list(range(min(n, k))), VX.Limits(gemm_coords_per_row=4))) == 4


def test_limits_from_dir(tmp_path):
    assert VX.Limits.from_dir(tmp_path) == VX.Limits()
    (tmp_path / "limits.json").write_text(json.dumps({"per_family": 7, "per_family_override": {"Gemm_v1": 3}}))
    lim = VX.Limits.from_dir(tmp_path)
    assert lim.for_family("Gemm_v1") == 3 and lim.for_family("SiluMul_v1") == 7


class _Eng:
    cp = {"repo": "tiny/model", "revision": "abc"}


def test_after_replay_exports_only_a_passing_pair_zero(attn_world, tmp_path):
    bmap, committed = attn_world.bmap_and_reader()
    args: tuple[Any, ...] = (_programs(attn_world), bmap, committed, _weights(attn_world), {}, None, {}, {}, {"r0": "d" * 64}, {"git_head": "f00"}, _Eng())
    assert VS.after_replay(None, {"result": True}, 0, ROOT, *args) is None
    assert VS.after_replay(str(tmp_path), {"result": None}, 0, ROOT, *args) is None
    assert VS.after_replay(str(tmp_path), {"result": True}, 1, ROOT, *args) is None and not (tmp_path / "export.json").exists()
    (tmp_path / "provenance.json").write_text(json.dumps({"row": 101, "research_run": "r-test"}))
    (tmp_path / "limits.json").write_text(json.dumps({"per_family": 3, "workers": 1}))
    s = VS.after_replay(str(tmp_path), {"result": True}, 0, ROOT, *args)
    p = s["provenance"]
    assert p["row"] == 101 and p["research_run"] == "r-test" and p["model"] == "tiny/model" and p["versions"] == {"git_head": "f00"}
    assert all(c["drawn"] <= 3 for c in s["by_family"].values()) and VX.verify(tmp_path)["ok"]


@pytest.mark.slow
def test_a_spent_time_budget_stops_the_draw_by_name(attn_world, tmp_path):
    """The forked pool's check of the budget (`Limits.workers` is 8 unless set); `test_vu_store_budget` holds the serial one's."""
    s = _export(attn_world, tmp_path, limits=VX.Limits(per_family=64, max_seconds=0.0))
    assert s["sets"] == [] and s["by_family"]
    assert all(c.get("drawn", 0) == 0 and c["not_drawn_time_budget"] > 0 for c in s["by_family"].values())


def test_families_are_interleaved_in_the_draw(attn_world):
    bmap, committed = attn_world.bmap_and_reader()
    from verity_vllm.check.replay.index import ProgramIndex
    from verity_vllm.check.replay.population import population
    pi = ProgramIndex(attn_world.inst)
    pool = population(pi, "r0")[0]
    fams = [v["family"] for v in VX.select(pool, VX.Limits(per_family=64), VX.selection_key(ROOT))]
    n = len(set(fams))
    assert len(set(fams[:n])) == n                         # the first round holds one VU of every family


@pytest.mark.slow
def test_the_forked_pool_writes_the_serial_sets(attn_world, tmp_path):
    a = _export(attn_world, tmp_path / "serial")
    b = _export(attn_world, tmp_path / "forked", limits=VX.Limits(per_family=64, workers=2))
    assert VS._limits_for(tmp_path, {"evaluator": {"retention": "device"}}, 0.9).workers == 1 and VS._limits_for(tmp_path, {}, 0.9).workers == 8
    assert VS._limits_for(tmp_path, {}, 0.1).workers == 1                  # memory-tight: serial, never a pool that could OOM the Commit
    h = VS.memory_headroom()
    assert h is None or 0.0 <= h <= 1.0
    assert [(x["set"], x["n"], x["content_digest"]) for x in a["sets"]] == [(x["set"], x["n"], x["content_digest"]) for x in b["sets"]]


def test_a_cap_below_the_stratum_count_reaches_every_shape():
    pool = [{"family": "Gemm_v1", "stratum": f"Gemm_v1{{K={k},N=8}}|r0|rank0|mid|m{j}|w", "i": 100 * k + j, "spec": f"Gemm_v1{{K={k},N=8}}"}
            for k in (1536, 576) for j in range(50)]
    got = VX.select(pool, VX.Limits(per_family=20), VX.selection_key(ROOT))
    ks = {v["spec"] for v in got}
    assert len(got) == 20 and ks == {"Gemm_v1{K=1536,N=8}", "Gemm_v1{K=576,N=8}"}, ks


def test_the_store_holds_the_units_and_a_query_cuts_the_sets(attn_world, tmp_path):
    s = _export(attn_world, tmp_path, limits=VX.Limits(per_family=64, workers=1, gemm_coords_per_row=2))
    root = tmp_path / "store"
    assert VS.verify_store(root)["ok"] and s["store"]["values"] > 0
    run = json.loads((root / "run.json").read_text())
    assert run["schema"] == VS.STORE_SCHEMA and run["provenance"]["weights"]["rows_per_gemm"] == 2
    units = [json.loads(x) for p in root.glob("programs/*/units.jsonl") for x in p.read_text().splitlines()]
    gemm_ev = [ev for u in units for ev in u.get("evaluations") or [] if ev["family"] == "Gemm_v1"]
    assert gemm_ev and all(len(ev["args"][1]["sample"]["rows"]) == 2 and ev["args"][1]["weight"]["sha256"] for ev in gemm_ev)
    default = {x["set"]: x["content_digest"] for x in s["sets"]}
    only, fam = VS.extract(root, tmp_path / "gemm", where=f"Gemm_v1{{K={H},*}}")
    assert {x["set"] for x in only} == {f"gemm-coordinate-ampere-bf16-k{H}"} and set(fam) == {"Gemm_v1"}
    assert only[0]["content_digest"] == default[only[0]["set"]]
    mod, _ = VS.extract(root, tmp_path / "mlp", module="model.layers.0.mlp")
    assert mod and all(x["set"].startswith(("gemm-coordinate", "silumul")) for x in mod)
    full, _ = VS.extract(root, tmp_path / "full", where="Gemm_v1", weights=_weights(attn_world),
                         limits=VX.Limits(per_family=64, workers=1, gemm_coords_per_row=1000))
    assert sum(x["n"] for x in full) > sum(x["n"] for x in only)          # the checkpoint fallback cuts any row
    assert VX.verify(tmp_path)["ok"]


def _e4m3(rng, shape):
    b = rng.integers(0, 256, size=shape, dtype=np.int64)
    b[(b & 0x7F) == 0x7F] &= ~0x08                                          # no NaN codes
    return b.astype(np.uint8)


def _store_one(tmp_path, fam, st, ops, out, refs, limits):
    """One evaluation stored as `vu_store` stores it (a unit of one evaluation), then the default extraction into `<tmp>/sets`."""
    key = VX.selection_key(ROOT)
    blobs: dict = {}
    ev = VS._evaluation((key, ("row", "r0", 0, 0)), fam, st, ops, out, refs, blobs, limits)
    root = tmp_path / "store"
    (root / "values").mkdir(parents=True)
    (root / "programs" / "p").mkdir(parents=True)
    for h, (dt, b) in blobs.items():
        (root / "values" / f"{h}.{dt}").write_bytes(b)
    unit = {"vu": 0, "request": "r0", "i": 0, "engine_step": 0, "op_path": "model.layers.0.mlp", "row": 0, "phase": "decode", "family": fam,
            "evaluations": [ev]}
    (root / "programs" / "p" / "units.jsonl").write_text(json.dumps(unit) + "\n")
    (root / "run.json").write_text(json.dumps({"schema": VS.STORE_SCHEMA, "provenance": {"run_root": ROOT.hex()}, "limits": dataclasses.asdict(limits)}))
    sets, _ = VS.extract(root, tmp_path / "sets")
    return root, ev, sets


@pytest.mark.parametrize("N", [256, 192])
def test_an_fp8_block_coordinate_is_the_row_word(tmp_path, N):
    """N = 192 ends mid-block: its last 64 coordinates read the weight-scale row cdiv(N, G) - 1."""
    rng = np.random.default_rng(7)
    K, G = 256, 128

    def f32(a) -> np.ndarray:
        return np.asarray(a, dtype=np.float32).view(np.uint32)

    xq, w, sx, ws = _e4m3(rng, K), _e4m3(rng, (N, K)), f32(rng.uniform(1e-3, 0.1, K // G)), f32(rng.uniform(1e-3, 0.1, (-(-N // G), K // G)))
    st, ops = {"K": K, "N": N, "G": G}, [xq, sx, w.reshape(-1), ws.reshape(-1)]
    out = EV.evaluate("ScaledMmFp8BlockSharedScale_v1", st, [[xq], [sx], [], []], [None, None, ops[2], ops[3]])
    insts = VX.decompose("ScaledMmFp8BlockSharedScale_v1", st, ops, out, lambda n, k: list(range(0, n, n // k)), VX.Limits(gemm_coords_per_row=8))
    assert len(insts) == 8 and {i.set_name for i in insts} == {f"gemm-coordinate-fp8-block{G}-hopper-e4m3-k{K}"}
    for i in insts:
        n = i.sub["n"]
        assert np.array_equal(i.ports["w"], w[n]) and np.array_equal(i.ports["sw"], ws[n // G]) and int(i.ports["y"][0]) == int(out["0"][n])
    with pytest.raises(VX.DecompositionMismatch):
        VX.decompose("ScaledMmFp8BlockSharedScale_v1", st, ops, {"0": out["0"] ^ 1}, lambda n, k: [0], VX.Limits())
    lim = VX.Limits(gemm_coords_per_row=5)
    _root, ev, sets = _store_one(tmp_path, "ScaledMmFp8BlockSharedScale_v1", st, ops, out, [None, None, {"field": "w", "range": [0, N * K]},
                                                                                   {"field": "ws", "range": [0, ws.size]}], lim)
    wa = ev["args"][2]
    assert wa["weight"]["dtype"] == "u8" and wa["weight"]["shape"] == [N, K] and len(wa["sample"]["rows"]) == 5 and "sample" not in ev["args"][3]
    assert [(s["set"], s["n"]) for s in sets] == [(f"gemm-coordinate-fp8-block{G}-hopper-e4m3-k{K}", 5)] and VX.verify(tmp_path)["ok"]
    full, _ = VS.extract(tmp_path / "store", tmp_path / "ckpt", weights=lambda f, lo, hi: ops[2] if f == "w" else None)
    assert full[0]["content_digest"] == sets[0]["content_digest"]                # the checkpoint's e4m3 rows cut the same coordinates


def test_moe_expert_coordinates_are_the_row_words(tmp_path):
    rng = np.random.default_rng(11)
    E, K, N = 3, 32, 5
    x, slab = _bf16(rng, K), _bf16(rng, E * N * K)
    e: np.ndarray = np.asarray([1], dtype=np.uint32)
    g: np.ndarray = np.asarray([0.375], dtype=np.float32).view(np.uint32)
    st, pick, lim = {"E": E, "K": K, "N": N}, (lambda n, k: list(range(min(n, k)))), VX.Limits(gemm_coords_per_row=N)
    up = EV.evaluate("MoeExpertGemm_v1", st, [[x], [e], []], [None, None, slab])
    down = EV.evaluate("MoeExpertGemmW_v1", st, [[x], [e], [g], []], [None, None, None, slab])
    a = VX.decompose("MoeExpertGemm_v1", st, [x, e, slab], up, pick, lim)
    b = VX.decompose("MoeExpertGemmW_v1", st, [x, e, g, slab], down, pick, lim)
    W = slab.reshape(E, N, K)[1]
    assert [i.sub["n"] for i in a] == [i.sub["n"] for i in b] == list(range(N))
    assert all(np.array_equal(i.ports["w"], W[i.sub["n"]]) and i.sub["expert"] == 1 for i in a + b)
    assert [int(i.ports["y"][0]) for i in a] == up["up"].tolist() and [int(i.ports["y"][0]) for i in b] == down["down"].tolist()
    assert {i.set_name for i in b} == {f"moe-expert-routed-gemm-coordinate-ampere-bf16-k{K}"} and set(b[0].ports) == {"x", "w", "g", "y"}
    root, ev, sets = _store_one(tmp_path, "MoeExpertGemmW_v1", st, [x, e, g, slab], down, [None, None, None, {"field": "w2", "range": [0, slab.size]}],
                                VX.Limits(gemm_coords_per_row=3))
    assert ev["args"][3]["weight"]["expert"] == 1 and ev["args"][3]["weight"]["shape"] == [N, K] and len(ev["args"][3]["sample"]["rows"]) == 3
    assert VS.verify_store(root)["ok"] and [s["n"] for s in sets] == [3] and VX.verify(tmp_path)["ok"]


AMPERE, HOPPER, FA2, FA3 = "AmpereBF16TcDot16_v2", "HopperBF16WgmmaDot16_v1", "Fa2InvSum_v1", "Fa3InvSum_v1"
GEMM_RULE = "y = f2fp_bf16(ascending k16 chain of DOT over (x, w) from +0)"
ROUTED_RULE = "y = f2fp_bf16(FMUL(ascending k16 chain of DOT over (x, w) from +0, g))"
FP8_RULE = "per K-tile kb ascending: temp = G/32 DOT steps from +0 over (x, w); acc = FFMA(temp, FMUL(sx[kb], sw[kb]), acc) from +0; y = cvt.rn.bf16(acc)"
HEAD_RULE = "one query at position T-1 over keys 0..T-1 of its KV head, ceil(T/BN) key blocks last -> first (the Definition's head)"
ROPE_RULE = "out[i] = bf16_rn(RN32(x[i]*c[i] - x[i+R]*s[i])), out[i+R] = bf16_rn(RN32(x[i+R]*c[i] + x[i]*s[i])), cs = [c | s], R = D/2"
COORDS = [{"n": n, "N": 5} for n in range(5)]
ROUTED = [{**c, "expert": 2, "E": 3} for c in COORDS]
HEADS = [{"head": h, "kv_head": h // 2, "T": 3, "NH": 4, "KVH": 2} for h in range(4)]
WORDS = "0x[0-9a-f]{4} != row word 0x[0-9a-f]{4}$"
#: family -> (the set, the relation beside its `definition`, the ports with the output last, every instance's `sub`, the mismatch it names)
CUTS: dict[str, tuple[str, dict[str, Any], str, list[dict[str, int]], str]] = {
    "Gemm_v1": ("gemm-coordinate-ampere-bf16-k32", {"subcircuit": "GemmCoordinate<32>", "dot": AMPERE, "statics": {"K": 32}, "rule": GEMM_RULE},
                "x w y", COORDS, f"coordinate 0: {WORDS}"),
    "Gemm_v2": ("gemm-coordinate-hopper-bf16-wgmma-k32", {"subcircuit": "GemmCoordinate<32>", "dot": HOPPER, "statics": {"K": 32}, "rule": GEMM_RULE},
                "x w y", COORDS, f"coordinate 0: {WORDS}"),
    "ScaledMmFp8BlockSharedScale_v1": ("gemm-coordinate-fp8-block32-hopper-e4m3-k64",
                                       {"subcircuit": "ScaledMmFp8BlockCoordinate<64,32>", "dot": "HopperE4m3QgmmaDot32_v1", "statics": {"K": 64, "G": 32},
                                        "rule": FP8_RULE}, "x sx w sw y", COORDS, f"coordinate 0: {WORDS}"),
    "MoeExpertGemm_v1": ("moe-expert-gemm-coordinate-ampere-bf16-k32",
                         {"subcircuit": "GemmCoordinate<32>", "dot": AMPERE, "statics": {"K": 32}, "rule": GEMM_RULE},
                         "x w y", ROUTED, f"expert 2 coordinate 0: {WORDS}"),
    "MoeExpertGemm_v2": ("moe-expert-gemm-coordinate-hopper-bf16-wgmma-k32",
                         {"subcircuit": "GemmCoordinate<32>", "dot": HOPPER, "statics": {"K": 32}, "rule": GEMM_RULE},
                         "x w y", ROUTED, f"expert 2 coordinate 0: {WORDS}"),
    "MoeExpertGemmW_v1": ("moe-expert-routed-gemm-coordinate-ampere-bf16-k32",
                          {"subcircuit": "RoutedGemmCoordinate<32>", "dot": AMPERE, "statics": {"K": 32}, "rule": ROUTED_RULE},
                          "x w g y", ROUTED, f"expert 2 coordinate 0: {WORDS}"),
    "MoeExpertGemmW_v2": ("moe-expert-routed-gemm-coordinate-hopper-bf16-wgmma-k32",
                          {"subcircuit": "RoutedGemmCoordinate<32>", "dot": HOPPER, "statics": {"K": 32}, "rule": ROUTED_RULE},
                          "x w g y", ROUTED, f"expert 2 coordinate 0: {WORDS}"),
    "Attention_v3": ("attention-head-fa2-d16-bn16",
                     {"subcircuit": "AttentionHead<T,16>", "dot": AMPERE, "inv": FA2, "statics": {"D": 16, "BN": 16}, "rule": HEAD_RULE},
                     "q k v out", HEADS, "head 0: 16 words differ from the row$"),
    "Attention_v2": ("attention-head-hopper-bf16-wgmma-fa3invsum_v1-d16-bn16",
                     {"subcircuit": "AttentionHead<T,16>", "dot": HOPPER, "inv": FA3, "statics": {"D": 16, "BN": 16}, "rule": HEAD_RULE},
                     "q k v out", HEADS, "head 0: 16 words differ from the row$"),
    "RoPE_v1": ("rope-head-d16", {"subcircuit": "RoPEHead<16>", "statics": {"D": 16}, "rule": ROPE_RULE},
                "x cs out", [{"head": h, "NHEADS": 2} for h in range(2)], "head 0: differs from the row$"),
}


def _every(n, k):
    return list(range(n))


def _no_pick(n, k):
    raise AssertionError("a stored sample of weight rows is its own choice of coordinates")


def _evaluate(fam, st, ops):
    return EV.evaluate(fam, st, [[a] for a in ops], [None] * len(ops))


def _plain(insts):
    return [(i.set_name, i.relation, i.sub, {k: (str(v.dtype), v.tolist()) for k, v in i.ports.items()}) for i in insts]


@pytest.fixture(scope="module")
def cut_rows():
    """One row of every family `DECOMPOSE` cuts: family -> (statics, operands)."""
    rng = np.random.default_rng(5)

    def f32(a) -> np.ndarray:
        return np.asarray(a, dtype=np.float32).view(np.uint32)

    e: np.ndarray = np.asarray([2], dtype=np.uint32)
    x, w, slab, g = _bf16(rng, 32), _bf16(rng, 5 * 32), _bf16(rng, 3 * 5 * 32), f32([0.375])
    gemm, moe, attn = {"K": 32, "N": 5}, {"E": 3, "K": 32, "N": 5}, {"T": 3, "NH": 4, "KVH": 2, "D": 16, "BN": 16}
    heads = [_bf16(rng, 4 * 16), _bf16(rng, 3 * 2 * 16), _bf16(rng, 3 * 2 * 16)]
    return {"Gemm_v1": (gemm, [x, w]), "Gemm_v2": ({**gemm, "DOT": HOPPER}, [x, w]),
            "ScaledMmFp8BlockSharedScale_v1": ({"K": 64, "N": 5, "G": 32}, [_e4m3(rng, 64), f32(rng.uniform(1e-3, 0.1, 2)), _e4m3(rng, 5 * 64),
                                                                            f32(rng.uniform(1e-3, 0.1, 2))]),
            "MoeExpertGemm_v1": (moe, [x, e, slab]), "MoeExpertGemm_v2": ({**moe, "DOT": HOPPER}, [x, e, slab]),
            "MoeExpertGemmW_v1": (moe, [x, e, g, slab]), "MoeExpertGemmW_v2": ({**moe, "DOT": HOPPER}, [x, e, g, slab]),
            "Attention_v3": (attn, heads), "Attention_v2": ({**attn, "DOT": HOPPER, "INV": FA3}, heads),
            "RoPE_v1": ({"NHEADS": 2, "D": 16}, [_bf16(rng, 2 * 16), _bf16(rng, 16)])}


@pytest.mark.parametrize("fam", sorted(VX.DECOMPOSE))
def test_a_row_is_cut_into_the_instances_of_its_subcircuit(cut_rows, fam):
    """The set, the relation and the `sub` of every instance of a row, and the template `template_of` reads off the row's spec."""
    (st, ops), (name, rel, ports, subs, miss) = cut_rows[fam], CUTS[fam]
    out = _evaluate(fam, st, ops)
    row = next(iter(out.values()))
    insts = VX.decompose(fam, st, ops, out, _every, VX.Limits())
    assert [i.sub for i in insts] == subs
    assert all(i.set_name == name and i.relation == {"definition": fam, **rel} and set(i.ports) == set(ports.split()) for i in insts)
    assert np.concatenate([i.ports[ports.split()[-1]] for i in insts]).tolist() == row.tolist()        # the instances' outputs are the row's words
    spec = f"{fam}{{{','.join(f'{k}={v}' for k, v in st.items())}}}"
    assert VX.template_of(spec) == (rel["subcircuit"].partition("<")[0], {**rel["statics"], **{k: rel[k] for k in ("dot", "inv") if k in rel}}, len(insts))
    with pytest.raises(VX.DecompositionMismatch, match=f"^{fam} {miss}"):
        VX.decompose(fam, st, ops, {"0": row ^ 1}, _every, VX.Limits())


def test_a_spec_its_template_cannot_read_is_the_row_itself():
    for spec, st in (("Gemm_v2{K=32,N=5}", {"K": 32, "N": 5}), ("Gemm_v2{K=32,N=5,DOT=NoSuchDot_v1}", {"K": 32, "N": 5, "DOT": "NoSuchDot_v1"}),
                     ("Attention_v2{T=3,NH=4,KVH=2,D=16,BN=16}", {"T": 3, "NH": 4, "KVH": 2, "D": 16, "BN": 16}), ("Gemm_v1{N=5}", {"N": 5})):
        assert VX.template_of(spec) == (spec.partition("{")[0], st, 1)


@pytest.mark.parametrize("fam", sorted(VX.WEIGHT_ROWS_OF))
def test_a_stored_sample_of_weight_rows_cuts_the_whole_operands_coordinates(cut_rows, fam):
    st, ops = cut_rows[fam]
    out = _evaluate(fam, st, ops)
    g, rows_of = VX.WEIGHT_ROWS_OF[fam]
    rows, extra = rows_of(st, ops)
    assert rows.shape == (st["N"], st["K"]) and rows.dtype == ops[g].dtype and extra == ({"expert": 2} if "E" in st else {})
    sel = [1, 3]
    stored = VX.decompose(fam, st, [*ops[:g], VX.RowSample(sel, rows[sel]), *ops[g + 1:]], out, _no_pick, VX.Limits())
    assert [i.sub["n"] for i in stored] == sel and _plain(stored) == _plain(VX.decompose(fam, st, ops, out, lambda n, k: sel, VX.Limits()))


@pytest.mark.parametrize("fam", ["Gemm_v1", "MoeExpertGemm_v1", "MoeExpertGemmW_v1"])
def test_a_non_finite_coordinate_is_not_an_instance(cut_rows, fam):
    """The families whose row kernel evaluates such a row; the other coordinate families' kernels decline it, so no tap hands it over."""
    st, ops = cut_rows[fam]
    ops = [a.copy() for a in ops]
    VX.WEIGHT_ROWS_OF[fam][1](st, ops)[0][2, 0] = 0x7F80                    # +inf in weight row 2: coordinate 2 is outside the twin's domain
    cut = VX.decompose(fam, st, ops, _evaluate(fam, st, ops), _every, VX.Limits())
    assert [i.sub["n"] for i in cut] == [0, 1, 3, 4]


def test_any_other_family_is_exported_as_the_row_itself():
    st = {"N": 4, "I": 2, "EPS": "1e-06", "MODE": "rn"}
    a: np.ndarray = np.asarray([-1, 2, -3, 4], dtype=np.int32)
    b: np.ndarray = np.asarray([0.5, -0.25], dtype=np.float32)
    o: np.ndarray = np.asarray([7, 8], dtype=np.uint16)
    [inst] = VX.decompose("Made_v1", st, [a, b], {"0": o, "out": o, "aux": o.copy()}, _no_pick, VX.Limits())
    tag = hashlib.sha256(b'{"EPS": "1e-06", "MODE": "rn"}').hexdigest()[:8]
    assert inst.set_name == f"made-v1-i2-n4-{tag}" == VX.decompose("Made_v1", dict(reversed(st.items())), [a], {"0": o}, _no_pick, VX.Limits())[0].set_name
    assert VX.decompose("Made_v1", {"N": 4, "I": 2}, [a], {"0": o}, _no_pick, VX.Limits())[0].set_name == "made-v1-i2-n4"
    assert {k: v for k, v in inst.relation.items() if k != "rule"} == {"subcircuit": "Made_v1", "definition": "Made_v1", "statics": st}
    assert inst.relation["rule"].startswith("the registered row kernel of Made_v1 (")
    assert inst.sub == {} and _plain([inst])[0][3] == {"in0": ("uint32", [0xFFFFFFFF, 2, 0xFFFFFFFD, 4]), "in1": ("uint32", [0x3F000000, 0xBE800000]),
                                                       "out.0": ("uint16", [7, 8]), "out.aux": ("uint16", [7, 8])}       # unsigned views, aliases dropped
    with pytest.raises(TypeError, match="no word type for complex128"):
        VX.decompose("Made_v1", st, [np.zeros(2, dtype=np.complex128)], {"0": o}, _no_pick, VX.Limits())
    assert VX.decompose("Embedding_v1", {"V": 4}, [a], {"0": o}, _no_pick, VX.Limits()) == []


def test_the_selection_is_keyed_by_the_run_root_alone_and_the_budget_has_its_defaults():
    assert VX.selection_key(bytearray(ROOT)) == derive(b"verity-vllm benchmark selection", "verity/vllm-vu-export/select/v1", {"run_root": ROOT})
    assert dataclasses.asdict(VX.Limits()) == {"per_family": 256, "per_family_override": {}, "gemm_coords_per_row": 32, "heads_per_row": 4,
                                               "max_instances_per_set": 8192, "max_row_words": 1 << 22, "max_seconds": 600.0,
                                               "max_population_seconds": 900.0, "workers": 8}


REF = ["r0", 3, "model.layers.0.self_attn", 0, "decode", 0]


def test_a_set_is_written_as_its_manifest_says(cut_rows, tmp_path):
    st, ops = cut_rows["RoPE_v1"]
    a, b = VX.decompose("RoPE_v1", st, ops, _evaluate("RoPE_v1", st, ops), _every, VX.Limits())
    w = VX._SetWriter(a.set_name, a.relation)
    w.add(a, [7, *REF])
    w.add(b, [7, *REF])
    summary = w.write(tmp_path / "sets", {"row": "test"})
    d = tmp_path / "sets" / "rope-head-d16"
    files = {f.name: {"sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "bytes": f.stat().st_size} for f in d.iterdir() if f.name != "manifest.json"}
    assert sorted(files) == ["cs.u16", "index.json", "out.u16", "x.u16"]
    assert (d / "x.u16").read_bytes() == np.concatenate([a.ports["x"], b.ports["x"]]).astype("<u2").tobytes()       # instance-major, little-endian
    body = {"schema": "vllm-vu-set/v1", "set": "rope-head-d16", "relation": a.relation, "n": 2,
            "ports": {k: {"dtype": "u16", "file": f"{k}.u16", "words": 16} for k in ("cs", "out", "x")},
            "files": {k: v for k, v in files.items() if k != "index.json"}}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert json.loads((d / "manifest.json").read_text()) == {
        **body, "files": files, "content_digest": digest, "provenance": {"row": "test"},
        "content_digest_rule": "sha256 of the canonical JSON of {schema, set, relation, n, ports, files without index.json}"}
    assert json.loads((d / "index.json").read_text()) == {
        "schema": "vllm-vu-set/v1/index", "set": "rope-head-d16", "columns": ["id", "vu", "request", "engine_step", "op_path", "row", "phase", "source", "sub"],
        "rows": [[0, 7, *REF, {"head": 0, "NHEADS": 2}], [1, 7, *REF, {"head": 1, "NHEADS": 2}]]}
    assert summary == {"set": "rope-head-d16", "n": 2, "bytes": sum(f["bytes"] for f in files.values()), "content_digest": digest, "relation": "RoPEHead<16>"}
    other = VX._SetWriter(a.set_name, a.relation)                           # the same instances of another VU, under another provenance
    other.add(a, [8, "r1", 4, "model.layers.1.self_attn", 1, "prefill", 1])
    other.add(b, [8, "r1", 4, "model.layers.1.self_attn", 1, "prefill", 1])
    assert other.write(tmp_path / "other", {"row": "another"})["content_digest"] == digest
    with pytest.raises(VX.DecompositionMismatch, match=r"^set rope-head-d16: ports \['cs', 'x'\] != \['cs', 'out', 'x'\]$"):
        w.add(VX.Instance(a.set_name, a.relation, {"x": a.ports["x"], "cs": a.ports["cs"]}, {}), [7, *REF])
    mixed = VX._SetWriter("mixed", {})
    for word in (np.uint16, np.uint32):
        mixed.add(VX.Instance("mixed", {}, {"p": np.zeros(2, dtype=word)}, {}), [7, *REF])
    with pytest.raises(VX.DecompositionMismatch, match=r"^set mixed port p: word types \['uint16', 'uint32'\]$"):
        mixed.write(tmp_path / "sets", {})


def test_verify_names_the_instances_that_fail_their_relation(cut_rows, tmp_path):
    st, ops = cut_rows["RoPE_v1"]
    head = VX.decompose("RoPE_v1", st, ops, _evaluate("RoPE_v1", st, ops), _every, VX.Limits())[0]
    silu: np.ndarray = _bf16(np.random.default_rng(9), 16)
    [row] = VX.decompose("SiluMul_v1", {"I": 8}, [silu], _evaluate("SiluMul_v1", {"I": 8}, [silu]), _no_pick, VX.Limits())
    (tmp_path / "sets").mkdir()
    assert VX.verify(tmp_path) == {"ok": False, "sets": []}                 # no set: nothing was verified
    for inst, port, n, good in ((head, "out", 20, (4, 9)), (row, "out.out", 3, (0, 2))):
        w = VX._SetWriter(inst.set_name, inst.relation)
        for j in range(n):
            ports = {k: v.copy() for k, v in inst.ports.items()}
            ports[port][0] ^= int(j not in good)                            # an output word of every other instance is flipped
            w.add(dataclasses.replace(inst, ports=ports), [100 + j, *REF])
        w.write(tmp_path / "sets", {})
    wrong = [j for j in range(20) if j not in (4, 9)]
    sets = [{"set": "rope-head-d16", "ok": False, "n": 20, "bad": wrong[:16], "bad_n": 18}, {"set": "silumul-v1-i8", "ok": False, "n": 3, "bad": [1], "bad_n": 1}]
    assert VX.verify(tmp_path) == {"ok": False, "sets": sets} and VX.verify_set(tmp_path / "sets" / "rope-head-d16") == sets[0]
