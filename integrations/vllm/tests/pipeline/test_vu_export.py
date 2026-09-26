"""The VU exporter (`verity_vllm.pipeline.vu_export`) on the replay's honest test Programs: the sets it writes are grouped by
subcircuit, content-addressed and re-evaluate under their relation; a VU that does not replay bit-exactly is never exported."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity_vllm.check.replay import evaluate as EV
from verity_vllm.pipeline import vu_export as VX
from verity_vllm.pipeline import vu_store as VS
from tests.check.test_sampled_replay import H, I, V, Honest, _bf16, _program, _programs, _weights

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
    _export(attn_world, tmp_path, limits=VX.Limits(per_family=64, gemm_coords_per_row=1000))
    d = tmp_path / "sets" / f"gemm-coordinate-ampere-bf16-k{H}"
    man = json.loads((d / "manifest.json").read_text())
    x = np.frombuffer((d / "x.u16").read_bytes(), "<u2").reshape(man["n"], H)
    w = np.frombuffer((d / "w.u16").read_bytes(), "<u2").reshape(man["n"], H)
    y = np.frombuffer((d / "y.u16").read_bytes(), "<u2")
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
    args = (_programs(attn_world), bmap, committed, _weights(attn_world), {}, None, {}, {}, {"r0": "d" * 64}, {"git_head": "f00"}, _Eng())
    assert VS.after_replay(None, {"result": True}, 0, ROOT, *args) is None
    assert VS.after_replay(str(tmp_path), {"result": None}, 0, ROOT, *args) is None
    assert VS.after_replay(str(tmp_path), {"result": True}, 1, ROOT, *args) is None and not (tmp_path / "export.json").exists()
    (tmp_path / "provenance.json").write_text(json.dumps({"row": 101, "research_run": "r-test"}))
    (tmp_path / "limits.json").write_text(json.dumps({"per_family": 3}))
    s = VS.after_replay(str(tmp_path), {"result": True}, 0, ROOT, *args)
    p = s["provenance"]
    assert p["row"] == 101 and p["research_run"] == "r-test" and p["model"] == "tiny/model" and p["versions"] == {"git_head": "f00"}
    assert all(c["drawn"] <= 3 for c in s["by_family"].values()) and VX.verify(tmp_path)["ok"]


def test_a_spent_time_budget_stops_the_draw_by_name(attn_world, tmp_path):
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
