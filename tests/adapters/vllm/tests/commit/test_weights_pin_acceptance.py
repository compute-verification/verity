"""Acceptance tests for the weights-of-record pin (W1/W2, the engine-assumption fields and stamped records), CPU only.

The pin resolves the Program's prescribed `weights` parameter to the REGISTERED checkpoint (manifests/checkpoints.json shard sha256s)
through the implementation's composition rule, ONCE per workload (a record beside the Program), and compares the served parameters with it
by digest -- no rehash of the model per request.  These tests hold, on a synthetic checkpoint + Programs (no torch, no pod):

  1. B1 / B2 / B8 of the SAME model: the record is a function of (weights struct, checkpoint) only -- identical fields, digests and
     root_of_record across sequence shapes; each B's run accepts a record derived from one of ITS components; the default-shape record
     of a mixed B>=2 workload is refused by name, a stamped record of ANOTHER workload too.
  2. the workload's component pick, run on synthetic workload dirs -- first manifest
     component, build_request only when it declares that component's shape.
  3. Shared weights across request subcircuits: the GLOBAL manifest carries every weight field ONCE (not x B) and binding collapses them
     to ONE registered-input identity per rank; an omitted registration is a named missing identity, a duplicate is a reported copy.
  4. Wrong registered root / wrong checkpoint context: shard sha != pin (W1), no manifest entry, another repo / revision (W2), another
     Program (W2) -- each refused by name; the verdict text names the pin outcome (WEIGHTS PIN ok / missing / mismatch).
  5. Substituted weight input: a whole served tensor from another model -- named; the checkpoint FILE substituted on disk -- W1 named.
  6. TP shard / reference mismatch: a rank served with the other rank's slice of ONE field, a TP2 record against a TP1 (unsharded) model,
     a world/rank the Program does not declare -- each named.
  7. Missing metadata field vs real inequality: a record without engine_assumptions / analytic fields on an honest
     model still passes with its engine constants listed unpinned by name; a real digest inequality fails, never repaired.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests.check.test_weights_of_record import (  # noqa: F401 (fixture)
    PREFIX,
    _program,
    _served,
    _write_program,
    checkpoint,
    live_digests,
    write_safetensors,
)
from verity_vllm.check import weights_of_record as WR
from verity_vllm.pipeline import cli

pytestmark = pytest.mark.usefixtures("code_versions_stood_in")

# the three panel shapes of one model: B1 = the envelope, B2 / B8 mixed (no request has both maxima -> the default shape is NOT a component)
B1_SHAPES = [(9, 2)]
B2_SHAPES = [(4, 2), (9, 1)]
B8_SHAPES = [(9, 1), (8, 2), (7, 1), (6, 2), (5, 1), (4, 2), (3, 1), (2, 2)]


def _row(tmp_path, name, prog, shapes, *, workload_digest, default=(9, 2)):
    """A workload dir as the Build leaves it: build_request (default wrapper shape) + one component per declared shape, the global
    Program's workload digest, the GLOBAL manifest (program_digest = workload digest, component_digests)."""
    root = tmp_path / name
    comps, cdig = {}, []
    for i, (lp, t) in enumerate(shapes):
        dig = hashlib.sha512(f"{name}/LP{lp}_T{t}".encode()).hexdigest()
        _write_program(root / f"build_request_LP{lp}_T{t}", prog, dig, lp, t)
        comps[f"r{i}"] = {"program_digest": dig, "LP": lp, "T": t}
        cdig.append(dig)
    ddig = hashlib.sha512(f"{name}/default".encode()).hexdigest()
    _write_program(root / "build_request", prog, ddig, *default)
    (root / "build_workload").mkdir()
    (root / "build_workload" / "workload_program.json").write_text(json.dumps({"workload_digest": workload_digest}))
    (root / "manifest.json").write_text(json.dumps({"program_digest": workload_digest, "component_digests": sorted(set(cdig)), "components": comps}))
    return root, {workload_digest, *cdig}, ddig


def test_b1_b2_b8_same_model_one_pin_per_row_and_the_default_shape_record_is_refused_at_b_ge_2(checkpoint, tmp_path):
    """B1, B2 and B8 on the same model: the default-shape record fails at B>=2 and the component-derived record passes."""
    d, man, t = checkpoint
    prog, fields = _program()
    live = live_digests(_served(t, fields))
    W1, W2, W8 = "1" * 128, "2" * 128, "8" * 128
    r1, set1, d1 = _row(tmp_path, "b1", prog, B1_SHAPES, workload_digest=W1)
    r2, set2, d2 = _row(tmp_path, "b2", prog, B2_SHAPES, workload_digest=W2)
    r8, set8, d8 = _row(tmp_path, "b8", prog, B8_SHAPES, workload_digest=W8)
    # (a) the pin is a function of (weights struct, checkpoint): every component of every B yields the SAME fields/digests/root
    recs = {}
    for root in (r1, r2, r8):
        for comp in WR.component_program_dirs(str(root)):
            recs[comp] = WR.build_record(comp, str(d), checkpoints_manifest=str(man))
    roots = {r["root_of_record"] for r in recs.values()}
    assert len(roots) == 1, roots
    digs = {tuple((f["field"], f["digest"]) for f in r["fields"]) for r in recs.values()}
    assert len(digs) == 1 and all(r["complete"] and r["of_record"] for r in recs.values())
    assert all(r["compared"] == len(fields) for r in (WR.check(rec, live) for rec in recs.values()))      # one compare per field, B-independent
    # (b) each B accepts a record derived from one of ITS components (un-stamped, single-digest rule = the row's --program-dir path)
    for root, ofr in ((r1, set1), (r2, set2), (r8, set8)):
        comp = WR.component_program_dirs(str(root))[0]
        res = WR.check(recs[comp], live, program_digests_of_record=ofr)
        assert res["result"] is True and res["program_of_record"] is True, (root.name, res["why"])
    # (c) the DEFAULT-shape record of a mixed B>=2 workload is refused by name with ZERO mismatched fields ...
    for root, ofr, ddig in ((r2, set2, d2), (r8, set8, d8)):
        rec_default = WR.build_record(str(root / "build_request"), str(d), checkpoints_manifest=str(man))
        assert rec_default["root_of_record"] in roots                                   # same weights, only the Program identity differs
        res = WR.check(rec_default, live, program_digests_of_record=ofr)
        assert res["result"] is False and res["mismatched"] == [] and res["missing"] == []
        assert res["why"].startswith(f"record built for Program {ddig[:16]} != Program of record"), res["why"]
    # ... while at B1 the default shape IS the Program of record
    rec_b1 = WR.build_record(str(r1 / "build_request"), str(d), checkpoints_manifest=str(man))
    assert WR.check(rec_b1, live, program_digests_of_record={W1, d1})["result"] is True
    # (d) stamped records (--programs-root): the B8 record is of record for the B8 run only -- refused for the B2 run and vice versa
    st8 = WR.stamp_of_record_set(WR.build_record(WR.component_program_dirs(str(r8))[0], str(d), checkpoints_manifest=str(man)), str(r8))
    st2 = WR.stamp_of_record_set(WR.build_record(WR.component_program_dirs(str(r2))[0], str(d), checkpoints_manifest=str(man)), str(r2))
    assert st8["of_record"] is True and set(st8["of_record_set"]["program_digests"]) == set8
    assert st2["of_record"] is True and set(st2["of_record_set"]["program_digests"]) == set2
    assert WR.check(st8, live, program_digests_of_record=set8)["result"] is True
    assert WR.check(st2, live, program_digests_of_record=set2)["result"] is True
    x = WR.check(st8, live, program_digests_of_record=set2)
    assert x["result"] is False and "record's of-record set != the run's" in x["why"]
    assert WR.check(st2, live, program_digests_of_record=set8)["result"] is False


def test_substituted_weight_input_is_named_whether_in_memory_or_on_disk(checkpoint, tmp_path):
    """A served tensor taken from ANOTHER model (whole field, plausible values) fails by field name; a checkpoint
    shard substituted on disk (another model's bytes under the pinned file name) fails W1 by shard name BEFORE any field compare."""
    d, man, t = checkpoint
    prog, fields = _program()
    rec = WR.weights_of_record(prog, WR.CheckpointIndex(str(d), checkpoints_manifest=str(man)), prefix=PREFIX)
    served = _served(t, fields)
    other = np.random.default_rng(11).random(served["model.layers.0.self_attn.o_proj.weight"].shape, dtype=np.float32)
    sub = dict(served, **{"model.layers.0.self_attn.o_proj.weight": WR._f32_to_bf16_bits(other)})
    res = WR.check(rec, live_digests(sub), program_digests_of_record={"d" * 128})
    assert res["result"] is False and [m["served_name"] for m in res["mismatched"]] == ["model.layers.0.self_attn.o_proj.weight"]
    assert res["why"].startswith("1 served weight(s) differ from the checkpoint of record: model.layers.0.self_attn.o_proj.weight")
    # the substituted model served in FULL (every tensor from elsewhere) names every field, root unchanged (the record never re-derives from the engine)
    rng = np.random.default_rng(12)
    full = {k: (WR._f32_to_bf16_bits(rng.random(v.shape, dtype=np.float32)) if v.dtype == np.uint16 else v) for k, v in served.items()}
    resf = WR.check(rec, live_digests(full))
    n_bf16_fields = sum(1 for _n, _s, bits in fields if bits == 16)         # 12: every bf16 field named (tied lm_head through embed_tokens), the f32 kv scale untouched
    assert resf["result"] is False and len(resf["mismatched"]) == n_bf16_fields == 12
    assert resf["root_of_record"] == rec["root_of_record"]
    # on disk: the shard's bytes replaced (same names, other values) -> the sha differs from the checkpoints.json pin -> W1, named
    sub_dir = tmp_path / "snap_sub"
    sub_dir.mkdir()
    t2 = {k: (dt, (WR._f32_to_bf16_bits(rng.random(a.shape, dtype=np.float32)) if dt == "BF16" else a)) for k, (dt, a) in t.items()}
    write_safetensors(str(sub_dir / "model.safetensors"), t2)
    (sub_dir / "config.json").write_text(json.dumps(json.loads((d / "config.json").read_text())))
    man_sub = tmp_path / "ck_sub.json"                                     # the pin still names the ORIGINAL sha for this path
    orig_sha = json.loads(Path(man).read_text())["checkpoints"][0]["files"]["model.safetensors"]["sha256"]
    Path(man_sub).write_text(json.dumps({"checkpoints": [{"repo": "test/tiny", "revision": "abc", "local_path": str(sub_dir), "files": {"model.safetensors": {"sha256": orig_sha}}}]}))
    rec_sub = WR.weights_of_record(prog, WR.CheckpointIndex(str(sub_dir), checkpoints_manifest=str(man_sub)), prefix=PREFIX)
    assert rec_sub["of_record"] is False and rec_sub["checkpoint"]["shards"]["model.safetensors"]["pinned"] is False
    # the engine served exactly the substituted disk bytes: self-consistent, yet refused -- 'served == disk' is not the pin
    served_sub = _served(t2, fields)
    r = WR.check(rec_sub, live_digests(served_sub))
    assert r["result"] is False and r["mismatched"] == [] and "not the pinned ones (W1)" in r["why"] and orig_sha[:16] in r["why"]


def test_tp_shard_reference_mismatch_is_named(checkpoint):
    """Rank-local shards <-> the registered source: the record of rank r is the per-part slice of the registered
    tensors; ONE field served from the other rank's slice is named; a TP2 record against an unsharded (TP1) model is named; a Program
    that declares a world the checkpoint geometry cannot be split into is unresolved (fail closed)."""
    d, man, t = checkpoint
    tp0, tp1 = {"world": 2, "rank": 0}, {"world": 2, "rank": 1}
    prog0, f0 = _program(tp=tp0)
    rec0 = WR.weights_of_record(prog0, WR.CheckpointIndex(str(d), checkpoints_manifest=str(man)), prefix=PREFIX)
    assert rec0["complete"] and rec0["tp"] == tp0
    by = {f["field"]: f for f in rec0["fields"]}
    assert by["model.model.layers.0.self_attn.qkv_proj.weight"]["shard"].startswith("rank 0/2: per-part dim-0 slice")
    assert by["model.model.layers.0.self_attn.o_proj.weight"]["shard"].startswith("rank 0/2: dim-1 slice")
    assert by["model.model.norm.weight"]["shard"].startswith("rank 0/2: replicated")
    assert by["model.model.layers.1.mlp.experts.w13_weight"]["shard"] == "rank 0/2 (per-part dim 0)" and by["model.model.layers.1.mlp.experts.w2_weight"]["shard"] == "rank 0/2 (dim 1)"
    s0, s1 = _served(t, f0, tp=tp0), _served(t, f0, tp=tp1)
    assert WR.check(rec0, live_digests(s0))["result"] is True
    # one field served from the OTHER rank's slice (a mis-wired shard load): exactly that field is named
    mixed = dict(s0, **{"model.layers.0.mlp.down_proj.weight": s1["model.layers.0.mlp.down_proj.weight"]})
    r = WR.check(rec0, live_digests(mixed))
    assert r["result"] is False and [m["served_name"] for m in r["mismatched"]] == ["model.layers.0.mlp.down_proj.weight"]
    # a rank-0 record against an UNSHARDED (TP1) served model: every sharded field differs, the replicated norms do not
    prog_full, ff = _program()
    full = _served(t, ff)
    rf = WR.check(rec0, live_digests(full))
    assert rf["result"] is False
    named = {m["served_name"] for m in rf["mismatched"]}
    assert "model.layers.0.self_attn.qkv_proj.weight" in named and "model.norm.weight" not in named and "model.layers.0.input_layernorm.weight" not in named
    # a TP1 record against a rank-0 shard: named the same way (the reference relation is checked in both directions)
    rec_full = WR.weights_of_record(prog_full, WR.CheckpointIndex(str(d), checkpoints_manifest=str(man)), prefix=PREFIX)
    assert WR.check(rec_full, live_digests(s0))["result"] is False
    # a declared world the geometry cannot be split into (3 does not divide the fused rows): unresolved by name, fail closed
    prog3, _ = _program(tp={"world": 3, "rank": 0})
    rec3 = WR.weights_of_record(prog3, WR.CheckpointIndex(str(d), checkpoints_manifest=str(man)), prefix=PREFIX)
    assert rec3["complete"] is False and rec3["unresolved"] and WR.check(rec3, live_digests(s0))["result"] is False
    assert "incomplete" in WR.check(rec3, live_digests(s0))["why"]


def test_missing_metadata_field_is_not_a_pin_mismatch(checkpoint):
    """A record lacking a metadata field (no engine_assumptions / analytic fields, engine constants listed
    unpinned) is a weaker but SOUND pin on an honest model -- accepted, the gap named in `unpinned`; a real digest inequality on the same record
    fails by name -- the two are different outcomes, the former is never reported as the latter and the latter never repaired."""
    d, man, t = checkpoint
    prog, fields = _program()
    rec = WR.weights_of_record(prog, WR.CheckpointIndex(str(d), checkpoints_manifest=str(man)), prefix=PREFIX)
    legacy = json.loads(json.dumps(rec))
    legacy.pop("engine_assumptions")
    legacy.pop("n_analytic", None)
    legacy.pop("analytic_fields", None)
    for f in legacy["fields"]:                                    # the older record shape: the kv scale listed by name, no digest
        if f["field"].endswith("attn._k_scale"):
            f.update({"digest": None, "pinned": False, "why": "engine constant (not a checkpoint tensor)"})
            f.pop("rule", None)
            f.pop("sources", None)
    legacy["unpinned"] = ["model.model.layers.0.self_attn.attn._k_scale"]
    legacy["n_pinned"] = rec["n_pinned"] - 1
    served = _served(t, fields)
    ok = WR.check(legacy, live_digests(served), program_digests_of_record={"d" * 128}, kv_cache_dtype="auto")
    assert ok["result"] is True and ok["compared"] == len(fields) - 1 and ok["unpinned"] == ["model.model.layers.0.self_attn.attn._k_scale"]
    assert "not authenticated by this pin" in ok["unpinned_note"] and ok["engine_assumptions"] is None
    # the same legacy record, one served element flipped: a real inequality -- named, not repaired
    bad = dict(served)
    w = bad["model.layers.0.mlp.gate_up_proj.weight"].copy()
    w[0, 0] ^= 0x0001
    bad["model.layers.0.mlp.gate_up_proj.weight"] = w
    r = WR.check(legacy, live_digests(bad), program_digests_of_record={"d" * 128})
    assert r["result"] is False and [m["served_name"] for m in r["mismatched"]] == ["model.layers.0.mlp.gate_up_proj.weight"]
    # a served kv scale != 1.0 under the legacy record is NOT caught (unpinned by name) -- the current record catches it: the repair is re-derivation
    sc = dict(served, **{"model.layers.0.self_attn.attn._k_scale": np.float32(0.5).reshape(()).view(np.uint32)})
    assert WR.check(legacy, live_digests(sc))["result"] is True
    assert WR.check(rec, live_digests(sc))["result"] is False


@pytest.mark.parametrize("name", ["b1", "b2"])
def test_pin_record_is_built_once_per_row_not_per_request(checkpoint, tmp_path, name):
    """No rehash per request: the CLI derives ONE record per workload (beside the picked Program); the run reuses it: building the
    record touches the checkpoint (sha256 of the shards + composition) exactly once, check() reads digests only."""
    d, man, t = checkpoint
    prog, fields = _program()
    shapes = B1_SHAPES if name == "b1" else B2_SHAPES
    root, ofr, _ = _row(tmp_path, name, prog, shapes, workload_digest="c" * 128)
    out = root / "weights_of_record.json"
    rc = cli.run(WR, ["--programs-root", str(root), "--checkpoint-dir", str(d), "--checkpoints", str(man), "--out", str(out)])
    rec = json.loads(Path(out).read_text())
    assert rc == 0 and rec["of_record"] is True and set(rec["of_record_set"]["program_digests"]) == ofr
    live = live_digests(_served(t, fields))
    n_reads = {"n": 0}
    orig = WR.CheckpointIndex.array

    def counting(self, nm):
        n_reads["n"] += 1
        return orig(self, nm)

    WR.CheckpointIndex.array = counting
    try:
        for _ in range(len(shapes) * 4):                       # B requests x pairs: the check never opens the checkpoint again
            assert WR.check(rec, live, program_digests_of_record=ofr)["result"] is True
    finally:
        WR.CheckpointIndex.array = orig
    assert n_reads["n"] == 0
