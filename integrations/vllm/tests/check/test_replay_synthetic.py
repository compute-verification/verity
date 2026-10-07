"""replay.py on the synthetic toy program (no capture artifacts, no checkpoint): the folded toy Program is evaluated
once with the reference evaluator, snapshots are *derived* from that transcript (as `test_compare_synthetic` does), and
the three tiers must agree with it bit-exactly; corrupted evidence / corrupted weights must be localised.

The toy log's sampled tokens are fabricated (`test_fold_synthetic.SAMPLED = [9, 3]`), so a replay of the toy weights
does not reproduce them.  `consistent_toy` rebuilds the log with the tokens the Program actually computes (two
fixed-point passes: the step-1 token depends on the step-0 token it is fed), which is what a real capture is."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from verity_vllm.check import fold_compare as cmp
from verity_vllm.check.replay import replay
from verity_vllm.observe.fold.fold import fold
from tests.observe import test_fold_synthetic as tfs
from tests.check.test_compare_synthetic import ToyWeights, _evaluate_program, _write_snapshots

REQ = tfs.REQ


def _computed_tokens(r, weights) -> list[int]:
    """The tokens the folded Program selects (root outputs of the reference evaluation)."""
    t = _evaluate_program(r, weights)
    return [int(t[g]) for g in r.program.circuit.root_outputs()]


@pytest.fixture(scope="module")
def consistent_toy():
    """(FoldResult, ToyWeights, transcript, tokens) of a toy log whose sampled tokens are the Program's own."""
    if not tfs._real_memory_available():
        tfs.install_stub_memory()
    original = list(tfs.SAMPLED)
    try:
        r = fold(tfs.toy_log(), tfs.toy_profile(), strict=True)
        weights = ToyWeights(r.params[-1][1].leaves)
        t0 = _computed_tokens(r, weights)[0]                     # step 0 depends on the prompt only
        tfs.SAMPLED[:] = [t0, original[1]]
        r = fold(tfs.toy_log(), tfs.toy_profile(), strict=True)
        t1 = _computed_tokens(r, weights)[1]                     # step 1 depends on t0
        tfs.SAMPLED[:] = [t0, t1]
        r = fold(tfs.toy_log(), tfs.toy_profile(), strict=True)
        t = _evaluate_program(r, weights)
        toks = [int(t[g]) for g in r.program.circuit.root_outputs()]
        assert toks == [t0, t1] and r.sampled == {REQ: [(0, t0), (1, t1)]}
        yield r, weights, t, toks
    finally:
        tfs.SAMPLED[:] = original


@pytest.fixture(scope="module")
def snapshots(consistent_toy, tmp_path_factory):
    r, weights, t, _ = consistent_toy
    d = tmp_path_factory.mktemp("snaps")
    _write_snapshots(r, t, d)
    return cmp.SnapshotStore(d)


def test_tier_a_covers_every_spec(consistent_toy, snapshots):
    r, weights, _, _ = consistent_toy
    a = replay.tier_a(r, snapshots, weights, k=2, seed=0)
    assert a["status"] == "pass", a
    assert a["uncovered_specs"] == [] and a["n_specs_covered"] == a["n_specs"] == len({m["spec"] for m in r.instances})
    assert all(e["status"] == "pass" for e in a["instances"])
    for spec, c in a["coverage"].items():
        # k per (spec, regime) stratum: the toy's specs may run in both regimes
        assert c["granularity"] == "instance" and c["pass"] == c["required"] == sum(min(2, r["instances"]) for r in c["regimes"].values()) and c["fail"] == 0, (spec, c)
    # deterministic: same inputs, same report (modulo timings)
    b = replay.tier_a(r, snapshots, weights, k=2, seed=0)
    strip = lambda rep: [(e["node"], e["status"], e.get("n_diff")) for e in rep["instances"]]
    assert strip(a) == strip(b)


def test_tier_a_descends_into_child_units(consistent_toy, snapshots):
    r, weights, _, _ = consistent_toy
    a = replay.tier_a(r, snapshots, weights, k=2, seed=3, max_arg_leaves=100)
    assert a["status"] == "pass", a
    # D85: the population summary rides along (informational) -- oversize specs report their tiling
    vp = a["vu_population"]
    assert vp["status"] == "ok", vp
    assert vp["oversize_instances"] > 0 and vp["count"] >= vp["instances"]
    assert set(vp["tiling"]) == {s for s, c in a["coverage"].items() if c["granularity"].startswith("child:")}
    assert all(t["covered_returns"] <= t["returns"] for t in vp["tiling"].values())
    assert vp["partial_tiling_specs"] == [] and vp["zero_mass_specs"] == []
    gran = {s.split("{")[0]: c["granularity"] for s, c in a["coverage"].items()}
    assert gran["Embedding_v1"] == "child:GatherBf16x16_v1"          # one gather gate of the batch (a primitive member)
    assert gran["Gemm_v1"] == "child:GemmCoordinate_v1{K=16}"       # one output coordinate (a composite batch member)
    assert gran["RMSNormFusedCuda_v2"] == "instance"
    kids = [e for e in a["instances"] if e["granularity"].startswith("child:")]
    assert kids and all(e["status"] == "pass" and e["n_returns"] == 1 for e in kids)
    assert {e["n_args"] for e in kids if "Gather" in e["granularity"]} == {1 + tfs.C["V"]}
    assert {e["n_args"] for e in kids if "GemmCoordinate" in e["granularity"]} == {2 * tfs.C["H"]}
    # different members are drawn for the k evaluations of one spec
    members = [e["child"]["path"][-1] for e in kids if "Gather" in e["granularity"]]
    assert len(set(members)) == len(members)


def test_gate_and_reference_resolution_agree(consistent_toy, snapshots):
    """`Replayer.gate_words` (by gate index, the child path) equals `operand_words` (by reference, the instance path)."""
    r, weights, _, _ = consistent_toy
    rep = replay.Replayer(r, snapshots, weights, chain=False)
    for m in r.instances:
        k = m["node"]
        by_ref, _ = rep.operand_words(k)
        by_gate = rep.gate_words(rep.call(k).args)
        assert by_gate == by_ref, m["name"]


def test_tier_b_each_step_passes(consistent_toy, snapshots):
    r, weights, _, toks = consistent_toy
    for step, tok in zip((0, 1), toks):
        b = replay.tier_b(r, snapshots, weights, step=step)
        assert b["status"] == "pass", b
        assert b["n_evaluated"] == b["n_instances"] == b["n_compared"] and b["n_diverged"] == 0 and b["n_skipped"] == 0
        assert b["partial"] is False and b["first_divergence"] is None
        assert b["tokens"] == {"computed": [tok], "sampled": [tok], "equal": True}
    # decode operands produced at step 0 are boundaries (from the snapshot), everything inside step 1 is chained
    b1 = replay.tier_b(r, snapshots, weights, step=1)
    att = next(e for e in b1["instances"] if e["spec"].startswith("Attention"))
    assert att["boundary"] > 0 and att["partial"] == []
    gemms = [e for e in b1["instances"] if e["spec"].startswith("Gemm")]
    assert gemms and all(e["boundary"] == 0 and e["partial"] == [] for e in gemms)


def test_tier_b_budget_skips_are_partial_not_failures(consistent_toy, snapshots):
    r, weights, _, toks = consistent_toy
    b = replay.tier_b(r, snapshots, weights, step=1, max_arg_leaves=100)
    assert b["status"] == "pass" and b["n_diverged"] == 0 and b["partial"] is True, b
    assert b["n_skipped"] == b["n_budget_skips"] > 0
    skipped = {e["node"] for e in b["instances"] if e["status"] == "skipped"}
    consumers = [e for e in b["instances"] if e.get("partial")]
    assert consumers and all(set(e["partial"]) <= skipped for e in consumers)
    assert b["tokens"]["equal"] is True and b["tokens"]["computed"] == [toks[1]]
    # a prefix
    p = replay.tier_b(r, snapshots, weights, step=0, max_instances=5)
    assert p["status"] == "pass" and p["n_evaluated"] == 5 and p["n_skipped"] == p["n_instances"] - 5 == p["n_budget_skips"]
    assert p["tokens"]["computed"] == [None] and p["tokens"]["equal"] is None
    # a layer selection keeps the non-layer instances (embedding, final norm, lm_head, sampler)
    l = replay.tier_b(r, snapshots, weights, step=1, layers=[])
    kept = [e for e in l["instances"] if e["status"] != "skipped"]
    assert kept and all("layers." not in e["name"] for e in kept)
    assert {e["spec"].split("{")[0] for e in kept} == {"Embedding_v1", "RMSNormFusedCuda_v2", "Gemm_v1", "TokenSelect_v1"}
    assert l["status"] == "pass" and l["partial"] is True and l["tokens"]["equal"] is True


def test_tier_b_localises_a_corrupted_snapshot(consistent_toy, tmp_path):
    r, weights, t, _ = consistent_toy
    att = next(m for m in r.instances if m["spec"].startswith("Attention"))
    from verity.ml.circuits.refs import runs
    bad = next(idx for space, idx, _, _ in runs(r.body.nodes[att["node"]].args[1].refs) if space == "n")
    bad_meta = next(m for m in r.instances if m["node"] == bad)
    _write_snapshots(r, t, tmp_path, corrupt_node=bad)
    store = cmp.SnapshotStore(tmp_path)
    b = replay.tier_b(r, store, weights, step=bad_meta["step"])
    assert b["status"] == "fail" and b["n_diverged"] == 1                # the computed values are right; one snapshot is wrong
    assert b["first_divergence"]["node"] == bad and b["first_divergence"]["n_diff"] == tfs.C["KVH"] * tfs.C["D"]
    a = replay.tier_a(r, store, weights, k=3, seed=0)
    assert a["status"] == "fail" and a["coverage"][bad_meta["spec"]]["fail"] >= 1


def test_tier_b_localises_a_wrong_weight_and_cascades(consistent_toy, snapshots):
    """A flipped checkpoint bit: the first consumer of that leaf diverges, its computed value flows on, downstream
    diverges too (the cascade), and the first divergence names the culprit."""
    r, weights, _, _ = consistent_toy
    from verity.ml.circuits.refs import Affine, Coll
    wt = r.params[-1][1]
    leaf = Coll(wt, Affine("p", 1, 0, wt.leaves))["layers"][0]["o_w"].refs[5][2]
    bad_w = ToyWeights(wt.leaves)
    bad_w.w = bad_w.w.copy()
    bad_w.w[leaf] ^= 0x0040
    b = replay.tier_b(r, snapshots, bad_w, step=0)
    assert b["status"] == "fail" and b["n_diverged"] > 1, b["n_diverged"]
    first = b["first_divergence"]
    o_proj = [m for m in r.instances if m["step"] == 0 and m["name"].endswith("o_proj")]
    assert first["node"] == o_proj[0]["node"] and first["spec"].startswith("Gemm"), first
    assert b["divergences"][0]["node"] == first["node"]
    assert b["tokens"]["equal"] in (False, True)                            # reported either way; the status is already fail


def test_tier_c_end_to_end_tokens(consistent_toy):
    r, weights, _, toks = consistent_toy
    c = replay.tier_c(r, weights)
    assert c["status"] == "pass", c
    assert c["tokens"][REQ] == {"steps": [0, 1], "computed": toks, "sampled": toks, "equal": True, "first_mismatch_step": None}
    assert c["n_evaluated"] == c["n_instances"] == len(r.instances) and c["n_skipped"] == 0
    assert [s["status"] for s in c["per_step"]] == ["pass", "pass"]
    assert all(s["tokens"]["equal"] is True for s in c["per_step"])
    # a step prefix
    c0 = replay.tier_c(r, weights, steps=[0])
    assert c0["status"] == "pass" and c0["tokens"][REQ]["computed"] == toks[:1]
    # a budget skip of the lm_head cascades (no interior snapshots to fall back on): reported as skipped, never as pass
    cb = replay.tier_c(r, weights, max_arg_leaves=100)
    assert cb["status"] == "skipped" and cb["n_skipped"] > 0 and cb["tokens"][REQ]["equal"] is None


def test_tier_c_reports_token_mismatch_and_continues_with_the_computed_token(consistent_toy):
    """The fabricated toy log ([9, 3]) is not what the Program computes: tier C reports the first mismatching step and
    keeps feeding the *computed* token forward (the decode Embedding is wired to the TokenSelect output)."""
    _, weights, _, toks = consistent_toy
    fabricated = [9, 3]
    original = list(tfs.SAMPLED)
    try:
        tfs.SAMPLED[:] = fabricated
        r = fold(tfs.toy_log(), tfs.toy_profile(), strict=True)
    finally:
        tfs.SAMPLED[:] = original
    assert r.sampled == {REQ: [(0, 9), (1, 3)]} and toks != fabricated
    c = replay.tier_c(r, weights)
    assert c["status"] == "fail" and "token" in c["status_reason"]
    tk = c["tokens"][REQ]
    assert tk["sampled"] == fabricated and tk["equal"] is False and tk["first_mismatch_step"] == 0
    # step 1 was computed from the computed step-0 token, so it equals the consistent replay's step-1 token
    assert tk["computed"] == toks
    assert [s["tokens"]["equal"] for s in c["per_step"]] == [False, False]
    assert c["n_diverged"] == 0 and c["n_skipped"] == 0                    # no interior comparison in tier C: only tokens


def test_reports_are_json(consistent_toy, snapshots):
    r, weights, _, _ = consistent_toy
    for rep in (replay.tier_a(r, snapshots, weights, k=1, seed=0), replay.tier_b(r, snapshots, weights, step=0), replay.tier_c(r, weights)):
        s = json.dumps(cmp._jsonable(rep))
        assert json.loads(s)["status"] in replay.TIER_STATUSES
        assert "seconds" in rep and "max_rss_gb" in rep
        print(replay.summary_line(rep))


def test_tier_a_served_tokens_and_strata(consistent_toy, snapshots):
    r, weights, _, _ = consistent_toy
    a = replay.tier_a(r, snapshots, weights, k=2, seed=0)
    t = a["tokens"]
    assert t["evaluated"] > 0 and t["mismatch"] == [] and t["all_rows_equal"] is True and 0 in t["steps"]
    assert a["strata"] >= a["n_specs"]
    assert all("regime" in e for e in a["instances"])


def test_tier_chain_computed_intermediates(consistent_toy, snapshots):
    r, weights, _, _ = consistent_toy
    c = replay.tier_chain(r, snapshots, weights, n=2, seed=1, depth=3, max_nodes=6)
    assert c["status"] == "pass", c
    assert c["n_slices"] >= 1 and c["n_edges_chained"] >= 1
    for sl in c["slices"]:
        assert sl["status"] == "pass" and sl["edges_chained"] >= 1
        assert all(m["status"] == "pass" for m in sl["members"])
        assert any(m["chained_from"] for m in sl["members"])
