"""verity_vllm/program/kernels/twins.py — every twin against the gate evaluator on synthetic small statics (laptop-fast), the
Replayer's `--evaluator` switch on the synthetic toy (gates / twins / auto agree word for word), the `--check`
evidence file, and — behind `CAPTURE_BIG=1` like `test_replay_m1.py` — tier b step 0 of M1 (the whole 32-row
SmolLM2 prefill, 9666 instances) with `--evaluator twins`: 0 divergences, `partial == False`, < 1.5 GB."""
from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import numpy as np
import pytest

from verity_vllm.check.replay import replay
from verity_vllm.program.kernels import kernel_registry as KR
from verity_vllm.program.kernels import twins as TW
from verity_vllm.pipeline import cli
from tests.check.test_replay_synthetic import consistent_toy, snapshots  # noqa: F401 - fixtures
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.defs import bind

FAMILIES = sorted(TW.FAMILIES)

# every specialization present in the two logs the sweep covers (resolve_log `by_spec`): SmolLM2-135M (M1) and
# Qwen2.5-1.5B (M6) — a twin must claim each (the CAPTURE_BIG test below evaluates M1's for real)
M1_STATICS = {
    "Gemm_v1": [{"K": 576, "N": 960}, {"K": 576, "N": 576}, {"K": 576, "N": 3072}, {"K": 1536, "N": 576}, {"K": 576, "N": 49152}],
    "RMSNormTriton_v1": [{"N": 576, "EPS": 1e-05}], "RMSNormFusedCuda_v2": [{"N": 576, "EPS": 1e-05}],
    "RoPE_v1": [{"NHEADS": 9, "D": 64}, {"NHEADS": 3, "D": 64}], "SiluMul_v1": [{"I": 1536}],
    "Embedding_v1": [{"V": 49152, "H": 576}], "TokenSelect_v1": [{"V": 49152}],
    "Attention_v3": [{"T": T, "NH": 9, "KVH": 3, "D": 64, "BN": 128} for T in range(1, 48)],
}
M6_STATICS = {
    "Gemm_v1": [{"K": 1536, "N": 2048}, {"K": 1536, "N": 1536}, {"K": 1536, "N": 17920}, {"K": 8960, "N": 1536}, {"K": 1536, "N": 151936}],
    "BiasAdd_v1": [{"N": 2048}], "RMSNormTriton_v1": [{"N": 1536, "EPS": 1e-06}], "RMSNormFusedCuda_v2": [{"N": 1536, "EPS": 1e-06}],
    "RoPE_v1": [{"NHEADS": 12, "D": 128}, {"NHEADS": 2, "D": 128}], "SiluMul_v1": [{"I": 8960}],
    "Embedding_v1": [{"V": 151936, "H": 1536}], "TokenSelect_v1": [{"V": 151936}],
    "Attention_v3": [{"T": T, "NH": 12, "KVH": 2, "D": 128, "BN": 64} for T in range(1, 48)],
}


# ---------------------------------------------------------------------------------------------------------
# twin == gates on synthetic small statics
# ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("base", FAMILIES)
def test_twin_matches_gates_on_random_small_statics(base):
    cls = TW.FAMILIES[base]
    rng = np.random.default_rng(zlib.crc32(base.encode()))
    rnd = TW.check_instances(cls, cls.small_statics, rng, 8, TW.RANDOM_KINDS)
    assert rnd["status"] == "pass" and rnd["mismatches"] == [] and rnd["n"] >= 6, rnd
    assert len(rnd["statics"]) >= 2, rnd["statics"]                       # several structural variants were drawn
    edge = TW.check_instances(cls, cls.small_statics, rng, 8, TW.EDGE_KINDS)
    assert edge["mismatches"] == [], edge
    assert edge["n"] >= 2, edge                                            # NaN / inf / subnormal / boundary words
    if base == "Attention_v3":                                             # declines (not mismatches) on non-finite words
        assert edge["n_declined"] > 0 and all(k.startswith("non-finite") for k in edge["declined"]), edge["declined"]
    else:
        assert edge["n_declined"] == 0


@pytest.mark.parametrize("model,statics", [("m1", M1_STATICS), ("m6", M6_STATICS)])
def test_twins_cover_every_logged_specialization(model, statics):
    seen = set()
    for base, lst in statics.items():
        for st in lst:
            spec = bind(TW.DEFINITIONS[base], **st)
            assert KR.covers(spec, TW.KERNEL) is None, (spec.id, KR.covers(spec, TW.KERNEL))
            assert spec.base_id == base and TW.FAMILIES[base].twin_id
            seen.add(TW.FAMILIES[base].twin_id)
    assert seen == {TW.FAMILIES[b].twin_id for b in statics}


def test_twins_decline_statics_outside_the_body():
    assert "K=24" in KR.covers(bind(TW.DEFINITIONS["Gemm_v1"], K=24, N=2), TW.KERNEL)      # K not a multiple of 16
    assert KR.covers(bind(TW.DEFINITIONS["Gemm_v1"], K=32, N=2), TW.KERNEL) is None


def test_gemm_twin_falls_back_to_the_primitives_on_non_finite_words():
    """`tc_model.cpp` has no inf/NaN extension: coordinates with a non-finite operand or result take the registered
    primitives' path and still equal the gates."""
    spec = bind(TW.DEFINITIONS["Gemm_v1"], K=32, N=3)
    tw = KR.instance(spec, TW.KERNEL)
    rng = np.random.default_rng(1)
    x = TW.rand_bf16(rng, 32, "gauss")
    w = TW.rand_bf16(rng, 3 * 32, "gauss").reshape(3, 32)
    w[0, 5] = 0x7F80                    # +inf in coordinate 0
    w[1, 7] = 0x7FC1                    # NaN payload in coordinate 1
    w[2, :] = 0x7F00                    # huge finite words in coordinate 2 (the chain may or may not overflow)
    args = [x.astype(np.uint64), w.reshape(-1).astype(np.uint64)]
    got, expect = tw(*args, **spec.statics), np.array(evaluate(spec, *(a.tolist() for a in args)), dtype=np.uint64)
    assert np.array_equal(got, expect), (got, expect)
    assert TW._nonfinite16(got.astype(np.uint16))[:2].all()


def test_attention_twin_declines_non_finite_operands():
    spec = bind(TW.DEFINITIONS["Attention_v3"], T=3, NH=1, KVH=1, D=16, BN=128)
    tw = KR.instance(spec, TW.KERNEL)
    rng = np.random.default_rng(2)
    args = TW.AttentionTwin.random_args(spec.statics, rng, "gauss")
    tw(*args, **spec.statics)                                             # finite: claimed
    args[1][0] = 0x7FC0                                                   # a NaN key word
    with pytest.raises(KR.Declined):
        tw(*args, **spec.statics)


# ---------------------------------------------------------------------------------------------------------
# the Replayer's evaluator switch (synthetic toy: 12 specializations of 8 families; its D=8 attention has no twin —
# the FA2 model needs D % 16 == 0 — so those 3 instances take the gates in every mode, which the reports must say)
# ---------------------------------------------------------------------------------------------------------


def _twinned(r) -> set[int]:
    """Nodes of the toy whose specialization has a twin."""
    from verity_vllm.check.fold_compare import _attach_specs
    _attach_specs(r)
    return {m["node"] for m in r.instances if KR.covers(m["_spec"], TW.KERNEL) is None}


def test_tier_b_evaluators_agree_and_are_reported(consistent_toy, snapshots):  # noqa: F811
    r, weights, _, toks = consistent_toy
    twinned = _twinned(r)
    assert twinned and len(twinned) == len(r.instances) - 3, len(twinned)
    for step, tok in zip((0, 1), toks):
        metas = [m for m in r.instances if m["step"] == step]
        n_tw = sum(1 for m in metas if m["node"] in twinned)
        for mode in ("gates", "twins", "auto"):
            b = replay.tier_b(r, snapshots, weights, step=step, evaluator=mode)
            assert b["status"] == "pass" and b["n_diverged"] == 0 and b["partial"] is False, b
            assert b["tokens"] == {"computed": [tok], "sampled": [tok], "equal": True} and b["evaluator"] == mode
            ev = b["evaluators"]
            assert ev["mode"] == mode and ev["declined"] == 0
            if mode == "gates":
                assert ev == {"twins": 0, "gates": b["n_evaluated"], "declined": 0, "mode": "gates", "self_check": {}}
                assert all(e["evaluator"] == "gates" and e["twin_id"] is None for e in b["instances"])
            else:
                assert ev["twins"] == n_tw and ev["gates"] == b["n_evaluated"] - n_tw
                assert set(ev["self_check"]) == {m["spec"] for m in metas if m["node"] in twinned}
                assert all(sc["status"] == "pass" and sc["n_random"] >= 3 and sc["mismatches"] == [] for sc in ev["self_check"].values())
                for e in b["instances"]:
                    if e["node"] in twinned:
                        assert e["evaluator"] == "twins" and e["twin_id"] == ev["self_check"][e["spec"]]["twin_id"], e
                    else:
                        assert e["evaluator"] == "gates" and e["twin_id"] is None and e["spec"].startswith("Attention_v3"), e
            # the report keeps every field h2's gates read
            for key in ("status", "partial", "n_diverged", "tokens", "n_from_earlier_steps", "n_earlier_step_operands"):
                assert key in b


def test_twins_and_gates_compute_identical_words(consistent_toy, snapshots):  # noqa: F811
    r, weights, _, _ = consistent_toy
    twinned = _twinned(r)
    reps = {mode: replay.Replayer(r, snapshots, weights, chain=True, snapshots=True, evaluator=mode) for mode in ("gates", "twins")}
    for m in r.instances:                                                 # both steps, node order, no release: whole table
        k = m["node"]
        entries = {mode: rep.replay_node(k, compare=True) for mode, rep in reps.items()}
        assert entries["gates"]["evaluator"] == "gates", entries
        assert entries["twins"]["evaluator"] == ("twins" if k in twinned else "gates"), entries
        assert np.array_equal(reps["gates"].computed[k], reps["twins"].computed[k]), m["name"]
        assert entries["gates"]["n_args"] == entries["twins"]["n_args"] and entries["gates"]["status"] == entries["twins"]["status"] == "pass"
    assert reps["twins"].evaluator_counts == {"twins": len(twinned), "gates": len(r.instances) - len(twinned)}
    assert reps["gates"].evaluator_counts == {"gates": len(r.instances)}


def test_tier_c_with_twins_reproduces_the_tokens(consistent_toy):  # noqa: F811
    r, weights, _, toks = consistent_toy
    twinned = _twinned(r)
    c = replay.tier_c(r, weights, evaluator="twins")
    assert c["status"] == "pass" and c["n_skipped"] == 0 and c["n_evaluated"] == len(r.instances), c
    assert c["evaluators"]["twins"] == len(twinned) and c["evaluators"]["gates"] == len(r.instances) - len(twinned)
    assert list(c["tokens"].values())[0]["computed"] == toks
    assert sum(s["evaluators"]["twins"] for s in c["per_step"]) == len(twinned)


def test_failed_self_check_falls_back_in_auto_and_errors_in_twins(consistent_toy, snapshots, monkeypatch):  # noqa: F811
    r, weights, _, _ = consistent_toy
    real = TW.admit

    def rigged(spec, n=3, seed=0):
        rep = real(spec, n=n, seed=seed)
        if spec.base_id == "SiluMul_v1":
            rep = {**rep, "status": "fail", "mismatches": [{"spec": spec.id, "kind": "rigged", "n_diff": 1}]}
        return rep

    monkeypatch.setattr(TW, "admit", rigged)
    b = replay.tier_b(r, snapshots, weights, step=0, evaluator="auto")
    assert b["status"] == "pass" and b["n_diverged"] == 0
    silu = [e for e in b["instances"] if e["spec"].startswith("SiluMul")]
    attn = [e for e in b["instances"] if e["spec"].startswith("Attention")]      # no twin for D=8: gates in every mode
    assert silu and all(e["evaluator"] == "gates" and e["twin_id"] is None for e in silu)
    assert b["evaluators"]["gates"] == len(silu) + len(attn) and b["evaluators"]["twins"] == b["n_evaluated"] - len(silu) - len(attn)
    assert b["evaluators"]["self_check"]["SiluMul_v1{I=16}"]["status"] == "fail"
    b = replay.tier_b(r, snapshots, weights, step=0, evaluator="twins")
    assert b["status"] == "fail" and b["n_errors"] == len(silu), (b["status"], b.get("status_reason"), b["errors"][:2])
    assert all("failed its self-check" in e["reason"] for e in b["errors"])


def test_replayer_rejects_unknown_evaluator(consistent_toy):  # noqa: F811
    r, weights, _, _ = consistent_toy
    with pytest.raises(ValueError):
        replay.Replayer(r, None, weights, evaluator="fast")


# ---------------------------------------------------------------------------------------------------------
# the evidence file
# ---------------------------------------------------------------------------------------------------------


def test_check_writes_the_evidence_schema(tmp_path):
    out = tmp_path / "twins.json"
    assert cli.main(["twins", "--check", "--n-random", "4", "--n-edge", "4", "--seed", "7", "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["schema"] == TW.SCHEMA == "verity-gen/twins/v1" and rep["status"] == "pass" and rep["log"] is None
    assert set(rep["families"]) == {cls.twin_id for cls in TW.FAMILIES.values()}
    for fam in rep["families"].values():
        assert fam["status"] == "pass" and fam["n_random"] >= 3 and fam["mismatches"] == [] and fam["words"] > 0
        assert {"base_id", "n_random", "n_edge", "n_declined", "statics", "status", "seconds"} <= set(fam)
    assert set(rep["libraries"]) == {"tc_model", "fa2_model", "rms_triton_model"}


# ---------------------------------------------------------------------------------------------------------
# M1 (CAPTURE_BIG=1): the whole prefill through the twins
# ---------------------------------------------------------------------------------------------------------


def test_m1_tier_b_step0_full_prefill_with_twins(tmp_path):
    """The CLI in a subprocess (its own RSS high-water mark: the budget is per command, DESIGN §11): the whole prefill
    replayed through the twins against the step-0 snapshots — 2026-09-16 laptop: 1.0-1.4 GB."""
    import subprocess
    import sys
    from tests.check import test_replay_m1 as m1t
    why = m1t._available()
    if why is not None:
        pytest.skip(why)
    out = tmp_path / "replay_b0_twins.json"
    cmd = [sys.executable, "-m", "verity_vllm.pipeline.cli", "replay", "--log", str(m1t.LOG), "--snapshots", str(m1t.SNAPS), "--profile", m1t.PROFILE,
           "--tier", "b", "--step", "0", "--evaluator", "twins", "--out", str(out)]
    proc = subprocess.run(cmd, cwd=str(m1t.REPO), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    b = json.loads(out.read_text())
    print(f"\n{replay.summary_line(b)}")
    assert b["status"] == "pass" and b["evaluator"] == "twins", (b["status"], b.get("status_reason"), b["divergences"][:2], b["errors"][:2])
    assert b["n_instances"] == b["n_evaluated"] == b["n_compared"] == 9666 and b["n_diverged"] == 0 and b["n_skipped"] == 0
    assert b["partial"] is False and b["tokens"]["equal"] is True
    assert b["evaluators"]["twins"] == 9666 and b["evaluators"]["gates"] == 0 and b["evaluators"]["declined"] == 0
    assert all(sc["status"] == "pass" for sc in b["evaluators"]["self_check"].values())
    assert {e["spec"] for e in b["instances"]} == set(b["evaluators"]["self_check"])
    assert all(e["evaluator"] == "twins" and e["twin_id"] for e in b["instances"])
    assert b["max_rss_gb"] < m1t.RSS_LIMIT_GB, f"laptop budget: {b['max_rss_gb']:.2f} GB"
