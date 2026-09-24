"""replay.py on the M1 capture (SmolLM2-135M, 32-token prompt, 16 steps; snapshots at steps 0, 1, 8).  Skipped unless
the log, the snapshots, L3's profile and the checkpoint in the HF cache exist and `CAPTURE_BIG=1` (DESIGN §11 / D69:
full-size builds are gated like `test_fold_m1.py`; run `CAPTURE_BIG=1 pytest tests/check/test_replay_m1.py -s`).

Laptop budget (DESIGN §11 / the 2026-09-16 kernel panic): every test here stays under 1.5 GB and well under 3 minutes;
the reference evaluator costs ~40 us per gate, so a full decode step (7.8M gates) is ~5.3 min and the 32-row prefill
~3 h — those run on the pod (`python -m verity_vllm.check.replay --tier b --step N` without `--layers`).  Here tier B is
exercised on a layer subset of the decode step and on a prefix of the prefill; measured 2026-09-16: tier A k=2 30 s /
1.0 GB, tier B step 1 layers {0, 29} 23 s / 0.85 GB, tier B step 0 first 80 instances 26 s / 0.85 GB."""
from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "data" / "logs" / "m1.jsonl.gz"
SNAPS = REPO / "data" / "logs" / "m1"
PROFILE = "vllm_d9105ea80_sm89_eager"
REPO_ID = "HuggingFaceTB/SmolLM2-135M"
RSS_LIMIT_GB = 1.5

# the specializations no snapshot evidence can reach: attention at T = 32 + s exists only at decode step s, and the
# snapshots cover steps 0, 1, 8 (T=40 IS at step 8, but its K/V rows come from steps 2..7, which have no snapshots)
UNCOVERED = {f"Attention_v3{{T={T},NH=9,KVH=3,D=64,BN=128}}" for T in range(34, 48)}


def _available() -> str | None:
    if os.environ.get("CAPTURE_BIG") != "1":
        return "full-size build: set CAPTURE_BIG=1 (DESIGN §11)"
    if not LOG.exists():
        return f"{LOG} missing (L1)"
    if not (SNAPS / "snapshots.json").exists():
        return f"{SNAPS}/snapshots.json missing (L1 snapshots)"
    if importlib.util.find_spec(f"verity_vllm.observe.profiles.{PROFILE}") is None:
        return f"profile {PROFILE} missing (L3)"
    from verity_vllm.check.fold_compare import hf_safetensors_glob
    import glob
    if not glob.glob(hf_safetensors_glob(REPO_ID)):
        return f"{REPO_ID} safetensors not in the HF cache"
    try:
        from verity_vllm.observe.memory import MemoryModel
        MemoryModel()
    except NotImplementedError:
        return "memory.py is a skeleton (L2)"
    return None


pytestmark = pytest.mark.skipif(_available() is not None, reason=str(_available()))


@pytest.fixture(scope="module")
def m1():
    from verity_vllm.check.fold_compare import SnapshotStore, WeightSource, fold_log, load_profile
    t0 = time.time()
    r = fold_log(LOG, load_profile(PROFILE), strict=None, with_program=True)
    store = SnapshotStore(SNAPS)
    cos_sin = SNAPS / "cos_sin_cache.npy"
    weights = WeightSource(r.params[-1][1], cos_sin=cos_sin if cos_sin.exists() else None, repo=REPO_ID)
    if not weights.available():
        pytest.skip("weights / cos_sin table unavailable")
    print(f"\nM1 fold+Program {time.time() - t0:.1f}s, {len(r.body.nodes)} instances, snapshot steps {store.steps}")
    return r, store, weights


def _under_budget(rep: dict, seconds: float) -> None:
    assert rep["max_rss_gb"] < RSS_LIMIT_GB, rep["max_rss_gb"]
    assert rep["seconds"] < seconds, rep["seconds"]


def test_tier_a_k2_covers_every_reachable_spec(m1):
    from verity_vllm.check import replay
    r, store, weights = m1
    a = replay.tier_a(r, store, weights, k=2, seed=0)
    print(replay.summary_line(a))
    _under_budget(a, 150)
    assert a["snapshot_steps"] == [0, 1, 8]
    specs = {m["spec"] for m in r.instances}
    assert a["n_specs"] == len(specs) == 59
    # every evaluation that ran passed: the bodies are exact wherever there is evidence
    assert all(e["status"] in ("pass", "skipped") for e in a["instances"]), [e for e in a["instances"] if e["status"] not in ("pass", "skipped")]
    assert a["summary"]["pass"] == 2 * (len(specs) - len(UNCOVERED)) == 90
    # ... and the uncovered list is exactly the evidence gap, with its reasons (the pass rule is not weakened for it)
    assert a["status"] == "fail"
    assert {u["spec"] for u in a["uncovered_specs"]} == UNCOVERED
    for u in a["uncovered_specs"]:
        if "T=40," in u["spec"]:
            assert "0 of 2 required" in u["reason"] and "not snapshotted" in u["reason"], u
        else:
            assert u["reason"].startswith("no snapshot step contains it"), u
    for spec in specs - UNCOVERED:
        c = a["coverage"][spec]
        assert c["pass"] == 2 and c["fail"] == 0 and c["error"] == 0 and c["evaluated"] == 2, (spec, c)
    # the oversize instances are covered at a child unit whose arguments fit
    cov = a["coverage"]
    assert cov["Embedding_v1{V=49152,H=576}"]["granularity"] == "child:GatherBf16x49152_v1"
    assert cov["Embedding_v1{V=49152,H=576}"]["child_arg_leaves"] == 1 + 49152
    assert cov["Gemm_v1{K=576,N=49152}"]["granularity"] == "child:GemmCoordinate_v1{K=576}"
    assert cov["Gemm_v1{K=576,N=49152}"]["child_arg_leaves"] == 2 * 576
    assert cov["Gemm_v1{K=576,N=49152}"]["evaluable"] == 3 and cov["TokenSelect_v1{V=49152}"]["evaluable"] == 3
    assert all(c["granularity"] == "instance" for s, c in cov.items() if "49152" not in s and c["granularity"] is not None)
    for e in a["instances"]:
        if e["granularity"].startswith("child:"):
            assert e["status"] == "pass" and e["n_returns"] == 1 and e["n_args"] in (1 + 49152, 2 * 576), e


def test_tier_b_decode_step_layers_subset(m1):
    """Step 1 (decode, T=33) on layers 0 and 29 plus the non-layer instances under the laptop budget: 0 divergences;
    Embedding and lm_head Gemm (28M leaves) skipped under `max_arg_leaves`, their consumers `partial` (fed from the
    snapshot), the sampled token reproduced."""
    from verity_vllm.check import replay
    r, store, weights = m1
    b = replay.tier_b(r, store, weights, step=1, max_arg_leaves=4_000_000, layers=[0, 29])
    print(replay.summary_line(b))
    _under_budget(b, 150)
    assert b["status"] == "pass", b.get("status_reason")
    assert b["n_instances"] == 304 and b["n_diverged"] == 0 and b["first_divergence"] is None
    assert b["n_skipped"] == b["n_budget_skips"] == 2 + 28 * 10
    assert b["n_evaluated"] == b["n_compared"] == 304 - b["n_skipped"] == 22
    assert b["partial"] is True
    assert len(b["skipped"]) == 20 and b["skipped"][0]["spec"].startswith("Embedding")        # the ≤20 excerpt
    budget = {e["spec"].split("{")[0]: e["reason"] for e in b["instances"]
              if e["status"] == "skipped" and "max_arg_leaves" in e["reason"]}
    assert set(budget) == {"Embedding_v1", "Gemm_v1"} and "28312128 argument leaves" in budget["Gemm_v1"]
    lm_head = next(e["node"] for e in b["instances"] if e["spec"] == "Gemm_v1{K=576,N=49152}")
    ts = next(e for e in b["instances"] if e["spec"].startswith("TokenSelect"))
    assert ts["status"] == "pass" and ts["partial"] == [lm_head]          # logits from the lm_head's snapshot
    assert b["tokens"] == {"computed": [253], "sampled": [253], "equal": True}
    norm0 = next(e for e in b["instances"] if e["spec"].startswith("RMSNormTriton"))
    emb = next(e["node"] for e in b["instances"] if e["spec"].startswith("Embedding"))
    assert norm0["status"] == "pass" and norm0["partial"] == [emb]
    # inside the selected layers everything chains: `partial` only ever names skipped producers of this step, layer 0
    # falls back only where it reads the skipped Embedding (input norm; the fused post-attention norm's residual), and
    # attention's only outside reads are the KV rows
    skipped_nodes = {e["node"] for e in b["instances"] if e["status"] == "skipped"}
    for e in b["instances"]:
        if e["status"] == "pass":
            assert set(e["partial"]) <= skipped_nodes, e
            if "layers.0/" in e["name"]:
                assert e["partial"] == ([emb] if e["name"].endswith("layernorm") else []), e
            if e["spec"].startswith("Attention"):
                assert e["boundary"] == 64                                   # 32 K + 32 V rows from the prefill step
            elif "layers." in e["name"]:
                assert e["boundary"] == 0, e


def test_tier_b_prefill_prefix(m1):
    """Step 0 (prefill, 32 rows) is ~3 h on the laptop; the first 80 instances (32 Embedding rows skipped under the
    budget, 32 layer-0 input norms, 16 qkv Gemm rows) chain with 0 divergences in ~30 s."""
    from verity_vllm.check import replay
    r, store, weights = m1
    b = replay.tier_b(r, store, weights, step=0, max_arg_leaves=4_000_000, max_instances=80)
    print(replay.summary_line(b))
    _under_budget(b, 150)
    assert b["status"] == "pass", b.get("status_reason")
    assert b["n_instances"] == 9666 and b["n_diverged"] == 0
    assert b["n_evaluated"] == b["n_compared"] == 48 and b["n_skipped"] == b["n_budget_skips"] == 9666 - 48
    assert b["tokens"] == {"computed": [None], "sampled": [359], "equal": None}   # the sampler is beyond the prefix
    ev = [e for e in b["instances"] if e["status"] == "pass"]
    assert [e["spec"].split("{")[0] for e in ev] == ["RMSNormTriton_v1"] * 32 + ["Gemm_v1"] * 16
    assert all(len(e["partial"]) == 1 for e in ev[:32]) and all(e["partial"] == [] for e in ev[32:])
