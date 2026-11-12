"""Gate G2: the M1 log (`data/logs/m1.jsonl.gz`, produced on the pod) resolves with 0 Unsupported under
the pinned profile, with the expected instance counts per step.  Skipped when the log is absent."""

from __future__ import annotations

from pathlib import Path

import pytest

from verity_vllm.observe.fold.resolve_log import resolve_log
from verity_vllm.observe.fold.profiles.vllm_d9105ea80_sm89_eager import C, PROFILE, check_header

LOG = Path(__file__).resolve().parents[2] / "data" / "logs" / "m1.jsonl.gz"

pytestmark = pytest.mark.skipif(not LOG.exists(), reason=f"M1 log absent: {LOG}")


@pytest.fixture(scope="module")
def resolved():
    header, results, roots, notes = resolve_log(LOG, PROFILE)
    return header, results, roots


@pytest.mark.slow
def test_header_matches_profile(resolved):
    header, results, roots = resolved
    assert header is not None
    assert check_header(header) == []


@pytest.mark.slow
def test_zero_unsupported_over_all_steps(resolved):
    _, results, _ = resolved
    assert len(results) == 16, [r.step for r in results]
    errors = [(r.step, r.error) for r in results if r.error]
    assert errors == []
    residue = [(r.step, u.op, u.reason) for r in results for u in r.unsupported]
    assert residue == []


def test_instance_counts_per_step(resolved):
    _, results, _ = resolved
    NL = C["NL"]
    for r in results:
        R = r.rows
        fam = r.spec_family_rows()
        L = fam.get("TokenSelect_v1", 0)
        assert L >= 1, (r.step, fam)
        expected = {
            "Gemm_v1": 4 * NL * R + L,             # qkv, o, gate_up, down per layer per row; lm_head per logits row
            "RMSNormTriton_v1": R,                 # layer 0 input norm
            "RMSNormFusedCuda_v2": 2 * NL * R,
            "RoPE_v1": 2 * NL * R,                 # q and k
            "Attention_v3": NL * R,
            "SiluMul_v2": NL * R,
            "Embedding_v1": R,
            "TokenSelect_v1": L,
        }
        assert fam == expected, (r.step, r.phase_hint, fam, expected)
        # one Attention_v3{T} compute per layer per distinct T in the step (rows grouped by T)
        att = {k: v for k, v in r.spec_computes().items() if k.startswith("Attention_v3")}
        assert att and all(v == NL for v in att.values()), (r.step, att)


def test_prefill_then_decode(resolved):
    _, results, _ = resolved
    assert results[0].phase_hint == "prefill" and results[0].rows == 32
    assert all(r.phase_hint == "decode" and r.rows == 1 for r in results[1:])
    # prefill: one Attention_v3{T} per T = 1..32 per layer; decode: one per layer
    pre = results[0].spec_computes()
    assert sum(v for k, v in pre.items() if k.startswith("Attention_v3")) == C["NL"] * 32
    for r in results[1:]:
        att = {k: v for k, v in r.spec_computes().items() if k.startswith("Attention_v3")}
        assert len(att) == 1 and next(iter(att.values())) == C["NL"], (r.step, att)


def test_weight_roots_are_classified(resolved):
    from verity_vllm.observe.fold.profiles.vllm_d9105ea80_sm89_eager import root_class
    _, _, roots = resolved
    names = [rt.name for rs in roots.values() for rt in rs if rt.role in ("weight", "buffer")]
    assert names, "no weight roots in the preamble"
    unknown = [n for n in names if root_class(n) == "unknown"]
    assert unknown == []
    assert "model.embed_tokens.weight" in names and "model.norm.weight" in names
