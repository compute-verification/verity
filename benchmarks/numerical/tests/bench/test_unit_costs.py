"""F6's unit-cost inventory: pinned Boolean counts of small subcircuits, the exact affine check, the draw and the work model."""
from __future__ import annotations

from fractions import Fraction
from math import comb

import pytest

from verity_numerical.bench import unit_costs as U

AMPERE = {"datatype": "bf16", "semantics": "sm80.mma.m16n8k16.bf16"}
HOPPER = {"datatype": "bf16", "semantics": "sm90.wgmma.m64n8k16.bf16"}


def _counts(template, params, statics=None):
    c = U.build(template, params, statics)
    return {m: c[m] for m in U.METRICS}


@pytest.mark.parametrize("template, params, want", [
    ("gemm-coordinate", {"K": 32, **AMPERE},
     {"inputs": 1024, "ands": 16294, "output_copies": 16, "committed": 17334, "nnz_AB": 474604, "lookups": 0}),
    ("gemm-coordinate", {"K": 32, **HOPPER},
     {"inputs": 1024, "ands": 15511, "output_copies": 16, "committed": 16551, "nnz_AB": 475144, "lookups": 0}),
    ("bias-add", {"N": 16}, {"inputs": 512, "ands": 11472, "output_copies": 256, "committed": 12240, "nnz_AB": 218800, "lookups": 0}),
    ("token-select-greedy", {"V": 64},
     {"inputs": 1024, "ands": 6788, "output_copies": 32, "committed": 7844, "nnz_AB": 386746, "lookups": 0}),
])
def test_pinned_counts(template, params, want):
    assert _counts(template, params) == want


def test_gemm_is_affine_in_k():
    pts = [(k, _counts("gemm-coordinate", {"K": k, **AMPERE})) for k in (32, 48, 64)]
    fit = U.affine(pts)
    assert fit["inputs"] == (Fraction(32), Fraction(0))
    assert {m: int(v) for m, v in U.at(fit, 32).items()} == pts[0][1]


def test_affine_refuses_a_bent_metric():
    pts = [(1, {"ands": 10}), (2, {"ands": 20}), (3, {"ands": 31})]
    with pytest.raises(ValueError, match="ands is not affine"):
        U.affine(pts, ("ands",))
    with pytest.raises(ValueError, match="at least three points"):
        U.affine(pts[:2], ("ands",))


def test_argmax_scan_law_predicts_a_later_band():
    builds = {v: _counts("token-select-greedy", {"V": v}) for v in range(2, 20)}
    law = U.argmax_scan_law(builds)
    assert law.constants == {"alpha": 103, "beta": 163, "inputs_per_V": 16}
    assert law(70) == _counts("token-select-greedy", {"V": 70})
    bent = {**builds, 15: {**builds[15], "ands": builds[15]["ands"] + 1, "committed": builds[15]["committed"] + 1}}
    with pytest.raises(ValueError, match="breaks at V = 15"):
        U.argmax_scan_law(bent)


def test_missing_piece_is_reported_not_guessed():
    job, c = U._build_job(U._key("gumbel-top-p-token-select", {"V": 64}))
    assert c["error"].startswith("NoPiece:")


def test_sample_size_is_the_record_k():
    assert U.sample_size() == 27_713


def test_draw_count_against_the_exact_hypergeometric():
    N, n, K, tail = 30, 7, 12, 1e-3
    pmf = [comb(n, k) * comb(N - n, K - k) / comb(N, K) for k in range(n + 1)]
    want = next(k for k in range(n + 1) if sum(pmf[k + 1:]) <= tail)
    mean, q = U.draw_count(N, n, K, tail)
    assert (mean, q) == (K * n / N, want)
    assert U.draw_count(10, 3, 4, 0.05) == (1.2, 2)


def test_work_model():
    assert U.work("gemm-coordinate", {"K": 2048, **AMPERE}) == {"ops_per_input": 4096, "bytes_per_input": 4096}
    assert U.work("scaled-mm-fp8-block-coordinate", {"K": 2560, "G": 128, "semantics": "sm90.wgmma.m64n8k32.e4m3"}) == \
        {"ops_per_input": 5180, "bytes_per_input": 2640}
    assert U.work("attention-head", {"D": 64, "BN": 128, "semantics": "sm80.fa2.bf16"}, 100) == \
        {"ops_per_input": 25900, "bytes_per_input": 25856}


def test_attention_t_of_a_stratum():
    full = {"parameters": {"NB": 3, "last": "full"}}
    part = {"parameters": {"NB": 3, "last": "partial"}}
    assert U._t_stats(full, 128)["mean"] == 384
    assert (U._t_stats(part, 128)["min"], U._t_stats(part, 128)["max"]) == (257, 383)
    runs = {"parameters": {"NB": 1, "last": "partial"}, "variable_counts": {"T": {"runs": [{"min": 1, "max": 3, "vus_per_value": 2}]}}}
    assert U._t_stats(runs, 128)["mean"] == 2
