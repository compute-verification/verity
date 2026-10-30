"""Core's `IntegrityProfile` counts what the Lean audit theorem bounds.

`Audit/OneStage.lean`'s `audit_count` bounds Pr[accept and at least K wrong units] by `L.miss K + εks + δlink`, and for
the uniform k-subset `Law.subset_miss` makes the sampling term C(n - K, k)/C(n, k). The profile's `accept(K)` for one
level drawn `Subset(k)` must be that ratio, and its wrong-unit bound the largest count the curve does not exclude at
delta. This pins both statements' text, so a change on either side fails here."""
from __future__ import annotations

from fractions import Fraction
from math import comb
from pathlib import Path

import pytest

from verity.ml.circuits.partition import Level
from verity.core.protocols.profile import IntegrityProfile, Subset

AUDIT = Path(__file__).resolve().parents[4] / "Security" / "Proofs" / "Flock" / "Soundness" / "Audit"
SUBSET_MISS = "(subset n k hk).miss K = ((n - K).choose k : ℝ≥0∞) / (n.choose k : ℝ≥0∞)"
AUDIT_COUNT = "L.miss K + εks + δlink"

#: The one-stage audits (lane one-stage-e2e): n units, k drawn, delta, the most wrong units admitted.
RUNS = [(1024, 16, 0.01, 254), (1024, 256, 2.0 ** -20, 46), (344, 128, 2.0 ** -20, 28), (183_680, 256, 2.0 ** -20, 9675)]


@pytest.mark.skipif(not (AUDIT / "Law.lean").exists(), reason="the audit layer's Lean is not on this branch")
def test_the_lean_statements_are_the_ones_the_profile_computes():
    law = (AUDIT / "Law.lean").read_text()
    one_stage = (AUDIT / "OneStage.lean").read_text()
    assert "theorem subset_miss" in law and SUBSET_MISS in law
    assert "theorem audit_count" in one_stage and AUDIT_COUNT in one_stage


@pytest.mark.parametrize("n, k, delta, at_most", RUNS)
def test_the_profile_is_the_lean_count_curve(n, k, delta, at_most):
    profile = IntegrityProfile((Level("unit"),), (n,), {"unit": Subset(k)}, delta)
    top, d = comb(n, k), Fraction(delta)
    miss = lambda K: Fraction(comb(n - K, k), top)  # noqa: E731  Law.subset_miss, as stated
    assert all(profile.accept(K) == miss(K) for K in (0, 1, at_most, at_most + 1, n - k, n))
    assert miss(at_most) >= d > miss(at_most + 1)
    assert int(profile.worst_case(lambda m: m)) == at_most
