"""Red-team F12: a run claims the interactive 2^-128 union only with an authenticated live coin log (numpy / CPU only)."""
from __future__ import annotations

import pytest

from archive.direct.ligerito.proof import FS_QUERIES_LOG2, soundness_claim

SND = {"total_log2": -128.01, "terms_log2": {"L1.proximity ((1+rho)/2)^|S|": -129.2, "L1.field k' n/|F|": -158.0,
                                             "ZC.sumcheck (3 col_vars + 7 + 4)/|F|": -178.9}}


def test_only_authenticated_live_claims_the_union():
    assert soundness_claim(SND, "live", verifier_record=True)["claimed_log2"] == SND["total_log2"]
    for coins, record in (("live", False), ("local", False), ("local", True)):
        c = soundness_claim(SND, coins, verifier_record=record)
        assert c["claimed_log2"] is None and c["basis"].startswith("none") and c["interactive_union_log2"] == SND["total_log2"]


def test_fiat_shamir_is_q_times_the_worst_term():
    c = soundness_claim(SND, "fiat-shamir", verifier_record=True)
    assert c["worst_term"].startswith("L1.proximity")
    assert c["claimed_log2"] == pytest.approx(FS_QUERIES_LOG2 - 129.2) and c["claimed_log2"] > -128
