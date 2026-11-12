"""`test_gen_dense2`'s analytic rotary bound against Phi-3's captured cos/sin table, a run artefact written on a GPU pod, which
no tracked file holds. Skipped where it is absent.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.engine.test_gen_dense2 import phi3
from verity_vllm.program import model as analytic


def test_phi3_captured_rotary_table_is_within_the_analytic_bound():
    p = Path(__file__).resolve().parents[2] / "out" / "gen" / "lanes" / "dense2" / "runs" / "phi3_mini_32x16" / "capture" / "cos_sin_cache.npy"
    if not p.exists():
        pytest.skip("run artefact not present")
    r = analytic.check_cos_sin(phi3.C, p)
    assert r["status"] == "pass" and r["n_outside_bound"] == 0 and r["max_ulp"] <= 1, {k: r[k] for k in ("n_outside_bound", "max_ulp", "n_diff")}
