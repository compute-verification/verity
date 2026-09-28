"""The γ accounting reproduces the NCP numbers, sub-word moves priced."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("pouw_gamma", Path(__file__).parents[1] / "gamma.py")
G = importlib.util.module_from_spec(spec)
sys.modules["pouw_gamma"] = G
spec.loader.exec_module(G)


def test_ncp_matches_new_crypto():
    assert G.row("ncp-v1", 4096, 4096, 4096).gamma == "0.694%"          # 0.69% at 4,096²
    assert G.row("ncp-v1", 4096, 65536, 65536).gamma == "0.512%"        # 0.51% at the milestone
    assert G.row("ncp-v1", 2048, 2048, 2048).gamma == "0.887%"          # the benchmark model's worst shape


def test_ncp_hashing_is_394_per_mac():
    assert round(G.row("ncp-v1", 4096, 4096, 4096).hash_units_per_useful_mac) == 394


def test_pearl_has_no_gamma_and_a_floor():
    r = G.row("pearl-fp8-v4", 2048, 2048, 2048)
    assert r.gamma is None and float(r.gamma_floor.rstrip("%")) > 1.0


def test_uncovered_lm_head():
    import json
    model = json.loads(G.WORKLOAD.read_text())["model"]
    assert 0.09 < G.uncovered_fraction(model) < 0.11
