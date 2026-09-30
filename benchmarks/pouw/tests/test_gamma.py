"""The γ accounting reproduces the research store's NCP numbers (``docs/pouw/new-crypto.md`` §2, sub-word moves priced)."""

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


def test_the_total_is_the_headline():
    r = G.row("ncp-v1", 4096, 4096, 4096)
    assert r.binding == "shake256" and G.HASHES["shake256"].status == "measured"
    assert float(r.total_vs_plain_gemm) == float(r.arithmetic_vs_plain_gemm) + 0.75 * G.HASHES["shake256"].units_per_byte
    assert 396 < float(r.total_vs_plain_gemm) < 398 and 3.0 < float(r.arithmetic_vs_plain_gemm) < 3.01


def test_unbound_rows_carry_no_gamma():
    r = G.row("ncp-v1", 4096, 4096, 4096)
    unbound, *bound = r.bindings
    assert unbound["hash"] is None and unbound["gamma"] is None
    assert unbound["total_vs_plain_gemm"] == r.arithmetic_vs_plain_gemm
    assert [b["hash"] for b in bound] == list(G.HASHES) and all(b["gamma"] == r.gamma for b in bound)


def test_estimated_hash_rates_scale_shake256():
    s = G.HASHES["shake256"]
    for h in ("sha256", "turboshake128", "blake3"):
        v = G.HASHES[h]
        assert v.status == "estimated" and abs(v.gbps * v.units_per_byte - s.gbps * s.units_per_byte) < 1e-6


def test_pearl_has_no_gamma_and_a_floor():
    r = G.row("pearl-fp8-v4", 2048, 2048, 2048)
    assert r.gamma is None and float(r.gamma_floor.rstrip("%")) > 1.0


def test_pearl_floor_charges_b_f_a():
    assert G.row("pearl-fp8-v4", 8192, 8192, 8192).gamma_floor == "3.044%"   # (2r/n + 2r/m + 4r/k + 4/k)/Ω, r = 32


def test_uncovered_lm_head():
    import json
    model = json.loads(G.WORKLOAD.read_text())["model"]
    assert 0.09 < G.uncovered_fraction(model) < 0.11
