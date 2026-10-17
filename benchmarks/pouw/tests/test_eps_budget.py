"""eps_budget: Dither's skippable share as served, by design (D0–D5) and SKIP's per-component designs."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("pouw_eps_budget", Path(__file__).parents[1] / "eps_budget.py")
E = importlib.util.module_from_spec(spec)
sys.modules["pouw_eps_budget"] = E
spec.loader.exec_module(E)

CELLS = [(m, c, ph) for m in E.MODELS for c in E.CONTEXTS for ph in E.PHASES]


def test_the_strips_are_their_rank_over_depth():
    assert E.products("record", "QKᵀ", 128) == (128, 64, 64)            # L/d = 25% logical, two limbs executed
    assert E.products("A", "linears", 4096) == (4096, 0, 0)
    assert E.products("B", "QKᵀ", 128) == (128, 64, 0)                   # r = k/2 on one limb: checked
    assert E.products("B32", "linears", 4096) == (4096, 64, 32)          # CO's gap r/(k + r)
    assert E.products("C", "linears", 4096) == (3 * 4096, 0, 32)         # NCP: L of 3k known at the zero instance
    assert E.products("A", "P·V", 1024) == (1024, 66, 33)


def test_record_skips_the_strips_lm_head_and_decode():
    p = E.point("llama-3.1-8b", 2048, "prefill", "D0")
    assert p.eps_mac == pytest.approx(0.0113, abs=5e-4)
    assert set(p.parts["macs_skip"]) == {"strip linears", "strip qk", "strip pv", "lm_head"}
    for m, c, _ in CELLS:
        assert E.summary(m, c, "decode", "record").eps_mac > 1        # the clean products are the honest majority


@pytest.mark.parametrize("design", ["A", "B"])
def test_both_sided_and_co_dither_meet_the_target_everywhere(design):
    for m, c, ph in CELLS:
        s = E.summary(m, c, ph, design)
        assert s.eps_mac < 1e-3 and s.eps_time < 1e-3, (m, c, ph)
        assert s.gamma < 2.2, (m, c, ph)
        skipped = {r["component"] for r in s.rows if r["eps_mac"] or r["eps_time"]}
        assert skipped == {"P·V"}, (m, c, ph)


def test_co_at_rank_32_and_ncp_miss_it_on_q_k():
    for design in ("B32", "C"):
        s = E.summary("llama-3.1-8b", 2048, "prefill", design)
        qk = next(r for r in s.rows if r["component"] == "QKᵀ")
        assert s.eps_mac > 3e-3 and qk["eps_mac"] > 1e-3, design
    assert 4 < E.summary("llama-3.1-70b", 2048, "prefill", "C").gamma < 5


@pytest.mark.parametrize("design", ["CB", "Bc"])
def test_the_hybrids_meet_the_target_everywhere(design):
    lin = {"CB": "C", "Bc": "co32c"}[design]
    assert E.products(design, "linears", 4096) == E.products(lin, "linears", 4096)
    assert E.products(design, "QKᵀ", 128) == E.products("B", "QKᵀ", 128)
    for m, c, ph in CELLS:
        s = E.summary(m, c, ph, design)
        assert s.eps_mac < 3e-3 and s.eps_time < 3e-3 and s.gamma < 10, (m, c, ph)
    assert E.products("Bc", "linears", 4096) == (4096, 3 * 32, 0)       # every digit limb is a committed word


def test_decode_hides_the_edit_under_dram_and_the_zero_weight_gap_is_reported():
    for m, c, _ in CELLS:
        assert E.summary(m, c, "decode", "A").gamma < 1.05
        assert E.summary(m, c, "decode", "record").gamma > 1.05
        assert E.decode_dram(m, c) > 1


def test_compiled_protocol_adds_the_spot_checks_share():
    assert E.compiled_extra() == pytest.approx(0.002, abs=1e-5)


def test_main_writes_both_tables(tmp_path):
    assert E.main(["--md", str(tmp_path / "t.md"), "--json", str(tmp_path / "t.json")]) == 0
    text = (tmp_path / "t.md").read_text()
    assert "D0 ε MAC / time" in text and "A: ε_skip MAC / time, γ" in text and "| **whole** |" in text
