"""gemm_rate/attn_rate.py's arithmetic: the model's shapes from its HF config, causal attention's useful MACs, chunked
prefill's steps, a window's MACs per SM clock against the dense rate and theta, and the table's shares beside the
linears."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gemm_rate"))
import attn_rate as ar


def test_the_default_model_is_the_served_1b_profile():
    m = ar.model(ar.CONFIG)
    assert (m["hq"], m["hkv"], m["d"], m["hidden"], m["ffn"], m["layers"]) == (32, 8, 64, 2048, 8192, 16)
    assert m["linears"] == {"qkv": (3072, 2048), "o": (2048, 2048), "gate_up": (16384, 2048), "down": (2048, 8192)}


def test_attention_counts_each_row_against_the_keys_at_or_before_it():
    m = ar.model(ar.CONFIG)
    rows = sum(range(1, 1025))
    assert ar.attn_macs(m, [1024]) == 2 * 32 * 64 * rows
    assert ar.attn_macs(m, ar.parse_prompt("8x1024")) == 8 * ar.attn_macs(m, [1024])
    assert ar.lin_macs(m, 8192) == 8192 * (3072 * 2048 + 2048 * 2048 + 16384 * 2048 + 2048 * 8192)


def test_prompts_and_chunks():
    assert ar.parse_prompt("16384") == [16384]
    assert ar.parse_prompt("4x1024") == [1024] * 4
    assert ar.chunks(16384, 8192) == [(0, 8192), (8192, 8192)]
    assert ar.chunks(10000, 4096) == [(0, 4096), (4096, 4096), (8192, 1808)]
    assert ar.chunks(1024, 0) == [(0, 1024)]


@pytest.mark.parametrize("kind", ["bf16", "s8"])
def test_a_window_at_f_mhz_has_theta_at_f_equal_to_its_share_of_peak(kind):
    clocks = 132 * 1830e6 * 2.0
    w = {"calls": 100, "span_ns": 2e9, "sm_clocks": clocks, "sms": 132}
    macs_call = int(0.75 * clocks * ar.PEAK[kind] / 100)
    p = ar.point(w, macs_call, kind)
    assert p["sm_mhz"] == pytest.approx(1830)
    assert p["of_peak"] == pytest.approx(0.75, rel=1e-9)
    assert p["theta_1830"] == pytest.approx(p["of_peak"], rel=1e-9)
    assert p["theta_1980"] == pytest.approx(p["of_peak"] * 1830 / 1980, rel=1e-9)
    assert p["us_call"] == pytest.approx(2e4)


def _row(arm, prompt, t, form, us, r=0):
    return {"kind": "window", "sms": 132, "arm": arm, "prompt": prompt, "t": t, "form": form, "round": r,
            "of_peak": 0.5, "theta_1830": 0.5, "theta_1980": 0.46, "sm_mhz": 1830.0, "us_call": us}


def test_the_table_sets_attention_beside_the_linears_at_the_same_rows():
    m = ar.model(ar.CONFIG)
    rows = [_row("fa3", "8x1024", 8192, "whole", us) for us in (90.0, 100.0, 300.0)]
    rows += [_row("lin", "M8192", 8192, "whole", 900.0), _row("fa3", "16384", 16384, "chunked", 500.0),
             _row("fa3", "16384", 16384, "whole", 400.0), _row("lin", "M16384", 16384, "whole", 1600.0),
             {"kind": "window", "sms": 0, "arm": "fa3", "prompt": "8x1024", "form": "whole", "t": 8192}]
    text = ar.table(rows, m, 8192)
    assert "| fa3 | 8x1024 | whole | 0.500 | 0.500 | 0.460 | 1830 | 100.0 | 3 |" in text
    ma, ml = ar.attn_macs(m, [1024] * 8), ar.lin_macs(m, 8192)
    assert f"| fa3 | 8x1024 | whole | {ma / (ma + ml):.3f} | {100 / 1000:.3f} | 100.0 | 900.0 |" in text
    ma, ml = ar.attn_macs(m, [16384]), ar.lin_macs(m, 16384)
    assert f"| fa3 | 16384 | chunked | {ma / (ma + ml):.3f} | {500 / 2300:.3f} | 500.0 | 1800.0 |" in text
    assert f"| fa3 | 16384 | whole | {ma / (ma + ml):.3f} | {400 / 2000:.3f} | 400.0 | 1600.0 |" in text
