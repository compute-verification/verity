"""share.py's `Model` on the lines of a made-up dcap run: what it fits (the read clock, the DRAM rate, lm_head's and
the linears' curves, the attention's line, the step's δ, FA2's efficiency) and what it reads back from the fit (a
step's and a prompt's time, and a row of the tables' ratios). Each number is chosen so the fit can be stated by hand."""

from __future__ import annotations

import pytest
import share as S

MHZ = 2000
CAP = 1024 * 188 * MHZ * 1e6  # the int8 peak at the read clock, MACs a second


def _lin(mhz, m, ms, lm=None):
    algos = [{"name": "o", "ms_alone": 9.0}] + ([{"name": "lm_head", "ms_alone": lm}] if lm else [])
    return {"mode": "lin", "mhz": mhz, "M": m, "ms": ms, "algos": algos}


def _at(mode, mhz, ctx, ms):
    return {"mode": mode, "mhz": mhz, "M": 16, "L": ctx, "ms": ms}


ROWS = [
    # dcap's two kinds of gate line, which have no clock
    {"mode": "gate", "what": "cublaslt", "m": 16, "ok": True},
    {"mode": "gate", "what": "attention", "B": 16, "L": 2048, "ok": True},
    {"mode": "bw", "mhz": 1993, "gb_per_s": 900.0}, {"mode": "bw", "mhz": 1994, "gb_per_s": 1000.0},
    # out of order; up to 256 rows a line's time holds lm_head's, which is its second algorithm here
    _lin(1995, 8192, 384.0), _lin(1996, 256, 16.0, lm=4.0), _lin(1997, 1, 2.0, lm=0.5), _lin(1998, 16, 4.0, lm=1.0),
    # 16 rows at L = 2,048 i read 2^31 i bytes of KV: 0.5 ms and 4 ms each, 0.1 ms over at the ends and under between
    _at("att", 1999, 2048, 4.6), _at("att", 2000, 4096, 8.4), _at("att", 2001, 6144, 12.4),
    _at("att", 2002, 8192, 16.6),
    # a whole step: its linears (4 ms), its attention as measured, and 0.7 ms and 0.1 ms more
    _at("step", 2003, 2048, 9.3), _at("step", 2004, 8192, 20.7),
]


@pytest.fixture(scope="module")
def m():
    return S.Model(ROWS)


def _att(p, eta):
    return S.causal(p) / (eta * CAP) * 1e3  # a causal prompt's attention at a share eta of the peak, ms


def test_the_fit_of_a_runs_lines(m):
    assert m.bw == 1e12  # the fastest read
    # of the twelve clocks in order, the one at half the 14 lines: the gate lines are counted, so it is the eighth,
    # one past the median's two (1998 and 1999)
    assert m.mhz == MHZ
    assert m.capacity == CAP and m.capacity_max == 1024 * 188 * 2430e6
    assert (m.lm_m, m.lm_ms) == ([1, 16, 256], [0.5, 1.0, 4.0])
    assert m.lin_m == [1, 16, 256, 8192] and m.lin_body == [1.5, 3.0, 12.0, 384.0]  # without lm_head
    assert [r["M"] for r in m.lin_rows] == m.lin_m
    assert sorted(m.att) == [(16, 2048), (16, 4096), (16, 6144), (16, 8192)]
    assert sorted(m.step) == [(16, 2048), (16, 8192)]
    # the least-squares line through the four attention points: 4 ms for each 2^31 bytes, and 0.5 ms
    assert m.att_bw == pytest.approx(250 * 2 ** 31) and m.att_c0 == pytest.approx(0.5)
    assert m.delta == pytest.approx(0.4)  # the mean of 0.7 and 0.1
    assert m.eta_fa2 == pytest.approx(S.causal(8192) / (0.07311 * CAP))  # causal 8,192 in FA2's 73.11 ms


def test_lm_head_and_the_linears_are_read_from_their_curves(m):
    assert m.lm(64) == pytest.approx(2.0)  # halfway in log rows from (16, 1) to (256, 4)
    assert m.lin(1) == 2.0 and m.lin(8192) == 384.0
    assert m.lin(16) == pytest.approx(4.0) and m.lin(256) == pytest.approx(16.0)  # the measured lines, lm_head in
    assert m.lin(64) == pytest.approx(6.0 + 2.0)  # the body halfway from 3 to 12, and lm_head for 64 rows
    assert m.lin(257) == pytest.approx(12.0 * 257 / 256)  # past 256 rows: no lm_head unless asked for
    assert m.lin(512) == pytest.approx(24.0)
    assert m.lin(8192, heads=16) == pytest.approx(385.0)  # a prefill batch of 16 prompts
    assert m.lin(64, heads=0) == pytest.approx(6.0)


def test_a_steps_time_is_the_measured_one_where_there_is_one_and_the_models_elsewhere(m):
    assert m.att_ms(16, 2048) == pytest.approx(4.5) and m.att_ms(512, 2048) == pytest.approx(128.5)
    assert m.step_ms(16, 2048) == pytest.approx(9.3 + 1.09)  # measured, and a step's non-GEMM time
    assert m.step_ms(16, 2048, measured=False) == pytest.approx(4.0 + 4.5 + 0.4 + 1.09)  # linears, attention, δ
    assert m.step_ms(16, 4096) == pytest.approx(4.0 + 8.5 + 0.4 + 1.09)  # no step line there: the model's
    assert m.step_ms(512, 2048) == pytest.approx(24.0 + 128.5 + 0.4 + 38.37 / 16)  # 512 tokens' non-GEMM time


def test_a_prompts_time_packs_prompts_into_a_batch_of_8192_tokens(m):
    assert m.prefill_ms(8192, 0.5) == pytest.approx(384.5 + _att(8192, 0.5) + 38.37)  # one prompt, one lm_head row
    assert m.prefill_ms(512, 0.25) == pytest.approx(385.0 / 16 + _att(512, 0.25) + 38.37 / 16)  # 16 prompts a batch
    assert m.prefill_ms(16384, 0.5) == pytest.approx(768.5 + _att(16384, 0.5) + 76.74)  # longer than a batch: alone


def test_a_rows_ratios_are_of_the_read_clocks_capacity_and_of_the_2430_mhz_one(m):
    r = m.row(0.25 * CAP * 0.010, 10.0, fill=0.1)  # a quarter of the read clock's capacity for 10 ms
    at_max = 0.25 * MHZ / 2430
    assert r == pytest.approx({"nominal/capacity": 0.25, "nominal/capacity_2430": at_max, "certified/nominal": 0.99,
                               "tau": S.A_V99 * at_max, "certified/capacity": 0.2475,
                               "forced_share": 0.99 * S.A_V99 * at_max, "hidden": 1.0, "filler_share": 0.1})
    busy = m.row(4 * m.capacity_max * 0.010, 10.0)  # four times the 2430 MHz capacity: τ = 4 A
    assert busy["tau"] == pytest.approx(4 * S.A_V99) and busy["filler_share"] == 0.0
    assert busy["hidden"] == pytest.approx((1 - 0.99 * 4 * S.A_V99) / S.A_V99)
