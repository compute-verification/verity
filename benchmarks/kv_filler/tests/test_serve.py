from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve  # noqa: E402


def test_filler_by_phase_splits_each_record_pro_rata_over_decode_prefill_and_between():
    steps = [{"t0": 1.0, "t1": 2.0, "tokens": 4, "reqs": 4, "prefill_tokens": 0},
             {"t0": 2.5, "t1": 3.0, "tokens": 900, "reqs": 2, "prefill_tokens": 0},  # prefills by its tokens
             {"t0": 3.0, "t1": 4.0, "tokens": 8, "reqs": 8, "prefill_tokens": 0}]
    offset_ms = -1000.0  # host seconds = (gpu ms + 1000) / 1000
    recs = [(500.0, 1500.0, 100, 0, 0, 0),  # host 1.5 .. 2.5: half in decode, half between
            (1500.0, 2500.0, 100, 0, 0, 1),  # host 2.5 .. 3.5: half prefill, half decode
            (3500.0, 3500.0, 7, 0, 0, 2)]  # an instant: between
    got = serve.filler_by_phase(steps, recs, offset_ms)
    assert got == {"decode": pytest.approx(100), "prefill": pytest.approx(50), "between": pytest.approx(57)}


def test_within_keeps_only_the_bytes_read_inside_the_trace():
    offset_ms = -1000.0
    recs = [(-500.0, 500.0, 100, 0),  # host 0.5 .. 1.5: the token gate's half before the trace
            (500.0, 1500.0, 100, 1),  # host 1.5 .. 2.5: inside
            (2500.0, 3500.0, 100, 2),  # host 3.5 .. 4.5: half after
            (3800.0, 3800.0, 7, 3),  # an instant after
            (1000.0, 1000.0, 5, 4)]  # an instant inside
    got = serve.within(recs, offset_ms, 1.0, 4.0)
    assert [r[3] for r in got] == [0, 1, 2, 4]
    assert [r[2] for r in got] == [pytest.approx(50), pytest.approx(100), pytest.approx(50), 5]
    assert got[0][:2] == pytest.approx((0.0, 500.0)) and got[2][:2] == pytest.approx((2500.0, 3000.0))
