from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze  # noqa: E402
import lag  # noqa: E402

TB = lag.KV_TOKEN_BYTES


def step(t0, t1, tokens, reqs, prefill, kv):
    return {"t0": t0, "t1": t1, "tokens": tokens, "reqs": reqs, "prefill_tokens": prefill, "kv_tokens": kv}


def test_shortfall_splits_eta_by_phase_and_takes_filler_off_the_total():
    # R = 1 GB/s over 10 s: a decode second reading 0.5 GB, half a second's gap, a prefill second reading 0.1 GB,
    # half a second before an empty step of 1 s (idle: the empty step schedules nothing), and 2 GB of filler
    w = 10**8
    steps = [step(0, 1, 2, 2, 0, (4 * 10**8) // TB), step(1.5, 2.5, 100, 1, 100, 0), step(3, 4, 0, 0, 0, 0)]
    e = analyze.shortfall(steps, 10.0, w, 2 * 10**9, 1.0, token_bytes=TB)
    assert e["decode"]["of_R"] == pytest.approx((w + (4 * 10**8) // TB * TB) / 1e9)
    assert e["prefill"] == {"s": 1, "of_R": pytest.approx(0.1), "eta": pytest.approx(0.09)}
    assert e["empty"] == {"s": 1, "of_R": 0.0, "eta": pytest.approx(0.1)}
    assert e["gaps"] == {"s": 0.5, "of_R": 0.0, "eta": pytest.approx(0.05)}
    assert e["idle"] == {"s": pytest.approx(6.5), "of_R": 0.0, "eta": pytest.approx(0.65)}
    assert e["eta_before_filler"] == pytest.approx(1 - (e["decode"]["of_R"] + 0.1) / 10)
    assert e["eta"] == pytest.approx(e["eta_before_filler"] - 0.2)


def test_phases_counts_a_requests_tokens_past_its_first_as_prefill():
    # vLLM's batch queue puts num_computed_tokens a step ahead: a record may say 0 prefill tokens for a chunk
    s = analyze.phases([step(0, 1, 8192, 8, 0, 0), step(1, 2, 9, 9, 0, 0)])
    assert [x["prefill_tokens"] for x in s] == [8184, 0]
