from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lag  # noqa: E402

TB = lag.KV_TOKEN_BYTES


def step(t0, t1, tokens, prefill, kv):
    return {"t0": t0, "t1": t1, "tokens": tokens, "prefill_tokens": prefill, "kv_tokens": kv}


def test_a_side_stream_encode_lags_a_step_at_any_rate():
    steps = [step(0, 0.1, 1000, 1000, 1000), step(0.1, 0.11, 1, 0, 1001), step(0.11, 0.12, 1, 0, 1002)]
    r = lag.replay(steps, lag.Rates(1e12, 1e12, 1e12))
    assert r["plain_bytes"] == [0, 1000 * TB, TB]


def test_a_prefill_burst_is_read_plain_until_drained():
    # 1,000 prompt tokens (131 MB), then decode steps of 10 ms beside an encode of 3.1 GB/s: about 42 steps' worth
    steps = [step(0, 0.05, 1000, 1000, 1000)] + [step(0.05 + 0.01 * i, 0.06 + 0.01 * i, 1, 0, 1001 + i) for i in range(80)]
    r = lag.replay(steps, lag.Rates(decode_GBps=TB * 1000 / 0.42 / 1e9, prefill_GBps=0.0, idle_GBps=3.47))
    plain = r["plain_bytes"]
    assert plain[0] == 0
    assert plain[1] == 1000 * TB  # the whole prompt is pending when the first decode step reads it
    drained = next(i for i, b in enumerate(plain[1:], 1) if b <= 2 * TB)
    assert 41 <= drained <= 46
    assert r["queue_peak_bytes"] == 1000 * TB


def test_idle_time_drains_at_the_idle_rate():
    steps = [step(0, 0.01, 100, 100, 100), step(1.01, 1.02, 1, 0, 101)]
    r = lag.replay(steps, lag.Rates(0.0, 0.0, idle_GBps=100 * TB / 1e9 / 2))
    assert r["plain_bytes"][1] == 50 * TB


def test_windows_spread_pro_rata():
    steps = [step(0.0, 1.0, 0, 0, 0), step(1.0, 3.0, 0, 0, 0)]
    g = lag.windows(steps, [1e9, 4e9], 0.0, 3.0, 1.0, 1.0)
    assert g == [1.0, 2.0, 2.0]


def test_draining_slows_only_the_steps_that_drain():
    steps = [step(0, 0.1, 10, 10, 10), step(0.1, 0.2, 1, 0, 11), step(0.2, 0.3, 1, 0, 12)]
    r = lag.replay(steps, lag.Rates(decode_GBps=200 * TB / 1e9, prefill_GBps=0.0, idle_GBps=0.0, decode_slowdown=0.5))
    assert r["plain_bytes"] == [0, 10 * TB, TB]
    assert abs(r["added_s"] - 0.1) < 1e-12  # the first decode step drains 10 tokens and its own 1; the next drains 1
    assert abs(r["busy_s"] - 0.3) < 1e-12


def test_fused_puts_a_prefill_encode_on_its_step():
    steps = [step(0, 0.1215, 4096, 4096, 0), step(0.1215, 0.1415, 64, 0, 4096)]
    f = lag.fused(steps, alone_GBps=3.47, decode_GBps=25.0)
    assert abs(f["added_s"] - (4096 * TB / 3.47e9 + 64 * TB / 25e9)) < 1e-12
    assert 0.15 < f["added_s"] < 0.16
