"""The calibration tool on synthetic honest traces."""

import io
import json
import math
import random
from contextlib import redirect_stdout
from fractions import Fraction

import pytest

from verity_network_warden import calibration, capacity
from verity_network_warden.schedule import ClockSync, LogicalFrame, SyncSet


def synthetic_trace(seed=0, sessions=40, window_buckets=20, frames_per_session=6, gap=1, latency=(1, 4)):
    """Sessions arrive at random buckets. Each session's first frame is ready 1–4 buckets after its request, and every
    later frame `gap` buckets after the previous one."""
    rng = random.Random(seed)
    anchors = sorted(rng.randrange(0, 5 * window_buckets) for _ in range(sessions))
    frames = []
    for s, a in enumerate(anchors):
        ready = a + rng.randint(*latency)
        for i in range(frames_per_session):
            frames.append({"session": f"s{s}", "index": i, "anchor": a, "ready": ready + i * gap})
    frames.sort(key=lambda f: (f["anchor"], f["session"], f["index"]))
    arrivals = [0] * (6 * window_buckets)
    for a in anchors:
        arrivals[a] += 1
    return {"bucket_ms": 100, "window_buckets": window_buckets, "frame_bytes": 4096, "queue_capacity": 4096,
            "configurations": [{"name": "synthetic", "egress": [{"link": "responses", "frames": frames}],
                                "ingress": [{"link": "requests", "arrivals": arrivals}]}]}


def test_percentile_nearest_rank():
    assert calibration.percentile([1, 2, 3, 4], 50) == 2
    assert calibration.percentile([1, 2, 3, 4], 100) == 4
    assert calibration.percentile([5], 0.01) == 5


def test_calibrate_honest_trace():
    report = calibration.calibrate(synthetic_trace())
    sync = report["clock_sync"]
    assert sync["size"] <= 16 * 8 and sync["bits_per_window"] <= 7
    assert sync["honest_miss_rate"] == 0 and sync["meets_target"]
    (egress,) = report["egress"]
    assert egress["allocation"] >= 1 and not egress["replay"]["overflow"]
    assert egress["replay"]["p99_delay_buckets"] <= 1
    assert 0 <= egress["padding_share"] <= 1
    assert 0 <= egress["shaping_delay_ms"]["p50"] <= egress["shaping_delay_ms"]["p99"] <= egress["shaping_delay_ms"]["max"]
    (ingress,) = report["ingress"]
    assert ingress["bucket_ms"] == 100  # τ_in's default
    assert ingress["rate_bps"] == capacity.ingress_rate(ingress["allocation"], 100)
    assert set(report["k_charge"]["share_of_k_by_span"]) == {"60s", "600s", "3600s", "86400s"}


def test_a_bucket_over_the_queue_capacity_overflows_at_every_allocation():
    """Five frames in one bucket with Q = 3 overflow whatever r is, so sizing raises instead of searching forever. At Q
    = 3, three frames still fit."""
    with pytest.raises(ValueError, match="hands 5 frames, more than the queue capacity Q = 3"):
        calibration.least_allocation([0] * 5, queue_capacity=3, max_delay=1)
    assert calibration.least_allocation([0] * 3, queue_capacity=3, max_delay=1) == 2
    with pytest.raises(ValueError, match="queue capacity"):
        calibration.size_ingress_ms([10.0] * 5, 100, queue_capacity=3)


def test_ingress_allocation_is_minimal():
    trace = synthetic_trace(seed=1, sessions=80)
    arrivals = trace["configurations"][0]["ingress"][0]["arrivals"]
    r = calibration.size_ingress(arrivals, 100)["allocation"]
    if r > 1:
        hands = [t for t, c in enumerate(arrivals) for _ in range(c)]
        ok, delays = calibration.replay_fifo(hands, r - 1, 1 << 16)
        assert not (ok and calibration.percentile(delays, 99) <= 1)


def test_wider_ingress_bucket_cuts_the_rate():
    trace = synthetic_trace(seed=2)
    at_100 = calibration.calibrate(trace)["k_charge"]["ingress_bps"]
    at_1000 = calibration.calibrate(trace, ingress_bucket_ms=1000)["k_charge"]["ingress_bps"]
    assert at_1000 < at_100


def test_slow_sessions_are_paced_below_one_frame_per_bucket():
    """Sessions that produce one frame every 3 buckets are met by ρ = 1/3."""
    report = calibration.calibrate(synthetic_trace(seed=3, gap=3, frames_per_session=12))
    sync = report["clock_sync"]
    assert Fraction(sync["rhos"][0]) <= Fraction(1, 3)
    assert sync["honest_miss_rate"] == 0 and sync["meets_target"]


def test_a_mid_session_stall_is_an_honest_miss():
    """A session that stalls for 30 buckets after its third frame is late for every value in Σ_sync, so its window
    misses."""
    trace = synthetic_trace(seed=4)
    frames = trace["configurations"][0]["egress"][0]["frames"]
    for f in frames:
        if f["session"] == "s0" and f["index"] >= 3:
            f["ready"] += 30
    sync = calibration.calibrate(trace)["clock_sync"]
    assert sync["honest_miss_rate"] > 0 and not sync["meets_target"]


def test_prover_does_not_trade_pace_for_one_bucket_of_offset():
    """The first frame is ready 2 buckets after the anchor and the rest pace at 4 per bucket from bucket 3. ρ = 1 is met
    at δ = 2, and ρ = 4 only at δ = 3. The prover takes (3, 4): its frames wait at most a bucket, where (2, 1) would hold
    the 40th frame 29 buckets."""
    frames = [LogicalFrame(b"", 10, i) for i in range(40)]
    ready = [12] + [13 + i // 4 for i in range(1, 40)]
    syncs, hands, misses = calibration.choose_syncs(frames, ready, SyncSet(tuple(range(6)), (1, 4)), 600)
    assert misses == 0 and syncs == [ClockSync(3, 4)]
    assert max(h - r for h, r in zip(hands, ready)) <= 1


def paced_configuration(name, rate, windows, window_buckets=20):
    """One session per window, anchored 2 buckets in, producing frames at `rate` per bucket from the next bucket."""
    frames = []
    for w in range(windows):
        anchor = w * window_buckets + 2
        for i in range(2 * rate.numerator + 1):
            frames.append({"session": f"{name}{w}", "index": i, "anchor": anchor,
                           "ready": anchor + 1 + i * rate.denominator // rate.numerator})
    return {"name": name, "egress": [{"link": "responses", "frames": frames}]}


def test_weighted_percentile_with_unit_weights_is_the_percentile():
    rng = random.Random(7)
    for n in (1, 2, 5, 40):
        xs = [Fraction(rng.randrange(1, 50), rng.randrange(1, 9)) for _ in range(n)]
        for p in (0.01, 25, 50, 99.9, 100):
            assert calibration.weighted_percentile(xs, [1] * n, p) == calibration.percentile(xs, p)
    assert calibration.weighted_percentile([Fraction(2), Fraction(1, 2)], [1, 3], 50) == Fraction(1, 2)
    with pytest.raises(ValueError):
        calibration.weighted_percentile([Fraction(1)], [0], 50)


def test_top_rho_is_the_median_over_windows():
    """One configuration spans 8 windows at one frame every 2 buckets, and two others span one window each at 2 frames
    per bucket. The median over configurations is 2, the rate of 2 of the 10 windows; over windows it is 1/2, so Σ_sync's
    ρ stops at 1/2. Cutting the long configuration into 8 one-window ones leaves Σ_sync as it was."""
    half, two = Fraction(1, 2), Fraction(2)
    configs = [paced_configuration("a", half, 8), paced_configuration("b", two, 1), paced_configuration("c", two, 1)]
    trace = {"bucket_ms": 100, "window_buckets": 20, "frame_bytes": 4096, "queue_capacity": 4096,
             "configurations": configs}
    old = calibration.choose_sync_set([1] * 10, [half, two, two], [half, two, two])
    new = calibration.choose_sync_set([1] * 10, [half, two, two], [half, two, two], typical_weights=[8, 1, 1])
    assert max(old.rhos) == two and new.rhos == (half,)
    sync = calibration.calibrate(trace)["clock_sync"]
    assert sync["rhos"] == ["1/2"] and sync["honest_miss_rate"] == 0
    a = configs[0]["egress"][0]["frames"]
    cut = [{"name": f"a{w}", "egress": [{"link": "responses",
                                         "frames": [dict(f, anchor=f["anchor"] - 20 * w, ready=f["ready"] - 20 * w)
                                                    for f in a if f["anchor"] // 20 == w]}]} for w in range(8)]
    assert calibration.calibrate(dict(trace, configurations=cut + configs[1:]))["clock_sync"] == sync


def test_fraction_at_most_rounds_down():
    assert calibration.fraction_at_most(Fraction(1, 3)) == Fraction(1, 3)
    assert calibration.fraction_at_most(Fraction(2, 161)) == Fraction(2, 161)  # an 8.05 s step at 100 ms buckets
    assert calibration.fraction_at_most(0.0751, 64) == Fraction(3, 40)
    assert calibration.fraction_at_most(1e-4) == Fraction(1, 1024)
    for x in (0.0124, 0.07, 0.5, 1.9, 10.4):
        f = calibration.fraction_at_most(x)
        assert f <= Fraction(x) and Fraction(x) - f < Fraction(1, 1024)


def test_step_major_order_interleaves_windows():
    """A batching engine's canonical order is step-major: a session anchored in window 0 keeps producing while a
    session anchored in window 1 starts, one frame each per step. No window misses, and the second session isn't held
    behind the first."""
    frames = []
    for step in range(30):
        frames.append({"session": "a", "index": step, "anchor": 15, "ready": 16 + step})
        if step >= 10:
            frames.append({"session": "b", "index": step - 10, "anchor": 25, "ready": 26 + step - 10})
    trace = {"bucket_ms": 100, "window_buckets": 20, "frame_bytes": 4096, "queue_capacity": 4096,
             "configurations": [{"name": "interleaved", "egress": [{"link": "responses", "frames": frames}]}]}
    sync = calibration.calibrate(trace)["clock_sync"]
    assert sync["honest_miss_rate"] == 0
    assert sync["first_frame_latency_ms"]["p99"] <= 200
    lf = [LogicalFrame(b"", f["anchor"], f["index"]) for f in frames]
    syncs, hands, misses = calibration.choose_syncs(lf, [f["ready"] for f in frames],
                                                    SyncSet((0, 1, 2), (Fraction(1, 2), 1)), 20)
    assert misses == 0 and len(syncs) == 2
    assert all(h >= f["ready"] for h, f in zip(hands, frames))
    assert hands[frames.index(next(f for f in frames if f["session"] == "b"))] <= 27


def test_ingress_from_timestamps():
    """Arrival times allow an ingress bucket finer than the trace bucket, and the delivery delay is at most a bucket
    plus the FIFO delay."""
    arrivals = [0.0, 3.0, 3.5, 30.0, 61.0, 61.0, 61.0, 250.0]
    at_25 = calibration.size_ingress_ms(arrivals, 25)
    assert at_25["allocation"] >= 1
    assert at_25["delivery_delay_ms"]["max"] <= 25 * (1 + at_25["allocation"])
    hands = sorted(math.ceil(t / 25) for t in arrivals)
    if at_25["allocation"] > 1:
        ok, delays = calibration.replay_fifo(hands, at_25["allocation"] - 1, 1 << 16)
        assert not (ok and calibration.percentile(delays, 99) <= 1)
    trace = synthetic_trace()
    trace["configurations"][0]["ingress"] = [{"link": "requests", "arrival_ms": arrivals}]
    (ingress,) = calibration.calibrate(trace, ingress_bucket_ms=25)["ingress"]
    assert ingress["allocation"] == at_25["allocation"] and ingress["bucket_ms"] == 25


def test_cli(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(synthetic_trace()))
    out = io.StringIO()
    with redirect_stdout(out):
        assert calibration.main([str(path), "--ingress-bucket-ms", "200", "--target-k-share", "0.01"]) == 0
    report = json.loads(out.getvalue())
    assert report["ingress"][0]["bucket_ms"] == 200
    assert "meets_target" in report["k_charge"]["target"]


def test_cli_reports_a_trace_no_allocation_carries(tmp_path, capsys):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"window_buckets": 20, "queue_capacity": 3, "configurations": [
        {"name": "burst", "ingress": [{"link": "requests", "arrivals": [5]}]}]}))
    assert calibration.main([str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: a bucket hands 5 frames, more than the queue capacity Q = 3" in captured.err
