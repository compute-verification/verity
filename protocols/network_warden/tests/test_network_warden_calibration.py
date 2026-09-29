"""The calibration tool on synthetic honest traces."""

import io
import json
import math
import random
from contextlib import redirect_stdout
from fractions import Fraction

from verity_network_warden import calibration, capacity


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
    (ingress,) = report["ingress"]
    assert ingress["bucket_ms"] == 100  # decision D's default
    assert ingress["rate_bps"] == capacity.ingress_rate(ingress["allocation"], 100)
    assert set(report["k_charge"]["share_of_k_by_span"]) == {"60s", "600s", "3600s", "86400s"}


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


def test_fraction_at_most_rounds_down():
    assert calibration.fraction_at_most(Fraction(1, 3)) == Fraction(1, 3)
    assert calibration.fraction_at_most(0.0751) == Fraction(3, 40)
    assert calibration.fraction_at_most(0.001) == Fraction(1, 64)
    for x in (0.07, 0.5, 1.9, 10.4):
        assert calibration.fraction_at_most(x) <= Fraction(x)


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
