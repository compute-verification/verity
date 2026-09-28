"""The calibration tool on synthetic honest traces."""

import io
import json
import random
from contextlib import redirect_stdout

from verity_network_timing import calibration, capacity


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


def test_slow_sessions_are_honest_misses():
    """Sessions that produce one frame every 3 buckets can't meet ρ ≥ 1 frame per bucket, so their windows miss."""
    report = calibration.calibrate(synthetic_trace(seed=3, gap=3, frames_per_session=12))
    assert report["clock_sync"]["honest_miss_rate"] > 0 and not report["clock_sync"]["meets_target"]


def test_cli(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(synthetic_trace()))
    out = io.StringIO()
    with redirect_stdout(out):
        assert calibration.main([str(path), "--ingress-bucket-ms", "200", "--target-k-share", "0.01"]) == 0
    report = json.loads(out.getvalue())
    assert report["ingress"][0]["bucket_ms"] == 200
    assert "meets_target" in report["k_charge"]["target"]
