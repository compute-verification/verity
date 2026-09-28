"""Calibrating constant-rate release from honest timing traces (the calibration spec; timing-channel doc §3).

From honest runs on the target hardware it picks, per boundary link:
- the egress allocation r;
- the clock-sync set Σ_sync, the (δ, ρ) values a window may declare;
- the ingress grid.

It reports the honest miss rate, the costs and the capacity. The protocol fixes τ_b, the frame size and the security
target; the ingress bucket (decision D, 100 ms by default) is a parameter.

The trace is JSON, with one entry per configuration:

    {"bucket_ms": 100, "window_buckets": 600, "frame_bytes": 4096, "queue_capacity": 65536,
     "configurations": [
       {"name": "peak-decode",
        "egress": [{"link": "responses",
                    "frames": [{"session": "s1", "index": 0, "anchor": 12, "ready": 14}, ...]}],
        "ingress": [{"link": "requests", "arrivals": [0, 2, 1, ...]}]}]}

- **Egress frames** are listed in the Program's canonical order, with anchors non-decreasing. `anchor` is the recorded
  bucket of the request a frame answers, and `ready` is the bucket in which the frame became ready to send.
- **Ingress `arrivals`** counts frames per trace bucket.

Percentiles use the nearest-rank method.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict, deque
from collections.abc import Sequence

from verity_network_timing import capacity
from verity_network_timing.schedule import ClockSync, LogicalFrame, SyncSet, eligible

SPANS_S = (60, 600, 3600, 86400)


def percentile(xs: Sequence[float], p: float) -> float:
    """The nearest-rank percentile, with p in [0, 100]."""
    if not xs:
        raise ValueError("percentile of an empty sequence")
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))]


def first_frame_latencies(frames: Sequence[dict]) -> list[int]:
    """Ready bucket minus anchor, for every session's first frame."""
    return [f["ready"] - f["anchor"] for f in frames if f["index"] == 0]


def session_rates(frames: Sequence[dict]) -> list[float]:
    """Each session's production rate, in frames per bucket, from its first to its last frame."""
    by_session: dict[str, list[dict]] = defaultdict(list)
    for f in frames:
        by_session[f["session"]].append(f)
    rates = []
    for fs in by_session.values():
        if len(fs) > 1:
            fs = sorted(fs, key=lambda f: f["index"])
            rates.append((len(fs) - 1) / max(1, fs[-1]["ready"] - fs[0]["ready"]))
    return rates


def choose_sync_set(latencies: Sequence[int], low_rates: Sequence[float], typical_rates: Sequence[float],
                    n_delta: int = 16, n_rho: int = 8) -> SyncSet:
    """The spec's §2:
    - **δ** runs from 0 up to the p99.99 first-frame latency, in `n_delta` steps;
    - **ρ** runs from the p0.01 production rate (at least 1) up to the p50 rate, in `n_rho` values.

    `low_rates` should come from the heaviest configuration, and `typical_rates` from the lightest."""
    hi = math.ceil(percentile(latencies, 99.99)) if latencies else 0
    step = max(1, math.ceil(hi / max(1, n_delta - 1)))
    deltas = tuple(i * step for i in range(n_delta))
    lo_r = max(1, math.floor(percentile(low_rates, 0.01))) if low_rates else 1
    hi_r = max(lo_r, math.ceil(percentile(typical_rates, 50))) if typical_rates else lo_r
    if n_rho == 1 or hi_r == lo_r:
        rhos: tuple[int, ...] = (lo_r,)
    else:
        rhos = tuple(sorted({round(lo_r + i * (hi_r - lo_r) / (n_rho - 1)) for i in range(n_rho)}))
    return SyncSet(deltas, rhos)


def _hand(frames: Sequence[LogicalFrame], ready: Sequence[int], sync: ClockSync, last: int) -> tuple[bool, list[int]]:
    """Hand a window's frames under `sync`, after the previous hand time `last`. Returns whether every frame was ready
    by its logical bucket, and the actual hand times (a late frame goes when ready, delaying those behind it)."""
    ok, out, h_last, a_last = True, [], last, last
    for f, r in zip(frames, ready, strict=True):
        h_last = max(eligible(f, sync), h_last)
        ok = ok and r <= h_last
        a_last = max(h_last, r, a_last)
        out.append(a_last)
    return ok, out


def choose_syncs(frames: Sequence[LogicalFrame], ready: Sequence[int], sync_set: SyncSet,
                 window_buckets: int) -> tuple[list[ClockSync], list[int], int]:
    """Replay the honest prover's choice, window by window. It declares the met sync with the least latency (smallest δ,
    then largest ρ). A window that no value meets is a miss, which is a violation; the replay continues with the most
    lenient value. Returns the chosen syncs, the actual hand times and the number of misses."""
    if any(a.anchor > b.anchor for a, b in zip(frames, frames[1:])):
        raise ValueError("egress frames must be in canonical order with non-decreasing anchors")
    windows = (max((f.anchor for f in frames), default=0) // window_buckets) + 1
    order = sorted(sync_set, key=lambda s: (s.delta, -s.rho))
    lenient = max(sync_set, key=lambda s: (s.delta, -s.rho))
    syncs, hands, misses, last, k = [], [], 0, 0, 0
    for w in range(windows):
        j = k
        while j < len(frames) and frames[j].anchor // window_buckets == w:
            j += 1
        chunk, chunk_ready = frames[k:j], ready[k:j]
        for sync in order:
            ok, out = _hand(chunk, chunk_ready, sync, last)
            if ok:
                break
        else:
            misses += 1
            sync = lenient
            _, out = _hand(chunk, chunk_ready, sync, last)
        syncs.append(sync)
        hands.extend(out)
        last = out[-1] if out else last
        k = j
    return syncs, hands, misses


def replay_fifo(hands: Sequence[int], allocation: int, queue_capacity: int) -> tuple[bool, list[int]]:
    """Replay the constant-rate FIFO warden on frames handed at `hands` (non-decreasing buckets). Returns whether the
    queue stayed within its capacity, and each frame's queueing delay in buckets."""
    queue: deque[int] = deque()
    delays, k, t, ok = [], 0, 0, True
    while k < len(hands) or queue:
        while k < len(hands) and hands[k] == t:
            queue.append(t)
            k += 1
        ok = ok and len(queue) <= queue_capacity
        for _ in range(min(len(queue), allocation)):
            delays.append(t - queue.popleft())
        t += 1
    return ok, delays


def size_allocation(hands: Sequence[int], queue_capacity: int, headroom: float = 2.0,
                    max_delay: int = 1) -> dict:
    """The spec's §3. Start at r = ⌈headroom · p99.9⌉ of per-bucket demand, then raise r until there is no overflow and
    the p99 queueing delay is at most `max_delay` buckets."""
    if not hands:
        return {"allocation": 1, "demand": {}, "overflow": False, "p99_delay_buckets": 0, "utilization": 0.0}
    horizon = max(hands) + 1
    demand = [0] * horizon
    for h in hands:
        demand[h] += 1
    r = max(1, math.ceil(headroom * percentile(demand, 99.9)))
    while True:
        ok, delays = replay_fifo(hands, r, queue_capacity)
        p99 = percentile(delays, 99)
        if ok and p99 <= max_delay:
            break
        r += 1
    return {
        "allocation": r,
        "demand": {f"p{p}": percentile(demand, p) for p in (50, 99, 99.9)} | {"max": max(demand)},
        "overflow": not ok,
        "p99_delay_buckets": p99,
        "utilization": sum(demand) / (r * horizon),
    }


def size_ingress(arrivals: Sequence[int], trace_bucket_ms: int, ingress_bucket_ms: int = 100,
                 queue_capacity: int = 1 << 16, max_delay: int = 1) -> dict:
    """The smallest ingress allocation r_in that carries honest ingress without overflow, with a p99 delay of at most
    `max_delay` ingress buckets. The ingress bucket τ_in is decision D, 100 ms by default."""
    if ingress_bucket_ms % trace_bucket_ms:
        raise ValueError("the ingress bucket must be a multiple of the trace bucket")
    k = ingress_bucket_ms // trace_bucket_ms
    counts = [sum(arrivals[i : i + k]) for i in range(0, len(arrivals), k)]
    hands = [t for t, c in enumerate(counts) for _ in range(c)]
    r = 1
    while hands:
        ok, delays = replay_fifo(hands, r, queue_capacity)
        if ok and percentile(delays, 99) <= max_delay:
            break
        r += 1
    rate = capacity.ingress_rate(r, ingress_bucket_ms)
    return {"allocation": r, "bucket_ms": ingress_bucket_ms, "rate_bps": rate}


def calibrate(trace: dict, *, ingress_bucket_ms: int = 100, n_delta: int = 16, n_rho: int = 8,
              headroom: float = 2.0, epsilon: float = 1e-4, target_k_share: float | None = None,
              target_span_s: int = 3600) -> dict:
    """Size every link of the trace and report as the spec's §5 lays out."""
    bucket_ms = trace.get("bucket_ms", 100)
    T = trace["window_buckets"]
    frame_bytes = trace.get("frame_bytes", 4096)
    q = trace.get("queue_capacity", 1 << 16)
    configs = trace["configurations"]

    per_link_frames: dict[str, list[list[dict]]] = defaultdict(list)
    per_link_arrivals: dict[str, list[list[int]]] = defaultdict(list)
    latencies, low_rates, typical_rates = [], [], []
    for cfg in configs:
        cfg_rates = []
        for link in cfg.get("egress", []):
            per_link_frames[link["link"]].append(link["frames"])
            latencies += first_frame_latencies(link["frames"])
            cfg_rates += session_rates(link["frames"])
        for link in cfg.get("ingress", []):
            per_link_arrivals[link["link"]].append(link["arrivals"])
        if cfg_rates:
            low_rates.append(percentile(cfg_rates, 0.01))
            typical_rates.append(percentile(cfg_rates, 50))
    sync_set = choose_sync_set(latencies, low_rates, typical_rates, n_delta, n_rho)

    egress, windows_total, misses_total, chosen_deltas = [], 0, 0, []
    for name, runs in per_link_frames.items():
        hands_all, link_windows, link_misses = [], 0, 0
        offset = 0
        for frames in runs:
            lf = [LogicalFrame(b"", f["anchor"], f["index"]) for f in frames]
            syncs, hands, misses = choose_syncs(lf, [f["ready"] for f in frames], sync_set, T)
            link_windows += len(syncs)
            link_misses += misses
            chosen_deltas += [s.delta for s in syncs]
            hands_all += [h + offset for h in hands]
            offset = (max(hands_all) + 1) if hands_all else offset
        alloc = size_allocation(hands_all, q, headroom)
        r = alloc["allocation"]
        egress.append({
            "link": name, "direction": "egress", "frame_bytes": frame_bytes, "allocation": r,
            "bandwidth_bps": r * frame_bytes * 8 / (bucket_ms / 1000),
            "utilization": alloc["utilization"], "padding_share": 1 - alloc["utilization"],
            "demand": alloc["demand"], "replay": {"overflow": alloc["overflow"],
                                                  "p99_delay_buckets": alloc["p99_delay_buckets"]},
            "envelope_bits_per_window": capacity.envelope_bits(T, r),
            "honest_windows": link_windows, "honest_misses": link_misses,
        })
        windows_total += link_windows
        misses_total += link_misses

    ingress = []
    for name, runs in per_link_arrivals.items():
        joined = [c for run in runs for c in run]
        ingress.append({"link": name, "direction": "ingress"} | size_ingress(joined, bucket_ms, ingress_bucket_ms, q))

    miss_rate = misses_total / windows_total if windows_total else 0.0
    windows_per_day = 86400 * 1000 / (T * bucket_ms)
    ingress_bps = sum(i["rate_bps"] for i in ingress)
    k_charge = {f"{s}s": capacity.k_share(ingress_bps * s) for s in SPANS_S}
    report = {
        "grid": {"bucket_ms": bucket_ms, "window_buckets": T, "frame_bytes": frame_bytes, "queue_capacity": q},
        "egress": egress,
        "clock_sync": {
            "deltas": list(sync_set.deltas), "rhos": list(sync_set.rhos), "size": sync_set.size,
            "bits_per_window": sync_set.bits,
            "honest_miss_rate": miss_rate, "target": epsilon, "meets_target": miss_rate <= epsilon,
            "first_frame_latency_ms": ({"p50": percentile(chosen_deltas, 50) * bucket_ms,
                                        "p99": percentile(chosen_deltas, 99) * bucket_ms} if chosen_deltas else {}),
        },
        "capacity": {
            "egress_bits_per_day_one_clock": sync_set.bits * windows_per_day,
            "egress_bits_per_day_clock_per_link": sync_set.bits * windows_per_day * max(1, len(egress)),
        },
        "ingress": ingress,
        "k_charge": {"ingress_bps": ingress_bps, "share_of_k_by_span": k_charge,
                     "memory_after_wipe": "added from the wipe evidence; not measured here"},
        "provenance": {"configurations": [c.get("name", "") for c in configs]},
    }
    if target_k_share is not None:
        share = capacity.k_share(ingress_bps * target_span_s)
        report["k_charge"]["target"] = {"span_s": target_span_s, "share": target_k_share,
                                        "meets_target": share <= target_k_share}
    return report


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="network-timing-calibrate", description=__doc__.splitlines()[0])
    ap.add_argument("trace", help="honest timing trace (JSON)")
    ap.add_argument("--ingress-bucket-ms", type=int, default=100, help="the ingress bucket τ_in (decision D)")
    ap.add_argument("--deltas", type=int, default=16, help="offsets in Σ_sync")
    ap.add_argument("--rhos", type=int, default=8, help="rates in Σ_sync")
    ap.add_argument("--headroom", type=float, default=2.0)
    ap.add_argument("--epsilon", type=float, default=1e-4, help="target honest miss rate per window")
    ap.add_argument("--target-k-share", type=float, default=None)
    ap.add_argument("--target-span-s", type=int, default=3600)
    args = ap.parse_args(argv)
    with open(args.trace, encoding="utf-8") as fh:
        trace = json.load(fh)
    report = calibrate(trace, ingress_bucket_ms=args.ingress_bucket_ms, n_delta=args.deltas, n_rho=args.rhos,
                       headroom=args.headroom, epsilon=args.epsilon, target_k_share=args.target_k_share,
                       target_span_s=args.target_span_s)
    json.dump(report, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
