"""Calibrating constant-rate release from honest timing traces.

From honest runs on the target hardware it picks, per boundary link:
- the egress allocation r;
- the clock-sync set Σ_sync, the (δ, ρ) values a window may declare;
- the ingress grid.

It reports the honest miss rate, the costs and the capacity. The protocol fixes τ_b, the frame size and the security
target. The ingress bucket τ_in is 100 ms, chosen from a calibration of honest traces; the grid stays
configurable, so other buckets can be compared.

The trace is JSON, with one entry per configuration:

    {"bucket_ms": 100, "window_buckets": 600, "frame_bytes": 4096, "queue_capacity": 65536,
     "configurations": [
       {"name": "peak-decode",
        "egress": [{"link": "responses",
                    "frames": [{"session": "s1", "index": 0, "anchor": 12, "ready": 14}, ...]}],
        "ingress": [{"link": "requests", "arrival_ms": [3.2, 41.0, 41.5, ...]}]}]}

- **Egress frames** are listed in the Program's canonical order (step-major for a batching engine). `anchor` is the recorded
  bucket of the request a frame answers, and `ready` is the first bucket that starts after the frame is ready to send.
- **Ingress `arrival_ms`** is each frame's arrival time at the certifier, in ms. A trace may give `arrivals` instead, the
  count of frames per trace bucket; then the ingress bucket must be a multiple of the trace bucket.

Percentiles use the nearest-rank method.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict, deque
from collections.abc import Sequence
from fractions import Fraction

from verity.core.protocols.certifier import capacity
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, SyncSet, eligible

SPANS_S = (60, 600, 3600, 86400)
MAX_RHO_DENOMINATOR = 1024


def percentile(xs: Sequence[float], p: float) -> float:
    """The nearest-rank percentile, with p in [0, 100]."""
    if not xs:
        raise ValueError("percentile of an empty sequence")
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))]


def weighted_percentile(xs: Sequence[Fraction], weights: Sequence[int], p: float) -> Fraction:
    """The nearest-rank percentile of `xs` with each `xs[i]` repeated `weights[i]` times: of W = Σ weights, the least x
    whose cumulative weight reaches rank ⌈p/100 · W⌉ (at least 1). With unit weights it is `percentile`."""
    if len(xs) != len(weights) or any(w < 0 for w in weights) or sum(weights) == 0:
        raise ValueError("weighted percentile needs one non-negative weight per value and a positive total")
    total = sum(weights)
    rank, acc = min(total, max(1, math.ceil(p / 100 * total))), 0
    for x, w in sorted(zip(xs, weights, strict=True)):
        acc += w
        if acc >= rank:
            return x
    raise AssertionError("unreachable: the cumulative weight reaches the total")


def first_frame_latencies(frames: Sequence[dict]) -> list[int]:
    """Ready bucket minus anchor, for every session's first frame."""
    return [f["ready"] - f["anchor"] for f in frames if f["index"] == 0]


def session_rates(frames: Sequence[dict]) -> list[Fraction]:
    """Each session's production rate, in frames per bucket, from its first to its last frame."""
    by_session: dict[str, list[dict]] = defaultdict(list)
    for f in frames:
        by_session[f["session"]].append(f)
    rates = []
    for fs in by_session.values():
        if len(fs) > 1:
            fs = sorted(fs, key=lambda f: f["index"])
            rates.append(Fraction(len(fs) - 1, max(1, fs[-1]["ready"] - fs[0]["ready"])))
    return rates


def fraction_at_most(x: Fraction | float, max_denominator: int = MAX_RHO_DENOMINATOR) -> Fraction:
    """The largest p/q ≤ x with 1 ≤ q ≤ `max_denominator`, and at least 1/max_denominator. Rounding down keeps a rate
    lenient: a session at rate x still meets it."""
    x = Fraction(x)
    best = Fraction(1, max_denominator)
    for q in range(1, max_denominator + 1):
        best = max(best, Fraction(math.floor(x * q), q))
    return best


def choose_sync_set(latencies: Sequence[int], low_rates: Sequence[Fraction], typical_rates: Sequence[Fraction],
                    n_delta: int = 16, n_rho: int = 8, typical_weights: Sequence[int] | None = None) -> SyncSet:
    """Σ_sync:
    - **δ** runs from 0 up to the p99.99 first-frame latency, in `n_delta` steps;
    - **ρ** runs geometrically from the p0.01 of `low_rates` up to the p50 of `typical_rates` weighted by
      `typical_weights`, in `n_rho` values, each rounded down to a fraction with denominator at most 1024. ρ is below 1
      for sessions slower than one frame per bucket.

    `calibrate` passes one rate per configuration in each: its sessions' p0.01 production rate in `low_rates`, their
    p50 rate in `typical_rates`, and its windows in `typical_weights`, which is 1 + its largest anchor // T, the windows
    its egress frames span. The top ρ is then the median over windows (`weighted_percentile`): the least typical rate t
    such that the configurations with a typical rate at most t span at least ⌈W/2⌉ of all W windows. A configuration of
    n windows counts as n configurations of one window each, so cutting a run into more configurations (at a dropped
    window, say) doesn't move it. Without `typical_weights`, every configuration counts once."""
    hi = math.ceil(percentile(latencies, 99.99)) if latencies else 0
    step = max(1, math.ceil(hi / max(1, n_delta - 1)))
    deltas = tuple(i * step for i in range(n_delta))
    lo_r = fraction_at_most(percentile(low_rates, 0.01)) if low_rates else Fraction(1)
    weights = [1] * len(typical_rates) if typical_weights is None else typical_weights
    hi_r = max(lo_r, Fraction(weighted_percentile(typical_rates, weights, 50))) if typical_rates else lo_r
    if n_rho == 1 or hi_r == lo_r:
        return SyncSet(deltas, (lo_r,))
    ratio = float(hi_r / lo_r)
    rhos = {lo_r} | {fraction_at_most(lo_r * Fraction(ratio ** (i / (n_rho - 1)))) for i in range(1, n_rho)}
    return SyncSet(deltas, tuple(sorted(rhos)))


def choose_syncs(links: Sequence[tuple[Sequence[LogicalFrame], Sequence[int]]], sync_set: SyncSet,
                 window_buckets: int) -> tuple[list[ClockSync], list[list[int]], int]:
    """Replay the honest prover's choice for one configuration, window by window. `links` holds each egress link's
    frames and their ready buckets. Under one clock per spatial unit every egress link reads the window's sync, so the
    prover declares one value per window: the value met on every link with the least total hand time over the window's
    frames on all links, which is their total shaping delay; ties go to the smaller δ, then the larger ρ. (Taking the
    smallest δ first would pick a slow ρ to save one bucket, and hold every later frame for seconds.) A window that no
    value meets on every link is a miss, which is a violation; the replay continues with the most lenient value. Returns
    the chosen syncs, each link's actual hand times (a late frame goes when ready, delaying those behind it) and the
    number of missed windows.

    Each link's frames are in the Program's canonical order. For a batching engine that order is step-major, so a long
    session anchored in one window interleaves with sessions anchored in later ones, and anchors need not be monotone.
    On each link, each window's first frame must come after the first frame of every earlier window (requests are
    admitted in order). While window w is chosen, later windows take their most eager value (least δ, largest ρ). That
    only lowers hand times, so a window found met stays met once later windows are fixed. A larger δ or a smaller ρ
    only raises hand times, so a value met on every link exists exactly when the most lenient one is met on each."""
    wins = [[f.anchor // window_buckets for f in frames] for frames, _ in links]
    firsts: list[dict[int, int]] = []
    lasts: list[dict[int, int]] = []
    for win in wins:
        first: dict[int, int] = {}
        last: dict[int, int] = {}
        for k, w in enumerate(win):
            first.setdefault(w, k)
            last[w] = k
        if any(first[a] > first[b] for a, b in zip(sorted(first), sorted(first)[1:])):
            raise ValueError("each window's first frame must follow the first frame of every earlier window")
        firsts.append(first)
        lasts.append(last)
    if not any(wins):
        return [], [[] for _ in links], 0
    windows = max(max(win) for win in wins if win) + 1
    deltas, rhos = sorted(set(sync_set.deltas)), sorted(set(sync_set.rhos))
    eager, lenient = ClockSync(deltas[0], rhos[-1]), ClockSync(deltas[-1], rhos[0])
    syncs = [eager] * windows
    misses, h_prefix, k0 = 0, [0] * len(links), [0] * len(links)

    def cost(w: int, sync: ClockSync) -> int | None:
        """The window's total hand time on every link under `sync`, or None if a frame of the window is late."""
        syncs[w] = sync
        total = 0
        for i, (frames, ready) in enumerate(links):
            if w not in firsts[i]:
                continue
            h, win = h_prefix[i], wins[i]
            for k in range(firsts[i][w], lasts[i][w] + 1):
                h = max(eligible(frames[k], syncs[win[k]]), h)
                if win[k] == w:
                    if ready[k] > h:
                        return None
                    total += h
        return total

    for w in range(windows):
        if not any(w in first for first in firsts):
            continue
        for i, (frames, _) in enumerate(links):
            if w in firsts[i]:
                for k in range(k0[i], firsts[i][w]):
                    h_prefix[i] = max(eligible(frames[k], syncs[wins[i][k]]), h_prefix[i])
                k0[i] = firsts[i][w]
        best: tuple[int, int, int, ClockSync] | None = None
        for i_rho, rho in enumerate(rhos):
            # a larger δ only raises hand times, so the least met δ for this ρ is found by bisection
            if (c := cost(w, ClockSync(deltas[-1], rho))) is None:
                continue
            lo, hi, c_hi = 0, len(deltas) - 1, c
            while lo < hi:
                mid = (lo + hi) // 2
                if (c := cost(w, ClockSync(deltas[mid], rho))) is None:
                    lo = mid + 1
                else:
                    hi, c_hi = mid, c
            key = (c_hi, lo, -i_rho, ClockSync(deltas[lo], rho))
            if best is None or key[:3] < best[:3]:
                best = key
        if best is None:
            misses += 1
            syncs[w] = lenient
        else:
            syncs[w] = best[3]
    hands = []
    for (frames, ready), win in zip(links, wins):
        link_hands, h, actual = [], 0, 0
        for k, f in enumerate(frames):
            h = max(eligible(f, syncs[win[k]]), h)
            actual = max(h, ready[k], actual)
            link_hands.append(actual)
        hands.append(link_hands)
    return syncs, hands, misses


def replay_fifo(hands: Sequence[int], allocation: int, queue_capacity: int) -> tuple[bool, list[int]]:
    """Replay the constant-rate FIFO certifier on frames handed at `hands` (non-decreasing buckets). Returns whether the
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


def least_allocation(hands: Sequence[int], queue_capacity: int, max_delay: int, start: int = 1) -> int:
    """The least r ≥ `start` whose FIFO replay has no overflow and a p99 queueing delay of at most `max_delay` buckets.
    Both only improve as r grows, so doubling and then bisection finds it.

    At r equal to the largest bucket demand the queue empties every bucket, so that r fits unless the demand exceeds
    the queue capacity: then every r overflows, and this raises `ValueError`."""

    def fits(r: int) -> bool:
        ok, delays = replay_fifo(hands, r, queue_capacity)
        return ok and percentile(delays, 99) <= max_delay

    if max_delay < 0:
        raise ValueError("the delay budget is at least 0 buckets")
    if not hands:
        return start
    peak = max(Counter(hands).values())
    if peak > queue_capacity:
        raise ValueError(f"a bucket hands {peak} frames, more than the queue capacity Q = {queue_capacity}, "
                         "so every allocation overflows")
    lo, hi = start - 1, start
    while not fits(hi):
        lo, hi = hi, 2 * hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if fits(mid) else (mid, hi)
    return hi


def size_allocation(hands: Sequence[int], queue_capacity: int, headroom: float = 2.0,
                    max_delay: int = 1) -> dict:
    """The egress allocation. Start at r = ⌈headroom · p99.9⌉ of per-bucket demand, then raise r until there is no
    overflow and the p99 queueing delay is at most `max_delay` buckets."""
    if not hands:
        return {"allocation": 1, "demand": {}, "overflow": False, "p99_delay_buckets": 0, "utilization": 0.0}
    horizon = max(hands) + 1
    demand = [0] * horizon
    for h in hands:
        demand[h] += 1
    r = least_allocation(hands, queue_capacity, max_delay, max(1, math.ceil(headroom * percentile(demand, 99.9))))
    ok, delays = replay_fifo(hands, r, queue_capacity)
    p99 = percentile(delays, 99)
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
    `max_delay` ingress buckets. The ingress bucket τ_in is 100 ms."""
    if ingress_bucket_ms % trace_bucket_ms:
        raise ValueError("the ingress bucket must be a multiple of the trace bucket")
    k = ingress_bucket_ms // trace_bucket_ms
    counts = [sum(arrivals[i : i + k]) for i in range(0, len(arrivals), k)]
    hands = [t for t, c in enumerate(counts) for _ in range(c)]
    r = least_allocation(hands, queue_capacity, max_delay)
    rate = capacity.ingress_rate(r, ingress_bucket_ms)
    return {"allocation": r, "bucket_ms": ingress_bucket_ms, "rate_bps": rate}


def size_ingress_ms(arrival_ms: Sequence[float], ingress_bucket_ms: int = 100, queue_capacity: int = 1 << 16,
                    max_delay: int = 1) -> dict:
    """`size_ingress` from arrival times in ms, for any τ_in. A frame arriving at t is enqueued at the next bucket
    boundary, bucket ⌈t/τ_in⌉, and released after its FIFO delay. The report adds the delivery delay, release time
    minus arrival time, which the ingress grid adds to every request."""
    arrived = sorted(arrival_ms)
    hands = [math.ceil(t / ingress_bucket_ms) for t in arrived]
    r = least_allocation(hands, queue_capacity, max_delay)
    _, delays = replay_fifo(hands, r, queue_capacity)
    delivery = [(h + d) * ingress_bucket_ms - t for h, d, t in zip(hands, delays, arrived, strict=True)]
    return {
        "allocation": r, "bucket_ms": ingress_bucket_ms, "rate_bps": capacity.ingress_rate(r, ingress_bucket_ms),
        "delivery_delay_ms": ({p: percentile(delivery, q) for p, q in (("p50", 50), ("p99", 99))}
                              | {"max": max(delivery)}) if delivery else {},
    }


def calibrate(trace: dict, *, ingress_bucket_ms: int = 100, n_delta: int = 16, n_rho: int = 8,
              headroom: float = 2.0, epsilon: float = 1e-4, k_bits: float | None = None,
              target_k_share: float | None = None, target_span_s: int = 3600,
              ingress_max_delay_ms: int | None = None) -> dict:
    """Size every link of the trace and report the costs and the capacity. The ingress FIFO's p99 delay is at most one
    ingress bucket, or `ingress_max_delay_ms` when given. With `k_bits` (K, the policy's budget per isolation unit in
    bits) the ingress charge is also reported as a share of K, under the K it was computed with."""
    if target_k_share is not None and k_bits is None:
        raise ValueError("a target share of K needs K (k_bits)")
    in_delay = 1 if ingress_max_delay_ms is None else max(1, ingress_max_delay_ms // ingress_bucket_ms)
    bucket_ms = trace.get("bucket_ms", 100)
    T = trace["window_buckets"]
    frame_bytes = trace.get("frame_bytes", 4096)
    q = trace.get("queue_capacity", 1 << 16)
    configs = trace["configurations"]

    per_link_arrivals: dict[str, list[list[int]]] = defaultdict(list)
    per_link_arrival_ms: dict[str, list[list[float]]] = defaultdict(list)
    latencies, low_rates, typical_rates, typical_weights = [], [], [], []
    for cfg in configs:
        cfg_rates, cfg_windows = [], 0
        for link in cfg.get("egress", []):
            latencies += first_frame_latencies(link["frames"])
            cfg_rates += session_rates(link["frames"])
            cfg_windows = max(cfg_windows, max((f["anchor"] // T + 1 for f in link["frames"]), default=0))
        for link in cfg.get("ingress", []):
            if "arrival_ms" in link:
                per_link_arrival_ms[link["link"]].append(link["arrival_ms"])
            else:
                per_link_arrivals[link["link"]].append(link["arrivals"])
        if cfg_rates:
            low_rates.append(percentile(cfg_rates, 0.01))
            typical_rates.append(percentile(cfg_rates, 50))
            typical_weights.append(cfg_windows)
    sync_set = choose_sync_set(latencies, low_rates, typical_rates, n_delta, n_rho, typical_weights)

    # one clock per spatial unit: a window, and a miss, belong to the configuration, shared by its egress links
    windows_total, misses_total, chosen = 0, 0, []
    per_link_runs: dict[str, list[tuple[list[dict], list[int], int, int]]] = defaultdict(list)
    for cfg in configs:
        links = cfg.get("egress", [])
        lfs = [[LogicalFrame(b"", f["anchor"], f["index"]) for f in link["frames"]] for link in links]
        syncs, hands, misses = choose_syncs([(lf, [f["ready"] for f in link["frames"]]) for lf, link in zip(lfs, links)],
                                            sync_set, T)
        windows_total += len(syncs)
        misses_total += misses
        chosen += [syncs[w] for w in sorted({f.anchor // T for lf in lfs for f in lf})]
        for link, link_hands in zip(links, hands, strict=True):
            per_link_runs[link["link"]].append((link["frames"], link_hands, len(syncs), misses))

    egress = []
    for name, runs in per_link_runs.items():
        hands_all, link_windows, link_misses, held = [], 0, 0, []
        offset = 0
        for frames, hands, windows, misses in runs:
            link_windows += windows
            link_misses += misses
            held += [h - f["ready"] for h, f in zip(hands, frames, strict=True)]
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
            "shaping_delay_ms": ({p: percentile(held, q) * bucket_ms for p, q in (("p50", 50), ("p99", 99))}
                                 | {"max": max(held) * bucket_ms}) if held else {},
        })

    ingress = []
    for name, runs in per_link_arrivals.items():
        joined = [c for run in runs for c in run]
        ingress.append({"link": name, "direction": "ingress"}
                       | size_ingress(joined, bucket_ms, ingress_bucket_ms, q, in_delay))
    for name, runs in per_link_arrival_ms.items():
        joined_ms, offset = [], 0.0
        for run in runs:
            joined_ms += [t + offset for t in run]
            offset = (math.ceil(max(joined_ms, default=offset) / ingress_bucket_ms) + 2) * ingress_bucket_ms
        ingress.append({"link": name, "direction": "ingress"}
                       | size_ingress_ms(joined_ms, ingress_bucket_ms, q, in_delay))

    miss_rate = misses_total / windows_total if windows_total else 0.0
    windows_per_day = 86400 * 1000 / (T * bucket_ms)
    ingress_bps = sum(i["rate_bps"] for i in ingress)
    report = {
        "grid": {"bucket_ms": bucket_ms, "window_buckets": T, "frame_bytes": frame_bytes, "queue_capacity": q},
        "egress": egress,
        "clock_sync": {
            "deltas": list(sync_set.deltas), "rhos": [str(r) for r in sync_set.rhos], "size": sync_set.size,
            "bits_per_window": sync_set.bits,
            "honest_miss_rate": miss_rate, "target": epsilon, "meets_target": miss_rate <= epsilon,
            "declared": ({"delta": [min(s.delta for s in chosen), max(s.delta for s in chosen)],
                          "rho": [str(min(s.rho for s in chosen)), str(max(s.rho for s in chosen))]}
                         if chosen else {}),
            "first_frame_latency_ms": ({"p50": percentile([s.delta for s in chosen], 50) * bucket_ms,
                                        "p99": percentile([s.delta for s in chosen], 99) * bucket_ms}
                                       if chosen else {}),
        },
        "capacity": {
            "egress_bits_per_day_one_clock": sync_set.bits * windows_per_day,
            "egress_bits_per_day_clock_per_link": sync_set.bits * windows_per_day * max(1, len(egress)),
        },
        "ingress": ingress,
        "k_charge": {"ingress_bps": ingress_bps, "memory_after_wipe": "added from the wipe evidence; not measured here"},
        "provenance": {"configurations": [c.get("name", "") for c in configs]},
    }
    if k_bits is not None:
        report["k_charge"] |= {"k_bits": k_bits,
                               "share_of_k_by_span": {f"{s}s": capacity.k_share(ingress_bps * s, k_bits) for s in SPANS_S}}
    if target_k_share is not None:
        share = capacity.k_share(ingress_bps * target_span_s, k_bits)
        report["k_charge"]["target"] = {"span_s": target_span_s, "share": target_k_share,
                                        "meets_target": share <= target_k_share}
    return report


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="network-certifier-calibrate", description=__doc__.splitlines()[0])
    ap.add_argument("trace", help="honest timing trace (JSON)")
    ap.add_argument("--ingress-bucket-ms", type=int, default=100,
                    help="the ingress bucket τ_in: 100 ms; other values compare alternatives")
    ap.add_argument("--deltas", type=int, default=16, help="offsets in Σ_sync")
    ap.add_argument("--rhos", type=int, default=8, help="rates in Σ_sync")
    ap.add_argument("--headroom", type=float, default=2.0)
    ap.add_argument("--epsilon", type=float, default=1e-4, help="target honest miss rate per window")
    ap.add_argument("--k-bits", type=float, default=None,
                    help="K, the policy's budget per isolation unit in bits (the inference-only policy: 9.12e7); "
                         "without it the ingress charge is reported in bits only")
    ap.add_argument("--target-k-share", type=float, default=None, help="with --k-bits")
    ap.add_argument("--target-span-s", type=int, default=3600)
    ap.add_argument("--ingress-max-delay-ms", type=int, default=None,
                    help="p99 ingress FIFO delay budget (default: one ingress bucket)")
    args = ap.parse_args(argv)
    with open(args.trace, encoding="utf-8") as fh:
        trace = json.load(fh)
    try:
        report = calibrate(trace, ingress_bucket_ms=args.ingress_bucket_ms, n_delta=args.deltas, n_rho=args.rhos,
                           headroom=args.headroom, epsilon=args.epsilon, k_bits=args.k_bits,
                           target_k_share=args.target_k_share, target_span_s=args.target_span_s, ingress_max_delay_ms=args.ingress_max_delay_ms)
    except ValueError as e:
        print(f"{ap.prog}: error: {e}", file=sys.stderr)
        return 2
    json.dump(report, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
