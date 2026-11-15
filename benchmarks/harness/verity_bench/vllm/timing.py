"""What one timed run records beside its tokens: the decode time per token, the inter-token latencies and their
percentiles, the run's own trace of where its time went, and the host's probe around it."""

from __future__ import annotations

import gc
import os
import resource
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

# A step's host time may carry device work its own forward didn't enqueue; past this share of the decode time the
# run's trailing device work is too large to attribute, and the decode metric is dropped.
DECODE_TAIL_SHARE = 0.01


def decode_time_per_token(
    steps: list[Mapping[str, Any]], *, output_lag: int = 0, device_tail_s: float = 0.0, generated: int | None = None
) -> float | None:
    """Host time of the engine steps that returned a decode-only forward's tokens (a forward with decode tokens and no
    prefill tokens), over the tokens it decoded: per-token decode cost with prefill left out, which time per output
    token mixes in under mixed traffic. A step's host time holds the device time of the forward it waits on to return
    sampled tokens, which is its own forward's when `output_lag` is 0 and the forward `output_lag` iterations back's
    under async scheduling, so the time of iteration i + lag is charged to the forward of iteration i. None when no
    step qualifies, the steps carry no token counts (`run_requests_arrivals` without a reachable scheduler), or the
    device work left after the last step (`device_tail_s`) exceeds DECODE_TAIL_SHARE of the decode time, which shows
    device work the host step times didn't see. Also None when the steps' decode tokens add up to more than the run
    `generated`: a decode token is one generated token, so a larger total means the counts misclassify prompt tokens
    as decode (once, `arrivals.scheduled_tokens` looked prompt lengths up by the submitted request id, which this vLLM
    pin renames, so every prompt token counted as decode)."""

    if generated is not None and sum(int(s.get("decode_tokens") or 0) for s in steps) > generated:
        return None
    lag = int(output_lag)
    decode = [
        (float(steps[i + lag]["s"]), int(s["decode_tokens"]))
        for i, s in enumerate(steps[: len(steps) - lag])
        if s.get("forward") and s.get("prefill_tokens") == 0 and s.get("decode_tokens")
    ]
    tokens = sum(n for _, n in decode)
    seconds = sum(t for t, _ in decode)
    if not tokens or device_tail_s > DECODE_TAIL_SHARE * seconds:
        return None
    return seconds / tokens


def inter_token_latencies(returns: Sequence[tuple[float, Mapping[str, int]]]) -> list[float]:
    """One latency per generated token after each request's first: the time from the step that returned the request's
    previous tokens to the step that returned this one, shared evenly by the tokens a step returned at once. ``returns``
    holds, per engine step, the clock reading as it returned and each returned request's tokens generated so far. This
    is the per-token latency a streaming client sees; the first token's is ``time_to_first_token_s``."""

    last: dict[str, tuple[float, int]] = {}
    out: list[float] = []
    for t, counts in returns:
        for rid, n in counts.items():
            if rid in last:
                t0, n0 = last[rid]
                if n <= n0:
                    continue
                out.extend([(t - t0) / (n - n0)] * (n - n0))
            last[rid] = (t, n)
    return out


def percentile(values: list[float], q: int) -> float | None:
    """The q-th percentile (1..99), interpolated between the closest ranks; None without values."""

    if len(values) < 2:
        return values[0] if values else None
    return statistics.quantiles(values, n=100, method="inclusive")[q - 1]



def run_trace(timing: Mapping[str, Any], start: float, ready: float) -> dict[str, Any]:
    """A timed rep's own record of where its time went, small enough to keep in every rep: each `step()` iteration's host
    time and the prefill and decode tokens its schedule computed, each request's arrival and first output (seconds from
    the run's start, in request order), and how long after the engine became ready the run started."""

    def us(x: Any) -> float | None:
        return None if x is None else round(float(x), 6)

    steps = timing.get("steps") or []
    requests = list(timing["requests"].values())
    return {
        "since_ready_s": us(start - ready),
        "step_s": [us(s["s"]) for s in steps],
        "step_prefill_tokens": [s.get("prefill_tokens") for s in steps],
        "step_decode_tokens": [s.get("decode_tokens") for s in steps],
        "arrival_s": [us(r.get("arrival_s")) for r in requests],
        "first_output_s": [us(r.get("first_output_s")) for r in requests],
    }


class HostProbe:
    """The process's CPU time, context switches, page faults and garbage collections over a period, and the host's load
    when it opened: a slow rep with as much CPU time but more involuntary switches waited on a contended host, one with
    more CPU time ran slower code (or a slower clock), and one with neither waited on the device. `counters`, when
    given, is read at both ends and its deltas kept under `device`."""

    def __init__(self, counters: Callable[[], Mapping[str, float]] | None = None) -> None:
        self._counters = counters
        self.counted: dict[str, float] | None = dict(counters()) if counters is not None else None
        self.load = os.getloadavg()[0]
        self.gc_s = 0.0
        self._gc_at: float | None = None
        gc.callbacks.append(self._on_gc)
        self.thread = time.thread_time()
        self.usage = resource.getrusage(resource.RUSAGE_SELF)

    def _on_gc(self, phase: str, _info: Mapping[str, Any]) -> None:
        if phase == "start":
            self._gc_at = time.perf_counter()
        elif self._gc_at is not None:
            self.gc_s += time.perf_counter() - self._gc_at
            self._gc_at = None

    def close(self) -> dict[str, Any]:
        after = resource.getrusage(resource.RUSAGE_SELF)
        thread = time.thread_time() - self.thread
        if self._on_gc in gc.callbacks:
            gc.callbacks.remove(self._on_gc)
        b = self.usage
        start = self.counted
        counted = ({"device": {k: v - start.get(k, 0) for k, v in self._counters().items()}}
                   if self._counters is not None and start is not None else {})
        return {
            "loadavg_1m": round(self.load, 2),
            "process_cpu_s": round(after.ru_utime + after.ru_stime - b.ru_utime - b.ru_stime, 6),
            "thread_cpu_s": round(thread, 6),
            "involuntary_switches": after.ru_nivcsw - b.ru_nivcsw,
            "voluntary_switches": after.ru_nvcsw - b.ru_nvcsw,
            "minor_faults": after.ru_minflt - b.ru_minflt,
            "major_faults": after.ru_majflt - b.ru_majflt,
            "gc_s": round(self.gc_s, 6),
            **counted,
        }
