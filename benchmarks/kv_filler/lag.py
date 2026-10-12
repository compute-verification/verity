"""The page conversion's lag, replayed over a serving run's steps: every token a step schedules writes 128 KiB of KV
(Llama-3.1-8B, bf16) that must be encoded before it is registered; the encode drains a queue at the rate the phase
allows, and until a token's KV is encoded, each step that reads it reads unregistered bytes. Torch-free.

`replay(steps, rates)` walks the steps in order (each `{"t0", "t1", "tokens", "prefill_tokens", "kv_tokens"}`, times in
seconds, as `serve.py` records them) with `rates` the encode's GB/s beside a decode step, beside a prefill step and
alone (idle time between steps). A step's own writes join the queue at its end; the queue drains through the step (at
its phase's rate) and through the idle time after it. The plain bytes a step reads are the queue at its start, capped by
the step's context (`kv_tokens`): the pending tokens belong to running requests, which read their whole context. So an
encode on a side stream lags by at least a step, whatever its rate. A step that drains the queue is slowed by its
phase's `*_slowdown` (the encode's measured cost beside it), which `added_s` sums.

`fused(steps, ...)` is the other arm: the encode in the KV write, no lag, its cost on the step. A step with prefill
tokens adds its tokens' encode at the alone rate (prefill and encode are both compute-bound and serialize: measured
+150 ms on a 121 ms prefill of 4,096 tokens at k = 2, where this adds 155 ms); a decode step adds its tokens' encode
at `decode_GBps`, the encode's bytes over the time it added to a decode step.
"""
from __future__ import annotations

from dataclasses import dataclass

KV_TOKEN_BYTES = 32 * 2 * 8 * 128 * 2


@dataclass(frozen=True)
class Rates:
    decode_GBps: float
    prefill_GBps: float
    idle_GBps: float
    decode_slowdown: float = 0.0
    prefill_slowdown: float = 0.0


def replay(steps: list[dict], rates: Rates, token_bytes: int = KV_TOKEN_BYTES) -> dict:
    queue = 0.0
    plain: list[float] = []
    peak, lag_s, t_prev, added = 0.0, [], None, 0.0
    for s in steps:
        if t_prev is not None and s["t0"] > t_prev:
            queue = max(0.0, queue - rates.idle_GBps * 1e9 * (s["t0"] - t_prev))
        read = min(queue, s["kv_tokens"] * token_bytes)
        plain.append(read)
        pre = s["prefill_tokens"] > 0
        rate = rates.prefill_GBps if pre else rates.decode_GBps
        if queue > 0 and rate > 0:
            added += (s["t1"] - s["t0"]) * (rates.prefill_slowdown if pre else rates.decode_slowdown)
        queue = max(0.0, queue - rate * 1e9 * (s["t1"] - s["t0"])) + s["tokens"] * token_bytes
        peak = max(peak, queue)
        lag_s.append(queue / (rates.decode_GBps * 1e9) if rates.decode_GBps else float("inf") if queue else 0.0)
        t_prev = s["t1"]
    return {"plain_bytes": plain, "plain_total": sum(plain), "queue_peak_bytes": peak,
            "lag_s_max": max(lag_s, default=0.0),  # lag: the queue at a step's end over the decode-side rate
            "added_s": added, "busy_s": sum(s["t1"] - s["t0"] for s in steps)}


def fused(steps: list[dict], alone_GBps: float, decode_GBps: float, token_bytes: int = KV_TOKEN_BYTES) -> dict:
    added = sum(s["tokens"] * token_bytes / ((alone_GBps if s["prefill_tokens"] > 0 else decode_GBps) * 1e9)
                for s in steps)
    busy = sum(s["t1"] - s["t0"] for s in steps)
    return {"added_s": added, "busy_s": busy, "overhead": added / busy if busy else 0.0}


def windows(steps: list[dict], plain: list[float], t0: float, t1: float, width: float, R_GBps: float) -> list[float]:
    """γ_lag per window: the plain bytes read inside it (a step's pro rata) over R·T."""
    n = int((t1 - t0) // width)
    out = [0.0] * n
    for s, b in zip(steps, plain):
        a, z = s["t0"], s["t1"]
        if z <= a or not b:
            continue
        for i in range(max(0, int((a - t0) // width)), min(n - 1, int((z - t0) // width)) + 1):
            lo, hi = t0 + i * width, t0 + (i + 1) * width
            ov = min(z, hi) - max(a, lo)
            if ov > 0:
                out[i] += b * ov / (z - a)
    return [v / (R_GBps * 1e9 * width) for v in out]
