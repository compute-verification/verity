"""The page conversion's lag, replayed over a serving run's steps: every token a step schedules writes 128 KiB of KV
(Llama-3.1-8B, bf16) that must be encoded before it is registered; the encode drains a queue at the rate the phase
allows, and until a token's KV is encoded, each step that reads it reads unregistered bytes. Torch-free.

`replay(steps, rates)` walks the steps in order (each `{"t0", "t1", "tokens", "prefill_tokens", "kv_tokens"}`, times in
seconds, as `serve.py` records them) with `rates` the encode's GB/s beside a decode step, beside a prefill step and
alone (idle time between steps). A step's own writes join the queue at its end; the queue drains through the step (at
its phase's rate) and through the idle time after it. The plain bytes a step reads are the queue at its start, capped by
the step's context (`kv_tokens`): the pending tokens belong to running requests, which read their whole context. So an
encode on a side stream lags by at least a step, whatever its rate (an encode fused into the KV write has no lag and
puts its cost on the step instead).
"""
from __future__ import annotations

from dataclasses import dataclass

KV_TOKEN_BYTES = 32 * 2 * 8 * 128 * 2


@dataclass(frozen=True)
class Rates:
    decode_GBps: float
    prefill_GBps: float
    idle_GBps: float


def replay(steps: list[dict], rates: Rates, token_bytes: int = KV_TOKEN_BYTES) -> dict:
    queue = 0.0
    plain: list[float] = []
    peak, lag_s, t_prev = 0.0, [], None
    for s in steps:
        if t_prev is not None and s["t0"] > t_prev:
            queue = max(0.0, queue - rates.idle_GBps * 1e9 * (s["t0"] - t_prev))
        read = min(queue, s["kv_tokens"] * token_bytes)
        plain.append(read)
        rate = rates.prefill_GBps if s["prefill_tokens"] > 0 else rates.decode_GBps
        queue = max(0.0, queue - rate * 1e9 * (s["t1"] - s["t0"])) + s["tokens"] * token_bytes
        peak = max(peak, queue)
        lag_s.append(queue / (rates.decode_GBps * 1e9) if rates.decode_GBps else float("inf") if queue else 0.0)
        t_prev = s["t1"]
    return {"plain_bytes": plain, "plain_total": sum(plain), "queue_peak_bytes": peak,
            "lag_s_max": max(lag_s, default=0.0)}  # lag: the queue at a step's end over the decode-side rate


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
