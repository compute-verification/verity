"""bench.py's analysis, numpy only: per arm and shape (or a layer's linear set), the achieved TOPS and bytes/s against the s8
peak P and the measured DRAM bandwidth B, and two ridges:

    ridge_card   M* where 2 M W / P = bytes(M) / B, W the weights' elements and bytes(M) = W + M K + M N out + 4 (M + N)
                 summed over the set: the card's roofline crossing for that shape
    ridge_arm    the same with the arm's own best achieved TOPS and bytes/s over the sweep in place of P and B: where its time
                 stops being flat in M (weight streaming) and starts growing with it (tensor cores)

and the arm's time, TOPS and fractions of peak and bandwidth at each ridge, its time interpolated linearly in log t over log M
between the sweep's points. The ratio of the two fractions is 2 M W B / (bytes(M) P) whatever the time, so a crossing read
off it would only restate the card's ridge; ridge_arm is what the measurement adds."""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from fixture import model


def bytes_of(n: int, k: int, m: float, out_bytes: int) -> float:
    return n * k + m * k + m * n * out_bytes + 4 * (m + n)


def interp(ms: list[int], ts: list[float], m: float) -> float:
    return float(np.exp(np.interp(math.log(m), np.log(ms), np.log(ts))))


def ridge(w: int, bytes_fn: Callable[[float], float], p_tops: float, b_gbs: float) -> float:
    """M where 2 M W / P = bytes(M) / B, bytes affine in M; inf where compute never catches up."""
    b0, b1 = bytes_fn(0), bytes_fn(1) - bytes_fn(0)
    a = 2 * w / (p_tops * 1e12) - b1 / (b_gbs * 1e9)
    return math.inf if a <= 0 else b0 / (b_gbs * 1e9) / a


def curve(w: int, bytes_fn: Callable[[float], float], us_by_m: dict[int, float], peak: float, bw: float) -> dict:
    ms = sorted(us_by_m)
    ts = [us_by_m[m] * 1e-6 for m in ms]
    tops = [2 * m * w / t / 1e12 for m, t in zip(ms, ts)]
    gbs = [bytes_fn(m) / t / 1e9 for m, t in zip(ms, ts)]
    doc = {"ms": ms, "us": [t * 1e6 for t in ts], "tops": tops, "frac_peak": [v / peak for v in tops], "gbs": gbs,
           "frac_bw": [v / bw for v in gbs], "ridge_card": ridge(w, bytes_fn, peak, bw),
           "ridge_arm": ridge(w, bytes_fn, max(tops), max(gbs)), "at_ridge": {}}
    for label in ("card", "arm"):
        mm = doc[f"ridge_{label}"]
        if math.isfinite(mm) and len(ms) > 1:
            t = interp(ms, ts, mm)
            doc["at_ridge"][label] = {"m": mm, "us": t * 1e6, "tops": 2 * mm * w / t / 1e12,
                                      "frac_peak": 2 * mm * w / t / 1e12 / peak, "frac_bw": bytes_fn(mm) / t / 1e9 / bw}
    return doc


def summarize(rows: list[dict], models: tuple[str, ...], peak: float, bw: float, out_bytes: dict[str, int]) -> dict:
    """rows: bench.jsonl's ({"n", "k", "m", "us": {arm: microseconds}}) -> per linear and per model's layer set, per arm."""
    by: dict[tuple[int, int, str], dict[int, float]] = {}
    for r in rows:
        for arm, us in r["us"].items():
            by.setdefault((r["n"], r["k"], arm), {})[r["m"]] = us
    out: dict = {"peak_tops": peak, "bw_gbs": bw, "linears": {}, "layers": {}}
    for (n, k, arm), tm in by.items():
        out["linears"].setdefault(f"{n},{k}", {})[arm] = curve(
            n * k, lambda m, n=n, k=k, a=arm: bytes_of(n, k, m, out_bytes[a]), tm, peak, bw)
    for name in models:
        lin = [(n, k) for _, n, k, _ in model(name).linears]
        for arm in sorted({arm for (_, _, arm) in by}):
            curves = [by.get((n, k, arm), {}) for n, k in lin]
            if not all(curves):
                continue
            common = sorted(set.intersection(*(set(c) for c in curves)))
            if not common:
                continue
            tm = {m: sum(c[m] for c in curves) for m in common}
            out["layers"].setdefault(name, {})[arm] = curve(
                sum(n * k for n, k in lin), lambda m, a=arm: sum(bytes_of(n, k, m, out_bytes[a]) for n, k in lin), tm, peak, bw)
    return out
