"""`prove_time.sh`'s summary: per statement the timed sessions' median prove time (`benchmarks/flock_s8_block/summary.py`'s
`session_set`), and per block type the fit of that time against the statement's blocks N, T(N) = fixed + N · per_block
(least squares over its N), its seconds a block amortized at each N (T(N) / N), and with ``--card`` (a GPU's run) the
block's proof overhead: seconds a block times the card's peak MACs a second over the block's MACs, marginal (``per_block``)
and amortized. A coordinate's MACs are its K; the card's peak is its s8 rate (`verity.hardware`; e4m3 runs at that rate on
the cards here, their ``weight.e4m3`` 1). A type without a static K (a census type) gets its seconds a block and no
overhead.

    python prove_time.py OUT_DIR [--card SKU]     prints the summary as JSON
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _session_set():
    spec = importlib.util.spec_from_file_location("s8_summary", ROOT / "benchmarks/flock_s8_block/summary.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.session_set


def peak_macs(card: str) -> float:
    sys.path.insert(0, str(ROOT))
    from verity import hardware
    c = hardware.load(card)
    return c["rate"]["tensor"]["value"] * c["sm"]["count"]["value"] * c["clock"]["sm"]["value"] * 1e6


def fit(points: list[tuple[int, float]]) -> tuple[float, float] | None:
    """(fixed, per_block) of T = fixed + N · per_block by least squares, or None under two distinct N."""
    if len({n for n, _ in points}) < 2:
        return None
    m = len(points)
    sx, sy = sum(n for n, _ in points), sum(t for _, t in points)
    sxx, sxy = sum(n * n for n, _ in points), sum(n * t for n, t in points)
    b = (m * sxy - sx * sy) / (m * sxx - sx * sx)
    return (sy - b * sx) / m, b


def main(out: Path, card: str | None) -> dict:
    session_set = _session_set()
    peak = peak_macs(card) if card else None
    res = {"card": card, "peak_macs_per_s": peak, "statements": {}, "types": {}, "table": []}
    types = {}
    for dev in ("gpu", "cpu"):
        for d in sorted((out / dev).glob("*-n*")):
            st = json.loads((out / "stage" / d.name / "stage.json").read_text())
            s = session_set(d, gpu=dev == "gpu")
            rec = {"device": dev, "tag": st["tag"], "n": st["n"], "k_log": st["k_log"], "unit_log": st.get("unit_log"),
                   "vus_per_block": st["vus_per_block"],
                   "ands": st["ands"], "macs": st.get("statics", {}).get("K"), "public_inputs": st.get("public_inputs", []),
                   "accepted": s.get("accepted"), "prove_total_s": s.get("prove_total_s"),
                   "prove_total_all_s": s.get("prove_total_all_s"), "session_s": s.get("session_s"), "buckets": s.get("buckets"),
                   "gpu_util_mean_pct": s.get("gpu_util_mean_pct"), "sm_clock_mean_mhz": s.get("sm_clock_mean_mhz"),
                   "error": s.get("error")}
            res["statements"][f"{dev}/{d.name}"] = rec
            if rec["prove_total_s"] is not None:
                types.setdefault(f"{dev}/{st['tag']}", []).append(rec)
    for key, recs in sorted(types.items()):
        recs.sort(key=lambda r: r["n"])
        macs, r0 = recs[0]["macs"], recs[0]
        f = fit([(r["n"], r["prove_total_s"]) for r in recs])
        t = {"macs": macs, "k_log": r0["k_log"], "unit_log": r0["unit_log"], "vus_per_block": r0["vus_per_block"], "ands": r0["ands"],
             "cells_per_block": (1 << r0["k_log"]) / r0["vus_per_block"],
             "amortized": {r["n"]: {"s_per_block": r["prove_total_s"] / r["n"]} for r in recs}}
        if peak and macs:
            for r in recs:
                t["amortized"][r["n"]]["overhead"] = r["prove_total_s"] / r["n"] * peak / macs
        if f:
            t.update(fixed_s=f[0], per_block_s=f[1], residual=[r["prove_total_s"] - (f[0] + r["n"] * f[1]) for r in recs])
            if peak and macs:
                t["overhead"] = f[1] * peak / macs
        res["types"][key] = t
        line = f"{key}: k_log {t['k_log']} unit_log {t['unit_log']} G {t['vus_per_block']} ANDs {t['ands']:,}"
        if f:
            line += f"; fixed {f[0]:.3f} s, {f[1] * 1e3:.3f} ms a block" + (f", overhead {t['overhead']:.3g}" if "overhead" in t else "")
        line += "; amortized " + ", ".join(f"N={n} {a['s_per_block'] * 1e3:.3f} ms" + (f" ({a['overhead']:.3g})" if "overhead" in a else "")
                                           for n, a in t["amortized"].items())
        res["table"].append(line)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--card", default=None, help="the GPU's card (verity.hardware SKU) for the proof overhead")
    a = ap.parse_args()
    print(json.dumps(main(a.out, a.card), indent=1, default=str))
