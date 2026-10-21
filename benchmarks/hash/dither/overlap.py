"""What runs beside the units, from an sbench `prof` capture (nsys SQLite export, as stepprof.py): for every kernel on
the units stream inside the arm's longer NVTX range, its duration and the share of it that each other part (stepprof's
PARTS by name) covered, then per kernel name a least-squares fit dur = b0 + sum_c b_c share_c: b0 is the kernel alone,
b_c what part c beside it for its whole run adds, and sum b_c share_c over the step what the units lose to each part.

    python overlap.py gate.sqlite [--arm ARM] > overlap.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from bisect import bisect_right
from pathlib import Path

import numpy as np

import stepprof as SP

CLASSES = ("units: noise (separate launch)", "A table (h1c_a_call)", "units: seeds",
           "GEMM with the z/y epilogue (dithera_mm)")
SHORT = {CLASSES[0]: "noise", CLASSES[1]: "a_table", CLASSES[2]: "seeds", CLASSES[3]: "dithera_mm"}


def covered(s: int, e: int, iv: list[tuple[int, int]], ends: list[int]) -> int:
    """The time in [s, e) under the disjoint sorted intervals iv (ends: their ends)."""
    t = 0
    for a, b in iv[bisect_right(ends, s):]:
        if a >= e:
            break
        t += min(b, e) - max(a, s)
    return t


def analyse(path: str, arm: str | None) -> dict:
    db = sqlite3.connect(path)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    ranges: dict[str, list[tuple[int, int]]] = {}
    for start, end, text, text_id in db.execute(
            "SELECT start, end, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        name = text if text is not None else strings.get(text_id)
        if name and name.startswith(SP.ST.PROF_RANGE):
            ranges.setdefault(name.removeprefix(SP.ST.PROF_RANGE).rsplit("/n", 1)[0], []).append((int(start), int(end)))
    arm = arm or sorted(ranges)[0]
    start, end = max(ranges[arm], key=lambda r: r[1] - r[0])
    ks = [(strings.get(n, str(n)), int(s), int(e), int(q)) for n, s, e, q in db.execute(
        "SELECT demangledName, start, end, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start >= ? AND end <= ?",
        (start, end))]
    units = SP.PARTS[4][0]
    by_q: dict[int, int] = {}
    for name, s, e, q in ks:
        if SP.part(name, None) == units:
            by_q[q] = by_q.get(q, 0) + 1
    uq = max(by_q, key=by_q.get)
    iv = {c: [] for c in CLASSES}
    for name, s, e, q in ks:
        c = SP.part(name, None)
        if c in iv:
            iv[c].append((s, e))
    merged = {}
    for c, v in iv.items():
        v.sort()
        m: list[list[int]] = []
        for a, b in v:
            if m and a <= m[-1][1]:
                m[-1][1] = max(m[-1][1], b)
            else:
                m.append([a, b])
        merged[c] = ([(a, b) for a, b in m], [b for _, b in m])
    rows: dict[str, list[list[float]]] = {}
    for name, s, e, q in ks:
        if q != uq or SP.part(name, None) != units or e <= s:
            continue
        d = e - s
        key = name[:name.index(">(") + 1] if ">(" in name else name.split("(")[0]
        rows.setdefault(key.removeprefix("void "), []).append([d / 1e3] + [covered(s, e, *merged[c]) / d for c in CLASSES])
    doc = {"arm": arm, "range_ms": (end - start) / 1e6, "units_stream": uq, "kernels": {}}
    for name, r in sorted(rows.items(), key=lambda kv: -sum(x[0] for x in kv[1])):
        a = np.array(r)
        y, X = a[:, 0], np.hstack([np.ones((len(a), 1)), a[:, 1:]])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        doc["kernels"][name] = {
            "n": len(a), "median_us": float(np.median(y)), "sum_ms": float(y.sum() / 1e3),
            "mean_share": {SHORT[c]: float(a[:, 1 + i].mean()) for i, c in enumerate(CLASSES)},
            "alone_us": float(beta[0]), "per_full_overlap_us": {SHORT[c]: float(beta[1 + i]) for i, c in enumerate(CLASSES)},
            "lost_ms": {SHORT[c]: float(beta[1 + i] * a[:, 1 + i].sum() / 1e3) for i, c in enumerate(CLASSES)},
        }
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sqlite")
    ap.add_argument("--arm")
    a = ap.parse_args()
    print(json.dumps(analyse(str(Path(a.sqlite)), a.arm), indent=1))


if __name__ == "__main__":
    main()
