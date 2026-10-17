"""EPS1's readings of the opcode probes (note:20261017T0310Z-report-eps1-program): joint facets, side units, clock.

  python eps1.py facets OUT/rows.jsonl [--margin 0.02]   # frontier.cu's MMA + X points -> facets (SMSP-cycles a warp
                                                         # instruction, x (1 - margin)), with the points each is tight at
  python eps1.py side OUT/rows.jsonl                     # side.cu: honest alone, and each unit alone and beside it
  python eps1.py clock OUT/sustain.jsonl [--skip-s 3]    # power.cu: each kernel's clock, power and work rate a phase

A facet is a row c_M n_MMA + c_X n_X <= T over warp instructions on one SMSP in T SMSP-cycles: the upper hull of the
measured points (x_MMA, x_X) in warp instructions a SMSP-cycle (the rows' rate / 4), each alone rate on its axis, with
every coefficient scaled by 1 - margin so a program measured at the hull still satisfies it with the margin to spare.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

IMMA = "IMMA.16832.S8.S8"
SMSPS = 4


def load(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def loop_rate(row: dict, pattern: str) -> float:
    """The row's rate (warp instructions/SM/clk) of the loop whose pattern is `pattern`; 0 when no loop counts it."""
    return sum(v for k, v in row["rates"].items() if k.split(":", 1)[1] == pattern)


def hull(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """The upper-right hull from the point of largest x to the point of largest m (m, x >= 0), in increasing m."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    up: list[tuple[float, float, str]] = []
    for p in pts:
        while len(up) >= 2:
            (m1, x1, _), (m2, x2, _) = up[-2], up[-1]
            if (m2 - m1) * (p[1] - x1) - (x2 - x1) * (p[0] - m1) >= 0:
                up.pop()
            else:
                break
        up.append(p)
    top = max(range(len(up)), key=lambda i: (up[i][1], -up[i][0]))
    right = max(range(len(up)), key=lambda i: (up[i][0], up[i][1]))
    return up[top:right + 1]


def facets(points: list[tuple[float, float, str]], margin: float) -> list[dict]:
    """Each hull edge as c_M m + c_X x <= 1 (SMSP-cycles a warp instruction), scaled by 1 - margin, plus the two axes."""
    h = hull(points)
    out = [{"MMA": 0.0, "X": (1 - margin) / h[0][1], "tight": [h[0][2]]}]
    for (m1, x1, n1), (m2, x2, n2) in zip(h, h[1:]):
        det = m1 * x2 - m2 * x1
        if det == 0:
            continue
        cm, cx = (x2 - x1) / det, (m1 - m2) / det
        if cm <= 0 or cx <= 0:
            continue
        out.append({"MMA": (1 - margin) * cm, "X": (1 - margin) * cx, "tight": [n1, n2]})
    out.append({"MMA": (1 - margin) / h[-1][0], "X": 0.0, "tight": [h[-1][2]]})
    return out


def frontier_points(rows: list[dict], x: str, pattern: str) -> list[tuple[float, float, str]]:
    by = {r["probe"]: r for r in rows}
    pts = [(loop_rate(by["imma_ref"], IMMA) / SMSPS, 0.0, "imma_ref"), (0.0, loop_rate(by[f"alone_{x}"], pattern) / SMSPS,
                                                                       f"alone_{x}")]
    for name, r in by.items():
        if name.startswith((f"mix_{x}_", f"spec_{x}_w")):
            pts.append((loop_rate(r, IMMA) / SMSPS, loop_rate(r, pattern) / SMSPS, name))
    return pts


OPS = {"imad": "IMAD", "hfma2": "HFMA2*", "lop3": "LOP3.LUT", "shfl": "SHFL.BFLY"}


def cmd_facets(a) -> int:
    rows = load(a.rows)
    for x, pat in OPS.items():
        try:
            pts = frontier_points(rows, x, pat)
        except KeyError as e:
            print(f"{x}: no probe {e}", file=sys.stderr)
            continue
        print(f"## MMA + {x} ({pat}); points (MMA, {x}) a SMSP-cycle:")
        for m, v, n in sorted(pts):
            print(f"  {n:16s} MMA {m:.5f} ({16 * m:.4f} of peak) {x} {v:.5f}  MMA cycles {m * 16.064:.3f}"
                  f" + {x} {v:.4f}")
        for f in facets(pts, a.margin):
            print(f"  facet MMA {f['MMA']:.3f} + {x.upper()} {f['X']:.3f} SMSP-cycles  (tight at {', '.join(f['tight'])})")
    return 0


#: words a warp instruction of a side loop's other patterns, by unit (the counted pattern's are the probe's `words`)
MORE_WORDS = {"redblk": {"UBLKRED*": 256}}


def side_words(row: dict, unit: str) -> float:
    """The side role's words/SM/clk: every pattern of its loop at its words a warp instruction."""
    words = {row["counted"]: row["words"] or 0, **MORE_WORDS.get(unit, {})}
    return sum(r * words.get(k.split(":", 1)[1], 0) for k, r in row["rates"].items() if k.split(":", 1)[1] != IMMA)


def cmd_side(a) -> int:
    rows = {r["probe"]: r for r in load(a.rows)}
    base = rows.get("honest") or {}
    hm = loop_rate(base, IMMA) if base else float("nan")
    print(f"honest alone: IMMA {hm:.5f}/SM/clk = {hm / 4 * 16:.4f} of the tensor pipe")
    print(f"{'unit':10s} {'alone words/SM/clk':>18s} {'beside: words/SM/clk':>20s} {'MMA kept':>9s} {'stop gap':>8s}")
    for name in sorted(n for n in rows if n.startswith("hs_")):
        u = name[3:]
        hs, sa = rows[name], rows.get("sa_" + u)
        beside = side_words(hs, u)
        alone = side_words(sa, u) if sa else float("nan")
        ref = rows.get("honest_cl" if u.startswith("dsm") else "honest")
        kept = loop_rate(hs, IMMA) / loop_rate(ref, IMMA) if ref else float("nan")
        print(f"{u:10s} {alone:18.4f} {beside:20.4f} {kept:9.4f} {hs.get('stop_gap') or 0:8.4f}")
    return 0


def windows(lines: list[dict], skip_s: float) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = {}
    for ln in lines:
        if ln.get("op") != "sustain":
            continue
        out.setdefault((ln["phase"], ln["probe"]), []).append(ln)
    # a probe sustained twice in a phase (the first and last honest window): split at the restart of `launch`
    split: dict[tuple[str, str], list[dict]] = {}
    for (ph, pr), ls in out.items():
        runs, cur = [], []
        for ln in ls:
            if ln["launch"] == 0 and cur:
                runs.append(cur)
                cur = []
            cur.append(ln)
        runs.append(cur)
        for i, r in enumerate(runs):
            split[(ph, pr if i == 0 else f"{pr}#{i + 1}")] = [ln for ln in r if ln["t"] >= skip_s]
    return split


MMAS = {"p_honest": 8, "p_half": 4, "p_mma": 8, "p_mma0": 8, "p_fold": 0, "p_ffma": 0, "p_mixed": 4}


def cmd_clock(a) -> int:
    w = windows(load(a.sustain), a.skip_s)
    print(f"{'phase':14s} {'kernel':12s} {'clock64/gt MHz':>14s} {'min':>8s} {'NVML MHz':>9s} {'W':>6s} {'C':>4s} "
          f"{'reasons':>8s} {'IMMA/SM/clk':>11s} {'IMMA/SM/us':>10s}")
    for (ph, pr), ls in w.items():
        if not ls:
            continue
        mhz = statistics.median(ln["clock_mhz_median"] for ln in ls)
        lo = min(ln["clock_mhz_min"] for ln in ls)
        nv = statistics.median(ln["nvml_sm_mhz"] for ln in ls)
        watts = statistics.median(ln["nvml_mw"] for ln in ls) / 1e3
        temp = statistics.median(ln["temp_c"] for ln in ls)
        reasons = 0
        for ln in ls:
            reasons |= int(ln["throttle"], 16)
        nm = MMAS.get(pr.split("#")[0], 0)
        per_clk = statistics.median(16 * ln["iters"] * nm / ln["cyc_median"] for ln in ls)
        print(f"{ph:14s} {pr:12s} {mhz:14.1f} {lo:8.1f} {nv:9.0f} {watts:6.1f} {temp:4.0f} {reasons:#8x} {per_clk:11.5f} "
              f"{per_clk * mhz:10.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("facets")
    f.add_argument("rows", type=Path)
    f.add_argument("--margin", type=float, default=0.02)
    s = sub.add_parser("side")
    s.add_argument("rows", type=Path)
    c = sub.add_parser("clock")
    c.add_argument("sustain", type=Path)
    c.add_argument("--skip-s", type=float, default=3.0)
    a = p.parse_args(argv)
    return {"facets": cmd_facets, "side": cmd_side, "clock": cmd_clock}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
