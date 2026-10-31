"""The epsilon program's readings of the opcode probes: joint facets, side units, clock.

  python eps1.py facets OUT/rows.jsonl... [--margin 0.02] [--prior rows.csv] [--emit EPOCH]
      # frontier.cu's MMA + X points (every run given) -> facets (SMSP-cycles a warp instruction, x (1 - margin)), with
      # the points each is tight at; --prior adds the points the published K* facets were fitted to (their vertices),
      # --emit prints the facets as rows.csv and census/EPOCH.csv lines
  python eps1.py side OUT/rows.jsonl                     # side.cu: honest alone, and each unit alone and beside it
  python eps1.py clock OUT/sustain.jsonl [--skip-s 3]    # power.cu: each kernel's clock, power and work rate a phase
  python eps1.py ceiling OUT                             # clock_ceiling.sh: each clock knob and the clock it reached
  python eps1.py uniform OUT                             # uniform.cu: issue slots a uniform instruction, each setting
  python eps1.py l2gather OUT                            # l2gather.cu: bytes a SM a clock and TB/s a gather path, best
  python eps1.py jitter OUT                              # clock_calib.sh: auditor-side u of host-timing a reference kernel

A facet is a row c_M n_MMA + c_X n_X <= T over warp instructions on one SMSP in T SMSP-cycles: the upper hull of the
measured points (x_MMA, x_X) in warp instructions a SMSP-cycle (the rows' rate / 4), each alone rate on its axis, with
every coefficient scaled by 1 - margin so a program measured at the hull still satisfies it with the margin to spare.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from fnmatch import fnmatchcase
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
    # a facet within 0.2% (of each coefficient's largest) of one at least as tight on both coefficients adds nothing: an
    # axis, or an edge between near-collinear points; dropping it loosens Π by at most that much, inside the margin
    tol = {k: 0.002 * max(f[k] for f in out) for k in ("MMA", "X")}
    keep = list(out)
    for f in out:
        if any(g is not f and all(g[k] >= f[k] - tol[k] for k in tol) for g in keep):
            keep.remove(f)
    return keep


def frontier_points(rows: list[dict], x: str, pattern: str) -> list[tuple[float, float, str]]:
    pts = []
    for r in rows:
        name = r["probe"]
        if name == "imma_ref" or name == f"alone_{x}" or name.startswith((f"mix_{x}_", f"spec_{x}_w")):
            pts.append((loop_rate(r, IMMA) / SMSPS, loop_rate(r, pattern) / SMSPS, f"{name}@{r['run']}"))
    if not pts:
        raise KeyError(f"no {x} probe")
    return pts


def prior_points(rows_csv: Path, prefix: str, kind: str, margin: float) -> tuple[list[tuple[float, float, str]], set[str]]:
    """The points a published set of facets (rows.csv's `prefix`N rows, MMA a + KIND b) was fitted through, and the runs
    those rows cite: each consecutive pair's intersection and the MMA-axis end, at 1 - margin of the facets (a published
    facet sits at 1 - margin on its points). The KIND-axis end is left out: the run's own alone probe gives it (KSHFL0's is
    the ASSUMED SHFL)."""
    pat = re.compile(rf"MMA ([0-9.]+) \+ {kind} ([0-9.]+) SMSP-cycles")
    fs, runs = [], set()
    with rows_csv.open(newline="") as f:
        for d in csv.DictReader(f):
            m = pat.search(d["description"])
            if re.fullmatch(rf"{prefix}\d+", d["row"]) and m:
                fs.append((float(m.group(1)), float(m.group(2)), d["row"]))
                runs |= set(d["runs"].split())
    fs.sort()
    if not fs:
        return [], runs
    s = 1 - margin
    pts = []
    for (a1, b1, n1), (a2, b2, n2) in zip(fs, fs[1:]):
        det = a1 * b2 - a2 * b1
        pts.append((s * (b2 - b1) / det, s * (a1 - a2) / det, f"{n1}/{n2}"))
    return pts + [(s / fs[-1][0], 0.0, f"{fs[-1][2]}:axis")], runs


OPS = {"imad": "IMAD", "hfma2": "HFMA2*", "lop3": "LOP3.LUT", "shfl": "SHFL.BFLY", "iadd3": "IADD3", "ffma": "FFMA",
       "idp4a": "IDP.4A*", "popc": "POPC"}
#: each family's census kind, its published rows (VF3f) and the rows its facets become; iadd3,
#: ffma, idp4a and popc are sm89/scalar.cu's
FAMILY = {"imad": ("IMAD", "KIMAD", "EIMAD"), "hfma2": ("HFMA2", "KHFMA2", "EHFMA2"), "lop3": ("ALU", "KALU", "EALU"),
          "shfl": ("SHFL", "KSHFL", "ESHFL"), "iadd3": ("ALU", "KIADD3", "EIADD3"), "ffma": ("FFMA", "KFFMA", "EFFMA"),
          "idp4a": ("IDP4A", "KIDP4A", "EIDP4A"), "popc": ("XU", "KXU", "EXU")}


def cmd_facets(a) -> int:
    rows = [r for p in a.rows for r in load(p)]
    runs = " ".join(sorted({r["run"] for r in rows}))
    for x, pat in OPS.items():
        try:
            pts = frontier_points(rows, x, pat)
        except KeyError as e:
            print(f"{x}: {e}", file=sys.stderr)
            continue
        kind, prior, new = FAMILY[x]
        prior_runs = set()
        if a.prior:
            more, prior_runs = prior_points(a.prior, prior, kind, a.margin)
            pts += more
        fs = facets(pts, a.margin)
        if a.emit:
            for i, f in enumerate(fs):
                tight = [t.split("@")[0] for t in f["tight"]]
                cite = " ".join(sorted(set(runs.split()) | (prior_runs if any(t.startswith(prior) for t in tight)
                                                             else set())))
                desc = (f"MMA {f['MMA']:.3f} + {kind} {f['X']:.3f} SMSP-cycles (EPS1 frontier, {x} for {kind}"
                        f"{', with ' + prior + chr(39) + 's points' if a.prior else ''}, 2% noise margin; tight at "
                        f"{' and '.join(tight)})")
                print(f'rows.csv: {new}{i},joint,{cite},"{desc}"')
                for k, c in (("MMA", f["MMA"]), (kind, f["X"])):
                    if c > 0:
                        print(f"census/{a.emit}.csv: {k},{new}{i},,,{c:.9g},,{a.emit},,{cite}")
            continue
        print(f"## MMA + {x} ({pat}); points (MMA, {x}) a SMSP-cycle:")
        for m, v, n in sorted(pts):
            print(f"  {n:40s} MMA {m:.5f} ({16 * m:.4f} of peak) {x} {v:.5f}")
        for f in fs:
            print(f"  facet MMA {f['MMA']:.3f} + {kind} {f['X']:.3f} SMSP-cycles  (tight at {', '.join(f['tight'])})")
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


MMAS = {"p_honest": 8, "p_half": 4, "p_mma": 8, "p_mma0": 8, "p_fold": 0, "p_ffma": 0, "p_mixed": 4, "p_honest16": 8,
        "p_mma16": 8, "p_mma16z": 8, "p_light": 0}


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


def _val(d: dict, *path: str) -> object:
    for k in path:
        d = d.get(k, {}) if isinstance(d, dict) else {}
    v = d.get("value", d.get("error", "-")) if isinstance(d, dict) and d else "-"
    return v[0] if isinstance(v, list) and len(v) == 1 else v


def cmd_ceiling(a) -> int:
    """clock_ceiling.sh: each knob's rc, what NVML then reports, and the clock each kernel ran at by the host's clock."""
    out = a.out
    found = json.loads((out / "found.json").read_text())
    trials = {t["trial"]: t for t in load(out / "trials.jsonl")} if (out / "trials.jsonl").exists() else {}
    states = {s["trial"]: s for s in load(out / "states.jsonl")} if (out / "states.jsonl").exists() else {}
    calib: dict[str, dict[str, list[dict]]] = {}
    for c in load(out / "calib.jsonl") if (out / "calib.jsonl").exists() else []:
        calib.setdefault(c["trial"], {}).setdefault(c["probe"], []).append(c)
    fmax = float(found["max_mhz"])
    print(f"GPU {found['uuid']}: NVML max SM clock {found['max_mhz']} MHz, power limit {found['found_pl_w']} W, "
          f"vboost {found['found_vboost']} of {found['max_vboost']}")
    st = states.get("found", {})
    print(f"  GPC VF offset range {_val(st, 'gpc_vf_offset_range')}, memory {_val(st, 'mem_vf_offset_range')}, "
          f"P0 graphics offsets {st.get('clock_offsets_p0', {}).get('graphics', '-')}, "
          f"power limit constraints (mW) {_val(st, 'power_limit_constraints_mw')}")
    print(f"\n{'trial':26s} {'rc':>3s} {'NVML now':>8s} {'max':>6s} {'app':>6s} {'kernel':9s} {'host MHz':>9s} "
          f"{'gt MHz':>9s} {'gt min':>9s} {'NVML':>5s} {'W':>5s} {'reasons':>8s}  output")
    best = (0.0, "", "")
    for t in dict.fromkeys([*trials, *calib]):
        tr, s = trials.get(t, {}), states.get(t, {})
        first = True
        for k, cs in (calib.get(t) or {"-": []}).items():
            host = max((c["clock_mhz_host"] for c in cs), default=0.0)
            gt = max((c["clock_mhz_gt_max"] for c in cs), default=0.0)
            gt_lo = min((c.get("clock_mhz_gt_min", c["clock_mhz_gt_max"]) for c in cs), default=0.0)
            if host > best[0]:
                best = (host, t, k)
            nvm = max((c["nvml_sm_mhz"] for c in cs), default=0)
            watts = max((c["nvml_mw"] for c in cs), default=0) / 1e3
            reasons = 0
            for c in cs:
                reasons |= int(c["throttle"], 16)
            head = (f"{t:26s} {str(tr.get('rc', '')):>3s} {str(_val(s, 'clock_now', 'sm')):>8s} "
                    f"{str(_val(s, 'clock_max', 'sm')):>6s} {str(_val(s, 'applications', 'graphics')):>6s}") if first \
                else " " * 54
            msg = " ".join(tr.get("out", "").split())[:90] if first else ""
            print(f"{head} {k:9s} {host:9.1f} {gt:9.1f} {gt_lo:9.1f} {nvm:5d} {watts:5.0f} {reasons:#8x}  {msg}")
            first = False
    if best[0]:
        print(f"\nhighest host-timed clock: {best[0]:.1f} MHz ({best[2]} at {best[1]}), "
              f"{100 * (best[0] / fmax - 1):+.3f}% against NVML's max {fmax:.0f}")
    return 0


#: VF2's UNI class: every uniform-datapath opcode, the constant and special-register loads into uniform registers, and
#: the moves into them
UNIFORM = re.compile(r"U[A-Z].*|LDCU|R2UR|S2UR|CS2UR|VOTEU")
SASS_INSN = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)")


def loop_ops(sass: str, kernel: str, head: str, tail: str) -> list[str]:
    """The opcodes of `kernel`'s instructions from `head` to `tail` (the gate's loop) in a cuobjdump listing."""
    body = sass.split(f"Function : {kernel}\n", 1)[1].split("Function : ", 1)[0]
    lo, hi = int(head, 16), int(tail, 16)
    return [m.group(2) for m in SASS_INSN.finditer(body) if lo <= int(m.group(1), 16) <= hi]


def slope(pts: list[tuple[float, float]]) -> float:
    mx, my = statistics.fmean(p[0] for p in pts), statistics.fmean(p[1] for p in pts)
    return sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)


def cmd_uniform(a) -> int:
    """uniform.cu: each loop's vector (V) and uniform (U) instructions a warp-iteration from its SASS, its SMSP-cycles a
    warp-iteration (T), and the issue cost of a uniform instruction: T's slope in U within each fixed stream and step."""
    out = a.out
    sass = (out / "sass" / "uniform.sass").read_text()
    loops = {v["probe"]: v["verdict"]["loops"][0] for v in json.loads((out / "gate.json").read_text())["verdicts"]
             if v["passed"] and v["verdict"].get("loops")}
    pts = {}
    for r in load(out / "rows.jsonl"):
        lp = loops.get(r["probe"])
        if not lp:
            continue
        ops = [o for o in loop_ops(sass, r["probe"], lp["head"], lp["tail"]) if o != "NOP"]
        u = sum(1 for o in ops if UNIFORM.fullmatch(o.split(".")[0]))
        n = sum(1 for o in ops if fnmatchcase(o, r["counted"]))
        pts[r["probe"]] = (len(ops) - u, u, SMSPS * n / r["rate"], r.get("tags") or [])
    print(f"{'probe':9s} {'V':>4s} {'U':>4s} {'T':>8s} {'V/T':>6s} {'(V+U)/T':>8s} {'U/T':>6s} {'slots/U vs K=0':>14s}  tags")
    for p, (v, u, t, tags) in pts.items():
        base = pts.get(p.split("_")[0] + "_0")
        c = (t - base[2]) / (u - base[1]) if base and u != base[1] else float("nan")
        print(f"{p:9s} {v:4d} {u:4d} {t:8.3f} {v / t:6.3f} {(v + u) / t:8.3f} {u / t:6.3f} {c:14.3f}  {' '.join(tags)}")
    print("SMSP-cycles a uniform instruction, each fixed stream's UIMAD+ULOP3 points: between neighbours in U, and the "
          "least-squares slope")
    for fam in sorted({p.split("_")[0] for p in pts}):
        fp = sorted((u, t) for p, (_, u, t, _) in pts.items() if p.split("_")[0] == fam and p.split("_")[1].isdigit())
        if len(fp) >= 2:
            steps = " ".join(f"{u0}->{u1}:{(t1 - t0) / (u1 - u0):.3f}" for (u0, t0), (u1, t1) in zip(fp, fp[1:]))
            print(f"  {fam}: {steps}  fit {slope(fp):.3f} over U {fp[0][0]}..{fp[-1][0]}")
    return 0


#: qstar's fastest table kernel (fr16_c32, benchmarks/pouw/qstar) sustains this
QSTAR_TBS = 3.3


def cmd_l2gather(a) -> int:
    """l2gather.cu: data-dependent row gathers from an L2-resident table at fresh random indices (a cheater's). Each
    probe's sustained bytes a SM a clock (rate * words * 4, the warp instruction moving W*32 bytes) and TB/s (* nSM *
    clock); alone (gl_/gt_/gn_) for the peak, beside the honest IMMA step (gh_, the gather's rate over the honest stop)
    for the MMA rate kept. The best sustained of each, and whether any path beats QSTAR's 3.3 TB/s table kernel."""
    rows = {r["probe"]: r for r in load(a.out / "rows.jsonl")}
    hmma = loop_rate(rows["honest"], IMMA) if "honest" in rows else float("nan")
    print(f"honest alone: IMMA {hmma:.5f}/SM/clk")

    def read(r: dict) -> tuple[float, float, float, float]:
        """gather rate, bytes a SM a clock, TB/s, and MMA kept (nan when alone)."""
        bpc = r["rate"] * (r["words"] or 0) * 4
        tbs = bpc * r["blocks"] * (r["sm_mhz"] or 0) / 1e6
        beside = any(k.split(":", 1)[1] == IMMA for k in r["rates"])
        kept = loop_rate(r, IMMA) / hmma if beside and hmma else float("nan")
        return r["rate"], bpc, tbs, kept

    print(f"{'probe':15s} {'note':17s} {'W':>3s} {'rate':>8s} {'B/SM/clk':>9s} {'TB/s':>6s} {'MMA kept':>9s}  tags")
    best_alone = best_beside = None
    for name, r in rows.items():
        if name == "honest":
            continue
        rate, bpc, tbs, kept = read(r)
        w = int(round((r["words"] or 0) / 8))
        tag = " ".join(r.get("tags") or [])
        print(f"{name:15s} {r.get('note') or '':17s} {w:3d} {rate:8.4f} {bpc:9.3f} {tbs:6.3f} {kept:9.4f}  {tag}")
        if kept == kept:  # beside (MMA kept is a number)
            if best_beside is None or bpc > best_beside[1]:
                best_beside = (name, bpc, tbs, kept)
        elif best_alone is None or bpc > best_alone[1]:
            best_alone = (name, bpc, tbs)
    if best_alone:
        print(f"best alone:  {best_alone[0]} {best_alone[1]:.3f} B/SM/clk = {best_alone[2]:.3f} TB/s")
    if best_beside:
        print(f"best beside: {best_beside[0]} {best_beside[1]:.3f} B/SM/clk = {best_beside[2]:.3f} TB/s, "
              f"MMA kept {best_beside[3]:.4f}")
    peak = max((read(r)[2] for n, r in rows.items() if n != "honest"), default=0.0)
    verb = "beats" if peak > QSTAR_TBS else "does not beat"
    print(f"fastest path {peak:.3f} TB/s {verb} QSTAR's {QSTAR_TBS} TB/s table kernel")
    return 0


def cmd_jitter(a) -> int:
    """clock_calib.sh: the auditor-side uncertainty of timing a known-length reference kernel with the host's own clock,
    with no trust in the chip (reading C). Each (trial, length) group's host-timed SM clock over the reps: its mean, the
    random spread (stddev and half-range), and the systematic gap below the chip's own %globaltimer clock, which is the
    launch latency and shrinks as the kernel runs longer. Timing can only lower-bound the clock (a prover runs slower at
    will), so u is the floor an auditor quotes for that bound: the random spread plus the residual gap at the longest
    length."""
    rows = [r for r in load(a.out / "jitter.jsonl") if r.get("op") == "calib"]
    groups: dict[tuple[str, float], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["trial"], r["len_s"]), []).append(r)
    print(f"{'trial':8s} {'len_s':>7s} {'reps':>4s} {'host MHz':>9s} {'stddev':>7s} {'half-range':>10s} {'rel u%':>7s} "
          f"{'gt MHz':>9s} {'gap%':>6s} {'NVML':>5s}")
    best: dict[str, tuple[float, float, float]] = {}
    for (t, ln), cs in sorted(groups.items()):
        host = [c["clock_mhz_host"] for c in cs]
        gt = [c["clock_mhz_gt_median"] for c in cs]
        mean = statistics.fmean(host)
        sd = statistics.stdev(host) if len(host) > 1 else 0.0
        half = (max(host) - min(host)) / 2
        rel = 100 * half / mean if mean else 0.0
        gtm = statistics.fmean(gt)
        gap = 100 * (gtm - mean) / gtm if gtm else 0.0
        nvm = statistics.median(c["nvml_sm_mhz"] for c in cs)
        print(f"{t:8s} {ln:7.3f} {len(cs):4d} {mean:9.1f} {sd:7.2f} {half:10.2f} {rel:7.3f} {gtm:9.1f} {gap:6.3f} "
              f"{nvm:5.0f}")
        if t not in best or ln > best[t][0]:
            best[t] = (ln, rel, gap)
    for t, (ln, rel, gap) in best.items():
        print(f"{t}: at {ln:.3f}s the host clock pins the SM clock to within {rel:.3f}% (random) and sits {gap:.3f}% "
              f"below the chip's timer (launch latency); u ~ {rel + gap:.3f}%")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("facets")
    f.add_argument("rows", type=Path, nargs="+")
    f.add_argument("--margin", type=float, default=0.02)
    f.add_argument("--prior", type=Path, help="rows.csv: add the points each family's published facets were fitted to")
    f.add_argument("--emit", metavar="EPOCH", help="print the facets as rows.csv and census.csv lines from EPOCH")
    s = sub.add_parser("side")
    s.add_argument("rows", type=Path)
    c = sub.add_parser("clock")
    c.add_argument("sustain", type=Path)
    c.add_argument("--skip-s", type=float, default=3.0)
    ce = sub.add_parser("ceiling")
    ce.add_argument("out", type=Path, help="a clock_ceiling.sh run's out/ (found.json, trials, states, calib)")
    u = sub.add_parser("uniform")
    u.add_argument("out", type=Path, help="a uniform.cu run's out/ (rows.jsonl, gate.json, sass/)")
    g = sub.add_parser("l2gather")
    g.add_argument("out", type=Path, help="an l2gather.cu run's out/ (rows.jsonl)")
    j = sub.add_parser("jitter")
    j.add_argument("out", type=Path, help="a clock_calib.sh run's out/ (jitter.jsonl)")
    a = p.parse_args(argv)
    return {"facets": cmd_facets, "side": cmd_side, "clock": cmd_clock, "ceiling": cmd_ceiling, "uniform": cmd_uniform,
            "l2gather": cmd_l2gather, "jitter": cmd_jitter}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
