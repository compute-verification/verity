"""N3Window_v3's ANDs on every link-window, beside N2's head checks: the simulator's and the lab's link-windows as built,
and served rows' busy link-windows at their shape, as v3 and as `n3_v4`'s N3Window_v4, lever by lever.

A link-window is its two port-windows, egress and ingress, and N3 runs one unit on each port-window that recorded frames
(`n3_unit.Setup.attach`). `n3_unit.size` counts a unit's ANDs by tracing it whole. At a served window's 35,000 frames of
4,096 bytes that doesn't fit in memory: N2's frames root (`N3FramesRoot`) and the salt's XOR (`N3Xor`) each hold a
list of every frame bit. `composed` counts the same window from its parts instead, as `N3Window_v3`'s body calls them:
each part traced at the window's statics, the per-frame parts times the frames, the frames root as `leaf_blocks(B)`
compressions a frame and 2 a node of its depth-D tree (`nodes`), and the XOR as none, since it is XORs alone.
`checked` requires `composed` to equal `size`, which traces both, at every N of a ladder. `composed_v4` and
`n3_v4.size` are the same for v4.

`composed_v5` is `n3_v5`'s N3Window_v5, and `levers` prices NET-52's levers on the busy link-windows from it.

    python -m workload.n3_cost --out FILE [--lab RUN_DIR] [--sim] [--check N ...] [--check-v4 N ...]
        [--runs EGRESS INGRESS] [--served N ...] [--levers [--check-v5 N ...]]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from verity.ml.circuits.defs import bind
from verity.ml.commitments.gates.frame_v3 import FrameV3Leaf, FrameV3Node, index_bytes
from verity.ml.commitments.gates.hm96 import Hm96Sha512Leaf
from verity.ml.commitments.gates.lms import LmsEqual, Sha512Compress
from verity.ml.commitments.gates.sha512 import pad_tail

from . import n3_unit as U
from . import n3_v4 as V4
from . import n3_v5 as V5

N2_LINK_WINDOW = 2 * (1_876_070 + 807 + 19_199 + 1_333_292 + 3)
"""N2's two head checks a link-window at T = 600 buckets and a link name of 9 bytes, 6,458,742 ANDs: each head's
opening, its fields, its counts against the profile, its signed digest (the record digest salted by the certifier) and
the 3 ANDs that join them (`head_check.size`)."""

SERVED = {"T": 600, "R": 1686, "B": 4096, "L": 9, "windows_a_day": 1440, "egress_frames": (34_499, 35_386),
          "ingress_share": 21_372 / 715_340}
"""llama8b-node1's served rows (the consistency draft, note:network-accounting/20261006T0633Z-draft-circuit-certificate-
consistency, run r20261006-035423-2cd5): windows of T = 600 buckets of 100 ms and r = 1,686 slots of 4,096 bytes, so
1,440 a day; busy egress link-windows of 34,499 to 35,386 real frames; and 21,372 ingress frames to 715,340 egress ones
over the ten configurations, which `served` takes as each link-window's ingress share."""


def leaf_blocks(B: int) -> int:
    """The compressions of one frame's leaf in N2's frames root, after the tag's block."""
    return (B + len(pad_tail(128 + B))) // 128


def nodes(N: int, D: int) -> int:
    """The nodes a depth-D frames tree of N leaves hashes, each level's odd node paired with the empty subtree."""
    level, total = N, 0
    for _ in range(D):
        level = (level + 1) // 2
        total += level
    return total


def composed(f: U.Format, N: int, G: int, salted: bool, rec: U.Shape, memo: dict, FB: int | None = None) -> dict:
    """`size` of `unit(f, N, G, salted, rec)`, by part, from the parts its body calls; with frames of FB bytes, the
    salt and the frames root at FB (the parts of v5's units on narrower rows that read the frames whole)."""
    def a(d) -> int:
        return U._ands(d, memo)

    n, FB = N - salted, f.B if FB is None else FB
    out = dict.fromkeys(("row_reads", "descriptor_reads", "record_opening", "segments", "frames", "frames_root",
                         "other"), 0)
    if salted:
        out["other"] += (a(bind(U.N3Slot, N=N)) + N * a(bind(U.N3Gate, W=8 * FB)) + a(bind(U.N3SaltEq, B=FB))
                         + n * a(bind(U.N3Mux, W=8 * FB)))
    if n:
        frame = bind(U.N3Frame, FMT=tuple(f), G=G)
        seg = sum(U._times(x) * a(x.fn) for x in frame.body.nodes if getattr(x.fn, "name", "") == "N3Seg")
        out["other"] += a(bind(U.N3Positions, DL=f.DL, N=n))
        out["descriptor_reads"] = n * a(bind(U.N3HidingRead, N_BITS=f.LW, D=f.DL))
        out["row_reads"] = n * G * a(bind(U.N3RowRead, DEPTHS=f.DEPTHS, N_BITS=16 * f.B))
        out["segments"], out["frames"] = n * seg, n * (a(frame) - seg)
    out["frames_root"] = a(bind(Sha512Compress)) * (leaf_blocks(FB) * N + 2 * nodes(N, rec.D))
    out["record_opening"] = a(bind(U.N3IsZero, W=8)) + a(bind(U.N3HidingRead, N_BITS=8 * (rec.HL + 64), D=2))
    out["other"] += salted + (2 + G) * n + 2 - 1
    out["ands"] = sum(out.values())
    return out


def composed_v4(f: U.Format, N: int, G: int, salted: bool, rec: U.Shape, R: int, memo: dict,
                FB: int | None = None) -> dict:
    """`n3_v4.size` of `n3_v4.unit(f, N, G, salted, rec, R)`, by part, from the parts its body calls: v3's, but for the
    descriptors' one run and the rows' R runs, and the ok bits the final AND joins."""
    def a(d) -> int:
        return U._ands(d, memo)

    out = composed(f, N, G, salted, rec, memo, FB)
    n = N - salted
    out["other"] -= salted + (2 + G) * n + 2 - 1
    T, Sr, zw = n * G, min(n * G, 1 << f.DB), max(1, (n * G).bit_length())
    if n:
        out["descriptor_reads"] = (n * a(bind(V4.N3HidingLeaf, N_BITS=f.LW, D=f.DL)) + a(bind(V4.N3Range, S=n, D=f.DL))
                                   + a(bind(LmsEqual, N=64)) + a(bind(V4.N3Fits, D=f.DL, N=n)))
        per_read = (a(bind(U.N3Pick, U=f.U)) + a(bind(V4.N3HidingLeaf, N_BITS=16 * f.B, D=f.DB))
                    + a(bind(U.N3IsZero, W=16 * f.B)) + a(bind(V4.N3Runs, R=R)))
        per_run = (T * (a(bind(V4.N3Carry)) + a(bind(V4.N3Skip, W=zw))) + (T - 1) * a(bind(U.N3Mux, W=512))
                   + sum(npos * a(bind(V4.N3CompactStep, W=W)) for npos, W in V4.compaction(T, Sr))
                   + Sr * a(bind(U.N3Mux, W=512)) + a(bind(U.N3Pick, U=f.U)) + a(bind(V4.N3Range, S=Sr, D=f.DB))
                   + a(bind(V4.N3RunRoot, DEPTHS=f.DEPTHS)) + a(bind(V4.N3RunFits, DEPTHS=f.DEPTHS, T=T, S=Sr)))
        out["row_reads"] = T * per_read + R * per_run
    out["other"] += salted + (2 + T + R * (T + 2) + n if n else 0) + 2 - 1
    out["ands"] = sum(v for k, v in out.items() if k != "ands")
    return out


def templated(out: dict, f: U.Format, N: int, G: int, salted: bool, tpl: tuple, memo: dict,
              FB: int | None = None) -> dict:
    """A composition with its frames (of FB bytes, B unless given) placed by the template (`n3_v5.N3FrameFixed`)
    instead of their descriptors: v4's or v5's with lever 2, whose other parts don't read the placement."""
    n = N - salted
    if n:
        frame = bind(V5.N3FrameFixed, FMT=tuple(f), G=G, TPL=tpl, FB=f.B if FB is None else FB)
        seg = sum(U._times(x) * U._ands(x.fn, memo) for x in frame.body.nodes if getattr(x.fn, "name", "") == "N3Seg")
        out = dict(out, frames=n * (U._ands(frame, memo) - seg))
        out["ands"] = sum(v for k, v in out.items() if k != "ands")
    return out


def composed_v5(f: U.Format, N: int, G: int, salted: bool, rec: U.Shape, R: int, memo: dict, tpl: tuple = (),
                FB: int | None = None) -> dict:
    """`n3_v5.size` of `n3_v5.unit(f, N, G, salted, rec, R, tpl, FB)`, by part, from the parts its body calls: v4's,
    but for the rows' runs, each read hashing one B-byte row and each run its head, and with a template its frames'
    (of FB bytes)."""
    def a(d) -> int:
        return U._ands(d, memo)

    out = composed_v4(f, N, G, salted, rec, R, memo, FB)
    n = N - salted
    T, W = n * G, 8 * f.B
    Sr, zw = min(T + 1, 1 << f.DB), max(1, (T + 1).bit_length())
    if n:
        leaf, pick = a(bind(V4.N3HidingLeaf, N_BITS=W, D=f.DB)), a(bind(U.N3Pick, U=f.U))
        per_read = pick + a(bind(V5.N3Next, D=f.DB)) + leaf + a(bind(V5.N3RunsOne, R=R))
        per_run = (pick + leaf + T * (a(bind(V4.N3Carry)) + a(bind(V4.N3Skip, W=zw)) + a(bind(V5.N3Advance, W=W))
                                      + a(bind(U.N3Gate, W=2 * W)))
                   + (T - 1) * a(bind(U.N3Mux, W=512))
                   + sum(len(ps) * a(bind(V4.N3CompactStep, W=w)) for ps, w in V5.compaction(T, Sr - 1))
                   + (Sr - 1) * a(bind(U.N3Mux, W=512)) + a(bind(V4.N3Range, S=Sr, D=f.DB))
                   + a(bind(V4.N3RunRoot, DEPTHS=f.DEPTHS)) + a(bind(V4.N3RunFits, DEPTHS=f.DEPTHS, T=T + 1, S=Sr)))
        out["row_reads"] = T * per_read + R * per_run
        out["ands"] = sum(v for k, v in out.items() if k != "ands")
    return templated(out, f, N, G, salted, tpl, memo, FB) if tpl != () else out


def checked(f: U.Format, rec: U.Shape, ladder: list[int], G: int, salted: bool, memo: dict,
            R: int | None = None, v5: tuple | None = None, FB: int | None = None) -> list[dict]:
    """`composed` beside `size` at each N of the ladder (v4's with R runs, given R; v5's with R runs and the template
    `v5`, given it, and frames of FB bytes); raises where they differ."""
    out = []
    for N in ladder:
        t0 = time.perf_counter()
        if R is None:
            whole, parts = U.size(U.unit(f, N, G, salted, rec), memo), composed(f, N, G, salted, rec, memo)
        elif v5 is None:
            whole = V4.size(V4.unit(f, N, G, salted, rec, R), memo)
            parts = composed_v4(f, N, G, salted, rec, R, memo)
        else:
            whole = V5.size(V5.unit(f, N, G, salted, rec, R, v5, FB), memo)
            parts = composed_v5(f, N, G, salted, rec, R, memo, v5, FB)
        if whole != parts:
            raise AssertionError(f"N = {N}: the window traces to {whole}, its parts to {parts}")
        out.append({"N": N, "G": G, "salted": salted, **({} if R is None else {"R": R}),
                    **({} if v5 is None else {"template": v5 != ()}), **({} if FB is None else {"FB": FB}),
                    "ands": whole["ands"],
                    "seconds": round(time.perf_counter() - t0, 1)})
        print("check", json.dumps(out[-1]), flush=True)
    return out


def fit(rows: list[dict]) -> dict:
    """The least-squares line ands = fixed + per_frame · N over the rows, and its largest relative residual."""
    xs, ys = [r["N"] for r in rows], [r["ands"] for r in rows]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    return {"fixed": round(a), "per_frame": round(b), "points": len(xs),
            "max_relative_residual": max(abs(y - (a + b * x)) / y for x, y in zip(xs, ys))}


LEVERS = (("descriptors", "descriptor_reads"), ("rows", "row_reads"), ("other", "other"))
"""v4's changes, one at a time: its part group against v3's, the rest of the window v3's. The groups are disjoint and
every other group is the same in both, so v3 less the three deltas is v4."""


def served(lab_fmt: U.Format | None, ladder: list[int], check: list[int], check_v4: list[int] = (),
           runs: tuple[int, int] = (1, 2)) -> dict:
    """Served rows' port-windows at their shape, and their busy link-windows, as v3 and as v4. The format is the lab's
    at B = 4,096 with a prefix of 256 bytes and a suffix of 64 (a guess at a served event's JSON around its crossing
    text; the lab's are 30 and 2), in three sizes of run: the lab's depths, one busy window's (every outbox 2^16 rows
    and the layout two ports of 2^16 positions, the least that hold 35,386 frames' reads), and a day's (2^26 and two
    ports of 2^26). Egress port-windows read one segment and are unsalted, ingress ones two and salted, as the lab's
    are; v4 opens egress reads as runs[0] runs and ingress ones as runs[1]. At the lab's depths v4 has no unit: a
    window's n descriptors are one run, which a layout of 2^6 positions doesn't hold. `check` holds v3's `composed` to
    the trace at the lab's depths, and `check_v4` v4's at one window's."""
    S = SERVED
    rec = U.Shape(S["L"], S["T"], S["R"])
    base = lab_fmt or U.Format(S["B"], 30, 2, 2, (1, 0, 0, 0), 6)
    fmts = {"lab": base._replace(P=256, Q=64),
            "window": base._replace(P=256, Q=64, DEPTHS=(16,) * len(base.DEPTHS), DL=17),
            "day": base._replace(P=256, Q=64, DEPTHS=(26,) * len(base.DEPTHS), DL=27)}
    out: dict = {"shape": {"T": rec.T, "R": rec.R, "B": S["B"], "L": rec.L, "D": rec.D}, "v4_runs": list(runs),
                 "runs": {}}
    egress = sorted(set(ladder) | set(S["egress_frames"]))
    ingress = sorted({max(2, round(S["ingress_share"] * n)) for n in S["egress_frames"]} | set(ladder))
    sides = (("egress", egress, 1, False, runs[0]), ("ingress", ingress, 2, True, runs[1]))
    for name, f in fmts.items():
        memo: dict = {}
        holds = max(egress) <= 1 << f.DL
        run: dict = {"format": f._asdict(), "check": [], "check_v4": [], "egress": [], "ingress": [],
                     "v4": {"egress": [], "ingress": []}}
        if name == "lab":
            run["check"] = checked(f, rec, check, 1, False, memo) + checked(f, rec, check, 2, True, memo)
        if name == "window":
            run["check_v4"] = (checked(f, rec, check_v4, 1, False, memo, runs[0])
                               + checked(f, rec, check_v4, 2, True, memo, runs[1]))
        for key, ns, G, salted, R in sides:
            for N in ns:
                t0 = time.perf_counter()
                run[key].append({"N": N, "G": G, "salted": salted, **composed(f, N, G, salted, rec, memo),
                                 "seconds": round(time.perf_counter() - t0, 1)})
                print(name, json.dumps(run[key][-1]), flush=True)
                if holds:
                    t0 = time.perf_counter()
                    run["v4"][key].append({"N": N, "G": G, "salted": salted, "R": R,
                                           **composed_v4(f, N, G, salted, rec, R, memo),
                                           "seconds": round(time.perf_counter() - t0, 1)})
                    print(name, "v4", json.dumps(run["v4"][key][-1]), flush=True)
        run["egress_fit"], run["ingress_fit"] = fit(run["egress"]), fit(run["ingress"])
        run["parts"] = part_ands(f, memo)
        if holds:
            run["v4"]["egress_fit"], run["v4"]["ingress_fit"] = fit(run["v4"]["egress"]), fit(run["v4"]["ingress"])
            n_out = max(S["egress_frames"])
            run["v4"]["run_cost"] = {"egress": run_cost(f, n_out, 1, False, rec, runs[0], memo),
                                     "ingress": run_cost(f, max(2, round(S["ingress_share"] * n_out)), 2, True, rec,
                                                         runs[1], memo)}
        busy = []
        for n_out in S["egress_frames"]:
            n_in = max(2, round(S["ingress_share"] * n_out))
            e = next(r for r in run["egress"] if r["N"] == n_out)
            i = next(r for r in run["ingress"] if r["N"] == n_in)
            row = {"egress_frames": n_out, "ingress_frames": n_in, "egress": e["ands"], "ingress": i["ands"],
                   "n3": e["ands"] + i["ands"], "n3_a_day": (e["ands"] + i["ands"]) * S["windows_a_day"],
                   "n2": N2_LINK_WINDOW, "n2_a_day": N2_LINK_WINDOW * S["windows_a_day"],
                   "floor": e["frames_root"] + i["frames_root"]}
            e4 = next((r for r in run["v4"]["egress"] if r["N"] == n_out), None)
            i4 = next((r for r in run["v4"]["ingress"] if r["N"] == n_in), None)
            if e4 and i4:
                row["v4"] = e4["ands"] + i4["ands"]
                row["v4_floor"] = e4["frames_root"] + i4["frames_root"]
                row["levers"] = {lever: (e[g] + i[g]) - (e4[g] + i4[g]) for lever, g in LEVERS}
                row["parts"] = {g: (e[g] + i[g], e4[g] + i4[g]) for g in e if g not in ("N", "G", "salted",
                                                                                        "seconds")}
            busy.append(row)
        run["busy_link_windows"] = busy
        out["runs"][name] = run
    return out


SERVED_TEMPLATES = {"egress": V5.tpl_of(U.B64, 256, 4032, bytes(256), bytes(64)),
                    "ingress": V5.tpl_of(U.RAW, 256, 4032, bytes(256), bytes(64))}
"""Lever 2's templates at the served shape, the format's guess fixed: every frame's prefix 256 bytes and its suffix 64
at 4,032, the 3,776 bytes between its text, base64 of up to 2,832 data bytes on egress (an event's) and raw on
ingress. Their bytes are constants, which cost no ANDs, so only the mode and the offsets price."""


def shared_first_blocks(f: U.Format, N: int, G: int, salted: bool, R: int, memo: dict) -> dict:
    """Lever 3's estimate on v5, by the group it saves in: one compression for each node past the first on every level
    of the descriptors' range and of each run's, were a level's shared first block (the fixed prefix, the domain id
    and the level) compressed once. Not built: it needs a node part that takes a midstate."""
    n = N - salted
    if not n:
        return {"descriptor_reads": 0, "row_reads": 0}
    Sr, c = min(n * G + 1, 1 << f.DB), U._ands(bind(Sha512Compress), memo)
    return {"descriptor_reads": c * sum(V4.width(n, k + 1, f.DL) - 1 for k in range(f.DL)),
            "row_reads": c * R * sum(V4.width(Sr, k + 1, f.DB) - 1 for k in range(f.DB))}


def canonical_geometry(f: U.Format, N: int, salted: bool, rec: U.Shape, tpl: tuple, memo: dict) -> dict:
    """An estimate, not a unit: a window whose frames are its port's outbox rows. Each of the port's frames carries
    one row of C = s - p data bytes (base64) or s - p bytes (raw) at the template's place, so the outbox is per port
    and its row j is frame j's data, zero past its message's last byte; frame j then reads row j at a public position,
    with no descriptor, no segment and no run. Priced from the parts such a unit would call: v5's frames root, record
    opening and salt, each row's hiding leaf (`N3HidingLeaf` of C bytes), one range opening of the window's rows in
    the port's tree (`N3Range`), and each frame's placement from its row (`N3PlaceFixed`, r its own)."""
    def a(d) -> int:
        return U._ands(d, memo)

    mode, p, s, _, _ = V5.template(tpl, f.B)
    C = (s - p) * 3 // 4 if mode == U.B64 else s - p
    n = N - salted
    out = composed(f, N, 1, salted, rec, memo)
    out = {k: out[k] for k in ("record_opening", "frames_root")} | {"other": 0}
    if salted:
        out["other"] = (a(bind(U.N3Slot, N=N)) + N * a(bind(U.N3Gate, W=8 * f.B)) + a(bind(U.N3SaltEq, B=f.B))
                        + n * a(bind(U.N3Mux, W=8 * f.B)))
    out["row_reads"] = n * a(bind(V4.N3HidingLeaf, N_BITS=8 * C, D=f.DB)) + a(bind(V4.N3Range, S=n, D=f.DB))
    out["frames"] = n * a(bind(V5.N3PlaceFixed, FMT=tuple(f), TPL=tpl, FB=f.B))
    out["other"] += salted + n + 3 - 1
    out["ands"] = sum(out.values())
    return out


EGRESS_ROWS = (64, 16, 256, 512)
"""Lever 4's egress outbox rows, in bytes (the first; the others its sensitivity): a served egress frame is one token's
event (`network_traces/record.py`), so its data is one token's text, and rows that hold one are enough (512 hold any
token of GLM-5.3's vocabulary, whose longest is 512 bytes and 124 of 154,820 longer than 64). Ingress frames are a
prompt's bytes packed whole, and keep rows of B."""


def levers(lab_fmt: U.Format | None, check_v5: list[int], runs: tuple[int, int] = (1, 2),
           egress_rows: tuple[int, ...] = EGRESS_ROWS) -> dict:
    """NET-52's levers on served rows' busy link-windows, at one busy window's depths and a day's (`served`'s formats):
    v4; lever 1, v5's rows of B bytes; lever 2, the frames' placement a template's (`SERVED_TEMPLATES`, the descriptor
    then only r and the segments, P = Q = 1), on v4 and on v5; lever 4 on both, egress rows of `egress_rows[0]` bytes
    under frames of B (the rest of `egress_rows` beside it); and lever 3's estimate on v5 with lever 2, and with 2
    and 4.
    Each lever's delta and the cumulative total are against the floor, N2's frames root of the link-window's frames,
    which every variant computes as v3 does. `check_v5` holds `composed_v5` to v5's trace at one window's depths,
    without the template, with it, and with it on lever 4's rows."""
    S = SERVED
    rec = U.Shape(S["L"], S["T"], S["R"])
    base = lab_fmt or U.Format(S["B"], 30, 2, 2, (1, 0, 0, 0), 6)
    fmts = {"window": base._replace(P=256, Q=64, DEPTHS=(16,) * len(base.DEPTHS), DL=17),
            "day": base._replace(P=256, Q=64, DEPTHS=(26,) * len(base.DEPTHS), DL=27)}
    out: dict = {"shape": {"T": rec.T, "R": rec.R, "B": S["B"], "L": rec.L, "D": rec.D}, "runs": list(runs),
                 "templates": {k: {"mode": v[0], "p": v[1], "s": v[2]} for k, v in SERVED_TEMPLATES.items()},
                 "depths": {}}
    for name, f in fmts.items():
        memo: dict = {}
        ft = f._replace(P=1, Q=1)
        run: dict = {"format": f._asdict(), "check_v5": [], "busy_link_windows": []}
        if name == "window":
            run["check_v5"] = (checked(f, rec, check_v5, 1, False, memo, runs[0], ())
                               + checked(f, rec, check_v5, 2, True, memo, runs[1], ())
                               + checked(ft, rec, check_v5, 1, False, memo, runs[0], SERVED_TEMPLATES["egress"])
                               + checked(ft, rec, check_v5, 2, True, memo, runs[1], SERVED_TEMPLATES["ingress"])
                               + checked(ft._replace(B=egress_rows[0]), rec, check_v5, 1, False, memo, runs[0],
                                         SERVED_TEMPLATES["egress"], f.B))
        for n_out in S["egress_frames"]:
            n_in = max(2, round(S["ingress_share"] * n_out))
            sides = (("egress", n_out, 1, False, runs[0]), ("ingress", n_in, 2, True, runs[1]))
            row: dict = {"egress_frames": n_out, "ingress_frames": n_in, "variants": {}, "egress_rows": {}}
            for key, N, G, salted, R in sides:
                t0 = time.perf_counter()
                tpl = SERVED_TEMPLATES[key]
                v = {"v4": composed_v4(f, N, G, salted, rec, R, memo),
                     "lever_1": composed_v5(f, N, G, salted, rec, R, memo),
                     "lever_2": templated(composed_v4(ft, N, G, salted, rec, R, memo), ft, N, G, salted, tpl, memo),
                     "levers_1_2": composed_v5(ft, N, G, salted, rec, R, memo, tpl)}
                fr = ft._replace(B=egress_rows[0]) if key == "egress" else ft
                v["levers_1_2_4"] = composed_v5(fr, N, G, salted, rec, R, memo, tpl, f.B)
                for k, of, fk in (("levers_1_2_3", "levers_1_2", ft), ("levers_1_2_3_4", "levers_1_2_4", fr)):
                    v[k] = dict(v[of])
                    for g, saved in shared_first_blocks(fk, N, G, salted, R, memo).items():
                        v[k][g] -= saved
                    v[k]["ands"] = sum(x for g, x in v[k].items() if g != "ands")
                v["geometry_estimate"] = canonical_geometry(ft, N, salted, rec, tpl, memo)
                for RB in egress_rows[1:]:
                    e = composed_v5(ft._replace(B=RB), N, G, salted, rec, R, memo, tpl, f.B) if key == "egress" else (
                        v["levers_1_2"])
                    row["egress_rows"].setdefault(RB, {})[key] = e["ands"]
                for k, parts in v.items():
                    row["variants"].setdefault(k, {})[key] = parts
                print(name, key, N, json.dumps({k: x["ands"] for k, x in v.items()}),
                      round(time.perf_counter() - t0, 1), flush=True)
            floor = sum(row["variants"]["v4"][k]["frames_root"] for k in ("egress", "ingress"))
            row["floor"] = floor
            for k, parts in row["variants"].items():
                total = parts["egress"]["ands"] + parts["ingress"]["ands"]
                if any(parts[s]["frames_root"] != row["variants"]["v4"][s]["frames_root"] for s in ("egress", "ingress")):
                    raise AssertionError(f"{k}'s frames root isn't v4's")
                row[k] = {"ands": total, "x_floor": round(total / floor, 3), "a_day": total * S["windows_a_day"]}
            for RB, sides_ands in row["egress_rows"].items():
                total = sum(sides_ands.values())
                row["egress_rows"][RB] = {"ands": total, "x_floor": round(total / floor, 3)}
            v4 = row["v4"]["ands"]
            row["deltas"] = {k: v4 - row[k]["ands"] for k in row["variants"] if k != "v4"}
            run["busy_link_windows"].append(row)
        out["depths"][name] = run
    return out


def part_ands(f: U.Format, memo: dict) -> dict:
    """The parts a frame's cost is made of, at the format: what lever 3's estimates are composed from."""
    def a(d) -> int:
        return U._ands(d, memo)

    frame = bind(U.N3Frame, FMT=tuple(f), G=1)
    seg = a(bind(U.N3Seg, B=f.B))
    return {"sha512_compress": a(bind(Sha512Compress)), "row_digest": a(bind(U.N3RowDigest, N_BITS=16 * f.B)),
            "half_row_digest": a(bind(U.N3RowDigest, N_BITS=8 * f.B)),
            "descriptor_digest": a(bind(U.N3RowDigest, N_BITS=f.LW)), "hm96_leaf": a(bind(Hm96Sha512Leaf)),
            "frame_v3_leaf": a(bind(FrameV3Leaf, B=index_bytes(f.DB), SCHEMA=U.SCHEMA)),
            "frame_v3_node": a(bind(FrameV3Node, B=index_bytes(f.DB - 1))), "row_is_zero": a(bind(U.N3IsZero, W=16 * f.B)),
            "segment": seg, "frame_less_segment": a(frame) - seg,
            "frames_root_a_frame": a(bind(Sha512Compress)) * (leaf_blocks(f.B) + 2)}


def run_cost(f: U.Format, N: int, G: int, salted: bool, rec: U.Shape, R: int, memo: dict) -> int:
    """What one more of v4's runs adds to a window."""
    return composed_v4(f, N, G, salted, rec, R + 1, memo)["ands"] - composed_v4(f, N, G, salted, rec, R, memo)["ands"]


def link_windows(sized: list[dict], link_of) -> list[dict]:
    """Each link-window's N3 ANDs: its port-windows' units, summed, with each port-window's frames."""
    by: dict = defaultdict(lambda: {"ports": {}, "n3": 0})
    for r in sized:
        lw = by[(link_of(r["port"]), r["window"])]
        lw["ports"][r["port"]] = {"recorded": r["recorded"], "ands": r.get("ands", 0), "by": r["by"]}
        lw["n3"] += r.get("ands", 0)
    return [{"link": k[0], "window": k[1], **v, "n2_head_checks": 2} for k, v in sorted(by.items())]


def sim() -> dict:
    """The simulator's honest run, as `n3_unit sim` runs it by default."""
    from .block import Context
    from .n3 import merge
    from .run import run_memory
    from .run import statement as run_statement

    config = {"workload": "mlp", "spec": {"d": 4, "layers": 3, "requests": 3, "groups": 1}, "blocks": 3,
              "devices": 2, "frame_bytes": 64, "seed": 0}
    ctx = Context(config)
    results = run_memory(ctx)
    public, witness = run_statement(ctx, results)
    records, _ = merge(results)
    setup, windows = U.sim_windows(public, witness, records, 4)
    return {"config": config, "format": setup.fmt._asdict(),
            "link_windows": link_windows(U._sized(windows, circuit=False), lambda p: setup.links[p][0])}


def lab(run: Path) -> tuple[dict, U.Format]:
    """A lab run's windows, as `n3_unit lab` builds them."""
    here = Path(__file__).resolve().parents[1]
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    import records  # noqa: F401, PLC0415  (puts network_traces on the path, for active_replay)

    import active_replay as ar  # noqa: PLC0415

    from . import lab as L

    public = json.loads((run / "public.json").read_text())
    witness = json.loads((run / "witness.json").read_text())
    report = ar.Setup(json.loads((run / "report.json").read_text()))
    B = public["frame_bytes"]
    published = {block: [ar.unpack_record(x, B) for f in (L.EGRESS, L.INGRESS)
                         for x in ar.read_blobs(run / block / f"{f}.records")]
                 for block in sorted(p.name for p in run.iterdir() if (p / f"{L.EGRESS}.records").exists())}
    setup, windows = U.lab_windows(public, witness, published, grids=(report.egress_params, report.ingress_params),
                                   sync_set=report.sync_set)
    return {"run": str(run), "format": setup.fmt._asdict(),
            "link_windows": link_windows(U._sized(windows, circuit=False), lambda p: p.split(":")[0])}, setup.fmt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lab", type=Path, help="a lab run's directory (lab_run --out); its format is the served one's base")
    ap.add_argument("--sim", action="store_true", help="the simulator's honest run, 64-byte frames, 4 buckets a window")
    ap.add_argument("--check", type=int, nargs="*", default=[2, 3, 64, 257, 1024],
                    help="the N at which `composed` must equal `size`, at the served shape")
    ap.add_argument("--check-v4", type=int, nargs="*", default=[2, 3, 64],
                    help="the N at which `composed_v4` must equal `n3_v4.size`, at one busy window's depths")
    ap.add_argument("--runs", type=int, nargs=2, default=[1, 2], metavar=("EGRESS", "INGRESS"),
                    help="v4's runs a port-window, egress and ingress")
    ap.add_argument("--served", type=int, nargs="*", default=[2, 16, 256, 4096, 16384],
                    help="the N of the served ladder, beside the busy windows' own")
    ap.add_argument("--levers", action="store_true", help="NET-52's levers on the busy link-windows instead of `served`")
    ap.add_argument("--check-v5", type=int, nargs="*", default=[2, 3, 64],
                    help="with --levers, the N at which `composed_v5` must equal `n3_v5.size`, at one busy window's depths")
    args = ap.parse_args(argv)
    out: dict = {}
    fmt = None
    if args.sim:
        out["sim"] = sim()
        args.out.write_text(json.dumps(out, indent=1))
    if args.lab:
        out["lab"], fmt = lab(args.lab)
        args.out.write_text(json.dumps(out, indent=1))
    if args.levers:
        out["levers"] = levers(fmt, args.check_v5, tuple(args.runs))
        args.out.write_text(json.dumps(out, indent=1))
        print(json.dumps({k: [{x: r[x] for x in r if x != "variants"} for r in v["busy_link_windows"]]
                          for k, v in out["levers"]["depths"].items()}, indent=1))
        return 0
    out["served"] = served(fmt, args.served, args.check, args.check_v4, tuple(args.runs))
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v["busy_link_windows"] for k, v in out["served"]["runs"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
