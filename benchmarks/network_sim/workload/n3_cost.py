"""N3Window_v3's ANDs on every link-window, beside N2's head checks: the simulator's and the lab's link-windows as built,
and served rows' busy link-windows at their shape.

A link-window is its two port-windows, egress and ingress, and N3 runs one unit on each port-window that recorded frames
(`n3_unit.Setup.attach`). `n3_unit.size` counts a unit's ANDs by tracing it whole. At a served window's 35,000 frames of
4,096 bytes that doesn't fit in memory: N2's frames root (`N3FramesRoot`) and the salt's XOR (`N3Xor`) each hold a
list of every frame bit. `composed` counts the same window from its parts instead, as `N3Window_v3`'s body calls them:
each part traced at the window's statics, the per-frame parts times the frames, the frames root as `leaf_blocks(B)`
compressions a frame and 2 a node of its depth-D tree (`nodes`), and the XOR as none, since it is XORs alone.
`checked` requires `composed` to equal `size`, which traces both, at every N of a ladder.

    python -m workload.n3_cost --out FILE [--lab RUN_DIR] [--sim] [--check N ...] [--served N ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from verity.ml.circuits.defs import bind
from verity.ml.commitments.gates.lms import Sha512Compress
from verity.ml.commitments.gates.sha512 import pad_tail

from . import n3_unit as U

N2_LINK_WINDOW = 2 * (1_876_070 + 807 + 19_199 + 2)
"""N2's two head checks a link-window at T = 600 buckets and a link name of 9 bytes, 3,792,156 ANDs: each head's
opening, its fields, its counts against the profile and the 2 ANDs that join them (`head_check.size`)."""

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


def composed(f: U.Format, N: int, G: int, salted: bool, rec: U.Shape, memo: dict) -> dict:
    """`size` of `unit(f, N, G, salted, rec)`, by part, from the parts its body calls."""
    def a(d) -> int:
        return U._ands(d, memo)

    n = N - salted
    out = dict.fromkeys(("row_reads", "descriptor_reads", "record_opening", "segments", "frames", "frames_root",
                         "other"), 0)
    if salted:
        out["other"] += (a(bind(U.N3Slot, N=N)) + N * a(bind(U.N3Gate, W=8 * f.B)) + a(bind(U.N3SaltEq, B=f.B))
                         + n * a(bind(U.N3Mux, W=8 * f.B)))
    if n:
        frame = bind(U.N3Frame, FMT=tuple(f), G=G)
        seg = sum(U._times(x) * a(x.fn) for x in frame.body.nodes if getattr(x.fn, "name", "") == "N3Seg")
        out["other"] += a(bind(U.N3Positions, DL=f.DL, N=n))
        out["descriptor_reads"] = n * a(bind(U.N3HidingRead, N_BITS=f.LW, D=f.DL))
        out["row_reads"] = n * G * a(bind(U.N3RowRead, DEPTHS=f.DEPTHS, N_BITS=16 * f.B))
        out["segments"], out["frames"] = n * seg, n * (a(frame) - seg)
    out["frames_root"] = a(bind(Sha512Compress)) * (leaf_blocks(f.B) * N + 2 * nodes(N, rec.D))
    out["record_opening"] = a(bind(U.N3IsZero, W=8)) + a(bind(U.N3HidingRead, N_BITS=8 * (rec.HL + 64), D=2))
    out["other"] += salted + (2 + G) * n + 2 - 1
    out["ands"] = sum(out.values())
    return out


def checked(f: U.Format, rec: U.Shape, ladder: list[int], G: int, salted: bool, memo: dict) -> list[dict]:
    """`composed` beside `size` at each N of the ladder; raises where they differ."""
    out = []
    for N in ladder:
        t0 = time.perf_counter()
        whole, parts = U.size(U.unit(f, N, G, salted, rec), memo), composed(f, N, G, salted, rec, memo)
        if whole != parts:
            raise AssertionError(f"N = {N}: the window traces to {whole}, its parts to {parts}")
        out.append({"N": N, "G": G, "salted": salted, "ands": whole["ands"],
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


def served(lab_fmt: U.Format | None, ladder: list[int], check: list[int]) -> dict:
    """Served rows' port-windows at their shape, and their busy link-windows. The format is the lab's at B = 4,096 with
    a prefix of 256 bytes and a suffix of 64 (a guess at a served event's JSON around its crossing text; the lab's
    are 30 and 2), in two sizes of run: the lab's depths, and a day's (every outbox 2^26 rows, the layout two ports of
    2^26 positions). Egress port-windows read one segment and are unsalted, ingress ones two and salted, as the lab's
    are."""
    S = SERVED
    rec = U.Shape(S["L"], S["T"], S["R"])
    base = lab_fmt or U.Format(S["B"], 30, 2, 2, (1, 0, 0, 0), 6)
    fmts = {"lab": base._replace(P=256, Q=64),
            "day": base._replace(P=256, Q=64, DEPTHS=(26,) * len(base.DEPTHS), DL=27)}
    out: dict = {"shape": {"T": rec.T, "R": rec.R, "B": S["B"], "L": rec.L, "D": rec.D}, "runs": {}}
    egress = sorted(set(ladder) | set(S["egress_frames"]))
    ingress = sorted({max(2, round(S["ingress_share"] * n)) for n in S["egress_frames"]} | set(ladder))
    for name, f in fmts.items():
        memo: dict = {}
        run = {"format": f._asdict(), "check": [], "egress": [], "ingress": []}
        if name == "lab":
            run["check"] = checked(f, rec, check, 1, False, memo) + checked(f, rec, check, 2, True, memo)
        for key, ns, G, salted in (("egress", egress, 1, False), ("ingress", ingress, 2, True)):
            for N in ns:
                t0 = time.perf_counter()
                run[key].append({"N": N, "G": G, "salted": salted, **composed(f, N, G, salted, rec, memo),
                                 "seconds": round(time.perf_counter() - t0, 1)})
                print(name, json.dumps(run[key][-1]), flush=True)
        run["egress_fit"], run["ingress_fit"] = fit(run["egress"]), fit(run["ingress"])
        busy = []
        for n_out in S["egress_frames"]:
            n_in = max(2, round(S["ingress_share"] * n_out))
            e = next(r["ands"] for r in run["egress"] if r["N"] == n_out)
            i = next(r["ands"] for r in run["ingress"] if r["N"] == n_in)
            busy.append({"egress_frames": n_out, "ingress_frames": n_in, "egress": e, "ingress": i, "n3": e + i,
                         "n3_a_day": (e + i) * S["windows_a_day"], "n2": N2_LINK_WINDOW,
                         "n2_a_day": N2_LINK_WINDOW * S["windows_a_day"]})
        run["busy_link_windows"] = busy
        out["runs"][name] = run
    return out


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
    ap.add_argument("--served", type=int, nargs="*", default=[2, 16, 256, 4096, 16384],
                    help="the N of the served ladder, beside the busy windows' own")
    args = ap.parse_args(argv)
    out: dict = {}
    fmt = None
    if args.sim:
        out["sim"] = sim()
        args.out.write_text(json.dumps(out, indent=1))
    if args.lab:
        out["lab"], fmt = lab(args.lab)
        args.out.write_text(json.dumps(out, indent=1))
    out["served"] = served(fmt, args.served, args.check)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v["busy_link_windows"] for k, v in out["served"]["runs"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
