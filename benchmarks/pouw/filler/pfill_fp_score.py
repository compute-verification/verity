"""The proved-floor arms' runner for an arm with a fused decode step at some points
(`fused(phase, B, L)`, fused_score.py) and a phase filler at others (`phase(phase, B, L)`, psched_score.py): `fscore
fscore` where it runs a fused step, `pscore pscore` where it places a phase filler, the locked `dscore dscore`
elsewhere. It changes nothing in the scorer or either runner. The build, the gates before anything is timed and the ncu
rows are both runners' together (dgate, pgate at each phase S, an sgate a chunk shape, gate, fgate; ncu for dfill_k,
the counter's control, flin_k with and without filler warps, and dfill_p), and the report is the locked `report` with
both runners' own gates, each computed by its runner's `report` over the same out dir: fused_receipts, fused_dram and
fused_exact (fused_score.py), phase_dram, phase_receipts and sgates (psched_score.py).

  python3 benchmarks/pouw/filler/pfill_fp_score.py ARM [--rounds 8] [--reps 3] [--no-robust]   build, lease, measure, report
  python3 benchmarks/pouw/filler/pfill_fp_score.py measure ARM --bin DIR --out DIR             the measurement, in a lease
  python3 benchmarks/pouw/filler/pfill_fp_score.py report OUT_DIR                              score a measured out dir
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fused_score as FS  # noqa: E402
import psched_score as PS  # noqa: E402
import score as SC  # noqa: E402

A, S = SC.A, SC.S
FUSED_GATES = ("fused_receipts", "fused_dram", "fused_exact")
PHASE_GATES = ("phase_dram", "phase_receipts", "sgates")


def drivers(arm, pts) -> dict:
    """Each point's driver: fscore at a fused step, pscore at a phase filler, dscore elsewhere; never both at one point.
    Both runners' checks hold: a fused or phase point is on a split and pumps the arm's `filler`."""
    out = {}
    for ph, b, l in pts:
        q, p = FS.fused_of(arm, ph, b, l), PS.phase_of(arm, ph, b, l)
        if q is not None and p is not None:
            raise A.ArmError(f"{arm.name}: {ph} {b}/{l} has both a fused step and a phase filler")
        FS.filler_of(arm, ph, b, l), PS.filler_of(arm, ph, b, l)
        out[(ph, b, l)] = ("fscore", q) if q is not None else ("pscore", p) if p is not None else ("dscore", None)
    return out


def measure(arm, bdir: Path, out: Path, rounds: int, reps: int, robust: bool) -> int:
    env = dict(os.environ, DFILL_COINS=(out / "coins.txt").read_text().strip(), DFILL_CHUNK_LOG2=str(arm.filler.chunk_log2))
    fl, err, rc_log = arm.filler, out / "errors.txt", out / "rc.txt"

    def note(s):
        with open(rc_log, "a") as f:
            f.write(s + "\n")

    def at(f):
        return dict(env, DFILL_CHUNK_LOG2=str(f.chunk_log2))

    pts = S.points(robust)
    by = drivers(arm, pts)
    shapes = sorted({(q.W, q.NC, q.S) for d, q in by.values() if d == "pscore"})
    fillers = [fl] + sorted({FS.filler_of(arm, ph, b, l) for ph, b, l in pts
                             if A.placed(arm, ph, b, l).place != "none"} - {fl}, key=dataclasses.astuple)
    SC.smi(out)
    SC.sh(["nvidia-smi", "-q", "-d", "CLOCK,PERFORMANCE,POWER"] + (["-i", env["GPU_LEASE_UUID"]]
                                                                  if env.get("GPU_LEASE_UUID") else []),
          out=out / "smi-q.txt", err=err)
    gates = [("dgate", [str(bdir / "dcap"), "dgate"], env)]
    if shapes:
        gates += [("dgate", [str(bdir / "dcap"), "pgate", *sorted({str(s) for _, _, s in shapes})], env)]
    gates += [("sgate", [str(bdir / "dscore"), "sgate", str(f.W), str(f.NC), str(f.S)], at(f)) for f in fillers]
    gates += [("gate", [str(bdir / "dcap"), "gate"], env), ("gate", [str(bdir / "dcap"), "fgate"], env)]
    for name, cmd, e in gates:
        rc = SC.sh(cmd, out=out / f"{name}.jsonl", err=err, env=e)
        note(f"{' '.join(cmd[1:])} rc {rc}")
        if rc:
            return 5
    m = "gpu__time_duration.sum,dram__bytes.sum"
    for f in fillers:
        rc = SC.sh(["ncu", "--clock-control", "none", "-k", "regex:dfill_k", "--launch-count", "3", "--metrics", m,
                    "--csv", str(bdir / "dcap"), "done", str(f.W), str(f.NC), str(f.S)], out=out / "ncu-filler.csv",
                   err=err, env=at(f))
        note(f"ncu filler {dataclasses.astuple(f)} rc {rc}")
    rc = SC.sh(["ncu", "--clock-control", "none", "-k", "regex:bw_read", "--launch-count", "3", "--metrics", m, "--csv",
                str(bdir / "dcap"), "bw", "1", "3"], out=out / "ncu-control.csv", err=err, env=dict(env, DCAP_NOGRAPH="1"))
    note(f"ncu control rc {rc}")
    rc = SC.sh(["ncu", "--clock-control", "none", "-k", "regex:flin_k", "--launch-count", "64", "--metrics", m, "--csv",
                str(bdir / "dcap"), "flin", *FS.NCU_FLIN, "1"], out=out / "ncu-fused.csv", err=err, env=env)
    note(f"ncu flin {' '.join(FS.NCU_FLIN)} rc {rc}")
    for w, nc, s in shapes:
        rc = SC.sh(["ncu", "--clock-control", "none", "-k", "regex:dfill_p", "--launch-count", "3", "--metrics", m, "--csv",
                    str(bdir / "pscore"), "pncu", str(w), str(nc), str(s), str(PS.PNCU_TILES)], out=out / "ncu-phase.csv",
                   err=err, env=env)
        note(f"ncu phase W {w} NC {nc} S {s} rc {rc}")
    SC.smi(out)
    for i, (ph, b, l) in enumerate(pts):
        p, (d, q) = A.placed(arm, ph, b, l), by[(ph, b, l)]
        f = FS.filler_of(arm, ph, b, l)
        if d == "fscore":
            cmd = [str(bdir / "fscore"), "fscore", str(b), str(l), str(p.fill_sms), str(f.W), str(f.NC), str(f.S),
                   *FS.fused_args(q), str(rounds), str(reps), str(i)]
        elif d == "pscore":
            cmd = [str(bdir / "pscore"), "pscore", str(b), str(l), str(p.fill_sms), str(f.W), str(f.NC), str(f.S),
                   str(q.W), str(q.NC), str(q.S), str(q.tiles), str(rounds), str(reps), str(i)]
        else:
            cmd = [str(bdir / "dscore"), "dscore", "step" if ph == "decode" else "prefill", str(b), str(l), p.place,
                   str(p.fill_sms), str(f.W), str(f.NC), str(f.S), str(rounds), str(reps), str(i)]
        rc = SC.sh(cmd, out=out / "dscore.jsonl", err=err, env=at(f))
        note(f"{cmd[1]} {ph} {b} {l} {p.place} {p.fill_sms} filler {dataclasses.astuple(f)} "
             f"{d} {dataclasses.astuple(q) if q else None} rc {rc}")
        SC.smi(out)
    SC.control(arm, fl, bdir, out, env, rounds, reps, len(pts), note)
    return 0


def report(out: Path, arm) -> dict:
    """The locked report with both runners' gates: fused_score.report's, then psched_score.report's (each reruns the
    locked report over the same out dir, so the second's res is the locked one with its own gates), merged."""
    fused = FS.report(out, arm)
    res = PS.report(out, arm)
    g = res["gates"]
    for k in FUSED_GATES:
        g[k] = fused["gates"][k]
    g["ok"] = g["ok"] and all(g[k]["ok"] for k in FUSED_GATES)
    res["scored"] = g["ok"] and res["score_if_gated"] is not None
    res["SCORE"] = res["score_if_gated"] if res["scored"] else None
    if "score_under_ch_if_gated" in res:
        res["score_under_ch"] = res["score_under_ch_if_gated"] if res["scored"] else None
    res["driver"] = ("pfill_fp_score.py (arm-side: fscore.cu for the fused points, pscore.cu for the phase points, the "
                     "locked dscore elsewhere)")
    res["fused_points"] = fused["fused_points"]
    res["harness"] = fused.get("harness")
    (out / "score.json").write_text(json.dumps(res, indent=1))
    return res


def show(res: dict) -> None:
    SC.show(res)
    for k in FUSED_GATES + PHASE_GATES:
        v = {kk: vv for kk, vv in res["gates"][k].items() if kk != "bound_share_of_hbm"}
        print(f"  gate {k}: {'pass' if res['gates'][k]['ok'] else 'FAIL'}  {v}")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "measure":
        ap = argparse.ArgumentParser(prog="pfill_fp_score.py measure")
        ap.add_argument("arm")
        ap.add_argument("--bin", type=Path, required=True)
        ap.add_argument("--out", type=Path, required=True)
        ap.add_argument("--rounds", type=int, default=8)
        ap.add_argument("--reps", type=int, default=3)
        ap.add_argument("--no-robust", action="store_true")
        a = ap.parse_args(argv[1:])
        return measure(A.load(a.arm), a.bin, a.out, a.rounds, a.reps, not a.no_robust)
    if argv and argv[0] == "report":
        ap = argparse.ArgumentParser(prog="pfill_fp_score.py report")
        ap.add_argument("out", type=Path)
        a = ap.parse_args(argv[1:])
        res = report(a.out, A.load(json.loads((a.out / "arm.json").read_text())["spec"]))
        show(res)
        return 0
    ap = argparse.ArgumentParser(prog="pfill_fp_score.py", description="score a fused-and-phase arm with the locked scorer")
    ap.add_argument("arm", help="benchmarks/pouw/filler/FILE.py:Class, an arm with fused(phase, B, L) and phase(phase, B, L)")
    ap.add_argument("--out", type=Path, default=Path(os.environ["OUT"]) if os.environ.get("OUT") else None)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-robust", action="store_true")
    ap.add_argument("--max-min", type=int, default=30, help="the lease's bound in minutes")
    ap.add_argument("--by", default=os.environ.get("GPU_LEASE_WHO") or os.environ.get("RESEARCH_LANE"))
    a = ap.parse_args(argv)
    if a.out is None:
        ap.error("--out (or $OUT, as research run sets it)")
    arm = A.load(a.arm, record=True)
    if not (callable(getattr(arm, "fused", None)) and callable(getattr(arm, "phase", None))) or arm.filler is None:
        ap.error(f"{a.arm} needs fused(phase, B, L), phase(phase, B, L) and a filler")
    pts = S.points()
    by = drivers(arm, pts)
    out = a.out / arm.name
    out.mkdir(parents=True, exist_ok=True)
    doc = {**SC.arm_doc(arm), "spec": a.arm,
           "fused": {f"{ph} {b}/{l}": (dataclasses.asdict(q) if d == "fscore" else None) for (ph, b, l), (d, q) in by.items()},
           "phase": {f"{ph} {b}/{l}": (dataclasses.asdict(q) if d == "pscore" else None) for (ph, b, l), (d, q) in by.items()},
           "fillers": {f"{ph} {b}/{l}": dataclasses.asdict(FS.filler_of(arm, ph, b, l)) for ph, b, l in pts
                       if A.placed(arm, ph, b, l).place != "none"}}
    (out / "arm.json").write_text(json.dumps(doc, indent=1))
    (out / "coins.txt").write_text(secrets.token_hex(16))
    (out / "commit.txt").write_text(subprocess.run(["git", "-C", str(SC.REPO), "rev-parse", "HEAD"], capture_output=True,
                                                   text=True, check=False).stdout.strip())
    if os.environ.get("RESEARCH_RUN_ID"):
        (out / "run.txt").write_text(os.environ["RESEARCH_RUN_ID"])
    bdir = Path(tempfile.mkdtemp(prefix="pfill-fp-score."))
    try:
        SC.build(bdir, out)
        log = out / "build.log"
        for name in ("fscore", "pscore"):
            rc = SC.sh([SC.NVCC, *SC.NVFLAGS, "-o", str(bdir / name), str(HERE / f"{name}.cu"), *SC.LIBS], out=log, err=log)
            if rc:
                raise SystemExit(f"build of {name} failed (rc {rc}); see {log}")
        sass = subprocess.run(["cuobjdump", "-sass", str(bdir / "pscore")], capture_output=True, text=True,
                              check=False).stdout
        keep, lines = False, []
        for ln in sass.splitlines():
            if "Function : " in ln:
                keep = "dfill_p" in ln
            if keep:
                lines.append(ln)
        (out / "sass-dfill_p.txt").write_text("\n".join(lines))
        inner = [sys.executable, str(HERE / "pfill_fp_score.py"), "measure", a.arm, "--bin", str(bdir), "--out", str(out),
                 "--rounds", str(a.rounds), "--reps", str(a.reps)] + (["--no-robust"] if a.no_robust else [])
        rc = subprocess.run(["gpu-lease", "1", "--wait", "--max-min", str(a.max_min), "--", *inner], check=False).returncode
        with open(out / "rc.txt", "a") as f:
            f.write(f"lease rc {rc}\n")
    finally:
        shutil.rmtree(bdir, ignore_errors=True)
    res = report(out, arm)
    run_dir = Path(os.environ["RESEARCH_RUN_DIR"]) if os.environ.get("RESEARCH_RUN_DIR") else out
    SC.record(res, doc, run_dir, a.by)
    show(res)
    return 0 if rc == 0 else rc


if __name__ == "__main__":
    sys.exit(main())
