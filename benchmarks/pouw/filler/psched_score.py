"""The scheduled arm's runner: the locked scorer's run (`score.py`) for an arm that adds
a phase filler (`phase(phase, B, L)`, psched_arm.py), which filler-arm/v1 can't place, and may name a chunk shape per
point (`filler_at(phase, B, L)`, a `Filler`; the arm's `filler` where it has none, and always at a phase point), which v1
runs one of. It changes nothing in the scorer; it calls score.py's own functions wherever they apply and adds only what
these need: every shape the arm runs gets its own `dscore sgate` row and ncu DRAM rows (in sgate.jsonl after the arm's
`filler`'s, which the locked gate reads, and in ncu-filler.csv, whose every launch the locked DRAM gate bounds), and the
`sgates` gate requires a passing row for each.

  python3 benchmarks/pouw/filler/psched_score.py ARM [--rounds 8] [--reps 3] [--no-robust]   build, lease, measure, report
  python3 benchmarks/pouw/filler/psched_score.py measure ARM --bin DIR --out DIR             the measurement, in a lease

The run is score.py's, step for step: the arm's document, coins and commit; the build (score.build) and pscore.cu's; under
`gpu-lease 1 --wait` the gates before anything is timed, FILL's `dcap dgate`, the phase filler's `dcap pgate` (every
receipt of dfill_p at each phase S, its rows appended to dgate.jsonl, which the locked receipts gate reads), the scorer's
`dscore sgate` at the split's chunk shape and CAP's `dcap gate`; ncu's DRAM bytes for dfill_k and the counter's control as
score.py takes them, and for dfill_p (`pscore pncu`, ncu-phase.csv); then every suite point, through `pscore pscore` where
the arm places a phase filler and through the locked `dscore dscore` elsewhere. After the lease the locked `report`
scores the out dir (its own output kept as score-locked.json), and this adds the one gate it can't see, `phase_dram`:
dfill_p's DRAM bytes a second against the same 1% of the HBM bandwidth the same control measures. A failure there voids
the SCORE as any gate does. Then score.py's `record` writes the kernel-attempt/v1 record and `show` prints the result.
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
import score as SC

A, S = SC.A, SC.S
PNCU_TILES = 32


def phase_of(arm, ph, b, l):
    q = arm.phase(ph, b, l)
    if q is not None and A.placed(arm, ph, b, l).place != "split":
        raise A.ArmError(f"{arm.name}: a phase filler at {ph} {b}/{l} needs a split (pscore runs FILL v1's beside it)")
    return q


def filler_of(arm, ph, b, l):
    """The chunk shape at a point: `filler_at(phase, B, L)` when the arm has one, else its `filler`. A phase point keeps the
    arm's `filler` (pscore pumps it on the split)."""
    at = getattr(arm, "filler_at", None)
    f = at(ph, b, l) if callable(at) else arm.filler
    if not isinstance(f, A.Filler):
        raise A.ArmError(f"{arm.name}: the filler at {ph} {b}/{l} is {f!r}, not a Filler")
    if phase_of(arm, ph, b, l) is not None and f != arm.filler:
        raise A.ArmError(f"{arm.name}: a phase point ({ph} {b}/{l}) runs the arm's filler, not {f}")
    return f


def measure(arm, bdir: Path, out: Path, rounds: int, reps: int, robust: bool) -> int:
    env = dict(os.environ, DFILL_COINS=(out / "coins.txt").read_text().strip(), DFILL_CHUNK_LOG2=str(arm.filler.chunk_log2))
    fl, err, rc_log = arm.filler, out / "errors.txt", out / "rc.txt"

    def note(s):
        with open(rc_log, "a") as f:
            f.write(s + "\n")

    def at(f):
        return dict(env, DFILL_CHUNK_LOG2=str(f.chunk_log2))

    pts = S.points(robust)
    shapes = sorted({(q.W, q.NC, q.S) for ph, b, l in pts if (q := phase_of(arm, ph, b, l))})
    fillers = [fl] + sorted({filler_of(arm, ph, b, l) for ph, b, l in pts
                             if A.placed(arm, ph, b, l).place != "none"} - {fl}, key=dataclasses.astuple)
    SC.smi(out)
    SC.sh(["nvidia-smi", "-q", "-d", "CLOCK,PERFORMANCE,POWER"] + (["-i", env["GPU_LEASE_UUID"]]
                                                                  if env.get("GPU_LEASE_UUID") else []),
          out=out / "smi-q.txt", err=err)
    gates = [("dgate", [str(bdir / "dcap"), "dgate"], env),
             ("dgate", [str(bdir / "dcap"), "pgate", *sorted({str(s) for _, _, s in shapes})], env)]
    gates += [("sgate", [str(bdir / "dscore"), "sgate", str(f.W), str(f.NC), str(f.S)], at(f)) for f in fillers]
    gates += [("gate", [str(bdir / "dcap"), "gate"], env)]
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
    for w, nc, s in shapes:
        rc = SC.sh(["ncu", "--clock-control", "none", "-k", "regex:dfill_p", "--launch-count", "3", "--metrics", m, "--csv",
                    str(bdir / "pscore"), "pncu", str(w), str(nc), str(s), str(PNCU_TILES)], out=out / "ncu-phase.csv",
                   err=err, env=env)
        note(f"ncu phase W {w} NC {nc} S {s} rc {rc}")
    SC.smi(out)
    for i, (ph, b, l) in enumerate(pts):
        p, q = A.placed(arm, ph, b, l), phase_of(arm, ph, b, l)
        if q is not None:
            cmd = [str(bdir / "pscore"), "pscore", str(b), str(l), str(p.fill_sms), str(fl.W), str(fl.NC), str(fl.S),
                   str(q.W), str(q.NC), str(q.S), str(q.tiles), str(rounds), str(reps), str(i)]
        else:
            f = filler_of(arm, ph, b, l)
            cmd = [str(bdir / "dscore"), "dscore", "step" if ph == "decode" else "prefill", str(b), str(l), p.place,
                   str(p.fill_sms), str(f.W), str(f.NC), str(f.S), str(rounds), str(reps), str(i)]
        f = filler_of(arm, ph, b, l)
        rc = SC.sh(cmd, out=out / "dscore.jsonl", err=err, env=at(f))
        note(f"{cmd[1]} {ph} {b} {l} {p.place} {p.fill_sms} filler {dataclasses.astuple(f)} "
             f"phase {dataclasses.astuple(q) if q else None} rc {rc}")
        SC.smi(out)
    SC.control(arm, fl, bdir, out, env, rounds, reps, len(pts), note)
    return 0


def phase_dram(out: Path) -> dict:
    """dfill_p's DRAM bytes a second (its own launches) against S.DRAM_LIMIT of the HBM bandwidth the run's control measures,
    the arithmetic of S.dram_gate for the phase kernel."""
    def read(p):
        return p.read_text() if p.is_file() else ""
    ctl, ph = read(out / "ncu-control.csv"), read(out / "ncu-phase.csv")
    cb, ct = S.ncu_metric(ctl, "dram__bytes.sum", "bw_read"), S.ncu_metric(ctl, "gpu__time_duration.sum", "bw_read")
    fb, ft = S.ncu_metric(ph, "dram__bytes.sum", "dfill_p"), S.ncu_metric(ph, "gpu__time_duration.sum", "dfill_p")
    if not (cb and ct and fb and ft):
        return {"ok": False, "why": f"an ncu metric is missing (control {len(cb)}/{len(ct)}, phase {len(fb)}/{len(ft)})"}
    ratio = S._med(cb) / S.CONTROL_BYTES
    if not (S.CONTROL_RANGE[0] <= ratio <= S.CONTROL_RANGE[1]):
        return {"ok": False, "control_bytes_ratio": ratio, "why": f"the counter's control read {ratio:.3f} of 1 GiB"}
    hbm = S._med(cb) / S._med(ct)
    rate = max(b / t for b, t in zip(fb, ft))
    ok = rate <= S.DRAM_LIMIT * hbm
    return {"ok": ok, "control_bytes_ratio": ratio, "hbm_bytes_per_s": hbm, "phase_bytes_per_s": rate,
            "phase_bytes_per_launch": max(fb), "share_of_hbm": rate / hbm, "launches": len(fb),
            "why": None if ok else f"phase filler DRAM {rate / hbm:.2%} of HBM > {S.DRAM_LIMIT:.0%}"}


def report(out: Path, arm) -> dict:
    res = SC.report(out, arm)
    shutil.copy(out / "score.json", out / "score-locked.json")
    g = res["gates"]
    g["phase_dram"] = phase_dram(out)
    rows = SC.rows(out / "dscore.jsonl")
    ph_rows = [r for r in rows if r.get("driver") == "pscore"]
    # the phase filler's own rounds (its calibration, every arm and standalone block), which the replay gate checks
    unrounded = [f"decode {r['B']}/{r['L']}" for r in ph_rows if "phase_round" not in r["alone_fill"]
                 or any("phase_round" not in b[k] for b in r["blocks"] for k in ("arm", "solo") if k in b)
                 or any("solo" not in b for b in r["blocks"])]
    g["phase_receipts"] = {"ok": bool(ph_rows) and not unrounded and all(r["rec_bad"] == 0 and
                                                                         r["phase"]["tiles_short"] == 0 for r in ph_rows),
                           "unrounded": unrounded,
                           "points": len(ph_rows), "checked": sum(r["rec_checked"] for r in ph_rows),
                           "bad": sum(r["rec_bad"] for r in ph_rows),
                           "tiles_short": sum(r["phase"]["tiles_short"] for r in ph_rows)}
    shapes = {filler_of(arm, ph, b, l) for ph, b, l in S.points() if A.placed(arm, ph, b, l).place != "none"} | {arm.filler}
    sg = SC.rows(out / "sgate.jsonl")
    ran = {(r.get("W"), r.get("NC"), r.get("S"), r.get("tiles")) for r in sg if r.get("ok") is True}
    g["sgates"] = {"ok": bool(sg) and all(r.get("ok") is True for r in sg) and len(sg) >= len(shapes), "rows": len(sg),
                   "shapes": len(shapes), "ran": sorted(ran, key=str)}
    g["ok"] = all(g[k]["ok"] for k in ("phase_dram", "phase_receipts", "sgates")) and g["ok"]
    res["scored"] = g["ok"] and res["score_if_gated"] is not None
    res["SCORE"] = res["score_if_gated"] if res["scored"] else None
    res["driver"] = "psched_score.py (arm-side: pscore.cu for the phase points, the locked dscore elsewhere)"
    res["phase_points"] = {f"decode {r['B']}/{r['L']}": {**r["phase"], "fill_sms": r["fill_sms"]} for r in ph_rows}
    (out / "score.json").write_text(json.dumps(res, indent=1))
    return res


def show(res: dict) -> None:
    SC.show(res)
    for k in ("phase_dram", "phase_receipts", "sgates"):
        print(f"  gate {k}: {'pass' if res['gates'][k]['ok'] else 'FAIL'}  {res['gates'][k]}")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "measure":
        ap = argparse.ArgumentParser(prog="psched_score.py measure")
        ap.add_argument("arm")
        ap.add_argument("--bin", type=Path, required=True)
        ap.add_argument("--out", type=Path, required=True)
        ap.add_argument("--rounds", type=int, default=8)
        ap.add_argument("--reps", type=int, default=3)
        ap.add_argument("--no-robust", action="store_true")
        a = ap.parse_args(argv[1:])
        return measure(A.load(a.arm), a.bin, a.out, a.rounds, a.reps, not a.no_robust)
    if argv and argv[0] == "report":
        ap = argparse.ArgumentParser(prog="psched_score.py report")
        ap.add_argument("out", type=Path)
        a = ap.parse_args(argv[1:])
        res = report(a.out, A.load(json.loads((a.out / "arm.json").read_text())["spec"]))
        show(res)
        return 0
    ap = argparse.ArgumentParser(prog="psched_score.py", description="score a phase-filler arm with the locked scorer")
    ap.add_argument("arm", help="benchmarks/pouw/filler/FILE.py:Class, an arm with phase(phase, B, L)")
    ap.add_argument("--out", type=Path, default=Path(os.environ["OUT"]) if os.environ.get("OUT") else None)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-robust", action="store_true")
    ap.add_argument("--max-min", type=int, default=25, help="the lease's bound in minutes")
    ap.add_argument("--by", default=os.environ.get("GPU_LEASE_WHO") or os.environ.get("RESEARCH_LANE"))
    a = ap.parse_args(argv)
    if a.out is None:
        ap.error("--out (or $OUT, as research run sets it)")
    arm = A.load(a.arm)
    if not callable(getattr(arm, "phase", None)) or arm.filler is None:
        ap.error(f"{a.arm} has no phase(phase, B, L) or no filler: score it with score.py")
    out = a.out / arm.name
    out.mkdir(parents=True, exist_ok=True)
    doc = {**SC.arm_doc(arm), "spec": a.arm,
           "phase": {f"{ph} {b}/{l}": (dataclasses.asdict(q) if (q := phase_of(arm, ph, b, l)) else None)
                     for ph, b, l in S.points()},
           "fillers": {f"{ph} {b}/{l}": dataclasses.asdict(filler_of(arm, ph, b, l)) for ph, b, l in S.points()
                       if A.placed(arm, ph, b, l).place != "none"}}
    (out / "arm.json").write_text(json.dumps(doc, indent=1))
    (out / "coins.txt").write_text(secrets.token_hex(16))
    (out / "commit.txt").write_text(subprocess.run(["git", "-C", str(SC.REPO), "rev-parse", "HEAD"], capture_output=True,
                                                   text=True, check=False).stdout.strip())
    if os.environ.get("RESEARCH_RUN_ID"):
        (out / "run.txt").write_text(os.environ["RESEARCH_RUN_ID"])
    bdir = Path(tempfile.mkdtemp(prefix="psched-score."))
    try:
        SC.build(bdir, out)
        log = out / "build.log"
        rc = SC.sh([SC.NVCC, *SC.NVFLAGS, "-o", str(bdir / "pscore"), str(HERE / "pscore.cu"), *SC.LIBS], out=log, err=log)
        if rc:
            raise SystemExit(f"build of pscore failed (rc {rc}); see {log}")
        sass = subprocess.run(["cuobjdump", "-sass", str(bdir / "pscore")], capture_output=True, text=True,
                              check=False).stdout
        keep, lines = False, []
        for ln in sass.splitlines():
            if "Function : " in ln:
                keep = "dfill_p" in ln
            if keep:
                lines.append(ln)
        (out / "sass-dfill_p.txt").write_text("\n".join(lines))
        inner = [sys.executable, str(HERE / "psched_score.py"), "measure", a.arm, "--bin", str(bdir), "--out", str(out),
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
