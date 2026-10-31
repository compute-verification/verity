"""FILL's arm-side runner: the locked scorer's run (`score.py`) for an arm whose decode
step carries filler warps in its own kernels (`fused(phase, B, L)`, fused_arm.py), which filler-arm/v1 can't place, and
which may name a chunk shape per point (`filler_at(phase, B, L)`, as psched_score.py). It changes nothing in the scorer;
it calls score.py's own functions wherever they apply and adds only what the fused step needs.

  python3 benchmarks/pouw/filler/fused_score.py ARM [--rounds 8] [--reps 3] [--no-robust]   build, lease, measure, report
  python3 benchmarks/pouw/filler/fused_score.py measure ARM --bin DIR --out DIR             the measurement, in a lease
  python3 benchmarks/pouw/filler/fused_score.py report OUT_DIR                              score a measured out dir

The run is score.py's, step for step: the arm's document, coins and commit; the build (score.build) and fscore.cu's; under
`gpu-lease 1 --wait` the gates before anything is timed: FILL's `dcap dgate`, the scorer's `dscore sgate` at every chunk
shape the arm pumps (rows in sgate.jsonl, the arm's `filler`'s first, which the locked gate reads), CAP's `dcap gate`
and FILL's `dcap fgate` (flin bit for bit on every row of every linear at every tile height, alone and with filler
warps, and the attention kernels with filler warps against the host, every receipt against dfill_ref with the steps − 8
control), both into gate.jsonl, which the locked serving_exact gate reads; ncu's DRAM bytes for dfill_k and the
counter's control as score.py takes them, and for flin_k with and without its filler warps (ncu-fused.csv); then every
suite point, through `fscore fscore` where the arm runs a fused step and through the locked `dscore dscore` elsewhere,
and score.py's positive control. After the lease the locked `report` scores the out dir (its own output kept as
score-locked.json), and this adds the gates it can't see: `fused_receipts` (every fused row's receipts, device step
counts, rounds and output rows), `fused_dram` (the filler warps' global traffic, 44 bytes a slot a step at most, and
flin_k's measured DRAM bytes with its filler warps over without, each against the same 1% of the HBM bandwidth the run's
control measures), `fused_exact` (fgate's rows in gate.jsonl, every one passing) and `sgates`. Then score.py's `record`
writes the kernel-attempt/v1 record and `show` prints the result, with a second accounting beside it (`harness`): the
fused filler credited only over the measured step, the pump over the modelled non-GEMM time too (FILL's round-2
harness credited the pump alone there), which the locked arithmetic doesn't do (it credits each point's whole rate,
fixed work included, over the step and its non-GEMM time).
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
FUSED_SLOT_BYTES = 44  # a filler warp's global traffic a step: its receipt (8) and count (4), and the round's FState (32)
NCU_FLIN = ("32", "8", "4", "0")  # dcap flin M FW NC THR: 32 qkv launches with filler warps, then 32 without


def fused_of(arm, ph, b, l):
    q = arm.fused(ph, b, l)
    if q is not None and A.placed(arm, ph, b, l).place != "split":
        raise A.ArmError(f"{arm.name}: a fused step at {ph} {b}/{l} needs a split (fscore pumps FILL v1's beside it)")
    return q


def filler_of(arm, ph, b, l):
    """The chunk shape at a point: `filler_at(phase, B, L)` when the arm has one, else its `filler`; a fused point pumps
    the arm's `filler`."""
    at = getattr(arm, "filler_at", None)
    f = at(ph, b, l) if callable(at) else arm.filler
    if not isinstance(f, A.Filler):
        raise A.ArmError(f"{arm.name}: the filler at {ph} {b}/{l} is {f!r}, not a Filler")
    if fused_of(arm, ph, b, l) is not None and f != arm.filler:
        raise A.ArmError(f"{arm.name}: a fused point ({ph} {b}/{l}) pumps the arm's filler, not {f}")
    return f


def fused_args(q) -> list[str]:
    return [str(x) for x in (q.FW, q.FNC, q.AFW, q.ANC, q.APS, q.ALZ, q.LT, q.LM)]


def measure(arm, bdir: Path, out: Path, rounds: int, reps: int, robust: bool) -> int:
    env = dict(os.environ, DFILL_COINS=(out / "coins.txt").read_text().strip(), DFILL_CHUNK_LOG2=str(arm.filler.chunk_log2))
    fl, err, rc_log = arm.filler, out / "errors.txt", out / "rc.txt"

    def note(s):
        with open(rc_log, "a") as f:
            f.write(s + "\n")

    def at(f):
        return dict(env, DFILL_CHUNK_LOG2=str(f.chunk_log2))

    pts = S.points(robust)
    fillers = [fl] + sorted({filler_of(arm, ph, b, l) for ph, b, l in pts
                             if A.placed(arm, ph, b, l).place != "none"} - {fl}, key=dataclasses.astuple)
    SC.smi(out)
    SC.sh(["nvidia-smi", "-q", "-d", "CLOCK,PERFORMANCE,POWER"] + (["-i", env["GPU_LEASE_UUID"]]
                                                                  if env.get("GPU_LEASE_UUID") else []),
          out=out / "smi-q.txt", err=err)
    gates = [("dgate", [str(bdir / "dcap"), "dgate"], env)]
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
                str(bdir / "dcap"), "flin", *NCU_FLIN, "1"], out=out / "ncu-fused.csv", err=err, env=env)
    note(f"ncu flin {' '.join(NCU_FLIN)} rc {rc}")
    SC.smi(out)
    for i, (ph, b, l) in enumerate(pts):
        p, q = A.placed(arm, ph, b, l), fused_of(arm, ph, b, l)
        f = filler_of(arm, ph, b, l)
        if q is not None:
            cmd = [str(bdir / "fscore"), "fscore", str(b), str(l), str(p.fill_sms), str(f.W), str(f.NC), str(f.S),
                   *fused_args(q), str(rounds), str(reps), str(i)]
        else:
            cmd = [str(bdir / "dscore"), "dscore", "step" if ph == "decode" else "prefill", str(b), str(l), p.place,
                   str(p.fill_sms), str(f.W), str(f.NC), str(f.S), str(rounds), str(reps), str(i)]
        rc = SC.sh(cmd, out=out / "dscore.jsonl", err=err, env=dict(at(f), FLIN_STOP=str(q.ST)) if q else at(f))
        note(f"{cmd[1]} {ph} {b} {l} {p.place} {p.fill_sms} filler {dataclasses.astuple(f)} "
             f"fused {dataclasses.astuple(q) if q else None} rc {rc}")
        SC.smi(out)
    SC.control(arm, fl, bdir, out, env, rounds, reps, len(pts), note)
    return 0


def hbm(out: Path) -> tuple[float | None, str | None]:
    ctl = (out / "ncu-control.csv").read_text() if (out / "ncu-control.csv").is_file() else ""
    cb, ct = S.ncu_metric(ctl, "dram__bytes.sum", "bw_read"), S.ncu_metric(ctl, "gpu__time_duration.sum", "bw_read")
    if not (cb and ct):
        return None, "the counter's control has no rows"
    ratio = S._med(cb) / S.CONTROL_BYTES
    if not (S.CONTROL_RANGE[0] <= ratio <= S.CONTROL_RANGE[1]):
        return None, f"the counter's control read {ratio:.3f} of 1 GiB"
    return S._med(cb) / S._med(ct), None


def fused_dram(out: Path, frows: list[dict], stats: dict) -> dict:
    """The fused filler's DRAM traffic against S.DRAM_LIMIT of the HBM bandwidth the run's control measures, two ways:
    every slot's global traffic a step (FUSED_SLOT_BYTES, all of it assumed to reach DRAM) over the point's arm step, and
    flin_k's measured DRAM bytes a launch with its filler warps over without, over its time with them."""
    bw, why = hbm(out)
    if bw is None:
        return {"ok": False, "why": why}
    bound = {f"decode {r['B']}/{r['L']}": FUSED_SLOT_BYTES * r["fused"]["slots_a_step"]
             / (stats[("decode", r["B"], r["L"])]["arm_ms"] * 1e-3) / bw for r in frows}
    text = (out / "ncu-fused.csv").read_text() if (out / "ncu-fused.csv").is_file() else ""
    by, tm = S.ncu_metric(text, "dram__bytes.sum", "flin_k"), S.ncu_metric(text, "gpu__time_duration.sum", "flin_k")
    if len(by) < 64 or len(tm) < 64:
        return {"ok": False, "bound_share_of_hbm": bound, "why": f"ncu read {len(by)}/{len(tm)} of flin_k's 64 launches"}
    with_f, without = S._med(by[:32]), S._med(by[32:64])
    measured = max(0.0, with_f - without) / (S._med(tm[:32])) / bw
    worst = max(bound.values(), default=0.0)
    ok = worst <= S.DRAM_LIMIT and measured <= S.DRAM_LIMIT
    return {"ok": ok, "hbm_bytes_per_s": bw, "bound_share_of_hbm": bound, "bound_worst": worst,
            "flin": {"M FW NC THR": " ".join(NCU_FLIN), "bytes_with_filler": with_f, "bytes_without": without,
                     "share_of_hbm": measured},
            "why": None if ok else f"fused filler DRAM {max(worst, measured):.2%} of HBM > {S.DRAM_LIMIT:.0%}"}


def harness_accounting(rows: list[dict], stats: dict) -> dict:
    """The locked arithmetic with the fused filler's work credited over the measured step only (the pump keeps its rate
    over the modelled non-GEMM time): each fused point's fill rate becomes split + fixed × arm_ms ÷ (arm_ms + non-GEMM ×
    slowdown), the ratio the decode and mix entries both stretch it by."""
    alt = {k: dict(v) for k, v in stats.items()}
    for r in rows:
        if r.get("driver") != "fscore":
            continue
        k = ("decode", r["B"], r["L"])
        st = alt[k]
        split, fixed = S.fill_parts([b["arm"] for b in r["blocks"]], r["chunk_macs"])
        t1 = st["arm_ms"] + S.other_ms(r["B"]) * st["slowdown"]
        st["fill_rate"] = split + fixed * st["arm_ms"] / t1
        st["fixed_rate"], st["split_rate"] = fixed, split
    sc = S.score(alt)
    at = S.score(alt, {"serving": {**S.CERTIFIED["serving"], "fraction": 0.99}, "filler": S.CERTIFIED["filler"]})
    return {"SCORE": sc.get("SCORE"), "min": sc.get("min"), "min_mix": sc.get("min_mix"),
            "score_at_target": sc.get("score_at_target"),
            "entries": {e["entry"]: {"coverage": e["coverage"], "coverage_at_target": e["coverage_at_target"]}
                        for e in sc["entries"]},
            "entries_serving_099": {e["entry"]: e["coverage"] for e in at["entries"]}}


def rounded(r: dict) -> bool:
    """An fscore row's rounds are where the replay gate reads them: the fused step's warm-up and power rounds and every
    arm block's fused round with its fixed work, or, with no filler warps (no slots a step), none of them: the pump
    alone beside FILL's step, its rounds dscore's."""
    pump_only = r["fused"]["slots_a_step"] == 0
    fused_keys = ("fused_warm_round", "fused_power_round")
    if "round" not in r["alone_fill"] or any((k in r["alone_fill"]) == pump_only for k in fused_keys):
        return False
    return all("solo" in b and ("fused_round" in b["arm"]) != pump_only and bool(b["arm"].get("fixed_ms")) != pump_only
               for b in r["blocks"])


def report(out: Path, arm) -> dict:
    res = SC.report(out, arm)
    shutil.copy(out / "score.json", out / "score-locked.json")
    g = res["gates"]
    rows = SC.rows(out / "dscore.jsonl")
    stats = {("decode" if r["kind"] == "step" else "prefill", r["B"], r["L"]): S.point_stats(r) for r in rows}
    frows = [r for r in rows if r.get("driver") == "fscore"]
    want = sorted(f"decode {b}/{l}" for ph, b, l in S.points() if fused_of(arm, ph, b, l) is not None)
    got = sorted(f"decode {r['B']}/{r['L']}" for r in frows)
    unrounded = [f"decode {r['B']}/{r['L']}" for r in frows if not rounded(r)]
    g["fused_receipts"] = {
        "ok": bool(frows) and not unrounded and set(want) <= set(got)
        and all(r["rec_bad"] == 0 and r["fused"]["steps_bad"] == 0 and r["fused"]["out_mismatches"] == 0
                and r["fused"]["ctl_equal"] == 0 for r in frows),
        "points": len(frows), "missing": sorted(set(want) - set(got)), "unrounded": unrounded,
        "checked": sum(r["rec_checked"] for r in frows), "bad": sum(r["rec_bad"] for r in frows),
        "out_rows": sum(r["fused"]["out_rows"] * (len(r["blocks"]) + 1) for r in frows),
        "out_mismatches": sum(r["fused"]["out_mismatches"] for r in frows),
        "steps_bad": sum(r["fused"]["steps_bad"] for r in frows)}
    g["fused_dram"] = fused_dram(out, frows, stats)
    fg = [r for r in SC.rows(out / "gate.jsonl") if r.get("mode") == "fgate"]
    g["fused_exact"] = {"ok": bool(fg) and all(r.get("ok") is True for r in fg), "rows": len(fg),
                        "failed": [r.get("what") or f"M {r.get('M')} {r.get('linear')} FW {r.get('FW')}"
                                   for r in fg if r.get("ok") is not True]}
    shapes = {filler_of(arm, ph, b, l) for ph, b, l in S.points() if A.placed(arm, ph, b, l).place != "none"} | {arm.filler}
    sg = SC.rows(out / "sgate.jsonl")
    g["sgates"] = {"ok": bool(sg) and all(r.get("ok") is True for r in sg) and len(sg) >= len(shapes), "rows": len(sg),
                   "shapes": len(shapes)}
    g["ok"] = all(g[k]["ok"] for k in ("fused_receipts", "fused_dram", "fused_exact", "sgates")) and g["ok"]
    res["scored"] = g["ok"] and res["score_if_gated"] is not None
    res["SCORE"] = res["score_if_gated"] if res["scored"] else None
    if "score_under_ch_if_gated" in res:
        res["score_under_ch"] = res["score_under_ch_if_gated"] if res["scored"] else None
    res["driver"] = "fused_score.py (arm-side: fscore.cu for the fused points, the locked dscore elsewhere)"
    res["fused_points"] = {
        f"decode {r['B']}/{r['L']}": {
            **r["fused"], "fill_sms": r["fill_sms"], "step_sms": r["step_sms"],
            "split_mac_per_s": (sp := S.fill_parts([b["arm"] for b in r["blocks"]], r["chunk_macs"]))[0],
            "fixed_mac_per_s": sp[1]} for r in frows}
    res["harness"] = harness_accounting(rows, stats) if not res["missing"] else None
    if "score_under_ch_if_gated" in res and not res["missing"]:
        res["entries_under_ch"] = {e["entry"]: e["coverage"] for e in S.score(stats, S.certified(S.UNDER_CH))["entries"]}
    if "theorem-k" in S.FILLER_ENTRIES and not res["missing"] and arm.filler.noise in S.FILLER_ENTRIES["theorem-k"]["noise"]:
        # what the scorer would read with every filler MAC at Theorem K's entry, the fused filler warps' included (a
        # projection: v3 reads no theorem for a filler with an arm-side driver's fixed work)
        k = S.score(stats, S.certified("theorem-k"))
        res["projected_under_k"] = {"SCORE": k["SCORE"] if res["scored"] else None, "min": k.get("min"),
                                    "entries": {e["entry"]: e["coverage"] for e in k["entries"]}}
    (out / "score.json").write_text(json.dumps(res, indent=1))
    return res


def show(res: dict) -> None:
    SC.show(res)
    for k in ("fused_receipts", "fused_dram", "fused_exact", "sgates"):
        v = dict(res["gates"][k])
        v.pop("bound_share_of_hbm", None)
        print(f"  gate {k}: {'pass' if res['gates'][k]['ok'] else 'FAIL'}  {v}")
    pk = res.get("projected_under_k")
    if pk:
        print(f"\nprojected under Theorem K (every filler MAC at its entry, the fused filler warps' too): SCORE "
              f"{SC.fmt(pk['SCORE'])}, min {SC.fmt((pk['min'] or {}).get('coverage'))} at {(pk['min'] or {}).get('entry')}")
    h = res.get("harness")
    if h:
        print(f"\nharness accounting (every filler at 0.99, as SCORE under CH; fused filler over the measured step only, the "
              f"pump over non-GEMM too): SCORE "
              f"{SC.fmt(h['SCORE'])}, min {SC.fmt((h['min'] or {}).get('coverage'))} at {(h['min'] or {}).get('entry')}; "
              f"at 0.99 for all {SC.fmt(h['score_at_target'])}")
        ch = res.get("entries_under_ch")
        print("| entry | coverage (scorer) |" + (" scorer under CH |" if ch else "")
              + " coverage (harness) | serving 0.99 (harness) |\n|---|---|" + ("---|" if ch else "") + "---|---|")
        for e in res["entries"]:
            print(f"| {e['entry']} | {SC.fmt(e['coverage'])} |" + (f" {SC.fmt(ch[e['entry']])} |" if ch else "")
                  + f" {SC.fmt(h['entries'][e['entry']]['coverage'])} | {SC.fmt(h['entries_serving_099'][e['entry']])} |")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "measure":
        ap = argparse.ArgumentParser(prog="fused_score.py measure")
        ap.add_argument("arm")
        ap.add_argument("--bin", type=Path, required=True)
        ap.add_argument("--out", type=Path, required=True)
        ap.add_argument("--rounds", type=int, default=8)
        ap.add_argument("--reps", type=int, default=3)
        ap.add_argument("--no-robust", action="store_true")
        a = ap.parse_args(argv[1:])
        return measure(A.load(a.arm), a.bin, a.out, a.rounds, a.reps, not a.no_robust)
    if argv and argv[0] == "report":
        ap = argparse.ArgumentParser(prog="fused_score.py report")
        ap.add_argument("out", type=Path)
        a = ap.parse_args(argv[1:])
        res = report(a.out, A.load(json.loads((a.out / "arm.json").read_text())["spec"]))
        show(res)
        return 0
    ap = argparse.ArgumentParser(prog="fused_score.py", description="score a fused-filler arm with the locked scorer")
    ap.add_argument("arm", help="benchmarks/pouw/filler/FILE.py:Class, an arm with fused(phase, B, L)")
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
    if not callable(getattr(arm, "fused", None)) or arm.filler is None:
        ap.error(f"{a.arm} has no fused(phase, B, L) or no filler: score it with score.py")
    out = a.out / arm.name
    out.mkdir(parents=True, exist_ok=True)
    doc = {**SC.arm_doc(arm), "spec": a.arm,
           "fused": {f"{ph} {b}/{l}": (dataclasses.asdict(q) if (q := fused_of(arm, ph, b, l)) else None)
                     for ph, b, l in S.points()},
           "fillers": {f"{ph} {b}/{l}": dataclasses.asdict(filler_of(arm, ph, b, l)) for ph, b, l in S.points()
                       if A.placed(arm, ph, b, l).place != "none"}}
    (out / "arm.json").write_text(json.dumps(doc, indent=1))
    (out / "coins.txt").write_text(secrets.token_hex(16))
    (out / "commit.txt").write_text(subprocess.run(["git", "-C", str(SC.REPO), "rev-parse", "HEAD"], capture_output=True,
                                                   text=True, check=False).stdout.strip())
    if os.environ.get("RESEARCH_RUN_ID"):
        (out / "run.txt").write_text(os.environ["RESEARCH_RUN_ID"])
    bdir = Path(tempfile.mkdtemp(prefix="fused-score."))
    try:
        SC.build(bdir, out)
        log = out / "build.log"
        rc = SC.sh([SC.NVCC, *SC.NVFLAGS, "-o", str(bdir / "fscore"), str(HERE / "fscore.cu"), *SC.LIBS], out=log, err=log)
        if rc:
            raise SystemExit(f"build of fscore failed (rc {rc}); see {log}")
        inner = [sys.executable, str(HERE / "fused_score.py"), "measure", a.arm, "--bin", str(bdir), "--out", str(out),
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
