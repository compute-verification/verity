"""The certified filler's scorer (SCORER, cmp-13), locked: changes go to its owner, compute-accounting, never into an arm's
PR. The spec, its desiderata, metrics and gates: note:20261016T0435Z-report-filler-spec. The arithmetic: `scoring.py`;
the arm contract: `arms.py`; the device driver: `dscore.cu`.

  python3 benchmarks/pouw/filler/score.py ARM [--rounds 8] [--reps 3] [--no-robust]   build, lease one GPU, measure, report
  python3 benchmarks/pouw/filler/score.py measure ARM --bin DIR --out DIR             the measurement, inside a lease
  python3 benchmarks/pouw/filler/score.py report OUT_DIR [--json]                     score a measured run's out dir
  python3 benchmarks/pouw/filler/score.py --leaderboard [--refresh] [--local DIR ...] every filler attempt, best first

A climber's run, from a clean pushed checkout:

  research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=<lane> \\
    -- python3 benchmarks/pouw/filler/score.py <arm|benchmarks/pouw/filler/my_arm.py:MyArm>

`score.py ARM` builds dcap.cu and dscore.cu (outside the lease), then under `gpu-lease 1 --wait` runs the gates before
anything is timed: FILL's receipt gate (`dcap dgate`), the scorer's own at the arm's chunk shape (`dscore sgate`: poisoned
and launched, every receipt equal to the host's; poisoned with nothing launched, none; stale and skipped work, none) and
CAP's serving gate (`dcap gate`); a failed gate times nothing (exit 5). Then ncu's DRAM bytes for the filler's launches
beside the counter's positive control (1 GiB streamed), and every suite point: ROUNDS rounds in ABBA order of K reps of
serving alone on the whole GPU and K reps under the arm's placement, every rep's SM clock and board power read (clocks are
never set). Each filler round (a point's calibration and each arm block) numbers its tiles from a public range the scorer
fixes from the point and the round, then draws the round's coins from the OS; the replay gate checks the ranges and that
no tile id, coins or sampled seed repeats across rounds. After the lease it scores the run (`report`), writes `score.json`
and one `kernel-attempt/v1` record (`ledger/<arm>/filler-suite.json`, declared in the run's `outputs.json` so
`research run --on` publishes it).

An arm that fails a gate isn't scored: its SCORE is null and `score_if_gated` keeps the number for diagnosis.
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
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "harness"))
import arms as A  # noqa: E402
import scoring as S  # noqa: E402

NVCC = os.environ.get("NVCC", "/usr/local/cuda-13.0/bin/nvcc")
os.environ["PATH"] = f"{Path(NVCC).parent}:{os.environ.get('PATH', '')}"  # cuobjdump and ncu beside nvcc, as dfill.sh
NVFLAGS = ["-O3", "-std=c++17", "-gencode", "arch=compute_120a,code=sm_120a", "-Xptxas", "-v"]
LIBS = ["-lcublasLt", "-lnvidia-ml", "-ldl"]
GATES = ("slowdown", "receipts", "replay", "honest", "dram", "clocks", "serving_exact")
SMI_FIELDS = ("clocks.sm,clocks.max.sm,power.draw,power.limit,clocks.applications.graphics,"
              "clocks.default_applications.graphics,clocks_event_reasons.active")
SHAPE = "filler-suite"


def sh(cmd, out=None, err=None, env=None) -> int:
    with open(out, "a") if out else open(os.devnull, "w") as fo, open(err, "a") if err else open(os.devnull, "w") as fe:
        return subprocess.run(cmd, stdout=fo, stderr=fe, env=env).returncode


def arm_doc(arm) -> dict:
    sys.path.insert(0, str(HERE.parent / "harness"))
    import ledger  # noqa: PLC0415

    fl = dataclasses.asdict(arm.filler) if arm.filler else None
    variant = ledger.variant_record(family=f"filler-{arm.name}", device="sm120", schemes=["dfill-receipt"],
                                    sources=[REPO / s for s in arm.sources], root=REPO, loader="dscore",
                                    config={k: str(v) for k, v in (fl or {}).items()})
    pts = {f"{ph} {b}/{l}": dataclasses.asdict(A.placed(arm, ph, b, l)) for ph, b, l in S.points()}
    return {"interface": A.VERSION, "name": arm.name, "lineage": arm.lineage, "filler": fl, "choice": arm.choice,
            "sources": list(arm.sources), "variant": variant, "placements": pts}


def build(bdir: Path, out: Path) -> dict:
    log = out / "build.log"
    rec = {"nvcc": subprocess.run([NVCC, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1:]}
    for name, src in (("dcap", HERE.parent / "decode_capacity" / "dcap.cu"), ("dscore", HERE / "dscore.cu")):
        rc = sh([NVCC, *NVFLAGS, "-o", str(bdir / name), str(src), *LIBS], out=log, err=log)
        if rc:
            raise SystemExit(f"build of {name} failed (rc {rc}); see {log}")
    text = log.read_text()
    rec["spills"] = [ln.strip() for ln in text.splitlines() if "spill" in ln and "0 bytes spill" not in ln]
    sass = subprocess.run(["cuobjdump", "-sass", str(bdir / "dscore")], capture_output=True, text=True).stdout
    keep, lines = False, []
    for ln in sass.splitlines():
        if "Function : " in ln:
            keep = "dfill_k" in ln
        if keep:
            lines.append(ln)
    (out / "sass-dfill.txt").write_text("\n".join(lines))
    rec["dfill_sass"] = {"imma": sum("IMMA" in ln for ln in lines),
                         "memory_ops": sum(any(op in ln for op in ("LDG", "STG", "LDS", "STS", "LDL", "STL")) for ln in lines)}
    (out / "build.json").write_text(json.dumps(rec, indent=1))
    return rec


def smi(out: Path) -> None:
    uuid = os.environ.get("GPU_LEASE_UUID")
    cmd = ["nvidia-smi", f"--query-gpu={SMI_FIELDS}", "--format=csv,noheader,nounits"] + (["-i", uuid] if uuid else [])
    sh(cmd, out=out / "smi.csv", err=out / "errors.txt")


def measure(arm, bdir: Path, out: Path, rounds: int, reps: int, robust: bool) -> int:
    env = dict(os.environ, DFILL_COINS=(out / "coins.txt").read_text().strip())
    fl = arm.filler
    if fl:
        env["DFILL_CHUNK_LOG2"] = str(fl.chunk_log2)
    err, rc_log = out / "errors.txt", out / "rc.txt"

    def note(s):
        with open(rc_log, "a") as f:
            f.write(s + "\n")

    smi(out)
    sh(["nvidia-smi", "-q", "-d", "CLOCK,PERFORMANCE,POWER"] + (["-i", env["GPU_LEASE_UUID"]] if env.get("GPU_LEASE_UUID")
                                                                else []), out=out / "smi-q.txt", err=err)
    gates = [[str(bdir / "dcap"), "gate"], ]
    if fl:
        gates = [[str(bdir / "dcap"), "dgate"], [str(bdir / "dscore"), "sgate", str(fl.W), str(fl.NC), str(fl.S)]] + gates
    for cmd in gates:
        name = cmd[1]
        rc = sh(cmd, out=out / f"{name}.jsonl", err=err, env=env)
        note(f"{name} rc {rc}")
        if rc:
            return 5
    if fl:
        m = "gpu__time_duration.sum,dram__bytes.sum"
        rc = sh(["ncu", "--clock-control", "none", "-k", "regex:dfill_k", "--launch-count", "3", "--metrics", m, "--csv",
                 str(bdir / "dcap"), "done", str(fl.W), str(fl.NC), str(fl.S)], out=out / "ncu-filler.csv", err=err, env=env)
        note(f"ncu filler rc {rc}")
        rc = sh(["ncu", "--clock-control", "none", "-k", "regex:bw_read", "--launch-count", "3", "--metrics", m, "--csv",
                 str(bdir / "dcap"), "bw", "1", "3"], out=out / "ncu-control.csv", err=err,
                env=dict(env, DCAP_NOGRAPH="1"))
        note(f"ncu control rc {rc}")
    smi(out)
    for i, (ph, b, l) in enumerate(S.points(robust)):
        p = A.placed(arm, ph, b, l)
        f = fl or A.Filler()
        cmd = [str(bdir / "dscore"), "dscore", "step" if ph == "decode" else "prefill", str(b), str(l), p.place,
               str(p.fill_sms), str(f.W), str(f.NC), str(f.S), str(rounds), str(reps), str(i)]
        rc = sh(cmd, out=out / "dscore.jsonl", err=err, env=env)
        note(f"dscore {ph} {b} {l} {p.place} {p.fill_sms} rc {rc}")
        smi(out)
    return 0


def rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip().startswith("{")] if path.is_file() else []


def report(out: Path, arm=None) -> dict:
    doc = json.loads((out / "arm.json").read_text())
    if arm is None:
        try:
            arm = A.load(doc.get("spec") or doc["name"])
        except (A.ArmError, OSError, AttributeError):
            arm = None
    stats = {}
    drows = rows(out / "dscore.jsonl")
    for r in drows:
        stats[("decode" if r["kind"] == "step" else "prefill", r["B"], r["L"])] = S.point_stats(r)
    sc = S.score(stats)
    has_filler = doc["filler"] is not None
    sg = rows(out / "sgate.jsonl")
    def read(p):
        return p.read_text().strip() if p.is_file() else None

    dram = S.dram_gate(read(out / "ncu-filler.csv") if has_filler else None, read(out / "ncu-control.csv"))
    clocks = S.clock_gate(stats, S.parse_smi(read(out / "smi.csv") or ""))
    honest = (S.honesty(arm, stats, S.points()) if arm is not None else
              {"ok": False, "why": f"arm {doc['name']} can't be loaded to check its placement"})
    g = S.gates(stats, sc, has_filler=has_filler, sgate=sg[0] if sg else None, dgate=rows(out / "dgate.jsonl"),
                dram=dram, clocks=clocks, honest=honest,
                replay=S.replay_gate(drows) if has_filler else {"ok": True, "why": "no filler"})
    serving_gate = rows(out / "gate.jsonl")
    g["serving_exact"] = {"ok": bool(serving_gate) and all(r.get("ok") is not False for r in serving_gate),
                          "rows": len(serving_gate)}
    g["ok"] = g["ok"] and g["serving_exact"]["ok"]
    res = {"scorer": S.SCORER, "arm": doc["name"], "interface": doc["interface"], "lineage": doc["lineage"],
           "variant": doc["variant"]["id"], "choice": doc["choice"], "commit": read(out / "commit.txt"),
           "coins": read(out / "coins.txt"), "run": os.environ.get("RESEARCH_RUN_ID") or read(out / "run.txt"),
           "scored": g["ok"] and "SCORE" in sc, "SCORE": sc.get("SCORE") if g["ok"] else None,
           "score_if_gated": sc.get("SCORE"), "min": sc.get("min"), "min_mix": sc.get("min_mix"),
           "score_nominal": sc.get("score_nominal"), "score_at_target": sc.get("score_at_target"),
           "flags": sc["flags"], "certified": sc["certified"], "gates": g, "missing": sc["missing"],
           "entries": sc["entries"], "secondary": S.secondary(stats, sc),
           "points": {f"{k[0]} {k[1]}/{k[2]}": v for k, v in stats.items()}}
    (out / "score.json").write_text(json.dumps(res, indent=1))
    return res


def record(res: dict, doc: dict, run_dir: Path, by: str | None) -> dict:
    import ledger  # noqa: PLC0415

    lin = doc["lineage"]
    g = res["gates"]
    meta = {
        "schema": ledger.KIND, "time": ledger.now(), "run": res["run"], "run_id": res["run"], "harness": res["commit"],
        "tier": "2a", "by": by, "arm": res["arm"], "shape": SHAPE, "phase": "serving-suite", "scorer": S.SCORER,
        "variant": res["variant"], "variant_record": doc["variant"], "parent": lin.get("parent"),
        "approach": lin.get("approach"), "mechanism": sorted(set(lin.get("mechanism") or [])),
        "hypothesis": ledger.one_line(lin.get("hypothesis"), "the hypothesis"), "choice": res["choice"],
        "gates": {"version": S.SCORER, "arm_interface": doc["interface"]},
        "gate": {k: bool(v.get("ok")) for k, v in g.items() if isinstance(v, dict)} | {"ok": g["ok"]},
        "metrics": {"SCORE": res["SCORE"], "score_if_gated": res["score_if_gated"], "min": res["min"],
                    "min_mix": res["min_mix"], "score_nominal": res["score_nominal"],
                    "score_at_target": res["score_at_target"], "slowdown_worst": g["slowdown"]["worst"],
                    "slowdown_at": g["slowdown"]["at"], "dram_share_of_hbm": g["dram"].get("share_of_hbm"),
                    "entries": {e["entry"]: {"coverage": e["coverage"], "time": e["time"],
                                             "filler_share": e["filler_share"]} for e in res["entries"]},
                    "work_cv_max": res["secondary"]["work_cv_max"]},
        "flags": res["flags"], "certified": res["certified"], "coins": res["coins"],
    }
    rec = {"kind": ledger.KIND, "meta": ledger._clean(meta), "refs": {"parent": lin["parent"]}
           if str(lin.get("parent") or "").startswith("art:") else {}, "labels": [(ledger.KERNEL_OUTCOME, "screened")]}
    ledger.write(run_dir, [rec])
    if (run_dir / "job.json").is_file() or os.environ.get("RESEARCH_RUN_DIR"):
        ledger.write_outputs(run_dir, [rec])
    return rec


def fmt(x, p=3):
    return "–" if x is None else f"{x:.{p}f}" if isinstance(x, float) else str(x)


def show(res: dict) -> None:
    g = res["gates"]
    print(f"\n{res['arm']} ({res['variant']}), {S.SCORER}: SCORE {fmt(res['SCORE'])}"
          + ("" if res["scored"] else f" (not scored; {fmt(res['score_if_gated'])} if gated)")
          + f", min {fmt((res['min'] or {}).get('coverage'))} at {(res['min'] or {}).get('entry')}"
          + f" (mixes: {fmt((res['min_mix'] or {}).get('coverage'))} at {(res['min_mix'] or {}).get('entry')})"
          + f"; nominal ÷ capacity {fmt(res['score_nominal'])}, at 0.99 for all {fmt(res['score_at_target'])}")
    for f in res["flags"]:
        print(f"  FLAG: {f}")
    for name in GATES:
        v = g.get(name, {})
        extra = {"slowdown": lambda: f"worst {fmt(v.get('worst'))} at {v.get('at')} ({v.get('phase')})",
                 "replay": lambda: v.get("why") or f"{v.get('rounds')} rounds, {v.get('tiles'):.3g} tiles, ranges the "
                                   f"scorer's and disjoint, coins and {v.get('seeds_sampled')} sampled seeds distinct",
                 "receipts": lambda: v.get("why") or f"timed {v.get('timed_checked')} checked, {v.get('timed_bad')} bad",
                 "honest": lambda: v.get("why") or "; ".join(f"{k}: {', '.join(v[k])}" for k in
                                                             ("unstable_placement", "too_good", "over_peak") if v.get(k)),
                 "dram": lambda: v.get("why") if not v.get("share_of_hbm") else
                 f"{v['share_of_hbm']:.3%} of HBM, counter control {fmt(v.get('control_bytes_ratio'))}",
                 "clocks": lambda: f"{v.get('reps')} reps read; nvidia-smi between points: MHz {v.get('mhz')}, W "
                                   f"{v.get('watt')}",
                 }.get(name, lambda: "")()
        print(f"  gate {name}: {'pass' if v.get('ok') else 'FAIL'}  {extra}")
    print("\n| entry | coverage | filler share | time | slowdowns | τ (VF) |\n|---|---|---|---|---|---|")
    for e in res["entries"]:
        print(f"| {e['entry']} | {fmt(e['coverage'])} | {fmt(e['filler_share'])} | {fmt(e['time'])} | "
              + ", ".join(f"{k} {fmt(s)}" for k, s in e["slowdowns"].items()) + f" | {fmt(e['tau_vf'])} |")
    sec = res["secondary"]
    pw = [v for v in sec["power"].values() if all(S.isfinite(x) for x in v)]
    ck = [v for v in sec["clock"].values() if all(S.isfinite(x) for x in v)]
    setup = sorted(sec["placement_setup_ms"].values())
    print(f"\nsecondary: work CV max {fmt(sec['work_cv_max'])}; board W alone/arm, max "
          f"{fmt(max((a for a, _ in pw), default=None), 0)}/{fmt(max((b for _, b in pw), default=None), 0)}; SM MHz "
          f"alone/arm, min {fmt(min((a for a, _ in ck), default=None), 0)}/{fmt(min((b for _, b in ck), default=None), 0)}"
          f"; placement setup ms median {fmt(setup[len(setup) // 2] if setup else None, 0)}; choice: {res['choice']}")
    print("robust (not scored): " + ("; ".join(f"B {r['B']} L {r['L']}: {fmt(r['slowdown'])}x, coverage "
                                               f"{fmt(r['coverage'])}" for r in sec["robust"]) or "–"))
    if res["missing"]:
        print(f"missing entries: {res['missing']}")


def leaderboard(local: list[Path], refresh: bool, as_json: bool) -> int:
    import ledger  # noqa: PLC0415

    entries = []
    try:
        if refresh:
            from research.store import cli as SC  # noqa: PLC0415
            SC.main(["refresh"])
        entries = [(art, m) for art, m, _ in ledger.attempts() if m.get("scorer", "").startswith("filler-scorer/")]
    except Exception as ex:  # noqa: BLE001  the store is optional here; local run dirs still list
        print(f"leaderboard: the evidence store is not readable ({type(ex).__name__}: {ex}); local runs only",
              file=sys.stderr)
    for d in local:
        for p in sorted(Path(d).glob("**/ledger/*/filler-suite.json")):
            m = json.loads(p.read_text())["meta"]
            entries.append((str(p), m))
    seen, uniq = set(), []
    for art, m in entries:
        key = (m.get("run"), m.get("arm"))
        if key not in seen:
            seen.add(key)
            uniq.append((art, m))

    def gated(m):  # a record from a scorer without one of today's gates (replay came at 6d10838ff) isn't gated
        gt = m.get("gate") or {}
        return bool(gt.get("ok")) and all(k in gt for k in GATES)

    uniq.sort(key=lambda x: (not gated(x[1]), -((x[1]["metrics"].get("SCORE") or
                                                  x[1]["metrics"].get("score_if_gated") or 0))))
    if as_json:
        print(json.dumps([{"record": a, "gated": gated(m), **m} for a, m in uniq]))
        return 0
    print("| # | arm | variant | SCORE | min (entry) | worst slowdown | gates | run | scorer | parent | mechanism | flags |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, (art, m) in enumerate(uniq, 1):
        mt, gt = m["metrics"], m.get("gate") or {}
        sc = fmt(mt.get("SCORE")) if gated(m) else f"({fmt(mt.get('score_if_gated'))})"
        failed = [k for k, v in gt.items() if k != "ok" and not v] + [f"no {k}" for k in GATES if k not in gt]
        print(f"| {i} | {m['arm']} | {m.get('variant')} | {sc} | {fmt((mt.get('min') or {}).get('coverage'))} "
              f"({(mt.get('min') or {}).get('entry')}) | {fmt(mt.get('slowdown_worst'))} | "
              f"{'pass' if gated(m) else 'FAIL ' + ','.join(failed)} | {m.get('run')} | {(m.get('harness') or '')[:9]} | "
              f"{m.get('parent')} | {','.join(m.get('mechanism') or [])} | {'provisional' if m.get('flags') else ''} |")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--leaderboard":
        ap = argparse.ArgumentParser(prog="score.py --leaderboard")
        ap.add_argument("--leaderboard", action="store_true")
        ap.add_argument("--local", type=Path, action="append", default=[], help="run dirs not yet in the store")
        ap.add_argument("--refresh", action="store_true", help="refresh the store's index from its remote first")
        ap.add_argument("--json", action="store_true")
        a = ap.parse_args(argv)
        return leaderboard(a.local, a.refresh, a.json)
    if argv and argv[0] == "report":
        ap = argparse.ArgumentParser(prog="score.py report")
        ap.add_argument("out", type=Path)
        ap.add_argument("--json", action="store_true")
        a = ap.parse_args(argv[1:])
        res = report(a.out)
        print(json.dumps(res, indent=1)) if a.json else show(res)
        return 0
    if argv and argv[0] == "measure":
        ap = argparse.ArgumentParser(prog="score.py measure")
        ap.add_argument("arm")
        ap.add_argument("--bin", type=Path, required=True)
        ap.add_argument("--out", type=Path, required=True)
        ap.add_argument("--rounds", type=int, default=8)
        ap.add_argument("--reps", type=int, default=3)
        ap.add_argument("--no-robust", action="store_true")
        a = ap.parse_args(argv[1:])
        return measure(A.load(a.arm), a.bin, a.out, a.rounds, a.reps, not a.no_robust)
    ap = argparse.ArgumentParser(prog="score.py", description="score a filler arm: build, lease one GPU, measure, report",
                                 epilog="\n\n".join(__doc__.split("\n\n")[1:4]),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm", help=f"one of {sorted(A.ARMS)}, or benchmarks/pouw/filler/FILE.py:Class")
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
    out = a.out / arm.name
    out.mkdir(parents=True, exist_ok=True)
    doc = {**arm_doc(arm), "spec": a.arm}
    (out / "arm.json").write_text(json.dumps(doc, indent=1))
    (out / "coins.txt").write_text(secrets.token_hex(16))
    (out / "commit.txt").write_text(subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                                                   text=True).stdout.strip())
    if os.environ.get("RESEARCH_RUN_ID"):
        (out / "run.txt").write_text(os.environ["RESEARCH_RUN_ID"])
    bdir = Path(tempfile.mkdtemp(prefix="filler-score."))
    try:
        build(bdir, out)
        inner = [sys.executable, str(HERE / "score.py"), "measure", a.arm, "--bin", str(bdir), "--out", str(out),
                 "--rounds", str(a.rounds), "--reps", str(a.reps)] + (["--no-robust"] if a.no_robust else [])
        rc = subprocess.run(["gpu-lease", "1", "--wait", "--max-min", str(a.max_min), "--", *inner]).returncode
        with open(out / "rc.txt", "a") as f:
            f.write(f"lease rc {rc}\n")
    finally:
        shutil.rmtree(bdir, ignore_errors=True)
    res = report(out, arm)
    run_dir = Path(os.environ["RESEARCH_RUN_DIR"]) if os.environ.get("RESEARCH_RUN_DIR") else out
    record(res, doc, run_dir, a.by)
    show(res)
    return 0 if rc == 0 else rc


if __name__ == "__main__":
    sys.exit(main())
