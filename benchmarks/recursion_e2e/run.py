"""``verity bench recursion_e2e COMMAND``: the recursive session end to end on the CPU, timed stage by stage (README.md).

  run [--k K] [--n N] [--runs R] [--neg COIN ...] [--pcpus P] [--jobs J] [--baseline] [--keep-data]
      the inner statement (``N`` instances of ``GemmCoordinate_v2{K}``, `verity/ml/flock/pod/rec_reprice.py`) proved
      without --zk through the rec proxy (`verity_flock.rec_live`), V*'s statements over that session staged
      (`verity_flock.rec_vstage`) and proved with --zk, and each verified by the Lean verifier of record built at this
      commit (``flock-verify verify --zk --rec-inner-session --rec-inner-transcript``), once over the honest inner session
      and once per ``--neg`` coin over a copy of it with that coin of rep 0 flipped, which Lean must refuse. It runs
      ``85-rec-reprice.sh``'s steps build, inner, vstage, oprove and lean with ``GPU=0``, one research run's ``$OUT``
      for all of them; ``$OUT/recursion_e2e.json`` has each stage's wall time, peak memory and proof bytes. Exit 0 only
      when every step ran, every honest statement was accepted and every negative refused.
  summarize DIR
      ``DIR/recursion_e2e.json`` again from a run's ``out/`` (``DIR``), as ``run`` writes it.
  store RUN [--runs-dir DIR]
      a run's ``recursion_e2e.json`` (``research fetch RUN --all``) as one evidence/v1 record naming the run and its
      commit; prints its art: id. From a VM with the store's keys: a node's shell has none.

The recorded form, from a clean checkout of the commit to measure (the README has the rest):

    research run --on vy-nebius-cpu-5 --project verity --source . --cwd source --mem-gb 60 -- \\
      python3 benchmarks/recursion_e2e/run.py run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
POD = ROOT / "verity/ml/flock/pod/85-rec-reprice.sh"
SCHEMA = "verity/bench/recursion-e2e/v1"
RESULT = "recursion_e2e.json"
STEPS = ("build", "inner", "vstage", "oprove", "lean")
DEFAULT_WORK = "/workspace/jobs/rec-reprice"


def text(p: Path) -> str:
    return p.read_text(errors="replace") if p.exists() else ""


def jload(p: Path):
    t = text(p).strip()
    return json.loads(t) if t else None


def seconds(clock: str | None) -> float | None:
    """``/usr/bin/time -v``'s ``Elapsed (wall clock) time`` (``h:mm:ss`` or ``m:ss.ss``) in seconds."""
    if not clock:
        return None
    s = 0.0
    for part in clock.split(":"):
        s = 60 * s + float(part)
    return round(s, 2)


def timed(p: Path) -> dict:
    """A ``/usr/bin/time -v`` record: wall seconds, peak resident GB (the largest process), exit status."""
    t = text(p)

    def v(key):
        m = re.search(key + r".*: (\S+)", t)
        return m.group(1) if m else None

    rss = v(r"Maximum resident set size \(kbytes\)")
    rc = v(r"Exit status")
    return {"wall_s": seconds(v(r"Elapsed \(wall clock\) time")), "peak_gb": round(int(rss) / 2**20, 2) if rss else None,
            "rc": int(rc) if rc is not None else None}


def sizes(p: Path) -> dict[str, int]:
    """``du -sb`` lines: path (its last two parts) -> bytes."""
    out = {}
    for line in text(p).splitlines():
        f = line.split(None, 1)
        if len(f) == 2 and f[0].isdigit():
            out["/".join(Path(f[1]).parts[-2:])] = int(f[0])
    return out


def staged_units(st: dict | None) -> dict:
    """What `rec_vstage.stage` printed of each unit it staged: its template, instances, rows, k_log and seconds."""
    if not isinstance(st, dict):
        return {}
    keep = ("template", "instances", "unit_rows", "unit_log", "k_log", "stage_s")
    units = {}
    for name, u in (st.get("levels") or {}).items():
        units[name] = {k: u.get(k) for k in keep}
    if isinstance(st.get("cap"), dict) and "template" in st["cap"]:
        units["cap"] = {k: st["cap"].get(k) for k in keep}
    for name, u in (st.get("caps") or {}).items():
        units[f"cap-{name}"] = {k: u.get(k) for k in keep}
    for u in st.get("algebra") or []:
        units[f"alg-p{u.get('part')}"] = {k: u.get(k) for k in keep}
    return units


def lean_lines(log: str) -> dict[str, dict]:
    """``lean_verdicts.py``'s line per part (``lean PART[ (a forgery…)]: exit RC, accepted A, why W[  <- FAILS]``)."""
    out = {}
    for line in log.splitlines():
        m = re.match(r"(?:\+ )?lean (\S+?)(?: \(a forgery: Lean must refuse it\))?: exit (\S+), accepted (.*?), why (.*)$", line)
        if m:
            out[m.group(1)] = {"exit": None if m.group(2) == "None" else int(m.group(2)), "accepted": m.group(3),
                               "passes": not m.group(4).endswith("<- FAILS")}
    return out


def summarize(out: Path) -> dict:
    """``out/recursion_e2e.json`` from the steps' records in ``out`` (``85-rec-reprice.sh``'s ``summary-<step>.json``, its
    per-command ``/usr/bin/time`` records and this runner's ``step-<step>.time``)."""
    inputs = jload(out / "inputs.json") or {}
    steps = {s: {**timed(out / f"step-{s}.time"), "ran": (out / f"step-{s}.time").exists()} for s in STEPS}
    build = jload(out / "summary-build.json") or {}
    inner_s = jload(out / "summary-inner.json") or {}
    vstage = jload(out / "summary-vstage.json") or {}
    oprove = jload(out / "summary-oprove.json") or {}
    lean = jload(out / "summary-lean.json") or {}

    stage = build.get("stage") or {}
    proxy = inner_s.get("proxy") or {}
    session = sizes(out / "inner/session/sizes.txt")
    inner = {"stage_wall_s": seconds(stage.get("wall")), "stage_peak_gb": stage.get("peak_gb"),
             "statement_bytes": sizes(out / "stage/sizes.txt"),
             "prove_s": proxy.get("prove_s"), "prove_total_s": proxy.get("prove_total_s"), "e2e_s": proxy.get("e2e_s"),
             "proof_bytes": proxy.get("proof_bytes"), "prover_peak_gb": proxy.get("prover_peak_gb"),
             "prover_cpu_s": proxy.get("prover_cpu_s"), "k_log": proxy.get("k_log"), "dense_m": proxy.get("dense_m"),
             "instances": proxy.get("instances"), "session_bytes": session,
             "replay": (inner_s.get("replay_proxy") or {}).get("verdict"),
             "replay_wall_s": seconds((inner_s.get("replay_proxy") or {}).get("wall")), "vstar_wall_s": seconds(inner_s.get("vstar_wall"))}
    if "zk" in inner_s:
        z = inner_s["zk"]
        inner["baseline_zk"] = {k: z.get(k) for k in ("prove_total_s", "prove_s", "proof_bytes", "prover_peak_gb", "k_log",
                                                      "dense_m", "serve_verify_s", "accepted")}

    vst = {}
    for name, s in (vstage.get("statements") or {}).items():
        vst[name] = {"wall_s": seconds(s.get("wall")), "peak_gb": s.get("peak_gb"), "units": staged_units(s.get("stage"))}
    vstar = {"points_s": seconds((vstage.get("points") or {}).get("wall")), "fold_s": seconds((vstage.get("fold") or {}).get("wall")),
             "statements": vst, "staged_bytes": sum(sizes(out / "vstage/sizes.txt").values()) or None}

    ost = {}
    for name, s in (oprove.get("levels") or {}).items():
        lv = s.get("live") or {}
        ost[name] = {k: lv.get(k) for k in ("prove_total_s", "prove_s", "e2e_s", "proof_bytes", "prover_peak_gb", "prover_cpu_s",
                                            "serve_verify_s", "accepted", "k_log", "dense_m", "instances")}
        ost[name]["replay"] = (s.get("replay") or {}).get("verdict") if isinstance(s.get("replay"), dict) else s.get("replay")
        ost[name]["replay_wall_s"] = seconds(s.get("replay_wall"))
        if "error" in lv:
            ost[name]["error"] = lv["error"][-600:]

    def total(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return round(sum(xs), 2) if xs else None

    outer = {"statements": ost, "prove_total_s": total(s["prove_total_s"] for s in ost.values()),
             "proof_bytes": total(s["proof_bytes"] for s in ost.values()),
             "prover_peak_gb": max((s["prover_peak_gb"] for s in ost.values() if s["prover_peak_gb"] is not None), default=None),
             "accepted": all(s["accepted"] and all(x is True for x in s["accepted"]) for s in ost.values()) if ost else None}

    judged = lean_lines(text(out / "step-lean.log"))
    honest, negative, other = {}, {}, {}
    for name, s in (lean.get("levels") or {}).items():
        v = s.get("verdict") if isinstance(s.get("verdict"), dict) else {"why": str(s.get("verdict"))[-300:]}
        row = {"accepted": v.get("accepted"), "why": (v.get("why") or "")[:300], "verify_s": v.get("verify_s"),
               "setup_s": v.get("setup_s"), "inputs_s": v.get("inputs_s"), "wall_s": seconds(s.get("wall")),
               "peak_gb": s.get("peak_gb"), "passes": (judged.get(name) or {}).get("passes")}
        (negative if name.startswith("forged-") else other if name == "inner-statement" else honest)[name] = row
    verify = {"statements": honest, "inner_statement": other.get("inner-statement"),
              "wall_s_sum": total(r["wall_s"] for r in honest.values()),
              "wall_s_max": max((r["wall_s"] for r in honest.values() if r["wall_s"] is not None), default=None),
              "peak_gb": max((r["peak_gb"] for r in honest.values() if r["peak_gb"] is not None), default=None),
              "accepted": all(r["accepted"] is True for r in honest.values()) if honest else None}
    neg = {"statements": negative, "coins": inputs.get("neg"),
           "refused": all(r["accepted"] is not True and r["passes"] is not False for r in negative.values()) if negative else None}

    ok = (all(steps[s]["rc"] == 0 for s in STEPS) and verify["accepted"] is True and outer["accepted"] is True
          and (neg["refused"] is True or not inputs.get("neg")))
    res = {"schema": SCHEMA, "ok": ok, "inputs": inputs,
           "binaries": {"flock_circuit": build.get("binary"), "flock_verify": build.get("lean")},
           "steps": steps, "inner": inner, "vstar": vstar, "outer": outer, "verify": verify, "negative": neg,
           "headline": {"inner_prove_s": inner["prove_total_s"], "inner_proof_bytes": inner["proof_bytes"],
                        "inner_prover_peak_gb": inner["prover_peak_gb"], "vstar_stage_s": steps["vstage"]["wall_s"],
                        "vstar_stage_peak_gb": steps["vstage"]["peak_gb"], "outer_prove_s": outer["prove_total_s"],
                        "outer_proof_bytes": outer["proof_bytes"], "outer_prover_peak_gb": outer["prover_peak_gb"],
                        "outer_statements": len(ost), "lean_verify_s_sum": verify["wall_s_sum"],
                        "lean_verify_s_max": verify["wall_s_max"], "lean_peak_gb": verify["peak_gb"],
                        "negative_refused": neg["refused"]}}
    (out / RESULT).write_text(json.dumps(res, indent=1))
    return res


def git_head() -> str | None:
    p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else os.environ.get("RESEARCH_SOURCE_SHA")


def put(out: Path, res: dict) -> None:
    """The result into the store from inside the job, where the job has the store's keys; else `store RUN` from a VM."""
    if not shutil.which("research"):
        (out / "art.txt").write_text("not stored here: no `research` on PATH; run `store RUN` from a VM\n")
        return
    meta = {"benchmark": "recursion_e2e", "schema": SCHEMA, "run": os.environ.get("RESEARCH_RUN_ID"),
            "commit": res["inputs"].get("commit"), "ok": res["ok"]}
    p = subprocess.run(["research", "data", "put", "--kind", "evidence/v1", "--file", str(out / RESULT), "--preserve",
                        "--meta", json.dumps(meta)], capture_output=True, text=True)
    lines = (p.stdout or "").strip().splitlines()
    (out / "art.txt").write_text((lines[-1] if p.returncode == 0 and lines else
                                  f"not stored here (exit {p.returncode}): {(p.stderr or '')[-400:]}; run `store RUN` from a VM") + "\n")


def run(a) -> int:
    run_dir = Path(os.environ.get("RESEARCH_RUN_DIR") or a.run_dir).resolve()
    out = run_dir / "out"
    out.mkdir(parents=True, exist_ok=True)
    if not shutil.which("time") and not Path("/usr/bin/time").exists():
        raise SystemExit("recursion_e2e: needs /usr/bin/time (apt-get install time)")
    tag = a.tag or "recbench-" + (os.environ.get("RESEARCH_RUN_ID") or f"local-{os.getpid()}")
    work = Path(os.environ.get("FLOCK_WORK") or DEFAULT_WORK)
    ncpu = len(os.sched_getaffinity(0))
    pcpus = a.pcpus or max(2, ncpu * 3 // 4)
    inputs = {"commit": git_head(), "k": a.k, "n": a.n, "warm": 0, "runs": a.runs, "neg": a.neg, "baseline": a.baseline,
              "pcpus": pcpus, "cpus": ncpu, "jobs": a.jobs, "tag": tag, "flock_work": str(work),
              "statement": f"{a.n} x GemmCoordinate_v2{{K={a.k},DOT=HopperBF16WgmmaDot16_v1}} under Q_template_instance",
              "inner_statement": "verity/flock-circuit+plain-leaves, M0 (no --zk), FC_COINS=os, through rec_live's proxy",
              "outer_statement": "verity/flock-circuit --zk on loopback, every unit drawn",
              "pod_script": str(POD.relative_to(ROOT)), "run": os.environ.get("RESEARCH_RUN_ID")}
    (out / "inputs.json").write_text(json.dumps(inputs, indent=1))
    env = {**os.environ, "RESEARCH_RUN_DIR": str(run_dir), "FLOCK_WORK": str(work), "GPU": "0", "K": str(a.k), "N": str(a.n),
           "TAG": tag, "WARM": "0", "RUNS": str(a.runs), "PCPUS": str(pcpus), "JOBS": str(a.jobs), "FORGE": "",
           "NEG_INNER": " ".join(a.neg), "INNER_LOOPBACK": "1" if a.baseline else "0", "REPLAY": "1", "CHECKS": "1"}
    rc = 0
    try:
        for step in STEPS:
            with open(out / f"step-{step}.log", "w") as log:
                rc = subprocess.run(["/usr/bin/time", "-v", "-o", str(out / f"step-{step}.time"), "bash", str(POD)],
                                    cwd=ROOT, env={**env, "STEP": step}, stdout=log, stderr=subprocess.STDOUT).returncode
            print(f"recursion_e2e: {step} exit {rc}", flush=True)
            if rc != 0:
                break
    finally:
        res = summarize(out)
        if not a.keep_data:
            shutil.rmtree(work / "data" / tag, ignore_errors=True)
    put(out, res)
    print(json.dumps(res["headline"]))
    print(f"recursion_e2e: ok {res['ok']}; {out / RESULT}; {text(out / 'art.txt').strip()}")
    return 0 if res["ok"] and rc == 0 else 1


def store(a) -> int:
    subprocess.run(["research", "fetch", a.run, "--all"], check=True, capture_output=True)
    run_dir = a.runs_dir / a.run
    f = run_dir / "out" / RESULT
    if not f.is_file():
        print(f"{f}: no {RESULT}; store takes a `run` run", file=sys.stderr)
        return 1
    res = json.loads(f.read_text())
    job = jload(run_dir / "job.json") or {}
    meta = {"benchmark": "recursion_e2e", "schema": res["schema"], "run": a.run, "ok": res["ok"],
            "commit": ((job.get("run") or {}).get("remote") or {}).get("source_sha") or res["inputs"].get("commit")}
    p = subprocess.run(["research", "data", "put", "--kind", "evidence/v1", "--file", str(f), "--preserve", "--meta",
                        json.dumps(meta)], check=True, capture_output=True, text=True)
    print(p.stdout.strip().splitlines()[-1])
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="verity bench recursion_e2e", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the e2e, every step, into $RESEARCH_RUN_DIR/out (else RUN_DIR/out)")
    r.add_argument("--k", type=int, default=64, help="GemmCoordinate_v2's contraction (default 64)")
    r.add_argument("--n", type=int, default=16, help="instances of the inner statement, every one a unit (default 16)")
    r.add_argument("--runs", type=int, default=1, help="timed sessions per prove, no warm one (default 1)")
    r.add_argument("--neg", nargs="*", default=["challenge"], choices=["challenge", "query"],
                   help="inner coins to flip, one Lean negative each per V* statement (default challenge; none: --neg)")
    r.add_argument("--pcpus", type=int, default=0, help="the prover's cores (default 3/4 of the job's), the rest the verifier's")
    r.add_argument("--jobs", type=int, default=2, help="Lean verifications at once (default 2)")
    r.add_argument("--baseline", action="store_true", help="also today's --zk on the same statement with hm96 leaves, on loopback")
    r.add_argument("--keep-data", action="store_true", help="keep FLOCK_WORK/data/TAG (the staged statements and sessions)")
    r.add_argument("--tag", default=None, help="FLOCK_WORK/data/TAG (default recbench-RUN_ID)")
    r.add_argument("--run-dir", default="recursion_e2e-run", help="without $RESEARCH_RUN_DIR: where out/ goes")
    s = sub.add_parser("summarize", help="DIR/recursion_e2e.json from a run's out/")
    s.add_argument("dir", type=Path)
    t = sub.add_parser("store", help="a run's result into the store (from a VM with the store's keys)")
    t.add_argument("run")
    t.add_argument("--runs-dir", type=Path, default=Path.home() / ".research" / "runs")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return run(a)
    if a.cmd == "summarize":
        print(json.dumps(summarize(a.dir)["headline"]))
        return 0
    return store(a)


if __name__ == "__main__":
    sys.exit(main())
