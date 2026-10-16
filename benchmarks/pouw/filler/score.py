"""The certified filler's scorer (SCORER, cmp-13), locked: changes go to its owner, compute-accounting, never into an arm's
PR. The spec, its desiderata, metrics and gates: note:20261016T0435Z-report-filler-spec. The arithmetic: `scoring.py`;
the arm contract: `arms.py`; the device driver: `dscore.cu`.

  python3 benchmarks/pouw/filler/score.py ARM [--rounds 8] [--reps 3] [--no-robust]   build, lease one GPU, measure, report
  python3 benchmarks/pouw/filler/score.py measure ARM --bin DIR --out DIR             the measurement, inside a lease
  python3 benchmarks/pouw/filler/score.py report OUT_DIR [--json] [--as v2]           score a measured run's out dir
  python3 benchmarks/pouw/filler/score.py check ARM                                   the arm's contract and record limits
  python3 benchmarks/pouw/filler/score.py --leaderboard [--refresh] [--local DIR ...] every filler attempt, best first

A climber's run, from a clean pushed checkout:

  research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env GPU_LEASE_WHO=<lane> \\
    -- python3 benchmarks/pouw/filler/score.py <arm|benchmarks/pouw/filler/my_arm.py:MyArm>

`score.py ARM` builds dcap.cu and dscore.cu (outside the lease), then under `gpu-lease 1 --wait` runs the gates before
anything is timed: FILL's receipt gate (`dcap dgate`), the scorer's own at the arm's chunk shape (`dscore sgate`: poisoned
and launched, every receipt equal to the host's; poisoned with nothing launched, none; stale and skipped work, none) and
CAP's serving gate (`dcap gate`); a failed gate times nothing (exit 5). Then ncu's DRAM bytes for the filler's launches
beside the counter's positive control (1 GiB streamed), and every suite point: ROUNDS rounds in ABBA order of K reps of
serving alone on the whole GPU and K reps under the arm's placement, each arm block followed by a standalone block (the
filler alone on the same SMs as long), every rep's SM clock and board power read (clocks are never set), then serving
alone, the arm and the filler alone each sustained for a settled power window (the energy counter's watts; secondary).
Each filler round (a point's calibration, each arm and standalone block) numbers its tiles from a public range the scorer
fixes from the point and the round, then draws the round's coins from the OS; the replay gate checks the ranges and that
no tile id, coins or sampled seed repeats across rounds. The honest gate's too-good check compares each point's filler
beside serving with the median of its standalone blocks, its limit 1 + max(2%, their trimmed spread), and a positive control
(`control.jsonl`: at one point, a filler credited for S steps a tile that runs S - S/16 beside serving) must be caught.
After the lease it scores the run (`report`), writes `score.json` and one `kernel-attempt/v1` record
(`ledger/<arm>/filler-suite.json`, declared in the run's `outputs.json` so `research run --on` publishes it).

An arm that fails a gate isn't scored: its SCORE is null and `score_if_gated` keeps the number for diagnosis.

filler-scorer/v3, the lock (`scoring.SCORER`), refuses at load, before the build and the lease, an arm whose record the
ledger or the store would refuse (`arms.record_problems`); `check ARM` runs that check alone. Its SCORE is the proved
reading: the filler at the entry `scoring.filler_entry` decides from the arm's declared `Filler.noise` and the kernel
facts of the scorer's own build (`build.json`), with SCORE under CH (every filler at 0.99, v2's reading) beside it, and
each point's slowdown margin. `report OUT --as v2` reads a run as v2 did, without writing `score.json`.

filler-scorer/v4 (prepared, not the lock: `report OUT --as v4` previews it) reads the filler wherever it's called from.
The build also compiles each CUDA driver the arm lists (fscore.cu, pscore.cu) and records `scoring.program_facts` over
dscore's and the drivers' nvdisasm. The theorem entries' SASS checks read the `dfill_k`/`dfill_p` functions' bodies alone
(kernels or __noinline__ clones in serving kernels); any MMA, global store or register call outside them on a path the
filler warps run refuses the arm (gate `filler_paths`); other MMAs are serving's; a block's fixed work counts only from
spot-checked public tiles of a binary the scorer read (`credit_fixed`); and the honest gate bounds the whole kernel's
MACs, serving's and the filler warps', by its SMs' peak (`whole_of_peak`). `scoring.UNCHECKED_V4` is what it can't see.
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
CONTROL = ("decode", 32, 8192)  # the positive control's point, or the arm's first filler point if it places none there


def sh(cmd, out=None, err=None, env=None) -> int:
    with open(out, "a") if out else open(os.devnull, "w") as fo, open(err, "a") if err else open(os.devnull, "w") as fe:
        return subprocess.run(cmd, stdout=fo, stderr=fe, env=env).returncode


def arm_doc(arm) -> dict:
    sys.path.insert(0, str(HERE.parent / "harness"))
    import ledger  # noqa: PLC0415

    fl = {k: v for k, v in dataclasses.asdict(arm.filler).items() if v is not None} if arm.filler else None
    variant = ledger.variant_record(family=f"filler-{arm.name}", device="sm120", schemes=["dfill-receipt"],
                                    sources=[REPO / s for s in arm.sources], root=REPO, loader="dscore",
                                    config=A.config(arm.filler))
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
            keep = any(k in ln for k in S.FILLER_KERNELS)
        if keep:
            lines.append(ln)
    (out / "sass-dfill.txt").write_text("\n".join(lines))
    rec["dfill_sass"] = {"imma": sum("IMMA" in ln for ln in lines),
                         "memory_ops": sum(any(op in ln for op in ("LDG", "STG", "LDS", "STS", "LDL", "STL")) for ln in lines)}
    rec["nvcc_append_flags"] = os.environ.get("NVCC_APPEND_FLAGS") or None
    rec["kernel_facts"] = S.kernel_facts(sass)
    rec |= program_build(bdir, out, log)
    (out / "build.json").write_text(json.dumps(rec, indent=1))
    return rec


def nvdisasm(binary: Path) -> str:
    """Every cubin of `binary` through ``nvdisasm -c``, which names each __noinline__ device function's clone in its
    calling kernels (``cuobjdump -sass`` prints the clone unnamed inside the kernel)."""
    with tempfile.TemporaryDirectory(prefix="filler-nvd.") as d:
        subprocess.run(["cuobjdump", "-xelf", "all", str(binary.resolve())], cwd=d, capture_output=True, check=True)
        cubins = sorted(Path(d).glob("*.cubin"))
        if not cubins:
            raise RuntimeError(f"no cubin in {binary.name}")
        return "".join(subprocess.run(["nvdisasm", "-c", str(c)], capture_output=True, text=True, check=True).stdout
                       for c in cubins)


def program_build(bdir: Path, out: Path, log: Path) -> dict:
    """filler-scorer/v4's part of the build: each CUDA driver the arm lists in its sources besides dcap.cu and dscore.cu
    (fscore.cu, pscore.cu) compiled as ``bdir/<stem>`` with dscore's flags, and `scoring.program_facts` over dscore's and
    the drivers' nvdisasm. It never fails the build: what it can't build or read is recorded, and v4 reads such a build
    at the floor."""
    doc = json.loads((out / "arm.json").read_text()) if (out / "arm.json").is_file() else {}
    drivers, unbuilt = {"dscore": bdir / "dscore"}, {}
    for s in doc.get("sources") or ():
        p = REPO / s
        if p.suffix == ".cu" and p.name not in ("dcap.cu", "dscore.cu"):
            rc = sh([NVCC, *NVFLAGS, "-o", str(bdir / p.stem), str(p), *LIBS], out=log, err=log)
            if rc:
                unbuilt[p.stem] = f"build rc {rc}"
            else:
                drivers[p.stem] = bdir / p.stem
    try:
        pf, why = S.program_facts({k: nvdisasm(v) for k, v in drivers.items()}), None
    except (OSError, RuntimeError, subprocess.CalledProcessError) as ex:
        pf, why = None, f"{type(ex).__name__}: {ex}"
    return {"program_facts": pf, "program_facts_why": why, "drivers_unbuilt": unbuilt}


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
    pts = S.points(robust)
    for i, (ph, b, l) in enumerate(pts):
        p = A.placed(arm, ph, b, l)
        f = fl or A.Filler()
        cmd = [str(bdir / "dscore"), "dscore", "step" if ph == "decode" else "prefill", str(b), str(l), p.place,
               str(p.fill_sms), str(f.W), str(f.NC), str(f.S), str(rounds), str(reps), str(i)]
        rc = sh(cmd, out=out / "dscore.jsonl", err=err, env=env)
        note(f"dscore {ph} {b} {l} {p.place} {p.fill_sms} rc {rc}")
        smi(out)
    if fl:
        control(arm, fl, bdir, out, env, rounds, reps, len(pts), note)
    return 0


def control_point(arm, pts=None):
    """The positive control's point: CONTROL, unless the arm places no filler there, then its first filler point."""
    cand = [CONTROL] + list(pts or S.points())
    return next(((ph, b, l) for ph, b, l in cand if A.placed(arm, ph, b, l).place != "none"), None)


def control(arm, fl, bdir: Path, out: Path, env: dict, rounds: int, reps: int, index: int, note) -> int:
    """The honest gate's positive control (`control.jsonl`): the arm's filler shape and placement at `control_point`,
    its arm blocks running S - S/CHEAT steps a tile credited at S (DSCORE_CHEAT_S), its standalone blocks S. Its rounds
    take point index `index`, past the suite's, so its tile ranges are its own."""
    pt = control_point(arm)
    if pt is None:
        return 0
    ph, b, l = pt
    p = A.placed(arm, ph, b, l)
    cmd = [str(bdir / "dscore"), "dscore", "step" if ph == "decode" else "prefill", str(b), str(l), p.place,
           str(p.fill_sms), str(fl.W), str(fl.NC), str(fl.S), str(rounds), str(reps), str(index)]
    rc = sh(cmd, out=out / "control.jsonl", err=out / "errors.txt", env=dict(env, DSCORE_CHEAT_S=str(S.cheat_s(fl.S))))
    note(f"control {ph} {b} {l} {p.place} {p.fill_sms} S {fl.S} runs {S.cheat_s(fl.S)} rc {rc}")
    return rc


def control_stats(out: Path) -> dict | None:
    r = rows(out / "control.jsonl")
    if not r:
        return None
    r = r[0]
    return {**S.point_stats(r), "at": f"{'decode' if r['kind'] == 'step' else 'prefill'} {r['B']}/{r['L']}",
            "cheat_S": r.get("cheat_S"), "S": r["S"]}


def rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip().startswith("{")] if path.is_file() else []


def report(out: Path, arm=None, version: str | None = None) -> dict:
    """The run's result under `version` (the active `scoring.SCORER` by default), written to ``score.json`` only under the
    active one: another version's (``report --as v3``) is a preview."""
    version = version or S.SCORER
    doc = json.loads((out / "arm.json").read_text())
    if arm is None:
        try:
            arm = A.load(doc.get("spec") or doc["name"])
        except (A.ArmError, OSError, AttributeError):
            arm = None
    drows = rows(out / "dscore.jsonl")
    has_filler = doc["filler"] is not None
    build = json.loads((out / "build.json").read_text()) if (out / "build.json").is_file() else {}
    entry, pf, credit = None, build.get("program_facts"), None
    if version == S.V4 and has_filler:
        drows, credit = S.credit_fixed(drows, pf)
        unread = {r.get("driver", "dscore") for r in drows} - set(pf["binaries"] if pf else ())
        entry = S.filler_entry_v4(doc["filler"].get("noise"), pf, unread=sorted(unread))
    stats = {("decode" if r["kind"] == "step" else "prefill", r["B"], r["L"]): S.point_stats(r) for r in drows}
    if version == S.V3 and has_filler:
        fixed = any((b.get(side) or {}).get("fixed_ms") for r in drows for b in r["blocks"] for side in ("arm", "solo"))
        entry = S.filler_entry(doc["filler"].get("noise"), build.get("kernel_facts"), fixed_work=fixed)
    sc = S.score(stats, S.certified(entry["active"]) if entry else S.CERTIFIED)
    if entry and entry["applies"] != entry["active"]:
        entry["score_if_active"] = S.score(stats, S.certified(entry["applies"])).get("SCORE")
    under_ch = S.score(stats, S.certified(S.UNDER_CH)).get("SCORE") if entry else sc.get("SCORE")
    sg = rows(out / "sgate.jsonl")
    def read(p):
        return p.read_text().strip() if p.is_file() else None

    dram = S.dram_gate(read(out / "ncu-filler.csv") if has_filler else None, read(out / "ncu-control.csv"))
    clocks = S.clock_gate(stats, S.parse_smi(read(out / "smi.csv") or ""))
    whole = ({f"{'decode' if r['kind'] == 'step' else 'prefill'} {r['B']}/{r['L']}": S.whole_of_peak(r) for r in drows}
             if version == S.V4 else None)
    honest = (S.honesty(arm, stats, S.points(), control=control_stats(out), has_filler=has_filler, whole=whole)
              if arm is not None else {"ok": False, "why": f"arm {doc['name']} can't be loaded to check its placement"})
    g = S.gates(stats, sc, has_filler=has_filler, sgate=sg[0] if sg else None, dgate=rows(out / "dgate.jsonl"),
                dram=dram, clocks=clocks, honest=honest,
                replay=S.replay_gate(drows) if has_filler else {"ok": True, "why": "no filler"})
    serving_gate = rows(out / "gate.jsonl")
    g["serving_exact"] = {"ok": bool(serving_gate) and all(r.get("ok") is not False for r in serving_gate),
                          "rows": len(serving_gate)}
    if version == S.V4:
        g["filler_paths"] = S.paths_gate(pf) if has_filler else {"ok": True, "why": "no filler"}
    g["ok"] = g["ok"] and g["serving_exact"]["ok"] and g.get("filler_paths", {"ok": True})["ok"]
    res = {"scorer": version, "arm": doc["name"], "interface": doc["interface"], "lineage": doc["lineage"],
           "variant": doc["variant"]["id"], "choice": doc["choice"], "commit": read(out / "commit.txt"),
           "coins": read(out / "coins.txt"), "run": os.environ.get("RESEARCH_RUN_ID") or read(out / "run.txt"),
           "scored": g["ok"] and "SCORE" in sc, "SCORE": sc.get("SCORE") if g["ok"] else None,
           "score_if_gated": sc.get("SCORE"), "min": sc.get("min"), "min_mix": sc.get("min_mix"),
           "score_nominal": sc.get("score_nominal"), "score_at_target": sc.get("score_at_target"),
           "flags": sc["flags"], "certified": sc["certified"], "gates": g, "missing": sc["missing"],
           "entries": sc["entries"], "secondary": S.secondary(stats, sc),
           "points": {f"{k[0]} {k[1]}/{k[2]}": v for k, v in stats.items()}}
    if version in (S.V3, S.V4):
        res["margins"] = S.margins(stats)
        res["filler_entry"] = entry
        res |= {"score_under_ch": under_ch if g["ok"] else None, "score_under_ch_if_gated": under_ch}
    if version == S.V4:
        res["fixed_credit"] = credit
        res["program_facts_why"] = build.get("program_facts_why")
    if version == S.SCORER:
        (out / "score.json").write_text(json.dumps(res, indent=1))
    return res


def record(res: dict, doc: dict, run_dir: Path, by: str | None) -> dict:
    import ledger  # noqa: PLC0415

    lin = doc["lineage"]
    g = res["gates"]
    v3 = res["scorer"] in (S.V3, S.V4)
    meta = {
        "schema": ledger.KIND, "time": ledger.now(), "run": res["run"], "run_id": res["run"], "harness": res["commit"],
        "tier": "2a", "by": by, "arm": res["arm"], "shape": SHAPE, "phase": "serving-suite", "scorer": res["scorer"],
        "variant": res["variant"], "variant_record": doc["variant"], "parent": lin.get("parent"),
        "approach": lin.get("approach"), "mechanism": sorted(set(lin.get("mechanism") or [])),
        "hypothesis": ledger.one_line(lin.get("hypothesis"), "the hypothesis"),
        "choice": ledger.one_line(res["choice"], "choice") if v3 else res["choice"],
        "gates": {"version": res["scorer"], "arm_interface": doc["interface"]},
        "gate": {k: bool(v.get("ok")) for k, v in g.items() if isinstance(v, dict)} | {"ok": g["ok"]},
        "metrics": {"SCORE": res["SCORE"], "score_if_gated": res["score_if_gated"], "min": res["min"],
                    "min_mix": res["min_mix"], "score_nominal": res["score_nominal"],
                    "score_at_target": res["score_at_target"], "slowdown_worst": g["slowdown"]["worst"],
                    "slowdown_at": g["slowdown"]["at"], "dram_share_of_hbm": g["dram"].get("share_of_hbm"),
                    "entries": {e["entry"]: {"coverage": e["coverage"], "time": e["time"],
                                             "filler_share": e["filler_share"]} for e in res["entries"]},
                    "work_cv_max": res["secondary"]["work_cv_max"],
                    "honest_control": {k: (g["honest"].get("control") or {}).get(k)
                                       for k in ("beside_over_alone", "limit", "caught")}},
        "flags": res["flags"], "certified": res["certified"], "coins": res["coins"],
    }
    if v3:
        mg = res["margins"]
        meta["metrics"] |= {"margin_min": mg["min"], "no_margin": mg["no_margin"],
                            "margins": {k: m["margin"] for k, m in mg["points"].items() if m["scored"]},
                            "score_under_ch": res["score_under_ch"],
                            "score_under_ch_if_gated": res["score_under_ch_if_gated"]}
        fe = res["filler_entry"]
        meta["filler_entry"] = fe and {k: fe.get(k) for k in ("applies", "active", "noise", "fraction", "checks",
                                                              "failed", "kernel", "score_if_active")}
    if res.get("fixed_credit"):
        meta["fixed_credit"] = {"credited": res["fixed_credit"]["credited"],
                                "dropped": len(res["fixed_credit"]["dropped"])}
    rec = {"kind": ledger.KIND, "meta": ledger._clean(meta), "refs": {"parent": lin["parent"]}
           if str(lin.get("parent") or "").startswith("art:") else {}, "labels": [(ledger.KERNEL_OUTCOME, "screened")]}
    ledger.write(run_dir, [rec])
    if (run_dir / "job.json").is_file() or os.environ.get("RESEARCH_RUN_DIR"):
        ledger.write_outputs(run_dir, [rec])
    return rec


def fmt(x, p=3):
    return "–" if x is None else f"{x:.{p}f}" if isinstance(x, float) else str(x)


def control_line(c: dict | None) -> str:
    if c is None:
        return "no filler, no control"
    if "why" in c:
        return f"control: {c['why']}"
    return (f"control (S {c['S']} credited, {c['cheat_S']} run beside serving, at {c['point']}): beside ÷ standalone "
            f"{fmt(c['beside_over_alone'])} vs limit {fmt(c['limit'])}, {'caught' if c['caught'] else 'NOT CAUGHT'}")


def show_margins(mg: dict) -> None:
    print(f"\n| point | slowdown | margin (limit {mg['limit']}) | worst / best round's | placement |\n|---|---|---|---|---|")
    for name, m in sorted(mg["points"].items(), key=lambda x: (not x[1]["scored"], x[1]["margin"])):
        r = m["rounds"] or [None, None]
        print(f"| {name}{'' if m['scored'] else ' (robust, unscored)'} | {fmt(m['slowdown'])} | {fmt(m['margin'], 4)} | "
              f"{fmt(r[0], 4)} / {fmt(r[1], 4)} | {m['place']} {m['fill_sms'] or ''} |")
    lo = mg["min"] or {}
    print(f"least margin {fmt(lo.get('margin'), 4)} at {lo.get('at')}; no margin (a round at or past the limit): "
          f"{', '.join(mg['no_margin']) or 'none'}")


def show(res: dict) -> None:
    g = res["gates"]
    print(f"\n{res['arm']} ({res['variant']}), {res['scorer']}"
          + ("" if res["scorer"] == S.SCORER else f" (a preview: the active lock is {S.SCORER})")
          + f": SCORE {fmt(res['SCORE'])}"
          + ("" if res["scored"] else f" (not scored; {fmt(res['score_if_gated'])} if gated)")
          + f", min {fmt((res['min'] or {}).get('coverage'))} at {(res['min'] or {}).get('entry')}"
          + f" (mixes: {fmt((res['min_mix'] or {}).get('coverage'))} at {(res['min_mix'] or {}).get('entry')})"
          + f"; nominal ÷ capacity {fmt(res['score_nominal'])}, at 0.99 for all {fmt(res['score_at_target'])}")
    if "score_under_ch_if_gated" in res:
        print(f"  SCORE under CH (every filler at 0.99; CH withdrawn, decision 27; "
              + ("of the fixed work v4 credits" if res["scorer"] == S.V4 else "v2's reading, for comparison")
              + f"): {fmt(res['score_under_ch'])}" + ("" if res["scored"] else
                                                      f" ({fmt(res['score_under_ch_if_gated'])} if gated)"))
    for f in res["flags"]:
        print(f"  FLAG: {f}")
    fe = res.get("filler_entry")
    if fe:
        print(f"  filler entry: {fe['applies']} applies ({fe['fraction']}), declared noise {fe['noise']}"
              + ("" if fe["applies"] == fe["active"] else
                 f"; inactive, so SCORE reads {fe['active']} ({fmt(fe.get('score_if_active'))} once it is active)")
              + "".join(f"\n    not {k}: {w}" for k, ws in fe["failed"].items() for w in ws)
              + ("\n    unchecked: " + "; ".join(fe["unchecked"]) if fe["unchecked"] else ""))
        kf = fe.get("kernel") or {}
        if "callers" in kf:
            print(f"    filler functions {len(kf['functions'] or [])}, their callers {len(kf['callers'])}, serving MMAs "
                  f"in {len(kf['serving_mma'])} functions")
    fc = res.get("fixed_credit")
    if fc:
        print(f"  fixed work credited in {fc['credited']} blocks, dropped in {len(fc['dropped'])}"
              + (f": {fc['dropped'][0]}" + (" ..." if len(fc["dropped"]) > 1 else "") if fc["dropped"] else ""))
    for name in GATES + (("filler_paths",) if "filler_paths" in g else ()):
        v = g.get(name, {})
        extra = {"slowdown": lambda: f"worst {fmt(v.get('worst'))} at {v.get('at')} ({v.get('phase')})",
                 "replay": lambda: v.get("why") or f"{v.get('rounds')} rounds, {v.get('tiles'):.3g} tiles, ranges the "
                                   f"scorer's and disjoint, coins and {v.get('seeds_sampled')} sampled seeds distinct",
                 "receipts": lambda: v.get("why") or f"timed {v.get('timed_checked')} checked, {v.get('timed_bad')} bad",
                 "honest": lambda: v.get("why") or "; ".join(
                     [f"{k}: {', '.join(v[k])}" for k in ("unstable_placement", "too_good", "over_peak", "no_reference",
                                                          "over_peak_whole") if v.get(k)]
                     + ([f"whole kernel ≤ {fmt(v['whole_of_peak_max'])} of peak"] if v.get("whole_of_peak_max") else [])
                     + [control_line(v.get("control"))]),
                 "filler_paths": lambda: v.get("why") or f"{v.get('callers')} callers, nothing outside dfill_k on "
                                         "their filler warps' paths",
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
    if "margins" in res:
        show_margins(res["margins"])
    sec = res["secondary"]
    ck = [v for v in sec["clock"].values() if all(S.isfinite(x) for x in v)]
    setup = sorted(sec["placement_setup_ms"].values())

    def wmax(k):
        return fmt(max((v[k] for v in sec["power"].values() if S.isfinite(v.get(k)) and v[k] > 0), default=None), 0)
    print(f"\nsecondary: work CV max {fmt(sec['work_cv_max'])}; board W (settled) serving alone/arm/filler alone, max "
          f"{wmax('alone')}/{wmax('arm')}/{wmax('filler_alone')}; SM MHz "
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

    def readings(m):  # (proved SCORE, SCORE under CH), each the gated number or None; v1's and v2's SCORE was CH's
        mt, ok = m["metrics"], gated(m)
        if m.get("scorer") in ("filler-scorer/v1", S.V2):
            return None, mt.get("SCORE" if ok else "score_if_gated")
        return mt.get("SCORE" if ok else "score_if_gated"), mt.get("score_under_ch" if ok else "score_under_ch_if_gated")

    uniq.sort(key=lambda x: (not gated(x[1]), *(-(r if r is not None else -1) for r in readings(x[1]))))
    if as_json:
        print(json.dumps([{"record": a, "gated": gated(m), "proved": readings(m)[0], "under_ch": readings(m)[1], **m}
                          for a, m in uniq]))
        return 0
    print("SCORE is the proved reading (filler-scorer/v3); SCORE under CH reads every filler at 0.99 (CH withdrawn, "
          "decision 27), as v2's SCORE did. A v2 record has no proved SCORE; an ungated one's numbers are in ().")
    print("| # | arm | variant | SCORE | SCORE under CH | min (entry) | worst slowdown | gates | run | scorer | parent | "
          "mechanism | flags |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, (art, m) in enumerate(uniq, 1):
        mt, gt = m["metrics"], m.get("gate") or {}
        sc, ch = (fmt(r) if gated(m) or r is None else f"({fmt(r)})" for r in readings(m))
        failed = [k for k, v in gt.items() if k != "ok" and not v] + [f"no {k}" for k in GATES if k not in gt]
        print(f"| {i} | {m['arm']} | {m.get('variant')} | {sc} | {ch} | {fmt((mt.get('min') or {}).get('coverage'))} "
              f"({(mt.get('min') or {}).get('entry')}) | {fmt(mt.get('slowdown_worst'))} | "
              f"{'pass' if gated(m) else 'FAIL ' + ','.join(failed)} | {m.get('run')} | "
              f"{(m.get('scorer') or '').removeprefix('filler-scorer/')}@{(m.get('harness') or '')[:9]} | "
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
        ap.add_argument("--as", dest="version", choices=("v2", "v3", "v4"),
                        help="score under this version (default the active lock); another's is a preview, not written")
        a = ap.parse_args(argv[1:])
        res = report(a.out, version={"v2": S.V2, "v3": S.V3, "v4": S.V4}.get(a.version))
        print(json.dumps(res, indent=1)) if a.json else show(res)
        return 0
    if argv and argv[0] == "check":
        ap = argparse.ArgumentParser(prog="score.py check", description="an arm against the contract and its record's "
                                     "limits (filler-scorer/v3's load check), with no build or lease")
        ap.add_argument("arm")
        a = ap.parse_args(argv[1:])
        try:
            arm = A.load(a.arm, record=True)
        except A.ArmError as ex:
            print(f"refused: {ex}")
            return 2
        print(f"{arm.name}: the contract and the record's limits hold")
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
    arm = A.load(a.arm, record=S.SCORER in (S.V3, S.V4))
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
