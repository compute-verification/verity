"""The certified filler's scorer, locked: a change to it is its own change, never part of an arm's. The arithmetic:
`scoring.py`; the arm contract: `arms.py`; the device driver: `dscore.cu`.

  python3 benchmarks/pouw/filler/score.py ARM [--rounds 8] [--reps 3] [--no-robust]   build, lease one GPU, measure, report
  python3 benchmarks/pouw/filler/score.py measure ARM --bin DIR --out DIR             the measurement, inside a lease
  python3 benchmarks/pouw/filler/score.py report OUT_DIR [--json] [--as v4] [--coins]  score a measured run's out dir
  python3 benchmarks/pouw/filler/score.py tp-report OUT... --bin|--nvd X [--noise]    TP score of tpcomm.sh out dirs (v5)
  python3 benchmarks/pouw/filler/score.py defer-fit [--T 4096] [--json]               the deferred pass's calibration
  python3 benchmarks/pouw/filler/score.py check ARM                                   the arm's contract and record limits
  python3 benchmarks/pouw/filler/score.py --leaderboard [--refresh] [--local DIR ...] every filler attempt, best first

A climber's run, from a clean pushed checkout:

  research run --on vy-nebius-2 --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \\
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

filler-scorer/v3 refuses at load, before the build and the lease, an arm whose record the ledger or the store would
refuse (`arms.record_problems`); `check ARM` runs that check alone. Its SCORE is the proved reading: the filler at the
entry `scoring.filler_entry` decides from the arm's declared `Filler.noise` and the kernel facts of the scorer's own
build (`build.json`), with SCORE under CH (every filler at 0.99, v2's reading) beside it, and each point's slowdown
margin. `report OUT --as v2` (or `v3`) reads a run as that version did, without writing `score.json`.

filler-scorer/v4 (`report OUT --as v4`) keeps v3's load check and
reads the filler wherever it's called from.
The build also compiles each CUDA driver the arm lists (fscore.cu, pscore.cu) and records `scoring.program_facts` over
dscore's and the drivers' nvdisasm. The theorem entries' SASS checks read the `dfill_k`/`dfill_p` functions' bodies alone
(kernels or __noinline__ clones in serving kernels); any MMA, global store or register call outside them on a path the
filler warps run refuses the arm (gate `filler_paths`); other MMAs are serving's; a block's fixed work counts only from
spot-checked public tiles of a binary the scorer read (`credit_fixed`); and the honest gate bounds the whole kernel's
MACs, serving's and the filler warps', by its SMs' peak (`whole_of_peak`). `scoring.UNCHECKED_V4` is what it can't see.

filler-scorer/v5, the lock (`scoring.SCORER`; its fold check is a soundness correction), keeps v4's gates, the slowdown
gate on the measured step against clean serving. It reads the receipt's fold (`scoring.fold_reading`) from `Filler.fold`
or the lineage's receipt tag and the build's `fold_facts`: ram and the XOR-family folds halve a theorem entry (Theorem
K's proved half; their full reading, if the r10 select-charging lemma lands, is a secondary readout,
`scoring.SELECT_LEMMA`), and duo, whose receipt takes in every word (`scoring.FOLDS`), reads in full where the build
confirms it. Its SCORE is the measured filler part plus a MODELLED served part (`scoring.score_v5`;
`served_part_modelled` in score.json and the record): every served MAC at Theorem K's 0.9 over the wall time the edited
design would take (`scoring.served_model`'s named constants and their sources, decode's deferred pass calibrated to
measurement, `scoring.defer_fit`; `--coins` sets prefill's coins factor). `tp-report` reads tpcomm's TP suite
(`scoring.tp_score`). `defer-fit` prints the deferred pass's calibration and its residuals.
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
    for name, src in (("dcap", HERE / "dcap.cu"), ("dscore", HERE / "dscore.cu")):
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
    the drivers' nvdisasm, and filler-scorer/v5's `scoring.fold_facts` over the same at the arm's NC. It never fails the
    build: what it can't build or read is recorded, and v4 reads such a build at the floor."""
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
        listings = {k: nvdisasm(v) for k, v in drivers.items()}
        pf, why = S.program_facts(listings), None
        ff = S.fold_facts(listings, (doc.get("filler") or {}).get("NC", 4))
    except (OSError, RuntimeError, subprocess.CalledProcessError) as ex:
        pf, ff, why = None, None, f"{type(ex).__name__}: {ex}"
    return {"program_facts": pf, "program_facts_why": why, "drivers_unbuilt": unbuilt, "fold_facts": ff}


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


def report(out: Path, arm=None, version: str | None = None, coins: float = S.PREFILL_COINS) -> dict:
    """The run's result under `version` (the active `scoring.SCORER` by default), written to ``score.json`` only under the
    active one: another version's (``report --as v3``) is a preview. Under filler-scorer/v5, `coins` is prefill's
    per-stage coins factor (`scoring.PREFILL_COINS`)."""
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
    entry, pf, credit, fold, v5 = None, build.get("program_facts"), None, None, None
    if version in (S.V4, S.V5) and has_filler:
        drows, credit = S.credit_fixed(drows, pf)
        unread = {r.get("driver", "dscore") for r in drows} - set(pf["binaries"] if pf else ())
        if version == S.V4:
            entry = S.filler_entry_v4(doc["filler"].get("noise"), pf, unread=sorted(unread))
        else:
            declared, how = S.declared_fold(doc["filler"], doc.get("lineage"))
            flags = (build["nvcc_append_flags"] or "") if "nvcc_append_flags" in build else None
            fold = S.fold_reading(declared, doc["filler"].get("NC"), flags, build.get("fold_facts"), how)
            entry = S.filler_entry_v5(doc["filler"].get("noise"), pf, fold, unread=sorted(unread))
    stats = {("decode" if r["kind"] == "step" else "prefill", r["B"], r["L"]): S.point_stats(r) for r in drows}
    if version == S.V3 and has_filler:
        fixed = any((b.get(side) or {}).get("fixed_ms") for r in drows for b in r["blocks"] for side in ("arm", "solo"))
        entry = S.filler_entry(doc["filler"].get("noise"), build.get("kernel_facts"), fixed_work=fixed)
    if version == S.V5:
        sc = S.score(stats, S.certified_v5(entry) if entry else S.CERTIFIED)
        v5 = S.score_v5(stats, entry["active_fraction"] if entry else 0.0, coins)
        if entry and entry["applies"] != entry["active"]:
            entry["score_if_active"] = S.score_v5(stats, entry["fraction"], coins).get("SCORE")
        if entry and entry.get("if_fold"):
            x = S.score_v5(stats, entry["if_fold"]["active_fraction"], coins)
            entry["if_fold"] |= {"SCORE": x.get("SCORE"), "filler_part": x.get("filler_part")}
    else:
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
             if version in (S.V4, S.V5) else None)
    honest = (S.honesty(arm, stats, S.points(), control=control_stats(out), has_filler=has_filler, whole=whole)
              if arm is not None else {"ok": False, "why": f"arm {doc['name']} can't be loaded to check its placement"})
    g = S.gates(stats, sc, has_filler=has_filler, sgate=sg[0] if sg else None, dgate=rows(out / "dgate.jsonl"),
                dram=dram, clocks=clocks, honest=honest,
                replay=S.replay_gate(drows) if has_filler else {"ok": True, "why": "no filler"})
    serving_gate = rows(out / "gate.jsonl")
    g["serving_exact"] = {"ok": bool(serving_gate) and all(r.get("ok") is not False for r in serving_gate),
                          "rows": len(serving_gate)}
    if version in (S.V4, S.V5):
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
    if version in (S.V3, S.V4, S.V5):
        res["margins"] = S.margins(stats)
        res["filler_entry"] = entry
        res |= {"score_under_ch": under_ch if g["ok"] else None, "score_under_ch_if_gated": under_ch}
    if version in (S.V4, S.V5):
        res["fixed_credit"] = credit
        res["program_facts_why"] = build.get("program_facts_why")
    if version == S.V5:
        res |= {"SCORE": v5.get("SCORE") if g["ok"] else None, "score_if_gated": v5.get("SCORE"),
                "SCORE_is": "the measured filler part + the MODELLED served part (the harness serves plain int8)",
                "filler_part": v5.get("filler_part"), "served_part_modelled": v5.get("served_part_modelled"),
                "min": v5.get("min"), "scored": g["ok"] and "SCORE" in v5, "v4_form": sc.get("SCORE"),
                "entries_v5": v5["entries"], "served_model": v5["served_model"], "fold": fold}
        res.pop("score_under_ch")
        res.pop("score_under_ch_if_gated")
    if version == S.SCORER:
        (out / "score.json").write_text(json.dumps(res, indent=1))
    return res


def record(res: dict, doc: dict, run_dir: Path, by: str | None) -> dict:
    import ledger  # noqa: PLC0415

    lin = doc["lineage"]
    g = res["gates"]
    v3 = res["scorer"] in (S.V3, S.V4, S.V5)
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
                            "margins": {k: m["margin"] for k, m in mg["points"].items() if m["scored"]}}
        meta["metrics"] |= {k: res[k] for k in ("score_under_ch", "score_under_ch_if_gated", "filler_part",
                                                "served_part_modelled", "v4_form") if k in res}
        if res.get("fold"):
            meta["fold"] = {k: res["fold"][k] for k in ("declared", "signature", "fold", "share")}
        fe = res["filler_entry"]
        meta["filler_entry"] = fe and {k: fe.get(k) for k in ("applies", "active", "noise", "fraction", "checks",
                                                              "failed", "kernel", "score_if_active", "if_fold")}
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


def fit_line(f: dict) -> str:
    """`scoring.defer_fit`'s fitted quantities (s, u or d) as formulas."""
    return "; ".join(f"{q} = {' / '.join(f'TP {t} {v:.3f}' for t, v in f[q]['intercept'].items())} x (B/32)^"
                     f"{f[q]['batch']:.3f} x (L/2K)^{f[q]['context']:.3f} (rms {f[q]['rms']:.1%})"
                     for q in ("s", "u", "d") if q in f)


def show_served_model(sm: dict) -> None:
    c = sm["constants"]
    fits = {"DEFER_FIT": "one GPU's u (r3)", "DEFER_FIT_TP": "TP (r3 + r6)"}
    print("  MODELLED served component (the harness serves plain int8): "
          + ", ".join(f"{k} {fmt(v)}" for k, v in c.items() if k not in fits))
    for k, what in fits.items():
        if k in c:
            print(f"    deferred pass, {what}: {fit_line(c[k])}")
    for k, src in sm["sources"].items():
        print(f"    {k}: {src}")
    print(f"    model: {sm['model']}")


def show_defer_fit(res: dict) -> None:
    q = res["form"]
    print(f"the deferred pass at T = {res['T']}, {res['n']} TP points ({q} form): {fit_line(res)}")
    print(f"  the added time's largest error {res['slowdown_err_pts_max']:.2f} points of step slowdown; the filler's "
          f"remaining share at the plain pass, error rms {res['fill_err_pts_rms']:.2f} and largest "
          f"{res['fill_err_pts_max']:.2f} points of C; residual = fit ÷ measured − 1 (positive: the model overstates)")
    print(f"\n| TP | B | L | s | s fit | resid | {q} | {q} fit | resid | added ms | error ms | slowdown pts | error pts "
          "| fill share | fit | error pts |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in res["points"]:
        print(f"| {p['tp']} | {p['B']} | {p['L']} | {p['s']:.3f} | {p['s_fit']:.3f} | {p['s_resid']:+.1%} | "
              f"{p[q]:.3f} | {p[q + '_fit']:.3f} | {p[q + '_resid']:+.1%} | {p['added_ms']:.2f} | "
              f"{p['added_err_ms']:+.2f} | {p['slowdown_pts']:.1f} | {p['slowdown_err_pts']:+.2f} | "
              f"{p['fill_share']:.4f} | {p['fill_share_fit']:.4f} | {p['fill_err_pts']:+.2f} |")
    print("\none GPU's shown share (STAGE-COST, B 32): " + ", ".join(f"T {t} {v:.3f}" for t, v in res["one_gpu"].items()))
    if res.get("one_gpu_u"):
        print(f"one GPU's displacement u(1 − s), u at least DEFER_OF_PEAK and TP 2's r3 fit: {fit_line(res['one_gpu_u'])}")
    print(f"runs: {res['runs']}")


def show_v5(res: dict) -> None:
    fd, fe = res.get("fold"), res.get("filler_entry") or {}
    print(f"  SCORE = measured filler part {fmt(res['filler_part'])} + modelled served part {fmt(res['served_part_modelled'])}"
          f" (v4's arithmetic, served plain at {S.CERTIFIED['serving']['fraction']}, with v5's fold: "
          f"{fmt(res['v4_form'])})")
    if fd:
        print(f"  fold: {fd['fold']} (share {fd['share']}), declared {fd['declared']} by {fd['declared_by']}, the SASS "
              f"shows {fd['signature']}, NC {fd['nc']}, NVCC_APPEND_FLAGS {fd['flags']!r}"
              + "".join(f"\n    {w}" for w in fd["why"]))
        print(f"  filler entry {fe.get('applies')}: {fe.get('unfolded')} unfolded, {fe.get('fraction')} folded")
        if fe.get("if_fold"):
            x = fe["if_fold"]
            print(f"  secondary readout, not SCORE: {fmt(x.get('SCORE'))} (filler part {fmt(x.get('filler_part'))}, "
                  f"entry {x['fraction']}) if {x['if']}")
    show_served_model(res["served_model"])
    print("\n| entry | coverage | filler part | served part (modelled) | t1 measured ms | t1 modelled ms | filler MACs measured → "
          "modelled |\n|---|---|---|---|---|---|---|")
    for e in res["entries_v5"]:
        print(f"| {e['entry']} | {fmt(e['coverage'])} | {fmt(e['filler_part'])} | {fmt(e['served_part_modelled'])} | "
              f"{fmt(e['t1_measured_ms'], 2)} | {fmt(e['t1_ms'], 2)} | {e['filler_measured_macs']:.3g} → "
              f"{e['filler_macs']:.3g} |")


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
    if res["scorer"] == S.V5:
        show_v5(res)
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
    print("SCORE is the proved reading (filler-scorer/v3, v4 and v5). A v5 SCORE is the measured filler part plus a "
          "MODELLED served part (the harness serves plain int8), both shown; v3's and v4's read serving plain at 0.225. "
          "SCORE under CH reads every filler at 0.99 (CH withdrawn, decision 27), as v2's SCORE did. A v2 record has no "
          "proved SCORE; an ungated one's numbers are in ().")
    print("| # | arm | variant | SCORE | filler part | served part (modelled) | SCORE under CH | min (entry) | "
          "worst slowdown | gates | run | scorer | parent | mechanism | flags |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, (art, m) in enumerate(uniq, 1):
        mt, gt = m["metrics"], m.get("gate") or {}
        sc, ch = (fmt(r) if gated(m) or r is None else f"({fmt(r)})" for r in readings(m))
        failed = [k for k, v in gt.items() if k != "ok" and not v] + [f"no {k}" for k in GATES if k not in gt]
        print(f"| {i} | {m['arm']} | {m.get('variant')} | {sc} | {fmt(mt.get('filler_part'))} | "
              f"{fmt(mt.get('served_part_modelled'))} | {ch} | {fmt((mt.get('min') or {}).get('coverage'))} "
              f"({(mt.get('min') or {}).get('entry')}) | {fmt(mt.get('slowdown_worst'))} | "
              f"{'pass' if gated(m) else 'FAIL ' + ','.join(failed)} | {m.get('run')} | "
              f"{(m.get('scorer') or '').removeprefix('filler-scorer/')}@{(m.get('harness') or '')[:9]} | "
              f"{m.get('parent')} | {','.join(m.get('mechanism') or [])} | {'provisional' if m.get('flags') else ''} |")
    return 0


#: what the TP score's gates can't check from tpcomm.sh's out dirs
TP_UNCHECKED = (
    "replay: tpcomm's filler rounds carry no tile range the scorer fixes, so no anti-replay check",
    "serving_exact: tpcomm records no serving gate rows",
    "honest: no positive control and no standalone blocks (the too-good check); the whole-of-peak bound is read",
    "prefill: the TP suite is decode only, so the modelled served part has no prefill term")


def _text(p: Path) -> str | None:
    return p.read_text().strip() if p.is_file() else None


def tp_report(outs: list[Path], listings: dict | None, noise: str | None, fold: str | None,
              facts_from: str | None = None) -> dict:
    """The TP score (filler-scorer/v5): tpcomm.sh's out dirs (the TP suite), read together. The filler is read at v5's
    entry (`scoring.filler_entry_v5`) from the TP build's program and fold facts at `scoring.TP_FILLER` (`listings`: the
    tpcomm binary's nvdisasm; None reads the filler at the floor), with the declared noise and fold; each (TP, B, L) at
    `scoring.tp_entry`; the gates are `scoring.tp_gates`, each out dir's DRAM (ncu, `scoring.TP_NCU_KERNEL`) and clocks
    (application clocks at their default, every rep's clock read), and the whole kernel's MACs, served and filler,
    within each rank's peak."""
    sets, by = [], {}
    for out in outs:
        for sd in sorted(p for p in out.glob("set*") if p.is_dir()):
            rws = S.tp_rows({int(p.stem.removeprefix("rank")): p.read_text() for p in sd.glob("rank*.jsonl")})
            sets.append({"dir": f"{out.parent.name}/{sd.name}", "settings": _text(sd / "settings.txt") or "", "rows": rws})
            for k, arms_ in rws["arms"].items():
                for arm, reps in arms_.items():
                    by.setdefault(k, {}).setdefault(arm, []).extend(reps)
    pf = S.program_facts(listings, S.TP_FILLER) if listings else None
    ff = S.fold_facts(listings, 4, S.TP_FILLER) if listings else None
    fd = S.fold_reading(fold, None, None, ff, "--fold" if fold else "undeclared")
    entry = S.filler_entry_v5(noise, pf, fd)
    entries = {k: S.tp_entry(v, k, entry["active_fraction"]) for k, v in sorted(by.items())
               if v.get("base") and v.get("phase")}
    g = S.tp_gates(sets, entries, pf)
    dram, clocks = [], []
    for out in outs:
        dram.append({"dir": out.parent.name, **S.dram_gate(_text(out / "ncu-filler.csv"), _text(out / "ncu-control.csv"),
                                                          S.TP_NCU_KERNEL)})
        smi = S.parse_smi("\n".join(ln.split(",", 2)[2] for ln in (_text(out / "gpu.txt") or "").splitlines()
                                    if ln.count(",") >= 8))
        set_ = [r for r in smi if r.get("app_mhz") not in (None, r.get("default_app_mhz"))]
        clocks.append({"dir": out.parent.name, "ok": bool(smi) and not set_, "smi_rows": len(smi),
                       "app_clock_set": len(set_), "mhz": [min(r["mhz"] for r in smi), max(r["mhz"] for r in smi)]
                       if smi else None})
    unread = [f"{s['dir']} {k} {a}" for s in sets for k, by_ in s["rows"]["arms"].items() for a, reps in by_.items()
              for ranks in reps for x in ranks if any(len(r) < 3 or not r[1] for r in x["reps"])]
    whole = [((x["served_macs"] + x["fill_macs_rep"]) / (S.cap(x["mhz"]) * x["ms"] * 1e-3), f"{k} rank {x['rank']}")
             for k, v in by.items() for ranks in v.get("phase", []) for x in ranks]
    g["dram"] = {"ok": bool(dram) and all(d["ok"] for d in dram), "dirs": dram}
    g["clocks"] = {"ok": bool(clocks) and all(c["ok"] for c in clocks) and not unread, "dirs": clocks,
                   "reps_without_clock": unread[:4]}
    wmax = max(whole, default=(None, None))
    g["honest"] = {"ok": bool(whole) and wmax[0] <= 1.0, "whole_of_peak_max": wmax[0], "at": wmax[1]}
    g["ok"] = all(v["ok"] for v in g.values())
    sc = S.tp_score(list(entries.values()))
    return {"scorer": S.V5, "reading": "SCORE-TP" + ("" if S.SCORER == S.V5 else " (a preview)"),
            "outs": [str(o) for o in outs],
            "commits": sorted({_text(o / "commit.txt") or "?" for o in outs}), "facts_from": facts_from,
            "SCORE_TP": sc.get("SCORE_TP") if g["ok"] else None, "score_tp_if_gated": sc.get("SCORE_TP"),
            "SCORE_TP_is": "the measured filler part + the MODELLED served part (tpcomm serves plain int8)",
            "filler_part": sc.get("filler_part"), "served_part_modelled": sc.get("served_part_modelled"),
            "v4_form": sc.get("v4_form"),
            "by_tp": sc["by_tp"], "missing": sc["missing"], "entries": list(entries.values()), "gates": g,
            "filler_entry": entry, "fold": fd, "served_model": S.served_model(), "unchecked": list(TP_UNCHECKED)}


def show_tp(res: dict) -> None:
    g, fe, fd = res["gates"], res["filler_entry"], res["fold"]
    print(f"\n{res['reading']}, {res['scorer']}: SCORE-TP {fmt(res['SCORE_TP'])}"
          + ("" if res["SCORE_TP"] is not None else f" (not scored; {fmt(res['score_tp_if_gated'])} if gated)")
          + f" = measured filler part {fmt(res['filler_part'])} + modelled served part {fmt(res['served_part_modelled'])}; "
          f"v4's form (served plain at {S.CERTIFIED['serving']['fraction']}) {fmt(res['v4_form'])}")
    print(f"  runs at {', '.join(c[:9] for c in res['commits'])}; filler facts from {res['facts_from'] or 'the runs'}")
    print(f"  filler entry {fe['applies']} ({fe['fraction']}, {fe['unfolded']} unfolded); fold {fd['fold']} (SASS "
          f"{fd['signature']}, declared {fd['declared']})" + "".join(f"\n    not {k}: {w}" for k, ws in fe["failed"].items()
                                                              for w in ws[:2]))
    show_served_model(res["served_model"])
    for name, v in g.items():
        if name != "ok":
            print(f"  gate {name}: {'pass' if v['ok'] else 'FAIL'}  "
                  + {"slowdown": lambda: f"worst {fmt(v.get('worst'))} at {v.get('at')}",
                     "filler_paths": lambda: v.get("why") or f"{v.get('callers')} callers",
                     "honest": lambda: f"whole kernel ≤ {fmt(v.get('whole_of_peak_max'))} of peak",
                     "receipts": lambda: f"{v['checked']} checked" + (f", bad: {v['bad']}" if v["bad"] else ""),
                     "pf_gate": lambda: f"{v['ranks']} ranks" + (f": {v['why']}" if v["why"] else "")}.get(name, lambda: "")())
    print("  unchecked: " + "; ".join(res["unchecked"]))
    print("\n| entry | slowdown | coverage | filler part | served part (modelled) | v4 form | step ms base / phase / modelled |\n"
          "|---|---|---|---|---|---|---|")
    for e in res["entries"]:
        print(f"| {e['entry']} | {fmt(e['slowdown'], 4)} | {fmt(e['coverage'])} | {fmt(e['filler_part'])} | "
              f"{fmt(e['served_part_modelled'])} | {fmt(e['coverage_v4_form'])} | {fmt(e['t0_ms'], 2)} / "
              f"{fmt(e['t1_measured_ms'], 2)} / {fmt(e['t1_ms'], 2)} |")
    for t, v in res["by_tp"].items():
        print(f"TP {t}: coverage {fmt(v['coverage'])} = filler {fmt(v['filler_part'])} + modelled served "
              f"{fmt(v['served_part_modelled'])} (v4 form "
              f"{fmt(v['coverage_v4_form'])})")
    if res["missing"]:
        print(f"missing entries: {', '.join(res['missing'])}")


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
        ap.add_argument("--as", dest="version", choices=("v2", "v3", "v4", "v5"),
                        help="score under this version (default the active lock); another's is a preview, not written")
        ap.add_argument("--coins", type=float, default=S.PREFILL_COINS,
                        help="v5: prefill's per-stage coins over its GEMMs (scoring.PREFILL_COINS)")
        a = ap.parse_args(argv[1:])
        res = report(a.out, version={"v2": S.V2, "v3": S.V3, "v4": S.V4, "v5": S.V5}.get(a.version), coins=a.coins)
        print(json.dumps(res, indent=1)) if a.json else show(res)
        return 0
    if argv and argv[0] == "tp-report":
        ap = argparse.ArgumentParser(prog="score.py tp-report", description="SCORE-TP (filler-scorer/v5) of "
                                     "tpcomm.sh out dirs, together")
        ap.add_argument("outs", type=Path, nargs="+")
        src = ap.add_mutually_exclusive_group()
        src.add_argument("--bin", type=Path, help="the tpcomm binary the runs ran (read with nvdisasm)")
        src.add_argument("--nvd", type=Path, help="its nvdisasm -c listing")
        ap.add_argument("--facts-from", help="a label: where --bin/--nvd's build comes from, if not the runs' commit")
        ap.add_argument("--noise", default=None, help=f"the TP filler's declared operands, one of {A.NOISE}")
        ap.add_argument("--fold", default=None, help=f"its declared fold, one of {A.FOLDS} (undeclared: ram)")
        ap.add_argument("--json", action="store_true")
        a = ap.parse_args(argv[1:])
        listings = {"tpcomm": nvdisasm(a.bin)} if a.bin else {"tpcomm": a.nvd.read_text()} if a.nvd else None
        res = tp_report(a.outs, listings, a.noise, a.fold, a.facts_from)
        print(json.dumps(res, indent=1, default=str)) if a.json else show_tp(res)
        return 0
    if argv and argv[0] == "defer-fit":
        ap = argparse.ArgumentParser(prog="score.py defer-fit", description="the deferred pass's calibration "
                                     "(scoring.defer_fit): s and d (or u) fitted to HC-COMM's TP points (r3 and r6 at "
                                     "T 4096, the fit TP scores with), each point's residual, and one GPU's shown "
                                     "share by T and its u")
        ap.add_argument("--T", type=int, default=S.DEFER_T)
        ap.add_argument("--form", choices=("d", "u"), default="d")
        ap.add_argument("--json", action="store_true")
        a = ap.parse_args(argv[1:])
        one_u = {k: S.DEFER_FIT[k] for k in ("T", "tps", "u")} if a.T == S.DEFER_T else None
        res = {**S.defer_fit(a.T, S.DEFER_TP + S.DEFER_TP_R6, a.form), "one_gpu": {t: S.defer_shown_1gpu(t) for t in
                                                                                  S.DEFER_1GPU},
               "one_gpu_u": one_u, "runs": S.DEFER_TP_RUNS}
        print(json.dumps(res, indent=1)) if a.json else show_defer_fit(res)
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
    arm = A.load(a.arm, record=S.SCORER in (S.V3, S.V4, S.V5))
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
