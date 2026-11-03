"""The standard-workloads panel's traces: chosen steps of published records, recorded by te_core.capture.script under
this checkout's te_core and measured (te_core.capture.circuit.measure), as the scorer's re-trace does for its records.

    python trace.py pack --out PACK.tar --plan PLAN.json          # here: what each trace reads, into one tar
    python trace.py node --pack PACK.tar --out DIR [--workers N]   # on a CPU node (job.sh)

PLAN.json is a list of {"run": RUN, "art": ART, "steps": [STEP, ...]}: each record's `run/run.json`, `island.tar` and
`candidate.json` come from the evidence store, its measurement commit's covert_witness tree (without te_core) and its
start's `start.json` (the pin's starts_store.json) with them, and this checkout's te_core beside. The node writes
DIR/traces/RUN-STEP.json for each step: the capture's output (te_core.capture.script's CLI fields) and `measured`, the
circuit's quantities.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CW = "benchmarks/covert_witness"
REPO = Path(__file__).resolve().parents[4]
#: a step late in a long record replays every step before it
TRACE_TIMEOUT = 3 * 3600


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def archive(rev: str, sub: str, dest: Path, strip: int, exclude: str = "") -> None:
    dest.mkdir(parents=True, exist_ok=True)
    a = subprocess.run(["git", "archive", "--format=tar", rev, sub], cwd=REPO, check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dest), f"--strip-components={strip}",
                    *([f"--exclude={exclude}"] if exclude else [])], input=a, check=True)


def fetch(art: str, dest: Path, *paths: str) -> None:
    args = [x for p in paths for x in ("--path", p)]
    subprocess.run(["research", "data", "fetch", art, "--to", str(dest), *args], check=True, capture_output=True)


def pack(plan: list[dict], out: Path) -> dict:
    d = Path(tempfile.mkdtemp(prefix="sw-trace-pack-"))
    try:
        runs, metas, starts = [], set(), set()
        for item in plan:
            rd = d / "runs" / item["run"]
            fetch(item["art"], rd, "run/run.json", "island.tar", "candidate.json", "island.json")
            cand = json.loads((rd / "candidate.json").read_text())
            isl = json.loads((rd / "island.json").read_text())
            runs.append({"run": item["run"], "art": item["art"], "steps": item["steps"], "start": cand["start"],
                         "meas": cand["measurement_commit"], "island": isl["name"], "program": isl["program"]})
            metas.add(cand["measurement_commit"])
            starts.add((cand["start"], cand["measurement_commit"]))
        for m in metas:
            archive(m, CW, d / "meas" / m[:12], 2, exclude="te_core")
        archive("HEAD", f"{CW}/te_core", d / "te", 2)
        for name, m in starts:
            store = json.loads(git("show", f"{m}:{CW}/starts_store.json"))["starts"][name]
            tmp = d / "starts-fetch" / name
            fetch(store["art"], tmp, f"{store.get('path', 'starts')}/{name}/start.json")
            (d / "starts").mkdir(exist_ok=True)
            shutil.copy(next(tmp.rglob("start.json")), d / "starts" / f"{name}.start.json")
        shutil.rmtree(d / "starts-fetch", ignore_errors=True)
        man = {"te_commit": git("rev-parse", "HEAD"), "te_tree": git("rev-parse", f"HEAD:{CW}/te_core"), "runs": runs}
        (d / "manifest.json").write_text(json.dumps(man, indent=1))
        with tarfile.open(out, "w") as tf:
            tf.add(d, arcname=".")
        return man
    finally:
        shutil.rmtree(d, ignore_errors=True)


def workdir(root: Path, r: dict) -> Path:
    """The measurement tree with this te_core and the record's island, as the harness ran the program."""
    w = root / "work" / r["run"]
    if w.exists():
        return w
    shutil.copytree(root / "pack" / "meas" / r["meas"][:12], w, symlinks=True)
    os.symlink(root / "pack" / "te" / "te_core", w / "te_core")
    isl = w / "islands" / r["island"]
    shutil.rmtree(isl, ignore_errors=True)
    isl.mkdir(parents=True)
    with tarfile.open(next((root / "pack" / "runs" / r["run"]).rglob("island.tar"))) as tf:
        tf.extractall(isl, filter="data")
    return w


def one(root: str, r: dict, step: int, out: str) -> dict:
    root, out = Path(root), Path(out)
    sys.path.insert(0, str(root / "pack" / "te"))
    from te_core.capture.circuit import StepCircuit, measure

    t0 = time.time()
    name = f"{r['run']}-{step}"
    run_json = next((root / "pack" / "runs" / r["run"]).rglob("run.json"))
    run = json.loads(run_json.read_text())
    args = list(run.get("script_args") or [])
    if "--out" in args:
        args[args.index("--out") + 1] = str(root / "sub" / name)
    wd = workdir(root, r)
    tr, log, circ = root / "t" / f"{name}.json", out / "logs" / f"{name}.log", root / "t" / f"{name}.npz"
    for p in (tr, log):
        p.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "te_core.capture.script", "--script", f"islands/{r['island']}/{r['program']}",
           "--run", str(run_json), "--step", str(step), "--starts", str(root / "pack" / "starts"), "--out", str(tr),
           "--log", str(log), "--circuit", str(circ), "--", *args]
    env = dict(os.environ, PYTHONPATH=str(wd), AGENT_GUARD_LOCAL_OK="1", CW_DEVICE="cpu", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    res = {"run": r["run"], "art": r["art"], "step": step, "args": args}
    try:
        p = subprocess.run(cmd, cwd=wd, env=env, capture_output=True, text=True, timeout=TRACE_TIMEOUT)
        if p.returncode:
            res["error"] = "\n".join((p.stderr or f"rc {p.returncode}").strip().splitlines()[-30:])
        else:
            res["trace"] = json.loads(tr.read_text())
            res["measured"] = measure(StepCircuit.load(circ))
    except subprocess.TimeoutExpired:
        res["error"] = f"trace timed out after {TRACE_TIMEOUT} s"
    except Exception:  # noqa: BLE001 - every trace's failure is recorded with the others
        res["error"] = traceback.format_exc()[-3000:]
    res["wall_s"] = round(time.time() - t0, 1)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    (out / "traces" / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
    circ.unlink(missing_ok=True)
    return {k: res.get(k) for k in ("run", "step", "wall_s")} | {"error": (res.get("error") or "")[-300:]}


def node(pack_tar: Path, out: Path, workers: int) -> int:
    root = Path(tempfile.mkdtemp(prefix="sw-trace-", dir=os.environ.get("TMPDIR") or None))
    try:
        with tarfile.open(pack_tar) as tf:
            tf.extractall(root / "pack", filter="fully_trusted")
        man = json.loads((root / "pack" / "manifest.json").read_text())
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps(man, indent=1))
        for r in man["runs"]:
            workdir(root, r)
        jobs = [(r, s) for r in man["runs"] for s in r["steps"]]
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            futs = [ex.submit(one, str(root), r, s, str(out)) for r, s in jobs]
            for fut in as_completed(futs):
                print(json.dumps(fut.result()), flush=True)
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    n = sub.add_parser("node")
    n.add_argument("--pack", type=Path, required=True)
    n.add_argument("--out", type=Path, required=True)
    n.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.cmd == "pack":
        print(json.dumps(pack(json.loads(a.plan.read_text()), a.out)["runs"]))
        return 0
    return node(a.pack, a.out, a.workers)


if __name__ == "__main__":
    raise SystemExit(main())
