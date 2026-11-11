"""te_core 4.0's measured quantities for the scorer's ledger: each valid run's step traced, measured and ingested again
(coordinator 01:22Z: the refresh runs on te_core 4.0, which scores formalism v2's quantities measured on each record's
traced step; its rows need `ledger ingest --structures`).

    python trace40.py pack --out PACK.tar [--te TE40_WT] [--all]   # here: the runs with no trace yet, and what tracing reads
    python trace40.py node --pack PACK.tar --out DIR [--workers N]  # on a CPU node, under a venv with torch (job40.sh)
    python trace40.py take RUN [--src DIR]                          # here: a node run's out/ into the trace cache
    python trace40.py ledger [--out LEDGER]                         # here: the measured ledger the refresh scores

The node step is the toolsmith's re-trace (art:191e5aa65ab1's helpers/driver.py, art:7bbd4cd32918's ingest): each run is
recorded under fake tensors by `te_core.capture.script` at the first of its dominant update structure's steps whose
forward reads an applied update (of the first three), its circuit measured (`te_core.capture.circuit.measure`), a
sampled head's rows taken at the counter's width and seed-subspace's d at the run's, then `te_core.ledger.ingest` with
that structure on every updated step structure. `ledger` links the scorer's ledger ($TE_LEDGER) and replaces each
traced run's submissions with the node's under their ids, so routes keep their names; a run with no trace keeps its
row, which 4.0's rescore leaves out as not_measured. A run whose trace shows an affected share of 0 or u(1) = 0 beside
a nonzero E is refused, not scored (coordinator 04:40Z): before te_core 4.2, which refuses it itself, its rows are
marked invalid. The cache is per te_core tree, so a new te_core traces again.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
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

HOME = Path.home()
STATE = Path(os.environ.get("TE_SCORER", HOME / ".research" / "te-scorer"))
LEDGER = Path(os.environ.get("TE_LEDGER", HOME / ".research" / "te-ledger"))
MEASURED = Path(os.environ.get("TE_LEDGER_40", HOME / ".research" / "te-ledger-40"))
TE40_WT = Path(os.environ.get("TE40_WT", "/tmp/scorer/te40"))
REPO = Path(os.environ.get("TE_REPO", "/workspace"))
#: the commit whose pinned.json and starts_store.json say which records are traced and from which starts
PIN_REV = os.environ.get("TE_PIN_REV", "origin/cursor/covert-witness-4207")
CW = "benchmarks/covert_witness"
V3_TOKENIZER = "bpe32k"
#: a run whose dominant update structure starts late replays every step before it (a 4-arm shallow run: 3,493 steps);
#: TE_TRACE_TIMEOUT sets it for a node run that retries one that timed out (coordinator 21:47Z: once, at twice it)
TRACE_TIMEOUT = int(os.environ.get("TE_TRACE_TIMEOUT") or 5400)
#: the node step's version: a run that failed under an older one is traced again
CODE = 2
#: the formalism's fields in bits of the update or the gradient, which seed-subspace's d scales
D_SCALED = ("u1_bits", "update_cut_bits", "grad_cut_bits", "state_bits", "delta_grad_bits")
#: plan keys the trace's printed plan and the run's notes both carry but don't compare (plans.py's SKIP)
SKIP = {"args", "steps", "data", "arms", "seconds", "s", "loss", "step", "first_order", "train_steps", "data_steps",
        "flops_per_step", "probe_flops", "time", "wall", "lowest_trained", "spent"}
#: the `invalid` reason `ledger` gives a refused run's records (`refused`), which te_core.frontier leaves out
REFUSED = ("capture: affected share 0 or u(1) = 0 beside a nonzero update, refused, not scored (coordinator 04:40Z: "
           "the trace missed the update's path into the forward)")


def git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def te_tree(te: Path = TE40_WT) -> str:
    """The te_core tree the trace cache is keyed by."""
    return git("rev-parse", f"HEAD:{CW}/te_core", cwd=te)


def cache(te: Path = TE40_WT) -> Path:
    return STATE / "trace40" / te_tree(te)[:12]


# -- here: which runs, and their inputs


def at_pin(name: str, rev: str = PIN_REV) -> dict:
    return json.loads(git("show", f"{rev}:{CW}/{name}"))


def pins(rev: str = PIN_REV) -> list[str]:
    """$TE_V1_PINS, else the commits te_core.panels reads at `rev` (`frontier.pinned(pilot=True)`): the pin's, its
    same_version's and its pilot's."""
    if os.environ.get("TE_V1_PINS"):
        return os.environ["TE_V1_PINS"].split(",")
    p = at_pin("pinned.json", rev)
    return [p["commit"], *p.get("same_version", []), *p.get("pilot", [])]


def pilot_start(rev: str = PIN_REV) -> tuple[set[str], set[str]]:
    """(the pin's commits when it measures data v3, the starts on another tokenizer): te_core.frontier leaves a v3
    record from such a start out as `pilot-start`, so it isn't traced."""
    p = at_pin("pinned.json", rev)
    v3 = {p["commit"], *p.get("same_version", [])} if p.get("data") == "v3" else set()
    starts = at_pin("starts_store.json", rev)["starts"]
    return v3, {n for n, s in starts.items() if (s.get("shape") or {}).get("tokenizer") != V3_TOKENIZER}


def scope() -> list[dict]:
    """Runs in the scorer's ledger whose record the scorer holds valid at a pin (`pins`), each with its first ledger
    row's agent, direction, mechanism, parent and reference (what ingest reads), oldest first; a v3 record from a pilot
    start is left out (`pilot_start`)."""
    measured, (v3, pilot) = set(pins()), pilot_start()
    first = {}
    for line in (LEDGER / "index.jsonl").open():
        r = json.loads(line)
        if r.get("run_id") and (r["run_id"] not in first or r["written_ts"] < first[r["run_id"]]["written_ts"]):
            first[r["run_id"]] = r
    out = {}
    for line in (STATE / "index.jsonl").open():
        x = json.loads(line)
        if not (x.get("valid") and x.get("measurement_commit") in measured and x.get("run") in first
                and x.get("island_tree")) or (x["measurement_commit"] in v3 and x.get("start") in pilot):
            continue
        o = first[x["run"]]
        out[x["run"]] = {"run_id": x["run"], "record": x["id"].removeprefix("art:"), "agent": o["agent"],
                         "direction": o["direction"], "mechanism": o["mechanism"], "parent": o.get("parent", ""),
                         "reference": bool(o.get("reference")), "start": o.get("start", ""), "island": x["island"],
                         "program": x["program"], "island_tree": x["island_tree"], "meas": x["measurement_commit"],
                         "written_ts": o["written_ts"]}
    return sorted(out.values(), key=lambda r: r["written_ts"])


def done(c: Path) -> set[str]:
    """Runs traced, or failed under this CODE."""
    traced = json.loads((c / "traced.json").read_text()) if (c / "traced.json").exists() else {}
    return {k for k, v in traced.items() if not v.get("error") or v.get("code") == CODE}


def archive(rev: str, dest: Path, sub: str = "", strip: int = 0, exclude: str = "") -> None:
    dest.mkdir(parents=True, exist_ok=True)
    a = subprocess.run(["git", "archive", "--format=tar", rev, *([sub] if sub else [])], cwd=REPO, check=True,
                       capture_output=True).stdout
    cmd = ["tar", "-x", "-C", str(dest), f"--strip-components={strip}"]
    if exclude:
        cmd.append(f"--exclude={exclude}")
    subprocess.run(cmd, input=a, check=True)


def starts_dir(dest: Path, names: set[str], rev: str = PIN_REV) -> None:
    """NAME.start.json (what te_core.capture.script's --starts reads) for each start in `names`, by its start.json's
    sha256 in starts_store.json at `rev`: the scorer's copy where it holds one, else from the evidence store, which
    takes the scorer (it imports only beside the te_core its refresh pins)."""
    starts = at_pin("starts_store.json", rev)["starts"]
    dest.mkdir(parents=True)
    for n in sorted(names):
        sha = starts[n]["files"]["start.json"]
        held = STATE / "starts-store" / f"{sha}.json"
        if not held.exists():
            import scorer as S

            S.blob(S.store_remote(), sha, held)
        if hashlib.sha256(held.read_bytes()).hexdigest() != sha:
            raise SystemExit(f"{held} is not the start {n}'s start.json ({sha})")
        shutil.copy(held, dest / f"{n}.start.json")


def pack(out: Path, te: Path, everything: bool, only: set[str] | None = None, skip: set[str] = frozenset()) -> list[dict]:
    """PACK.tar for the runs in `scope` with no trace in this te_core's cache (`only`: those runs or islands): runs.json,
    each run's run.json (gzip) and meta.json, the measurement trees (covert_witness without te_core), the island trees,
    te_core and the starts."""
    c = cache(te)
    have = set() if everything else done(c)
    runs = [r for r in scope() if r["run_id"] not in have | skip and (not only or r["run_id"] in only or r["island"] in only)]
    if not runs:
        return []
    d = Path(tempfile.mkdtemp(prefix="trace40-pack-"))
    try:
        for r in runs:
            src = STATE / "records" / r["record"]
            dst = d / "runs" / r["run_id"]
            dst.mkdir(parents=True)
            with (src / "run" / "run.json").open("rb") as f, gzip.open(dst / "run.json.gz", "wb", compresslevel=1) as g:
                shutil.copyfileobj(f, g)
            if (src / "meta.json").exists():
                shutil.copy(src / "meta.json", dst / "meta.json")
        for m in sorted({r["meas"] for r in runs}):
            archive(m, d / "meas" / m[:12], CW, strip=2, exclude="te_core")
        for t in sorted({r["island_tree"] for r in runs}):
            archive(t, d / "islands" / t[:12])
        archive(git("rev-parse", "HEAD", cwd=te), d / "te", f"{CW}/te_core", strip=2)
        starts_dir(d / "starts", {r["start"] for r in runs})
        (d / "runs.json").write_text(json.dumps(runs, indent=1))
        (d / "manifest.json").write_text(json.dumps({"te_commit": git("rev-parse", "HEAD", cwd=te),
                                                     "te_tree": te_tree(te), "runs": len(runs)}))
        with tarfile.open(out, "w") as tf:
            tf.add(d, arcname=".")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return runs


# -- on the node: trace, measure, ingest (the toolsmith's driver and ingest)


def sig(run: dict, s: dict) -> tuple:
    st = run["structures"].get(s.get("struct")) or {}
    return tuple(sorted((r[0], r[1], r[4]) for r in st.get("updated", [])))


def regions_of(run: dict, n: int = 3) -> list[int]:
    """The first `n` update steps, after the run's first update, whose U is the run's dominant U (by updated MACs)."""
    steps = run["steps"]
    w = collections.Counter()
    for s in steps:
        g = sig(run, s)
        if g:
            w[g] += sum(r[2] for r in run["structures"][s["struct"]]["updated"])
    if not w:
        return []
    dom = max(w, key=w.get)
    upd = [s["step"] for s in steps if (s.get("by") or {}).get("other/param", 0) > 0]
    if not upd:
        return []
    return [s["step"] for s in steps if s["step"] > upd[0] and sig(run, s) == dom
            and (s.get("by") or {}).get("other/param", 0) > 0][:n]


def run_key(r: dict, args: list[str]) -> str:
    args = list(args)
    if "--out" in args:
        args[args.index("--out") + 1] = "OUT"
    return hashlib.sha256(json.dumps([r["meas"], r["island_tree"], r["program"], args]).encode()).hexdigest()[:16]


def flat(d: dict, pre: str = ""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, f"{pre}{k}.")
        elif isinstance(v, (int, float, str, bool)) or v is None:
            yield pre + k, v


def plan_differ(meta: Path, log: Path) -> dict:
    """The plan the program printed under fake tensors beside the one the run recorded (meta.json's notes), on the keys
    both carry, where they differ (plans.py): a program that plans from values plans from the recorder's stand-ins."""
    notes, printed = {}, {}
    m = json.loads(meta.read_text()) if meta.exists() else {}
    for k, v in flat({k: v for k, v in (m.get("notes") or {}).items() if k not in SKIP}):
        notes.setdefault(k, v)
        if k.rsplit(".", 1)[-1] not in SKIP:
            notes.setdefault(k.rsplit(".", 1)[-1], v)
    for line in log.read_text(errors="ignore").splitlines() if log.exists() else []:
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict):
            for k, v in flat({k: v for k, v in d.items() if k not in SKIP}):
                printed.setdefault(k, v)
    return {k: (printed[k], notes[k]) for k in sorted(set(notes) & set(printed)) if printed[k] != notes[k]
            and not (isinstance(notes[k], (int, float)) and isinstance(printed[k], (int, float))
                     and abs(printed[k] - notes[k]) <= 0.01 * max(abs(notes[k]), 1))}


def subspace_at_run_d(res: dict, differ: dict) -> tuple[dict, dict | None]:
    """seed-subspace's update is its step's d coefficients, and d is planned from the clock and the cap, so the trace's d
    is the fake clock's: its cut and the formalism's d-sized quantities are scaled to the d the run recorded."""
    d = differ.get("d")
    if not d or len(res["structure"]["updated"]) == 0:
        return res, None
    traced_d, run_d = d
    assert len(res["cut"]) == 1 and res["cut"][0][1] == res["cut_bits"] and res["cut_bits"] % traced_d == 0, res["cut"]
    scale = run_d / traced_d
    f = dict(res["formalism"])
    scaled = {k: f[k] for k in D_SCALED if f.get(k) is not None and f[k] % traced_d == 0}
    for k in scaled:
        f[k] = f[k] * scale
    if "u1_bits" in scaled:
        f["u_in"] = [[x, b * scale] for x, b in f["u_in"]]
    info = {"traced_d": traced_d, "run_d": run_d, "scaled": scaled, "kept": [k for k in D_SCALED if k not in scaled]}
    cut = res["cut_bits"] // traced_d * run_d
    st = dict(res["structure"], cut_bits=cut, beta=res["structure"]["beta"] * scale, formalism=f)
    return dict(res, structure=st, cut_bits=cut, beta=st["beta"], formalism=f), info


def head_at_largest(res: dict, run: dict, d_model: int) -> tuple[dict, dict | None]:
    """A sampled head's width is ``unique(targets)``, which the recorder takes at its largest (the micro-batch's tokens):
    its rows (the trace's widest) are scaled so the trace's forward GEMMs are the counter's. The formalism's quantities
    are the widest head's (its update reads the head's whole table either way)."""
    by = next(s for s in run["steps"] if s["step"] == res["step"]).get("by") or {}
    rec, other = sum(v for k, v in by.items() if k.startswith("fwd/")), by.get("other/param", 0)
    fwd = res["forward_gemm_macs"]
    for total in (rec, rec + other):
        if abs(fwd - total) <= 0.01 * total:
            return dict(res, recorded_forward_macs=total), None
    res = dict(res, recorded_forward_macs=rec)
    if fwd < rec:
        # no head the recorder widened: the trace's forward is short of the counter's, and its rows stay as traced
        return res, {"unscaled": "the trace's forward GEMMs are below the counter's", "fwd_ratio": fwd / rec}
    rows = res["structure"]["updated"]
    top = max(r[0] for r in rows)
    head = sum(r[0] for r in rows if r[0] == top) * res["tokens"]
    s = 1 - (fwd - rec) / head
    assert 0 < s < 1, (fwd, rec, head)
    new = [[r[0] * s, r[1], r[2] * s, r[3], r[4]] if r[0] == top else r for r in rows]
    per_token = sum(r[0] for r in new)
    info = {"traced_width": top // d_model, "width": round(top // d_model * s), "scale": s, "fwd_ratio_before": fwd / rec}
    st = dict(res["structure"], updated=new, beta=res["cut_bits"] / per_token)
    return dict(res, structure=st, beta=st["beta"], updated_macs=round(per_token * res["tokens"]),
                forward_gemm_macs=fwd - (1 - s) * head), info


def workdir(root: Path, r: dict) -> Path:
    """The measurement tree with this te_core and the run's island tree: the program's directory as the harness ran it."""
    d = root / "work" / f"{r['meas'][:12]}-{r['island_tree'][:12]}-{r['island']}"
    if d.exists():
        return d
    tmp = d.with_name(d.name + ".part")
    shutil.copytree(root / "pack" / "meas" / r["meas"][:12], tmp, symlinks=True, copy_function=os.link)
    os.symlink(root / "pack" / "te" / "te_core", tmp / "te_core")
    isl = tmp / "islands" / r["island"]
    if isl.is_symlink() or isl.is_file():
        isl.unlink()
    elif isl.exists():
        shutil.rmtree(isl)
    shutil.copytree(root / "pack" / "islands" / r["island_tree"][:12], isl, symlinks=True, copy_function=os.link)
    tmp.rename(d)
    return d


def trace(root: Path, r: dict, run_json: Path, args: list[str], step: int, k: str) -> tuple[dict, Path, Path]:
    wd = workdir(root, r)
    args = list(args)
    if "--out" in args:
        args[args.index("--out") + 1] = str(root / "out" / r["run_id"])
    (root / "out" / r["run_id"]).mkdir(parents=True, exist_ok=True)
    name = f"{r['run_id']}-{k}-{step}"
    out, log, circ = root / "traces" / f"{name}.json", root / "logs" / f"{name}.log", root / "circuits" / f"{name}.npz"
    for p in (out, log, circ):
        p.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "te_core.capture.script", "--script", f"islands/{r['island']}/{r['program']}",
           "--run", str(run_json), "--step", str(step), "--starts", str(root / "pack" / "starts"), "--out",
           str(out) + ".tmp", "--log", str(log), "--circuit", str(circ), "--", *args]
    env = dict(os.environ, PYTHONPATH=str(wd), AGENT_GUARD_LOCAL_OK="1", CW_DEVICE="cpu", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    try:
        p = subprocess.run(cmd, cwd=wd, env=env, capture_output=True, text=True, timeout=TRACE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"error": f"trace timed out after {TRACE_TIMEOUT} s"}, log, circ
    if p.returncode:
        err = (p.stderr or "").strip().splitlines() or [f"killed with no output (rc {p.returncode})"]
        return {"error": "\n".join(err[-25:])}, log, circ
    os.replace(str(out) + ".tmp", out)
    return json.loads(out.read_text()), log, circ


def one(root: str, r: dict, out: str) -> dict:
    """Trace, measure and ingest one run; its records go to out/records/RUN.jsonl.gz as it ends, so a node run that
    times out keeps the runs it finished."""
    root, out = Path(root), Path(out)
    sys.path.insert(0, str(root / "pack" / "te"))
    from te_core.capture.circuit import StepCircuit, measure
    from te_core.ledger import Ledger, ingest

    rid, t0 = r["run_id"], time.time()
    rd = root / "runs" / rid
    rd.mkdir(parents=True, exist_ok=True)
    with gzip.open(root / "pack" / "runs" / rid / "run.json.gz", "rb") as f, (rd / "run.json").open("wb") as g:
        shutil.copyfileobj(f, g)
    run = json.loads((rd / "run.json").read_text())
    summary = {"island": r["island"], "agent": r["agent"], "code": CODE}
    try:
        tree = ((run.get("evaluation") or {}).get("island") or {}).get("tree")
        if tree and tree != r["island_tree"]:
            raise ValueError(f"run.json's island tree {tree} is not the record's {r['island_tree']}")
        args = list(run.get("script_args") or [])
        k = run_key(r, args)
        res, step, log = {"error": "no update step with the dominant U"}, None, None
        # a step whose forward reads no applied update yet (a first zeroth-order step: its own perturbations of θ only)
        # has no update-dependent MAC, so the next step of the same structure represents the run
        for step in regions_of(run):
            res, log, circ = trace(root, r, rd / "run.json", args, step, k)
            if res.get("updated_macs") != 0:
                break
        summary |= {"step": step, "key": k, "seconds": res.get("seconds")}
        if "structure" not in res:
            return summary | {"error": (res.get("error") or "no structure")[-2000:]}
        res = dict(res, step=step, formalism=measure(StepCircuit.load(circ)))
        res["structure"] = dict(res["structure"], formalism=res["formalism"])
        cfg = json.loads((root / "pack" / "starts" / f"{r['start']}.start.json").read_text())["config"]
        res, heads = head_at_largest(res, run, cfg["d_model"])
        differ = plan_differ(root / "pack" / "runs" / rid / "meta.json", log)
        res, sub = subspace_at_run_d(res, differ) if r["island"] == "seed-subspace" else (res, None)
        structures = {s["struct"]: res["structure"] for s in run["steps"]
                      if (run["structures"].get(s.get("struct")) or {}).get("updated")}
        recs = ingest(rd, Ledger(root / "ledgers" / rid, r["agent"]), direction=r["direction"],
                      mechanism=r["mechanism"], parent=r["parent"], run_id=rid, reference=r["reference"],
                      structures=structures)
        (out / "records").mkdir(parents=True, exist_ok=True)
        with gzip.open(out / "records" / f"{rid}.jsonl.gz", "wt") as f:
            f.writelines(json.dumps(x, sort_keys=True) + "\n" for x in recs)
        f = res["formalism"]
        return summary | {"rows": len(recs), "heads": heads, "subspace": sub, "plan_differ": differ,
                          "formalism": {x: f.get(x) for x in ("phi", "affected_share", "token_factor", "a_bits",
                                                              "u1_bits", "tail_share", "delta_grad_bits")},
                          "wall": round(time.time() - t0, 1)}
    except Exception:
        return summary | {"error": traceback.format_exc()[-2000:]}
    finally:
        (rd / "run.json").unlink(missing_ok=True)
        shutil.rmtree(root / "ledgers" / rid, ignore_errors=True)


def node(pack_tar: Path, out: Path, workers: int) -> int:
    root = Path(tempfile.mkdtemp(prefix="trace40-", dir=os.environ.get("TMPDIR") or None))
    try:
        with tarfile.open(pack_tar) as tf:
            tf.extractall(root / "pack", filter="fully_trusted")
        runs = json.loads((root / "pack" / "runs.json").read_text())
        for r in runs:
            workdir(root, r)
        # the largest record first, so the long traces don't trail
        runs.sort(key=lambda r: -(root / "pack" / "runs" / r["run_id"] / "run.json.gz").stat().st_size)
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text((root / "pack" / "manifest.json").read_text())
        results = {}
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(one, str(root), r, str(out)): r["run_id"] for r in runs}
            for fut in as_completed(futs):
                rid = futs[fut]
                try:
                    results[rid] = fut.result()
                except Exception:
                    results[rid] = {"error": traceback.format_exc()[-2000:]}
                x = results[rid]
                print(rid, x.get("island"), x.get("step"), x.get("rows"), x.get("seconds"),
                      (x.get("error") or "").strip().splitlines()[-1:] or "", flush=True)
                (out / "traced.json").write_text(json.dumps(results, indent=1, sort_keys=True, default=str))
        with tarfile.open(out / "logs.tar.gz", "w:gz") as tf:
            if (root / "logs").exists():
                tf.add(root / "logs", arcname="logs")
        bad = {k: v for k, v in results.items() if v.get("error")}
        print(f"{len(results) - len(bad)} of {len(results)} runs traced and ingested; {len(bad)} not:",
              collections.Counter(v.get("island") for v in bad.values()).most_common(), flush=True)
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


# -- here: the node's output into the cache, and the measured ledger


def take(run: str, te: Path, src: Path | None = None) -> dict:
    """A node run's out/ (fetched to ~/.research/runs/RUN, or `src` where it is here already) into this te_core's cache:
    records and traced.json."""
    if src is None:
        # a fetch the store fails for a moment (exit 4, 19:19Z) would send the shard's runs to be traced again
        for wait in (0, 15, 60):
            time.sleep(wait)
            r = subprocess.run(["research", "fetch", run, "--all"], capture_output=True, check=False)
            if not r.returncode:
                break
        r.check_returncode()
        src = HOME / ".research" / "runs" / run / "out"
    man = json.loads((src / "manifest.json").read_text())
    c = cache(te)
    if man["te_tree"] != te_tree(te):
        raise SystemExit(f"{run} traced te_core tree {man['te_tree']}, not this checkout's {te_tree(te)}")
    (c / "records").mkdir(parents=True, exist_ok=True)
    for p in (src / "records").glob("*.jsonl.gz") if (src / "records").exists() else []:
        shutil.copy(p, c / "records" / p.name)
    traced = json.loads((c / "traced.json").read_text()) if (c / "traced.json").exists() else {}
    new = json.loads((src / "traced.json").read_text())
    traced |= {k: v | {"by": run} for k, v in new.items()}
    (c / "traced.json").write_text(json.dumps(traced, indent=1, sort_keys=True, default=str))
    return new


def refused(c: Path) -> set[str]:
    """Traced runs whose step shows an affected share of 0 or u(1) = 0 while the run's E is nonzero."""
    traced = json.loads((c / "traced.json").read_text()) if (c / "traced.json").exists() else {}
    e = {}
    for line in (STATE / "index.jsonl").open():
        x = json.loads(line)
        if x.get("source") == "launch.py" and x.get("run"):
            e[x["run"]] = x.get("E_bytes") or 0
    return {k for k, v in traced.items() if not v.get("error") and e.get(k, 0) > 0
            and 0 in ((v.get("formalism") or {}).get("affected_share"), (v.get("formalism") or {}).get("u1_bits"))}


def traced_rows(te: Path, over: list[Path] = ()) -> tuple[dict, set[str]]:
    """Each traced (run, checkpoint)'s first record, from the newest of `te` and `over` (oldest first) whose cache
    traced the run, and the refused runs (`refused`). A run's records file is read from that one cache only, and its
    checkpoints share one copy of the run's trace: at 23:18Z every layer's records, each with its own copy, took the
    build past this VM's 15 GB."""
    newest, bad = {}, set()
    for t in (te, *over):
        c, runs = cache(t), set()
        for p in sorted((c / "records").glob("*.jsonl.gz")) if (c / "records").exists() else []:
            with gzip.open(p, "rt") as f:
                if f.readline():
                    runs.add(p.name.removesuffix(".jsonl.gz"))
                    newest[p.name.removesuffix(".jsonl.gz")] = p
        bad = (bad - runs) | (set() if "\ndef refusal(" in git("show", f"HEAD:{CW}/te_core/formalism.py", cwd=t)
                              else refused(c))
    traced = {}
    for run, p in newest.items():
        shared = []
        with gzip.open(p, "rt") as f:
            for line in f:
                x = json.loads(line)
                if x["run_id"] != run:
                    raise SystemExit(f"{p} holds {x['run_id']}'s records, not {run}'s")
                fm = x.get("formalism")
                tr = fm.get("trace") if isinstance(fm, dict) else None
                if isinstance(tr, dict):
                    same = next((s for s in shared if s == tr), None)
                    if same is None:
                        shared.append(tr)
                    else:
                        fm["trace"] = same
                traced.setdefault((run, x["checkpoint_step"]), x)
    return traced, bad


def ledger(out: Path, te: Path, over: list[Path] = ()) -> dict:
    """The scorer's ledger with each traced run's submissions replaced by the node's (on the original, so a field the
    harness wrote and the node's re-ingest lacks, such as v3's data and model, stays; under the ledger's id, agent and
    write time), a refused run's (`refused`) marked invalid with REFUSED where te_core refuses none itself (4.2's
    `formalism.refusal`), every other file linked; written to `out` whole, then swapped in. With `over`, newer
    captures' checkouts, oldest first, a run a newer one's cache traced takes every checkpoint from there, so each
    run's rows are one capture's (`traced_rows`)."""
    traced, bad = traced_rows(te, over)
    by_id = {}
    for line in (LEDGER / "index.jsonl").open():
        r = json.loads(line)
        by_id[r["id"]] = (r.get("run_id"), r.get("checkpoint_step"))
    tmp = out.with_name(out.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    n = collections.Counter()
    used = set()
    for p in sorted((LEDGER / "records").glob("*/*")):
        dst = tmp / "records" / p.parent.name / p.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        key = by_id.get(p.stem) if p.suffix == ".json" else None
        if key in traced:
            old = json.loads(p.read_text())
            new = old | traced[key] | {k: old[k] for k in ("id", "agent", "written_at", "written_ts")}
            if key[0] in bad:
                new["invalid"] = [*(new.get("invalid") or []), REFUSED]
                n["refused"] += 1
            dst.write_text(json.dumps(new, sort_keys=True) + "\n")
            used.add(key)
            n["replaced"] += 1
        else:
            os.link(p, dst)
            n["linked" if p.suffix == ".json" else "other"] += 1
    n["traced rows with no ledger row"] = len(set(traced) - used)
    n["refused_runs"] = sorted(bad)
    old = out.with_name(out.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if out.exists():
        out.rename(old)
    tmp.rename(out)
    shutil.rmtree(old, ignore_errors=True)
    return dict(n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--te", type=Path, default=TE40_WT)
    p.add_argument("--all", action="store_true", help="every run in scope, traced or not")
    p.add_argument("--only", default="", help="comma-separated run ids or islands")
    p.add_argument("--skip", default="", help="comma-separated run ids (another node run has them)")
    n = sub.add_parser("node")
    n.add_argument("--pack", type=Path, required=True)
    n.add_argument("--out", type=Path, required=True)
    n.add_argument("--workers", type=int, default=64)
    t = sub.add_parser("take")
    t.add_argument("run")
    t.add_argument("--te", type=Path, default=TE40_WT)
    t.add_argument("--src", type=Path, default=None, help="the run's out/ here (default: `research fetch RUN --all`'s)")
    led = sub.add_parser("ledger")
    led.add_argument("--out", type=Path, default=MEASURED)
    led.add_argument("--te", type=Path, default=TE40_WT)
    led.add_argument("--over", type=Path, action="append", default=[],
                     help="a newer capture's checkout, whose traced runs replace --te's and earlier --over's; repeatable")
    a = ap.parse_args()
    if a.cmd == "pack":
        runs = pack(a.out, a.te, a.all, {x for x in a.only.split(",") if x} or None, {x for x in a.skip.split(",") if x})
        print(json.dumps({"runs": len(runs), "pack": str(a.out) if runs else None,
                          "bytes": a.out.stat().st_size if runs else 0}))
    elif a.cmd == "node":
        return node(a.pack, a.out, a.workers)
    elif a.cmd == "take":
        new = take(a.run, a.te, a.src)
        print(json.dumps({"runs": len(new), "errors": sum(bool(v.get("error")) for v in new.values())}))
    else:
        print(json.dumps(ledger(a.out, a.te, a.over)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
