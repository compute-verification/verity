"""training/algorithm-policy rebuilt with every rotation product priced as the GEMM it ran as, and what moved.

    python scorer/replot.py node --state K --inputs IN --scratch S --out OUT     # on the scorer's node
    python scorer/replot.py diff --out FILE LABEL=PANEL.json LABEL=PANEL.json [...]

Run from benchmarks/covert_witness with the scorer's interpreter (K/venv's). te_core.rotations names a rotating run's
rotation products `tf32`; te_core.ledger's ingest applies it to a new record. The ledger's rows were ingested before
it, so `node` builds the panel again from the scorer's state, changing nothing in K:

1. the price: te_core.gates_main, te_core's gate table against main's (IN/main/training_slowdown), statement for
   statement. The table's own check (`python -m te_core.gates --check`) compiles the Definitions, which the scorer's
   interpreter doesn't hold: main's check runs it on main's copy;
2. the measured ledger as the last refresh built it (trace40.py's `ledger`, on a copy of K/ledger, from the capture
   layers the last refresh's meta.json names, each checked by its te_core tree);
3. the control: te_core.algorithm_policy on that ledger (v1cycle's POLICY_IO, the last refresh's pins, ladder fit,
   gate factors and failed traces). It reproduces the last refresh's tables where nothing but te_core's VERSION moved,
   so every later difference is the rotations';
4. a copy of the measured ledger with each rotating run's rows as ingest writes them now (`rotations.repriced`, from
   the run's record in K/state): which runs rotate is read from every row's program and stored arguments, and a record
   te_core.rotations refuses (`Unread`) keeps its rows, listed with why;
5. the builder on that copy, twice: as te_core reads it (the price, and te_core.tiles' 32-bit operands for a row
   named `tf32`), and with the tiles' width for such a row held at a bf16 row's (`PRICE_ONLY`), which no panel
   publishes: the second is what the price alone moves;
6. the finals table from its own inputs (islands/integrator/finals_table.py: the reruns' records fetched from the
   store and ingested afresh, so the hook names their rotations), and the overlay (plot2_final.py) of each finals
   table over each search table: the served pair, the control, the price alone, and the one to publish (OUT/overlay);
7. each panel's rows, cell by cell, across those builds (`compare`), under OUT/diff.

`diff` is `compare` on stored panels.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CW = HERE.parent
sys.path.insert(0, str(CW))
#: the builder with the tiles reading a `tf32` row's operands at a bf16 row's width: the price's effect alone
PRICE_ONLY = 'from te_core import tiles\ntiles.WIDTH = dict(tiles.WIDTH) | {"gemm/tf32": tiles.WIDTH["gemm/bf16"]}\n'
#: the uncut tables a build of te_core.algorithm_policy writes
TABLES = ("algorithm-policy.json", "frontier-fit.json", "frontier-points.json", "coverage.json", "refused.json")
#: a panel's fields that say when and by what it was built, never compared as content
STAMPS = ("updated_at", "source")
#: the column that carries the builder's VERSION, which every rebuilt row changes
VERSION_COLUMN = "te_core"
T0 = time.monotonic()


def say(text: str) -> None:
    print(f"{datetime.now(UTC).strftime('%H:%M:%SZ')} +{time.monotonic() - T0:6.0f}s {text}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def run(cmd: list, log: Path, cwd: Path = CW, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """`cmd` with its output in `log`; a failure raises with the log's tail."""
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, capture_output=True, text=True, check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(r.stdout + r.stderr)
    if check and r.returncode:
        raise SystemExit(f"replot: {' '.join(str(c) for c in cmd[:4])} ... exited {r.returncode}: "
                         f"{(r.stderr or r.stdout).strip()[-1500:]}")
    return r


def dump(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")


# -- panels, cell by cell


def run_of(record: str) -> str:
    """The run of a row's record cell (`RUN@STEP`, the overlay's with its split after a semicolon)."""
    return str(record).split(";")[0].rsplit("@", 1)[0]


def table(panel: dict) -> dict[tuple, dict]:
    """A panel's rows by (run, gain target), or by its first column where it has neither; a key met twice is refused."""
    cols = panel["columns"]
    out = {}
    for r in panel["rows"]:
        d = dict(zip(cols, canon(r)))
        key = (run_of(d["record"]), d.get("gain_target")) if "record" in d else (str(r[0]), len(out))
        if key in out:
            raise SystemExit(f"replot: two rows of {panel.get('id')} are {key}")
        out[key] = d
    return out


def canon(v):
    """A JSON value with every whole number a float: the route serves a float past 2^53 as its digits, an int."""
    if isinstance(v, int) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, list):
        return [canon(x) for x in v]
    if isinstance(v, dict):
        return {k: canon(x) for k, x in v.items()}
    return v


def change(a, b) -> float | None:
    """b/a - 1 for two numbers, None otherwise."""
    def num(x) -> bool:
        return isinstance(x, (int, float)) and not isinstance(x, bool)
    return b / a - 1 if num(a) and num(b) and a else None


def compare(panels: dict[str, dict], rotating: set[str] = frozenset(), base: str | None = None,
            last: str | None = None) -> dict:
    """The panels' rows cell by cell, in the order given. Per row whose cells differ between any two panels: its key,
    whether its run rotates, and each differing column's value in every panel with its change from `base` (the first
    panel unless named) to `last` (the last unless named). `VERSION_COLUMN` is counted, never listed. A row missing
    from a panel is listed under `only`."""
    names = list(panels)
    base, last = base or names[0], last or names[-1]
    tabs = {n: table(p) for n, p in panels.items()}
    cols = [c for c in panels[names[0]]["columns"] if all(c in p["columns"] for p in panels.values())]
    keys = list(dict.fromkeys(k for t in tabs.values() for k in t))
    moved, only = [], []
    for k in keys:
        have = [n for n in names if k in tabs[n]]
        if len(have) < len(names):
            only.append({"run": k[0], "g": k[1], "in": have})
            continue
        cells = {}
        for c in cols:
            vals = [tabs[n][k][c] for n in names]
            if c != VERSION_COLUMN and any(v != vals[0] for v in vals[1:]):
                cells[c] = dict(zip(names, vals)) | {"change": change(tabs[base][k][c], tabs[last][k][c])}
        if cells:
            moved.append({"run": k[0], "g": k[1], "rotating": k[0] in rotating, "cells": cells,
                          "between": [f"{a}/{b}" for a, b in zip(names, names[1:])
                                      if any(tabs[a][k][c] != tabs[b][k][c] for c in cells)]})
    head = {n: {f: p.get(f) for f in ("id", *STAMPS, "note")} | {"rows": len(p["rows"]), "columns": p["columns"],
                VERSION_COLUMN: sorted({str(r[p["columns"].index(VERSION_COLUMN)]) for r in p["rows"]})
                if VERSION_COLUMN in p["columns"] else None} for n, p in panels.items()}
    return {"panels": head, "base": base, "last": last, "rows": len(keys), "moved": moved, "only": only,
            "moved_rotating": sum(m["rotating"] for m in moved),
            "moved_other": sum(not m["rotating"] for m in moved)}


def same_rows(a: dict, b: dict) -> dict:
    """Whether two panels hold the same columns and rows, and which of their other fields differ."""
    return {"columns": a["columns"] == b["columns"], "rows": canon(a["rows"]) == canon(b["rows"]),
            "row_counts": [len(a["rows"]), len(b["rows"])],
            "differ": sorted(k for k in set(a) | set(b) if k not in ("columns", "rows") and a.get(k) != b.get(k))}


def deep(a, b, path: str = "", skip: tuple[str, ...] = ()) -> list[dict]:
    """Every leaf where two JSON documents differ, by path; a list of dicts that carry `run` is walked by run."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [d for k in dict.fromkeys([*a, *b]) if k not in skip
                for d in deep(a.get(k), b.get(k), f"{path}/{k}", skip)]
    if isinstance(a, list) and isinstance(b, list):
        def by(xs: list) -> dict | None:
            runs = [x.get("run") if isinstance(x, dict) else None for x in xs]
            return dict(zip(runs, xs)) if xs and None not in runs and len(set(runs)) == len(runs) else None
        ka, kb = by(a), by(b)
        if ka is not None and kb is not None:
            return [d for k in dict.fromkeys([*ka, *kb]) for d in deep(ka.get(k), kb.get(k), f"{path}/{k}", skip)]
        if len(a) == len(b):
            return [d for i, (x, y) in enumerate(zip(a, b)) for d in deep(x, y, f"{path}/{i}", skip)]
    return [] if canon(a) == canon(b) else [{"at": path, "before": a, "after": b, "change": change(a, b)}]


# -- on the node


def layers(K: Path, meta: dict) -> tuple[Path, list[Path]]:
    """The capture's checkout and the newer layers' the last refresh scored over (its meta.json's `trace`, `over` and
    `held`), each found by its te_core tree, which keys its trace cache."""
    def tree(p: Path) -> str:
        return git("rev-parse", "HEAD:benchmarks/covert_witness/te_core", cwd=p)[:12]
    te40 = K / "wt" / "te40"
    if tree(te40) != meta["trace"]["te_tree"]:
        raise SystemExit(f"replot: {te40} is at te_core tree {tree(te40)}, and the last refresh traced on "
                         f"{meta['trace']['te_tree']}: a refresh has run since the tables this rebuilds")
    found = {}
    for p in sorted((K / "wt").glob("te-next*")):
        found.setdefault(tree(p), p)
    over = []
    for v in meta.get("over") or []:
        t = meta["held"][v]["te_tree"]
        if t not in found:
            raise SystemExit(f"replot: no checkout under {K / 'wt'} at te_core tree {t} (the last refresh's layer {v}); "
                             f"found {sorted(found)}")
        over.append(found[t])
    return te40, over


def scorer_env(K: Path, S: Path) -> dict:
    """refresh-node.sh's environment, with the ledger a copy under `S` and the measured ledger beside it."""
    return os.environ | {"TE_SCORER_STATE": str(K / "state"), "TE_SCORER": str(K / "state"),
                         "TE_LEDGER": str(S / "ledger"), "TE_LEDGER_40": str(S / "measured"),
                         "TE_REPO": str(K / "repo.git"), "TE40_WT": str(K / "wt" / "te40"),
                         "TE_RESCORE_WT": str(K / "wt" / "rescore"), "TE_NEXT_WT": str(K / "wt" / "te-next"),
                         "TE_OUT_ROOT": str(K / "v1"), "AGENT_GUARD_LOCAL_OK": "1"}


def facts(K: Path) -> dict:
    """What the node holds that this rebuild reads: the refreshes, each checkout's commit, te_core tree and VERSION,
    and the interpreter."""
    wts = {}
    for p in sorted((K / "wt").iterdir()):
        try:
            init = git("show", "HEAD:benchmarks/covert_witness/te_core/__init__.py", cwd=p)
            wts[p.name] = {"commit": git("rev-parse", "HEAD", cwd=p),
                           "te_tree": git("rev-parse", "HEAD:benchmarks/covert_witness/te_core", cwd=p),
                           "te_core": next((x.split('"')[1] for x in init.splitlines() if x.startswith("VERSION = ")), "")}
        except (subprocess.CalledProcessError, OSError) as e:
            wts[p.name] = {"error": str(e)[-200:]}
    import matplotlib
    import numpy

    return {"refreshes": sorted(p.name for p in (K / "v1").iterdir() if p.name[:2] == "20")[-8:], "worktrees": wts,
            "python": sys.version.split()[0], "numpy": numpy.__version__, "matplotlib": matplotlib.__version__,
            "cores": os.cpu_count(), "research": shutil.which("research")}


def measured_ledger(K: Path, S: Path, meta: dict, logs: Path) -> dict:
    """Step 2 (module docstring): K/ledger's records copied and indexed, then trace40.py's measured ledger of them."""
    env = scorer_env(K, S)
    shutil.copytree(K / "ledger" / "records", S / "ledger" / "records")
    run([sys.executable, "-m", "te_core.ledger", "--root", S / "ledger", "index"], logs / "ledger-index.log", env=env)
    theirs, mine = K / "ledger" / "index.jsonl", S / "ledger" / "index.jsonl"
    te40, over = layers(K, meta)
    r = run([sys.executable, HERE / "trace40.py", "ledger", "--out", S / "measured", "--te", te40,
             *(x for o in over for x in ("--over", o))], logs / "trace40-ledger.log", cwd=HERE, env=env)
    built = json.loads(r.stdout.strip().splitlines()[-1])
    was = (meta.get("trace") or {}).get("ledger")
    return {"records": sum(1 for _ in (S / "ledger" / "records").glob("*/*.json")),
            "index_as_the_node_holds_it": theirs.exists() and sha256(theirs) == sha256(mine),
            "capture": str(te40), "over": [str(o) for o in over], "built": built, "last_refresh_built": was,
            "as_last_refresh": built == was}


def linked(src: Path, dst: Path) -> None:
    """`src`'s records under `dst`, each file a hard link."""
    for p in sorted((src / "records").glob("*/*")):
        d = dst / "records" / p.parent.name / p.name
        d.parent.mkdir(parents=True, exist_ok=True)
        os.link(p, d)


def gate_factors(K: Path, stamp: str) -> Path:
    """The analysis's gate factors the refresh at `stamp` built with, else the ones a refresh seeds its builds with
    (v1cycle's `--gates`), else the newest build's."""
    seeded = sorted((K / "v1").glob("*/algorithm-policy/analysis-gates.json"))
    last = K / "v1" / stamp / "algorithm-policy" / "analysis-gates.json"
    found = next((p for p in (last, K / "v1" / "trial" / "gates.json", *reversed(seeded)) if p.exists()), None)
    if found is None:
        raise SystemExit(f"replot: no gate factors under {K / 'v1'} (trial/gates.json, or a refresh's "
                         "algorithm-policy/analysis-gates.json), and the scorer's interpreter can't compile them")
    return found


def builder(out: Path, root: Path, K: Path, meta: dict, cap: Path, failed: Path, policy_io: str,
            before: str = "") -> subprocess.Popen:
    """te_core.algorithm_policy as the scorer's refresh runs it (v1cycle's POLICY_IO; nothing published), started."""
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(gate_factors(K, meta["stamp"].replace(":", "")), out / "analysis-gates.json")
    args = ["--out", str(out), "--root", str(root), "--scorer", str(K / "state"),
            *(x for p in meta["pins"] for x in ("--pin", p)), "--ladder-fit", str(cap)]
    env = {k: v for k, v in os.environ.items() if k not in ("TE_SERVED", "TE_TABLES_CITE", "TE_VM_PUT")}
    log = (out / "run.log").open("w")
    return subprocess.Popen([sys.executable, "-c", before + policy_io, *args], cwd=CW, stdout=log,
                            stderr=subprocess.STDOUT, env=env | {"TE_TRACE_FAILED": str(failed)})


def built(name: str, proc: subprocess.Popen, out: Path) -> dict[str, dict]:
    """The uncut tables of a finished build, by file name."""
    if proc.wait():
        raise SystemExit(f"replot: the {name} build exited {proc.returncode}: {(out / 'run.log').read_text()[-2000:]}")
    return {n: json.loads((out / "tables" / n).read_text()) for n in TABLES}


def reprice(measured: Path, priced: Path, K: Path) -> dict:
    """Step 4 (module docstring): `measured`'s records under `priced`, a rotating run's rows rewritten and every other
    file linked; what it found, per run."""
    from te_core import gates, rotations

    by_run: dict[str, list[Path]] = collections.defaultdict(list)
    first: dict[str, dict] = {}
    n = collections.Counter()
    for p in sorted((measured / "records").glob("*/*")):
        row = json.loads(p.read_text()) if p.suffix == ".json" else None
        run_id = row.get("run_id") if row and row.get("kind") == "submission" else None
        if not run_id:
            d = priced / "records" / p.parent.name / p.name
            d.parent.mkdir(parents=True, exist_ok=True)
            os.link(p, d)
            n["other files"] += 1
            continue
        by_run[run_id].append(p)
        said = "args" in row and isinstance(row.get("island"), dict)
        if run_id not in first or (said and not first[run_id]["said"]):
            first[run_id] = {"program": row.get("program"), "evaluation": {"island": row.get("island")},
                             "script_args": row.get("args"), "said": said}
    records: dict[str, list] = collections.defaultdict(list)
    for line in (K / "state" / "index.jsonl").open():
        x = json.loads(line)
        if x.get("run") and str(x.get("id", "")).startswith("art:"):
            records[x["run"]].append((bool(x.get("valid")), x["id"].removeprefix("art:")))

    def record_of(run_id: str) -> tuple[dict | None, str | None]:
        for _, art in sorted(records.get(run_id, []), reverse=True):
            f = K / "state" / "records" / art / "run" / "run.json"
            if f.exists():
                return json.loads(f.read_text()), art
        return None, None

    def gates_of(row: dict) -> float | None:
        return sum(x["gates"] for x in (row.get("formalism") or {}).get("parts") or []) or None

    price = gates.per_mac(f"gemm/{rotations.TF32}") - gates.per_mac("gemm/bf16")
    out = {"runs": {}, "unread": {}, "unknown_program": {}, "no_record": [], "no_arguments": [],
           "gates_added_per_mac": price}
    for run_id, paths in by_run.items():
        head, record, art = first[run_id], None, None
        if not head["said"]:
            record, art = record_of(run_id)
            if record is None:
                out["no_arguments"].append(run_id)
            else:
                head = record
        found, why = rotations.rules(head), rotations.unknown(head)
        if why:
            out["unknown_program"][run_id] = why
        new = {}
        if found and not why:
            if record is None:
                record, art = record_of(run_id)
            if record is None:
                out["no_record"].append(run_id)
            else:
                try:
                    if rotations.rules(record) != found:
                        raise rotations.Unread(f"its rows' arguments give {found}, its record's {rotations.rules(record)}")
                    _, counts = rotations.named(record)
                    steps = collections.Counter(s.get("struct") for s in record["steps"])
                    for p in paths:
                        row = json.loads(p.read_text())
                        new[p] = (row, rotations.repriced(row, record))
                    cps = {}
                    for row, now in new.values():
                        macs = sum(m for k, (_, m) in now["macs_log"]["ops"].items() if k.endswith(" " + rotations.TF32))
                        cps[str(row["checkpoint_step"])] = {"rotation_macs": macs, "gates_added": macs * price,
                                                           "C_before": gates_of(row), "C_after": gates_of(now)}
                    out["runs"][run_id] = {
                        "rules": found, "record": f"art:{art}", "island": list(rotations.island(record)),
                        "start": (record.get("start") or {}).get("name"), "rows": len(paths),
                        "structures": {sid: c | {"steps": steps.get(sid, 0)} for sid, c in counts.items()},
                        "checkpoints": dict(sorted(cps.items(), key=lambda kv: int(kv[0])))}
                except rotations.Unread as e:
                    out["unread"][run_id] = str(e)
                    new = {}
        for p in paths:
            d = priced / "records" / p.parent.name / p.name
            d.parent.mkdir(parents=True, exist_ok=True)
            if p in new and new[p][1] is not new[p][0]:
                d.write_text(json.dumps(new[p][1], sort_keys=True) + "\n")
                n["rows rewritten"] += 1
            else:
                os.link(p, d)
                n["rows linked"] += 1
    return out | {"counts": dict(n), "rotating_runs": len(out["runs"]), "runs_in_ledger": len(by_run)}


def finals(S: Path, IN: Path, logs: Path, inputs: dict) -> dict:
    """Step 6's finals table: islands/integrator/finals_table.py on its own inputs, the reruns' records fetched again."""
    script = CW / "islands" / "integrator" / "finals_table.py"
    f = IN / "finals"
    run([sys.executable, script, "fetch", "--launches", f / "launches-snap.jsonl", "--to", S / "finals-records",
         "--arts", f / "arts.json"], logs / "finals-fetch.log")
    now = json.loads((S / "finals-records" / "fetch.json").read_text())
    was = json.loads((f / "fetch.json").read_text())
    run([sys.executable, script, "build", "--records", S / "finals-records", "--policy", f / "policy",
         "--policy-art", inputs["finals_policy_art"], "--launches", f / "launches-snap.jsonl",
         "--ladder-fit", f / "CAP.json", "--out", S / "finals"], logs / "finals-build.log")
    from te_core import rotations

    reruns = {}
    for d in sorted(p for p in (S / "finals-records").iterdir() if (p / "out" / "run" / "run.json").exists()):
        record = json.loads((d / "out" / "run" / "run.json").read_text())
        try:
            found = rotations.rules(record)
            reruns[d.name] = {"rules": found, "unknown": rotations.unknown(record),
                              "rotation_macs": rotations.macs(record) if found else 0}
        except rotations.Unread as e:
            reruns[d.name] = {"unread": str(e)}
    return {"without_a_record": sorted(now), "without_a_record_before": sorted(was),
            "same_records": sorted(now) == sorted(was), "reruns": reruns}


def overlay(finals_table: Path, search: Path, out: Path, log: Path) -> dict:
    run([sys.executable, HERE / "plot2_final.py", "build", "--finals", finals_table, "--policy", search, "--out", out],
        log)
    return json.loads((out / "tables" / "algorithm-policy.json").read_text())


def node(a) -> int:
    K, IN, S, OUT = a.state, a.inputs, a.scratch, a.out
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    inputs = json.loads((IN / "inputs.json").read_text())
    dump(OUT / "inputs.json", inputs | {"sha256": {str(p.relative_to(IN)): sha256(p) for p in sorted(IN.rglob("*"))
                                                    if p.is_file()}})
    meta = json.loads((IN / "last" / "meta.json").read_text())
    factors = gate_factors(K, meta["stamp"].replace(":", ""))
    dump(OUT / "node.json", facts(K) | {"gate_factors": {"path": str(factors), "sha256": sha256(factors)}})
    say("the price")
    held = run([sys.executable, "-m", "te_core.gates_main", IN / "main" / "training_slowdown"], logs / "gates-main.log",
               check=False)
    gates = {"rc": held.returncode, "main": json.loads(held.stdout) if held.stdout.strip().startswith("{") else None,
             "main_head": inputs.get("main_head")}
    dump(OUT / "gates.json", gates)
    if held.returncode:
        raise SystemExit(f"replot: te_core's gate table isn't main's: {gates}; {held.stderr.strip()[-600:]}")
    say("the measured ledger")
    led = measured_ledger(K, S, meta, logs)
    dump(OUT / "ledger.json", led)
    say(f"the measured ledger: {led['built']} (the last refresh's: {led['last_refresh_built']})")
    env = scorer_env(K, S)
    policy_io = subprocess.run([sys.executable, "-c", "import v1cycle; print(v1cycle.POLICY_IO)"], cwd=HERE, env=env,
                               check=True, capture_output=True, text=True).stdout
    failed = K / "v1" / meta["stamp"].replace(":", "") / "trace-failed.json"
    if not failed.exists():
        te40, over = layers(K, meta)
        code = ("import json, sys, v1cycle\nfrom pathlib import Path\n"
                "print(json.dumps(v1cycle.trace_failed([Path(p) for p in sys.argv[1:]]), indent=1))")
        failed = S / "trace-failed.json"
        failed.write_text(subprocess.run([sys.executable, "-c", code, str(te40), *map(str, over)], cwd=HERE, env=env,
                                         check=True, capture_output=True, text=True).stdout)
    shutil.copyfile(failed, OUT / "trace-failed.json")
    cap = IN / "finals" / "CAP.json"
    say("the control build starts")
    procs = {"control": builder(S / "control", S / "measured", K, meta, cap, failed, policy_io)}
    say("the rotating runs' rows")
    rot = reprice(S / "measured", S / "priced", K)
    dump(OUT / "rotations.json", rot)
    rotating = set(rot["runs"])
    say(f"{rot['rotating_runs']} rotating runs of {rot['runs_in_ledger']}: {rot['counts']}; unread {len(rot['unread'])}, "
        f"unknown programs {len(rot['unknown_program'])}, no record {len(rot['no_record'])}, no arguments "
        f"{len(rot['no_arguments'])}")
    linked(S / "priced", S / "priced-width")
    procs["priced"] = builder(S / "priced-out", S / "priced", K, meta, cap, failed, policy_io)
    procs["price"] = builder(S / "price-out", S / "priced-width", K, meta, cap, failed, policy_io, PRICE_ONLY)
    say("the finals table")
    fin = finals(S, IN, logs, inputs)
    dump(OUT / "finals" / "reruns.json", fin)
    for n in ("finals-table.json", "finals-table.md"):
        shutil.copyfile(S / "finals" / n, OUT / "finals" / n)
    shutil.copyfile(S / "finals-records" / "fetch.json", OUT / "finals" / "fetch.json")
    served_finals = IN / "served" / "finals-table.json"
    dump(OUT / "diff" / "finals-table.json",
         {"before": "the finals table the served overlay read", "after": "the finals table rebuilt",
          "differ": deep(json.loads(served_finals.read_text()), json.loads((S / "finals" / "finals-table.json").read_text()),
                         skip=("built_at", "te_core", "inputs"))})
    say("waiting for the three builds")
    dirs = {"control": S / "control", "priced": S / "priced-out", "price": S / "price-out"}
    tabs = {name: built(name, procs[name], dirs[name]) for name in ("control", "price", "priced")}
    for name, d in dirs.items():
        for f in (*TABLES, "coverage.md"):
            (OUT / "search" / name).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(d / "tables" / f, OUT / "search" / name / f)
        shutil.copyfile(d / "run.log", OUT / "search" / name / "run.log")
    shutil.copytree(S / "priced-out" / "console", OUT / "search" / "priced" / "console")
    last = {n: json.loads((IN / "last" / n).read_text()) for n in TABLES}
    ap, ff = TABLES[0], TABLES[1]
    dump(OUT / "diff" / "search.json",
         compare({"last": last[ap], "control": tabs["control"][ap], "price": tabs["price"][ap],
                  "priced": tabs["priced"][ap]}, rotating, base="control"))
    served_fit = json.loads((IN / "served" / "console-frontier-fit.json").read_text())
    new_fit = json.loads((S / "priced-out" / "console" / "frontier-fit.json").read_text())
    dump(OUT / "diff" / "frontier-fit.json",
         {"tables": {n: same_rows(last[ff], tabs[n][ff]) for n in tabs},
          "points": {n: same_rows(last[TABLES[2]], tabs[n][TABLES[2]]) if isinstance(last[TABLES[2]], dict)
                     and "rows" in last[TABLES[2]] else last[TABLES[2]] == tabs[n][TABLES[2]] for n in tabs},
          "refused": {n: last[TABLES[4]] == tabs[n][TABLES[4]] for n in tabs},
          "console_served_against_rebuilt": same_rows(served_fit, new_fit)})
    say("the overlays")
    new_finals = S / "finals" / "finals-table.json"
    over = {"served": json.loads((IN / "served" / "overlay.json").read_text()),
            "control": overlay(served_finals, S / "control" / "tables" / ap, S / "overlay-control", logs / "overlay-control.log"),
            "price": overlay(new_finals, S / "price-out" / "tables" / ap, S / "overlay-price", logs / "overlay-price.log"),
            "priced": overlay(new_finals, S / "priced-out" / "tables" / ap, OUT / "overlay", logs / "overlay.log")}
    for name in ("control", "price"):
        (OUT / "overlays" / name).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(S / f"overlay-{name}" / "tables" / ap, OUT / "overlays" / name / ap)
    dump(OUT / "diff" / "overlay.json", compare(over, rotating, base="control"))
    was = json.loads((IN / "served" / "console-algorithm-policy.json").read_text())
    now = json.loads((OUT / "overlay" / "console" / ap).read_text())
    served, rebuilt = set(table(was)), set(table(now))
    dump(OUT / "diff" / "console.json",
         {"served_rows": len(was["rows"]), "rebuilt_rows": len(now["rows"]),
          "served_rows_the_rebuild_drops": sorted(map(list, served - rebuilt)),
          "rows_the_rebuild_adds": sorted(map(list, rebuilt - served)),
          "bytes": len((OUT / "overlay" / "console" / ap).read_bytes()), "note": now["note"],
          "rows": compare({"served": was, "rebuilt": now}, rotating)})
    say("done")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("node", help="the rebuild, on the scorer's node")
    n.add_argument("--state", type=Path, required=True, help="the scorer's state on the node (refresh-node.sh's K)")
    n.add_argument("--inputs", type=Path, required=True, help="the last builds' stored tables and the finals' inputs")
    n.add_argument("--scratch", type=Path, required=True)
    n.add_argument("--out", type=Path, required=True)
    d = sub.add_parser("diff", help="stored panels' rows, cell by cell")
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--rotations", type=Path, default=None, help="`node`'s rotations.json, for which runs rotate")
    d.add_argument("panels", nargs="+", metavar="LABEL=PANEL.json")
    a = ap.parse_args(argv)
    if a.cmd == "node":
        return node(a)
    rotating = set(json.loads(a.rotations.read_text())["runs"]) if a.rotations else set()
    panels = {label: json.loads(Path(path).read_text()) for label, _, path in (x.partition("=") for x in a.panels)}
    out = compare(panels, rotating)
    dump(a.out, out)
    print(f"{out['rows']} rows, {len(out['moved'])} moved ({out['moved_rotating']} of rotating runs), "
          f"{len(out['only'])} in only some panels -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
