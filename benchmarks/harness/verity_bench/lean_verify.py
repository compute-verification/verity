"""`verity bench prove lean-verify`: Flock's Lean verifier (`flock-verify verify`) timed on one kept proof, and the
proof's bytes, as one record `verity bench prove` reads.

    verity bench prove lean-verify --lean flock-verify --proof TREE [--repeat 3] [--out DIR]
        [--session DIR] [--circuit F] [--public F] [--program F] [--partition F] [--registered F] [--tables DIR]
        [--stage F] [--word TYPE] [--definition ID] [--k-log K] [--blocks N] [--full N] [--statement ID] [--plain]
        [--verify-arg=ARG ...]

**The proof** is a tree that holds one session directory (`session.json` and each table's `<table>.rep0.bin` and
`<table>.rep1.bin`, the proof files) and the statement it was proved on: `circuit.txt`, the public file (`pub*.bin`),
and where the statement has them `program.json`, `partition.json`, `registered.json` and `tables/`. Each is found by
its name anywhere under `--proof`, and a name that two files carry is refused: its flag says which. `stage.json`, where
the tree holds the stager's, names the block type: the census type it was staged for (`word`), the staged Definition's
id (`definition`), log2 of a block's witness bits (`k_log`) and the statement's blocks (`blocks`, else `n`) of a full
one's (`full`, else 2^35 witness bits' worth); the flags give or override them.

**A run** is one `flock-verify verify` process, as `benchmarks/one_stage/a0.py` runs the verifier: the
statement's files held by content in the verifier's own archive (`flock-verify archive-put`, timed apart and in no
run's seconds), then `verify --archive --circuit --public [--partition --program --registered] --zk --session`. So a
run's wall seconds hold the verifier reading the statement, setting it up and verifying the one session. The command
runs it `--repeat` times and records each run: its wall and CPU seconds, peak RSS, exit code, verdict, and the phases
the verifier times itself (its inputs, the statement's setup, the session).

**A refusal is the result.** A run the verifier doesn't accept ends the command there, with no further run: the record
says `accepted: false` with the verifier's reason, and the command exits 1. Nothing is run again to get an acceptance.

Writes `DIR/lean-verify.json` (`SCHEMA`; DIR is `--out`, else the run's `$OUT`): the block type, the statement's and
the proof's files with their bytes, the runs, and the host (node, the cores the process may run on, the CPU model,
memory, the commit). `verity bench prove` reads it from a run labelled `run_of_record=prove-WORKLOAD-lean-verify`
(`prove_records`).

Standard library and `verity_bench.settings` only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from verity_bench import settings

SCHEMA = "verity-bench/lean-verify/v1"
#: the record's name in the directory the command writes
RECORD = "lean-verify.json"
STATEMENT_ID = "verity/flock-circuit"
#: a full statement's witness bits (`prove.STATEMENT_BITS`)
STATEMENT_BITS = 2**35
#: a phase the verifier times itself, as it prints it
PHASE = re.compile(r"^(?P<name>[^:\n]+): (?P<ms>\d+) ms$", re.MULTILINE)

#: the clock a run's wall seconds are read from
clock = time.monotonic


def spawn(cmd: Sequence[str], log: Path) -> tuple[int, str, dict]:
    """`cmd` to its end: its exit code, what it printed, and its own CPU seconds and peak RSS (`os.wait4`). `log` gets
    its stderr, then its stdout."""

    with open(log, "w") as f:
        p = subprocess.Popen(list(cmd), stdout=subprocess.PIPE, stderr=f, text=True)
        out = p.stdout.read()
        _, status, ru = os.wait4(p.pid, 0)
        p.returncode = os.waitstatus_to_exitcode(status)
        f.write(out)
    return p.returncode, out, {"cpu_s": round(ru.ru_utime + ru.ru_stime, 3), "user_s": round(ru.ru_utime, 3),
                               "maxrss_gib": round(ru.ru_maxrss / 2**20, 3)}


def _one(tree: Path, pattern: str, flag: str, given: str | None, *, required: bool = False,
         keep: Callable[[Path], bool] = Path.is_file) -> Path | None:
    """The one path under `tree` that `pattern` names, or `given`; None where nothing does and it isn't required."""

    if given:
        return Path(given)
    found = sorted(p for p in tree.rglob(pattern) if keep(p))
    if len(found) > 1:
        raise ValueError(f"{len(found)} paths under {tree} are named `{pattern}` ({', '.join(map(str, found[:4]))}): "
                         f"say which with {flag}")
    if not found and required:
        raise ValueError(f"nothing under {tree} is named `{pattern}`: give it with {flag}")
    return found[0] if found else None


def files(a: Any) -> dict[str, Path | None]:
    """The session directory and the statement's files, each `--proof`'s one path of its name or its flag's."""

    tree = Path(a.proof)
    record = _one(tree, "session.json", "--session", a.session and str(Path(a.session) / "session.json"), required=True)
    session = record.parent
    return {
        "session": session,
        "circuit": _one(tree, "circuit.txt", "--circuit", a.circuit, required=True),
        "public": _one(tree, "pub*.bin", "--public", a.public, required=True,
                       keep=lambda p: p.is_file() and p.parent != session),
        "partition": _one(tree, "partition.json", "--partition", a.partition),
        "program": _one(tree, "program.json", "--program", a.program),
        "registered": _one(tree, "registered.json", "--registered", a.registered),
        "tables": _one(tree, "tables", "--tables", a.tables, keep=Path.is_dir),
        "stage": _one(tree, "stage.json", "--stage", a.stage),
    }


def block(a: Any, stage: Mapping[str, Any]) -> dict:
    """The block type the statement is of: the flags', else `stage.json`'s."""

    def pick(flag: Any, *names: str) -> Any:
        return flag if flag is not None else next((stage[n] for n in names if stage.get(n) is not None), None)

    got = {"word": pick(a.word, "word"), "definition": pick(a.definition, "definition"), "k_log": pick(a.k_log, "k_log"),
           "blocks": pick(a.blocks, "blocks", "n")}
    if got["word"] is None and got["definition"] is None:
        raise ValueError("no block type: the proof holds no stage.json that names one (`word`, `definition`); give "
                         "--word (the census's name of the type) or --definition (the staged Definition's id)")
    for name in ("k_log", "blocks"):
        if not (isinstance(got[name], int) and not isinstance(got[name], bool) and got[name] > 0):
            raise ValueError(f"the statement's `{name}` is {got[name]!r}: give --{name.replace('_', '-')} (no stage.json "
                             "in the proof states it)")
    got["full"] = pick(a.full, "full") or max(1, STATEMENT_BITS >> got["k_log"])
    return got


def host() -> dict:
    """Where the verifier ran: the node, the cores this process may run on and the host's, its CPU model and memory,
    and the commit of the tree."""

    def read(path: str, key: str) -> str | None:
        try:
            return next((line.split(":", 1)[1].strip() for line in Path(path).read_text().splitlines()
                         if line.split(":", 1)[0].strip() == key), None)
        except OSError:
            return None

    kb = read("/proc/meminfo", "MemTotal")
    commit = settings.RESEARCH_SOURCE_SHA
    if not commit:
        p = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"], capture_output=True,
                           text=True, check=False)
        commit = p.stdout.strip() or None if p.returncode == 0 else None
    return {"node": platform.node(), "cores": len(os.sched_getaffinity(0)), "host_cores": os.cpu_count(),
            "cpu_model": read("/proc/cpuinfo", "model name") or platform.processor() or None,
            "mem_gib": round(int(kb.split()[0]) / 2**20, 1) if kb else None, "commit": commit}


def _sized(path: Path) -> dict:
    return {"name": path.name, "bytes": path.stat().st_size}


def measure(a: Any, out: Path, *, run: Callable[[Sequence[str], Path], tuple[int, str, dict]] | None = None,
            stamp: Callable[[], dict] | None = None) -> dict:
    """The record of `a.repeat` runs of the Lean verifier on the proof, stopped at the first run it doesn't accept.
    `run` starts a process (`spawn`) and `stamp` names the host (`host`)."""

    run, stamp = run or spawn, stamp or host
    if a.repeat < 1:
        raise ValueError(f"--repeat {a.repeat}: the verifier runs once or more")
    f = files(a)
    stage = json.loads(f["stage"].read_text()) if f["stage"] else {}
    of = block(a, stage)
    proofs = sorted(p for p in f["session"].glob("*.rep*.bin") if p.is_file())
    if not proofs:
        raise ValueError(f"{f['session']} holds no proof file (`<table>.rep0.bin`, `<table>.rep1.bin`)")
    held = [f[k] for k in ("circuit", "public", "partition", "program", "registered") if f[k]]
    tables = sorted(p for p in f["tables"].iterdir() if p.is_file()) if f["tables"] else []
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="lean-verify-archive-"))
    try:
        t = clock()
        rc, printed, _ = run([a.lean, "archive-put", str(scratch), *map(str, [*held, *tables])], out / "archive-put.log")
        put_s = clock() - t
        keys = {Path(line.split()[1]).name: line.split()[0] for line in printed.splitlines() if len(line.split()) == 2}
        if rc or any(p.name not in keys for p in held):
            raise ValueError(f"`{a.lean} archive-put` exited {rc} and named {len(keys)} of {len(held) + len(tables)} "
                             f"files ({out / 'archive-put.log'})")
        cmd = [a.lean, "verify", "--statement", a.statement, "--archive", str(scratch), "--circuit", keys[f["circuit"].name],
               "--public", keys[f["public"].name],
               *(x for k in ("partition", "program", "registered") if f[k] for x in (f"--{k}", keys[f[k].name])),
               *([] if a.plain else ["--zk"]), *a.verify_arg, "--session", str(f["session"])]
        runs = []
        for i in range(a.repeat):
            t = clock()
            rc, printed, used = run(cmd, out / f"verify-{i}.log")
            wall = clock() - t
            line = next((x for x in printed.splitlines() if x.startswith("VERDICT ")), None)
            verdict = json.loads(line.split(" ", 1)[1]) if line else {"accepted": False, "why": f"no verdict (rc {rc})"}
            log = (out / f"verify-{i}.log").read_text(errors="replace") if (out / f"verify-{i}.log").exists() else ""
            runs.append({"wall_s": round(wall, 3), **used, "rc": rc, "accepted": verdict.get("accepted") is True and rc == 0,
                         "why": verdict.get("why") or None,
                         "verdict_sha512": hashlib.sha512(line.encode()).hexdigest() if line else None,
                         "draw": {k: v for k, v in (verdict.get("unit_draw") or {}).items() if k != "units"} or None,
                         "phases_ms": {"session" if m["name"] == str(f["session"]) else m["name"]: int(m["ms"])
                                       for m in PHASE.finditer(log)}})
            if not runs[-1]["accepted"]:
                break
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {
        "schema": SCHEMA, "run": settings.RESEARCH_RUN_ID, "block": of,
        "statement": {"id": a.statement, "zk": not a.plain, "circuit_sha512": keys[f["circuit"].name],
                      "files": [_sized(p) for p in held], "tables": [_sized(p) for p in tables]},
        "proof": {"files": [_sized(p) for p in proofs], "bytes": sum(p.stat().st_size for p in proofs),
                  "record_bytes": (f["session"] / "session.json").stat().st_size},
        "verifier": {"command": ["flock-verify", *cmd[1:]], "archive_put_s": round(put_s, 3), "repeat": a.repeat, "runs": runs},
        "accepted": len(runs) == a.repeat and all(r["accepted"] for r in runs),
        "host": stamp(),
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lean", required=True, help="flock-verify, the Lean verifier's executable")
    parser.add_argument("--proof", required=True, metavar="TREE", help="the kept proof: a session directory and its statement")
    parser.add_argument("--out", type=Path, default=None, help="where lean-verify.json and the logs go (default: $OUT)")
    parser.add_argument("--repeat", type=int, default=3, help="runs of the verifier, each recorded (default: 3)")
    for name, what in (("session", "the session directory"), ("circuit", "the circuit file"), ("public", "the public file"),
                       ("program", "the verifier's program"), ("partition", "the partition object"),
                       ("registered", "the registered reads"), ("tables", "the pinned tables' directory"),
                       ("stage", "the stager's stage.json")):
        parser.add_argument(f"--{name}", metavar="PATH", help=f"{what} (default: the one of its name under --proof)")
    parser.add_argument("--word", help="the census's name of the block type (default: stage.json's `word`)")
    parser.add_argument("--definition", help="the staged Definition's id (default: stage.json's)")
    parser.add_argument("--k-log", type=int, help="log2 of a block's witness bits (default: stage.json's)")
    parser.add_argument("--blocks", type=int, help="the statement's blocks (default: stage.json's `blocks`, else `n`)")
    parser.add_argument("--full", type=int, help="a full statement's blocks (default: stage.json's, else 2^35 witness bits' worth)")
    parser.add_argument("--statement", default=STATEMENT_ID, help=f"the statement `verify` sets up (default: {STATEMENT_ID})")
    parser.add_argument("--plain", action="store_true", help="a plain session: no --zk")
    parser.add_argument("--verify-arg", action="append", default=[], metavar="ARG",
                        help="one more argument of `verify`, repeatable (a plain session's: --verify-arg=--coins "
                             "--verify-arg=tree)")


def run(a: Any) -> int:
    out = a.out or Path(settings.OUT or Path(settings.RESEARCH_RUN_DIR) / "out")
    doc = measure(a, out)
    (out / RECORD).write_text(json.dumps(doc, indent=1) + "\n")
    runs = doc["verifier"]["runs"]
    print("LEAN-VERIFY " + json.dumps({"accepted": doc["accepted"], "block": doc["block"], "proof_bytes": doc["proof"]["bytes"],
                                       "wall_s": [r["wall_s"] for r in runs], "maxrss_gib": [r["maxrss_gib"] for r in runs],
                                       "why": runs[-1]["why"], "host": doc["host"]}))
    if not doc["accepted"]:
        print(f"lean-verify: the Lean verifier refused run {len(runs)} of {doc['verifier']['repeat']}: {runs[-1]['why']} "
              f"({out / RECORD}); no run follows a refusal", flush=True)
    return 0 if doc["accepted"] else 1
