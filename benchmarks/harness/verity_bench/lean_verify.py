"""`verity bench prove lean-verify`: Flock's Lean verifier (its `verify`) timed on one kept proof, and the
proof's bytes, as one record `verity bench prove` reads.

    verity bench prove lean-verify --lean VERIFIER --proof TREE [--repeat 3] [--out DIR]
        [--session DIR] [--circuit F] [--public F] [--program F] [--partition F] [--registered F] [--tables DIR]
        [--stage F] [--word TYPE] [--definition ID] [--k-log K] [--blocks N] [--full N] [--statement ID] [--plain]
        [--verify-arg=ARG ...]

**The proof** is a tree that holds one session directory (`session.json` and each table's `<table>.rep0.bin` and
`<table>.rep1.bin`, the proof files) and the statement it was proved on: `circuit.txt`, the public file (`pub*.bin`),
and where the statement has them the verifier's program (`program*.json`), `partition.json`, `registered.json` and
`tables/`. Each is found by its name anywhere under `--proof` (so a tree may hold several arts' files side by side, as `lean_verify.sh --with`
lays them), and a name that two files carry is refused: its flag says which. `stage.json`, where
the tree holds the stager's, names the block type: the census type it was staged for (`word`), the staged Definition's
id (`definition`), log2 of a block's witness bits (`k_log`) and the statement's blocks (`blocks`, else `n`) of a full
one's (`full`, else 2^35 witness bits' worth); the flags give or override them.

**What is held to the circuit first, and what isn't.** The circuit's second line (`META {...}`) binds a partition
(`partition`: a `rule` and a `digest`) and states a `program_sha512`. Before any verifier process starts, the command
hashes a partition file under the bound rule (SHA-512 of the rule, a zero byte and the object's canonical JSON) and
refuses one whose digest isn't the bound one, or one the circuit binds nothing for. Under the provisional rule
(`PARTITION_COVER`) the verifier takes no partition object and refuses a session given one, so a partition file that
is the bound cover is held back from it, and the record says so. After `archive-put`, the circuit's SHA-512 is held to
`stage.json`'s `circuit_sha512` where it states one.

**The program is not held to the circuit.** A unit statement's `program_sha512` is no digest of a program file: the
stager writes there the SHA-512 of its `program_digests`, the unit's name with a SHA-256 over the staged Definition's
id and the row layout, which no program descriptor holds. So the command hashes no program against it. It reads the
program only to record which was handed (its digest as the verifier computes it, the SHA-512 of the descriptor's
canonical JSON without `annotations`, and the Definition it names, `program_fn`) and to refuse one that names another
Definition than `stage.json`'s `word`: a comparison of two names, which binds nothing.

**The label says what an accept holds.** At `--zk` on a circuit under the provisional rule the verifier takes the
units as the circuit states them and compares no Program with the circuit, so an accept there is recorded as
`COVER_LABEL`, word for word, and never as the verifier knowing what the statement computes. Any other accept is
labelled by its partition rule alone (`label`). `verity bench prove` prints the label with the seconds.

**A run** is one process of the verifier's `verify`, as `benchmarks/one_stage/a0.py` runs the verifier: the
statement's files held by content in the verifier's own archive (its `archive-put`, timed apart and in no
run's seconds), then `verify --archive --circuit --public [--partition --program --registered] --zk --session`. So a
run's wall seconds hold the verifier reading the statement, setting it up and verifying the one session. The command
runs it `--repeat` times and records each run: its wall and CPU seconds, peak RSS, exit code, verdict, and the phases
the verifier times itself (its inputs, the statement's setup, the session).

**A refusal is the result.** A run the verifier doesn't accept ends the command there, with no further run: the record
says `accepted: false` with the verifier's reason, and the command exits 1. Nothing is run again to get an acceptance.

Writes `DIR/lean-verify.json` (`SCHEMA`; DIR is `--out`, else the run's `$OUT`): the block type, the statement's and
the proof's files with their bytes, the runs, the label of an accept, and the host (node, the cores the process may run on, the CPU model,
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
#: the rules a circuit binds its partition under, as the Lean verifier names them (`Flock.HmRow`): an object the
#: verifier reads (`--partition`), and the provisional cover, for which it takes none
PARTITION_V1, PARTITION_COVER = "verity/partition/v1", "flock-circuit/unit-cover/v0"
#: what an accept at `--zk` on a circuit under `PARTITION_COVER` holds, in the words `verity bench prove` prints
COVER_LABEL = "accepted on the provisional unit cover, units as stated, the Program not held to the circuit"
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
        "program": _one(tree, "program*.json", "--program", a.program),
        "registered": _one(tree, "registered.json", "--registered", a.registered),
        "tables": _one(tree, "tables", "--tables", a.tables, keep=Path.is_dir),
        "stage": _one(tree, "stage.json", "--stage", a.stage),
    }


def bound(circuit: Path) -> dict:
    """What the circuit's second line (`META {...}`) states: its `program_sha512` (a unit statement's is the digest of
    its `program_digests`, not of a program file) and the partition's `rule` and `digest`, each None where the line
    states none. A circuit without that line states neither."""

    with open(circuit, "rb") as f:
        f.readline()
        line = f.readline()
    if not line.startswith(b"META "):
        return {"program_sha512": None, "partition": None}
    try:
        meta = json.loads(line[5:])
    except ValueError as e:
        raise ValueError(f"{circuit}: its second line is not `META` and a JSON object ({e})") from e
    part = meta.get("partition") if isinstance(meta, dict) else None
    return {"program_sha512": meta.get("program_sha512") if isinstance(meta, dict) else None,
            "partition": {"rule": part.get("rule"), "digest": part.get("digest")} if isinstance(part, dict) else None}


def _digest(obj: Any, rule: str = "") -> str:
    """SHA-512 of `obj`'s canonical JSON (compact, keys sorted, ASCII: the bytes the Lean verifier's `canon` writes),
    after `rule` and a zero byte where a rule tags it."""

    return hashlib.sha512((rule.encode() + b"\0" if rule else b"")
                          + json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _json(path: Path, flag: str) -> Any:
    try:
        return json.loads(path.read_bytes())
    except ValueError as e:
        raise ValueError(f"{path} ({flag}) is not JSON ({e}): no verifier was started") from e


def hold(f: Mapping[str, Path | None], stage: Mapping[str, Any]) -> dict:
    """What the circuit states (`bound`) and the files held to it. A partition file is held by its digest under the
    bound rule, as the verifier computes it (`HmRow.partitionDigest`; the provisional cover's the same bytes under its
    own rule): one that isn't the bound object, or that the circuit binds nothing for, is refused here. `held` names
    the files that passed, and `withheld` the ones the verifier isn't given: the partition under `PARTITION_COVER`,
    for which it takes no object. The program is held to nothing the circuit states. `program` is the one handed: its
    digest as the verifier computes it (`ProvedScope.programDigest`: the descriptor without `annotations`) and the
    Definition it names (`program_fn`), which is refused when `stage` names another as its `word`."""

    to = bound(f["circuit"])
    doc = {**to, "held": [], "withheld": [], "program": None}
    if f["program"]:
        desc = _json(f["program"], "--program")
        fn = desc.get("program_fn") if isinstance(desc, dict) else None
        if None not in (fn, stage.get("word")) and fn != stage["word"]:
            raise ValueError(f"{f['program']} is the program of another Definition than the statement's: its `program_fn` "
                             f"is {fn}, and {f['stage']} names the `word` {stage['word']}: no verifier was started (the "
                             "verifier compares no program with this circuit, so it would not refuse it)")
        doc["program"] = {"sha512": _digest({k: v for k, v in desc.items() if k != "annotations"}
                                            if isinstance(desc, dict) else desc), "fn": fn}
    if f["partition"]:
        rule = (to["partition"] or {}).get("rule")
        if rule not in (PARTITION_V1, PARTITION_COVER):
            raise ValueError(f"{f['circuit']} binds no partition under a rule this command hashes (its rule is {rule!r}; "
                             f"{PARTITION_V1} and {PARTITION_COVER} are), so nothing holds {f['partition']} to it: no "
                             "verifier was started")
        got = _digest(_json(f["partition"], "--partition"), rule)
        if got != to["partition"]["digest"]:
            raise ValueError(f"{f['partition']} is not the partition the circuit binds: its digest under {rule} is {got}, "
                             f"and {f['circuit']} binds {to['partition']['digest']}: no verifier was started")
        doc["held"].append("partition")
        if rule == PARTITION_COVER:
            doc["withheld"].append("partition")
    return doc


def label(to: Mapping[str, Any], zk: bool) -> str:
    """What an accept of this statement holds: `COVER_LABEL` at `--zk` under the provisional cover, else the partition
    rule the circuit binds and nothing more."""

    rule = (to["partition"] or {}).get("rule")
    if zk and rule == PARTITION_COVER:
        return COVER_LABEL
    return f"accepted {'at --zk' if zk else 'as a plain session'}, the circuit's partition under {rule or 'no rule'}"


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
        if not Path(path).exists():
            return None
        return next((line.split(":", 1)[1].strip() for line in Path(path).read_text().splitlines()
                     if line.split(":", 1)[0].strip() == key), None)

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
    to = hold(f, stage)
    handed = to.pop("program")
    f = {k: None if k in to["withheld"] else p for k, p in f.items()}
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
        if stage.get("circuit_sha512") not in (None, keys[f["circuit"].name]):
            raise ValueError(f"{f['circuit']} is not the circuit {f['stage']} names: its SHA-512 is "
                             f"{keys[f['circuit'].name]}, and `circuit_sha512` there is {stage['circuit_sha512']}: no "
                             "session was verified")
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
                         "phases_ms": {"session" if m["name"] == str(f["session"]) else m["name"]: int(m["ms"])
                                       for m in PHASE.finditer(log)}})
            if not runs[-1]["accepted"]:
                break
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    accepted = len(runs) == a.repeat and all(r["accepted"] for r in runs)
    return {
        "schema": SCHEMA, "run": settings.RESEARCH_RUN_ID, "block": of,
        "statement": {"id": a.statement, "zk": not a.plain, "circuit_sha512": keys[f["circuit"].name], "bound": to, "program": handed,
                      "files": [_sized(p) for p in held], "tables": [_sized(p) for p in tables]},
        "proof": {"files": [_sized(p) for p in proofs], "bytes": sum(p.stat().st_size for p in proofs),
                  "record_bytes": (f["session"] / "session.json").stat().st_size},
        "verifier": {"command": [Path(a.lean).name, *cmd[1:]], "archive_put_s": round(put_s, 3), "repeat": a.repeat, "runs": runs},
        "accepted": accepted, "label": label(to, not a.plain) if accepted else None,
        "host": stamp(),
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lean", required=True, help="the Lean verifier's executable, as `lake build` leaves it in verity/protocols/flock/verifier")
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
