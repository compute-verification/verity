"""A steady-state point's node side: one row of a steady-state spec (`steady_spec`) at one prover count, timed on one
card of a GPU node. `steady_point.sh` runs it from the shipped source, in the environment the spec's setup leaves, under
the `research run` that `steady_plan` launches through one of the node's GPU slots.

    python -m verity_bench.steady_node --spec SPEC --point POINT --identity IDENTITY

1. **Before any card is asked for** it holds the run to its point: the identity it was launched with is the one this
   tree gives the point (`steady_spec.point_identity`: a changed file of the closure is another point); the run's
   environment is the spec's; the node can run `gpu-lease` (`lease_tool`); the prover is the one the setup built
   (`VERITY_BENCH_STEADY_BIN`); and the run's CPUs are exactly the spec's `cores`, which the node's slot gives. It stages the row's statement with
   `benchmarks/served_units/stage.py`, or takes the one the row names as `staged`: a copy of a directory on the node,
   or a directory of an artifact fetched from the store (`fetched`), onto the node's disk and never into its memory.
   It refuses a statement whose circuit isn't the row's `pin` or whose units aren't its `n`.
2. **One lease of one card** (the harness kit's, `kit.lease`: `gpu-lease 1 --wait --max-min M`), asked of the cards on
   the NUMA node of the run's CPUs where the node says which they are (`cards_near`: a slot gives a run its CPUs and
   no card) and declaring no cores (`lease_argv`), and inside it the timed step (this module again, with `--inside`):
   it reads the card and the driver's version from the kit's host stamp (`kit.host_stamp`, the lease's card) and the
   toolkit's from the node's nvcc, and refuses a card or a version that isn't the spec's; records what else is on the
   node (`steady.card_file`) before and after;
   runs `benchmarks/one_stage/server_rate.py` at the point's load (`steady_spec.timed_sessions`); and keeps the row,
   with the versions it read (`steady_spec.VERSIONS`) and the period its sessions were sized by (`steady.PERIOD`, which
   `level` reads), and the sessions' record under `out/rate/`.
3. **After the lease has ended**, so that its usage line is among the run's files, the verdict: the row is a passed point
   when `steady_spec.point_why` finds nothing against it, from the run's own files. `result.json` passes with it,
   and a passed point's rate is its one measurement, appended by the kit's writer under the step's host stamp.

It exits 0 for a passed point, `steady_spec.NOT_PASSED` for a point that was timed and didn't pass, and `REFUSED` where
nothing was timed. Its records are `out/point.json` (the point, the prover, the cores, the cards its lease was asked
of, the card and the versions), `out/verdict.json` (each check's answer) and `out/rate/`; `result.json` lists every
file a verdict reads (`kept`), so the store holds what `plan` judges the run by. The statement and the provers' session
files stay in a scratch directory that it removes, also when the run is cancelled or timed out: TERM and INT end it
with `STOPPED`, after the command it was waiting for (`_run`) and with no result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from verity_config import sha256

from verity_bench import jobs, kit, settings, steady, steady_spec
from verity_bench.vllm.presets import GPUS

KIND = "verity-steady-point/v1"
POINT_FILE, VERDICT_FILE, TIMED_FILE = "out/point.json", "out/verdict.json", "out/timed.json"
REFUSED = 4
#: the exit code of a run that TERM or INT ended: 128 and TERM's number, for both
STOPPED = 128 + signal.SIGTERM
#: the seconds a command this process was waiting for has to end, once sent TERM, before it is killed
STOP_S = 30
#: the exit code `_ask` gives for a command the node couldn't run at all, as a shell gives it
NOT_RUN = 127
#: the lease tool in the tree, which a node has on its path as `gpu-lease`
LEASE_SOURCE = "tools/research/src/research/pods/sh/gpu_lease.sh"
#: the files of a staged statement the node side reads; the prover reads the rest
STAGED = ("stage.json", "circuit.txt", "pin.txt")
#: the store's fetch, from the shipped tree (`steady_point.sh` puts its package on the path)
FETCH = ("-m", "research", "data", "fetch")
#: where a node keeps files in its memory: no statement fetched from the store goes there
MEMORY = Path("/dev/shm")
#: where a node lists each NUMA node's CPUs (`nodeN/cpulist`) and each PCI device's NUMA node (`ADDRESS/numa_node`), and
#: the query that gives each card's index, as gpu-lease counts them, with its PCI address
NODES, DEVICES = Path("/sys/devices/system/node"), Path("/sys/bus/pci/devices")
CARDS = ("nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader")
#: the bytes kept of each of the server's logs, from its end
KEPT_LOG_BYTES = 200_000

#: a command's exit code, its output to a log file
Runner = Callable[[Sequence[str], Path], int]
#: a command's exit code and what it printed
Asker = Callable[[Sequence[str]], tuple[int, str]]
#: where a number is measured, as the kit stamps it (`kit.host_stamp`)
Stamper = Callable[[], Mapping[str, Any]]


class Refused(ValueError):
    """The point can't be timed as its spec says; nothing was timed."""


def _run(argv: Sequence[str], log: Path) -> int:
    """A command's exit code, its output in `log`. Where this process is ended while it waits (`_stopped`), the command
    is sent TERM and waited for first: a lease's holder then releases its card and stops its step before the scratch
    goes, where killing it would leave the step running."""

    with log.open("w") as f, subprocess.Popen(list(argv), stdout=f, stderr=subprocess.STDOUT) as process:
        try:
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=STOP_S)
            except subprocess.TimeoutExpired:
                process.kill()
            raise


def _ask(argv: Sequence[str]) -> tuple[int, str]:
    try:
        process = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    except OSError as error:
        return NOT_RUN, f"{type(error).__name__}: {error}"
    return process.returncode, process.stdout if process.returncode == 0 else process.stdout + process.stderr


def _write(path: Path, doc: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---- before the card ----------------------------------------------------------------------------------------------------------


def prover() -> dict[str, str]:
    """The prover the spec's setup built, with its SHA-256."""

    named = settings.VERITY_BENCH_STEADY_BIN
    if not named or not os.access(named, os.X_OK) or not Path(named).is_file():
        raise Refused(f"the spec's setup left no prover: VERITY_BENCH_STEADY_BIN ({named!r}) names no executable file")
    return {"path": named, "sha256": _digest(Path(named), "sha256")}


def own_cores(spec: steady_spec.SteadySpec, cpus: Sequence[int]) -> list[int]:
    """The run's CPUs, which are the spec's `cores` exactly: with fewer or more the row is another point's."""

    if len(cpus) != spec.cores:
        raise Refused(f"the run has {len(cpus)} CPUs ({_spans(cpus)}), and the `cores` of spec {spec.id!r} is {spec.cores}: "
                      "launch it through a slot of the node whose line gives that many")
    return sorted(cpus)


def _spans(cpus: Sequence[int]) -> str:
    spans: list[list[int]] = []
    for cpu in sorted(cpus):
        if spans and cpu == spans[-1][1] + 1:
            spans[-1][1] = cpu
        else:
            spans.append([cpu, cpu])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in spans)


def fetched(row: steady_spec.SteadyRow, scratch: Path, out: Path, run: Runner, ask: Asker) -> Path:
    """The statement a row names in the store (`SteadyRow.stored`), fetched into the scratch: the artifact's directory,
    read with the key `research run --store-read` puts in the run's environment. The fetch keeps what it reads in a
    store of its own under the scratch, removed once the files are out of it, so the node's store holds no copy and
    nothing stays on the node after the scratch. Before any byte is fetched it refuses a scratch in the node's memory,
    and one whose disk has less free than the statement twice, which the fetch holds until its store is removed."""

    art, directory = row.stored
    if MEMORY in scratch.resolve().parents:
        raise Refused(f"the `staged` of row {row.type!r} is fetched from the store onto the node's disk, and the scratch "
                      f"({scratch}) is in its memory: the spec's setup exports VERITY_BENCH_STEADY_SCRATCH, a directory "
                      "on the node's disk")
    store, got = scratch / "store", scratch / "fetched"
    fetch = [sys.executable, *FETCH, art, "--store", str(store), "--path", directory]
    rc, text = ask([*fetch, "--list", "--json"])
    try:
        size = sum(file["bytes"] for file in json.loads(text)) if rc == 0 else 0
    except (ValueError, TypeError, KeyError) as error:
        raise Refused(f"the store's listing of the `staged` of row {row.type!r}, {row.staged}, isn't each file with its "
                      f"bytes ({type(error).__name__}: {text.strip()[-300:]!r})") from error
    if not size:
        raise Refused(f"the store lists no file of the `staged` of row {row.type!r}, {row.staged} (exit {rc}: "
                      f"{text.strip()[-300:]!r}): a run reads the store with the key `research run --store-read` gives "
                      "it, which `verity-bench run SPEC --direct NODE` passes for a row whose `staged` names the store")
    free = shutil.disk_usage(scratch).free
    if free < 2 * size:
        raise Refused(f"the `staged` of row {row.type!r} is {size} bytes, held twice while it is fetched, and the disk "
                      f"of the scratch ({scratch}) has {free} free")
    rc = run([*fetch, "--to", str(got)], out / "fetch.log")
    shutil.rmtree(store, ignore_errors=True)
    if rc or not (got / directory).is_dir():
        raise Refused(f"fetching the `staged` of row {row.type!r}, {row.staged}, failed (exit {rc}): out/fetch.log")
    return got / directory


def statement(row: steady_spec.SteadyRow, scratch: Path, out: Path, run: Runner, ask: Asker = _ask) -> dict[str, Any]:
    """The row's statement where `server_rate.py` looks for it (`SCRATCH/z/NAME/cN/stage`): staged here, a copy of the
    one the row names on the node, or the one it names in the store (`fetched`). Its circuit's SHA-512 is computed on
    that copy, never taken from a file beside it."""

    stage = scratch / "z" / row.name / f"c{row.n}" / "stage"
    stage.parent.mkdir(parents=True)
    if row.stored is not None:
        fetched(row, scratch, out, run, ask).rename(stage)
        if not all((stage / name).is_file() for name in STAGED):
            raise Refused(f"the `staged` of row {row.type!r}, {row.staged}, holds no staged statement in the store "
                          f"({', '.join(STAGED)})")
    elif row.staged is not None:
        given = Path(row.staged)
        if not all((given / name).is_file() for name in STAGED):
            raise Refused(f"the `staged` of row {row.type!r}, {row.staged}, holds no staged statement on this node "
                          f"({', '.join(STAGED)})")
        shutil.copytree(given, stage)
    else:
        rc = run([sys.executable, steady_spec.STAGER, row.type, str(row.n), str(stage)], out / "stage.log")
        if rc or not all((stage / name).is_file() for name in STAGED):
            raise Refused(f"staging {row.type!r} at {row.n} units failed (exit {rc}): out/stage.log")
    pin = _digest(stage / "circuit.txt", "sha512")
    if pin != row.pin:
        raise Refused(f"the statement's circuit has SHA-512 {pin[:16]}..., and the `pin` of row {row.type!r} is "
                      f"{row.pin[:16]}...: another circuit takes no GPU time")
    if (said := (stage / "pin.txt").read_text().strip()) != pin:
        raise Refused(f"the statement's pin.txt, which the timing script writes in the row, says {said[:16]}..., and its "
                      f"circuit has SHA-512 {pin[:16]}...")
    record = json.loads((stage / "stage.json").read_text())
    if record.get("n") != row.n:
        raise Refused(f"the statement holds {record.get('n')!r} units, and the `n` of row {row.type!r} is {row.n}")
    return {"from": row.staged or steady_spec.STAGER, "pin": pin, "n": row.n,
            "stage": {key: record.get(key) for key in ("definition", "word", "k_log", "vus_per_block", "circuit_bytes")}}


def cards_near(cpus: Sequence[int], ask: Asker) -> dict[str, Any]:
    """The cards on the NUMA node of the run's CPUs (`cards`, by index), which the point's lease is asked of: a slot of
    the node gives a run its CPUs and no card, and a card across the machine from its provers' CPUs is another
    measurement. `cards` is None, with `why`, where the node doesn't say: the run's CPUs aren't on one NUMA node it
    lists (of a timed row, the check `one_numa_node` says so), or no card says it is on that node (a machine with one
    node, or a VM, says none). The lease then takes any card."""

    mine, held = set(cpus), []
    for listed in sorted(NODES.glob("node[0-9]*/cpulist")):
        spans = [span.split("-") for span in listed.read_text().strip().split(",") if span]
        if any(mine & set(range(int(span[0]), int(span[-1]) + 1)) for span in spans):
            held.append(int(listed.parent.name[4:]))
    if len(held) != 1:
        return {"numa_nodes": sorted(held), "cards": None, "why": f"the node lists {len(held)} NUMA nodes for the run's CPUs"}
    rc, text = ask(list(CARDS))
    where: dict[int, int] = {}
    for line in text.splitlines() if rc == 0 else ():
        index, _, address = (part.strip() for part in line.partition(","))
        # nvidia-smi writes the address with eight digits of domain and in capitals, the kernel with four and without
        said = DEVICES / address[-12:].lower() / "numa_node"
        node = said.read_text().strip() if index.isdigit() and said.is_file() else ""
        if re.fullmatch(r"-?\d+", node):
            where[int(index)] = int(node)
    if not (cards := sorted(index for index, node in where.items() if node == held[0])):
        return {"numa_nodes": held, "cards": None, "why": (
            f"none of the node's {len(where)} cards says it is on NUMA node {held[0]}" if rc == 0 else
            f"`{' '.join(CARDS)}` exited {rc}")}
    return {"numa_nodes": held, "cards": cards}


def lease_argv(spec: steady_spec.SteadySpec, inside: Sequence[str], on: Sequence[int] | None = None) -> list[str]:
    """The timed step under the kit's lease (`kit.lease`) of one card for the spec's bound, with what a point's lease
    says beside it, before the command: the cards it may be (`on`, gpu-lease's `--on`: `cards_near`), preemptible, as a
    guest's GPU job must be (a preempted point has no passing run, so the next `run` launches it again), and the host
    memory the spec admits. It declares no cores (gpu-lease's `--cores`): the run is pinned to its slot's, by which
    gpu-lease counts a lease that declares none, and whether the machine throttled them while the row ran is the
    row's check `no_throttling`."""

    minutes = spec.admission.max_minutes
    if minutes is None:
        raise Refused(f"spec {spec.id!r} has no admission max_minutes, which bounds the point's lease")
    near = ["--on", ",".join(map(str, on))] if on else []
    memory = ["--mem-gb", str(spec.admission.mem_gb)] if spec.admission.mem_gb is not None else []
    *lease, command = kit.lease(1, minutes)
    return [*lease, *near, "--preemptible", *memory, command, *inside]


def lease_tool(ask: Asker) -> None:
    """Refuses a node on which `gpu-lease` can't be run, before a statement is staged or fetched for a lease that could
    not be taken: the kit's lease calls it by its bare name."""

    tool = kit.lease(1, 1)[0]
    rc, text = ask([tool, "status"])
    if rc == NOT_RUN:
        raise Refused(f"`{tool}` can't be run on this node ({text.strip()[-200:]}), and the timed step runs under its "
                      f"lease of one card: put {LEASE_SOURCE} of this tree on the node's path as `{tool}`. A copy on "
                      "the node's own disk goes when the node is reset")


# ---- inside the lease ---------------------------------------------------------------------------------------------------------


def leased_card(stamp: Mapping[str, Any]) -> Mapping[str, Any]:
    """The kit's host stamp, once it is of the lease's one card: its name, its UUID and the driver's version."""

    gpus = stamp.get("gpus") or []
    if stamp.get("why") or len(gpus) != 1 or not stamp.get("driver"):
        raise Refused("the host stamp doesn't say what the lease's card is ("
                      + (stamp.get("why") or f"it names {len(gpus)} GPUs and the driver {stamp.get('driver')!r}")
                      + "): inside a gpu-lease of one card it is that card's, which gpu-lease names in GPU_LEASE_UUID")
    return stamp


def toolkit_on_node(ask: Asker) -> str:
    """The CUDA toolkit's version: what the nvcc on the path the spec's setup leaves says it is."""

    rc, text = ask(["nvcc", "--version"])
    found = re.search(r"release [\d.]+, V(\d+(?:\.\d+)+)", text) if rc == 0 else None
    if found is None:
        raise Refused("the nvcc on the path the spec's setup leaves doesn't say its version (exit "
                      f"{rc}: {text.strip()[-200:]!r}): the setup puts the toolkit the prover was built with first")
    return found.group(1)


def snapshot(path: Path, run_id: str, ask: Asker, now: Callable[[], float]) -> None:
    """What else is on the node now, as `steady.card_file` says a run keeps it: each card another run's lease holds and
    each compute process on a card, or a line saying which of the two the node gave no answer on."""

    lines = [f"epoch {int(now())}"]
    rc, text = ask(["gpu-lease", "status"])
    if rc != 0:
        lines.append(f"gpu-lease status gave no answer (exit {rc})")
    for line in text.splitlines() if rc == 0 else ():
        if re.match(r"\d+ held\b", line) and f" run={run_id} " not in f"{line} ":
            lines.append("lease " + line.split(" cmd=")[0][:200])
    cards_rc, cards = ask(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"])
    apps_rc, apps = ask(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"])
    if cards_rc != 0 or apps_rc != 0 or not cards.strip():
        lines.append("nvidia-smi gave no answer")
    else:
        index = {uuid.strip(): number.strip() for number, _, uuid in (line.partition(",") for line in cards.splitlines())}
        for line in apps.splitlines():
            fields = [field.strip().replace(" ", "") for field in line.split(",")]
            if len(fields) == 3 and fields[0]:
                lines.append(f"app GPU {index.get(fields[0], '?')} pid {fields[1]} {fields[2]}")
    path.write_text("\n".join(lines) + "\n")


def keep(scratch_rate: Path, out_rate: Path, item: steady_spec.SteadyPoint, read: Mapping[str, str]) -> int:
    """The step's records under `out/rate/`: the script's rows, each with the versions this step read on the node and
    the period its timed sessions were sized by (a row that declares one), the sessions' record, the device's samples
    and the end of each log. The verifiers' session files stay in the scratch. How many rows it kept."""

    out_rate.mkdir(parents=True, exist_ok=True)
    rates = scratch_rate / "rate.json"
    more = {steady_spec.VERSIONS: dict(read), **({steady.PERIOD: item.row.period} if item.row.period else {})}
    rows = [{**row, **more} for row in (json.loads(rates.read_text()) if rates.is_file() else [])]
    _write(out_rate / "rate.json", rows)
    step = scratch_rate / f"{item.row.name}-k{item.provers}"
    kept = out_rate / step.name
    kept.mkdir(exist_ok=True)
    for name in ("sessions.jsonl", "device.tsv"):
        if (step / name).is_file():
            shutil.copyfile(step / name, kept / name)
    for log in sorted(step.glob("*.log")):
        (kept / log.name).write_bytes(log.read_bytes()[-KEPT_LOG_BYTES:])
    return len(rows)


def timed(spec: steady_spec.SteadySpec, item: steady_spec.SteadyPoint, scratch: Path, run_dir: Path, *,
          run: Runner = _run, ask: Asker = _ask, now: Callable[[], float] = time.time,
          stamp: Stamper = kit.host_stamp) -> int:
    """The timed step, inside the lease: the script's exit code, or `REFUSED` with the reason in `TIMED_FILE`, which
    also keeps the host stamp the step ran under (`host`), for the run's measurement."""

    out = run_dir / "out"
    record: dict[str, Any] = {"refused": None}
    try:
        device = settings.CUDA_VISIBLE_DEVICES
        if not device or "," in device:
            raise Refused(f"the timed step runs under a gpu-lease of one card (CUDA_VISIBLE_DEVICES is {device!r})")
        host = leased_card(stamp())
        read = {"driver": host["driver"], "cuda_toolkit": toolkit_on_node(ask)}
        record |= {"device": device, "host": dict(host), steady_spec.VERSIONS: read}
        if GPUS[spec.card].nvidia_smi_name not in host["card"]:
            raise Refused(f"the lease's card is a {host['card']}, not the `card` of spec {spec.id!r} ({spec.card}: "
                          f"{GPUS[spec.card].nvidia_smi_name})")
        if read != steady_spec.versions(spec):
            raise Refused(f"the node's driver and CUDA toolkit are {read}, and spec {spec.id!r} says "
                          f"{steady_spec.versions(spec)}: a row at other versions is another point. Set the spec's "
                          "`driver` and `cuda_toolkit` to the node's, which times its rows again there")
        row = {"provers": item.provers}
        (out / "rate").mkdir(parents=True, exist_ok=True)
        snapshot(run_dir / steady.card_file(row, "start"), settings.RESEARCH_RUN_ID, ask, now)
        rc = run([sys.executable, steady_spec.TIMER, "--bin", prover()["path"], "--staged", str(scratch / "z"),
                  "--types", item.row.name, "--provers", str(item.provers), "--out", str(scratch / "rate"),
                  "--device", device, "--sessions", str(item.sessions), "--warm", str(item.row.warm),
                  "--ahead", str(spec.ahead)], out / "rate" / "server-rate.log")
        snapshot(run_dir / steady.card_file(row, "end"), settings.RESEARCH_RUN_ID, ask, now)
        record |= {"exit": rc, "rows": keep(scratch / "rate", out / "rate", item, read)}
    except Refused as refused:
        record["refused"] = str(refused)
        rc = REFUSED
    _write(run_dir / TIMED_FILE, record)
    return rc


# ---- the point ----------------------------------------------------------------------------------------------------------------


def kept(run_dir: Path) -> list[str]:
    """The run's files a verdict of it is computed from, with its records: everything under `out/`, and the lease's
    usage line. The run's result lists them: the store keeps the files a result lists, and `plan` judges the run by
    those."""

    files = sorted(path.relative_to(run_dir).as_posix() for path in (run_dir / "out").rglob("*") if path.is_file())
    return files + ([steady.LEASES] if (run_dir / steady.LEASES).is_file() else [])


def write_result(run_dir: Path, identity: str, item: steady_spec.SteadyPoint, why: str | None,
                 row: Mapping[str, Any] | None, host: Mapping[str, Any] | None = None) -> None:
    """The run's result (`kit.RESULT_SCHEMA`): its validation, which is the point's verdict, written here, and, for a
    passed point, its rate, the one measurement, appended by the kit's writer under `host`, the stamp the timed step
    ran under."""

    rate = row.get("statements_per_s") if row is not None and why is None else None
    evidence = [path for path in (steady.RATES, VERDICT_FILE, POINT_FILE) if (run_dir / path).is_file()]
    _write(run_dir / kit.RESULT, {
        "schema": kit.RESULT_SCHEMA, "kind": KIND, "run_id": settings.RESEARCH_RUN_ID,
        "workload_fingerprint": {"identity": identity, "point": item.to_json()},
        "validation": {"status": "passed" if why is None else "failed", "checker": "verity_bench.steady_spec.point_why",
                       "detail": why or f"{item.id}: the ten checks pass", "evidence": evidence},
        "measurements": [],
        "measurement_files": [steady.RATES] if (run_dir / steady.RATES).is_file() else [], "artifacts": kept(run_dir)})
    if rate is not None:
        if host is None:
            raise ValueError(f"{item.id} passed and {TIMED_FILE} holds no host stamp of its timed step, which its "
                             "measurement is written under")
        kit.Writer(run_dir, settings.RESEARCH_RUN_ID, host).measurement("statements_per_s", rate, "1/s")


def point(a: argparse.Namespace, *, run: Runner = _run, ask: Asker = _ask, now: Callable[[], float] = time.time,
          cpus: Callable[[], Sequence[int]] = lambda: sorted(os.sched_getaffinity(0)),
          stamp: Stamper = kit.host_stamp) -> int:
    root = Path.cwd()
    spec = steady_spec.load_spec(a.spec, root=root)
    item = steady_spec.point(spec, a.point)
    files = steady_spec.closure(root, spec)
    files_sha256 = sha256(files)
    identity = steady_spec.point_identity(spec, item, files_sha256)
    if identity != a.identity:
        raise ValueError(f"point identity changed: this tree gives {item.id} {identity}, and the run was launched as "
                         f"{a.identity}: a file its time depends on differs from the launching checkout's. "
                         + ("" if (root / ".git").exists() else
                            "This tree is no checkout, so every file under "
                            f"{', '.join(entry for entry in steady_spec.CLOSURE if (root / entry).is_dir())} counts: one "
                            "the launching checkout doesn't track (what a build or a setup left there) changes the "
                            "identity too. ")
                         + f"Its closure here is {len(files)} files (SHA-256 {files_sha256[:16]}...); `plan` prints "
                         "the launching checkout's as its `toolchain`")
    run_dir = Path(settings.RESEARCH_RUN_DIR).resolve()
    out = run_dir / "out"
    out.mkdir(parents=True, exist_ok=True)
    jobs.check_point_environment(spec, settings.environ())
    if a.inside is not None:
        return timed(spec, item, a.inside, run_dir, run=run, ask=ask, now=now, stamp=stamp)

    # a requeued point reruns whole in its run directory: no earlier attempt's verdict or record stands beside this one's
    for name in (kit.RESULT, POINT_FILE, VERDICT_FILE, TIMED_FILE, *(steady.card_file(item.to_json(), when)
                                                                        for when in ("start", "end"))):
        (run_dir / name).unlink(missing_ok=True)
    shutil.rmtree(out / "rate", ignore_errors=True)
    record: dict[str, Any] = {"schema": "verity-bench/steady-point/v0", "spec": spec.id, "point": item.to_json(),
                              "identity": identity, "files_sha256": files_sha256}
    scratch = None
    try:
        lease_tool(ask)
        record["prover"] = prover()
        own = own_cores(spec, cpus())
        record["cores"] = {"count": spec.cores, "list": _spans(own)}
        record["lease_on"] = cards_near(own, ask)
        scratch = Path(tempfile.mkdtemp(prefix="verity-bench-steady-", dir=settings.VERITY_BENCH_STEADY_SCRATCH))
        lease = lease_argv(spec, [sys.executable, "-m", "verity_bench.steady_node", "--spec", a.spec, "--point", a.point,
                                  "--identity", a.identity, "--inside", str(scratch)], record["lease_on"]["cards"])
        record["statement"] = statement(item.row, scratch, out, run, ask)
        record["lease_exit"] = run(lease, out / "lease.log")
        step = json.loads((run_dir / TIMED_FILE).read_text()) if (run_dir / TIMED_FILE).is_file() else {}
        record["timed"] = step
        if step.get("refused"):
            raise Refused(step["refused"])
        if not step:
            began = (run_dir / steady.card_file(item.to_json(), "start")).is_file()
            raise Refused(("the timed step was stopped before its end and left no row, as a preempted lease stops it"
                           if began else "the timed step never began")
                          + f": gpu-lease exited {record['lease_exit']} (out/lease.log)")
    except Refused as refused:
        record["refused"] = str(refused)
        _write(run_dir / POINT_FILE, record)
        write_result(run_dir, identity, item, f"refused, nothing timed: {refused}", None)
        print(f"steady point {item.id}: refused, nothing timed: {refused}", file=sys.stderr)
        return REFUSED
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
    _write(run_dir / POINT_FILE, record)

    files = steady.files_in(run_dir)
    row = steady_spec.own_row(files, item)
    why = row if isinstance(row, str) else steady_spec.point_why(row, files, spec, item, own=True)
    _write(run_dir / VERDICT_FILE, {
        "identity": identity, "point": item.id, "passed": why is None, "why": why,
        "verdict": None if isinstance(row, str) else steady.verdict(row, files).record()})
    write_result(run_dir, identity, item, why, None if isinstance(row, str) else row, record["timed"].get("host"))
    print(f"steady point {item.id}: " + ("passed" if why is None else f"not passed: {why}"))
    return 0 if why is None else steady_spec.NOT_PASSED


def parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m verity_bench.steady_node", description=__doc__.split("\n\n")[0])
    parser.add_argument("--spec", required=True)
    parser.add_argument("--point", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--inside", type=Path, default=None, metavar="SCRATCH",
                        help="the timed step, inside the lease the point took: the scratch directory its statement is in")
    return parser.parse_args(argv)


def _stopped(signum: int, frame: Any) -> None:
    raise SystemExit(STOPPED)


def stop_on_signals() -> None:
    """TERM (a cancel, a timeout) and INT end the run with `STOPPED` from wherever it is, so that `point` removes its
    scratch on the way out: a statement left in the scratch stays on the node, in memory where the scratch is one."""

    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, _stopped)


def main(argv: Sequence[str] | None = None) -> int:
    a = parse(argv)
    stop_on_signals()
    try:
        settings.validate(of_record=True)
        return point(a)
    except ValueError as error:
        print(f"steady point {a.point}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
