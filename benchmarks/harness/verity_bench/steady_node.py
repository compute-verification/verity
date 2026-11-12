"""A steady-state point's node side: one row of a steady-state spec (`steady_spec`) at one prover count, timed on one
card of a GPU node. `steady_point.sh` runs it from the shipped source, in the environment the spec's setup leaves, under
the `research run` that `steady_plan` launches through one of the node's GPU slots.

    python -m verity_bench.steady_node --spec SPEC --point POINT --identity IDENTITY

1. **Before any card is asked for** it holds the run to its point: the identity it was launched with is the one this
   tree gives the point (`steady_spec.point_identity`: a changed file of the closure is another point); the run's
   environment is the spec's; the prover is the one the setup built (`VERITY_BENCH_STEADY_BIN`); and the run's CPUs are
   exactly the spec's `cores`, which the node's slot gives. It stages the row's statement with
   `benchmarks/served_units/stage.py`, or copies the one the row names as `staged`, and refuses a statement whose
   circuit isn't the row's `pin` or whose units aren't its `n`.
2. **One lease of one card** (`gpu-lease 1 --wait --max-min M`), and inside it the timed step (this module again, with
   `--inside`): it reads the card, the driver's version and the toolkit's on the node and refuses a card or a version
   that isn't the spec's; records what else is on the node (`steady.card_file`) before and after;
   runs `benchmarks/one_stage/server_rate.py` at the point's load (`steady_spec.timed_sessions`); and keeps the row,
   with the versions it read (`steady_spec.VERSIONS`) and the period its sessions were sized by (`steady.PERIOD`, which
   `level` reads), and the sessions' record under `out/rate/`.
3. **After the lease has ended**, so that its usage line is among the run's files, the verdict: the row is a passed point
   when `steady_spec.point_why` finds nothing against it, from the run's own files. `result.json` passes with it.

It exits 0 for a passed point, `steady_spec.NOT_PASSED` for a point that was timed and didn't pass, and `REFUSED` where
nothing was timed. Its records are `out/point.json` (the point, the prover, the cores, the card and the versions),
`out/verdict.json` (each check's answer) and `out/rate/`; `result.json` lists every file a verdict reads (`kept`), so
the store holds what `plan` judges the run by. The statement and the provers' session files stay in a scratch directory
that it removes, also when the run is cancelled or timed out: TERM and INT end it with `STOPPED`, after the command
it was waiting for (`_run`) and with no result.
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

from verity_bench import jobs, settings, steady, steady_spec
from verity_bench.vllm.presets import GPUS

RESULT_SCHEMA = "research/result/v0.1"
KIND = "verity-steady-point/v1"
POINT_FILE, VERDICT_FILE, TIMED_FILE = "out/point.json", "out/verdict.json", "out/timed.json"
REFUSED = 4
#: the exit code of a run that TERM or INT ended: 128 and TERM's number, for both
STOPPED = 128 + signal.SIGTERM
#: the seconds a command this process was waiting for has to end, once sent TERM, before it is killed
STOP_S = 30
#: the files of a staged statement the node side reads; the prover reads the rest
STAGED = ("stage.json", "circuit.txt", "pin.txt")
#: the bytes kept of each of the server's logs, from its end
KEPT_LOG_BYTES = 200_000

#: a command's exit code, its output to a log file
Runner = Callable[[Sequence[str], Path], int]
#: a command's exit code and what it printed
Asker = Callable[[Sequence[str]], tuple[int, str]]


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
        return 127, f"{type(error).__name__}: {error}"
    return process.returncode, process.stdout


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


def statement(row: steady_spec.SteadyRow, scratch: Path, out: Path, run: Runner) -> dict[str, Any]:
    """The row's statement where `server_rate.py` looks for it (`SCRATCH/z/NAME/cN/stage`): staged here, or a copy of
    the one the row names. Its circuit's SHA-512 is computed on that copy, never taken from a file beside it."""

    stage = scratch / "z" / row.name / f"c{row.n}" / "stage"
    stage.parent.mkdir(parents=True)
    if row.staged is not None:
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


def lease_argv(spec: steady_spec.SteadySpec, inside: Sequence[str]) -> list[str]:
    """The timed step under one lease of one card, for the spec's bound: preemptible, as a guest's GPU job must be (a
    preempted point has no passing run, so the next `run` launches it again)."""

    minutes = spec.admission.max_minutes
    if minutes is None:
        raise Refused(f"spec {spec.id!r} has no admission max_minutes, which bounds the point's lease")
    memory = ["--mem-gb", str(spec.admission.mem_gb)] if spec.admission.mem_gb is not None else []
    return ["gpu-lease", "1", "--wait", "--max-min", str(minutes), "--preemptible", *memory, "--", *inside]


# ---- inside the lease ---------------------------------------------------------------------------------------------------------


def card_on_node(device: str, ask: Asker) -> dict[str, str]:
    """The leased card's driver version, name and UUID, as nvidia-smi gives them."""

    rc, text = ask(["nvidia-smi", "-i", device, "--query-gpu=driver_version,name,uuid", "--format=csv,noheader"])
    fields = [field.strip() for field in text.strip().splitlines()[0].split(",")] if rc == 0 and text.strip() else []
    if len(fields) != 3 or not all(fields):
        raise Refused(f"nvidia-smi doesn't say what GPU {device} is (exit {rc}: {text.strip()[:200]!r})")
    return dict(zip(("driver", "name", "uuid"), fields, strict=True))


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
          run: Runner = _run, ask: Asker = _ask, now: Callable[[], float] = time.time) -> int:
    """The timed step, inside the lease: the script's exit code, or `REFUSED` with the reason in `TIMED_FILE`."""

    out = run_dir / "out"
    record: dict[str, Any] = {"refused": None}
    try:
        device = settings.CUDA_VISIBLE_DEVICES
        if not device or "," in device:
            raise Refused(f"the timed step runs under a gpu-lease of one card (CUDA_VISIBLE_DEVICES is {device!r})")
        card = card_on_node(device, ask)
        read = {"driver": card["driver"], "cuda": toolkit_on_node(ask)}
        record |= {"device": device, "card": {"name": card["name"], "uuid": card["uuid"]}, steady_spec.VERSIONS: read}
        if GPUS[spec.card].nvidia_smi_name not in card["name"]:
            raise Refused(f"the lease's card is a {card['name']}, not the `card` of spec {spec.id!r} ({spec.card}: "
                          f"{GPUS[spec.card].nvidia_smi_name})")
        if read != steady_spec.versions(spec):
            raise Refused(f"the node's driver and CUDA toolkit are {read}, and spec {spec.id!r} says "
                          f"{steady_spec.versions(spec)}: a row at other versions is another point. Set the spec's "
                          "`driver` and `cuda` to the node's, which times its rows again there")
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
                 row: Mapping[str, Any] | None) -> None:
    rate = row.get("statements_per_s") if row is not None and why is None else None
    evidence = [path for path in (steady.RATES, VERDICT_FILE, POINT_FILE) if (run_dir / path).is_file()]
    _write(run_dir / "result.json", {
        "schema": RESULT_SCHEMA, "kind": KIND, "run_id": settings.RESEARCH_RUN_ID,
        "workload_fingerprint": {"identity": identity, "point": item.to_json()},
        "validation": {"status": "passed" if why is None else "failed", "checker": "verity_bench.steady_spec.point_why",
                       "detail": why or f"{item.id}: the ten checks pass", "evidence": evidence},
        "measurements": [] if rate is None else [{"name": "statements_per_s", "value": rate, "unit": "1/s"}],
        "measurement_files": [steady.RATES] if (run_dir / steady.RATES).is_file() else [], "artifacts": kept(run_dir)})


def point(a: argparse.Namespace, *, run: Runner = _run, ask: Asker = _ask, now: Callable[[], float] = time.time,
          cpus: Callable[[], Sequence[int]] = lambda: sorted(os.sched_getaffinity(0))) -> int:
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
        return timed(spec, item, a.inside, run_dir, run=run, ask=ask, now=now)

    # a requeued point reruns whole in its run directory: no earlier attempt's verdict or record stands beside this one's
    for name in ("result.json", POINT_FILE, VERDICT_FILE, TIMED_FILE, *(steady.card_file(item.to_json(), when)
                                                                        for when in ("start", "end"))):
        (run_dir / name).unlink(missing_ok=True)
    shutil.rmtree(out / "rate", ignore_errors=True)
    record: dict[str, Any] = {"schema": "verity-bench/steady-point/v0", "spec": spec.id, "point": item.to_json(),
                              "identity": identity, "files_sha256": files_sha256}
    scratch = None
    try:
        record["prover"] = prover()
        record["cores"] = {"count": spec.cores, "list": _spans(own_cores(spec, cpus()))}
        scratch = Path(tempfile.mkdtemp(prefix="verity-bench-steady-", dir=settings.VERITY_BENCH_STEADY_SCRATCH))
        lease = lease_argv(spec, [sys.executable, "-m", "verity_bench.steady_node", "--spec", a.spec, "--point", a.point,
                                  "--identity", a.identity, "--inside", str(scratch)])
        record["statement"] = statement(item.row, scratch, out, run)
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
    write_result(run_dir, identity, item, why, None if isinstance(row, str) else row)
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
