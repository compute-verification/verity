"""Resolve benchmark jobs, report completion, and dispatch recorded points."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_config import dumps, sha256, value

from verity_bench.vllm.presets import ResolvedSpec, SweepPoint, expand_sweep, resolve_spec

PASS_QUERY = """
SELECT id
FROM attempts
WHERE validation = 'passed' AND source_commit = '{commit}'
  AND EXISTS (
    SELECT 1 FROM attempt_params
    WHERE attempt = attempts.id AND key = 'identity' AND value = '{identity}'
  )
ORDER BY start_utc DESC, id DESC
LIMIT 1
""".strip()


@dataclass(frozen=True)
class Toolchain:
    commit: str
    uv_lock_sha256: str
    vllm_pin: str


@dataclass(frozen=True)
class PlannedPoint:
    point: SweepPoint
    identity: str
    passing_run: str | None


@dataclass(frozen=True)
class Plan:
    spec: ResolvedSpec
    spec_sha256: str
    toolchain: Toolchain
    points: tuple[PlannedPoint, ...]


CompletionLookup = Callable[[str, str], str | None]
Executor = Callable[[Sequence[str], Path], int]


def repository_root(start: str | Path | None = None) -> Path:
    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "uv.lock").is_file() and (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError(f"{here}: no Verity workspace root")


def _commit(root: Path) -> str:
    recorded = os.environ.get("RESEARCH_SOURCE_SHA")
    if recorded:
        return recorded
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def toolchain(root: Path, spec: ResolvedSpec, *, commit: str | None = None) -> Toolchain:
    return Toolchain(commit or _commit(root), _file_sha256(root / "uv.lock"), spec.preset.vllm_pin)


def point_identity(spec_sha256: str, pins: Toolchain, point: SweepPoint) -> str:
    return sha256({
        "resolved_spec_sha256": spec_sha256,
        "commit": pins.commit,
        "uv_lock_sha256": pins.uv_lock_sha256,
        "vllm_pin": pins.vllm_pin,
        "point": point.to_json(),
    })


def store_completion(identity: str, commit: str) -> str | None:
    """Newest passing attempt with this exact point identity, from the evidence index."""

    if not re.fullmatch(r"[0-9a-f]{64}", identity) or not re.fullmatch(r"[0-9a-f]{7,64}", commit):
        raise ValueError("benchmark completion lookup requires hexadecimal identities")
    query = PASS_QUERY.format(identity=identity, commit=commit)
    process = subprocess.run(
        ["research", "data", "sql", query, "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(f"could not query benchmark completion: {process.stderr.strip()}")
    rows = json.loads(process.stdout)
    return str(rows[0]["id"]) if rows else None


def make_plan(
    source: str | Path,
    *,
    root: Path | None = None,
    commit: str | None = None,
    lookup: CompletionLookup = store_completion,
) -> Plan:
    root = root or repository_root()
    spec = resolve_spec(source)
    pins = toolchain(root, spec, commit=commit)
    digest = spec.config_sha256
    points = tuple(
        PlannedPoint(point, identity, lookup(identity, pins.commit))
        for point in expand_sweep(spec.sweep.id)
        for identity in (point_identity(digest, pins, point),)
    )
    return Plan(spec, digest, pins, points)


def plan_json(plan: Plan) -> dict[str, Any]:
    return {
        "schema": "verity-bench/plan/v0",
        "resolved_spec_sha256": plan.spec_sha256,
        "resolved_spec": value(plan.spec),
        "toolchain": value(plan.toolchain),
        "points": [
            {"point": item.point.to_json(), "identity": item.identity, "passing_run": item.passing_run}
            for item in plan.points
        ],
    }


def _relative_spec(source: str | Path, root: Path) -> str:
    path = Path(source)
    if not path.is_file():
        path = root / "benchmarks" / "specs" / f"{source}.toml"
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        raise ValueError("a recorded benchmark spec must be inside the source tree") from None


QUEUE_KINDS = {1: "vllm-bench-1gpu", 8: "vllm-bench-8gpu"}


def queue_kind(gpu_count: int) -> str:
    """The queue kind (infra/cluster/kinds/benchmarks.toml) whose shape gives a point's job its CPUs and memory."""

    try:
        return QUEUE_KINDS[gpu_count]
    except KeyError:
        raise ValueError(
            f"no queue kind for a {gpu_count}-GPU spec: register one in infra/cluster/kinds/benchmarks.toml "
            "and name it in verity_bench.jobs.QUEUE_KINDS"
        ) from None


def research_command(plan: Plan, source: str | Path, item: PlannedPoint, root: Path) -> list[str]:
    spec = plan.spec
    command = [
        "research", "run",
        "--queue",
        "--kind", queue_kind(spec.preset.gpu_count),
        # A guest's GPU job must be; a stopped point reruns whole when requeued, and resume skips only passing points.
        "--preemptible",
        "--source", str(root),
        "--project", "verity",
        "--campaign", "vllm-bench",
        "--gpus", str(spec.preset.gpu_count),
        "--gpu-model", spec.preset.gpu_id,
        "--cwd", "source",
        "--require-result",
        "--tool", "verity_bench.tool:VLLM_BENCH_POINT",
    ]
    if spec.admission.max_minutes is not None:
        command += ["--max-min", str(spec.admission.max_minutes)]
    if os.environ.get("VERITY_BENCH_VENV"):
        command += ["--env", f"VERITY_BENCH_VENV={os.environ['VERITY_BENCH_VENV']}"]
    command += [
        "--",
        "bash", "benchmarks/harness/run_point.sh",
        "--spec", _relative_spec(source, root),
        "--point", item.point.id,
        "--identity", item.identity,
    ]
    return command


def _execute(command: Sequence[str], root: Path) -> int:
    return subprocess.run(command, cwd=root, check=False).returncode


def dispatch(plan: Plan, source: str | Path, *, root: Path | None = None, execute: Executor = _execute) -> int:
    """Launch one recorded research run for each incomplete point."""

    root = root or repository_root()
    for item in plan.points:
        if item.passing_run is not None:
            continue
        rc = execute(research_command(plan, source, item, root), root)
        if rc:
            return rc
    return 0


def write_resolved(plan: Plan, path: Path) -> None:
    path.write_bytes(dumps(plan_json(plan)) + b"\n")
