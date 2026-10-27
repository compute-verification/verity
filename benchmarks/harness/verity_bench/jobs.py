"""Resolve benchmark jobs, report completion, and dispatch recorded points."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_config import dumps, sha256, value

from verity_bench import settings
from verity_bench.vllm.presets import ResolvedSpec, SweepPoint, expand_sweep, resolve_spec


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
    recorded = settings.RESEARCH_SOURCE_SHA
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


# A point's runs start after its commit exists, and run ids begin with their UTC start; a day's margin covers clocks that
# disagree.
COMMIT_MARGIN_S = 86_400
PULL_JOBS = 64


def _show(root: Path, commit: str, field: str) -> str:
    process = subprocess.run(["git", "show", "-s", f"--format={field}", commit], cwd=root, capture_output=True, text=True)
    if process.returncode != 0:
        raise ValueError(f"git doesn't know commit {commit} here, so its runs can't be looked up: fetch it")
    return process.stdout.strip()


class StoreCompletion:
    """The newest attempt `research data lookup` finds for a point at its commit's tree, with validation passed and rc 0
    (a killed run's result.json may read as passed; a dirty tree has no reuse key): from this machine's catalog, else,
    unless `local_only`, from the store's remote, pulled once per commit since `COMMIT_MARGIN_S` before it; no reindex."""

    def __init__(self, store: Any = None, *, root: Path | None = None, local_only: bool = False,
                 log: Callable[[str], None] | None = None):
        self._store, self.root, self.local_only = store, root, local_only
        self._log = log or (lambda message: print(f"verity-bench: {message}", file=sys.stderr))
        self._pulled: set[str] = set()

    @property
    def store(self) -> Any:
        if self._store is None:
            from research.store.local import LocalStore

            self._store = LocalStore()
        return self._store

    def __call__(self, identity: str, commit: str) -> str | None:
        if not re.fullmatch(r"[0-9a-f]{64}", identity) or not re.fullmatch(r"[0-9a-f]{7,64}", commit):
            raise ValueError("benchmark completion lookup requires hexadecimal identities")
        found = self._local(identity, commit)
        if found is None and not self.local_only and commit not in self._pulled:
            self.pull_since(int(_show(self.root or repository_root(), commit, "%ct")))
            self._pulled.add(commit)
            found = self._local(identity, commit)
        return found

    def _local(self, identity: str, commit: str) -> str | None:
        from research.store import cli_prov, tools_registry

        root = self.root or repository_root()
        attempts = cli_prov.lookup(tools_registry.load("verity_bench.tool:VLLM_BENCH_POINT", root), root, store=self.store,
                                   params={"identity": identity}, source_tree=_show(root, commit, "%T"))["attempts"]
        passing = [a["id"] for a in attempts if a["status"].get("validation") == "passed" and a["status"].get("rc") == 0]
        return passing[-1] if passing else None

    def pull_since(self, started: int) -> int:
        """Fetch and index the remote's attempts started since `started` (Unix seconds, less `COMMIT_MARGIN_S`) that this
        store lacks; how many it indexed."""

        from research.store import ids
        from research.store.local import StoreError

        store, remote = self.store, self._remote()
        since = time.strftime("r%Y%m%d", time.gmtime(started - COMMIT_MARGIN_S))
        listed = [key[len("attempts/"):-len(".json")]
                  for key, _ in remote.list_times("attempts/", start_after=f"attempts/{since}")]
        missing = [run for run in listed if ids.is_run_id(run) and not store.attempt_path(run).exists()]
        scratch = Path(tempfile.mkdtemp(prefix=".verity-bench-pull-", dir=store.root / "scratch"))

        def get(run: str) -> tuple[str, bytes | None]:
            target = scratch / f"{run}.json"
            try:
                remote.get(f"attempts/{run}.json", target)
            except KeyError:
                return run, None
            return run, target.read_bytes()

        indexed, skipped = 0, []
        try:
            with ThreadPoolExecutor(max_workers=PULL_JOBS) as pool:
                for run, data in pool.map(get, missing):
                    try:
                        if data is None:
                            raise ValueError("vanished from the remote")
                        store.put_attempt(json.loads(data))
                        indexed += 1
                    except (ValueError, StoreError) as error:
                        skipped.append(f"{run}: {error}")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        self._log(f"pulled {indexed} attempts started since {since} from {remote.describe()} "
                  f"({len(listed) - len(missing)} of {len(listed)} were here)")
        for line in skipped[:5]:
            self._log(f"skipped attempt {line}")
        return indexed

    def pull_labels(self) -> int:
        """Fetch every label the remote has and this store lacks (one listing of the remote's labels); how many."""

        from research.store import labels_sync

        remote = self._remote()
        report = labels_sync.sync(self.store, push=False)
        self._log(f"pulled {len(report['pulled'])} labels from {remote.describe()} "
                  f"({report['already']} of {report['remote_before']} were here)")
        for line in report["bad"][:5]:
            self._log(f"skipped label {line}")
        return len(report["pulled"])

    def _remote(self) -> Any:
        remote = self.store.remote
        if remote is None:
            why = getattr(self.store, "remote_error", None) or "no remote is configured"
            raise RuntimeError(f"the evidence store has no remote here ({why}): a point another machine ran would read as "
                               "missing; pass --local-only to read this machine's catalog alone")
        return remote


def make_plan(
    source: str | Path,
    *,
    root: Path | None = None,
    commit: str | None = None,
    lookup: CompletionLookup | None = None,
) -> Plan:
    root = root or repository_root()
    lookup = lookup or StoreCompletion(root=root)
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


QUEUE_KINDS = {1: "vllm-bench-1gpu"}
# A point writes its records, out/ and what its setup builds (a few MB; the venv and model cache are shared). The
# queue's default 20 GB request would hold every point once the node's disk nears research disk-hold's 80%.
POINT_DISK_GB = 5


def queue_kind(gpu_count: int) -> str:
    """The queue kind (infra/cluster/kinds/benchmarks.toml) whose shape gives a point's job its CPUs and memory."""

    try:
        return QUEUE_KINDS[gpu_count]
    except KeyError:
        raise ValueError(
            f"no queue kind for a {gpu_count}-GPU spec: register one in infra/cluster/kinds/benchmarks.toml "
            "and name it in verity_bench.jobs.QUEUE_KINDS"
        ) from None


def point_environment(spec: ResolvedSpec) -> dict[str, str]:
    """What a point's job must run under: the spec's `env`, and its `setup` script (`run_point.sh` sources it before the
    point's Python starts) as VERITY_BENCH_SETUP."""

    return {**spec.env, **({"VERITY_BENCH_SETUP": spec.setup} if spec.setup else {})}


def check_point_environment(spec: ResolvedSpec, environ: Mapping[str, str]) -> None:
    expected = point_environment(spec)
    wrong = sorted(name for name, want in expected.items() if environ.get(name) != want)
    if "VERITY_BENCH_SETUP" not in expected and environ.get("VERITY_BENCH_SETUP"):
        wrong.append("VERITY_BENCH_SETUP")
    if wrong:
        raise ValueError(f"the point's environment differs from spec {spec.id!r}'s env and setup in {wrong}: "
                         "launch points with `verity-bench run SPEC`, which forwards them")


#: the GPU nodes whose GPUs `--direct` may take: each has one slot per GPU in GPU_SLOTS (the nebius nodes g0-g7, the
#: single-GPU DC VMs g0)
DIRECT_NODES = ("vy-nebius-1", "vy-nebius-2", "vy-nebius-3", "vy-dc-h200", "vy-dc-h200-2", "vy-dc-b200")
GPU_SLOTS = "/workspace/research/locks/gpu/slots"


def _env_flags(spec: ResolvedSpec) -> list[str]:
    flags = []
    if venv := settings.VERITY_BENCH_VENV:
        flags += ["--env", f"VERITY_BENCH_VENV={venv}"]
    for name, setting in point_environment(spec).items():
        flags += ["--env", f"{name}={setting}"]
    return flags


def _point_command(plan: Plan, source: str | Path, item: PlannedPoint, root: Path) -> list[str]:
    return [
        "bash", "benchmarks/harness/run_point.sh",
        "--spec", _relative_spec(source, root),
        "--point", item.point.id,
        "--identity", item.identity,
    ]


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
        "--disk-gb", str(POINT_DISK_GB),
        "--cwd", "source",
        "--require-result",
        "--tool", "verity_bench.tool:VLLM_BENCH_POINT",
    ]
    if spec.admission.max_minutes is not None:
        command += ["--max-min", str(spec.admission.max_minutes)]
    return [*command, *_env_flags(spec), "--", *_point_command(plan, source, item, root)]


def direct_minutes(spec: ResolvedSpec, node: str) -> float:
    """The spec's admission max_minutes, which bounds a direct point's run, slot and lease, once `node` is a GPU node."""

    if node not in DIRECT_NODES:
        raise ValueError(f"--direct {node}: not a GPU node with gpu slots ({', '.join(DIRECT_NODES)}), so a point there "
                         f"gets no GPU; a GPU node that has {GPU_SLOTS} joins verity_bench.jobs.DIRECT_NODES")
    if spec.admission.max_minutes is None:
        raise ValueError(f"--direct needs spec {spec.id!r}'s admission max_minutes: it bounds the run, its slot and its "
                         "GPU lease")
    return spec.admission.max_minutes


def direct_command(plan: Plan, source: str | Path, item: PlannedPoint, root: Path, node: str) -> list[str]:
    """The point as a `research run --on NODE` through one of the node's GPU slots, its GPUs from the node's gpu-lease:
    the queued point's tool, identity and environment, for a node whose launcher takes only slot runs."""

    spec = plan.spec
    minutes = direct_minutes(spec, node)
    return [
        "research", "run",
        "--on", node,
        "--project", "verity",
        "--campaign", "vllm-bench",
        "--source", str(root),
        "--cwd", "source",
        "--timeout", f"{minutes:g}m",
        "--require-result",
        "--tool", "verity_bench.tool:VLLM_BENCH_POINT",
        *_env_flags(spec),
        "--",
        "env", f"CHECK_SLOTS={GPU_SLOTS}", "python3", "tools/verity/check/slot.py", "--by", "benchmarks",
        "--run-max", f"{minutes * 60:g}",
        "--",
        # Without --max-min a lease has no end, which gpu-lease refuses while a booked period lies ahead; a guest's GPU
        # job is preemptible, as the queued point's is.
        "gpu-lease", str(spec.preset.gpu_count), "--wait", "--max-min", f"{minutes:g}", "--preemptible",
        "--",
        *_point_command(plan, source, item, root),
    ]


def _execute(command: Sequence[str], root: Path) -> int:
    return subprocess.run(command, cwd=root, check=False).returncode


def dispatch(plan: Plan, source: str | Path, *, root: Path | None = None, execute: Executor = _execute,
             direct: str | None = None) -> int:
    """Launch one recorded research run for each incomplete point: queued, or with `direct` on that node's GPU slots."""

    root = root or repository_root()
    if direct is not None:
        direct_minutes(plan.spec, direct)
    for item in plan.points:
        if item.passing_run is not None:
            continue
        command = (research_command(plan, source, item, root) if direct is None
                   else direct_command(plan, source, item, root, direct))
        rc = execute(command, root)
        if rc:
            return rc
    return 0


def write_resolved(plan: Plan, path: Path) -> None:
    path.write_bytes(dumps(plan_json(plan)) + b"\n")
