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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from verity_config import dumps, sha256, value

from verity_bench import kit, settings
from verity_bench.steady_spec import SteadyPoint, SteadySpec
from verity_bench.vllm.presets import ResolvedSpec, SweepPoint, expand_sweep, former_name, resolve_spec


@dataclass(frozen=True)
class Toolchain:
    commit: str
    uv_lock_sha256: str
    vllm_pin: str


@dataclass(frozen=True)
class PlannedPoint:
    """A point of a spec with its identity and its passing run. `why`: for a point with none, why the runs that could
    have been it aren't (a steady-state point's; a vLLM point's runs either passed or aren't found). `again`: named by
    `--again` (`again`), so `run` launches it though a passing run stands for it."""

    point: SweepPoint | SteadyPoint
    identity: str
    passing_run: str | None
    why: str | None = None
    again: bool = False


@dataclass(frozen=True)
class Plan:
    """A spec's points, each with its passing run. `toolchain`: what a point's identity pins beside its spec, a vLLM
    spec's `Toolchain` or a steady-state spec's files (`steady_plan.Sources`)."""

    spec: ResolvedSpec | SteadySpec
    spec_sha256: str
    toolchain: Any
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
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        raise ValueError(f"{root} is no git checkout and RESEARCH_SOURCE_SHA is unset, so nothing says which commit this "
                         "tree is: run under `research run`, which sets it (--tree-sha for a tree without .git)") from None


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
    (a killed run's result.json may read as passed; a dirty tree has no reuse key), and not started as a trial
    (`kit.trial_attempt`): from this machine's catalog, else,
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
        passing = [a["id"] for a in attempts if a["status"].get("validation") == "passed" and a["status"].get("rc") == 0
                   and not kit.trial_attempt(a)]
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
            {"point": item.point.to_json(), "identity": item.identity, "passing_run": item.passing_run,
             **({"why": item.why} if item.why else {}),
             **({"again": f"passed by {item.passing_run}"} if item.again and item.passing_run else {})}
            for item in plan.points
        ],
    }


def again(plan: Plan, names: Sequence[str]) -> Plan:
    """The plan with the points `names` names (`--again`, each by its id) launched again by `run`, though a passing run
    stands for one: a labelled row timed before the harness, or a run at another closure that the point's identity
    doesn't tell apart. Only named points: there is no form that launches every passed point again. The new run labels
    and supersedes nothing; the passing run stays the record until a run is labelled on purpose."""

    ids = [item.point.id for item in plan.points]
    if unknown := [name for name in names if name not in ids]:
        raise ValueError(f"--again {', '.join(unknown)}: not a point of spec {plan.spec.id!r}, whose points are "
                         f"{', '.join(ids)}")
    return replace(plan, points=tuple(replace(item, again=item.point.id in names) for item in plan.points))


def _relative_spec(source: str | Path, root: Path) -> str:
    path = Path(source)
    if not path.is_file():
        path = root / "benchmarks" / "harness" / "specs" / f"{former_name(source) or source}.toml"
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        raise ValueError("a recorded benchmark spec must be inside the source tree") from None


def point_environment(spec: ResolvedSpec | SteadySpec) -> dict[str, str]:
    """What a point's job must run under: the spec's `env`, and its `setup` script (`run_point.sh` sources it before the
    point's Python starts) as VERITY_BENCH_SETUP."""

    return {**spec.env, **({"VERITY_BENCH_SETUP": spec.setup} if spec.setup else {})}


def check_point_environment(spec: ResolvedSpec | SteadySpec, environ: Mapping[str, str]) -> None:
    expected = point_environment(spec)
    wrong = sorted(name for name, want in expected.items() if environ.get(name) != want)
    if "VERITY_BENCH_SETUP" not in expected and environ.get("VERITY_BENCH_SETUP"):
        wrong.append("VERITY_BENCH_SETUP")
    if wrong:
        raise ValueError(f"the point's environment differs from spec {spec.id!r}'s env and setup in {wrong}: "
                         "launch points with `verity-bench run SPEC`, which forwards them")


def lease_host() -> bool:
    """Whether gpu-lease hands out every GPU of this machine (research's `GPU_LEASE_HOST_FILE`, which the machine's setup
    writes): a run there sees no GPU until it takes a lease."""

    from research.remote import GPU_LEASE_HOST_FILE

    return Path(GPU_LEASE_HOST_FILE).exists()


def _lease(spec: ResolvedSpec) -> list[str]:
    """A vLLM point's wait for its cards where gpu-lease hands them out (`lease_host`): the spec's GPUs for its admission
    max_minutes, of the cards GPU_LEASE_ON lists and on the CPUs GPU_LEASE_CORES lists where they are set, preemptible as
    a guest's GPU job must be. Elsewhere none, and none in a job that holds its lease already (GPU_LEASE_UUID: the
    queue's, `research run --gpus N`'s): the point runs on the cards its job was given."""

    if settings.GPU_LEASE_UUID or not lease_host():
        return []
    if spec.admission.max_minutes is None:
        raise ValueError(f"spec {spec.id!r} has no admission max_minutes, which bounds the point's GPU lease")
    *lease, end = kit.lease(spec.preset.gpu_count, spec.admission.max_minutes, on=settings.GPU_LEASE_ON,
                            cores=settings.GPU_LEASE_CORES)
    return [*lease, "--preemptible", end]


def _steady_card() -> list[str]:
    """A steady-state point's card where GPU_LEASE_ON names one (`steady_node --card`): its node side takes the lease."""

    on = settings.GPU_LEASE_ON
    if on is None:
        return []
    if not on.isdigit():
        raise ValueError(f"GPU_LEASE_ON={on!r}: a steady-state point's lease is of one card, named by its index")
    return ["--card", on]


def point_command(plan: Plan, source: str | Path, item: PlannedPoint, root: Path) -> list[str]:
    """The point in this process's own run, on the machine it was started on, with the spec's env and setup: a vLLM
    point's `run_point.sh` in its lease (`_lease`), a steady-state point's `steady_point.sh`, which takes its own."""

    env = ["env", *(f"{name}={setting}" for name, setting in point_environment(plan.spec).items())]
    flags = ["--spec", _relative_spec(source, root), "--point", item.point.id, "--identity", item.identity]
    if isinstance(plan.spec, SteadySpec):
        return [*env, "bash", "benchmarks/harness/steady_point.sh", *flags, *_steady_card()]
    return [*_lease(plan.spec), *env, "bash", "benchmarks/harness/run_point.sh", *flags]


def booked(end: float | None, cores: str | None) -> list[str]:
    """A booked period's launch: the run takes the slot of the booked `cores`, and isn't started unless the spec's
    max_minutes fit before `end`. `research run` has neither (`--pool` takes the first free slot, `--timeout` is a length,
    not an end), so it is the launcher's, not the harness's."""

    raise NotImplementedError("infra's launcher goal infra/booked-launch: `research run` taking a booked period, its end "
                              "and its slot by the slot's cores")


def vllm_spec(plan: Plan) -> ResolvedSpec:
    """A vLLM plan's spec; TypeError on a steady-state plan's, whose points `steady_plan` runs."""

    if not isinstance(plan.spec, ResolvedSpec):
        raise TypeError(f"{plan.spec.id} is a steady-state spec: steady_plan runs its points")
    return plan.spec


def to_run(plan: Plan, point: str | None = None) -> PlannedPoint:
    """The one point a run runs: `point`, else the spec's only point, else the one point `--again` names."""

    ids = [item.point.id for item in plan.points]
    if point is None:
        named = [item.point.id for item in plan.points if item.again]
        if len(ids) != 1 and len(named) != 1:
            raise ValueError(f"spec {plan.spec.id!r} has {len(ids)} points ({', '.join(ids)}) and a run runs one: name it "
                             "with --point")
        point = ids[0] if len(ids) == 1 else named[0]
    found = [item for item in plan.points if item.point.id == point]
    if not found:
        raise ValueError(f"--point {point}: not a point of spec {plan.spec.id!r}, whose points are {', '.join(ids)}")
    return found[0]


def _execute(command: Sequence[str], root: Path) -> int:
    return subprocess.run(command, cwd=root, check=False).returncode


def dispatch(plan: Plan, source: str | Path, *, point: str | None = None, root: Path | None = None,
             execute: Executor = _execute) -> int:
    """Run the spec's one point (`to_run`) in this process's own run, on the machine it was started on
    (`point_command`), unless a passing run stands for it and `--again` doesn't name it; the point's exit code, or 0."""

    root = root or repository_root()
    item = to_run(plan, point)
    if item.passing_run is not None and not item.again:
        print(f"verity-bench: point {item.point.id} of spec {plan.spec.id!r} has its passing run {item.passing_run}, so "
              f"nothing runs (--again {item.point.id} runs it again)", file=sys.stderr)
        return 0
    return execute(point_command(plan, source, item, root), root)


def write_resolved(plan: Plan, path: Path) -> None:
    path.write_bytes(dumps(plan_json(plan)) + b"\n")
