"""A steady-state spec's store side: `plan`, `check` and `run` for a spec of `steady_spec.SCHEMA`. It runs on any machine
that reaches the store: no GPU, no node.

**A point's passing run** (`SteadyCompletion`) is the newest run whose row is a passed point of the spec
(`steady_spec.point_why`), judged now, from the run's files in the store, by the rules as this checkout has them: a
verdict a run wrote of itself is never read. Two kinds of run can be it:

- **a run of the point** (`tool.STEADY_POINT` with the point's identity, at any commit: the identity digests what the
  time depends on), which ran to its end (exit 0 or `steady_spec.NOT_PASSED`) from a clean tree;
- **a labelled row**: a row of a run of the workload's steady-state record (`prove_records.STEADY`: a labelled run, or a
  run a labelled art names), timed before the harness or outside it, whose `pin`, `provers` and `cores.count` are the
  point's. `LABELLED_ROW` says which row of such a run is judged against its files.

A point with no passing run is launched again by `run`, and `plan` says why each run that could have been it isn't
(`PlannedPoint.why`): the check it failed, or the one its files can't show.

**`run SPEC --direct NODE`** (`dispatch`) launches each point without a passing run as one recorded run on a GPU slot
of the node (`direct_command`), in the spec's order. A point that was timed and didn't pass doesn't stop the points
after it; a run that couldn't be launched, or that the node refused, does.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_config import sha256

from verity_bench import jobs, prove_records, served_table, steady, steady_spec
from verity_bench.steady_spec import SteadyPoint, SteadySpec

OWN, COMPLETED = "own", "completed"
#: Which row of a labelled run is judged against the run's files. `OWN`: the row in the run's own rate file, so a run
#: whose own row lacks what a check reads is "cannot tell" and is planned again. `COMPLETED`: where the record is an art
#: that holds the row as completed or corrected after the run (`prove_records.ART_RATES`), that row.
LABELLED_ROW = OWN
#: the runs `plan` names in a point's `why`, the newest first
WHYS = 3

#: a spec's point and its identity to its passing run and, without one, why the runs that could have been it aren't
Lookup = Callable[[SteadySpec, SteadyPoint, str], tuple[str | None, str | None]]


@dataclass(frozen=True)
class Sources:
    """What a steady-state point's identity pins beside its spec: the files of `steady_spec.closure`, by their digest."""

    files_sha256: str
    files: int


class RunFiles:
    """A run's files in the store as `steady.Files`, each fetched when it is first read."""

    def __init__(self, store: Any, art: str):
        self.store, self.art = store, art
        self._read: dict[str, str | None] = {}

    def __call__(self, path: str) -> str | None:
        if path not in self._read:
            held = self.store.listing(self.art, [path])
            self._read[path] = ((Path(self.store.fetch(self.art, paths=[path])) / path).read_text(errors="replace")
                                if held else None)
        return self._read[path]


def run_files(store: Any, run: str) -> RunFiles | None:
    """The files of a run that kept some, from this machine's store or its remote."""

    attempt = store.get_attempt(run)
    outputs = (attempt or {}).get("outputs") or {}
    art = outputs.get("run_files") or outputs.get("run_record")
    return RunFiles(store, art) if art else None


def _superseded(store: Any, target: str) -> bool:
    return bool(store.labels(target, key=served_table.SUPERSEDED_BY))


def labelled_rows(store: Any, record: str) -> list[tuple[str, Mapping[str, Any], RunFiles]]:
    """Each row of the record's runs, with its run and the run's files: a labelled run's own rows; the rows of each run
    a labelled art names; and for an art that holds one run's rows itself, that run's own rows or the art's
    (`LABELLED_ROW`). A superseded target, a run that kept no files and a rate file that can't be read give none."""

    found: list[tuple[str, Mapping[str, Any], RunFiles]] = []
    for target in served_table.records(store).get(record, []):
        if _superseded(store, target):
            continue
        completed = None
        if target.startswith("art:"):
            manifest = store.get_manifest(target)
            if not manifest.meta.get("runs"):
                continue
            runs = [run for run, _ in served_table.record_runs(store, [target]) if not _superseded(store, run)]
            if manifest.payload is not None and store.listing(target, [prove_records.ART_RATES]):
                if len(runs) != 1:
                    continue
                if LABELLED_ROW == COMPLETED:
                    path = prove_records.ART_RATES
                    completed = json.loads((Path(store.fetch(target, paths=[path])) / path).read_text())
        else:
            runs = [target]
        for run in runs:
            files = run_files(store, run)
            if files is None:
                continue
            rows = steady_spec.rows(files) if completed is None else completed
            if isinstance(rows, list):
                found += [(run, row, files) for row in rows if isinstance(row, Mapping)]
    return found


class SteadyCompletion:
    """The newest passing run of a steady-state point, from this machine's catalog and, unless `local_only`, the
    store's remote: its labels, and its attempts started since `since()` (when the spec was added: a point's identity
    names no commit, so every run since then can be the point's), pulled once."""

    def __init__(self, store: Any = None, *, root: Path | None = None, local_only: bool = False,
                 since: Callable[[], int] | None = None, log: Callable[[str], None] | None = None):
        self._base = jobs.StoreCompletion(store, root=root, local_only=local_only, log=log)
        self.root, self.local_only, self._since = root, local_only, since
        self._pulled = False
        self._labelled: dict[str, list[tuple[str, Mapping[str, Any], RunFiles]]] = {}

    @property
    def store(self) -> Any:
        return self._base.store

    def _pull(self) -> None:
        if self._pulled or self.local_only:
            return
        self._pulled = True
        self._base.pull_labels()
        if self._since is not None:
            self._base.pull_since(self._since())

    def _own(self, identity: str) -> list[str]:
        """The runs of the point that ran to their end from a clean tree, the newest first."""

        from research.store import cli_prov, tools_registry

        root = self.root or jobs.repository_root()
        attempts = cli_prov.lookup(tools_registry.load("verity_bench.tool:STEADY_POINT", root), root, store=self.store,
                                   params={"identity": identity}, any_source=True)["attempts"]
        return [a["id"] for a in reversed(attempts)
                if (a.get("status") or {}).get("rc") in (0, steady_spec.NOT_PASSED)
                and (a.get("source") or {}).get("dirty") is False]

    def __call__(self, spec: SteadySpec, item: SteadyPoint, identity: str) -> tuple[str | None, str | None]:
        self._pull()
        store, whys = self.store, []
        for run in self._own(identity):
            files = run_files(store, run)
            row = f"{steady.CANNOT_TELL}: the run kept no files" if files is None else steady_spec.own_row(files, item)
            why = row if isinstance(row, str) else steady_spec.point_why(row, files, spec, item, own=True)
            if why is None:
                return run, None
            whys.append(f"{run}: {why}")
        record = prove_records.WORKLOADS[spec.workload].record(prove_records.STEADY)
        if record not in self._labelled:
            self._labelled[record] = labelled_rows(store, record)
        for run, row, files in sorted(self._labelled[record], key=lambda found: found[0], reverse=True):
            if row.get("pin") != item.row.pin or row.get("provers") != item.provers:
                continue
            why = steady_spec.point_why(row, files, spec, item)
            if why is None:
                return run, None
            whys.append(f"{run} (labelled `{record}`): {why}")
        more = f"; and {len(whys) - WHYS} more" if len(whys) > WHYS else ""
        return None, ("; ".join(whys[:WHYS]) + more) or None


def _added(root: Path, source: str | Path) -> int:
    """When the spec's file was added to the tree: no run of its points is looked up from before."""

    path = jobs._relative_spec(source, root)
    if served_table.spec_path(served_table.spec_name(path)) != path:
        raise ValueError(f"{path} isn't a spec of benchmarks/specs, so its points' runs can't be looked up on the store's "
                         "remote since it was added: move it there, or pass --local-only")
    return served_table.spec_added(root, served_table.spec_name(path))


def make_plan(source: str | Path, *, root: Path | None = None, lookup: Lookup | None = None,
              local_only: bool = False) -> jobs.Plan:
    """The spec's points, each with its identity in this tree, its passing run and, without one, why. `lookup`: in
    place of the store's (`SteadyCompletion`, of this machine's catalog alone when `local_only`)."""

    root = root or jobs.repository_root()
    spec = steady_spec.load_spec(source, root=root)
    if spec.workload not in prove_records.WORKLOADS:
        raise ValueError(f"spec {spec.id!r}'s `workload` {spec.workload!r} has no steady-state record: the workloads are "
                         f"{', '.join(prove_records.WORKLOADS)} (`prove_records.WORKLOADS`)")
    files = steady_spec.closure(root, spec)
    digest = sha256(files)
    if lookup is None:
        lookup = SteadyCompletion(root=root, local_only=local_only, since=lambda: _added(root, source))
    points = []
    for item in spec.points:
        identity = steady_spec.point_identity(spec, item, digest)
        run, why = lookup(spec, item, identity)
        points.append(jobs.PlannedPoint(item, identity, run, why))
    return jobs.Plan(spec, sha256(spec, identity=True), Sources(digest, len(files)), tuple(points))


def direct_command(plan: jobs.Plan, source: str | Path, item: jobs.PlannedPoint, root: Path, node: str) -> list[str]:
    """The point as a `research run --on NODE` through one of the node's GPU slots, which gives it its cores. The node
    side takes the card itself, in one lease, so that the lease has ended and left its usage line among the run's
    files before the run's verdict is computed."""

    spec = plan.spec
    minutes = jobs.direct_minutes(spec, node)
    return [
        "research", "run",
        "--on", node,
        "--project", "verity",
        "--campaign", "steady-point",
        "--source", str(root),
        "--cwd", "source",
        "--timeout", f"{minutes:g}m",
        "--require-result",
        "--tool", "verity_bench.tool:STEADY_POINT",
        *[flag for name, setting in jobs.point_environment(spec).items() for flag in ("--env", f"{name}={setting}")],
        "--",
        "env", f"CHECK_SLOTS={jobs.GPU_SLOTS}", "python3", "tools/verity/check/slot.py", "--by", "benchmarks",
        "--run-max", f"{minutes * 60:g}",
        "--",
        "bash", "benchmarks/harness/steady_point.sh",
        "--spec", jobs._relative_spec(source, root),
        "--point", item.point.id,
        "--identity", item.identity,
    ]


def dispatch(plan: jobs.Plan, source: str | Path, *, root: Path | None = None, execute: jobs.Executor = jobs._execute,
             direct: str | None = None) -> int:
    """Launch one recorded run for each point without a passing run, on `direct`'s GPU slots, in the spec's order. A
    point timed and not passed goes on to the next; any other failure stops there. The first nonzero exit, or 0."""

    root = root or jobs.repository_root()
    if direct is None:
        raise ValueError(f"spec {plan.spec.id!r} is a steady-state spec: its points run on a node's own card, through a "
                         "GPU slot of it: `verity-bench run SPEC --direct NODE`")
    jobs.direct_minutes(plan.spec, direct)
    first = 0
    for item in plan.points:
        if item.passing_run is not None:
            continue
        rc = execute(direct_command(plan, source, item, root, direct), root)
        first = first or rc
        if rc not in (0, steady_spec.NOT_PASSED):
            break
    return first

