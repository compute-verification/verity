"""A steady-state spec's store side: `plan`, `check` and `run` for a spec of `steady_spec.SCHEMA`. It runs on any machine
that reaches the store: no GPU, no node.

**A point's passing run** (`SteadyCompletion`) is the newest run whose row is a passed point of the spec
(`steady_spec.point_why`), judged now, from the run's files in the store, by the rules as this checkout has them: a
verdict a run wrote of itself is never read. Two kinds of run can be it:

- **a run of the point** (`tool.STEADY_POINT` with the point's identity, at any commit: the identity digests what the
  time depends on), which ran to its end (exit 0 or `steady_spec.NOT_PASSED`) from a clean tree;
- **a labelled row**: a row of a run of the workload's steady-state record (`prove_records.STEADY`: a labelled run, or a
  run a labelled art names), timed before the harness or outside it, whose `pin`, `provers` and `cores.count` are the
  point's. The row judged against the run's files is the run's own, or, where a labelled art holds the run's rows as
  completed after it (`completed_rows`), the art's: such an art names that one run in its `runs`, carries the rows'
  derivation from the run's own records (`DERIVATION`), isn't superseded, is the one standing art that completes the
  run, and holds the run's own rows with nothing changed but what its derivation names (`completed`). So a run from
  before the timing script wrote a field a check reads counts on its completed row, and one with no such art is
  "cannot tell" and timed again.

A point with no passing run is launched again by `run`, and `plan` says why each run that could have been it isn't
(`PlannedPoint.why`): the check it failed, or the one its files can't show. A labelled run whose rows can't be read
at all (it kept no files, or its rate file is no list of rows) could have been any point, and is named in each.

**`run SPEC --direct NODE`** (`dispatch`) launches each point without a passing run as one recorded run on a GPU slot
of the node (`direct_command`), in the spec's order.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_config import sha256

from verity_bench import jobs, prove_records, served_table, steady, steady_spec
from verity_bench.steady_spec import SteadyPoint, SteadySpec

#: beside a labelled art's rows of a run (`prove_records.ART_RATES`): how each field the art added to the run's own rows
#: was derived from the run's own records. Its `rows` holds one record for each row, by the row's `type` and `provers`,
#: whose `fields` maps the fields it derived to how: a key names one field or several joined by commas, and a dotted
#: name is a field of a field (`stage.blocks`)
DERIVATION = "derivation.json"
#: the runs `plan` names in a point's `why`, the newest first
WHYS = 3
#: what a record's labels are on, by the start of the target's id: runs and arts (`served_table.records`)
RECORD_TARGETS = ("r", "art:")
#: the labels read from the remote at once: each is one small object, and the time is the remote's answers
PULL_JOBS = 64

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


def _named(art: str) -> str:
    return f"{art[:16]}…"


def _row_key(row: Mapping[str, Any]) -> str:
    """What names a row among a run's rows: its `type` and its `provers`."""

    return json.dumps([row.get("type"), row.get("provers")], sort_keys=True, default=repr)


@dataclass(frozen=True)
class Completed:
    """A labelled art's rows of one run, with the fields its derivation names for each row (by `_row_key`)."""

    art: str
    rows: list[Mapping[str, Any]]
    derived: Mapping[str, frozenset[str]]


def _text(store: Any, art: str, path: str) -> str:
    return (Path(store.fetch(art, paths=[path])) / path).read_text(errors="replace")


def _derived(text: str) -> dict[str, frozenset[str]] | str:
    """The fields a `DERIVATION` names for each row, or what a file that isn't one is instead."""

    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        return f"isn't JSON ({e.msg})"
    rows = doc.get("rows") if isinstance(doc, Mapping) else None
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) and isinstance(row.get("fields"), Mapping)
                                             for row in rows):
        return "isn't a record of the fields it derived for each row"
    return {_row_key(row): frozenset(name.strip() for key in row["fields"] for name in str(key).split(",") if name.strip())
            for row in rows}


def completed_rows(store: Any, art: str) -> tuple[list[str], Completed | str | None]:
    """A labelled art as one run's rows, completed after the run: the runs its meta names, and its rows where they
    stand to be taken as that run's (`completed` then holds them to the run's own). They stand when the art names the
    one run in `runs`, carries the rows' derivation (`DERIVATION`) and isn't superseded. Otherwise, in the rows' place:
    why they aren't taken, or None for an art that holds no rows (it stands for the runs it names)."""

    manifest = store.get_manifest(art)
    named = [run for run, _ in served_table.record_runs(store, [art])] if manifest.meta.get("runs") else []
    path = prove_records.ART_RATES
    if manifest.payload is None or not store.listing(art, [path]):
        return named, None
    if _superseded(store, art):
        return named, f"{_named(art)} is superseded"
    if len(named) != 1:
        return named, f"{_named(art)} holds `{path}` and names {len(named)} runs: an art's rows are one run's"
    if not store.listing(art, [DERIVATION]):
        return named, f"{_named(art)} holds its rows without their derivation from the run's records (`{DERIVATION}`)"
    derived = _derived(_text(store, art, DERIVATION))
    if isinstance(derived, str):
        return named, f"`{DERIVATION}` of {_named(art)} {derived}"
    try:
        rows = json.loads(_text(store, art, path))
    except json.JSONDecodeError as e:
        return named, f"`{path}` of {_named(art)} isn't JSON ({e.msg})"
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        return named, f"`{path}` of {_named(art)} isn't a list of rows"
    return named, Completed(art, rows, derived)


def _differences(own: Mapping[str, Any], theirs: Mapping[str, Any], within: str = "") -> list[tuple[str, str]]:
    """Each place a row differs from the run's own, by its dotted name: a field the own row lacks (`added`), one it
    holds and the other lacks (`dropped`), and one whose value is another (`changed`)."""

    found: list[tuple[str, str]] = []
    for key in sorted(own.keys() | theirs.keys(), key=str):
        name = f"{within}{key}"
        if key not in own:
            found.append((name, "added"))
        elif key not in theirs:
            found.append((name, "dropped"))
        elif isinstance(own[key], Mapping) and isinstance(theirs[key], Mapping):
            found += _differences(own[key], theirs[key], f"{name}.")
        elif own[key] != theirs[key] or type(own[key]) is not type(theirs[key]):
            found.append((name, "changed"))
    return found


def completed(own: list[Mapping[str, Any]], art: Completed) -> list[Mapping[str, Any]] | str:
    """A labelled art's rows as the run's `own` rows completed, or why they aren't that: each row of the art is one row
    of the run, by its `type` and `provers`, holds every field of it, and differs from it only by fields the run's row
    lacks and by fields the art's derivation names for that row (a name covers the fields under it). An art that
    changes anything else, a count of accepted sessions or an exit code, isn't the run's record of the row."""

    mine = {_row_key(row): row for row in own}
    if sorted(mine) != sorted(map(_row_key, art.rows)) or len(mine) != len(own):
        return (f"{_named(art.art)} holds {len(art.rows)} rows that aren't the run's {len(own)}, one for one by `type` "
                "and `provers`")
    for row in art.rows:
        names = art.derived.get(_row_key(row), frozenset())
        other = [f"`{name}` ({how})" for name, how in _differences(mine[_row_key(row)], row)
                 if how == "dropped" or (how == "changed" and not any(name == n or name.startswith(f"{n}.") for n in names))]
        if other:
            return (f"{_named(art.art)} differs from the run's own row of `{row.get('type')}` at {row.get('provers')} "
                    f"provers in {', '.join(other[:4])}{f' and {len(other) - 4} more' if len(other) > 4 else ''}: a "
                    f"completed row keeps every field of the run's and changes only those its `{DERIVATION}` names")
    return art.rows


@dataclass(frozen=True)
class Labelled:
    """A row of a run of a workload's steady-state record, with the run's files. `held`: for a row that isn't from the
    run's own rate file, where it is from; for one that is while an art holds rows of the run, why those aren't taken.
    Or (`row` None) a run of the record whose rows can't be read, `held` saying why."""

    run: str
    row: Mapping[str, Any] | None
    files: RunFiles | None
    held: str | None = None


def labelled_rows(store: Any, record: str) -> list[Labelled]:
    """Each row of the record's runs: the runs labelled into it and the runs its labelled arts name. A run's rows are
    its own rate file's, or, where one standing labelled art holds them completed (`completed_rows`, `completed`), the
    art's; two standing arts that complete one run complete it for neither. A run that kept no files, or whose rate
    file gives no rows, is one entry without a row, which says so. A superseded run and the runs of a superseded art
    give none."""

    runs: dict[str, None] = {}
    arts: dict[str, list[Completed]] = {}
    untaken: dict[str, str] = {}
    for target in served_table.records(store).get(record, []):
        if not target.startswith("art:"):
            runs[target] = None
            continue
        named, rows = completed_rows(store, target)
        if isinstance(rows, Completed):
            arts.setdefault(named[0], []).append(rows)
        elif rows is not None:
            untaken |= dict.fromkeys(named, rows)
        if not _superseded(store, target) and (rows is None or len(named) == 1):
            runs |= dict.fromkeys(named)
    found: list[Labelled] = []
    for run in runs:
        if _superseded(store, run):
            continue
        files = run_files(store, run)
        if files is None:
            found.append(Labelled(run, None, None, "the run kept no files"))
            continue
        own = steady_spec.rows(files)
        if isinstance(own, str):
            found.append(Labelled(run, None, files, own.removeprefix(f"{steady.CANNOT_TELL}: ")))
            continue
        standing = arts.get(run, [])
        taken = (completed(own, standing[0]) if len(standing) == 1 else untaken.get(run) if not standing else
                 f"{len(standing)} standing arts hold its rows completed ({', '.join(_named(c.art) for c in standing)}): "
                 "supersede all but one")
        if isinstance(taken, list):
            found += [Labelled(run, row, files, f"its row as completed in {_named(standing[0].art)}") for row in taken]
        else:
            found += [Labelled(run, row, files, f"its own row: {taken}" if taken else None) for row in own]
    return found


def pull_record_labels(store: Any, log: Callable[[str], None]) -> int:
    """Fetch the labels of the remote's runs and arts that this store lacks; how many. A record's targets are runs and
    arts (`RECORD_TARGETS`), and theirs are a fraction of the remote's labels, most of which are of other things: a
    machine whose store has never held them reads these in minutes, `PULL_JOBS` at once, where every label is several
    times as many. It says what it is doing before it starts: two listings of the remote, then one read for each label
    this store lacks."""

    from research.store import labels_sync

    log("pulling the labels of the store's runs and arts: two listings of the remote, then each label this store lacks")
    report = labels_sync.pull_prefixes(store, list(RECORD_TARGETS), jobs=PULL_JOBS)
    log(f"pulled {len(report['pulled'])} labels; the remote holds labels of {len(report['targets'])} runs and arts")
    for line in report["bad"][:5]:
        log(f"skipped label {line}")
    return len(report["pulled"])


class SteadyCompletion:
    """The newest passing run of a steady-state point, from this machine's catalog and, unless `local_only`, the
    store's remote: its runs' and arts' labels (`pull_record_labels`), and its attempts started since `since()` (when
    the spec was added: a point's identity names no commit, so every run since then can be the point's), pulled once."""

    def __init__(self, store: Any = None, *, root: Path | None = None, local_only: bool = False,
                 since: Callable[[], int] | None = None, log: Callable[[str], None] | None = None):
        self._log = log or (lambda message: print(f"verity-bench: {message}", file=sys.stderr, flush=True))
        self._base = jobs.StoreCompletion(store, root=root, local_only=local_only, log=self._log)
        self.root, self.local_only, self._since = root, local_only, since
        self._pulled = False
        self._labelled: dict[str, list[Labelled]] = {}

    @property
    def store(self) -> Any:
        return self._base.store

    def _pull(self) -> None:
        if self._pulled or self.local_only:
            return
        self._pulled = True
        since = self._since() if self._since is not None else None
        self._base._remote()
        pull_record_labels(self.store, self._log)
        if since is not None:
            self._base.pull_since(since)

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
        for found in sorted(self._labelled[record], key=lambda found: found.run, reverse=True):
            if found.row is None:
                whys.append(f"{found.run} (labelled `{record}`): {steady.CANNOT_TELL}: {found.held}")
                continue
            if found.row.get("pin") != item.row.pin or found.row.get("provers") != item.provers:
                continue
            why = steady_spec.point_why(found.row, found.files, spec, item)
            if why is None:
                return found.run, None
            whys.append(f"{found.run} (labelled `{record}`{f', {found.held}' if found.held else ''}): {why}")
        more = f"; and {len(whys) - WHYS} more" if len(whys) > WHYS else ""
        return None, ("; ".join(whys[:WHYS]) + more) or None


def _added(root: Path, source: str | Path) -> int:
    """When the spec's file was added to the tree: no run of its points is looked up from before."""

    path = jobs._relative_spec(source, root)
    if served_table.spec_path(served_table.spec_name(path)) != path:
        raise ValueError(f"{path} isn't a spec of {served_table.SPECS}, so its points' runs can't be looked up on the store's "
                         "remote since it was added: move it there, or pass --local-only")
    return served_table.spec_added(root, served_table.spec_name(path))


def make_plan(source: str | Path, *, root: Path | None = None, lookup: Lookup | None = None,
              local_only: bool = False) -> jobs.Plan:
    """The spec's points, each with its identity in this tree, its passing run and, without one, why. `lookup`: in
    place of the store's (`SteadyCompletion`, of this machine's catalog alone when `local_only`)."""

    root = root or jobs.repository_root()
    spec = steady_spec.load_spec(source, root=root)
    if spec.workload not in prove_records.WORKLOADS:
        raise ValueError(f"the `workload` of spec {spec.id!r}, {spec.workload!r}, has no steady-state record: the "
                         f"workloads are {', '.join(prove_records.WORKLOADS)} (`prove_records.WORKLOADS`)")
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
    files before the run's verdict is computed. A point whose row names its statement in the store runs with a
    read-only key to it."""

    spec = plan.spec
    return jobs.slot_run(
        spec, root, node, campaign="steady-point", tool="verity_bench.tool:STEADY_POINT",
        store_read=item.point.row.stored is not None,
        env=[flag for name, setting in jobs.point_environment(spec).items() for flag in ("--env", f"{name}={setting}")],
        job=["bash", "benchmarks/harness/steady_point.sh", "--spec", jobs._relative_spec(source, root),
             "--point", item.point.id, "--identity", item.identity])


def dispatch(plan: jobs.Plan, source: str | Path, *, root: Path | None = None, execute: jobs.Executor = jobs._execute,
             direct: str | None = None) -> int:
    """Launch one recorded run for each point without a passing run, on `direct`'s GPU slots: the harness's one
    dispatch (`jobs.dispatch`), with a steady-state point's command. A steady-state spec has no queued form."""

    if direct is None:
        raise ValueError(f"spec {plan.spec.id!r} is a steady-state spec: its points run on a node's own card, through a "
                         "GPU slot of it: `verity-bench run SPEC --direct NODE`")
    return jobs.dispatch(plan, source, root=root, execute=execute, direct=direct, command=direct_command)
