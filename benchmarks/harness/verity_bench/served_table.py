"""One table of every protocol's served overhead on Llama-3.1-8B, rendered from the evidence store.

A record is named in the store, never in code: a run labelled `run_of_record=NAME` is one of record NAME's runs, and an
art so labelled names its runs in its `runs` meta, in the order they were labelled (a target's latest label wins).

PoUW's run of record is its live rows (record `LIVE`): each a timed in-process vLLM run whose arm and control the run's
own `summary.json` and `stop.json` report, standing only when every gate it records passed and its coins were off-node.

Each served row (`PROTOCOLS`) is either a protocol's run of record (`Protocol.record`) or, without one, the newest passing
`vllm_bench_point` run of a spec and sweep point in the store's catalog (after pulling the remote's attempts started since
the spec was added), so a newer passing run supersedes an older one. A result labelled `superseded_by` is skipped. Ratios
are paired by rep from the result's own reps; caveats are the `label` and `note` assertions on the result and its run.

Beside a protocol's run of record, `BREAKDOWNS` puts where its time goes by span: `verity_bench.breakdown`'s rows (ms an
output token, not ratios) at each run of its record.
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_bench import jobs
from verity_bench.vllm import result as result_format
from verity_bench.vllm.presets import FORMER_SPEC, FORMER_SPECS, SOURCE_ROOT, SPEC_ROOT, expand_sweep, resolve_spec
from verity_bench.vllm.runner import summarize

SCHEMA = "verity-bench/served-table/v1"
SERVED = "stock-fp8@compiled"
PER_OUTPUT, PER_DECODE = "time_per_output_token_s", "decode_time_per_token_s"
CAVEAT_KEYS = ("label", "note")
LIVE_CAVEAT_KEYS = ("label", "row", "note")
SUPERSEDED_BY = "superseded_by"
RUN_OF_RECORD = "run_of_record"
POINT_TOOL = "vllm_bench_point"

# Validation comes from result.json alone; rc 0 rules out a run killed after writing it, and a dirty tree ran code its
# commit doesn't name (jobs.StoreCompletion's rule, keyed here by spec and point rather than by one commit's identity).
# The spec is named by either of its recorded paths (`spec_paths`).
PASSING = """
SELECT a.id, o.artifact, a.source_commit
FROM attempts a
JOIN attempt_params s ON s.attempt = a.id AND s.key = 'spec' AND s.value IN (?, ?)
JOIN attempt_params p ON p.attempt = a.id AND p.key = 'point' AND p.value = ?
JOIN attempt_outputs o ON o.attempt = a.id AND o.name = 'result'
WHERE a.tool = 'vllm_bench_point' AND a.validation = 'passed' AND a.rc = 0 AND a.source_dirty IS NOT 1
ORDER BY a.start_utc DESC, a.id DESC
""".strip()


@dataclass(frozen=True)
class Protocol:
    """A protocol's served rows: its `arm` over `stock-fp8@compiled`, and `own_arm` (default `arm`) over `eager_stock`,
    the eager stock engine its runs time beside it (None where their stock isn't eager). The rows are every point of
    each run of record `record`, or without one, the newest passing run of each point of each of `specs`."""

    name: str
    specs: tuple[str, ...]
    arm: str
    eager_stock: str | None
    own_arm: str | None = None
    record: str = ""


@dataclass(frozen=True)
class Live:
    """A live PoUW row: `arm`'s decode step over `control`'s, timed in run `run` on `device`, read from the
    `summary.json` and `stop.json` under `out` in the run's record (`sbench.sh`'s layout). Each is an
    entry of the `runs` meta of an art of record `LIVE`."""

    device: str
    run: str
    arm: str
    control: str
    out: str = "out"


PROTOCOLS = (
    Protocol("Pearl-C served (cross-check)",
             ("llama31-8b-1gpu-pouw-protocol-served", "llama31-8b-1gpu-pouw-protocol-served-b32"),
             "pouw:pearl-c-sm120-v1-h3", "stock-fp8", record="pearl-c-served"),
    Protocol("Pearl-C whole-step (cross-check)", (), "pouw-pearl-c-whole", "stock-fp8", record="pearl-c-whole-step"),
    # The bandwidth specs run every arm but the compiled one on decode-only CUDA graphs: no eager stock.
    Protocol("PoUS bandwidth audit", (), "bandwidth-audit", None, record="pous-bandwidth-audit"),
    Protocol("Commit", (), "commit@compiled", None, record="commit"),
    Protocol("H1 served, GPU commit path (cost only, in-process coins)", (), "pouw:eps1-sm120-h1fx1", "stock-int8",
             record="h1-served"),
)
# The factor that restates a ratio over batch-invariant eager stock as one over stock as served.
FACTORS = Protocol("conversion factor", (), SERVED, "stock-fp8", record="conversion-factor")
LIVE = "pouw-live"


@dataclass(frozen=True)
class Breakdown:
    """Where `protocol`'s time over its plain engine goes, by span (`verity_bench.breakdown`), at each run of record
    `record` (the breakdown spec's runs)."""

    protocol: str
    record: str


BREAKDOWNS = (Breakdown("PoUS bandwidth audit", "pous-bandwidth-audit-breakdown"),)


#: The specs' directory from the source tree's root, which a dispatched point records its spec under
SPECS = SPEC_ROOT.relative_to(SOURCE_ROOT).as_posix()


def spec_path(spec: str) -> str:
    """How a dispatched point records its spec (`jobs._relative_spec`)."""

    return f"{SPECS}/{spec}.toml"


def spec_paths(spec: str) -> tuple[str, str]:
    """The two paths a run of `spec` may have recorded: where the spec is, and where it was while the specs had a
    directory of their own (`presets.FORMER_SPEC`). A run of record keeps the path it ran under, so both are read."""

    return spec_path(spec), FORMER_SPEC.format(name=spec)


def spec_name(path: str) -> str:
    """The spec a recorded path names, under either of its directories (`spec_paths`); any other path stays a path."""

    for under in (SPECS, FORMER_SPECS):
        if path.startswith(f"{under}/"):
            return path.removeprefix(f"{under}/").removesuffix(".toml")
    return path.removesuffix(".toml")


def spec_added(root: Path, spec: str) -> int:
    """When the commit that added the spec was committed, under the earlier of its paths: none of its runs can start
    before."""

    process = subprocess.run(["git", "log", "--diff-filter=A", "--format=%ct", "--", *dict.fromkeys(spec_paths(spec))],
                             cwd=root, capture_output=True, text=True)
    times = [int(line) for line in process.stdout.split()]
    if process.returncode != 0 or not times:
        raise ValueError(f"git has no commit adding {spec_path(spec)} here, so its runs can't be looked up on the store's "
                         "remote: fetch the history (unshallow), or pass --local-only to read this machine's catalog alone")
    return min(times)


@dataclass(frozen=True)
class Found:
    run: str
    art: str
    commit: str
    reps: int
    point: Mapping[str, Any]
    caveats: list[dict[str, Any]]
    device: str | None = None


def _labels(store: Any, targets: Sequence[str]) -> list[dict[str, Any]]:
    found = [label for target in targets for label in store.labels(target)]
    return [{"target": label.target, "key": label.key, "value": label.value, "by": label.by, "ts": label.ts,
             "ref": label.ref} for label in sorted(found, key=lambda label: (label.ts, label.target, label.key))]


def _benchmark(store: Any, art: str) -> dict[str, Any]:
    tree = store.fetch(store.get_manifest(art).refs["run_files"], paths=["out/benchmark.json"])
    return json.loads((Path(tree) / "out" / "benchmark.json").read_text())


def _judge(store: Any, run: str, art: str, commit: str, arms: Sequence[str], points: Sequence[str] | None,
           record: str | None = None) -> tuple[list[Found], str | None]:
    """The result's points (those of `points`, else all) as rows, or why the result can't stand: a `superseded_by`
    label, an invalid result, or a point without every one of `arms` timed. The caveats are the result's, its run's and
    the `record` art's that named the run."""

    labels = _labels(store, [art, run, *([record] if record else [])])
    if superseded := [label["value"] for label in labels if label["key"] == SUPERSEDED_BY]:
        return [], f"superseded by {', '.join(map(str, superseded))}"
    doc = _benchmark(store, art)
    try:
        result_format.validate(doc)
    except ValueError as error:
        return [], f"invalid result: {error}"
    rows = [row for row in doc["points"] if points is None or row["point"]["id"] in points]
    if not rows:
        return [], f"no timed point {', '.join(points or ())}"
    for row in rows:
        if absent := [arm for arm in arms if arm not in row["arms"]]:
            return [], f"no timed {', '.join(absent)} at {row['point']['id']}"
    caveats = [label for label in labels if label["key"] in CAVEAT_KEYS]
    device = doc["identity"].get("gpu")
    return [Found(run, art, commit, int(doc["execution"]["reps"]), row, caveats, device) for row in rows], None


def newest_passing(store: Any, spec: str, point: str, arms: Sequence[str],
            skipped: list[dict[str, Any]]) -> Found | None:
    """The newest passing run of `spec` at `point` whose result isn't superseded and holds `arms`; each newer run passed
    over goes into `skipped` with why."""

    for run, art, commit in store.index.sql(PASSING, (*spec_paths(spec), point)):
        found, why = _judge(store, run, art, commit, arms, [point])
        if why is None:
            return found[0]
        skipped.append({"spec": spec, "point": point, "run": run, "art": art, "why": why})
    return None


def records(store: Any) -> dict[str, list[str]]:
    """Each record's targets in the catalog, the runs and arts whose latest `run_of_record` label names it, in the order
    each was first labelled."""

    latest = dict(store.index.sql("SELECT target, value FROM labels WHERE key = ? ORDER BY ts, path", (RUN_OF_RECORD,)))
    out: dict[str, list[str]] = {}
    for target, name in latest.items():
        out.setdefault(name, []).append(target)
    return out


def record_runs(store: Any, record: Sequence[str]) -> list[tuple[str, str | None]]:
    """Each run of a record's targets, with the art that named it (None for a labelled run): an art's `runs` meta is a
    list of run ids or a map from point to run id."""

    out: list[tuple[str, str | None]] = []
    for entry in record:
        if not entry.startswith("art:"):
            out.append((entry, None))
            continue
        runs = store.get_manifest(entry).meta.get("runs")
        listed = ([runs[point] for point in sorted(runs, key=_natural)] if isinstance(runs, Mapping)
                  else list(runs or ()))
        if not listed:
            raise ValueError(f"{entry}'s meta names no `runs`: label the record's runs instead")
        out += [(str(run), entry) for run in listed]
    return out


def _natural(point: str) -> list[Any]:
    """b8 before b32 before b128."""

    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", point)]


def _recorded(store: Any, run: str, arms: Sequence[str], record: str | None) -> tuple[str, list[Found], str | None]:
    """A run of record's spec and points as rows, or why it can't stand: the passing rule of `PASSING`, then `_judge`'s."""

    attempt = store.get_attempt(run)
    if attempt is None:
        return "", [], "not in the store or on its remote"
    spec = spec_name(str(attempt.get("params", {}).get("spec", "")))
    status, source = attempt.get("status") or {}, attempt.get("source") or {}
    art = (attempt.get("outputs") or {}).get("result")
    tool = (attempt.get("tool") or {}).get("name")
    if tool != POINT_TOOL or status.get("validation") != "passed" or status.get("rc") != 0 or source.get("dirty"):
        return spec, [], (f"not a passing {POINT_TOOL} run (tool {tool}, validation {status.get('validation')}, "
                          f"rc {status.get('rc')}, dirty {bool(source.get('dirty'))})")
    if not art:
        return spec, [], "no result output"
    found, why = _judge(store, run, art, str(source.get("commit") or ""), arms, None, record)
    return spec, found, why


def _arms(protocol: Protocol) -> list[str]:
    return list(dict.fromkeys([protocol.arm, *([protocol.eager_stock, protocol.own_arm or protocol.arm]
                                               if protocol.eager_stock else [])]))


def _points(protocol: Protocol, record: Sequence[str], store: Any,
            skipped: list[dict[str, Any]]) -> Iterator[tuple[str, str, Found | None, str | None]]:
    """(spec, point, found, the art that named the run) for every row of `protocol`, whose record's targets are
    `record`."""

    if record:
        for run, named_by in record_runs(store, record):
            spec, found, why = _recorded(store, run, _arms(protocol), named_by)
            if why is not None:
                skipped.append({"spec": spec, "point": None, "run": run, "art": named_by, "why": why})
                yield spec, "—", None, named_by
            for row in found:
                yield spec, row.point["point"]["id"], row, named_by
        return
    arms = list(dict.fromkeys([*_arms(protocol), SERVED]))
    for spec in protocol.specs:
        for point in expand_sweep(resolve_spec(spec).sweep.id):
            yield spec, point.id, newest_passing(store, spec, point.id, arms, skipped), None


def _source(spec: str, point: str, found: Found | None, record: str | None) -> dict[str, Any]:
    if found is None:
        return {"spec": spec, "point": point, "device": None, "run": None, "art": None, "commit": None, "reps": None,
                "caveats": [], "record": record}
    return {"spec": spec, "point": point, "device": found.device, "run": found.run, "art": found.art,
            "commit": found.commit, "reps": found.reps, "caveats": found.caveats, "record": record}


def _over(point: Mapping[str, Any], arm: str, over: str | None, metric: str) -> dict[str, Any] | None:
    return paired(point, arm, over, metric) if over and over in point["arms"] else None


def paired(point: Mapping[str, Any], arm: str, over: str, metric: str) -> dict[str, Any] | None:
    """`arm` over `over` at one point, paired by rep: mean, n and, from two reps, a 95% Student-t interval. None where a
    rep lacks the metric or a denominator is zero."""

    top = sorted(point["arms"][arm]["reps"], key=lambda rep: rep["rep"])
    bottom = sorted(point["arms"][over]["reps"], key=lambda rep: rep["rep"])
    pairs = [(a["metrics"].get(metric), b["metrics"].get(metric)) for a, b in zip(top, bottom, strict=True)]
    if not pairs or any(a is None or not b for a, b in pairs):
        return None
    return summarize([float(a) / float(b) for a, b in pairs])


def _live_failures(summary: Mapping[str, Any], stop: Mapping[str, Any], live: Live) -> list[str]:
    """Every gate the run records for `live.arm` that didn't pass; empty when the row stands."""

    arm, failures = live.arm, []
    if absent := [name for name in (arm, live.control) if name not in summary.get("arms", ())]:
        return [f"the run times no {', '.join(absent)}"]
    if failed := [name for name in summary.get("gate_failed", []) if name in (arm, live.control)]:
        failures.append(f"gate failed: {', '.join(failed)}")
    check = (summary.get("gate", {}).get(arm) or {}).get("check") or {}
    if not (check.get("ok") and check.get("complete")):
        failures.append("device gate not ok and complete")
    if not all(summary.get("gate_tokens_equal_ref_arm", {}).get(arm) or [False]):
        failures.append(f"gate tokens differ from {summary.get('ref_of', {}).get(arm)}'s")
    batch = summary.get("b")
    if not all(rep.get("requests_equal") == batch and not rep.get("first_diff")
               for rep in summary.get("decode_tokens_vs_ref", {}).get(arm) or [{}]):
        failures.append(f"timed tokens differ from {summary.get('ref_of', {}).get(arm)}'s")
    ended = summary.get("stop", {}).get(arm) or {}
    if ended.get("error") or ended.get("close_error"):
        failures.append(f"stopped with an error: {ended.get('error') or ended.get('close_error')}")
    verdict = ended.get("end_verdict") or {}
    if not (verdict.get("ok") and verdict.get("live_ok")):
        failures.append("no passing live window at the run's end (coins not live)")
    coins = (((stop.get(arm) or {}).get("end") or {}).get("live") or {}).get("coins") or {}
    if coins.get("source") != "off-node" or not coins.get("ok"):
        failures.append(f"coins {coins.get('source') or 'absent'}, not passing off-node coins")
    return failures


def live_row(store: Any, live: Live) -> tuple[dict[str, Any], str | None]:
    """One live row from its run's record, and why it can't stand (None when it stands)."""

    row: dict[str, Any] = {"device": live.device, "run": live.run, "arm": live.arm, "control": live.control,
                           "machine": None, "record": None, "commit": None, "batch": None, "t": None, "reps": None,
                           "step_ms": None, "control_ms": None, "ratio": None, "form": None, "coins": None,
                           "caveats": []}
    attempt = store.get_attempt(live.run)
    if attempt is None:
        return row, "not in the store or on its remote"
    status = attempt.get("status") or {}
    row.update(machine=(attempt.get("execution") or {}).get("machine"), commit=(attempt.get("source") or {}).get("commit"),
               record=(attempt.get("outputs") or {}).get("run_record"))
    if status.get("rc") != 0:
        return row, f"rc {status.get('rc')}"
    if not row["record"]:
        return row, "no run record"
    tree = Path(store.fetch(row["record"], paths=[f"{live.out}/summary.json", f"{live.out}/stop.json"]))
    summary = json.loads((tree / live.out / "summary.json").read_text())
    stop = json.loads((tree / live.out / "stop.json").read_text())
    row["caveats"] = [label for label in _labels(store, [live.run]) if label["key"] in LIVE_CAVEAT_KEYS]
    if failures := _live_failures(summary, stop, live):
        return row, "; ".join(failures)
    steps = summary["per_arm"][live.arm]["tpot_all"]
    controls = summary["per_arm"][live.control]["tpot_all"]
    check = summary["gate"][live.arm]["check"]
    coins = stop[live.arm]["end"]["live"]["coins"]
    row.update(batch=summary["b"], t=check.get("t"), reps=min(len(steps), len(controls)),
               step_ms=statistics.median(steps), control_ms=statistics.median(controls),
               form=check.get("form_basis") or check.get("form"), coins=coins["source"])
    row["ratio"] = row["step_ms"] / row["control_ms"]
    return row, None


def breakdown_points(store: Any, entry: Breakdown, record: Sequence[str],
                     skipped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`verity_bench.breakdown`'s attribution at every point of the runs of `entry`'s record, whose targets are `record`;
    a run that can't stand goes to `skipped`."""

    from verity_bench import breakdown

    points = []
    for run, named_by in record_runs(store, record):
        spec, found, why = _recorded(store, run, breakdown.ARMS, named_by)
        rows = []
        for row in found:
            try:
                rows.append({"protocol": entry.protocol, **_source(spec, row.point["point"]["id"], row, named_by),
                             **breakdown.attribution(row.point)})
            except ValueError as error:
                why = str(error)
        if why is not None:
            skipped.append({"spec": spec, "point": None, "run": run, "art": named_by, "why": why})
            continue
        points += rows
    return points


def live_runs(store: Any, record: Sequence[str]) -> list[Live]:
    """Each live row the arts of a record name, as their `runs` meta's entries."""

    if runs := [target for target in record if not target.startswith("art:")]:
        raise ValueError(f"{', '.join(runs)}: a live record is an art whose `runs` meta gives each run's device, arm and "
                         "control; label that art instead")
    return [Live(**entry) for art in record for entry in store.get_manifest(art).meta.get("runs") or ()]


def collect(store: Any, *, protocols: Sequence[Protocol] | None = None, factors: Protocol | None = None,
            live: str = LIVE, breakdowns: Sequence[Breakdown] | None = None) -> dict[str, Any]:
    """The table's rows from what `store` holds, its labels included."""

    protocols = PROTOCOLS if protocols is None else protocols
    factors = FACTORS if factors is None else factors
    breakdowns = BREAKDOWNS if breakdowns is None else breakdowns
    named = records(store)
    skipped: list[dict[str, Any]] = []
    live_rows = []
    for entry in live_runs(store, named.get(live, ())):
        row, why = live_row(store, entry)
        if why is not None:
            skipped.append({"spec": f"live {entry.device}", "point": None, "run": entry.run, "art": row["record"],
                            "why": why})
        else:
            live_rows.append(row)
    rows = []
    for protocol in protocols:
        for spec, point, found, record in _points(protocol, named.get(protocol.record, ()), store, skipped):
            own = protocol.own_arm or protocol.arm
            row = {"protocol": protocol.name, "arm": protocol.arm, "served_stock": SERVED,
                   "own_arm": own if protocol.eager_stock else None, "eager_stock": protocol.eager_stock,
                   **_source(spec, point, found, record), "over_served": None, "over_eager": None}
            if found is not None:
                row["over_served"] = {metric: _over(found.point, protocol.arm, SERVED, metric)
                                      for metric in (PER_OUTPUT, PER_DECODE)}
                if protocol.eager_stock:
                    row["over_eager"] = {PER_OUTPUT: _over(found.point, own, protocol.eager_stock, PER_OUTPUT)}
            rows.append(row)
    factor_rows = []
    for spec, point, found, record in _points(factors, named.get(factors.record, ()), store, skipped):
        row = {"arm": SERVED, "eager_stock": factors.eager_stock, **_source(spec, point, found, record), "factor": None}
        if found is not None:
            row["factor"] = {metric: _over(found.point, SERVED, factors.eager_stock, metric)
                             for metric in (PER_OUTPUT, PER_DECODE)}
        factor_rows.append(row)
    split = [point for entry in breakdowns
             for point in breakdown_points(store, entry, named.get(entry.record, ()), skipped)]
    return {"schema": SCHEMA, "served_stock": SERVED, "live": live_rows, "rows": rows, "factors": factor_rows,
            "breakdowns": split, "skipped": skipped}


def build(*, store: Any = None, root: Path | None = None, local_only: bool = False,
          added: Callable[[str], int] | None = None, log: Callable[[str], None] | None = None,
          protocols: Sequence[Protocol] | None = None, factors: Protocol | None = None,
          live: str = LIVE, breakdowns: Sequence[Breakdown] | None = None) -> dict[str, Any]:
    """`collect` after pulling the remote's labels, then its attempts started since the oldest spec without a record was
    added (`check`'s pull), unless `local_only`. A run of record is read by id, from the remote where this machine lacks
    it."""

    protocols = PROTOCOLS if protocols is None else protocols
    factors = FACTORS if factors is None else factors
    root = root or jobs.repository_root()
    completion = jobs.StoreCompletion(store, root=root, local_only=local_only, log=log)
    if not local_only:
        completion.pull_labels()
    named = records(completion.store)
    specs = [spec for protocol in (*protocols, factors) if protocol.record not in named for spec in protocol.specs]
    if not local_only and specs:
        when = added or (lambda spec: spec_added(root, spec))
        completion.pull_since(min(when(spec) for spec in specs))
    return collect(completion.store, protocols=protocols, factors=factors, live=live, breakdowns=breakdowns)
