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
from verity_bench.vllm.presets import expand_sweep, resolve_spec
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
PASSING = """
SELECT a.id, o.artifact, a.source_commit
FROM attempts a
JOIN attempt_params s ON s.attempt = a.id AND s.key = 'spec' AND s.value = ?
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
    `summary.json` and `stop.json` under `out` in the run's record (`hb_sbench.py`/`sbench.sh`'s layout). Each is an
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


def spec_path(spec: str) -> str:
    """How a dispatched point records its spec (`jobs._relative_spec`)."""

    return f"benchmarks/specs/{spec}.toml"


def spec_name(path: str) -> str:
    return path.removeprefix("benchmarks/specs/").removesuffix(".toml")


def spec_added(root: Path, spec: str) -> int:
    """When the commit that added the spec was committed: none of its runs can start before."""

    process = subprocess.run(["git", "log", "--diff-filter=A", "--format=%ct", "--", spec_path(spec)],
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

    for run, art, commit in store.index.sql(PASSING, (spec_path(spec), point)):
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


def _ratio(summary: Mapping[str, Any] | None) -> str:
    if summary is None or summary.get("mean") is None:
        return "—"
    ci = summary.get("ci95")
    return f"{summary['mean']:.3f}×" if ci is None else f"{summary['mean']:.3f}× ± {(ci[1] - ci[0]) / 2:.3f}"


def _cell(text: Any) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _caveat(caveats: Sequence[Mapping[str, Any]], keys: Sequence[str] = CAVEAT_KEYS) -> str:
    newest = {label["key"]: label["value"] for label in caveats}
    return _cell(" / ".join(str(newest[key]) for key in keys if key in newest))


def _art(art: str | None) -> str:
    return f"`{art[:16]}`" if art else "—"


def _sources(row: Mapping[str, Any]) -> list[str]:
    if row["run"] is None:
        return ["—", "no passing run", "—", ""]
    record = f" (record {_art(row['record'])})" if row.get("record") else ""
    return [str(row["reps"]), row["run"], _art(row["art"]) + record, _caveat(row["caveats"])]


def ms_cell(summary: Mapping[str, Any]) -> str:
    ci = summary.get("ci95")
    return f"{summary['mean']:.3f}" if ci is None else f"{summary['mean']:.3f} ± {(ci[1] - ci[0]) / 2:.3f}"


def _breakdown(points: Sequence[Mapping[str, Any]]) -> list[str]:
    from verity_bench.breakdown import AUDIT, AUDIT_SPANS, DECODE_ROWS, PLAIN, PLAIN_SPANS

    lines = [
        "",
        "## Where the bandwidth audit's time goes, by span (`verity-bench breakdown`)",
        "",
        "Each cell is ms an output token, a difference and not a ratio: the mean over the point's paired reps ± its "
        f"95% half-width (Student t). `{AUDIT_SPANS}` − `{PLAIN_SPANS}` is split into rows that sum to it: the "
        "top-up's contention from the arms with the top-up off, every other row a part of the spans arms with the "
        f"top-up off. `{AUDIT}` − `{PLAIN}` (spans off) is the audit's cost; the spans' marks add their own time to "
        "the spans arms. Each spans arm's step by part: `verity-bench breakdown`.",
    ]
    for protocol in dict.fromkeys(point["protocol"] for point in points):
        group = [point for point in points if point["protocol"] == protocol]
        rows = [{row["row"]: row for row in point["rows"]} for point in group]
        lines += ["", f"{protocol}:", "",
                  "| row | " + " | ".join(f"{point['point']}, ms an output token | share" for point in group) + " |",
                  "|---|" + "---:|---:|" * len(group),
                  f"| `{AUDIT}` − `{PLAIN}` (credited lines, spans off) | "
                  + " | ".join(f"{ms_cell(point['credited'])} | " for point in group) + " |",
                  f"| **`{AUDIT_SPANS}` − `{PLAIN_SPANS}`** | "
                  + " | ".join(f"**{ms_cell(point['total'])}** | 100%" for point in group) + " |"]
        for name in rows[0]:
            part = name.removeprefix("decode ")
            label = f"{name}: {DECODE_ROWS[part]}" if part in DECODE_ROWS else name
            cells = []
            for by_name in rows:
                row = by_name.get(name)
                share = "—" if row is None or row["share"] is None else f"{row['share']:.0%}"
                cells.append(("—" if row is None else ms_cell(row["ms_per_token"])) + f" | {share}")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("| sum of the rows | " + " | ".join(f"{point['sum']:.3f} | " for point in group) + " |")
        lines += [""] + [f"- {point['point']}: `{point['spec']}`, {point['reps']} reps, run {point['run']}, art "
                         f"{_art(point['art'])}" + (f" (record {_art(point['record'])})" if point["record"] else "")
                         + (f"; {caveat}" if (caveat := _caveat(point["caveats"])) else "") for point in group]
    return lines


def render(table: Mapping[str, Any]) -> str:
    rows, factors = table["rows"], table["factors"]
    lines = [
        "# Served overhead on Llama-3.1-8B",
        "",
        "## PoUW: run of record (live rows)",
        "",
        "Each row is one in-process vLLM run on one GPU: Llama-3.1-8B int8 (the verity-int8 pin), every engine on "
        "vLLM's defaults (torch.compile, CUDA graphs), prefix caching off, decode at batch B with max_num_batched_tokens "
        "T. A decode step is (t(N2) − t(N1)) / (N2 − N1) "
        "over two greedy generate calls of N1 = 32 and N2 = 160 output tokens per request, so prefill and per-call "
        "overhead cancel; it returns one token for each of the B requests. The ratio is the PoUW arm's median decode "
        "step over its control's (the int8-exact engine without PoUW, same run, same GPU), per step, not per token: "
        "it is not comparable with the per-token ratios over stock FP8 below. A row stands only when its run's gates "
        "all passed: the device gate on poisoned (0xA5) buffers, the arm's tokens equal to the int8-exact engine's at "
        "the gate and at every timed rep, and a passing live window at the run's end on off-node coins.",
        "",
        "| device | machine | B | T | PoUW decode step, ms | control | control's step, ms | PoUW / control, per decode "
        "step | reps | coins | run | run record | caveat |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---|---|",
    ]
    for row in table["live"]:
        lines.append("| " + " | ".join([
            row["device"], str(row["machine"]), str(row["batch"]), str(row["t"]), f"{row['step_ms']:.3f}",
            f"`{row['control']}`", f"{row['control_ms']:.3f}", f"{row['ratio']:.3f}×", str(row["reps"]), row["coins"],
            row["run"], _art(row["record"]), _caveat(row["caveats"], LIVE_CAVEAT_KEYS)]) + " |")
    lines += ["", "PoUW arms and their gates' form:", ""]
    lines += [f"- {row['device']} (`{row['run']}`): `{row['arm']}`; {_cell(row['form'])}" for row in table["live"]]
    lines += [
        "",
        "## Served rows, one GPU each (`vllm_bench_point`)",
        "",
        "The device is the GPU the run's engines saw (the result's `identity.gpu`); a ratio's arm and stock ran on it in "
        "the same run. "
        f"Ratios are a protocol's arm over `{table['served_stock']}` (stock FP8 as vLLM serves it), paired by rep: mean "
        "± 95% half-width (Student t), or the mean alone at one rep; lower is better. A row is its protocol's run of "
        "record (its runs, or an art's, labelled `run_of_record`) or, without one, the newest passing run of its spec and point in "
        "the evidence store; a result labelled `superseded_by` is skipped. The caveat is the newest `label` and `note` "
        "on the result and its run.",
        "",
        "| protocol | device | spec | point | / served stock, per output token | / served stock, per decode token "
        "| / eager stock, per output token | reps | run | art | caveat |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        eager = _ratio(row["over_eager"][PER_OUTPUT]) if row["over_eager"] else "—"
        served = row["over_served"] or {}
        lines.append("| " + " | ".join([row["protocol"], row["device"] or "—", f"`{row['spec']}`", row["point"],
                                        _ratio(served.get(PER_OUTPUT)), _ratio(served.get(PER_DECODE)), eager,
                                        *_sources(row)]) + " |")
    lines += ["", "Arms:", ""]
    seen = set()
    for row in rows:
        if row["protocol"] in seen:
            continue
        seen.add(row["protocol"])
        own = (f"; over eager stock, `{row['own_arm']}` / `{row['eager_stock']}`" if row["eager_stock"]
               else "; no eager stock in its runs")
        lines.append(f"- {row['protocol']}: `{row['arm']}` / `{row['served_stock']}`{own}.")
    lines += [
        "",
        f"## Conversion factors: `{table['served_stock']}` / batch-invariant eager `stock-fp8`",
        "",
        "| spec | point | per output token | per decode token | reps | run | art | caveat |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in factors:
        factor = row["factor"] or {}
        lines.append("| " + " | ".join([f"`{row['spec']}`", row["point"], _ratio(factor.get(PER_OUTPUT)),
                                        _ratio(factor.get(PER_DECODE)), *_sources(row)]) + " |")
    if table.get("breakdowns"):
        lines += _breakdown(table["breakdowns"])
    if table["skipped"]:
        lines += ["", "Passed over (a run that can't stand):", ""]
        lines += [f"- `{s['spec']}` {s['point'] or ''}: {s['run']} ({_art(s['art'])}): {_cell(s['why'])}"
                  for s in table["skipped"]]
    return "\n".join(lines) + "\n"
