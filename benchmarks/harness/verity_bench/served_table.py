"""One table of every protocol's served overhead on Llama-3.1-8B on one RTX PRO 6000, rendered from the evidence store.

Each row is the newest passing `vllm_bench_point` run of a spec and sweep point in the store's catalog (after pulling the
remote's attempts started since the spec was added), so a newer passing run supersedes an older one without anyone naming
art ids, and a result labelled `superseded_by` is skipped. Ratios are paired by rep from the result's own reps; caveats
are the `label` and `note` assertions on the result and its run.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_bench import jobs
from verity_bench.vllm import result as result_format
from verity_bench.vllm.presets import expand_sweep, resolve_spec
from verity_bench.vllm.runner import _summary

SCHEMA = "verity-bench/served-table/v0"
SERVED = "stock-fp8@compiled"
PER_OUTPUT, PER_DECODE = "time_per_output_token_s", "decode_time_per_token_s"
CAVEAT_KEYS = ("label", "note")
SUPERSEDED_BY = "superseded_by"

# Validation comes from result.json alone; rc 0 rules out a run killed after writing it, and a dirty tree ran code its
# commit doesn't name (jobs.PASS_QUERY's rule, keyed here by spec and point rather than by one commit's identity).
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
    """A protocol's served rows: its `arm` over `stock-fp8@compiled` in each of `specs`, and `own_arm` (default `arm`)
    over `eager_stock`, the spec's eager stock engine, where the spec runs one (None where its stock isn't eager)."""

    name: str
    specs: tuple[str, ...]
    arm: str
    eager_stock: str | None
    own_arm: str | None = None


PROTOCOLS = (
    Protocol("Pearl-C", ("llama31-8b-1gpu-pouw-protocol-served", "llama31-8b-1gpu-pouw-protocol-served-b32"),
             "pouw:pearl-c-sm120-v1-h3", "stock-fp8"),
    # The bandwidth specs run every arm but the compiled one on decode-only CUDA graphs: no eager stock.
    Protocol("PoUS bandwidth audit", ("llama31-8b-fp8-1gpu-bandwidth-served",), "bandwidth-audit", None),
    Protocol("Commit", ("llama31-8b-fp8-1gpu-commit-served",), "commit@compiled", "stock-fp8", own_arm="commit"),
    Protocol("H1 int8", ("llama31-8b-1gpu-pouw-h1-served",), "pouw:eps1-sm120-h1fx1", "stock-int8"),
)
# The factor that restates a ratio over batch-invariant eager stock as one over stock as served.
FACTORS = Protocol("conversion factor", ("llama31-8b-fp8-1gpu-compiled-full",), SERVED, "stock-fp8")


def spec_path(spec: str) -> str:
    """How a dispatched point records its spec (`jobs._relative_spec`)."""

    return f"benchmarks/specs/{spec}.toml"


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


def _labels(store: Any, targets: Sequence[str], pull: bool) -> list[dict[str, Any]]:
    if pull:
        from research.store import labels_sync  # noqa: PLC0415

        labels_sync.pull_targets(store, list(targets))
    found = [label for target in targets for label in store.labels(target)]
    return [{"target": label.target, "key": label.key, "value": label.value, "by": label.by, "ts": label.ts,
             "ref": label.ref} for label in sorted(found, key=lambda label: (label.ts, label.target, label.key))]


def _benchmark(store: Any, art: str) -> dict[str, Any]:
    tree = store.fetch(store.get_manifest(art).refs["run_files"], paths=["out/benchmark.json"])
    return json.loads((Path(tree) / "out" / "benchmark.json").read_text())


def _newest(store: Any, spec: str, point: str, arms: Sequence[str], pull: bool,
            skipped: list[dict[str, Any]]) -> Found | None:
    """The newest passing run of `spec` at `point` whose result isn't superseded and holds `arms`; each newer run passed
    over goes into `skipped` with why."""

    for run, art, commit in store.index.sql(PASSING, (spec_path(spec), point)):
        labels = _labels(store, [art, run], pull)
        why = None
        if superseded := [label["value"] for label in labels if label["key"] == SUPERSEDED_BY]:
            why = f"superseded by {', '.join(map(str, superseded))}"
        else:
            doc = _benchmark(store, art)
            try:
                result_format.validate(doc)
            except ValueError as error:
                why = f"invalid result: {error}"
            else:
                rows = [row for row in doc["points"] if row["point"]["id"] == point]
                absent = [arm for arm in arms if not rows or arm not in rows[0]["arms"]]
                if absent:
                    why = f"no timed {', '.join(absent)} at {point}"
        if why is not None:
            skipped.append({"spec": spec, "point": point, "run": run, "art": art, "why": why})
            continue
        return Found(run, art, commit, int(doc["execution"]["reps"]), rows[0],
                     [label for label in labels if label["key"] in CAVEAT_KEYS])
    return None


def paired(point: Mapping[str, Any], arm: str, over: str, metric: str) -> dict[str, Any] | None:
    """`arm` over `over` at one point, paired by rep: mean, n and, from two reps, a 95% Student-t interval. None where a
    rep lacks the metric or a denominator is zero."""

    top = sorted(point["arms"][arm]["reps"], key=lambda rep: rep["rep"])
    bottom = sorted(point["arms"][over]["reps"], key=lambda rep: rep["rep"])
    pairs = [(a["metrics"].get(metric), b["metrics"].get(metric)) for a, b in zip(top, bottom, strict=True)]
    if not pairs or any(a is None or not b for a, b in pairs):
        return None
    return _summary([float(a) / float(b) for a, b in pairs])


def _points(protocol: Protocol, pull: bool, store: Any,
            skipped: list[dict[str, Any]]) -> Iterator[tuple[str, str, Found | None]]:
    arms = [protocol.arm, SERVED, *([protocol.eager_stock, protocol.own_arm or protocol.arm] if protocol.eager_stock
                                    else [])]
    for spec in protocol.specs:
        for point in expand_sweep(resolve_spec(spec).sweep.id):
            yield spec, point.id, _newest(store, spec, point.id, list(dict.fromkeys(arms)), pull, skipped)


def _source(spec: str, point: str, found: Found | None) -> dict[str, Any]:
    if found is None:
        return {"spec": spec, "point": point, "run": None, "art": None, "commit": None, "reps": None, "caveats": []}
    return {"spec": spec, "point": point, "run": found.run, "art": found.art, "commit": found.commit,
            "reps": found.reps, "caveats": found.caveats}


def collect(store: Any, *, pull: bool, protocols: Sequence[Protocol] = PROTOCOLS,
            factors: Protocol = FACTORS) -> dict[str, Any]:
    """The table's rows from what `store`'s catalog holds; `pull` reads each result's labels through from the remote."""

    skipped: list[dict[str, Any]] = []
    rows = []
    for protocol in protocols:
        for spec, point, found in _points(protocol, pull, store, skipped):
            own = protocol.own_arm or protocol.arm
            row = {"protocol": protocol.name, "arm": protocol.arm, "served_stock": SERVED,
                   "own_arm": own if protocol.eager_stock else None, "eager_stock": protocol.eager_stock,
                   **_source(spec, point, found), "over_served": None, "over_eager": None}
            if found is not None:
                row["over_served"] = {metric: paired(found.point, protocol.arm, SERVED, metric)
                                      for metric in (PER_OUTPUT, PER_DECODE)}
                if protocol.eager_stock:
                    row["over_eager"] = {PER_OUTPUT: paired(found.point, own, protocol.eager_stock, PER_OUTPUT)}
            rows.append(row)
    factor_rows = []
    for spec, point, found in _points(factors, pull, store, skipped):
        row = {"arm": SERVED, "eager_stock": factors.eager_stock, **_source(spec, point, found), "factor": None}
        if found is not None:
            row["factor"] = {metric: paired(found.point, SERVED, factors.eager_stock, metric)
                             for metric in (PER_OUTPUT, PER_DECODE)}
        factor_rows.append(row)
    return {"schema": SCHEMA, "served_stock": SERVED, "rows": rows, "factors": factor_rows, "skipped": skipped}


def build(*, store: Any = None, root: Path | None = None, local_only: bool = False,
          added: Callable[[str], int] | None = None, log: Callable[[str], None] | None = None,
          protocols: Sequence[Protocol] = PROTOCOLS, factors: Protocol = FACTORS) -> dict[str, Any]:
    """`collect` after pulling the remote's attempts started since the oldest of the specs was added (#3406's pull),
    unless `local_only`."""

    root = root or jobs.repository_root()
    completion = jobs.StoreCompletion(store, root=root, local_only=local_only, log=log)
    if not local_only:
        when = added or (lambda spec: spec_added(root, spec))
        completion.pull_since(min(when(spec) for protocol in (*protocols, factors) for spec in protocol.specs))
    return collect(completion.store, pull=not local_only, protocols=protocols, factors=factors)


def _ratio(summary: Mapping[str, Any] | None) -> str:
    if summary is None or summary.get("mean") is None:
        return "—"
    ci = summary.get("ci95")
    return f"{summary['mean']:.3f}×" if ci is None else f"{summary['mean']:.3f}× ± {(ci[1] - ci[0]) / 2:.3f}"


def _cell(text: Any) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _caveat(caveats: Sequence[Mapping[str, Any]]) -> str:
    newest = {label["key"]: label["value"] for label in caveats}
    return _cell(" / ".join(str(newest[key]) for key in CAVEAT_KEYS if key in newest))


def _art(art: str | None) -> str:
    return f"`{art[:16]}`" if art else "—"


def _sources(row: Mapping[str, Any]) -> list[str]:
    if row["run"] is None:
        return ["—", "no passing run", "—", ""]
    return [str(row["reps"]), row["run"], _art(row["art"]), _caveat(row["caveats"])]


def render(table: Mapping[str, Any]) -> str:
    rows, factors = table["rows"], table["factors"]
    lines = [
        "# Served overhead on Llama-3.1-8B, one RTX PRO 6000",
        "",
        f"Ratios are a protocol's arm over `{table['served_stock']}` (stock FP8 as vLLM serves it), paired by rep: mean "
        "± 95% half-width (Student t), or the mean alone at one rep; lower is better. Each row is the newest passing run "
        "of its spec and point in the evidence store; a result labelled `superseded_by` is skipped. The caveat is the "
        "newest `label` and `note` on the result and its run.",
        "",
        "| protocol | spec | point | / served stock, per output token | / served stock, per decode token "
        "| / eager stock, per output token | reps | run | art | caveat |",
        "|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        eager = _ratio(row["over_eager"][PER_OUTPUT]) if row["over_eager"] else "—"
        served = row["over_served"] or {}
        lines.append("| " + " | ".join([row["protocol"], f"`{row['spec']}`", row["point"], _ratio(served.get(PER_OUTPUT)),
                                        _ratio(served.get(PER_DECODE)), eager, *_sources(row)]) + " |")
    lines += ["", "Arms:", ""]
    seen = set()
    for row in rows:
        if row["protocol"] in seen:
            continue
        seen.add(row["protocol"])
        own = (f"; over eager stock, `{row['own_arm']}` / `{row['eager_stock']}`" if row["eager_stock"]
               else "; no eager stock in its spec")
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
    if table["skipped"]:
        lines += ["", "Passed over (a newer passing run that can't stand):", ""]
        lines += [f"- `{s['spec']}` {s['point']}: {s['run']} ({_art(s['art'])}): {_cell(s['why'])}"
                  for s in table["skipped"]]
    return "\n".join(lines) + "\n"
