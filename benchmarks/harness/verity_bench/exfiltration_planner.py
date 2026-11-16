"""The compute audit's part of `verity bench exfiltration` (`verity_bench.exfiltration`): the planner's rows of record,
read from the store or from one run's record directory (`--from`), and each scope read at the share of compute."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from verity_bench.exfiltration_common import (
    BASIS_SOURCES,
    CASE_KEYS,
    COPY_BASIS,
    DEFAULT_BASIS,
    GB,
    PARTS,
    PLAN_FILE,
    PLAN_TOLERANCE,
    RECORD,
    ROW,
    ROWS_FILES,
    SOURCES,
    STEP,
    UNSTATED,
    VARIANT,
    WORLD,
    _same,
)
from verity_bench.served_table import RUN_OF_RECORD, SUPERSEDED_BY, record_runs, records


def _label(key: str = RUN_OF_RECORD, value: str = RECORD) -> str:
    return f"`research data label RUN {key} {value} --by LANE`"


def _run_files(store: Any, run: str, names: Sequence[str]) -> tuple[Path | None, list[str], str | None]:
    """Where the files ``names`` of a run's files that it holds are fetched to, and which it holds; or why it has none:
    the run is not in the store, or kept no run files."""

    attempt = store.get_attempt(run)
    if attempt is None:
        return None, [], "not in the store or on its remote"
    out = attempt.get("outputs") or {}
    files = out.get("run_files") or out.get("run_record")
    if not files:
        return None, [], "kept no run files"
    held = [name for name in names if store.listing(files, [name])]
    return (Path(store.fetch(files, paths=held)) if held else None), held, None


def read_run(store: Any, run: str) -> tuple[list[dict], dict | None, str | None]:
    """The planner's rows and the plan in a run's files, either of which it may lack, or why the run holds neither or
    is not read. The rows are its rows files' (`ROWS_FILES`), a case's over a scope the first file's that holds them; a
    later file that holds other rows of that case and scope leaves the run unread."""

    at, held, why = _run_files(store, run, (*ROWS_FILES, PLAN_FILE))
    return ([], None, why) if why else _rows_in(at, held)


#: a run's record file that names it and the commit it ran (`run.id`, `computation.tree_sha`), and, for a record
#: written by hand, what it says of itself (`made_up`), which a run's never holds
JOB = "job.json"


def record_job(record: Path) -> dict:
    """A record directory's `JOB`, or nothing where it holds none."""

    job = json.loads((record / JOB).read_text()) if (record / JOB).is_file() else {}
    if not isinstance(job, dict):
        raise ValueError(f"{record / JOB} is not a JSON object: a run's record names itself there")
    return job


def record_run(record: Path) -> str:
    """The run a record directory is of: its `JOB`'s `run.id`, else the directory's name."""

    return str((record_job(record).get("run") or {}).get("id") or record.resolve().name)


def read_record(record: Path) -> tuple[list[dict], dict | None, str | None]:
    """The planner's rows and the plan in a run's record directory, read as `read_run` reads a run's fetched files."""

    held = [name for name in (*ROWS_FILES, PLAN_FILE) if (record / name).is_file()]
    return _rows_in(record if held else None, held)


def _rows_in(at: Path | None, held: Sequence[str]) -> tuple[list[dict], dict | None, str | None]:
    if at is None:
        return [], None, f"holds no planner rows ({', '.join(f'`{n}`' for n in ROWS_FILES)}) and no plan (`{PLAN_FILE}`)"
    rows: list[dict] = []
    first: dict[tuple, tuple[str, list[str]]] = {}
    for name in (n for n in ROWS_FILES if n in held):
        cases: dict[tuple, list[dict]] = {}
        for line in (at / name).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                cases.setdefault(tuple(str(r.get(k)) for k in CASE_KEYS), []).append(r)
        for case, its in cases.items():
            text = sorted(json.dumps(r, sort_keys=True) for r in its)
            if case not in first:
                first[case] = (name, text)
                rows += its
            elif first[case][1] != text:
                return [], None, (f"holds row `{case[1]}` over the {case[3]} in `{first[case][0]}` and in `{name}`, and the "
                                  "two differ")
    return rows, json.loads((at / PLAN_FILE).read_text()) if PLAN_FILE in held else None, None


def _case(x: Mapping, row: str, step: str, world: str) -> bool:
    return (x.get("world"), x.get("row")) == (world, row) and str(x.get("step", "")).startswith(step)


def rows_of_record(store: Any, *, row: str = ROW, step: str = STEP, world: str = WORLD,
                   variant: str | None = None, record: Path | None = None) -> tuple[dict, dict, list[str], list[dict]]:
    """The case's rows of record by scope, each scope's from the one run that holds it, and its plans of record the
    same way, with the runs read and the runs left out with why. A run that holds a scope's plan may hold the plan's
    own row, at the plan's budget: it is the plan's, and is not read off. With ``variant``, only the rows the
    planner's driver names so (`VARIANT`) are read. With ``record`` (a run's record directory), its run is the one run
    of record, read from the directory with no store (`read_record`), and ``store`` is not read."""

    if record is None:
        labelled = record_runs(store, records(store).get(RECORD, []))
        if not labelled:
            raise ValueError(f"no planner rows of record: nothing in the store carries `{RUN_OF_RECORD}={RECORD}` "
                             f"(label each run that holds them: {_label()})")
        targets = sorted({run for run, _ in labelled}, reverse=True)
    else:
        targets = [record_run(record)]
    scopes: dict[str, tuple[str, list[dict]]] = {}
    plans: dict[str, tuple[str, dict]] = {}
    runs, refused = [], []
    for run in targets:
        found: list[dict] = []
        plan: dict | None = None
        superseded = [] if record is not None else [str(lab.value) for lab in store.labels(run, key=SUPERSEDED_BY)]
        if superseded:
            why: str | None = f"superseded by {', '.join(superseded)}"
        else:
            found, plan, why = read_run(store, run) if record is None else read_record(record)
            found = [r for r in found if not r.get("error") and _case(r, row, step, world)
                     and variant in (None, r.get(VARIANT))]
            if plan is not None and _case(plan, row, step, world):
                found = [r for r in found if (r["scope"], r["b10"]) != (plan["scope"], plan["beta"])]
            else:
                plan = None
            why = why or (None if found or plan else f"holds no row `{row}` at a step `{step}...` of the world `{world}`")
        if why:
            refused.append({"target": run, "record": RECORD, "why": why})
            continue
        runs.append(run)
        if plan:
            if plan["scope"] in plans:
                raise ValueError(f"{plans[plan['scope']][0]} and {run} both hold the {plan['scope']}'s plan and neither is "
                                 f"superseded: label the one the other replaces ({_label(SUPERSEDED_BY, 'NEWER')})")
            plans[plan["scope"]] = (run, plan)
        by: dict[str, list[dict]] = {}
        for r in found:
            by.setdefault(r["scope"], []).append(r)
        for scope, its in by.items():
            if scope in scopes:
                raise ValueError(f"{scopes[scope][0]} and {run} both hold the {scope}'s rows and neither is superseded: "
                                 f"label the one the other replaces ({_label(SUPERSEDED_BY, 'NEWER')})")
            if len(steps := sorted({r["step"] for r in its})) > 1:
                raise ValueError(f"`{step}` begins {len(steps)} steps of {run}'s row `{row}`: give more of the one meant "
                                 f"(--step): {'; '.join(steps)}")
            scopes[scope] = (run, its)
    if not scopes:
        left = "; ".join(f"{x['target']} {x['why']}" for x in refused) or f"{', '.join(runs)} hold a plan alone"
        fix = "give --from a run's record that does" if record is not None else f"label one that does: {_label()}"
        raise ValueError(f"no run of record holds the planner's rows: {left} ({fix})")
    for scope in sorted(set(plans) - set(scopes)):
        refused.append({"target": plans.pop(scope)[0], "record": RECORD,
                        "why": f"holds the {scope}'s plan, and no run of record holds that scope's rows"})
    read = {run for run, _ in (*scopes.values(), *plans.values())}
    return scopes, plans, [run for run in runs if run in read], refused


def _fit(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """The least-squares line through ``points``: its slope and intercept."""

    n = len(points)
    mx, my = sum(x for x, _ in points) / n, sum(y for _, y in points) / n
    slope = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x, _ in points)
    return slope, my - slope * mx


def _between(rows: Sequence[Mapping], key: str, beta: float, *, log: bool = False) -> float:
    """``key`` at the budget ``beta``, on the line through the two pinned budgets that bracket it (the nearest two
    where none do): linear in β, or with ``log`` ln(key) linear in ln β."""

    i = max(0, min(len(rows) - 2, sum(r["b10"] <= beta for r in rows) - 1))
    lo, hi = rows[i], rows[i + 1]
    if log:
        t = (math.log(beta) - math.log(lo["b10"])) / (math.log(hi["b10"]) - math.log(lo["b10"]))
        return math.exp(math.log(lo[key]) + (math.log(hi[key]) - math.log(lo[key])) * t)
    return lo[key] + (hi[key] - lo[key]) * (beta - lo["b10"]) / (hi["b10"] - lo["b10"])


def _requests(x: Mapping) -> float | None:
    """The requests a row or a plan is priced over: its sampled replays over the probability a request is sampled."""

    q, sample = x.get("q"), x.get("replay_sample")
    return sample / q if q and sample else None


def _priced(shares: Mapping[str, float], said: Mapping) -> dict[str, dict]:
    """A plan's proving time by how its block families are priced: each way's share of it and its families with
    theirs, largest first. ``shares`` are the plan's, by family; how a family is priced is the rows' account
    (``said``, their `split.families`): its share by way (`ways`), split in their proportion, else its ways
    (`priced`) joined, and `UNSTATED` for a family they don't name."""

    out: dict[str, dict] = {}
    for family, its in sorted(shares.items(), key=lambda kv: -kv[1]):
        f = said.get(family) or {}
        by = {k: v for k, v in (f.get("ways") or {}).items() if v > 0}
        total = sum(by.values())
        ways = ({k: v / total for k, v in sorted(by.items(), key=lambda kv: -kv[1])} if total > 0 else
                {" and ".join(sorted(f["priced"], reverse=True)) if f.get("priced") else UNSTATED: 1.0})
        for name, w in ways.items():
            way = out.setdefault(name, {"share": 0.0, "families": {}})
            way["share"] += its * w
            way["families"][family] = its * w
    return out


def _sources(split: Mapping) -> dict | None:
    """The share of a plan's proving each source prices, as its row's `split` records it: its `sources`, else its
    `basis` by `BASIS_SOURCES`, each of `SOURCES` at 0 where it prices none (`by`); and the share at the planner's
    assumed default (`default`, None where the split holds no `basis`). None where it records neither."""

    basis = split.get("basis") or {}
    by = dict(split.get("sources") or {})
    if not by:
        for key, share in basis.items():
            source = BASIS_SOURCES.get(key.split(":")[0]) or (key if key == COPY_BASIS else SOURCES[-1])
            by[source] = by.get(source, 0.0) + share
    if not by:
        return None
    return {"by": {s: by.get(s, 0.0) for s in SOURCES} | {s: v for s, v in by.items() if s not in SOURCES},
            "default": basis.get(DEFAULT_BASIS, 0.0) if basis else None}


def _own(run: str, row: Mapping) -> dict | None:
    """A row's own plan, where its `split` is one: it says the `q` and the sampled replays it is the plan of, and they
    are the row's. How its proving is priced (`_priced`) and what prices it (`_sources`), or None for a `split` that is
    not the row's plan."""

    split = row.get("split") or {}
    if not all(split.get(k) and row.get(k) and _same(split[k], row[k]) for k in ("q", "replay_sample")):
        return None
    families = split.get("families") or {}
    return {"of": "rows", "run": run, "beta": row["b10"],
            "ways": _priced({family: f["share"] for family, f in families.items()}, families), "sources": _sources(split)}


def _held(scope: str, run: str, rows: Sequence[Mapping], requests: float, of: str, plan: Mapping) -> dict:
    """``of``'s plan of a scope as these rows' plan: how its proving is priced (`_priced`), once it counts the rows'
    requests and bounds what they do at its budget. A plan records no source of its prices."""

    count, theirs = _requests(plan), _between(rows, "bound_bits", plan["beta"], log=True)
    if count is None or not _same(count, requests):
        raise ValueError(f"{of}'s plan of the {scope} is priced over {count or 0:.6g} requests and {run}'s rows over "
                         f"{requests:.6g}: it is not these rows' plan")
    if not math.isclose(plan["bound_bits"], theirs, rel_tol=PLAN_TOLERANCE):
        raise ValueError(f"{of}'s plan of the {scope} bounds {plan['bound_bits'] / GB:,.1f} GB at β = {plan['beta']:.4%}, "
                         f"where {run}'s rows give {theirs / GB:,.1f} GB: it is not these rows' plan")
    shares: dict[str, float] = {}
    for cell in (*plan["cells"], *plan.get("left_out", ())):
        family = cell["block"].split("{")[0]
        shares[family] = shares.get(family, 0.0) + cell["share"]
    near = min(rows, key=lambda r: abs(r["b10"] - plan["beta"]))
    return {"of": "plan", "run": of, "beta": plan["beta"],
            "ways": _priced(shares, (near.get("split") or {}).get("families") or {}), "sources": None}


def at_share(scope: str, run: str, rows: Sequence[Mapping], share: float,
             plan: tuple[str, Mapping] | None = None) -> dict:
    """One scope's rows read off at the audit budget where the total is ``share`` of the compute: the budget, whether
    the pinned budgets bracket it, the bound there at λ = 1 and λ times it, each part of the total, and what the
    rows hold fixed (λ, δ, the most bits a request's output holds, the scope's GPU-seconds, the requests they are
    priced over and the weights). How its proving is priced is ``plan``'s account, a run and the scope's plan it
    holds (`_held`); without one the rows' own plan's at the pinned budget nearest (`_own`), or None. The row at
    that budget's `note`, where it carries one, is the scope's."""

    rows = sorted(rows, key=lambda r: r["b10"])
    if len(kinds := sorted({str(r[VARIANT]) for r in rows if r.get(VARIANT) is not None})) > 1:
        raise ValueError(f"{run}'s {scope} rows are {len(kinds)} variants of the row ({', '.join(kinds)}), each on its "
                         "own premises: a run of record holds one, or --variant names the one meant")
    asked = [_requests(r) for r in rows]
    counts = [c for c in asked if c is not None]
    if len(counts) < len(asked):
        raise ValueError(f"{run}'s {scope} rows don't say what requests they are priced over (`replay_sample / q`)")
    fixed = {"lambda": {r["weights_bits"] / r["target_bits"] for r in rows}, "delta": {r["delta_seen"] for r in rows},
             "output_bits": {r["S"] for r in rows}, "gpu_seconds": {r["gpu_seconds"] for r in rows},
             "weights_bits": {r["weights_bits"] for r in rows}}
    if len({r["b10"] for r in rows}) < 2 or len({r["b10"] for r in rows}) != len(rows):
        raise ValueError(f"{run}'s {scope} rows pin {len(rows)} audit budgets, {len({r['b10'] for r in rows})} distinct: "
                         "the total's line needs two or more, each once")
    if moved := sorted([k for k, v in fixed.items() if len(v) != 1] + ["requests"] * (not _same(min(counts), max(counts)))):
        raise ValueError(f"{run}'s {scope} rows differ in {', '.join(moved)}, which one plan holds fixed over its "
                         "audit budgets")
    slope, intercept = _fit([(r["b10"], r["total"]) for r in rows])
    beta = (share - intercept) / slope
    if not beta > 0:
        raise ValueError(f"{run}'s {scope} rows spend {intercept:.2%} of the compute before any audit: no audit budget "
                         f"brings the total to {share:.2%}")
    lam = fixed["lambda"].pop()
    bits = _between(rows, "bound_bits", beta, log=True)
    near = min(rows, key=lambda r: abs(r["b10"] - beta))
    compute, _, period = scope.rpartition(" ")
    return {"scope": scope, "compute": compute, "period": period, "run": run, "step": rows[0]["step"],
            "mix": rows[0].get("mix"), "pinned": [r["b10"] for r in rows], "beta": beta,
            "inside": rows[0]["b10"] <= beta <= rows[-1]["b10"],
            "residual": max(abs(r["total"] - (intercept + slope * r["b10"])) for r in rows),
            **{k: v.pop() for k, v in fixed.items() if k != "lambda"}, "lambda": lam, "requests": counts[0],
            "bound": {"bits": bits, "GB": bits / GB}, "bits": bits * lam, "GB": bits * lam / GB,
            "parts": {p: _between(rows, p, beta) for p in PARTS if all(r.get(p) is not None for r in rows)},
            "priced": _held(scope, run, rows, counts[0], *plan) if plan else _own(run, near),
            "binding": _binding(run, near),
            **({"note": near["note"]} if near.get("note") else {})}


def _binding(run: str, row: Mapping) -> dict | None:
    """A row's binding bound as the planner wrote it (its `verification_phase`'s `binding`): the two-stage first-hit
    finder's bound of one verification phase of the row's design and of its scope of them, each counted in calls and
    in compressions, with the inputs it assumes and what Lean proves of one phase; beside it the row's budget and the
    phase's counts it is at. None for a row written before the planner computed one. The command computes none."""

    phase = row.get("verification_phase") or {}
    if not (bound := phase.get("binding")):
        return None
    return {"run": run, "beta": row["b10"], "scope": row["scope"],
            **{k: phase.get(k) for k in ("name", "seconds", "machines", "sampled", "sessions")}, **bound}
