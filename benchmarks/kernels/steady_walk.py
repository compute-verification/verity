"""steady_walk.py [--out DIR]: what walking the witness's units on the card buys the prover, read from the store and written
as `METRIC` into the run's result (`verity_bench.kit`), one member a pair of `PAIRS`: GPU-seconds a statement of the
pair's block at `PROVERS` prover, from the passing run of its walk spec's point (`FC_UNIT_WALK=1`) over the same from its
base spec's, its member the model and the card, both rates and both runs beside it. Each point is timed by
`uv run verity-bench run SPEC --direct NODE --card N --cores CPUS`. A pair whose points have no passing runs, whose
rows differ in more than the walk, or whose card no card record's `driver` names, writes its member without a value and
says why; a run where no pair has one is refused.

    research run --on auto --project verity --source . --cwd source --store-read -- uv run python benchmarks/kernels/steady_walk.py
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

METRIC = "kernels/witness-walk-on-card-over-host"
UNIT = "times host walk"
#: the plan of record's prover count a card (`verity bench prove --provers`' default)
PROVERS = 1


class Pair(NamedTuple):
    base: str
    walk: str
    type: str


PAIRS = (
    Pair("glm53-flock-steady-head-block-h200", "glm53-flock-steady-head-block-h200-walk",
         'census:SparseMlaHeadBlock_v2{DR=64,DV=512,NVIS=64,FIRST=false,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v2"},'
         'SCALE={"f64":"0x1.0000000000000p-4"}}'),
    Pair("llama32-1b-flock-steady-gemm-h200", "llama32-1b-flock-steady-gemm-h200-walk",
         'census:Gemm_v2{K=2048,N=16,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}'),
)


def gpu_seconds(row: Mapping[str, Any]) -> float:
    """A statement's GPU-seconds in a steady row: its GPUs over its statements a second."""
    return row["gpus"] / row["statements_per_s"]


def machine(spec_card: str, reported: str) -> tuple[str, str | None]:
    """The member's card: the ``short`` of the card record whose ``driver`` holds ``reported`` (`kit.card_short`), else
    the spec's card and why."""
    from verity_bench import kit
    from verity_bench.overhead import CARD_FILES

    try:
        return kit.card_short(reported, directory=CARD_FILES), None
    except SystemExit as refused:
        return spec_card, str(refused)


def why_not(pair: Pair, base: Mapping[str, Any], walk: Mapping[str, Any]) -> str | None:
    """Why the two rows don't differ in the walk alone, or None where they do."""
    if base.get("unit_walk") != [False]:
        return f"{pair.base}'s row ran unit_walk {base.get('unit_walk')!r}, not the host's walk alone ([False])"
    if walk.get("unit_walk") != [True]:
        return f"{pair.walk}'s row ran unit_walk {walk.get('unit_walk')!r}, not the card's walk in every rep ([True])"
    for key in ("type", "pin", "provers", "gpus"):
        if base.get(key) != walk.get(key):
            return f"the rows' {key} differ: {base.get(key)!r} and {walk.get(key)!r}"
    counts = [(row.get("cores") or {}).get("count") for row in (base, walk)]
    if counts[0] != counts[1]:
        return f"the rows' cores differ: {counts[0]} and {counts[1]}"
    return None


def entry(pair: Pair, model: str, card: str, base: Mapping[str, Any] | None = None, walk: Mapping[str, Any] | None = None,
          runs: Mapping[str, str] | None = None, why: str | None = None) -> dict:
    """One member's measurement, `card` its card's short name: the ratio of two passing rows that differ in the walk
    alone, else no value and why."""
    member = f"{model}-{card}"
    if why is None and base is not None and walk is not None:
        why = why_not(pair, base, walk)
    if why is not None or base is None or walk is None or runs is None:
        return {"member": member, "value": None, "why": why or "no rows", "inputs": [],
                "extra": {"block": pair.type, "provers": PROVERS}}
    on, off = gpu_seconds(walk), gpu_seconds(base)
    return {"member": member, "value": on / off, "why": None,
            "inputs": [f"{pair.walk} {runs['walk']}", f"{pair.base} {runs['base']}"],
            "extra": {"block": pair.type, "provers": PROVERS, "cores": base["cores"]["count"],
                      "walk_statements_per_s": walk["statements_per_s"], "host_statements_per_s": base["statements_per_s"],
                      "walk_gpu_seconds": on, "host_gpu_seconds": off, "walk_run": runs["walk"], "host_run": runs["base"]}}


def record(out: Path, entries: Sequence[dict], rows: Mapping[str, Any]) -> list[dict]:
    """Writes `METRIC`, one member an entry; refuses a run where no entry has a value."""
    if all(e["value"] is None for e in entries):
        raise SystemExit(f"steady_walk: no {METRIC}: " + "; ".join(f"{e['member']}: {e['why']}" for e in entries))
    from verity_bench import kit

    w = kit.writer(out)
    w.out.mkdir(parents=True, exist_ok=True)
    (w.out / "rows.json").write_text(json.dumps(rows, indent=1, sort_keys=True))
    valued = [e["member"] for e in entries if e["value"] is not None]
    w.validation("passed", "benchmarks/kernels/steady_walk.py",
                 f"{', '.join(valued)}: both points' runs passed the steady harness's checks, and their rows differ in the "
                 "walk alone", ["rows.json"])
    return [w.measurement(METRIC, e["value"], UNIT, inputs=e["inputs"], why=e["why"], mark="measured", member=e["member"],
                          extra=e["extra"]) for e in entries]


def row_and_card(store: Any, run: str, point: Any) -> tuple[Mapping[str, Any] | str, str | None]:
    """A run's row of `point` (or why it has none), and the card its result's host stamp names."""
    from verity_bench import steady_plan, steady_spec

    files = steady_plan.run_files(store, run)
    row = steady_spec.own_row(files, point) if files else "it kept no files"
    if isinstance(row, str):
        return f"{run}'s row of {point.id}: {row}", None
    result = store.get_manifest(store.get_attempt(run)["outputs"]["result"]).meta
    cards = {m["host"]["card"] for m in result["measurements"]}
    if len(cards) != 1:
        return f"{run}'s result names cards {sorted(cards)}, not one", None
    return row, cards.pop()


def passing(spec: str, block: str) -> tuple[Any, str | None, str | None, Any]:
    """The spec's point at `block` and `PROVERS`, its passing run (or None and why) and the spec."""
    from verity_bench import steady_plan

    plan = steady_plan.make_plan(spec)
    [item] = [p for p in plan.points if p.point.row.type == block and p.point.provers == PROVERS]
    if item.passing_run is None:
        return item.point, None, (f"{item.point.id} of {spec} has no passing run ({item.why}): time it with "
                                  f"`uv run verity-bench run {spec} --direct NODE --card N --cores CPUS`"), plan.spec
    return item.point, item.passing_run, None, plan.spec


def measure(store: Any, pair: Pair) -> tuple[dict, dict]:
    """The pair's entry and its rows."""
    from verity_bench import prove_common

    (bp, base_run, base_why, spec), (wp, walk_run, walk_why, _) = passing(pair.base, pair.type), passing(pair.walk, pair.type)
    model = prove_common.WORKLOADS[spec.workload].model.lower()
    if base_why or walk_why:
        return entry(pair, model, spec.card, why=base_why or walk_why), {}
    (base, base_card), (walk, walk_card) = row_and_card(store, base_run, bp), row_and_card(store, walk_run, wp)
    for found in (base, walk):
        if isinstance(found, str):
            return entry(pair, model, spec.card, why=found), {}
    if base_card != walk_card:
        return entry(pair, model, spec.card, why=f"the points ran on a {base_card} and a {walk_card}, not one card"), {}
    card, refused = machine(spec.card, walk_card)
    if refused:
        return entry(pair, model, card, why=refused), {}
    runs = {"base": base_run, "walk": walk_run}
    return entry(pair, model, card, base, walk, runs), {"walk": walk, "host": base, "runs": runs}


def main(out: Path) -> None:
    from research.store.local import LocalStore

    store = LocalStore()
    measured = [measure(store, pair) for pair in PAIRS]
    for m in record(out, [e for e, _ in measured], {e["member"]: rows for e, rows in measured if rows}):
        print(json.dumps({k: m.get(k) for k in ("name", "value", "unit", "member", "why", "walk_gpu_seconds",
                                                "host_gpu_seconds", "walk_run", "host_run")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="out", help="where the result goes outside a research run")
    main(Path(ap.parse_args().out))
