"""steady_walk.py [OUT]: what walking the witness's units on the card buys the prover, read from the store and written
as `METRIC` into the run's result (`verity_bench.kit`): GPU-seconds a statement of the head block's `FIRST=false` point
at `PROVERS` prover, from the passing run of `WALK`'s point (`FC_UNIT_WALK=1`) over the same from `BASE`'s, its member
the model and the card, both rates and both runs beside it. Each point is timed by
`uv run verity-bench run SPEC --direct NODE --card N --cores CPUS`; a point without a passing run is refused, naming it.

    research run --on auto --project verity --source . --cwd source --store-read -- uv run python benchmarks/kernels/steady_walk.py
"""
from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

METRIC = "kernels/witness-walk-on-card-over-host"
UNIT = "times host walk"
BASE, WALK = "glm53-flock-steady-head-block-h200", "glm53-flock-steady-head-block-h200-walk"
#: the plan of record's prover count a card (`verity bench prove --provers`' default)
PROVERS = 1
TYPE = ('census:SparseMlaHeadBlock_v2{DR=64,DV=512,NVIS=64,FIRST=false,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v2"},'
        'SCALE={"f64":"0x1.0000000000000p-4"}}')


def gpu_seconds(row: Mapping[str, Any]) -> float:
    """A statement's GPU-seconds in a steady row: its GPUs over its statements a second."""
    return row["gpus"] / row["statements_per_s"]


def machine(card: str) -> str:
    return "-".join(card.lower().split()).removeprefix("nvidia-")


def why_not(base: Mapping[str, Any], walk: Mapping[str, Any]) -> str | None:
    """Why the two rows don't differ in the walk alone, or None where they do."""
    if base.get("unit_walk") != [False]:
        return f"{BASE}'s row ran unit_walk {base.get('unit_walk')!r}, not the host's walk alone ([False])"
    if walk.get("unit_walk") != [True]:
        return f"{WALK}'s row ran unit_walk {walk.get('unit_walk')!r}, not the card's walk in every rep ([True])"
    for key in ("type", "pin", "provers", "gpus"):
        if base.get(key) != walk.get(key):
            return f"the rows' {key} differ: {base.get(key)!r} and {walk.get(key)!r}"
    counts = [(row.get("cores") or {}).get("count") for row in (base, walk)]
    if counts[0] != counts[1]:
        return f"the rows' cores differ: {counts[0]} and {counts[1]}"
    return None


def record(out: Path, base: Mapping[str, Any], walk: Mapping[str, Any], runs: Mapping[str, str], card: str,
           model: str) -> dict:
    """Writes `METRIC` for the two passing rows; refuses rows that differ in more than the walk."""
    if why := why_not(base, walk):
        raise SystemExit(f"steady_walk: no {METRIC}: {why}")
    from verity_bench import kit

    on, off = gpu_seconds(walk), gpu_seconds(base)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps({"walk": walk, "host": base, "runs": runs}, indent=1, sort_keys=True))
    w = kit.writer(out)
    w.validation("passed", "benchmarks/kernels/steady_walk.py",
                 "both points' runs passed the steady harness's checks, and their rows differ in the walk alone",
                 ["rows.json"])
    return w.measurement(
        METRIC, on / off, UNIT, mark="measured", member=f"{model}-{machine(card)}",
        inputs=[f"{WALK} {runs['walk']}", f"{BASE} {runs['base']}"],
        extra={"block": TYPE, "provers": PROVERS, "cores": base["cores"]["count"],
               "walk_statements_per_s": walk["statements_per_s"], "host_statements_per_s": base["statements_per_s"],
               "walk_gpu_seconds": on, "host_gpu_seconds": off, "walk_run": runs["walk"], "host_run": runs["base"]})


def row_and_card(store: Any, run: str, point: Any) -> tuple[Mapping[str, Any], str]:
    """A run's row of `point`, and the card its result's host stamp names."""
    from verity_bench import steady_plan, steady_spec

    files = steady_plan.run_files(store, run)
    row = steady_spec.own_row(files, point) if files else "it kept no files"
    if isinstance(row, str):
        raise SystemExit(f"steady_walk: {run}'s row of {point.id}: {row}")
    result = store.get_manifest(store.get_attempt(run)["outputs"]["result"]).meta
    cards = {m["host"]["card"] for m in result["measurements"]}
    if len(cards) != 1:
        raise SystemExit(f"steady_walk: {run}'s result names cards {sorted(cards)}, not one")
    return row, cards.pop()


def passing(spec: str) -> tuple[Any, str, str]:
    """The spec's point at `TYPE` and `PROVERS`, its passing run and the spec's workload."""
    from verity_bench import steady_plan

    plan = steady_plan.make_plan(spec)
    [item] = [p for p in plan.points if p.point.row.type == TYPE and p.point.provers == PROVERS]
    if item.passing_run is None:
        raise SystemExit(f"steady_walk: {item.point.id} of {spec} has no passing run ({item.why}): time it with "
                         f"`uv run verity-bench run {spec} --direct NODE --card N --cores CPUS`")
    return item.point, item.passing_run, plan.spec.workload


def main(out: Path) -> None:
    from research.store.local import LocalStore
    from verity_bench import prove_common

    (base_point, base_run, workload), (walk_point, walk_run, _) = passing(BASE), passing(WALK)
    store = LocalStore()
    (base, base_card), (walk, walk_card) = row_and_card(store, base_run, base_point), row_and_card(store, walk_run, walk_point)
    if base_card != walk_card:
        raise SystemExit(f"steady_walk: the points ran on a {base_card} and a {walk_card}, not one card")
    model = prove_common.WORKLOADS[workload].model.lower()
    m = record(out, base, walk, {"base": base_run, "walk": walk_run}, walk_card, model)
    print(json.dumps({k: m[k] for k in ("name", "value", "unit", "member", "walk_gpu_seconds", "host_gpu_seconds",
                                        "walk_run", "host_run")}), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else os.environ["OUT"]))
