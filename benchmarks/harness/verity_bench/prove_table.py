"""`verity bench prove`'s table (`prove_records`): a workload's inputs priced under a proof system, and the
measurements a runner records of its totals."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from verity_bench import jobs, kit, overhead_records, settings
from verity_bench import overhead as O
from verity_bench import prove as P
from verity_bench.prove_common import (
    CHECKS,
    LEAN,
    OWN_VERDICT,
    SAMPLING,
    WORKLOADS,
    record_name,
)
from verity_bench.prove_render import (
    _design,
    off_node_line,
)
from verity_bench.prove_store import (
    at_record,
    census,
    flock,
    lean,
    plan,
    serving,
    warm,
)
from verity_bench.served_table import RUN_OF_RECORD

#: the proof system whose table a runner records under its metric ids: SPOT, at the workload's sampling plan of record
RECORDED = "spot"
#: the prover GPUs a serving GPU at the exfiltration run of record's design, beside that table (`at_record`)
AT_RECORD = "gpus_per_serving_gpu_at_record"
#: the figures a runner records, by their key in the table's `totals` (`AT_RECORD`: the document's own), with the unit
#: each measurement carries
FIGURES = {"gpus_per_serving_gpu": "prover GPUs a serving GPU",
           "gpu_seconds_per_token": "prover GPU-seconds a served token",
           "lean_verifier_seconds_per_token": "Lean verifier seconds a served token",
           "proof_bytes_per_token": "proof bytes a served token",
           AT_RECORD: "prover GPUs a serving GPU"}
#: the target the goal benchmarks/proof-system-performance holds the prover GPUs a serving GPU to, at most this many:
#: proving's share of the 10% compute budget, in prover GPUs a serving GPU; replaced when the proof system's own
#: number is set
GPUS_PER_SERVING_GPU_TARGET = 0.0545
#: the figures whose measurements carry that target, lower being better; no other figure carries one
TARGETED = ("gpus_per_serving_gpu", AT_RECORD)
#: the unit of each Lean figure's fitted rate, as the table prints it
LEAN_RATE_UNITS = {"lean_verifier_seconds_per_token": "s an AND gate", "proof_bytes_per_token": "bytes a witness bit"}
#: the table a run's result keeps beside its measurements, relative to the run's directory
TABLE = "out/prove.json"


# ---- the table ---------------------------------------------------------------------------------------------------------------


def build(system: str, workload: str, *, sampling: str | None = None, provers: int = 1, cores: int = P.HOST_CORES,
          store: Any = None, root: Path | None = None, local_only: bool = False) -> dict:
    """A proof system's table on a workload from the store's records, each type priced at `provers` provers on a host
    of `cores` CPU cores a card: `prove.report`'s rows, sources and totals, with the inputs read, the runs and rows
    refused, the stated serving rate and the totals a monitoring period over all compute. SPOT's also holds the
    prover GPUs a serving GPU at the exfiltration run of record's design (`at_record`)."""

    if system != "spot" and sampling:
        raise ValueError(f"--sampling is SPOT's: {system} proves every block (pass --system spot)")
    if provers < 1:
        raise ValueError(f"--provers {provers}: a type is priced at one prover or more")
    if cores < 1:
        raise ValueError(f"--cores {cores}: a type is priced on a host of one core a card or more")
    root = root or jobs.repository_root()
    completion = jobs.StoreCompletion(store, root=root, local_only=local_only)
    if not local_only:
        completion.pull_labels()
    store, w = completion.store, WORKLOADS[workload]
    blocks, inputs = census(store, w)
    prover, more, refused = flock(store, w, blocks, provers, cores)
    timed, its, unread = lean(store, w, blocks)
    warmed, its_warm, unwarmed = warm(store, w, blocks)
    prover = replace(prover, lean=tuple(timed), warm=tuple(warmed))
    rate = serving(root, w)
    seconds = rate.gpu_seconds_per_token(blocks)
    inputs += more + its + its_warm
    refused += unread + unwarmed
    sampled = None
    priced: P.ProofSystem = prover
    if system == "spot":
        sampled, planned = plan(store, w, sampling)
        inputs.append(planned)
        priced = P.Spot(prover, sampled, rate)
    doc = P.report(priced.table(blocks), seconds)
    return {"schema": P.SCHEMA, "workload": w.name, "model": w.model, "cut": blocks.cut, "provers": provers, "cores": cores,
            "macs_in_blocks": blocks.macs_in_blocks, "inputs": inputs,
            "refused": refused, "serving": {"gpu_seconds_per_token": seconds, "status": "modeled", "basis": rate.basis},
            **doc, "periods": _periods(root, rate, doc["totals"]["gpus_per_serving_gpu"], sampled),
            **({} if sampled is None else {"at_record": at_record(store)})}


def _periods(root: Path, rate: P.Serving, ratio: float, sampled: P.SamplingPlan | None) -> dict:
    """The totals a monitoring period over all compute. With every block proved the prover GPUs a serving GPU hold for
    any compute and period, so the census's datacenter and the world's serving each get a week, a month and a year; a
    sampling plan is sized for one compute and period, and only that one is given."""

    c = overhead_records.census(root)
    gates = rate.flop_per_second * rate.mfu * c["gates_per_flop"]
    year = O.PERIODS_H["year"] * 3600
    if sampled is not None:
        fleets = {sampled.scope: sampled.serving_gpu_seconds / sampled.period_s}
        spans: dict[str, float] = {name: hours * 3600 for name, hours in O.PERIODS_H.items()
                                   if hours * 3600 == sampled.period_s} or {
            f"{sampled.period_s:g} s": sampled.period_s}
    else:
        fleets = {f"datacenter ({c['datacenter']})": float(c["gpus"]),
                  f"world ({c['world']})": c["world_flop_per_year"] / (rate.flop_per_second * rate.mfu * year)}
        spans = {name: hours * 3600 for name, hours in O.PERIODS_H.items()}
    return {"gates_basis": f"served AND gates at {c['gates_per_flop']:g} a FLOP ({c['gates_basis']}), a serving GPU at "
                           f"the stated MFU of its data-sheet peak",
            "only": None if sampled is None else "a sampling plan is sized for its compute and its period: no plan of "
                                                 "record covers another period, or a datacenter",
            "rows": [{"compute": fleet, "serving_gpus": gpus, "prover_gpus": ratio * gpus, "period": name, "period_s": span,
                      "served_gates": gpus * span * gates, "prover_gpu_seconds": ratio * gpus * span}
                     for fleet, gpus in fleets.items() for name, span in spans.items()]}


# ---- the result ------------------------------------------------------------------------------------------------------------


#: the kit's marks (`kit.MARKS`) from the strongest to the weakest: a total computed from two figures takes the weaker
#: of their marks
STRENGTH = ("measured", "partial", "modeled")


def _mark(doc: Mapping[str, Any], figure: str) -> tuple[float | None, str, str]:
    """A total's value, its mark (`kit.MARKS`) and that mark in the table's words. The proving totals: their status
    (`measured`, `modeled`, or `partial` with how many block types it leaves out), the share of the proving time
    steady-state rows measure and the range's other end. The Lean verifier's seconds and the proof's bytes: `not
    measured`, with no value and the mark `modeled` (no run of record measures every block type, so a figure the
    table has is in part fitted), else the status, each source's share of the figure (own, family, fitted) with the
    fitted rate, and the block type measured with what its accept holds (its record's `label`). The figure at the
    exfiltration run of record's design (`AT_RECORD`): modeled, with how it is read from that run's rows, or `not
    read` and why."""

    if figure == AT_RECORD:
        r = doc["at_record"]
        if r["gpus_per_serving_gpu"] is None:
            return None, r["status"], f"not read: {r['why']}"
        return r["gpus_per_serving_gpu"], r["status"], f"{r['status']}: {_design(r)}"
    t = doc["totals"]
    if figure in ("gpus_per_serving_gpu", "gpu_seconds_per_token"):
        mark = "partial" if t["missing"] else t["status"]
        status = (f"partial: {len(t['missing'])} block type(s) with no figure, whose proving it leaves out"
                  if t["missing"] else t["status"])
        low, high = t[f"{figure}_range"]
        other = "" if low == high else f"; range {low:.6g} to {high:.6g}, the other end {t['range_basis']}"
        own = (f"; at the sampling plan's own prices, which its planner sized the sample at: {t['planned'][figure]:.6g}"
               if t.get("planned") else "")
        return t[figure], mark, (f"{status}: steady-state measurements cover {t['steady_time_share']:.1%} of the proving time"
                           f"{' it prices' if t['missing'] else ''}{other}{own}")
    v = t[figure]
    if v is None:
        return None, "modeled", ("not measured: no Lean verifier run of record measures a block type "
                      f"(`{RUN_OF_RECORD}={record_name(doc['workload'], LEAN)}`)")
    m = t["lean_verify"][0]
    status = "modeled, not measured" if v["status"] == "modeled" else v["status"]
    shares = ", ".join(f"{word} {x['share'] or 0.0:.1%} ({x['block_types']} block types)"
                       for k, word in P.LEAN_SOURCES.items() for x in [v["sources"][k]])
    rate = ("no fitted rate" if v["fitted_rate"] is None else
            f"fitted rate {v['fitted_rate']:.4g} {LEAN_RATE_UNITS[figure]}")
    one = "one block type" if len(t["lean_verify"]) == 1 else f"{len(t['lean_verify'])} block types, the most proving time first"
    return v["value"], v["status"], (f"{status}: {shares} ({rate}); measured on {one}, `{m['block']}` ({m['label']}, run {m['run']}): "
                        f"{v['basis']}")


def measurements(doc: Mapping[str, Any], metrics: Mapping[str, str]) -> list[dict]:
    """The kit writer's measurements of a table, one a metric id of `metrics`: the total its
    key names (`FIGURES`), in that total's unit, with its mark (`_mark`). A total with a value has its mark in the
    table's words as the first of its inputs, since the kit holds a `why` only for a measurement with no value, then
    the records the table read, then each steady-state run its own verdict refused (`own_verdict`), with why; a total
    with none has those words as its `why`. The prover GPUs a serving GPU are the
    proving total over the stated serving rate, so their mark is the weaker of the two's (`STRENGTH`), first among
    their inputs, then proving's words and the serving rate's. Both proving totals then say how many of the priced
    steady-state rows ran with their card off their CPUs' NUMA node, how many on it and how many don't record it
    (`off_node_line`):
    a statement about the rows they rest on, which moves neither value nor mark. The figure at the exfiltration run of record's design
    (`AT_RECORD`) is read from that run alone, which its inputs name in place of the table's records. The figures of
    `TARGETED` carry `GPUS_PER_SERVING_GPU_TARGET` as their `target`, and, where they have a value, its check against
    it as their `verdict`: `meets` at or under it, `misses` over it."""

    if bad := sorted(set(metrics.values()) - FIGURES.keys()):
        raise ValueError(f"no total of the table is named {', '.join(bad)}: a metric records one of {', '.join(FIGURES)}")
    if AT_RECORD in metrics.values() and "at_record" not in doc:
        raise ValueError(f"`{AT_RECORD}` is SPOT's: only a table at a sampling plan holds it")
    targets = list(dict.fromkeys(t for i in doc["inputs"] for t in i["targets"]))
    targets += [f"refused {x['target']}: {x['why']}" for x in doc["refused"] if str(x["why"]).startswith(OWN_VERDICT)]
    s, out = doc["serving"], list[dict[str, Any]]()
    for name, figure in metrics.items():
        value, mark, said = _mark(doc, figure)
        if figure == AT_RECORD:
            run = doc["at_record"]["run"]
            read = [] if run is None else [run]
            out.append({"name": name, "value": value, "unit": FIGURES[figure], "mark": mark,
                        **({"why": said, "inputs": read} if value is None else {"inputs": [said, *read]})})
            continue
        if value is None:
            out.append({"name": name, "value": None, "unit": FIGURES[figure], "why": said, "inputs": targets, "mark": mark})
            continue
        inputs = [said]
        if figure == "gpus_per_serving_gpu":
            mark = max(mark, s["status"], key=STRENGTH.index)
            inputs = [mark, f"proving: {said}", f"serving rate ({s['status']}, a stated input): {s['basis']}"]
        if (off := doc["totals"].get("off_node")) and figure in ("gpus_per_serving_gpu", "gpu_seconds_per_token"):
            inputs.append(off_node_line(off))
        out.append({"name": name, "value": value, "unit": FIGURES[figure], "inputs": [*inputs, *targets], "mark": mark})
    for m, figure in zip(out, metrics.values(), strict=True):
        if figure in TARGETED:
            m["target"] = GPUS_PER_SERVING_GPU_TARGET
            if m["value"] is not None:
                m["verdict"] = "meets" if m["value"] <= GPUS_PER_SERVING_GPU_TARGET else "misses"
    return out


def validation(doc: Mapping[str, Any]) -> tuple[str, str]:
    """The run's validation of a table, its status and detail: `CHECKS` hold a figure when some block type is priced
    from steady-state rows, each of which passed them all, and no block type is left out of the totals; else it
    failed, and the detail says which."""

    t, checks = doc["totals"], ", ".join(f"`{c.__name__}`" for c in CHECKS)
    steady_types = next(x["block_types"] for x in doc["sources"] if x["source"] == P.STEADY)
    priced = (f"{steady_types} block type(s), {t['steady_time_share']:.1%} of the proving time, priced from steady-state "
              f"rows that each passed {checks}; a row that didn't prices nothing and is named on its type's row")
    if not steady_types:
        return "failed", f"no steady-state row passed {checks} and priced a block type: no total rests on one"
    if t["missing"]:
        return "failed", f"{priced}; {len(t['missing'])} block type(s) have no figure, and every total leaves them out"
    return "passed", priced


def record(doc: Mapping[str, Any], metrics: Mapping[str, str]) -> None:
    """Writes `RECORDED`'s table into this research run's result (`kit.writer`): the table as `TABLE`, named a
    measurement file and the validation's evidence, what ran (with the sampling plan) in the fingerprint,
    `measurements` under `metrics`' ids and `validation`. Another system's table records nothing: each metric is
    `RECORDED`'s."""

    if doc["system"].split("[", 1)[0] != RECORDED:
        return
    writer = kit.writer(settings.RESEARCH_RUN_DIR)
    table = writer.out / TABLE
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    plan = next(t for i in doc["inputs"] if i["slot"] == SAMPLING for t in i["targets"])
    writer.fingerprint(benchmark="prove", system=doc["system"], workload=doc["workload"], sampling=plan,
                       provers=doc["provers"], cores=doc["cores"])
    writer.measurement_file(TABLE)
    for m in measurements(doc, metrics):
        writer.measurement(**m)
    status, detail = validation(doc)
    writer.validation(status, "verity_bench.prove_records.CHECKS", detail, [TABLE])
