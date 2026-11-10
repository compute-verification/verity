"""`verity bench prove`'s store side: it reads a workload's inputs from the store's labelled records, prices the
workload under a proof system (`prove`) and renders the table. It runs on any machine that reaches the store: no GPU, no
node.

A record is named in the store, never in code (`served_table`'s rule). Each input is what carries the label
`run_of_record=prove-WORKLOAD-SLOT` and no `superseded_by`:

| slot | what carries the label | what is read |
| --- | --- | --- |
| `census` | one `block-census/v1` art | its `census.json` (`prove.Census.of`) |
| `cut` | one `partition` card (`spot-card/v1`) | the block types it recuts (`prove.Census.recut`) |
| `flock-single` | one `proof_overhead` card | each block type's single-session seconds (`prove.Session`) |
| `flock-steady` | any number of runs, or arts naming their `runs` | every proof server report in their run files (`REPORTS`): `server.json` with the `types.json` beside it (`prove.Steady`) |
| `spot-sampling` | one art holding `mix.json` | the sampling plan (`prove.SamplingPlan.of`), for `--system spot` |

The command refuses an input that isn't labelled, one that is superseded and a slot two unsuperseded targets claim, and
says which and how to label it. A steady-state run that is superseded, or that holds no report, is left out and listed
under "Refused". A report is a steady-state measurement when every session was accepted and its statement cache's hit
rate was flat (`report_why`); one that isn't is named on its block type's row.

The serving rate isn't a record: it is modeled from the census's MACs a served token, the fleet's assumed serving MFU
(`FLEETS`) and the card's data-sheet peak, and the output says so.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_bench import jobs, overhead_records
from verity_bench import overhead as O
from verity_bench import prove as P
from verity_bench.served_table import RUN_OF_RECORD, SUPERSEDED_BY, record_runs, records

CENSUS_KIND = "block-census/v1"
CENSUS_FILE = "census.json"
CARD_SCHEMA = "spot-card/v1"
PLAN_FILE = "mix.json"
#: a proof server's report in a run's files, and the staged types it proved beside it
REPORTS, TYPES = "*/server-k*/server.json", "types.json"
#: each fleet's assumed serving MFU
FLEETS = Path("benchmarks/rho/data/reference.toml")
SYSTEMS = ("flock", "spot")
CENSUS, CUT, STEADY, SINGLE, SAMPLING = "census", "cut", "flock-steady", "flock-single", "spot-sampling"


@dataclass(frozen=True)
class Workload:
    """A workload the command prices: the model its records name, the card its prover and its serving run on
    (`overhead.CARDS`), and the fleet whose serving MFU prices a served token (`FLEETS`)."""

    name: str
    model: str
    card: str
    fleet: str

    def record(self, slot: str) -> str:
        return f"prove-{self.name}-{slot}"


WORKLOADS = {w.name: w for w in (Workload("glm53", "GLM-5.3", "h200", "world-h200"),)}


# ---- records ---------------------------------------------------------------------------------------------------------------


def _superseded(store: Any, target: str) -> list[str]:
    return [str(lab.value) for lab in store.labels(target, key=SUPERSEDED_BY)]


def _label(name: str) -> str:
    return f"`research data label TARGET {RUN_OF_RECORD} {name} --by LANE`"


def of_record(store: Any, name: str, what: str, given: str | None = None) -> str:
    """The one target of record `name`: labelled and not superseded. `given` must be it."""

    labelled = records(store).get(name, [])
    if given is not None and given not in labelled:
        raise ValueError(f"{given} isn't a {what} of record: it doesn't carry `{RUN_OF_RECORD}={name}` (label it: {_label(name)})")
    found = labelled if given is None else [given]
    if not found:
        raise ValueError(f"no {what} of record: nothing in the store carries `{RUN_OF_RECORD}={name}` (label one: {_label(name)})")
    live = [t for t in found if not _superseded(store, t)]
    if not live:
        raise ValueError(f"the {what} of record is superseded: "
                         + ", ".join(f"{t} by {', '.join(_superseded(store, t))}" for t in found)
                         + f" (label what replaces it: {_label(name)})")
    if len(live) > 1:
        raise ValueError(f"{len(live)} targets carry `{RUN_OF_RECORD}={name}` and none is superseded: {', '.join(live)}; "
                         f"label each one a newer {what} replaces `{SUPERSEDED_BY}`")
    return live[0]


def _read(store: Any, art: str, path: str | None = None) -> dict:
    """An art's JSON: a file art's payload, or `path` of a tree's."""

    if store.get_manifest(art).payload.type == "file":
        return json.loads(Path(store.fetch(art)).read_text())
    if not store.listing(art, [path]):
        raise ValueError(f"{art} holds no `{path}`")
    return json.loads((Path(store.fetch(art, paths=[path])) / path).read_text())


def card(store: Any, art: str, slot: str, workload: Workload) -> dict:
    """A `spot-card/v1` card's value, checked to be `slot`'s card of the workload's model."""

    doc = _read(store, art)
    got = {k: doc.get(k) for k in ("schema", "slot", "model")}
    if got != {"schema": CARD_SCHEMA, "slot": slot, "model": workload.model}:
        raise ValueError(f"{art} isn't a `{slot}` card of {workload.model} ({CARD_SCHEMA}): it is {got}")
    return doc["value"]


def census(store: Any, workload: Workload) -> tuple[P.Census, list[dict]]:
    """The workload's census of record under its cut of record, and the two inputs."""

    art = of_record(store, workload.record(CENSUS), "block census")
    manifest = store.get_manifest(art)
    if manifest.kind != CENSUS_KIND or manifest.meta.get("model") != workload.model:
        raise ValueError(f"{art} isn't a block census of {workload.model} ({CENSUS_KIND}): it is a {manifest.kind} of "
                         f"{manifest.meta.get('model')}")
    whole = P.Census.of(_read(store, art, CENSUS_FILE))
    cut_art = of_record(store, workload.record(CUT), "cut")
    cut = card(store, cut_art, "partition", workload)
    recut = whole.recut(cut)
    return recut, [
        _input(CENSUS, workload, [art], f"{len(whole.blocks)} block types holding {whole.macs_in_blocks:.1%} of a served "
               f"token's {whole.served_macs_per_token:.4g} MACs (run {manifest.meta.get('run')})"),
        _input(CUT, workload, [cut_art], f"`{cut['name']}`, {len(recut.blocks)} block types under it: {cut.get('label')}")]


def report_why(report: Mapping[str, Any]) -> str | None:
    """Why a proof server's report isn't a steady-state measurement, or None when it is one."""

    if (report.get("host_limit") or {}).get("stopped_at_s"):
        return "stopped at the host's memory limit"
    if report.get("timed_out"):
        return "timed out"
    if any(report.get("rc") or [1]):
        return f"its provers exited {report.get('rc')}"
    if report.get("accepted") is not True:
        return "not every session was accepted"
    if report.get("flat") is not True:
        return "its statement cache's hit rate wasn't flat"
    return None


def reports(store: Any, run: str) -> tuple[list[P.Steady], str | None]:
    """Every staged type of every proof server report in a run's files, or why the run holds none."""

    attempt = store.get_attempt(run)
    if attempt is None:
        return [], "not in the store or on its remote"
    out = attempt.get("outputs") or {}
    files = out.get("run_files") or out.get("run_record")
    if not files:
        return [], "kept no run files"
    held = {row["path"] for row in store.listing(files, [REPORTS, REPORTS.rsplit("/", 1)[0] + "/" + TYPES])}
    paths = sorted(p for p in held if p.endswith("/server.json") and p.rsplit("/", 1)[0] + "/" + TYPES in held)
    if not paths:
        return [], f"holds no proof server report (`{REPORTS}` with its `{TYPES}`)"
    tree = Path(store.fetch(files, paths=[*paths, *(p.rsplit("/", 1)[0] + "/" + TYPES for p in paths)]))
    found = []
    for path in paths:
        where = path.rsplit("/", 1)[0]
        report, types = json.loads((tree / path).read_text()), json.loads((tree / where / TYPES).read_text())
        why = report_why(report)
        for name, staged in report["types"].items():
            gpu, verify = ((report.get(k) or {}).get(name) for k in ("gpu_s_by_type", "verify_s"))
            found.append(P.Steady(
                name, types[name]["definition"], staged["k_log"], staged["c"], staged["full"], run, where,
                gpu and gpu["mean"], verify and verify["mean"],
                why or (None if gpu else "no statement of it finished in the steady span")))
    return found, None


def flock(store: Any, workload: Workload) -> tuple[P.Flock, list[dict], list[dict]]:
    """Flock's prover from the workload's records: the steady-state runs', the single-session card's; the inputs, and
    the steady-state runs left out with why."""

    art = of_record(store, workload.record(SINGLE), "single-session card")
    value = card(store, art, "proof_overhead", workload)
    sessions = tuple(P.Session(b["block"], b["k_log"], b["seconds_per_block"], b["fixed_seconds"], b["statement_blocks"])
                     for b in value["blocks"])
    steady, runs, refused = [], [], []
    labelled = record_runs(store, records(store).get(workload.record(STEADY), []))
    for run in sorted({run for run, _ in labelled}, reverse=True):
        if superseded := _superseded(store, run):
            found, why = [], f"superseded by {', '.join(superseded)}"
        else:
            found, why = reports(store, run)
        if why:
            refused.append({"target": run, "record": workload.record(STEADY), "why": why})
            continue
        steady += found
        runs.append(run)
    measured = len({s.definition for s in steady if s.measured})
    return P.Flock(tuple(steady), sessions, art), [
        _input(STEADY, workload, runs, f"{measured} staged types measured at steady state" if runs else
               "none of record: no block type is measured at steady state"),
        _input(SINGLE, workload, [art], f"{len(sessions)} block types on the {value.get('device')}")], refused


def plan(store: Any, workload: Workload, given: str | None) -> tuple[P.SamplingPlan, dict]:
    """The workload's sampling plan of record (`given` must be it), and its input."""

    art = of_record(store, workload.record(SAMPLING), "sampling plan", given)
    got = P.SamplingPlan.of(_read(store, art, PLAN_FILE), art)
    return got, _input(SAMPLING, workload, [art], f"`{got.name}`, q = {got.q:.3g}, {got.scope}: {got.period_s:.4g} s on "
                       f"{got.serving_gpu_seconds / got.period_s:.4g} serving GPUs")


def serving(root: Path, workload: Workload) -> P.Serving:
    """The stated serving rate: the fleet's assumed serving MFU of the card's data-sheet peak."""

    gpu = O.CARDS[workload.card]
    peak = overhead_records.census(root)["flop_per_s"][gpu.gpu]
    mfu = tomllib.loads((root / FLEETS).read_text())["fleet"][workload.fleet]["mfu_serve"]
    return P.Serving(mfu["value"], peak,
                     f"MFU {mfu['value']:g} ({mfu['status']}, {mfu['low']:g} to {mfu['high']:g}: {mfu['source']}; {FLEETS} "
                     f"`{workload.fleet}`) of the {gpu.name}'s data-sheet peak, {peak:.5g} FLOP/s "
                     f"({overhead_records.CENSUS}/hardware.json `{gpu.datasheet}`), at {P.FLOP_PER_MAC} FLOP a MAC")


def _input(slot: str, workload: Workload, targets: list[str], what: str) -> dict:
    return {"slot": slot, "record": workload.record(slot), "targets": targets, "what": what}


# ---- the table ---------------------------------------------------------------------------------------------------------------


def build(system: str, workload: str, *, sampling: str | None = None, store: Any = None, root: Path | None = None,
          local_only: bool = False) -> dict:
    """A proof system's table on a workload from the store's records: `prove.report`'s rows, sources and totals, with
    the inputs read, the runs refused, the stated serving rate and the totals a monitoring period over all compute."""

    if system != "spot" and sampling:
        raise ValueError(f"--sampling is SPOT's: {system} proves every block (pass --system spot)")
    root = root or jobs.repository_root()
    completion = jobs.StoreCompletion(store, root=root, local_only=local_only)
    if not local_only:
        completion.pull_labels()
    store, w = completion.store, WORKLOADS[workload]
    blocks, inputs = census(store, w)
    prover, more, refused = flock(store, w)
    rate = serving(root, w)
    seconds = rate.gpu_seconds_per_token(blocks)
    inputs += more
    sampled = None
    if system == "spot":
        sampled, its = plan(store, w, sampling)
        inputs.append(its)
        prover = P.Spot(prover, sampled, rate)
    doc = P.report(prover.table(blocks), seconds)
    return {"schema": P.SCHEMA, "workload": w.name, "model": w.model, "cut": blocks.cut,
            "macs_in_blocks": blocks.macs_in_blocks, "inputs": inputs,
            "refused": refused, "serving": {"gpu_seconds_per_token": seconds, "status": "modeled", "basis": rate.basis},
            **doc, "periods": _periods(root, rate, doc["totals"]["gpus_per_serving_gpu"], sampled)}


def _periods(root: Path, rate: P.Serving, ratio: float, sampled: P.SamplingPlan | None) -> dict:
    """The totals a monitoring period over all compute. With every block proved the prover's GPUs a serving GPU hold for
    any compute and period, so the census's datacenter and the world's serving each get a week, a month and a year; a
    sampling plan is sized for one compute and period, and only that one is given."""

    c = overhead_records.census(root)
    gates = rate.flop_per_second * rate.mfu * c["gates_per_flop"]
    year = O.PERIODS_H["year"] * 3600
    if sampled is not None:
        fleets = {sampled.scope: sampled.serving_gpu_seconds / sampled.period_s}
        spans = {name: hours * 3600 for name, hours in O.PERIODS_H.items() if hours * 3600 == sampled.period_s} or {
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


# ---- rendering -----------------------------------------------------------------------------------------------------------------


def _n(x: float | None) -> str:
    if x is None:
        return "not measured"
    if x == 0:
        return "0"
    if abs(x) >= 1e6 or abs(x) < 1e-3:
        return f"{x:.3e}"
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.3g}"


def _range(pair: list[float]) -> str:
    return "" if pair[0] == pair[1] else f" (range {_n(pair[0])} to {_n(pair[1])})"


def _short(text: str | None, width: int = 64) -> str:
    text = text or "—"
    return text if len(text) <= width else text[:width - 1] + "…"


def _record(target: str | None) -> str:
    return "—" if target is None else target[:16] if target.startswith("art:") else target


def render(doc: Mapping[str, Any], rows: int | None = None) -> str:
    """The table as Markdown; `rows`: only the block types that take the most proving time, the rest in one row."""

    t, spot = doc["totals"], doc["system"].startswith("spot")
    lines = [f"# verity bench prove: {doc['system']} on {doc['workload']} ({doc['model']})", "",
             f"Inputs, each a labelled record of the store (`{RUN_OF_RECORD}`):", ""]
    lines += [f"- {i['slot']}: {', '.join(i['targets']) or 'none'} (`{i['record']}`): {i['what']}" for i in doc["inputs"]]
    s = doc["serving"]
    lines += [f"- serving rate ({s['status']}, a stated input, not a record): {_n(s['gpu_seconds_per_token'])} GPU-seconds a "
              f"served token, {_n(1 / s['gpu_seconds_per_token'])} tokens a serving GPU-second: {s['basis']}", "",
              f"| block type | k_log | blocks {'proved ' if spot else ''}per token | ms per block | source | record | "
              "share of proving time | note |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    shown = doc["rows"] if rows is None else doc["rows"][:rows]
    for r in shown:
        ms = "—" if r["seconds"] is None else _n(1e3 * r["seconds"])
        share = "—" if r["time_share"] is None else f"{r['time_share']:.2%}"
        lines.append(f"| `{_short(r['block'])}` | {'—' if r['k_log'] is None else r['k_log']} | {_n(r['per_token'])} | {ms} | "
                     f"{P.SOURCES[r['source']]} | {_record(r['record'])} | {share} | {r['note'] or ''} |")
    if rest := doc["rows"][len(shown):]:
        lines.append(f"| {len(rest)} more block types | | {_n(sum(r['per_token'] for r in rest))} | | | | "
                     f"{sum(r['time_share'] or 0.0 for r in rest):.2%} | |")
    lines += ["", "## At the sampling plan" if spot else "## Every block proved", "",
              f"- blocks {'proved ' if spot else ''}per token: {_n(t['blocks_per_token'])}",
              f"- prover GPU-seconds per token: {_n(t['gpu_seconds_per_token'])}{_range(t['gpu_seconds_per_token_range'])}",
              f"- {'served tokens covered' if spot else 'tokens proved'} per second per prover GPU: "
              f"{_n(t['tokens_per_gpu_second'])}{_range(t['tokens_per_gpu_second_range'])}",
              f"- prover GPUs per serving GPU: {_n(t['gpus_per_serving_gpu'])}{_range(t['gpus_per_serving_gpu_range'])}, "
              f"{t['status']}: steady-state measurements cover {t['steady_time_share']:.1%} of its proving time"]
    if t["range_basis"]:
        lines.append(f"- each range's other end: {t['range_basis']}")
    lines.append(f"- served MACs inside a proof block: {doc['macs_in_blocks']:.1%}; the rest are served rows no cut fits, "
                 "which no figure here prices")
    for name, key in (("verifier seconds", "verifier_seconds_per_token"), ("proof bytes", "proof_bytes_per_token")):
        if (v := t[key]) is None:
            lines.append(f"- {name} per token: not measured")
        else:
            basis = f" ({t['verifier_basis']})" if key.startswith("verifier") and t["verifier_basis"] else ""
            lines.append(f"- {name} per token: {_n(v['value'])} on the {v['block_types']} block types a record holds them "
                         f"for, {v['block_share']:.1%} of the blocks{basis}; the rest: not measured")
    missing = doc["sources"][-1]
    if missing["block_types"]:
        lines.append(f"- {missing['block_types']} block types ({missing['block_share']:.2%} of the blocks) have no figure: the "
                     "totals leave them out")
    lines += ["", "Share of the proving time each source covers:", "",
              "| source | block types | blocks per token | share of blocks | prover GPU-seconds per token | "
              "share of proving time |", "| --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {P.SOURCES[x['source']]} | {x['block_types']} | {_n(x['per_token'])} | "
              f"{'—' if x['block_share'] is None else format(x['block_share'], '.2%')} | {_n(x['gpu_seconds_per_token'])} | "
              f"{'—' if x['time_share'] is None else format(x['time_share'], '.2%')} |" for x in doc["sources"]]
    p = doc["periods"]
    lines += ["", f"Per monitoring period, over all compute ({p['gates_basis']}); every total is the table's ratio scaled, "
              "not measured at that size:", "",
              "| compute | period | serving GPUs | served AND gates | prover GPUs | prover GPU-seconds |",
              "| --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {x['compute']} | {x['period']} | {_n(x['serving_gpus'])} | {_n(x['served_gates'])} | {_n(x['prover_gpus'])} | "
              f"{_n(x['prover_gpu_seconds'])} |" for x in p["rows"]]
    if p["only"]:
        lines += ["", p["only"][0].upper() + p["only"][1:] + "."]
    if doc["refused"]:
        lines += ["", "Refused (not read):", ""]
        lines += [f"- {x['target']} (`{x['record']}`): {x['why']}" for x in doc["refused"]]
    return "\n".join(lines) + "\n"
