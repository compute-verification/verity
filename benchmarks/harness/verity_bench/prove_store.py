"""`verity bench prove`'s inputs, each read from the store's labelled records (`prove_records`): the census and its
cut, the steady-state, single-session and warm rows, the Lean verifier's records, the sampling plan, the prover GPUs
a serving GPU at the exfiltration run of record's design and the serving rate."""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from verity_bench import exfiltration as E
from verity_bench import lean_verify as L
from verity_bench import overhead as O
from verity_bench import overhead_records
from verity_bench import prove as P
from verity_bench import prove_flock as PF
from verity_bench.boolean import boolean_version
from verity_bench.prove_common import (
    _ABSENT,
    ART_RATES,
    CENSUS,
    CUT,
    LEAN,
    RATES,
    SAMPLING,
    SINGLE,
    STEADY,
    WARMED,
    Workload,
    _count,
    _field,
    _input,
    _read,
    _superseded,
    card,
    of_record,
)
from verity_bench.prove_rows import (
    off_node,
    own_verdict,
    steady_row,
)
from verity_bench.served_table import record_runs, records

CENSUS_KIND = "block-census/v1"
CENSUS_FILE = "census.json"
PLAN_FILE = "mix.json"
#: the host a run timed its rows on, in the run's files: its `HOST_CORES_FIELD` is the CPU cores the run's provers held
#: for its card, which a row of its `RATES` that states no `cores` is read at
HOST, HOST_CORES_FIELD = "out/host.json", "cpus_used"
#: a proof server's report in the files of a run from before `RATES`, the staged types it proved beside it, and its K
#: (`server-kK`) the provers it ran; read only where a run holds no `RATES`
REPORTS, TYPES = "*/server-k*/server.json", "types.json"
#: the Lean verifier's record in a run's files, as `verity bench prove lean-verify` writes it under the run's `$OUT`
LEAN_RECORD = "out/" + L.RECORD
#: what a `lean-verify` record holds for the command to read it
LEAN_FIELDS = ("block.k_log", "block.blocks", "block.full", "proof.bytes", "verifier.runs", "accepted", "label", "host.node",
               "host.cores", "host.cpu_model")
#: each fleet's assumed serving MFU
FLEETS = Path("benchmarks/slowdown/data/reference.toml")
#: the scope of the exfiltration run of record's rows that `AT_RECORD` is read over, at `exfiltration.SHARE`: the
#: planner's world year, `verity bench exfiltration`'s figure
DESIGN_SCOPE = "world year"
#: each card's group of a `stage_prove.sh` run: `prove_time.py`'s summary and `prove_time.sh`'s config, whose last
#: fields are the warm and timed sessions it ran a statement (`warm=W runs=R`)
WARM_SUMMARY, WARM_CONFIG = "out/gpu*/out/summary.json", "out/gpu*/out/config.txt"


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


def rates(store: Any, run: str) -> tuple[list[PF.Steady], list[str]]:
    """Every row of a run's `RATES` that names a staged Definition, or, in a run that holds none, every staged type of
    its proof server reports; and why the rest (or the run) can't be read. A `RATES` row that states no `cores` is
    read at the run's own (`HOST`'s `HOST_CORES_FIELD`), where the run's files state them; any other row that states
    none, and every staged type of a report, at `prove.UNSTATED_CORES`. A run whose own verdict is against its rows
    (`own_verdict`) has each refused with it, and the verdict first among its reasons."""

    attempt = store.get_attempt(run)
    if attempt is None:
        return [], ["not in the store or on its remote"]
    out = attempt.get("outputs") or {}
    files = out.get("run_files") or out.get("run_record")
    if not files:
        return [], ["kept no run files"]
    verdict = own_verdict(store, run, attempt)
    kept = {row["path"] for row in store.listing(files, [RATES, HOST])} & {RATES, HOST}
    if RATES in kept:
        tree = Path(store.fetch(files, paths=sorted(kept)))
        host = json.loads((tree / HOST).read_text()) if HOST in kept else None
        stated = host.get(HOST_CORES_FIELD) if isinstance(host, Mapping) else None
        found, whys = _rows(json.loads((tree / RATES).read_text()), run,
                            unstated=stated if _count(stated) else P.UNSTATED_CORES, refused=verdict)
        return found, [verdict, *whys] if verdict else whys
    if verdict:
        return [], [verdict]
    held = {row["path"] for row in store.listing(files, [REPORTS, REPORTS.rsplit("/", 1)[0] + "/" + TYPES])}
    paths = sorted(p for p in held if p.endswith("/server.json") and p.rsplit("/", 1)[0] + "/" + TYPES in held)
    if not paths:
        return [], [(f"holds no proof server rate (`{RATES}`, which `benchmarks/one_stage/server_rate.py` writes) and no "
                     f"proof server report (`{REPORTS}` with its `{TYPES}`)")]
    tree = Path(store.fetch(files, paths=[*paths, *(p.rsplit("/", 1)[0] + "/" + TYPES for p in paths)]))
    found = []
    for path in paths:
        where = path.rsplit("/", 1)[0]
        report, types = json.loads((tree / path).read_text()), json.loads((tree / where / TYPES).read_text())
        why, provers = report_why(report), int(where.rsplit("/server-k", 1)[1])
        for name, staged in report["types"].items():
            gpu, verify = ((report.get(k) or {}).get(name) for k in ("gpu_s_by_type", "verify_s"))
            found.append(PF.Steady(
                name, types[name]["definition"], staged["k_log"], staged["c"], staged["full"], run, where,
                gpu and gpu["mean"], verify and verify["mean"],
                why or (None if gpu else "no statement of it finished in the steady span"), provers=provers,
                off_node=off_node({}, run)))
    return found, []


def report_why(report: Mapping[str, Any]) -> str | None:
    """Why a proof server's report (`REPORTS`, from runs before `RATES`) isn't a steady-state measurement, or None when
    it is one."""

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


def _rows(doc: Any, run: str, art: str | None = None, unstated: int = P.UNSTATED_CORES,
          refused: str | None = None) -> tuple[list[PF.Steady], list[str]]:
    """A rate file's rows as `run`'s: those that name a staged Definition, and why the rest (or the file) can't be read.
    A row that states no `cores` is read at `unstated`; `refused`: why the run prices none of them (`own_verdict`)."""

    if not isinstance(doc, list) or not all(isinstance(row, Mapping) for row in doc):
        return [], [f"its `{RATES if art is None else ART_RATES}` isn't a list of rows"]
    read = [steady_row(row, run, art, unstated, refused) for row in doc]
    return [x for x in read if isinstance(x, PF.Steady)], [x for x in read if isinstance(x, str)]


def steady_rows(store: Any, target: str, superseded: bool = False) -> dict[str, tuple[list[PF.Steady], list[str]]]:
    """What a target of the steady-state record gives, by the run or art read: its rows that name a staged Definition,
    and why the rest (or the target) can't be read. A run gives its own `RATES`. An art that holds `ART_RATES` gives
    those rows in place of its run's own: they are that run's, completed or corrected after it, so its meta names the
    one run in `runs`, which the rows are ordered by and whose files aren't read; its attempt is, for the run's own
    verdict (`own_verdict`), which holds of its rows completed as of its own. Any other art stands for the runs its
    meta names. A superseded target gives nothing, unless `superseded` asks for its rows anyway (for their staged
    names)."""

    if not superseded and (by := _superseded(store, target)):
        return {target: ([], [f"superseded by {', '.join(by)}"])}
    if not target.startswith("art:"):
        return {target: rates(store, target)}
    manifest = store.get_manifest(target)
    if not manifest.meta.get("runs"):
        return {target: ([], [f"its meta names no `runs`: the run its `{ART_RATES}` is of, or the runs it stands for"])}
    runs = [run for run, _ in record_runs(store, [target])]
    if manifest.payload is None or not store.listing(target, [ART_RATES]):
        return {run: found for named in runs for run, found in steady_rows(store, named, superseded).items()}
    if len(runs) != 1:
        return {target: ([], [(f"holds `{ART_RATES}` and names {len(runs)} runs ({', '.join(runs)}): an art's rows are "
                               "one run's, so each run's go in an art of their own")])}
    verdict = own_verdict(store, runs[0], store.get_attempt(runs[0]))
    found, whys = _rows(_read(store, target, ART_RATES), runs[0], target, refused=verdict)
    return {target: (found, [verdict, *whys] if verdict else whys)}


def flock(store: Any, workload: Workload, census: P.Census, provers: int = 1,
          cores: int = P.HOST_CORES) -> tuple[PF.Flock, list[dict], list[dict]]:
    """Flock's prover at `provers` provers on a host of `cores` CPU cores a card from the workload's records: the
    steady-state runs' and arts' rows (`steady_rows`), the single-session card's; the inputs, and the steady-state
    runs, arts and rows left out with why. The steady-state input counts what the rows measure of `census`: its block
    types, and the rows that are none."""

    art = of_record(store, workload.record(SINGLE), "single-session card")
    value = card(store, art, "proof_overhead", workload)
    sessions = tuple(PF.Session(b["block"], b["k_log"], b["seconds_per_block"], b["fixed_seconds"], b["statement_blocks"])
                     for b in value["blocks"])
    steady, runs, refused = [], [], []
    read: dict[str, tuple[list[PF.Steady], list[str]]] = {}
    for target in records(store).get(workload.record(STEADY), []):
        read |= steady_rows(store, target)
    for target in sorted(read, reverse=True):
        found, whys = read[target]
        refused += [{"target": target, "record": workload.record(STEADY), "why": why} for why in whys]
        if found:
            steady += found
            runs.append(target)
    aliases = sorted({(s.definition, s.type) for target in records(store).get(workload.record(STEADY), [])
                      if _superseded(store, target) for found, _ in steady_rows(store, target, True).values() for s in found})
    prover = PF.Flock(tuple(steady), sessions, art, provers=provers, cores=cores, aliases=tuple(aliases),
                     boolean=boolean_version)
    table = prover.table(census)
    measured = sum(r.cost.source == P.STEADY for r in table.rows)
    count = f"{provers} prover{'' if provers == 1 else 's'}"
    unread = f"; {len(table.unread)} row(s) of no block type of the census, listed under Unread" if table.unread else ""
    return prover, [
        _input(STEADY, workload, runs, f"{measured} staged types measured at steady state at {count}{unread}" if runs else
               "none of record: no block type is measured at steady state"),
        _input(SINGLE, workload, [art], f"{len(sessions)} block types on the {value.get('device')}")], refused


def lean_verify(doc: Any, run: str) -> PF.LeanVerify | str:
    """A `lean-verify` record as the statement it measured, or why it measures nothing: another schema, a field it
    lacks (its `label` among them: an accept the record doesn't qualify is printed by no line), a run the Lean verifier refused (the refusal is the record's result, and prices nothing), or figures that
    aren't a statement's."""

    if not isinstance(doc, Mapping) or doc.get("schema") != L.SCHEMA:
        return f"its `{LEAN_RECORD}` isn't a `{L.SCHEMA}` record"
    if lacks := [name for name in LEAN_FIELDS if _field(doc, name) is _ABSENT]:
        return "its record lacks " + ", ".join(f"`{name}`" for name in lacks)
    block, runs = doc["block"], doc["verifier"]["runs"]
    if not isinstance(runs, list) or not runs or not all(isinstance(r, Mapping) for r in runs):
        return "its record holds no run of the verifier"
    if refused := [r for r in runs if r.get("accepted") is not True]:
        return f"the Lean verifier refused its statement: {refused[0].get('why')}"
    if doc["accepted"] is not True:
        return f"the Lean verifier ran {len(runs)} of the {doc['verifier'].get('repeat')} runs it set out to"
    if not (isinstance(doc["label"], str) and doc["label"].startswith("accepted")):
        return f"its `label` ({doc['label']!r}) doesn't say what its accept holds"
    word, definition = block.get("word"), block.get("definition")
    if not isinstance(word, str) and not isinstance(definition, str):
        return "its record names no block type (`block.word`, `block.definition`)"
    seconds = [r.get("wall_s") for r in runs]
    if not (all(_count(block[k]) for k in ("k_log", "blocks", "full")) and block["blocks"] <= block["full"]
            and _count(doc["proof"]["bytes"])
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and x > 0 for x in seconds)):
        return (f"its `block` ({ {k: block[k] for k in ('k_log', 'blocks', 'full')} }), `proof.bytes` "
                f"({doc['proof']['bytes']}) and runs' `wall_s` ({seconds}) aren't a statement's blocks, bytes and seconds")
    host, form = doc["host"], _field(doc, "statement.id")
    return PF.LeanVerify(P.definition_id(word) if isinstance(word, str) else definition, block["k_log"], block["blocks"],
                        block["full"], tuple(seconds), doc["proof"]["bytes"], run,
                        f"{host['node']} ({host['cores']} cores, {host['cpu_model']})", doc["label"],
                        form if isinstance(form, str) else None)


def lean_run(store: Any, run: str) -> PF.LeanVerify | str:
    """A run of record's `LEAN_RECORD` as the statement it measured, or why the run measures nothing."""

    if superseded := _superseded(store, run):
        return f"superseded by {', '.join(superseded)}"
    if run.startswith("art:"):
        return f"an art: the Lean verifier's record is a run's (`{LEAN_RECORD}`)"
    attempt = store.get_attempt(run)
    if attempt is None:
        return "not in the store or on its remote"
    out = attempt.get("outputs") or {}
    files = out.get("run_files") or out.get("run_record")
    if not files:
        return "kept no run files"
    if not store.listing(files, [LEAN_RECORD]):
        return f"holds no `{LEAN_RECORD}`, which `verity bench prove lean-verify` writes"
    return lean_verify(json.loads((Path(store.fetch(files, paths=[LEAN_RECORD])) / LEAN_RECORD).read_text()), run)


def lean(store: Any, workload: Workload, census: P.Census) -> tuple[list[PF.LeanVerify], list[dict], list[dict]]:
    """The statements of `census`'s block types the Lean verifier accepted, from the workload's `lean-verify` runs of
    record; their input, where a run carries the label; and the runs that measure nothing, with why."""

    held = {P.definition_id(b.name) for b in census.blocks}
    found, refused = [], []
    for run in sorted(records(store).get(workload.record(LEAN), []), reverse=True):
        got = lean_run(store, run)
        if isinstance(got, PF.LeanVerify) and got.definition not in held:
            got = f"its block type, `{got.definition}`, is no block type of the census under its cut"
        if isinstance(got, str):
            refused.append({"target": run, "record": workload.record(LEAN), "why": got})
        else:
            found.append(got)
    if not found and not refused:
        return [], [], []
    types = len({v.definition for v in found})
    return found, [_input(LEAN, workload, [v.run for v in found],
                          f"the Lean verifier (its `verify`) on one statement of each of {types} block type(s); "
                          "every other type's Lean verifier seconds and proof bytes are priced from them, its "
                          "Definition's at its size, else at the rate fitted over them" if found else
                          "none that measures a block type: the Lean verifier's seconds and the proof's bytes are not "
                          "measured")], refused


def warm_rows(summary: Any, config: str, run: str) -> list[PF.Warm | str]:
    """A group's statements proved on one card from warm sessions (`WARM_SUMMARY`, `WARM_CONFIG`), each as the
    statement it measured, or why it measures nothing: one on the CPU, one that names no census type (`word`), a
    session not accepted, fewer timed sessions than the group's RUNS, or figures that aren't a statement's. A full
    statement holds `prove.STATEMENT_BITS` over 2^k_log blocks."""

    got = re.search(r"\bwarm=(\d+) runs=(\d+)\s*$", config)
    statements = summary.get("statements") if isinstance(summary, Mapping) else None
    if not got or not isinstance(statements, Mapping):
        return [f"its `{WARM_CONFIG}` states no `warm=W runs=R`, or its `{WARM_SUMMARY}` no `statements`"]
    warm, runs = int(got[1]), int(got[2])
    out: list[PF.Warm | str] = []
    for key, st in sorted(statements.items()):
        name = key.partition("/")[2]
        if not isinstance(st, Mapping) or st.get("device") != "gpu":
            out.append(f"`{name}`: not proved on a card")
        elif not isinstance(st.get("word"), str):
            out.append(f"`{name}`: names no census type (`word`)")
        elif not (isinstance(st.get("accepted"), list) and st["accepted"] and all(a is True for a in st["accepted"])):
            out.append(f"`{name}`: not every session was accepted ({st.get('accepted')})")
        elif st.get("timed") != runs or len(st["accepted"]) != warm + runs:
            out.append(f"`{name}`: {st.get('timed')} timed of {len(st['accepted'])} sessions, not RUNS={runs} after WARM={warm}")
        elif not (_count(st.get("n")) and _count(st.get("k_log")) and isinstance(st.get("prove_total_s"), (int, float))
                  and st["prove_total_s"] > 0 and st["n"] <= max(1, P.STATEMENT_BITS >> st["k_log"])):
            out.append(f"`{name}`: its `n` ({st.get('n')}), `k_log` ({st.get('k_log')}) and `prove_total_s` "
                       f"({st.get('prove_total_s')}) aren't a statement's blocks and seconds")
        else:
            out.append(PF.Warm(P.definition_id(st["word"]), st["k_log"], st["n"], max(1, P.STATEMENT_BITS >> st["k_log"]),
                              st["prove_total_s"], warm, runs, run, name))
    return out


def warm_run(store: Any, run: str) -> list[PF.Warm | str]:
    """A run of record's statements proved from warm sessions, every card's group (`warm_rows`), or why the run
    measures nothing."""

    if superseded := _superseded(store, run):
        return [f"superseded by {', '.join(superseded)}"]
    attempt = store.get_attempt(run) if not run.startswith("art:") else None
    if attempt is None:
        return ["not a run in the store or on its remote"]
    out = attempt.get("outputs") or {}
    files = out.get("run_files") or out.get("run_record")
    if not files:
        return ["kept no run files"]
    held = {row["path"] for row in store.listing(files, [WARM_SUMMARY, WARM_CONFIG])}
    groups = sorted(p.rsplit("/", 1)[0] for p in held if p.endswith("/summary.json") and
                    p.rsplit("/", 1)[0] + "/config.txt" in held)
    if not groups:
        return [f"holds no `{WARM_SUMMARY}` with its `{WARM_CONFIG}`, which `stage_prove.sh` writes"]
    tree = Path(store.fetch(files, paths=[f"{g}/{f}" for g in groups for f in ("summary.json", "config.txt")]))
    return [x for g in groups
            for x in warm_rows(json.loads((tree / g / "summary.json").read_text()), (tree / g / "config.txt").read_text(), run)]


def warm(store: Any, workload: Workload, census: P.Census) -> tuple[list[PF.Warm], list[dict], list[dict]]:
    """The statements of `census`'s block types proved from warm sessions, from the workload's `flock-warm` runs of
    record; their input, where a run carries the label; and the statements and runs that measure nothing, with why."""

    held = {P.definition_id(b.name) for b in census.blocks}
    found, refused = [], []
    for run in sorted(records(store).get(workload.record(WARMED), []), reverse=True):
        for got in warm_run(store, run):
            if isinstance(got, PF.Warm) and got.definition not in held:
                got = f"`{got.statement}`: its block type, `{got.definition}`, is no block type of the census under its cut"
            if isinstance(got, str):
                refused.append({"target": run, "record": workload.record(WARMED), "why": got})
            else:
                found.append(got)
    if not found and not refused:
        return [], [], []
    return found, [_input(WARMED, workload, sorted({w.run for w in found}),
                          f"{len({w.definition for w in found})} block type(s) proved on one card from warm sessions"
                          if found else "none that measures a block type")], refused


def plan(store: Any, workload: Workload, given: str | None) -> tuple[P.SamplingPlan, dict]:
    """The workload's sampling plan of record (`given` must be it), and its input."""

    art = of_record(store, workload.record(SAMPLING), "sampling plan", given)
    got = P.SamplingPlan.of(_read(store, art, PLAN_FILE), art)
    return got, _input(SAMPLING, workload, [art], f"`{got.name}`, q = {got.q:.3g}, {got.scope}: {got.period_s:.4g} s on "
                       f"{got.serving_gpu_seconds / got.period_s:.4g} serving GPUs")


def at_record(store: Any) -> dict:
    """The prover GPUs a serving GPU at the exfiltration run of record's design, read as `verity bench exfiltration`
    reads it (`exfiltration.rows_of_record`, `exfiltration.at_share`): the planner's rows over `DESIGN_SCOPE` at
    `exfiltration.SHARE` of total compute, where its proving is `prove_share` of total compute at the audit budget
    `beta`, so a serving GPU holds prove_share / (1 - beta) prover GPUs. Modeled, as the planner's rows are; where the
    record can't be read, no value and `why`."""

    out = {"record": E.RECORD, "scope": DESIGN_SCOPE, "share": E.SHARE, "run": None, "prove_share": None, "beta": None,
           "gpus_per_serving_gpu": None, "status": "modeled", "why": None}
    try:
        scopes, plans, _, _ = E.rows_of_record(store)
        if DESIGN_SCOPE not in scopes:
            return out | {"why": f"no run of record holds the {DESIGN_SCOPE}'s rows (they hold {', '.join(sorted(scopes))})"}
        run, rows = scopes[DESIGN_SCOPE]
        p = E.at_share(DESIGN_SCOPE, run, rows, E.SHARE, plans.get(DESIGN_SCOPE))
    except ValueError as e:
        return out | {"why": str(e)}
    if "prove" not in p["parts"]:
        return out | {"run": p["run"], "why": f"{p['run']}'s {DESIGN_SCOPE} rows don't all hold their proving's share (`prove`)"}
    return out | {"run": p["run"], "prove_share": p["parts"]["prove"], "beta": p["beta"],
                  "gpus_per_serving_gpu": p["parts"]["prove"] / (1 - p["beta"])}


def serving(root: Path, workload: Workload) -> P.Serving:
    """The stated serving rate: the fleet's assumed serving MFU of the card's data-sheet peak."""

    gpu = O.CARDS[workload.card]
    peak = overhead_records.census(root)["flop_per_s"][gpu.gpu]
    mfu = tomllib.loads((root / FLEETS).read_text())["fleet"][workload.fleet]["mfu_serve"]
    return P.Serving(mfu["value"], peak,
                     f"MFU {mfu['value']:g} ({mfu['status']}, {mfu['low']:g} to {mfu['high']:g}: {mfu['source']}; {FLEETS} "
                     f"`{workload.fleet}`) of the {gpu.name}'s data-sheet peak, {peak:.5g} FLOP/s "
                     f"({overhead_records.CENSUS}/hardware.json `{gpu.datasheet}`), at {P.FLOP_PER_MAC} FLOP a MAC")
