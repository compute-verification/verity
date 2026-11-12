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
| `flock-steady` | any number of runs, or arts naming their `runs` | the rows of each run's `RATES`, the file `benchmarks/one_stage/server_rate.py` writes: a row a staged type and prover count (`prove.Steady`, `steady_row`); an art that holds `ART_RATES` gives those rows in place of its one run's own (`steady_rows`); a run from before `RATES` gives its proof server reports (`REPORTS`), a staged type of a `server-kK` report as its row at K provers |
| `spot-sampling` | one art holding `mix.json` | the sampling plan (`prove.SamplingPlan.of`), for `--system spot` |
| `lean-verify` | any number of runs of `verity bench prove lean-verify` | each run's `LEAN_RECORD`: one statement of a block type the Lean verifier accepted, its runs' seconds, its proof's bytes and its `label`, what the accept holds, which the Lean line prints word for word (`prove.LeanVerify`, `lean_verify`) |

The command refuses an input that isn't labelled, one that is superseded and a slot two unsuperseded targets claim, and
says which and how to label it. A steady-state run or art that is superseded, a run that holds neither `RATES` nor a
report and an art that names no run are left out and listed under "Refused", as is a row that names no staged Definition (one that
failed or was skipped).

A row is a steady-state measurement when it passes six checks of the proof server's steady-state rules
(`verity_bench.steady`) and holds what prices it (`unpriced`). The six are `CHECKS`, the rules' own `accepted`,
`exits`, `not_void`, `flat`, `no_throttling` and `every_prover_warmed`: every session it ran was accepted, none refused
and each filed by its verifier, every process exited 0, it has a rate, its statement cache's hit rate was flat, its
CPUs were never throttled, and each of its provers finished a warm session and filed one in the window. This reader
states none of them itself, so it and the harness give one answer on each. Only a pass counts: a row that fails a
check, or whose fields can't show it ("cannot tell"), prices nothing, and is named on its block type's row with each
check it didn't pass, the check's answer and its reason, in the rules' words, then with what it lacks to be priced
(`row_why`). The rules' four other checks are no condition here: three read a run's other files (its lease and card,
the card at its start and end, its window's sessions), and `one_numa_node` reads `cores.numa_nodes`, which a row may
lack. A labelled row is priced whatever they say (`steady.verdict` gives all ten, from the row and its run's files).
`report_why` judges a report.

A type is priced at the stated prover count (`--provers`), from the rows at that count only, and at
the stated host CPU cores a card (`--cores`): from the rows whose provers held that many, else from those that held
the fewest above it, named on the type's row and counted in the totals; a row timed at fewer cores prices nothing. A
row's cores are its `cores`, a count or a mapping with one as its `count` (`row_cores`); where a run's row states
none, its run's (`HOST`'s `HOST_CORES_FIELD`); and `prove.UNSTATED_CORES` for any other row that states none and for
every proof server report. A row
whose staged Definition is no block type of the census under its cut is on no row and in no figure: it is listed
under "Unread", with its own figure where it is a steady-state measurement.

A row prices the census type it was staged for (`stage.word`), whatever Definition was staged (`stage.definition`).
A type's row names the Definition its figure's statements are of (a row's `definition`): alone where it is the type's
own, as "its Boolean version" where the tree registers it as the type's (`boolean_version`), and as "another
Definition" where it doesn't, so a reader sees which form priced the type. A type priced from its family's row names
that row's Definition the same way.

The Lean verifier's seconds and the proof's bytes a served token are `not measured` with no `lean-verify` run of
record. With one, a block type the Lean verifier was run on takes its record's figures and every other type is scaled
from it (`prove`'s rules), and both totals print as `modeled`, each line naming what was measured (the block type, its
share of the proving time, its statement, the run and its host) and what was scaled, with the scaling's basis. A run
whose record the Lean verifier refused, that holds no record or whose block type the census doesn't hold measures
nothing and is listed under "Refused".

The serving rate isn't a record: it is modeled from the census's MACs a served token, the fleet's assumed serving MFU
(`FLEETS`) and the card's data-sheet peak, and the output says so.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from verity_bench import jobs, overhead_records, steady
from verity_bench import lean_verify as L
from verity_bench import overhead as O
from verity_bench import prove as P
from verity_bench.boolean import boolean_version
from verity_bench.served_table import RUN_OF_RECORD, SUPERSEDED_BY, record_runs, records

CENSUS_KIND = "block-census/v1"
CENSUS_FILE = "census.json"
CARD_SCHEMA = "spot-card/v1"
PLAN_FILE = "mix.json"
#: the proof server's rate in a run's files: `benchmarks/one_stage/server_rate.py`'s rows, one a staged type and prover count
RATES = steady.RATES
#: the same rows in an art that holds them itself: its one run's, as completed or corrected after the run
ART_RATES = "rate.json"
#: the host a run timed its rows on, in the run's files: its `HOST_CORES_FIELD` is the CPU cores the run's provers held
#: for its card, which a row of its `RATES` that states no `cores` is read at
HOST, HOST_CORES_FIELD = "out/host.json", "cpus_used"
#: a proof server's report in the files of a run from before `RATES`, the staged types it proved beside it, and its K
#: (`server-kK`) the provers it ran; read only where a run holds no `RATES`
REPORTS, TYPES = "*/server-k*/server.json", "types.json"
#: the checks of the proof server's steady-state rules (`steady.CHECKS`) a row is held to here, each the rules' own
#: function of the row, in the rules' order: every check of theirs that reads the row alone, but `one_numa_node`
CHECKS = (steady.accepted, steady.exits, steady.not_void, steady.flat, steady.no_throttling, steady.every_prover_warmed)
#: what a row holds, beside what those checks read, for the command to price it, by the script's names (`stage.word`,
#: the census type it was staged for, names its block type, and `stage.definition`, the form staged, where it has no
#: `stage.word`; `verify_s` is there only where the verifier was timed, and `cores`, the CPU cores its provers held for
#: its card, only on a row that states them: `row_cores`)
PRICE_FIELDS = ("gpus", "stage.k_log", "stage.blocks", "stage.full")
#: what `prove.HOST_CORES` is, as the table says it of the default
HOST_CORES_ARE = ", node A's share: the cores its provider gives a card"
#: the Lean verifier's record in a run's files, as `verity bench prove lean-verify` writes it under the run's `$OUT`
LEAN_RECORD = "out/" + L.RECORD
#: what a `lean-verify` record holds for the command to read it
LEAN_FIELDS = ("block.k_log", "block.blocks", "block.full", "proof.bytes", "verifier.runs", "accepted", "label", "host.node",
               "host.cores", "host.cpu_model")
#: each fleet's assumed serving MFU
FLEETS = Path("verity/protocols/inference_only/rho/data/reference.toml")
SYSTEMS = ("flock", "spot")
#: the block types with no figure the rendered table names, the most blocks first; `--json` names every one
MISSING_NAMED = 8
CENSUS, CUT, STEADY, SINGLE, SAMPLING = "census", "cut", "flock-steady", "flock-single", "spot-sampling"
LEAN = "lean-verify"


@dataclass(frozen=True)
class Workload:
    """A workload the command prices: the model its records name, the card its prover and its serving run on
    (`overhead.CARDS`), and the fleet whose serving MFU prices a served token (`FLEETS`)."""

    name: str
    model: str
    card: str
    fleet: str

    def record(self, slot: str) -> str:
        return record_name(self.name, slot)


def record_name(workload: str, slot: str) -> str:
    """The `run_of_record` value that names a workload's record of a slot."""

    return f"prove-{workload}-{slot}"


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


def _field(row: Mapping[str, Any], name: str) -> Any:
    """A row's field by its dotted name, or `_ABSENT`."""

    at: Any = row
    for key in name.split("."):
        if not isinstance(at, Mapping) or key not in at:
            return _ABSENT
        at = at[key]
    return at


_ABSENT = object()


def _count(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x > 0


def judged(row: Mapping[str, Any]) -> steady.RowVerdict:
    """A `RATES` row's answers on `CHECKS`, as the proof server's steady-state rules give them: a pass, a fail, or
    "cannot tell" where the row's fields don't show it. Only a row that passed them all is a steady-state measurement
    here."""

    return steady.RowVerdict(row.get("type"), row.get("provers"), tuple(check(row) for check in CHECKS))


def unpriced(row: Mapping[str, Any]) -> str | None:
    """Why a row's rate can't be priced whatever `CHECKS` say of it, or None when it can: a field of `PRICE_FIELDS` it
    lacks, by name; a prover count, a statement's blocks or the GPUs its provers held that aren't one; or verifier's
    seconds or cores that aren't what a row states them as."""

    if lacks := [name for name in PRICE_FIELDS if _field(row, name) is _ABSENT]:
        return "its row lacks " + ", ".join(f"`{name}`" for name in lacks)
    stage, provers, gpus = row["stage"], row.get("provers"), row["gpus"]
    if not (_count(provers) and _count(stage["blocks"]) and _count(stage["full"]) and stage["blocks"] <= stage["full"]):
        return (f"its `provers` ({provers}), `stage.blocks` ({stage['blocks']}) and `stage.full` ({stage['full']}) "
                "aren't a prover count and a statement's blocks of a full one's")
    if not (isinstance(gpus, (int, float)) and gpus > 0):
        return f"no rate to price: `statements_per_s` is {row.get('statements_per_s')} on `gpus` {gpus}"
    if not isinstance(row.get("verify_s"), (int, float, type(None))):
        return "its `verify_s` isn't a statement's seconds"
    if row_cores(row) is None:
        return f"its `cores` ({row['cores']}) isn't a count of CPU cores, or a mapping whose `count` is one"
    return None


def row_why(row: Mapping[str, Any]) -> str | None:
    """Why a `RATES` row isn't a steady-state measurement the table prices, or None when it is one: each check of
    `CHECKS` it didn't pass, with its answer and its reason (`judged`), then why it can't be priced (`unpriced`)."""

    return "; ".join(why for why in (judged(row).why, unpriced(row)) if why) or None


def row_cores(row: Mapping[str, Any], unstated: int = P.UNSTATED_CORES) -> int | None:
    """The CPU cores a row's provers held for its card: its `cores`, a count, or a mapping whose `count` is one (a row
    completed after its run states the cores' list and NUMA nodes beside it); `unstated` on a row that states none;
    None where what it states is neither."""

    stated = row.get("cores", unstated)
    if isinstance(stated, Mapping):
        stated = stated.get("count")
    return stated if _count(stated) else None


def steady_row(row: Mapping[str, Any], run: str, art: str | None = None, unstated: int = P.UNSTATED_CORES) -> P.Steady | str:
    """A `RATES` row as a staged type's measurement at its prover count and its cores: a statement's GPU-seconds are
    the GPUs its provers held over its statements a second, and its cores the CPU cores they held for its card
    (`row_cores`; `unstated` on a row that states none). Its block type is the census type it was staged for
    (`stage.word`), and the Definition staged (`stage.definition`) its `form` where that is another; a row with no
    `stage.word` is its `stage.definition`'s. A row that names no staged Definition can't be put on a block
    type's row, and is its reason instead. `art`: the art that holds the row as its `ART_RATES`, where the run's files
    don't."""

    name = f"`{row.get('type')}` at {row.get('provers')} provers"
    definition = _field(row, "stage.definition")
    if not isinstance(definition, str):
        why = (f"failed: {row['error']}" if "error" in row else f"skipped: {row['skipped']}" if "skipped" in row else
               "it lacks `stage.definition`")
        return f"its row {name}: {why}"
    why = row_why(row)
    stage = row["stage"]
    staged = definition
    if isinstance(stage.get("word"), str):
        definition = P.definition_id(stage["word"])
    form = None if staged == definition else staged
    blocks, full, provers = (x if _count(x) else 0 for x in (stage.get("blocks"), stage.get("full"), row.get("provers")))
    k_log = stage["k_log"] if isinstance(stage.get("k_log"), int) else 0
    held = RATES if art is None else ART_RATES
    cores = row_cores(row, unstated) or unstated
    if why:
        return P.Steady(str(row.get("type")), definition, k_log, blocks, full, run, held, refused=why, provers=provers,
                        art=art, cores=cores, form=form)
    return P.Steady(str(row.get("type")), definition, k_log, blocks, full, run, held, row["gpus"] / row["statements_per_s"],
                    row.get("verify_s"), provers=provers, art=art, cores=cores, form=form)


def rates(store: Any, run: str) -> tuple[list[P.Steady], list[str]]:
    """Every row of a run's `RATES` that names a staged Definition, or, in a run that holds none, every staged type of
    its proof server reports; and why the rest (or the run) can't be read. A `RATES` row that states no `cores` is
    read at the run's own (`HOST`'s `HOST_CORES_FIELD`), where the run's files state them; any other row that states
    none, and every staged type of a report, at `prove.UNSTATED_CORES`."""

    attempt = store.get_attempt(run)
    if attempt is None:
        return [], ["not in the store or on its remote"]
    out = attempt.get("outputs") or {}
    files = out.get("run_files") or out.get("run_record")
    if not files:
        return [], ["kept no run files"]
    kept = {row["path"] for row in store.listing(files, [RATES, HOST])} & {RATES, HOST}
    if RATES in kept:
        tree = Path(store.fetch(files, paths=sorted(kept)))
        host = json.loads((tree / HOST).read_text()) if HOST in kept else None
        stated = host.get(HOST_CORES_FIELD) if isinstance(host, Mapping) else None
        return _rows(json.loads((tree / RATES).read_text()), run, unstated=stated if _count(stated) else P.UNSTATED_CORES)
    held = {row["path"] for row in store.listing(files, [REPORTS, REPORTS.rsplit("/", 1)[0] + "/" + TYPES])}
    paths = sorted(p for p in held if p.endswith("/server.json") and p.rsplit("/", 1)[0] + "/" + TYPES in held)
    if not paths:
        return [], [f"holds no proof server rate (`{RATES}`, which `benchmarks/one_stage/server_rate.py` writes) and no "
                    f"proof server report (`{REPORTS}` with its `{TYPES}`)"]
    tree = Path(store.fetch(files, paths=[*paths, *(p.rsplit("/", 1)[0] + "/" + TYPES for p in paths)]))
    found = []
    for path in paths:
        where = path.rsplit("/", 1)[0]
        report, types = json.loads((tree / path).read_text()), json.loads((tree / where / TYPES).read_text())
        why, provers = report_why(report), int(where.rsplit("/server-k", 1)[1])
        for name, staged in report["types"].items():
            gpu, verify = ((report.get(k) or {}).get(name) for k in ("gpu_s_by_type", "verify_s"))
            found.append(P.Steady(
                name, types[name]["definition"], staged["k_log"], staged["c"], staged["full"], run, where,
                gpu and gpu["mean"], verify and verify["mean"],
                why or (None if gpu else "no statement of it finished in the steady span"), provers=provers))
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


def _rows(doc: Any, run: str, art: str | None = None, unstated: int = P.UNSTATED_CORES) -> tuple[list[P.Steady], list[str]]:
    """A rate file's rows as `run`'s: those that name a staged Definition, and why the rest (or the file) can't be read.
    A row that states no `cores` is read at `unstated`."""

    if not isinstance(doc, list) or not all(isinstance(row, Mapping) for row in doc):
        return [], [f"its `{RATES if art is None else ART_RATES}` isn't a list of rows"]
    read = [steady_row(row, run, art, unstated) for row in doc]
    return [x for x in read if isinstance(x, P.Steady)], [x for x in read if isinstance(x, str)]


def steady_rows(store: Any, target: str, superseded: bool = False) -> dict[str, tuple[list[P.Steady], list[str]]]:
    """What a target of the steady-state record gives, by the run or art read: its rows that name a staged Definition,
    and why the rest (or the target) can't be read. A run gives its own `RATES`. An art that holds `ART_RATES` gives
    those rows in place of its run's own: they are that run's, completed or corrected after it, so its meta names the
    one run in `runs`, which the rows are ordered by and which isn't read. Any other art stands for the runs its meta
    names. A superseded target gives nothing, unless `superseded` asks for its rows anyway (for their staged names)."""

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
        return {target: ([], [f"holds `{ART_RATES}` and names {len(runs)} runs ({', '.join(runs)}): an art's rows are "
                              "one run's, so each run's go in an art of their own"])}
    return {target: _rows(_read(store, target, ART_RATES), runs[0], target)}


def flock(store: Any, workload: Workload, census: P.Census, provers: int = 1,
          cores: int = P.HOST_CORES) -> tuple[P.Flock, list[dict], list[dict]]:
    """Flock's prover at `provers` provers on a host of `cores` CPU cores a card from the workload's records: the
    steady-state runs' and arts' rows (`steady_rows`), the single-session card's; the inputs, and the steady-state
    runs, arts and rows left out with why. The steady-state input counts what the rows measure of `census`: its block
    types, and the rows that are none."""

    art = of_record(store, workload.record(SINGLE), "single-session card")
    value = card(store, art, "proof_overhead", workload)
    sessions = tuple(P.Session(b["block"], b["k_log"], b["seconds_per_block"], b["fixed_seconds"], b["statement_blocks"])
                     for b in value["blocks"])
    steady, runs, refused = [], [], []
    read: dict[str, tuple[list[P.Steady], list[str]]] = {}
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
    prover = P.Flock(tuple(steady), sessions, art, provers=provers, cores=cores, aliases=tuple(aliases),
                     boolean=boolean_version)
    table = prover.table(census)
    measured = sum(r.cost.source == P.STEADY for r in table.rows)
    count = f"{provers} prover{'' if provers == 1 else 's'}"
    unread = f"; {len(table.unread)} row(s) of no block type of the census, listed under Unread" if table.unread else ""
    return prover, [
        _input(STEADY, workload, runs, f"{measured} staged types measured at steady state at {count}{unread}" if runs else
               "none of record: no block type is measured at steady state"),
        _input(SINGLE, workload, [art], f"{len(sessions)} block types on the {value.get('device')}")], refused


def lean_verify(doc: Any, run: str) -> P.LeanVerify | str:
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
    host = doc["host"]
    return P.LeanVerify(P.definition_id(word) if isinstance(word, str) else definition, block["k_log"], block["blocks"],
                        block["full"], tuple(seconds), doc["proof"]["bytes"], run,
                        f"{host['node']} ({host['cores']} cores, {host['cpu_model']})", doc["label"])


def lean_run(store: Any, run: str) -> P.LeanVerify | str:
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


def lean(store: Any, workload: Workload, census: P.Census) -> tuple[list[P.LeanVerify], list[dict], list[dict]]:
    """The statements of `census`'s block types the Lean verifier accepted, from the workload's `lean-verify` runs of
    record; their input, where a run carries the label; and the runs that measure nothing, with why."""

    held = {P.definition_id(b.name) for b in census.blocks}
    found, refused = [], []
    for run in sorted(records(store).get(workload.record(LEAN), []), reverse=True):
        got = lean_run(store, run)
        if isinstance(got, P.LeanVerify) and got.definition not in held:
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
                          "every other type's Lean verifier seconds and proof bytes are scaled from them" if found else
                          "none that measures a block type: the Lean verifier's seconds and the proof's bytes are not "
                          "measured")], refused


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


def build(system: str, workload: str, *, sampling: str | None = None, provers: int = 1, cores: int = P.HOST_CORES,
          store: Any = None, root: Path | None = None, local_only: bool = False) -> dict:
    """A proof system's table on a workload from the store's records, each type priced at `provers` provers on a host
    of `cores` CPU cores a card: `prove.report`'s rows, sources and totals, with the inputs read, the runs and rows
    refused, the stated serving rate and the totals a monitoring period over all compute."""

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
    prover = replace(prover, lean=tuple(timed))
    rate = serving(root, w)
    seconds = rate.gpu_seconds_per_token(blocks)
    inputs += more + its
    refused += unread
    sampled = None
    if system == "spot":
        sampled, its = plan(store, w, sampling)
        inputs.append(its)
        prover = P.Spot(prover, sampled, rate)
    doc = P.report(prover.table(blocks), seconds)
    return {"schema": P.SCHEMA, "workload": w.name, "model": w.model, "cut": blocks.cut, "provers": provers, "cores": cores,
            "macs_in_blocks": blocks.macs_in_blocks, "inputs": inputs,
            "refused": refused, "serving": {"gpu_seconds_per_token": seconds, "status": "modeled", "basis": rate.basis},
            **doc, "periods": _periods(root, rate, doc["totals"]["gpus_per_serving_gpu"], sampled)}


def _periods(root: Path, rate: P.Serving, ratio: float, sampled: P.SamplingPlan | None) -> dict:
    """The totals a monitoring period over all compute. With every block proved the prover GPUs a serving GPU hold for
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


def _missing_share(m: Mapping[str, Any]) -> str:
    """The share of a block type with no figure: of the blocks, and of the MACs where the census holds them."""

    blocks = f"{m['block_share']:.2%} of the blocks"
    return blocks if m["mac_share"] is None else f"{blocks}, {m['mac_share']:.2%} of the blocks' MACs"


def _above_line(above: Mapping[str, Any], partial: bool) -> str:
    """How much of the proving time rests on steady-state rows timed at more cores than the host holds: its share, and
    where it isn't 0 the block types, the cores their rows were timed at and how many each source holds."""

    head = (f"- steady-state rows timed above the host's {above['cores']} cores: {above['time_share'] or 0.0:.2%} of "
            f"{'the proving time it prices' if partial else 'its proving time'} rests on them")
    if not above["block_types"]:
        return head
    sources = "; ".join(f"{x['block_types']} {P.SOURCES[x['source']]}, {x['time_share'] or 0.0:.2%}" for x in above["sources"])
    return (f"{head}: {above['block_types']} block types no row at {above['cores']} cores measures, priced from rows timed "
            f"at {', '.join(map(str, above['timed_at']))} cores ({sources})")


def _planned_lines(planned: Mapping[str, Any]) -> list[str]:
    """The same blocks at the sampling plan's own prices, and where they leave the table's: each source's GPU-seconds
    at the table's prices over the plan's, and the block type that holds most of the difference."""

    lines = [f"- the same blocks at the sampling plan's own prices: {_n(planned['gpus_per_serving_gpu'])} prover GPUs per "
             "serving GPU. Its planner sized the sample at them; they are the plan's record's, no source this table "
             "reads, and hold whatever its planner counts in a block's price. Every other figure here prices a block "
             "from this table's sources"]
    ratios = ", ".join(
        f"{P.SOURCES[x['source']]} {x['gpu_seconds_per_token'] / x['planned_gpu_seconds_per_token']:.3g}x "
        f"({x['block_types']} block types)" for x in planned["sources"] if x["planned_gpu_seconds_per_token"])
    if top := planned["largest"]:
        lines.append(f"- where the two differ, this table's proving time over the plan's by this table's source: {ratios}; "
                     f"`{_short(top['block'])}` alone is {top['difference_share']:.1%} of the difference, "
                     f"{_n(1e3 * top['seconds'])} ms per block here and {_n(1e3 * top['planned_seconds'])} ms in the plan")
    return lines


def _lean_lines(doc: Mapping[str, Any]) -> list[str]:
    """The Lean verifier's seconds and the proof's bytes a served token, a line each: `not measured` with no statement
    measured, else the figure with its status, what was measured (the block type with its share of the proving time,
    its statement, the verifier's seconds on it or its proof's bytes, the run and the host) and what was scaled (the
    block types, their share of the figure, and the basis). The seconds' line says what the accept holds in the
    record's own words (its `label`)."""

    t = doc["totals"]
    seconds, size, measured = t["lean_verifier_seconds_per_token"], t["proof_bytes_per_token"], t["lean_verify"]
    if not measured:
        return ["- Lean verifier seconds per token: not measured (no Lean verifier run of record measures a block type: "
                f"label one {_label(record_name(doc['workload'], LEAN))})", "- proof bytes per token: not measured"]
    m = measured[0]
    of = "the proving time at the sampling plan" if doc["system"].startswith("spot") else "the proving time"
    share = "none" if m["time_share"] is None else f"{m['time_share']:.1%}"
    fill = "a full statement" if m["blocks"] == m["full"] else f"{m['blocks']}/{m['full']} of a full statement"
    block = (f"`{m['block']}` ({share} of {of}), in its statement of {m['blocks']} blocks of 2^{m['k_log']} "
             f"witness bits each ({fill})")
    runs = "one run" if len(m["seconds"]) == 1 else (f"the median of {len(m['seconds'])} runs: "
                                                     + ", ".join(_n(x) for x in m["seconds"]) + " s")
    more = "" if len(measured) == 1 else "; also measured: " + ", ".join(
        f"`{x['block']}` ({_n(x['statement_seconds'])} s and {_n(x['proof_bytes'])} bytes a statement, run "
        f"{x['run']})" for x in measured[1:])
    one = "one block type" if len(measured) == 1 else f"{len(measured)} block types, the most proving time first"

    def line(name: str, v: Mapping[str, Any], what: str, unit: str) -> str:
        status = "modeled, not measured" if v["status"] == "modeled" else v["status"]
        rate = "the measured statement's" if len(measured) == 1 else "the measured statements' median"
        scaled = (f" Scaled: the other {v['scaled_block_types']} block types, {1 - v['measured_share']:.1%} of this figure, "
                  f"each at {rate} {unit} a witness bit times its 2^k_log bits: {v['basis']}." if v["scaled_block_types"] else "")
        bare = (f" In neither: {v['unpriced_block_types']} block type(s) with no k_log." if v["unpriced_block_types"] else "")
        return f"- {name} per token: {_n(v['value'])}, {status}. Measured: {one}, {what}{more}.{scaled}{bare}"

    return [line("Lean verifier seconds", seconds,
                 f"{block}: {m['label']}, by the Lean verifier's `verify` in {_n(m['statement_seconds'])} s ({runs}; a run is "
                 f"one process, the statement read and set up in it), in run {m['run']} on {m['host']}", "seconds"),
            line("proof bytes", size,
                 f"{block}, whose proof files are {_n(m['proof_bytes'])} bytes in all (run {m['run']})", "bytes")]


def _record(target: str | None) -> str:
    return "—" if target is None else target[:16] if target.startswith("art:") else target


def render(doc: Mapping[str, Any], rows: int | None = None) -> str:
    """The table as Markdown; `rows`: only the block types that take the most proving time, the rest in one row."""

    t, spot = doc["totals"], doc["system"].startswith("spot")
    planned = t.get("planned")
    lines = [f"# verity bench prove: {doc['system']} on {doc['workload']} ({doc['model']})", "",
             f"Inputs, each a labelled record of the store (`{RUN_OF_RECORD}`):", ""]
    lines += [f"- {i['slot']}: {', '.join(i['targets']) or 'none'} (`{i['record']}`): {i['what']}" for i in doc["inputs"]]
    lines.append(f"- prover count (`--provers`, a stated input): {doc['provers']}; a type is measured at steady state only "
                 "by rows at that count, and each such row names it")
    lines.append(f"- host cores (`--cores`, a stated input): {doc['cores']} CPU cores a card"
                 f"{HOST_CORES_ARE if doc['cores'] == P.HOST_CORES else ''}; a type is priced from the rows whose provers "
                 "held that many, else from those that held the fewest above it, which its row names; a row that held "
                 f"fewer prices nothing, and one whose record states no cores is read as {P.UNSTATED_CORES}")
    s = doc["serving"]
    lines += [f"- serving rate ({s['status']}, a stated input, not a record): {_n(s['gpu_seconds_per_token'])} GPU-seconds a "
              f"served token, {_n(1 / s['gpu_seconds_per_token'])} tokens a serving GPU-second: {s['basis']}", "",
              f"| block type | k_log | blocks {'proved ' if spot else ''}per token | ms per block | "
              f"{'ms per block in the plan | ' if planned else ''}source | record | share of proving time | note |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |" + (" --- |" if planned else "")]
    shown = doc["rows"] if rows is None else doc["rows"][:rows]
    for r in shown:
        ms = "—" if r["seconds"] is None else _n(1e3 * r["seconds"])
        share = "—" if r["time_share"] is None else f"{r['time_share']:.2%}"
        its = f"{_n(1e3 * r['planned_seconds'])} | " if planned else ""
        lines.append(f"| `{_short(r['block'])}` | {'—' if r['k_log'] is None else r['k_log']} | {_n(r['per_token'])} | {ms} | "
                     f"{its}{P.SOURCES[r['source']]} | {_record(r['record'])} | {share} | {r['note'] or ''} |")
    if rest := doc["rows"][len(shown):]:
        lines.append(f"| {len(rest)} more block types | | {_n(sum(r['per_token'] for r in rest))} | | "
                     f"{'| ' if planned else ''}| | {sum(r['time_share'] or 0.0 for r in rest):.2%} | |")
    lines += ["", "## At the sampling plan" if spot else "## Every block proved", "",
              f"- blocks {'proved ' if spot else ''}per token: {_n(t['blocks_per_token'])}",
              f"- prover GPU-seconds per token: {_n(t['gpu_seconds_per_token'])}{_range(t['gpu_seconds_per_token_range'])}",
              f"- {'served tokens covered' if spot else 'tokens proved'} per second per prover GPU: "
              f"{_n(t['tokens_per_gpu_second'])}{_range(t['tokens_per_gpu_second_range'])}",
              f"- prover GPUs per serving GPU: {_n(t['gpus_per_serving_gpu'])}{_range(t['gpus_per_serving_gpu_range'])}, "
              f"{t['status']}: steady-state measurements cover {t['steady_time_share']:.1%} of "
              f"{'the proving time it prices' if t['missing'] else 'its proving time'}"]
    if t["missing"]:
        named = "; ".join(f"`{_short(m['block'])}` ({_missing_share(m)})" for m in t["missing"][:MISSING_NAMED])
        more = len(t["missing"]) - MISSING_NAMED
        lines.append(f"- partial: {len(t['missing'])} block type(s) with no figure, whose proving every total here leaves "
                     f"out: {named}" + (f"; and {more} more, which `--json` names" if more > 0 else ""))
    if above := t["above_cores"]:
        lines.append(_above_line(above, bool(t["missing"])))
    if planned:
        lines += _planned_lines(planned)
    if t["range_basis"]:
        lines.append(f"- each range's other end: {t['range_basis']}")
    lines.append(f"- served MACs inside a proof block: {doc['macs_in_blocks']:.1%}; the rest are served rows no cut fits, "
                 "which no figure here prices")
    if (v := t["verifier_seconds_per_token"]) is None:
        lines.append("- verifier seconds per token: not measured")
    else:
        basis = f" ({t['verifier_basis']})" if t["verifier_basis"] else ""
        lines.append(f"- verifier seconds per token: {_n(v['value'])} on the {v['block_types']} block types a record holds "
                     f"them for, {v['block_share']:.1%} of the blocks{basis}; the rest: not measured")
    lines += _lean_lines(doc)
    lines += ["", "Share of the proving time each source covers:", "",
              "| source | block types | blocks per token | share of blocks | prover GPU-seconds per token | "
              "share of proving time |", "| --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {P.SOURCES[x['source']]} | {x['block_types']} | {_n(x['per_token'])} | "
              f"{'—' if x['block_share'] is None else format(x['block_share'], '.2%')} | {_n(x['gpu_seconds_per_token'])} | "
              f"{'—' if x['time_share'] is None else format(x['time_share'], '.2%')} |" for x in doc["sources"]]
    p = doc["periods"]
    lines += ["", f"Per monitoring period, over all compute ({p['gates_basis']}); every total is the table's ratio scaled, "
              f"not measured at that size{', and partial as that ratio is' if t['missing'] else ''}:", "",
              "| compute | period | serving GPUs | served AND gates | prover GPUs | prover GPU-seconds |",
              "| --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {x['compute']} | {x['period']} | {_n(x['serving_gpus'])} | {_n(x['served_gates'])} | {_n(x['prover_gpus'])} | "
              f"{_n(x['prover_gpu_seconds'])} |" for x in p["rows"]]
    if p["only"]:
        lines += ["", p["only"][0].upper() + p["only"][1:] + "."]
    if doc["refused"]:
        lines += ["", "Refused (not read):", ""]
        lines += [f"- {x['target']} (`{x['record']}`): {x['why']}" for x in doc["refused"]]
    if doc["unread"]:
        lines += ["", "Unread (labelled steady-state rows whose staged Definition is no block type of this census under "
                  "its cut; they are in no figure above):", ""]
        lines += [f"- {_record(x['record'])}{'' if x['record'] == x['run'] else ', run ' + x['run']}: `{x['type']}` at "
                  f"{x['provers']} prover{'' if x['provers'] == 1 else 's'}, `{x['definition']}`: "
                  + (f"not a steady-state measurement: {x['why']}" if x["seconds"] is None else
                     f"measured at steady state, {_n(1e3 * x['seconds'])} ms per block") for x in doc["unread"]]
    return "\n".join(lines) + "\n"
