"""A steady-state row of `verity bench prove`'s store side (`prove_records`): the checks it is judged by, what it
lacks to be priced, its cores, its run's own verdict and whether its card was on its CPUs' NUMA node."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, TypeGuard

from verity_bench import prove as P
from verity_bench import prove_flock as PF
from verity_bench import steady, steady_node
from verity_bench.prove_common import (
    _ABSENT,
    ART_RATES,
    CHECKS,
    OWN_VERDICT,
    RATES,
    STEADY,
    WORKLOADS,
    _count,
    _field,
)
from verity_bench.steady import CARD_NUMA

#: the tool a run of the harness's steady-state point is recorded under (`verity_bench.tool.STEADY_POINT`'s name): its
#: attempt carries the point's verdict on all ten checks as its result's validation (`own_verdict`)
STEADY_POINT_TOOL = "steady_point"
#: a scan of the runs' own host records, for rows that don't record `CARD_NUMA`: the record whose runs it read
#: (`record`), the runs it read (`read`), those of them whose card was on another NUMA node than their CPUs (`off`,
#: each with both nodes), and in `source` what was read (`OffNodeScan`)
OFF_NODE = Path(__file__).resolve().parent / "data" / "off_node.json"
#: what a row holds, beside what those checks read, for the command to price it, by the script's names (`stage.word`,
#: the census type it was staged for, names its block type, and `stage.definition`, the form staged, where it has no
#: `stage.word`; `verify_s` is there only where the verifier was timed, and `cores`, the CPU cores its provers held for
#: its card, only on a row that states them: `row_cores`)
PRICE_FIELDS = ("gpus", "stage.k_log", "stage.blocks", "stage.full")


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


def _node(x: Any) -> TypeGuard[int]:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


_RUN_ID = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}")


@dataclass(frozen=True)
class OffNodeScan:
    """A scan of runs' own host records (`OFF_NODE`): the record whose runs it read, the runs it read (`read`), and
    those of them whose card was on another NUMA node than their CPUs (`off`)."""

    record: str
    read: frozenset[str]
    off: frozenset[str]

    def of(self, run: str) -> bool | None:
        """Whether `run`'s card was off its CPUs' node by the scan, None where the scan didn't read `run`."""

        return (run in self.off) if run in self.read else None


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if dup := sorted({k for k, _ in pairs if sum(k == j for j, _ in pairs) > 1}):
        raise ValueError(f"lists {', '.join(dup)} more than once")
    return dict(pairs)


@cache
def off_node_scan(path: Path = OFF_NODE) -> OffNodeScan:
    """`OFF_NODE`'s scan, refused unless it names a workload's steady-state record, lists each run it read once and
    by its id, and lists each off-node run once, among those it read, with a card's node and CPUs' nodes that aren't
    that one node alone."""

    try:
        doc = json.loads(path.read_text(), object_pairs_hook=_no_duplicate_keys)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e
    record, read, off = doc["record"], doc["read"], doc["off"]
    if record not in {w.record(STEADY) for w in WORKLOADS.values()}:
        raise ValueError(f"{path}: `record` ({record!r}) is no workload's steady-state record")
    if bad := sorted(r for r in read if not (isinstance(r, str) and _RUN_ID.fullmatch(r))):
        raise ValueError(f"{path}: `read` lists what isn't a run id: {', '.join(map(str, bad))}")
    if len(set(read)) != len(read):
        raise ValueError(f"{path}: `read` lists {', '.join(sorted({r for r in read if read.count(r) > 1}))} more than once")
    for run, x in off.items():
        if run not in read:
            raise ValueError(f"{path}: off-node run {run} isn't among the runs the scan read")
        card, cpus = x.get("card_numa_node"), x.get("cpu_numa_nodes")
        if not (_node(card) and isinstance(cpus, list) and cpus and all(_node(n) for n in cpus) and cpus != [card]):
            raise ValueError(f"{path}: run {run} lists no card's NUMA node and CPUs' NUMA nodes other than it alone")
    return OffNodeScan(record, frozenset(read), frozenset(off))


def off_node(row: Mapping[str, Any], run: str, scan: OffNodeScan | None = None) -> bool | None:
    """Whether a row's card was on another NUMA node than its CPUs, None where that isn't recorded. A row that records
    its card's node (`CARD_NUMA`) and its CPUs' (`cores.numa_nodes`) is off when they aren't that one node and on
    otherwise. A row that doesn't (or records null) is as its run is by the scan (`off_node_scan`): off where it lists
    the run off-node, on where it read the run and doesn't, and not recorded where it didn't read the run, whatever
    the run's id."""

    card, cpus = row.get(CARD_NUMA), _field(row, "cores.numa_nodes")
    if _node(card) and isinstance(cpus, list) and cpus and all(_node(n) for n in cpus):
        return cpus != [card]
    return (off_node_scan() if scan is None else scan).of(run)


def steady_row(row: Mapping[str, Any], run: str, art: str | None = None, unstated: int = P.UNSTATED_CORES,
               refused: str | None = None) -> PF.Steady | str:
    """A `RATES` row as a staged type's measurement at its prover count and its cores: a statement's GPU-seconds are
    the GPUs its provers held over its statements a second, and its cores the CPU cores they held for its card
    (`row_cores`; `unstated` on a row that states none). Its block type is the census type it was staged for
    (`stage.word`), and the Definition staged (`stage.definition`) its `form` where that is another; a row with no
    `stage.word` is its `stage.definition`'s. A row that names no staged Definition can't be put on a block
    type's row, and is its reason instead. `art`: the art that holds the row as its `ART_RATES`, where the run's files
    don't. `refused`: why the row's run prices none of its rows (`own_verdict`), which the row is refused with in
    place of `CHECKS`' answers. Whether its card was off its CPUs' NUMA node is `off_node`'s."""

    name = f"`{row.get('type')}` at {row.get('provers')} provers"
    definition = _field(row, "stage.definition")
    if not isinstance(definition, str):
        why: str | None = (f"failed: {row['error']}" if "error" in row else f"skipped: {row['skipped']}" if "skipped" in row else
               "it lacks `stage.definition`")
        return f"its row {name}: {why}"
    why = refused or row_why(row)
    stage = row["stage"]
    staged = definition
    if isinstance(stage.get("word"), str):
        definition = P.definition_id(stage["word"])
    form = None if staged == definition else staged
    blocks, full, provers = (x if _count(x) else 0 for x in (stage.get("blocks"), stage.get("full"), row.get("provers")))
    k_log = stage["k_log"] if isinstance(stage.get("k_log"), int) else 0
    held = RATES if art is None else ART_RATES
    cores = row_cores(row, unstated) or unstated
    off = off_node(row, run)
    if why:
        return PF.Steady(str(row.get("type")), definition, k_log, blocks, full, run, held, refused=why, provers=provers,
                        art=art, cores=cores, form=form, off_node=off)
    return PF.Steady(str(row.get("type")), definition, k_log, blocks, full, run, held, row["gpus"] / row["statements_per_s"],
                    row.get("verify_s"), provers=provers, art=art, cores=cores, form=form, off_node=off)


def own_verdict(store: Any, run: str, attempt: Mapping[str, Any] | None) -> str | None:
    """Why a run prices none of its rows by its own verdict, or None where it has none against them. A run of the
    harness's steady-state point (its attempt's tool `STEADY_POINT_TOOL`) judged its row by all ten of the rules'
    checks as it ended, and wrote that verdict as its result's validation, which its attempt carries: `passed` leaves
    its rows to `CHECKS`; `failed` refuses them, naming the run and the checks its verdict file
    (`steady_node.VERDICT_FILE`, fetched only for such a run) says it didn't pass; and no validation at all can't be
    told, and refuses them too, as a row whose fields can't show a check prices nothing. Any other run, one timed
    outside the harness's point or before it, has no verdict of its own: None, and `CHECKS` alone judge its rows."""

    if ((attempt or {}).get("tool") or {}).get("name") != STEADY_POINT_TOOL:
        return None
    status = ((attempt or {}).get("status") or {}).get("validation")
    if status == "passed":
        return None
    if status != "failed":
        return (f"{OWN_VERDICT} can't be told: run {run} is a steady-state point's, and its attempt carries no verdict "
                f"(its result's validation is {status!r})")
    return f"{OWN_VERDICT} failed: run {run} {_failed_checks(store, attempt or {})}"


def _failed_checks(store: Any, attempt: Mapping[str, Any]) -> str:
    """The checks a failed steady-state point's verdict file says its row didn't pass, each with its answer, or what its
    verdict says instead where it names none."""

    out = attempt.get("outputs") or {}
    files, path = out.get("run_files") or out.get("run_record"), steady_node.VERDICT_FILE
    if not files or not store.listing(files, [path]):
        return f"keeps no `{path}` naming the checks it didn't pass"
    try:
        doc = json.loads((Path(store.fetch(files, paths=[path])) / path).read_text())
    except json.JSONDecodeError as e:
        return f"keeps a `{path}` that isn't JSON ({e.msg})"
    checks = ((doc.get("verdict") or {}).get("checks") or {}) if isinstance(doc, Mapping) else {}
    named = [f"`{name}` ({c.get('status')})" for name, c in checks.items()
             if isinstance(c, Mapping) and c.get("status") != steady.PASS]
    if named:
        return "didn't pass " + ", ".join(named)
    why = doc.get("why") if isinstance(doc, Mapping) else None
    return f"wasn't a passed point: {why}" if why else f"keeps a `{path}` that names no check it didn't pass"
