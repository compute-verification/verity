"""`verity bench prove`'s table as text (`prove_records`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from verity_bench import prove as P
from verity_bench.prove_common import (
    LEAN,
    SAMPLING,
    _label,
    record_name,
)
from verity_bench.served_table import RUN_OF_RECORD

#: what `prove.HOST_CORES` is, as the table says it of the default
HOST_CORES_ARE = ", node A's share: the cores its provider gives a card"
#: the block types with no figure the rendered table names, the most blocks first; `--json` names every one
MISSING_NAMED = 8


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


def off_node_line(off: Mapping[str, Any]) -> str:
    """How many of the steady-state rows that price a type ran with their card off their CPUs' NUMA node, how many on
    it and how many don't record it, each with its share of the proving time, then the rows priced from the
    single-session card, whose table records no CPUs (`prove.report`'s `off_node`)."""

    def share(kind: Mapping[str, Any]) -> str:
        return f"{kind['time_share'] or 0.0:.4%}"

    return (f"card off its CPUs' NUMA node: {off['off']['rows']} of {off['rows']} priced steady rows, "
            f"{share(off['off'])} of the proving time; on its node by the runs' own records: {off['on']['rows']} rows, "
            f"{share(off['on'])}; not recorded: {off['not_recorded']['rows']} rows, {share(off['not_recorded'])}; "
            f"priced from the single-session card, whose table records no CPUs: {off['single']['rows']} rows, "
            f"{share(off['single'])}")


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

    lines = [(f"- the same blocks at the sampling plan's own prices: {_n(planned['gpus_per_serving_gpu'])} prover GPUs per "
              "serving GPU. Its planner sized the sample at them; they are the plan's record's, no source this table "
              "reads, and hold whatever its planner counts in a block's price. Every other figure here prices a block "
              "from this table's sources")]
    ratios = ", ".join(
        f"{P.SOURCES[x['source']]} {x['gpu_seconds_per_token'] / x['planned_gpu_seconds_per_token']:.3g}x "
        f"({x['block_types']} block types)" for x in planned["sources"] if x["planned_gpu_seconds_per_token"])
    if top := planned["largest"]:
        lines.append(f"- where the two differ, this table's proving time over the plan's by this table's source: {ratios}; "
                     f"`{_short(top['block'])}` alone is {top['difference_share']:.1%} of the difference, "
                     f"{_n(1e3 * top['seconds'])} ms per block here and {_n(1e3 * top['planned_seconds'])} ms in the plan")
    return lines


def _design(r: Mapping[str, Any]) -> str:
    """How the figure at the exfiltration run of record's design is read from that run's rows (`at_record`)."""

    return (f"run {r['run']} (`{RUN_OF_RECORD}={r['record']}`), the planner's {r['scope']} at {r['share']:.0%} of total "
            f"compute: its proving's {r['prove_share']:.3%} of total compute over 1 - its audit budget, "
            f"{r['beta']:.3%}")


def _record_lines(doc: Mapping[str, Any]) -> list[str]:
    """The prover GPUs a serving GPU at the sampling plan of record and at the exfiltration run of record's design,
    side by side, each naming its source: the plan's art, and the run whose rows it is read from (`at_record`)."""

    t, r = doc["totals"], doc["at_record"]
    plan = next(x for i in doc["inputs"] if i["slot"] == SAMPLING for x in i["targets"])
    head = "- prover GPUs per serving GPU at"
    lines = [f"{head} the sampling plan of record ({plan}): {t['gpus_per_serving_gpu']:.4g}, {t['status']}"]
    if r["gpus_per_serving_gpu"] is None:
        return [*lines, (f"{head} the exfiltration run of record's design (`{RUN_OF_RECORD}={r['record']}`): not read: "
                         f"{r['why']}")]
    return [*lines, (f"{head} the exfiltration run of record's design: {r['gpus_per_serving_gpu']:.4g}, {r['status']}: "
                     f"{_design(r)}")]


def _lean_lines(doc: Mapping[str, Any]) -> list[str]:
    """The Lean verifier's seconds and the proof's bytes a served token, a line each: `not measured` with no statement
    measured, else the figure with its status, what was measured (the block type with its share of the proving time,
    its statement, the verifier's seconds on it or its proof's bytes, the run and the host) and, where it is modeled,
    how much of it each source prices (own, family, fitted: the block types, their share of the figure, and each
    one's rate) with the basis. The seconds' line says what the accept holds in the record's own words (its
    `label`)."""

    t = doc["totals"]
    seconds, size, measured = t["lean_verifier_seconds_per_token"], t["proof_bytes_per_token"], t["lean_verify"]
    if not measured:
        return [("- Lean verifier seconds per token: not measured (no Lean verifier run of record measures a block type: "
                 f"label one {_label(record_name(doc['workload'], LEAN))})"), "- proof bytes per token: not measured"]
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

    def line(name: str, v: Mapping[str, Any], what: str, unit: str, per: str, size: str) -> str:
        status = "modeled, not measured" if v["status"] == "modeled" else v["status"]
        how = {P.LEAN_MEASURED: "its own statement's",
               P.LEAN_FAMILY: f"its Definition's measured statements' {unit} {per} (the dearest where several are) "
                              f"times its {size}",
               P.LEAN_FITTED: f"{'—' if v['fitted_rate'] is None else _n(v['fitted_rate'])} {unit} {per} (every "
                              f"measured statement's {unit} over their {size}) times its {size}"}
        parts = [f"{P.LEAN_SOURCES[k]} {x['block_types']} block type(s), {x['share'] or 0.0:.1%}"
                 + ("" if k == P.LEAN_MEASURED else f", each at {how[k]}") for k, x in v["sources"].items() if x["block_types"]]
        shares = f" Of this figure: {'; '.join(parts)}: {v['basis']}." if v["status"] == "modeled" else ""
        bare = (f" In none: {v['unpriced_block_types']} block type(s) with no {size}." if v["unpriced_block_types"] else "")
        return f"- {name} per token: {_n(v['value'])}, {status}. Measured: {one}, {what}{more}.{shares}{bare}"

    return [line("Lean verifier seconds", seconds,
                 f"{block}: {m['label']}, by the Lean verifier's `verify` in {_n(m['statement_seconds'])} s ({runs}; a run is "
                 f"one process, the statement read and set up in it), in run {m['run']} on {m['host']}", "seconds",
                 "an AND gate", "AND gates"),
            line("proof bytes", size,
                 f"{block}, whose proof files are {_n(m['proof_bytes'])} bytes in all (run {m['run']})", "bytes",
                 "a witness bit", "2^k_log witness bits")]


#: what each mark of a block type's Lean verifier row rests on (`P.LEAN_MARKS`), as its table's heading words it
LEAN_ROWS = ("The Lean verifier's seconds, a row a block type (measured: the median of its own statement's runs over its "
             "blocks; modeled: its Definition's measured statement's seconds an AND gate times its own AND gates; "
             "assumed: the rate fitted over every measured statement times its AND gates, which no measurement of its "
             "Definition backs; none: no AND gates to price it by)")


def _lean_rows(doc: Mapping[str, Any], rows: int | None) -> list[str]:
    """The Lean verifier's table: each block type's seconds a block, its mark, the run that measured the statement it
    rests on and that statement, in the main table's order; with `rows`, the rest in one row counting their marks."""

    shown = doc["rows"] if rows is None else doc["rows"][:rows]
    lines = ["", LEAN_ROWS + ":", "", "| block type | Lean verifier ms per block | mark | record | rests on |",
             "| --- | --- | --- | --- | --- |"]
    for r in shown:
        ms = "—" if r["lean_verifier_seconds"] is None else _n(1e3 * r["lean_verifier_seconds"])
        lines.append(f"| `{_short(r['block'])}` | {ms} | {r['lean_mark'] or 'none'} | {r['lean_record'] or '—'} | "
                     f"{_lean_basis(r)} |")
    if rest := doc["rows"][len(shown):]:
        marks = [r["lean_mark"] or "none" for r in rest]
        counts = ", ".join(f"{mark} {marks.count(mark)}" for mark in (*P.LEAN_MARKS.values(), "none") if mark in marks)
        lines.append(f"| {len(rest)} more block types | | {counts} | | |")
    return lines


def _lean_basis(r: Mapping[str, Any]) -> str:
    if r["lean_record"] is None:
        return "the fitted rate" if r["lean_mark"] else "—"
    blocks, full = r["lean_statement_blocks"]
    whose = "its own statement" if r["lean_mark"] == "measured" else f"`{_short(r['lean_definition'])}`'s statement"
    size = "a full statement" if blocks == full else f"{blocks}/{full} of a full statement"
    form = r["lean_form"]
    read = ("its record names no statement id" if form is None else
            f"measured from {P.LEAN_FORMS.get(form, f'statement `{form}`')}")
    return f"{whose} of {blocks} blocks ({size}), {read}"


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
    lines += [(f"- serving rate ({s['status']}, a stated input, not a record): {_n(s['gpu_seconds_per_token'])} GPU-seconds a "
               f"served token, {_n(1 / s['gpu_seconds_per_token'])} tokens a serving GPU-second: {s['basis']}"), "",
              (f"| block type | k_log | blocks {'proved ' if spot else ''}per token | ms per block | "
               f"{'ms per block in the plan | ' if planned else ''}source | record | share of proving time | note |"),
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
              (f"- {'served tokens covered' if spot else 'tokens proved'} per second per prover GPU: "
               f"{_n(t['tokens_per_gpu_second'])}{_range(t['tokens_per_gpu_second_range'])}"),
              (f"- prover GPUs per serving GPU: {_n(t['gpus_per_serving_gpu'])}{_range(t['gpus_per_serving_gpu_range'])}, "
               f"{t['status']}: steady-state measurements cover {t['steady_time_share']:.1%} of "
               f"{'the proving time it prices' if t['missing'] else 'its proving time'}")]
    if t["missing"]:
        named = "; ".join(f"`{_short(m['block'])}` ({_missing_share(m)})" for m in t["missing"][:MISSING_NAMED])
        more = len(t["missing"]) - MISSING_NAMED
        lines.append(f"- partial: {len(t['missing'])} block type(s) with no figure, whose proving every total here leaves "
                     f"out: {named}" + (f"; and {more} more, which `--json` names" if more > 0 else ""))
    if "at_record" in doc:
        lines += _record_lines(doc)
    if above := t["above_cores"]:
        lines.append(_above_line(above, bool(t["missing"])))
    if off := t.get("off_node"):
        lines.append(f"- {off_node_line(off)}")
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
              ("| source | block types | blocks per token | share of blocks | prover GPU-seconds per token | "
               "share of proving time |"), "| --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {P.SOURCES[x['source']]} | {x['block_types']} | {_n(x['per_token'])} | "
              f"{'—' if x['block_share'] is None else format(x['block_share'], '.2%')} | {_n(x['gpu_seconds_per_token'])} | "
              f"{'—' if x['time_share'] is None else format(x['time_share'], '.2%')} |" for x in doc["sources"]]
    lines += _lean_rows(doc, rows)
    p = doc["periods"]
    lines += ["", (f"Per monitoring period, over all compute ({p['gates_basis']}); every total is the table's ratio scaled, "
              f"not measured at that size{', and partial as that ratio is' if t['missing'] else ''}:"), "",
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
        lines += ["", ("Unread (labelled steady-state rows whose staged Definition is no block type of this census under "
                  "its cut; they are in no figure above):"), ""]
        lines += [f"- {_record(x['record'])}{'' if x['record'] == x['run'] else ', run ' + x['run']}: `{x['type']}` at "
                  f"{x['provers']} prover{'' if x['provers'] == 1 else 's'}, `{x['definition']}`: "
                  + (f"not a steady-state measurement: {x['why']}" if x["seconds"] is None else
                     f"measured at steady state, {_n(1e3 * x['seconds'])} ms per block") for x in doc["unread"]]
    return "\n".join(lines) + "\n"
