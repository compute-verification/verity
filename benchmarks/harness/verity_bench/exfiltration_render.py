"""The document of `verity bench exfiltration` (`verity_bench.exfiltration`) as the Markdown it prints (`render`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from verity_bench.exfiltration_common import (
    BOUNDS,
    DEFAULT_BASIS,
    ERRORS,
    LEAN_BOUND,
    LEAN_NUMBERS,
    MARKS,
    NO_BOUND,
    NOT_PROVED,
    NOT_RECORDED,
    NUMERALS,
    ON_CPUS,
    PARTS,
    PLAN_FILE,
    PRINTED,
    READER,
    ROW,
    RUN_NOT_PROVED,
    SECURITY,
    SOURCES,
    TOGETHER,
    VERDICTS,
    WHOLE,
)
from verity_bench.exfiltration_document import _timed_alone
from verity_bench.exfiltration_planner import _label
from verity_bench.exfiltration_text import (
    _listed,
    _n,
    _names,
    _on,
    _p,
    _part,
    _pct,
    _table,
)
from verity_bench.served_table import RUN_OF_RECORD

#: a word `exfiltration/v3` keeps in the document's inputs (their `part` and `source`) until a format version, and the
#: Glossary's word the print says for it (GLOSSARY.md, Older names)
HELD_WORDS = {"network certifier": "network attester"}


def _said(text: str) -> str:
    """A document's input field as the print says it: each held word (`HELD_WORDS`) in the Glossary's."""

    for held, word in HELD_WORDS.items():
        text = text.replace(held, word)
    return text


def _amount(x: Any, error: float | None) -> str:
    """A figure's amount as a table prints it: `NO_BOUND` where its error, summed over everything it covers, is 1 or
    more."""

    return NO_BOUND if error is not None and error >= 1 else _n(x)


def _readers(share: float) -> str:
    """Which of the three readers of the planner's rows is the headline figure, and why the other two differ."""

    return (f"this is the headline figure: λ times the bound where the row's total is {_pct(share)} of compute, on a "
            "least-squares line of the total through every pinned budget. The planner's driver reads the same rows two "
            "other ways: `headline plan` prints the bound at one pinned budget, the plan's, whose total is that row's and "
            f"not set to {_pct(share)}, at λ = 1; `headline rows` prints λ times the bound at {_pct(share)}, with the total read "
            "between the two pinned budgets that bracket it alone.")


def _how(priced: Mapping | None) -> str:
    """How a scope's proving time is priced, by its plan: each way's share and the largest families; or that no plan
    says."""

    if priced is None:
        return ("how the row's proving time is priced is not read, and none of it is marked measured: the rows' `split` "
                "is not their own plan (it doesn't say the row's `q` and `replay_sample`: it is written before the row's "
                f"levers multiply its blocks), and no run of record holds the row's plan (`{PLAN_FILE}`)")
    largest = sorted(((s, family, way) for way, p in priced["ways"].items() for family, s in p["families"].items()),
                     reverse=True)[:3]
    plan = "its own plan at the pinned budget nearest" if priced["of"] == "rows" else "its plan"
    alone = "measured" in priced["ways"] and _timed_alone(priced, priced["ways"]["measured"]["share"])[1]
    return (f"of the row's proving time, by {plan}, β = {_part(priced['beta'])} ({priced['run']}), "
             "the share in block families priced each way is "
            + ", ".join(f"{_pct(p['share'])} {way}" for way, p in priced["ways"].items())
            + ": the largest families are " + ", ".join(f"{family} ({_pct(s)}, {way})" for s, family, way in largest)
            + (f". Of the families priced measured, {alone}" if alone else ""))


def _by_source(sources: Mapping | None) -> str:
    """The share of a row's proving each source prices and the share at the planner's assumed default (`_sources`),
    each `NOT_RECORDED` where the row's plan doesn't say."""

    default = None if sources is None else sources["default"]
    by = sources["by"] if sources else dict.fromkeys(SOURCES)
    return (", ".join(f"{s} {NOT_RECORDED if v is None else _pct(v)}" for s, v in by.items())
            + f"; at the planner's assumed default (`{DEFAULT_BASIS}`) {NOT_RECORDED if default is None else _pct(default)}")


def _called(name: str, stop: Mapping) -> str:
    """What the sum under the row ``name`` is printed as: `TOGETHER`'s, but for a row the calculator reads from
    verdict records whose stop class the reader's answer (``stop``, `build`'s `certifier`'s `stop`) charges its whole
    schedule on some execution phase, which is `WHOLE`'s, with the execution phases a link is held on where it holds
    any."""

    if name not in stop["rows"] or stop["below"]:
        return TOGETHER[name]
    return WHOLE[name] + (f" on all but {_n(stop['held_phases'])} execution phase(s) of a link" if stop["held_phases"] else "")


def _labelled(year: Mapping, key: str, set_time: str, stop: Mapping) -> list[str]:
    """The world year's figures (``key`` of each of its sums) that are printed only under a label, one clause a
    label: every row that carries its own, and every other row at the jitter bound ``set_time``; each called as the
    reader's answer ``stop`` has it (`_called`)."""

    own: dict[str, list[str]] = {}
    other: dict[str, list[str]] = {}
    for name in TOGETHER:
        t, what = year["together"][name], _called(name, stop)
        if t["label"]:
            own.setdefault(t["label"], []).append(f"{_n(t[key])} GB {what}")
        else:
            other.setdefault(t["with_set_time_card"]["label"], []).append(f"{_n(t['with_set_time_card'][key])} GB {what}")
    return [f"{label}: " + "; and ".join(filter(None, (
                "; ".join(own.get(label, ())),
                f"at D_j = {set_time}, " + "; ".join(other[label]) if label in other else "")))
            for label in dict.fromkeys((*own, *other))]


def _lean(lean: Mapping) -> str:
    """What Lean proves at numbers of one verification phase's conflict and first-hit part, as a binding bound says
    it: the instance the phase is within and its bits, or `NOT_PROVED` with the nearest instance and each numeral
    the phase is past."""

    if lean["theorem"]:
        return (f"`{LEAN_NUMBERS}.{lean['theorem']}` holds the part of it that is linear in the prover's calls to at "
                f"most 2^-{lean['bits']}, beside each session's knowledge error and the rewinding part")
    past = "; ".join(f"{NUMERALS.get(name, name)} at most {v['instance']:,} there and {v['design']:,} here"
                     for name, v in lean["past"].items())
    return (f"{NOT_PROVED}: the nearest instance, `{LEAN_NUMBERS}.{lean['nearest']}`, at most 2^-{lean['bits']}, has "
            f"{past}")


def _security(bound: Mapping | None) -> list[str]:
    """The security level's lines: a row's binding bound (`_binding`) over one verification phase and over its scope
    of them, each counted in calls and in compressions, what it assumes, and what Lean proves of each; or that the
    rows hold none."""

    if bound is None:
        return [(f"- **the {SECURITY}**: not read. The rows of record hold no binding bound (`verification_phase.binding`): "
                f"the planner wrote them before it computed one. Run it again and label its run ({_label()}).")]
    design, (calls, compressions) = bound["design"], (bound["counted"][c] for c in ("calls", "compressions"))
    scope = bound["scope"].replace(" ", "-")
    return [
        (f"- **the {SECURITY}** (modeled): the probability that an accepted run opens a value off the committed trace, by "
        f"the bound Lean states of a two-stage sample counted by classes of gates (`{LEAN_BOUND}`), every run at "
        f"{design['calls_a_run']} calls of SHA-512, at the row's own design at the pinned budget nearest, β = {_part(bound['beta'])} ({bound['run']}). "
        f"**Over one verification phase: at most {calls['phase']['bound']}** counted in calls, as the Lean counts, and "
        f"{compressions['phase']['bound']} in compressions. **Over a {scope} of phases**, {_n(design['phases'])} of "
        f"them: **at most {calls['phases']['bound']}** in calls and {compressions['phases']['bound']} in compressions. "
        f"A verification phase is a {bound['name']}'s {_n(bound['seconds'])} s on {_n(bound['machines'])} machines: its "
        f"first stage samples {_n(bound['sampled'])} replay blocks, and its second {_n(bound['proof_blocks'])} of their "
        f"proof blocks in {_n(design['sessions'])} sessions, the largest of {_n(bound['tables'])} "
        f"table{'s' * (bound['tables'] != 1)}."),
        f"- the {SECURITY} assumes: "
        + "; ".join(i["input"] if i["value"] is None else f"{_n(i['value'])} {i['input']}" for i in bound["assumed"])
        + ". It counts each session's knowledge error and the run's own opening checks as 0.",
        (f"- what Lean proves of the {SECURITY}: over one verification phase counted in calls, {_lean(calls['lean'])}. "
        f"Counted in compressions, {_lean(compressions['lean'])}. Over a {scope} of phases, {NOT_PROVED}: no theorem "
        "sums a monitoring period's verification phases.")]


def _share(cs: Mapping) -> tuple[list[str], list[str]]:
    """The share of compute a served run measures (`build`'s `compute_share`), beside the given share every other
    figure is read at: a line, and with a run's record a table of its parts."""

    given = f"the given {_pct(cs['given']['share'])} of total compute ({cs['given']['mark']})"
    if cs["value"] is None:
        lines = [f"**The audit's share of compute: {cs['why']}.** Every figure below is read at {given}.", ""]
    else:
        parts = ", ".join(f"{name.replace('_', ' ')} {_part(p['share'])}" + (f" ({ON_CPUS})" if p["on_cpus"] else "")
                          for name, p in cs["parts"].items())
        there = (f"its part has no figure at that share: {cs['no_GB']}" if cs["GB"] is None else
                 f"at that share, β = {_part(cs['beta'])}{'' if cs['inside'] else ' (extrapolated)'}, the compute "
                 f"audit's part over the world's year is {_n(cs['GB'])} GB")
        lines = [(f"**The audit's share of compute: {_part(cs['value'])} ({cs['mark']})**, on {cs['run']}: the audit's "
                 f"GPU-seconds over the window's serving, {_n(cs['serving_gpu_s'])} GPU-seconds, in parts {parts}. "
                 f"{there[0].upper()}{there[1:]}, beside {_n(cs['given']['GB'])} GB at {given}, which every figure "
                 "below is read at."), ""]
    if cs["run"] is None:
        return lines, []
    gaps = {g["field"]: g for g in cs["missing"]}

    def source(name: str, said: str) -> str:
        return said + (f"; lacks {', '.join(f'`{k}`' for k in gaps[name]['lacks'])}: {gaps[name]['why']}"
                       if name in gaps else "")

    return lines, [f"## The audit's share of compute on {cs['run']}", "",
            ("Each part of the audit's GPU time on the served run, and the window's serving it is a share of, read from "
            "the run's record. A part the record lacks is marked missing with its gap, and the share is not claimed "
            "without it."), "",
            *_table(("part", "what", "GPU-seconds", "share of the window's serving", "mark", "source"),
                    [("serving", "the window's serving", _n(cs["serving_gpu_s"]), "",
                      "missing" if "serving" in gaps else "measured", source("serving", "`serving`")),
                     *((name, p["what"], _n(p["gpu_s"]) + (f", {ON_CPUS}" if p["on_cpus"] else ""), _part(p["share"]),
                        p["mark"], source(name, p["source"]))
                       for name, p in cs["parts"].items())]), ""]


def render(doc: Mapping[str, Any]) -> str:
    """``doc`` (`build`) as text: the figure and what it is, the inputs with their marks, each part, and the two side
    by side for every scope and monitoring period. The network attester's figures of record come first; a figure
    that is under a label is printed in a line, a row or a column that carries that label, as the document gives
    it. Every line of the network attester's and of the two together carries its errors and the δ of the scans they
    sum, and a figure whose error is 1 or more is printed as `NO_BOUND`, with no amount."""

    share, net, w, cert = doc["share"], doc["network_check"], doc["weights"], doc["certifier"]
    head = next(p for p in doc["compute_audit"] if p["scope"] == "world year")
    planned = [p for p in doc["compute_audit"] if p["priced"]]
    year = next(p for p in doc["periods"] if (p["compute"], p["period"]) == ("world", "year"))
    of, set_time, card, default = cert["of_record"], cert["set_time"], cert["with_set_time_card"]["label"], cert["default_class"]
    count, sums, delta, eps, error = year["output_count"], year["together"], cert["scan_delta"], year["eps"], year["error"]
    record = [name for name in TOGETHER if not sums[name]["label"]]
    labels = {name: sums[name]["label"] for name in TOGETHER if sums[name]["label"]}
    marks = {m: sum(i["mark"] == m for i in doc["inputs"]) for m in MARKS}
    stop, charged = doc.get("stop_class"), cert["stop"]
    # the rows that take the output count at 0, and where; the rows of the sums that would, were an execution phase held
    at_zero, held = count["at_zero_in"], _on(charged)
    would = [name for name in TOGETHER if name in charged["rows"]]
    measured, parts = _share(doc["compute_share"])
    outside = "" if head["inside"] else ", outside the pinned budgets: extrapolated"
    what = ", the plan of record's row, with every modeled lever at its measured value" if doc["row"] == ROW else ""
    what += f", in its variant `{doc['variant']}`" if doc["variant"] else ""
    label = f", {doc['label'] or head.get('note')}" if doc["label"] or head.get("note") else ""
    sources = head["priced"] and head["priced"]["sources"]
    plan = f" (its own plan at β = {_part(head['priced']['beta'])}, {head['priced']['run']})" if sources else ""
    a = doc["at_assumed_prices"]
    assumed = [] if a is None else [
        f"- row `{a['row']}`, {a['label']}: "
        + (f"not read: {a['why']}." if a["GB"] is None else
           f"{_n(a['GB'])} GB, at β = {_part(a['beta'])}{'' if a['inside'] else ', extrapolated'} ({a['run']}), failing "
           f"with at most δ = {_n(a['delta'])} over the monitoring period. It is not the figure. What prices its "
           f"proving time: {_by_source(a['sources'])}.")]
    on = f"modeled, on an assumed class of at most {_n(cert['response_ids'])} ids a response"
    if eps < 1:
        scan = next(r for r in cert["rows"] if r["row"] == "scan")
        attester = [
            f"- **the network attester's part: {_n(sums['scan']['certifier_GB'])} GB** ({on}) {TOGETHER['scan']}, "
            f"{_n(scan['term'][of]['bits_a_request'])} bits a request, and "
            + ", ".join(f"{_n(sums[name]['certifier_GB'])} GB {_called(name, charged)}" for name in record if name != "scan")
            + f", each at D_j = {of}, its term of record, and each failing with at most ε = {_p(eps)} over the world's "
            f"year, at a scan's δ of {_p(delta['value'])}."
            + ("" if default["response_ids"] == cert["response_ids"] else
               f" On the class of at most {_n(default['response_ids'])} ids a response that `verity bench certifier_term` "
               "assumes by default, the scan's is "
               + (f"{_n(default['value'])} {default['unit']}." if default["eps"] < 1 else
                  f"{NO_BOUND}, at an ε of {_n(default['eps'])}.")),
            *(f"- the network attester's part {clause}; ε = {_p(eps)}."
              for clause in _labelled(year, "certifier_GB", set_time, charged))]
    else:
        attester = [(f"- **the network attester's part: {NO_BOUND}** ({on}). Its ε over the world's year is {_n(eps)}, "
                     "and at 1 or more it bounds nothing, so no amount is printed under any of its rows.")]
    if error < 1:
        together = [
            "- **together**, a link's execution phase carrying K = schedules + cells + jitter + min(entries, advice + "
            f"the compute audit's part), at D_j = {of}, with no advice priced and with a response's output count taken "
            f"at 0 under {_names(at_zero) + held if at_zero else 'no row'}: "
            + "; ".join(f"{_n(sums[name]['GB'])} GB {_called(name, charged)}" for name in record)
            + f"; each failing with at most {_n(error)}, the two parts' errors summed.",
            *(f"- together, {clause}; error {_n(error)}." for clause in _labelled(year, "GB", set_time, charged))]
    else:
        together = [(f"- **together: {NO_BOUND}**. The two parts' errors over the world's year sum to {_n(error)}, and at "
                    "1 or more the sum bounds nothing, so no amount is printed for the two together.")]
    lines = [
        (f"# verity bench exfiltration: {doc['model']}'s exfiltration capacity over a monitoring period, at "
        f"{_pct(share)} of total compute"), "",
        *measured,
        f"Over the world's year, at {_pct(share)} of total compute, in two parts:", "",
        (f"- **the compute audit's part{label}: {_n(head['GB'])} GB** (modeled), {_n(head['GB'] / w['GB'])} copies of the weights "
        f"({w['label']}, {_n(w['GB'])} GB) and {_n(year['compute_audit']['bits_a_request'])} bits a request. It is the "
        f"planner's row `{doc['row']}`{what}, read at the audit budget where its total is {_pct(share)} of the compute, "
        f"β = {_part(head['beta'])}{outside}: its bound, "
        f"{_n(head['bound']['GB'])} GB, times λ = {_n(head['lambda'])}. The row is priced over "
        f"{_n(head['requests'])} requests, the count the network attester's part is over."),
        f"- {_readers(share)}",
        f"- {_how(head['priced'])}.",
        f"- what prices the row's proving time, as its rows record it{plan}: {_by_source(sources)}.",
        *assumed,
        *attester,
        *together,
        "- the output count: "
        + (f"{_n(count['GB'])} GB ({count['mark']}), {_n(count['bits_a_request'])} bits a request" if eps < 1 else
           f"{NO_BOUND} ({count['mark']}), under the network attester's ε")
        + ". Until a served audit has sampled a stop block the server chooses a response's output count, and the network "
        f"attester's row `{count['row']}` charges it as its schedules. "
        + (f"The sums under {_names(at_zero)} take it at 0{held}, as they price no advice." if at_zero else
           f"No sum takes it at 0: those under {_names(would)} hold it too, their stop class charged its whole schedule.")
        + ("" if stop is None else f" The stop class's verdicts of record (`{stop['record']}`, "
           f"{', '.join(stop['runs']) or 'no run read'}): {stop['said']}."),
        f"- the stop class of the network attester's rows {_names(charged['rows'])} is charged "
        + ("below its schedule" if charged["below"] else "its whole schedule" if not charged["held_phases"] else
           f"below its schedule on {_n(charged['held_phases'])} execution phase(s) of a link and its whole schedule on "
           "every other execution phase of a monitoring period")
        + (f", by the answer of the stop verdicts' reader (`{READER}.of_runs`) on the runs that carry `{VERDICTS}` "
           f"({', '.join(charged['runs'])}"
           + (f", of which superseded: {', '.join(charged['superseded'])}" if charged["superseded"] else "") + ")"
           if charged["runs"] else
           ", as the calculator charges it with no record read: no run carries the stop class's record, so the stop "
           f"verdicts' reader (`{READER}.of_runs`) was not asked")
        + ". A sum takes the output count at 0 only on an execution phase the reader's answer holds, never because "
          "its row's schedules read 0:",
        *(f"  - {line}" for line in charged["why"]),
        (f"- the compute audit's part fails with at most δ = {_n(head['delta'])} over the monitoring period. The network "
        f"attester's ε over the world's year is each execution phase's scan's δ ({delta['mark']}), "
        f"{_p(delta['value'])}, summed over every link's execution phases, {_n(year['phases'])} of them: {_p(eps)}. "
        f"The two together fail with at most the sum of both, {_n(error)}. A figure whose error, summed over "
        f"everything it covers, is 1 or more is printed as \"{NO_BOUND}\", with that error and no amount."),
        (f"- with the network check's own proving at {_pct(net['share'])} of total compute, which the planner's rows "
        f"don't count (assumed, no run), the audit budget is β = {_part(net['beta'])} and the compute audit's part "
        f"{_n(net['GB'])} GB."),
        *_security(head["binding"]),
        f"- of {len(doc['inputs'])} inputs, {marks['measured']} {'is' if marks['measured'] == 1 else 'are'} measured, "
        + _listed([f"{marks['modeled']} modeled", f"{marks['simulated']} simulated", f"{marks['assumed']} assumed",
                   *(f"{marks[m]} {BOUNDS.get(m) or PRINTED.get(m, m)}"
                     for m in (*BOUNDS, "missing", RUN_NOT_PROVED) if marks[m])]) + ".", "",
        f"Inputs, each a labelled record of the store (`{RUN_OF_RECORD}`):", "",
        f"- compute-audit: {', '.join(doc['runs'])} (`{doc['record']}`): the planner's rows, "
        f"{sum(len(p['pinned']) for p in doc['compute_audit'])} of row `{doc['row']}` at step `{head['step']}` over "
        f"{len(doc['compute_audit'])} scopes"
        + "".join(f", and the {p['scope']}'s plan ({p['priced']['run']})" for p in planned if p["priced"]["of"] == "plan"),
        *([] if stop is None else [
            (f"- the stop class: {', '.join(stop['runs']) or 'no run read'} (`{stop['record']}`): "
            f"{len(stop['verdicts'])} verdicts, {stop['said']}")]),
        ("- the network attester's part and the requests both parts count over (modeled or assumed, read from the tree, "
        "not records): `verity bench certifier_term`'s inputs, among them the request it counts and what that request "
        "is, in the table below"), "",
        "## Inputs", "",
        *_table(("part", "input", "value", "unit", "mark", "source"),
                [(_said(i["part"]), i["input"], _n(i["value"]), i["unit"], PRINTED.get(i["mark"], i["mark"]),
                  _said(i["source"]))
                 for i in doc["inputs"]]), "",
        *parts,
        f"## The compute audit's part at {_pct(share)} of total compute", "",
        ("Each scope's rows of record, read at the audit budget β where their total is the share. The requests are the "
        "count the rows are priced over. The parts are shares of total compute; the wait is part of serving."), "",
        *_table(("scope", "run", "β", "bound at λ = 1, GB", "GB", "x weights", "requests", *PARTS),
                [(p["scope"], p["run"], _part(p["beta"]) + ("" if p["inside"] else " (extrapolated)"), _n(p["bound"]["GB"]),
                  _n(p["GB"]), _n(p["bits"] / w["bits"]), _n(p["requests"]), *(_part(p["parts"].get(part)) for part in PARTS))
                 for p in doc["compute_audit"]]), "",
        *(planned and (
            ("Each scope's proving time by how its block families are priced, by its plan: the rows' own at the pinned "
            "budget nearest β, or a plan of record at its own budget. How each family is priced is the rows' account."), "",
            *_table(("scope", "plan", "at β", "priced", "share of proving time", "block families", "the largest"),
                    [(p["scope"], p["priced"]["run"] + (" (rows)" if p["priced"]["of"] == "rows" else ""),
                      _part(p["priced"]["beta"]), way, _part(x["share"]), str(len(x["families"])),
                      ", ".join(f"{family} {_part(s)}" for family, s in list(x["families"].items())[:3]))
                     for p in planned for way, x in p["priced"]["ways"].items()]), "")),
        "## The network attester's part over the world's year", "",
        (f"`verity bench certifier_term`'s rows (`{cert['schema']}`) at its request count `{cert['request_count']}`, for a "
        f"class of at most {_n(cert['response_ids'])} ids a response. The term is the schedules, the fail cells and the "
        f"jitter, and the term of record is the one at D_j = {of}. The term at D_j = {set_time}, and every figure of a "
        f"row whose name carries it, is printed only under the network attester's label, \"{card}\". The entries are "
        "the compute side's. Each row carries the δ of its scans and its ε over the world's year, their sum over every "
        "link's execution phases; a row whose ε is 1 or more has no amount."), "",
        *(f"- {name} ({'; '.join(filter(None, (k['standing'], k.get('mark'))))}): {k['counts']}"
          for name, k in cert["legend"].items()), "",
        *_table(("row", "schedules, GB", "cells, GB", f"term at D_j = {of}, GB", "x weights",
                 f"term at D_j = {set_time} {card}, GB", "entries, GB", "scan's δ", "ε"),
                [(r["row"] + (f", {r['with_set_time_card']}" if r["with_set_time_card"] else ""),
                  *((NO_BOUND,) * 6 if r["eps"] >= 1 else
                    (_n(r["schedules"]["GB"]), _n(r["cells"]["GB"]), _n(r["term"][of]["GB"]),
                     _n(r["term"][of]["of_weights"]), _n(r["term"][set_time]["GB"]),
                     _n(r["entries"] and r["entries"]["GB"], "the compute side's"))),
                  _n(delta["value"]), _n(r["eps"])) for r in cert["rows"]]), "",
        f"The entries as the scan's check charges them: {cert['entries_rule']}.", "",
        "## Each monitoring period over all compute: the two parts side by side, in GB", "",
        f"The network attester's term is the one of record, at D_j = {of}, and `together` adds the compute audit's part "
        "to it, or the entries where the row charges them and they are fewer. The output count is the schedules of "
        f"the row `{count['row']}`, what a response's output count is charged while the server chooses it "
        f"({count['mark']}): "
        + (f"the sum under {_names([name for name in at_zero if name in record])} takes it at 0{held}."
           if any(name in record for name in at_zero) else "no sum of this table takes it at 0."), "",
        ("Each part's amounts are followed by its error over the line's scope and monitoring period: the compute "
        "audit's δ; the network attester's ε, its scans' δ summed over every link of the scope and every execution "
        "phase of the period; and `together`'s, the sum of the two. A figure whose error is 1 or more is printed as "
        f"\"{NO_BOUND}\"."), "",
        *_table(("compute", "period", "GPU-seconds", "requests", "compute audit", "bits a request", "compute audit's δ",
                 *(f"network attester, {name}" for name in record), "scan's δ", "network attester's ε",
                 *(f"together, {name}" for name in record), "together's error", f"output count ({count['mark']})"),
                [(p["compute"], p["period"], _n(p["gpu_seconds"]), _n(p["requests"]),
                  _n(p["compute_audit"] and p["compute_audit"]["GB"], "no rows of record"),
                  _n(p["compute_audit"] and p["compute_audit"]["bits_a_request"]),
                  _n(p["compute_audit"] and p["compute_audit"]["delta"]),
                  *(_amount(p["together"][name]["certifier_GB"], p["eps"]) for name in record),
                  _n(delta["value"]), _n(p["eps"]),
                  *(_amount(p["together"][name]["GB"], p["error"]) for name in record), _n(p["error"]),
                  _amount(p["output_count"]["GB"], p["eps"]))
                 for p in doc["periods"]]), "",
        f"## Each monitoring period over all compute, {card}: in GB", "",
        f"No amount of this table is of record. Each is under the network attester's label, \"{card}\", which its "
        f"column carries: every amount of a row that carries the label, and every other row's at D_j = {set_time}."
        + (f" The sum under {_names([name for name in at_zero if name in labels])} takes the output count "
           f"at 0{held} too." if any(name in labels for name in at_zero) else "")
        + " A line's errors are those of the table above.", "",
        *_table(("compute", "period",
                 *(f"{part}, {name}, {label}" for name, label in labels.items()
                   for part in ("network attester", "together")),
                 *(f"{part}, {name}, D_j = {set_time} {card}" for part in ("network attester", "together")
                   for name in record),
                 "scan's δ", "network attester's ε", "together's error"),
                [(p["compute"], p["period"],
                  *(_amount(p["together"][name][key], p[e]) for name in labels for key, e in ERRORS.items()),
                  *(_amount(p["together"][name]["with_set_time_card"][key], p[e]) for key, e in ERRORS.items()
                    for name in record),
                  _n(delta["value"]), _n(p["eps"]), _n(p["error"]))
                 for p in doc["periods"]]),
    ]
    if doc["refused"]:
        lines += ["", "Refused (not read):", "", *(f"- {x['target']} (`{x['record']}`): {x['why']}" for x in doc["refused"])]
    return "\n".join(lines)
