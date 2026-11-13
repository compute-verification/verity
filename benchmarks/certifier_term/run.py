"""`verity bench certifier_term [ARGS]`: the network certifier's term for the served model at the load of a node, the
census's reference datacenter and the world, over a week, a month and a year, in bits and GB (the README beside this
file). It runs the calculator of K, `benchmarks/network_traces/leak_bounds.py`, with ARGS, and prints what that prints.

With `--out DIR` it also writes its result, through the harness kit's writer for `DIR` (`verity_bench.kit`) and through
nothing else; the kit says where the file is, `DIR/result.json` outside a research run (`benchmarks/harness/README.md`).
The result holds the fields every resource's benchmark reports (`measurements`), for the row the term of record is read
from, the scan at two release schedules, at each scope and monitoring period the calculator's document holds. Every
figure is the model's and its name says so (`MODELED`); none is computed from a run's K.

- `metric`: the term of record as the document gives it, in GB, beside the document's target. Its row is the scan's
  over the world's year (`RECORD`).
- the card's figure, a `metric` named under the card's mark (`CARD`: `simulated`): the same row's term at the other
  jitter bound, what a network certifier whose network card releases at a set time would be charged, as the document
  gives it under its label (`CARD_FIGURE`), in GB beside the same target. It is no term of record. Its `inputs` open
  with the card as the document states it: that it is simulated, and the clause the figure rests on.
- `free`: the term K in bits, which the check bounds from above, beside the same target in bits.
- `capacity`: what the scope's certified boundary links carry out over the period, in bits. It is read from the
  registered hardware and never copied: `--registration FILE` names a registration log, and a machine's uplink is the
  link of its host that a network certifier splits (`uplink`), at its `beta` bits a tick. `beta` is each way and the
  term is egress, so it counts once. A scope's machines are each taken as registered so. With no registration it has
  no value, and its `inputs` say why.
- `verified`: capacity less free, where both have a value.
- `error` (`ERROR`): the row's ε, the calculator's `eps`, a union bound over the period's execution phases.

A row whose error is 1 or more has no bound at its period (`NO_BOUND`): its `free` and `verified`, and `metric` and the
card's figure where the row is the term of record's, are written with no value and that reason in their `inputs`; its
`error` is written.

Each measurement's `inputs` name what it was computed from, and among them the class the term is at, the most ids a
response holds, and the scan's δ its error is at, as the document states them (`at_class`, `at_delta`).
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import math
import sys
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, NamedTuple

#: the directory of the calculator of K, which is loaded from its file
CALCULATOR = Path(__file__).resolve().parents[1] / "network_traces"
#: what every measurement's name starts with: the benchmark, and that the figure is the model's
MODELED = "certifier_term/modeled"
#: the row the term of record is read from: the scan, as `main` charges it
ROW = "scan"
#: the scope and monitoring period of that row, the document's `value`: the world's year
RECORD = ("world", "year")
#: the document's key for the card's figure, that row's term at the other jitter bound with its label and its mark,
#: and the document's input that is the card, whose source is what the figure rests on
CARD_FIGURE, CARD = "with_set_time_card", "set_time_card"
#: the last word of the error's name, the field every resource's benchmark reports its error under
ERROR = "error"
#: the calculator's key for a row's ε, which the error is
EPS = "eps"
#: the document's input that is the class the term is at: the most ids a response holds
CLASS = "response_ids"
#: the document's input that is the δ of each execution phase's scan, which a row's ε sums
DELTA = "scan_delta"
#: why a row whose error is 1 or more carries no amount
NO_BOUND = "no bound at this period"
NO_REGISTRATION = ("absent: no registration is given (--registration FILE), and capacity is read only from a "
                   "registered link's beta")
SCALED = ("modeled: each machine of the scope registered as this one, so the scope's machines times beta times the "
          "period's ticks; beta is each way and the term is egress, so one direction")
UNION = (f"{EPS}, the calculator's key: a union bound over the period's execution phases, their number times "
         f"{DELTA}; at 1 or more it bounds nothing")


class Uplink(NamedTuple):
    """A machine's certified boundary links as a registration registers them: their `beta` summed, in bits a tick each
    way, of which the term's, egress, is one; a tick's seconds; and what a measurement's inputs say of where they were
    read."""
    beta: int
    tick: Fraction
    said: str


def calculator() -> Any:
    """The calculator of K, `leak_bounds`, loaded from its directory."""
    if (path := str(CALCULATOR)) not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("leak_bounds")


def uplink(path: Path, links: int) -> Uplink:
    """A machine's uplink as the registration log at ``path`` registers it (`verity.hardware.registration`'s committed
    log): the site its site entries register (`Site.of_log`) holds one host, and each of that host's boundary links is
    split by a network certifier (`Site.is_boundary`, `Site.certified`), ``links`` of them, the model's links a machine.
    The log is read as given: its check is its verifier's."""
    from verity.hardware import registration
    from verity.hardware.site import TICK, Entry, Site

    try:
        composite, log = registration.log_of(json.loads(path.read_text()))
    except (OSError, ValueError) as e:
        raise SystemExit(f"certifier_term: {path} is no registration log: {e}") from e
    site = Site.of_log([e for e in log if isinstance(e, Entry)])
    hosts, boundary = [x.id for x in site.locs if x.kind == "host"], site.boundary()
    split = [k for k in boundary if site.certified(k)]
    if len(hosts) != 1 or len(split) != links or len(boundary) != len(split):
        raise SystemExit(f"certifier_term: {path} doesn't register the model's machine, one host whose {links} boundary "
                         f"link(s) a network certifier splits: it registers the hosts {hosts}, whose boundary links are "
                         f"{[k.id for k in boundary]}, {[k.id for k in split]} of them certified")
    return Uplink(sum(k.beta for k in split), TICK,
                  f"registration {registration.head(composite, log)}, read as given: host {hosts[0]}'s certified "
                  f"boundary link(s) {', '.join(f'{k.id} at beta {k.beta}' for k in split)}, in bits each way a tick of "
                  f"{TICK} s (verity.hardware.site)")


def stated(key: str, entry: dict) -> str:
    """An input of the calculator's document as a measurement's inputs name it: its value, its mark and its source."""
    value = " ".join(str(x) for x in (entry["value"], entry["unit"]) if x != "")
    return f"{key} = {value} ({entry['mark']}): {entry['source']}"


def at_class(doc: dict) -> str:
    """The class the document's terms are at (`CLASS`), as the document states it."""
    return stated(CLASS, doc["inputs"][CLASS])


def at_delta(lb: Any, doc: dict) -> str:
    """The scan's δ the document's ε are at (`DELTA`): as the document states it where it is the calculator's default,
    and where the command gives another, that it is given and assumed, no scan having run at it."""
    entry = doc["inputs"][DELTA]
    if entry["value"] == float(lb.SCAN_DELTA):
        return stated(DELTA, entry)
    return (f"{DELTA} = {entry['value']} ({entry['mark']}): each execution phase's scan's δ as the command gives it "
            "(--scan-delta), to the document's float; no scan has run at it")


def unbounded(row: dict) -> list[str]:
    """Why a row of the document carries no amount, as a measurement's inputs say it: nothing where its error is under
    1, and where it is 1 or more that it has no bound at its period (`NO_BOUND`), a union bound there bounding nothing."""
    return [] if row[EPS] < 1 else [f"absent: {NO_BOUND}: the row's error, {row[EPS]}, is 1 or more"]


def document(lb: Any, argv: Sequence[str]) -> dict:
    """The calculator ``lb``'s document on ``argv``, as its `main` prints it under `--json`."""
    with contextlib.redirect_stdout(io.StringIO()) as text:
        lb.main([*argv, "--json"])
    return json.loads(text.getvalue())


def card_bound(lb: Any) -> str:
    """The jitter bound the card's figure is at, as a row's `term` names it: the calculator's other than the one of
    record."""
    other, = (str(dj) for dj in lb.JITTERS if dj != lb.OF_RECORD)
    return other


def record(lb: Any, argv: Sequence[str], doc: dict) -> dict:
    """The row the term of record is read from (`ROW`, `RECORD`): ``doc``'s where the command ``argv`` asks for its
    scope and monitoring period, and else the same command's document once it asks for them too. It is the row whose
    term the document gives as its `value`, and whose term at the other jitter bound it gives as the card's figure
    (`CARD_FIGURE`), or the runner refuses."""

    def of(d: dict) -> dict | None:
        return next((r for r in d["rows"] if (r["row"], r["scope"], r["period"]) == (ROW, *RECORD)), None)

    row = of(doc) or of(document(lb, [*argv, "--scope", RECORD[0], "--period", RECORD[1]]))
    if row is None or row["term"][str(lb.OF_RECORD)]["GB"] != doc["value"]:
        raise SystemExit(f"certifier_term: the calculator's document gives a term of record, {doc['value']} "
                         f"{doc['unit']}, that isn't its row {ROW}'s over the {RECORD[0]}'s {RECORD[1]} at D_j = "
                         f"{lb.OF_RECORD}: the runner writes no result it can't name the row of")
    if row["term"][card_bound(lb)]["GB"] != doc[CARD_FIGURE]["value"]:
        raise SystemExit(f"certifier_term: the calculator's document gives a figure {doc[CARD_FIGURE]['label']}, "
                         f"{doc[CARD_FIGURE]['value']} {doc[CARD_FIGURE]['unit']}, that isn't its row {ROW}'s over the "
                         f"{RECORD[0]}'s {RECORD[1]} at D_j = {card_bound(lb)}: the runner writes no result it can't "
                         "name the row of")
    return row


def measurements(lb: Any, doc: dict, head: dict, up: Uplink | None = None) -> list[dict]:
    """The calculator ``lb``'s document ``doc`` in the fields every resource's benchmark reports, as the kit writer's
    measurements, each named under `MODELED`: the term of record (`metric`), whose row is ``head`` (`record`); the
    card's figure, ``head``'s term at the other jitter bound, a `metric` named under the card's mark; and for
    each scope and monitoring period the document holds, that row's `free`, `capacity`, `verified` and error (`ERROR`),
    the row being the scan's at the release schedules of record (`ROW`). `metric`, the card's figure and `free` stand
    against the document's target, in their own units. Capacity is the scope's machines times a machine's uplink ``up``
    one way over the period, rounded up; with no uplink, capacity and verified have no value and their inputs say why.
    A row whose error is 1 or more carries no `free`, no `verified` and, where it is ``head``, no `metric` and no card's
    figure, and their inputs say why (`unbounded`); its error is written. Every measurement's inputs name the class the
    term is at and the scan's δ its error is at (`at_class`, `at_delta`), `metric`'s, the card's figure's and `free`'s
    among the document's inputs; the card's figure's open with the card (`CARD`), whose source is the clause the
    figure rests on."""
    at, target = str(lb.OF_RECORD), doc["target"]["value"]
    basis = [at_class(doc), at_delta(lb, doc)]
    by = {k: basis[1] if k == DELTA else stated(k, i) for k, i in doc["inputs"].items()}
    said, card = list(by.values()), doc[CARD_FIGURE]

    def stands(value: float, ceiling: float, none: list[str]) -> str | None:
        return None if none else "over target" if value > ceiling else "within target"

    none = unbounded(head)
    out = [{"name": f"{MODELED}/metric", "value": None if none else doc["value"], "unit": doc["unit"], "target": target,
            "verdict": stands(doc["value"], target, none), "inputs": [f"modeled: {doc['name']}", *none, *said]},
           {"name": f"{MODELED}/{card['mark']}/metric", "value": None if none else card["value"], "unit": card["unit"],
            "target": target, "verdict": stands(card["value"], target, none),
            "inputs": [f"modeled: the row of {MODELED}/metric at D_j = {card_bound(lb)}, {card['label']}: no term of "
                       "record", by[CARD], *none, *(s for k, s in by.items() if k != CARD)]}]
    for r in doc["rows"]:
        if r["row"] != ROW:
            continue
        name = f"{MODELED}/{r['scope']}/{r['period']}"
        row = (f"modeled: benchmarks/network_traces/leak_bounds.py's row {ROW} at D_j = {at}, scope {r['scope']}, "
               f"period {r['period']}, request count {r['request_count']}")
        free, ceiling, none = r["term"][at]["bits"], target * lb.GB, unbounded(r)
        capacity = None if up is None else math.ceil(r["machines"] * up.beta * Fraction(lb.PERIODS[r["period"]]) / up.tick)
        absent = [NO_REGISTRATION] if up is None else []
        out += [
            {"name": f"{name}/free", "value": None if none else free, "unit": "bit", "target": ceiling,
             "verdict": stands(free, ceiling, none), "inputs": [row, *none, *said]},
            {"name": f"{name}/capacity", "value": capacity, "unit": "bit",
             "inputs": [*(absent or [up.said, SCALED]), row, *basis]},
            {"name": f"{name}/verified", "value": None if absent or none else capacity - free, "unit": "bit",
             "inputs": [f"{name}/capacity", f"{name}/free", *absent, *none, *basis]},
            {"name": f"{name}/{ERROR}", "value": r[EPS], "unit": "probability", "inputs": [row, UNION, *basis]},
        ]
    return out


def main(argv: Sequence[str]) -> int:
    """The calculator's `main` on ``argv`` but for the runner's own options, and with `--out` its result, which is
    written before anything is printed: a refusal prints nothing of the calculator's."""
    ap = argparse.ArgumentParser(prog="verity bench certifier_term", usage=argparse.SUPPRESS, add_help=False,
                                 allow_abbrev=False)
    group = ap.add_argument_group("the runner's own options")
    group.add_argument("--out", type=Path, metavar="DIR",
                       help="also write the result through the harness kit's writer for DIR, which says where the file "
                            "is (DIR/result.json outside a research run): for the scan at two release schedules, each "
                            f"scope and monitoring period's free, capacity, verified and {ERROR}, the term of record, "
                            "and beside it the same row's figure with a simulated set-time card, every one modeled; a "
                            "row whose error is 1 or more carries no amount")
    group.add_argument("--registration", type=Path, metavar="FILE",
                       help="with --out: the registration log a machine's certified boundary link is read from, for "
                            "capacity and verified (default: none, and they are written with no value and why)")
    own, rest = ap.parse_known_args(argv)
    if own.registration is not None and own.out is None:
        ap.error("--registration goes with --out: it is read for the result's capacity")
    lb, asked = calculator(), bool({"-h", "--help"} & set(rest))
    if own.out is None or asked:
        try:
            return lb.main(rest)
        except SystemExit as done:
            if asked and not done.code:
                print("\n" + ap.format_help(), end="")
            raise
    doc = document(lb, rest)
    head = record(lb, rest, doc)
    up = None if own.registration is None else uplink(own.registration, doc["inputs"]["links"]["value"])
    from verity_bench import kit

    try:
        writer = kit.writer(own.out)
        for m in measurements(lb, doc, head, up):
            writer.measurement(**m)
    except ValueError as e:
        raise SystemExit(f"certifier_term: {e}") from e
    return lb.main(rest)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
