"""The stop class's verdicts that `verity bench exfiltration` (`verity_bench.exfiltration`) reads: each verdict's record,
the runs that hold them, and whether they complete a class."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from verity_bench.exfiltration_common import (
    AGREED,
    CLASS_FIELDS,
    CLASS_NAME,
    CONTROLS,
    ISSUE_ID,
    PIECES,
    RATE,
    RATE_TEXT,
    RUN_NOT_PROVED,
    VERDICT_FIELDS,
    VERDICT_READS,
    VERDICTS,
    VERDICTS_FILE,
    VERIFIER_FIELDS,
    _reader,
)
from verity_bench.exfiltration_planner import _label, _run_files
from verity_bench.exfiltration_text import _listed
from verity_bench.served_table import SUPERSEDED_BY, record_runs, records


def verdict(x: Any, where: str) -> dict:
    """A verdict's record (`VERDICT_FIELDS`), refused where it lacks a field, by the field's name, or where a field
    holds what that field can't."""

    if not isinstance(x, Mapping):
        raise ValueError(f"{where} is a verdict's record, one JSON object, not {type(x).__name__}")
    if lacks := [f"`{k}`" for k in VERDICT_FIELDS if k not in x]:
        raise ValueError(f"{where} lacks {', '.join(lacks)}: a verdict's record holds "
                         f"{', '.join(f'`{k}`' for k in VERDICT_FIELDS)} (`VERDICT_FIELDS` in "
                         "benchmarks/harness/verity_bench/exfiltration.py)")
    v = x["verifier"]
    if lacks := [f"`verifier.{k}`" for k in VERIFIER_FIELDS
                 if not (isinstance(v, Mapping) and isinstance(v.get(k), str) and v[k])]:
        raise ValueError(f"{where} lacks {', '.join(lacks)}: a verdict's verifier is its module and the commit it ran at")

    def text(k: str) -> bool:
        return isinstance(x[k], str) and bool(x[k])

    for ok, why in ((x["piece"] in PIECES, f"`piece` is one of {', '.join(PIECES)}"),
                    (x["control"] is None or x["control"] in CONTROLS,
                     f"`control` is null on the honest run, else one of {', '.join(CONTROLS)}"),
                    (text("run"), "`run` is the id of the run that produced the verdict"),
                    (x["verdict"] in VERDICT_READS, f"`verdict` is one of {', '.join(VERDICT_READS)}"),
                    (x["proved_by"] is None or text("proved_by"),
                     "`proved_by` is null, or the Lean guarantee whose proof covers the verifier"),
                    (x["proved_by"] is not None or text("open"),
                     "`open` states what stands between the verdict and proved, while `proved_by` is null"),
                    (x["verdict"] != "withheld" or text("open"),
                     "`open` states what the verifier withheld its accept for, on a `withheld` verdict"),
                    (not (isinstance(x["open"], str) and ISSUE_ID.search(x["open"])),
                     ("`open` names an issue, and the print carries no issue's id: state what is open in words (e.g. "
                     "`the two-round form's soundness is not proved`), and put the issue on the board"))):
        if not ok:
            raise ValueError(f"{where} holds {', '.join(f'`{k}` {x[k]!r}' for k in VERDICT_FIELDS if k != 'verifier')}: "
                             f"{why}")
    out = {k: x[k] for k in VERDICT_FIELDS} | {"verifier": {k: v[k] for k in VERIFIER_FIELDS}}
    if x["control"] is not None:
        return out
    rate = RATE_TEXT.match(x["rate"]) if isinstance(x.get("rate"), str) else None
    for k, ok, why in (("registration", text, "the digest of the class's boundary record"),
                       ("stream_registration", text, "the digest of its own stream's registration"),
                       ("stream_names_boundary", lambda k: isinstance(x[k], bool),
                        "true or false, the verifier's recorded check that its stream's registration names the boundary"),
                       ("serve", text, "the store id of the served run the class came from"),
                       ("class", lambda k: isinstance(x[k], str) and bool(CLASS_NAME.match(x[k])),
                        "`w<window>/p<P>-o<O>`, with `-nostop` for a class with no stop block"),
                       ("rate", lambda k: rate is not None and math.gcd(int(rate[1]), int(rate[2])) == 1,
                        "`NUM/DEN` in lowest terms, the rate the registration record states")):
        if k in x and not ok(k):
            raise ValueError(f"{where} holds `{k}` {x[k]!r}: `{k}` is {why}")
    return out | {k: x[k] for k in CLASS_FIELDS if k in x}


def _what(v: Mapping) -> str:
    return PIECES[v["piece"]] + ("" if v["control"] is None else f", the control with {CONTROLS[v['control']]}")


def verdict_runs(store: Any) -> dict[str, list[str]]:
    """Every run that carries `run_of_record=VERDICTS`, each with the runs that replace it (`SUPERSEDED_BY`; none
    where it is not superseded). It is the one list of the record's runs: `stop_verdicts` reads or leaves out each
    of them, the print names each as read or as refused, and `stop_charge` hands the same runs to the stop
    verdicts' reader, those with a replacement as its superseded ones."""

    return {run: [str(lab.value) for lab in store.labels(run, key=SUPERSEDED_BY)]
            for run in sorted({run for run, _ in record_runs(store, records(store).get(VERDICTS, []))})}


def stop_charge(root: Path, store: Any, labelled: Mapping[str, Sequence[str]]) -> Any:
    """How the stop class is charged by the records the runs of ``labelled`` (`verdict_runs`) hold: the stop
    verdicts' reader's answer (`READER`'s `of_runs`), which `attester` hands the calculator. The reader is handed
    every run of ``labelled`` and, as superseded, those another run replaces; this reads none of their files itself."""

    return _reader().of_runs(store, list(labelled), [run for run, by in labelled.items() if by])


def stop_verdicts(store: Any, labelled: Mapping[str, Sequence[str]]) -> tuple[list[dict], list[str], list[dict]] | None:
    """The stop class's verdicts of record (`verdict`), the runs read and the runs left out with why; None where
    nothing in the store carries `run_of_record=VERDICTS`. ``labelled`` is the record's runs (`verdict_runs`): every
    one not superseded is read, each verdict's record a line of its `VERDICTS_FILE`, and a verdict on one piece and
    control that two records hold is refused."""

    if not labelled:
        return None
    held: dict[tuple, tuple[str, dict]] = {}
    runs, refused = [], []
    for run, superseded in labelled.items():
        at, why = None, None
        if superseded:
            why = f"superseded by {', '.join(superseded)}"
        else:
            at, _, why = _run_files(store, run, (VERDICTS_FILE,))
            why = why or (None if at else f"holds no verdicts (`{VERDICTS_FILE}`)")
        if why or at is None:
            refused.append({"target": run, "record": VERDICTS, "why": why})
            continue
        for n, line in enumerate((at / VERDICTS_FILE).read_text().splitlines(), 1):
            if not line.strip():
                continue
            where = f"{run}'s `{VERDICTS_FILE}` line {n}"
            try:
                v = verdict(json.loads(line), where)
            except json.JSONDecodeError as e:
                raise ValueError(f"{where} is no JSON object: {e}") from e
            key = (v["piece"], v["control"])
            if key in held:
                other = held[key][0]
                raise ValueError(f"{run} holds two verdicts on {_what(v)}" if other == run else
                                 f"{other} and {run} both hold a verdict on {_what(v)} and neither is superseded: label "
                                 f"the one the other replaces ({_label(SUPERSEDED_BY, 'NEWER')})")
            held[key] = (run, v)
        runs.append(run)
    order = list(PIECES)
    return sorted((v for _, v in held.values()), key=lambda v: (order.index(v["piece"]), v["control"] or "")), runs, refused


def _by(v: Mapping) -> str:
    return f"`{v['verifier']['module']}` at {v['verifier']['commit']}"


def _read(v: Mapping) -> str:
    return f"on {v['run']}, withheld: {v['open']}" if v["verdict"] == "withheld" else f"is `{v['verdict']}` on {v['run']}"


def stop_class(verdicts: Sequence[Mapping], runs: Sequence[str]) -> dict:
    """Whether the stop class's ``verdicts`` (`stop_verdicts`) put a run behind the output count's 0 in the sums that
    take it at 0 (`complete`; which sums those are is the reader's answer, never this): each of `PIECES` accepted
    and complete on its honest run, and
    each of `CONTROLS` refused, on a piece, by the verifier that accepted that piece, and none accepted; a `withheld`
    verdict is neither, so it keeps the class short of complete and names what it was withheld for, and a control on a
    withheld piece refused by that piece's verifier counts as refused.

    The four honest verdicts are of one class only where each names it (`CLASS_FIELDS`) with its verifier's recorded
    check that its stream's registration names the class's boundary true (`stream_names_boundary`, which this never
    recomputes), they agree on `AGREED`, and their rate is the row's, `RATE`; otherwise `lacks` names the verdict that
    lacks a field or records a failed check, or the field the verdicts disagree on with each value's pieces. A control
    is held to its piece and its verifier (module and commit) alone, never to the class.

    Where it is complete, `class` is what the verdicts agree on and `mark` is what the output count's 0 is marked:
    `measured` only where every piece's record names the Lean guarantee that proves its verifier (`proved_by`), else
    `RUN_NOT_PROVED`. Where it is not, `lacks` names what is missing, `class` and the mark are None and the output count
    keeps its own. `said` is what the print says of the class."""

    honest = {v["piece"]: v for v in verdicts if v["control"] is None}
    lacks = []
    for piece, what in PIECES.items():
        if piece not in honest:
            lacks.append(f"no verdict on {what}")
        elif (v := honest[piece])["verdict"] != "accept":
            lacks.append(f"{what} {_read(v)}")
    named = [(PIECES[piece], honest[piece]) for piece in PIECES if piece in honest]
    for what, v in named:
        if lack := [f"`{k}`" for k in CLASS_FIELDS if k not in v]:
            lacks.append(f"{what}'s verdict on {v['run']} lacks {', '.join(lack)}")
        elif v["stream_names_boundary"] is not True:
            lacks.append(f"{what}'s verdict on {v['run']} records that its stream's registration, "
                         f"`{v['stream_registration']}`, doesn't name the class's boundary (`stream_names_boundary` false)")
    for k in AGREED:
        held: dict[str, list[str]] = {}
        for what, v in named:
            if k in v:
                held.setdefault(v[k], []).append(what)
        if len(held) > 1:
            lacks.append(f"the honest verdicts disagree on `{k}`: "
                         + _listed([f"`{x}` ({_listed(pieces)})" for x, pieces in held.items()]))
        elif k == "rate" and held and (rate := next(iter(held))) != RATE:
            lacks.append(f"the honest verdicts are at rate {rate}, and the row assumes {RATE}: every stop and tie block "
                         "proved")
    for control, what in CONTROLS.items():
        tried = [v for v in verdicts if v["control"] == control]
        if not tried:
            lacks.append(f"no negative control with {what}")
        for v in tried:
            accepted = honest.get(v["piece"])
            if v["verdict"] != "refuse":
                lacks.append(f"the control with {what} on {PIECES[v['piece']]} {_read(v)}, not refused")
            elif (accepted is None or accepted["verdict"] not in ("accept", "withheld")
                  or accepted["verifier"] != v["verifier"]):
                lacks.append(f"the control with {what} is refused on {v['run']} by {_by(v)}, not by a verifier that "
                             f"accepted the honest run's verdict on {PIECES[v['piece']]}")
    out = {"record": VERDICTS, "runs": list(runs), "complete": not lacks, "mark": None, "class": None, "lacks": lacks,
           "verdicts": list(verdicts)}
    if lacks:
        return out | {"said": "not complete: " + "; ".join(lacks)}
    one = {k: honest["stop_record"][k] for k in AGREED}
    on = f"complete on class `{one['class']}` of serve {one['serve']} at rate {one['rate']}"
    unproved = [piece for piece in PIECES if honest[piece]["proved_by"] is None]
    if not unproved:
        return out | {"mark": "measured", "class": one, "said": f"{on} (measured: proved by "
                      + _listed([f"`{g}`" for g in dict.fromkeys(honest[p]["proved_by"] for p in PIECES)]) + ")"}
    return out | {"mark": RUN_NOT_PROVED, "class": one,
                  "said": f"{on} (run, not proved: {_listed(list(dict.fromkeys(honest[p]['open'] for p in unproved)))}"
                          f"; Python verifier for {_listed([PIECES[p] for p in unproved])})"}
