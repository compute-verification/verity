"""The stop class's verdict records of a served run, and how they charge the design's row (`stop`).

`verity bench certifier_term`'s row `design` charges its stop class below its schedule only as a served run's verdict
records say. This module reads those records and the signed heads they are of, asks the network certifier's own rule
(`class_charge.charge`) how each class of each head is charged, and hands the calculator the answer
(`leak_bounds.Stop`). It decides nothing itself, and nothing but the records moves the answer.

**Where they are.** In the evidence store, in the run files of the runs that carry `run_of_record=RECORD`, which the
caller names (`read`'s ``runs``; the label's readers are the harness's). A run holds one or both of:

- `RECORDS_FILE`: one verdict record a line, an object with the fields `class_charge.FIELDS`;
- `HEADS_FILE`: one signed head a line, an object with the serve it is of (``serve``) and the registration of that
  execution phase's connection rows (``registration``), as `connection_rows.registration` writes it.

**A head** (`head_of`) is the registration's own: its link, execution phase, parameters digest and tables as listed,
held to the digest its certificate names. No signature is checked here: the certificate verifies under the network
certifier's key and session, which a calculator doesn't hold, so a head read here is as good as the run that kept it.

**The answer** (`stop`) is below the schedule only where every head read is of one serve, some class of some head
holds a count above 0, and every class with a count above 0, in every head read, is charged below its schedule by
the rule. One class at its whole schedule keeps every connection at its whole schedule, and each line says which
verdict it lacks. Each verdict read is handed on under the mark `leak_bounds.RUN_NOT_PROVED`, whatever its
``proved_by`` says: a run is no proof, and what a record's ``proved_by`` is worth is its reader's to check."""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import leak_bounds as LB
from verity.protocols.certifier.verifier import class_charge as CC
from verity.protocols.certifier.verifier import connection_rows as C
from verity.protocols.certifier.verifier import content_rows as CT

#: the record, in the evidence store, of the runs that hold the stop class's verdict records
RECORD = "exfiltration-glm53-stop-verdicts"
#: a run's files that are read: its verdict records and its signed heads, one JSON object a line
RECORDS_FILE, HEADS_FILE = "stop-verdicts.jsonl", "stop-heads.jsonl"
#: a head line's fields
SERVE, REGISTRATION = "serve", "registration"


def head_of(doc: Mapping[str, Any]) -> C.Head:
    """The head a registration of an execution phase's connection rows lists, held to the digest its certificate
    names, or ValueError naming what doesn't read. Its signature is not checked (the module docstring's "A head")."""
    try:
        if doc["format"] != C.REGISTRATION:
            raise ValueError(f"its format is {doc['format']!r}, not {C.REGISTRATION!r}")
        bits = doc["word_bits"]
        head = C.Head(doc["link"], doc["window"], bytes.fromhex(doc["parameters"]),
                      tuple(C.Entry(t["name"], t["rows"], t["words"] * bits // 8, bytes.fromhex(t["domain"]),
                                    bytes.fromhex(t["root"])) for t in doc["tables"]))
        signed = doc["certificate"]["digest"]
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"a registration of connection rows doesn't read: {e!r}") from None
    if signed != head.digest().hex():
        raise ValueError("the registration's certificate doesn't name its head's digest")
    return head


def classes(head: C.Head) -> list[int]:
    """The classes a head lists, by index: those whose request table it names."""
    return sorted(int(e.name.removeprefix(f"{CT.REQUEST}.")) for e in head.tables
                  if e.name.startswith(f"{CT.REQUEST}.") and e.name.removeprefix(f"{CT.REQUEST}.").isdecimal())


def _whole(why: str, *more: str) -> LB.Stop:
    return LB.Stop(False, (f"each connection is charged its whole schedule: {why}", *more))


def stop(heads: Iterable[Mapping[str, Any]], records: Iterable[Mapping[str, Any]],
         left: Sequence[str] = ()) -> LB.Stop:
    """How the design's stop class is charged (the module docstring's "The answer"), from head lines and verdict
    records as `read` returns them; ``left`` are lines on what was not read, which the answer carries."""
    heads, records = list(heads), list(records)
    if not heads and not records:
        return LB.Stop(False, (*LB.NO_RECORDS.why, *left))
    if not heads:
        return _whole(f"{len(records)} verdict record(s) were read and no signed head (`{HEADS_FILE}`), so no "
                      "class's count is known", *left)
    read: dict[tuple[str, int], tuple[str, C.Head]] = {}
    for line in heads:
        try:
            serve, head = line[SERVE], head_of(line[REGISTRATION])
        except (KeyError, TypeError, ValueError) as e:
            return _whole(f"a head line doesn't read: {e}", *left)
        if type(serve) is not str or not serve:
            return _whole(f"a head line names no serve: {serve!r}", *left)
        if read.setdefault((head.link, head.window), (serve, head)) != (serve, head):
            return _whole(f"two heads are given for execution phase {head.window} of the link {head.link}", *left)
    if len(serves := sorted({serve for serve, _ in read.values()})) != 1:
        return _whole(f"the heads read are of {len(serves)} serves ({', '.join(serves)}), and the verdicts are "
                      "each from a run on one", *left)
    serve, = serves
    try:
        charges = [(head, CC.charge(head, cls, serve, records))
                   for _, (_, head) in sorted(read.items()) for cls in classes(head)]
    except ValueError as e:
        return _whole(str(e), *left)
    lines = [f"the link {head.link}, execution phase {head.window}, {c.why()}" for head, c in charges]
    held = [c for _, c in charges if c.count]
    below = bool(held) and all(c.below for c in held)
    counted: list[Mapping[str, Any]] = []
    for _, c in charges:
        counted += [r for r in c.read if not any(r is seen for seen in counted)]
    rows, of = sum(c.count for c in held), f"{len(read)} execution phase(s) of the serve {serve}"
    if not held:
        first = f"each connection is charged its whole schedule: no class holds a row in the {of} read"
    elif below:
        first = (f"each connection is charged below its schedule: the {rows} row(s) of {len(held)} class(es) in the "
                 f"{of} read each have their {len(CC.VERDICTS)} verdicts, accepted and complete")
    else:
        lacking = sum(1 for c in held if not c.below)
        first = (f"each connection is charged its whole schedule: {lacking} of the {len(held)} class(es) that hold "
                 f"a row in the {of} read lack a verdict")
    other = len(records) - len(counted)
    return LB.Stop(below, (first, *lines,
                           *([f"{other} verdict record(s) read are of no class of a head read on that serve, and "
                              "count for nothing"] if other else []), *left),
                   tuple({**{k: r.get(k) for k in CC.FIELDS}, "mark": LB.RUN_NOT_PROVED} for r in counted), serve)


def _lines(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read(store: Any, runs: Iterable[str]) -> tuple[list[Any], list[Any], list[str]]:
    """The head lines and the verdict records in the run files of ``runs``, in the runs' order, and a line for each
    run that is not in the store, kept no run files or holds neither file. ``store`` is the evidence store as the
    harness reads it (`get_attempt`, `listing`, `fetch`)."""
    heads: list[Any] = []
    records: list[Any] = []
    left: list[str] = []
    for run in runs:
        attempt = store.get_attempt(run)
        out = (attempt or {}).get("outputs") or {}
        files = out.get("run_files") or out.get("run_record")
        held = [name for name in (RECORDS_FILE, HEADS_FILE) if files and store.listing(files, [name])]
        if not held:
            left.append(f"{run} is not read: it " + ("is not in the store or on its remote" if attempt is None else
                                                      "kept no run files" if not files else
                                                      f"holds neither `{RECORDS_FILE}` nor `{HEADS_FILE}`"))
            continue
        at = Path(store.fetch(files, paths=held))
        records += _lines(at / RECORDS_FILE) if RECORDS_FILE in held else []
        heads += _lines(at / HEADS_FILE) if HEADS_FILE in held else []
    return heads, records, left


def of_runs(store: Any, runs: Iterable[str]) -> LB.Stop:
    """How the design's stop class is charged by the records ``runs`` hold in ``store`` (`read`, `stop`). With no
    run it is its whole schedule, and the line says no run carries the record."""
    runs = list(runs)
    if not runs:
        return LB.Stop(False, (*LB.NO_RECORDS.why, f"no run in the store carries `run_of_record={RECORD}`"))
    try:
        return stop(*read(store, runs))
    except (OSError, ValueError) as e:
        return _whole(f"the records of {', '.join(runs)} don't read: {e}")
