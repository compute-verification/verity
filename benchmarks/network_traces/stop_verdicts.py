"""The stop class's verdict records of a served run, and how they charge the rows that read them (`stop`).

`verity bench certifier_term`'s rows that hold the lengths by a stop block charge their stop class below its
schedule only as a served run's verdict records say. This module reads those records and the signed heads they
name, asks the network certifier's own rule (`class_charge.charge`) how each class of each head is charged, and
hands the calculator the answer (`leak_bounds.Stop`). It decides nothing itself, and nothing but the records moves
the answer.

**Where they are.** In the evidence store, in the run files of the runs that carry `run_of_record=RECORD`, which the
caller names, every one of them, with those another run replaces (`read`'s ``runs`` and ``superseded``; the label's
readers are the harness's). A run holds, each one JSON object a line:

- its verdict records, in every file named as `RECORDS_FILES`: an object with the fields `class_charge.FIELDS`, as
  the verdict's own verifier writes it, naming its run, the served run its class came from, the digest of what that
  serve registered, and its verifier's module and commit (`class_charge.reads`; a record that lacks one doesn't
  read, whatever its format, and the line says which it lacks). Which of the four verdicts a record is, is read from
  its verdict's ``format`` (`FORMATS`). A verdict whose verifier's format is not listed there reads as missing, and
  a record of such a format is counted in a line and charges nothing;
- `REGISTRATIONS_FILE`: the registration of each execution phase's connection rows that its verdicts were decided
  under, as `connection_rows.registration` writes it. A record names its head by digest alone, and a class's count
  is read from the head.

**A superseded run** adds no accept and takes away no refusal. The caller hands `read` every run that carries the
record and says which of them another run replaces. A superseded run's files are read as any run's, but for its
verdict records that are accepted and complete, which count for nothing: nothing retries a verdict, so what a run
refused on a head stands whatever is labelled later, as does an accept of its that is not complete, a record of its
that doesn't read and a registration it kept. A record counts as a superseded run's where that run's files hold it
or it names that run as the one that produced it. So superseding a run never raises what is held. A superseded run
that is not among the runs handed holds everything at its whole schedule: what it refused was not read.

**A head** (`head_of`) is the registration's own: its link, execution phase, parameters digest and tables as listed,
held to the digest its certificate names. No signature is checked here: the certificate verifies under the network
certifier's key and session, which a calculator doesn't hold, so a head read here is as good as the run that kept it.
A verdict's own verifier checks the certificate before it accepts.

**The answer** (`stop`) counts the execution phases that are held: those whose head was read, in which some class
holds a count above 0 and every class with a count above 0 is charged below its schedule by the rule. One class at
its whole schedule keeps its execution phase at its whole schedule, and each line says which verdict it lacks.
Every other execution phase is charged its whole schedule, one whose registration no run kept among them: the
calculator counts a row's execution phases from its monitoring period (`leak_bounds.Stop.share`), never from the
registrations read here. So a run's files with an execution phase left out charge what they would with that
execution phase refused, and since nothing but verdicts on a head's digest holds an execution phase, a
registration nobody signed holds none. A verdict is held to a head by the digest it names, so one decided under
another head, another serve's among them, counts for nothing. Registrations or records that don't read, and two
heads given for one execution phase of a link, hold no execution phase at all. Each verdict that is on a head read
is handed on under the mark `leak_bounds.RUN_NOT_PROVED`, whatever its ``proved_by`` says: a run is no proof, and
what a record's ``proved_by`` is worth is its reader's to check.

**Not one class.** The four verdicts count only as one class's: of one class, in one execution phase of a link, on
one signed head, and of one serve under one registration, as their records name the two (`class_charge`'s "One
serve, one registration"). Where each of the four was read and no class that holds a row has them so, the answer
holds nothing and its first line says `NOT_ONE_CLASS` and what the four differ in: where no class has all four on
its head, the fields of a verdict that say where it is (`DIFFER`); where one has, the serve or the registration its
records name (`ONE`)."""
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
#: a run's files that are read: its verdict records, in every file so named, and the registrations they name
RECORDS_FILES, REGISTRATIONS_FILE = "out/*-verdicts.jsonl", "out/registrations.jsonl"
#: which of the four verdicts a record is, by the ``format`` of its verdict
FORMAT = "format"
FORMATS = {"verity-network-certifier/response-tie-verdict/v1": CC.RESPONSE_TIE}
#: what the answer's first line says where each of the four verdicts was read and they are not one class's
NOT_ONE_CLASS = "not one class: unchecked"
#: where a verdict says it is, and what four verdicts that are not one class's can differ in, each with its fields:
#: of a verdict where no class has all four on its head, and of a record (`class_charge.ONE`) where one has
PLACE = (CC.LINK, CC.WINDOW, CC.CLASS, CC.HEAD)
DIFFER = (((CC.CLASS,), "the class (`class`)"),
          ((CC.LINK, CC.WINDOW), "the execution phase (`link`, `window`)"),
          ((CC.HEAD,), "the signed head (`head`) of the network certifier's rows"))
ONE = {CC.SERVE: "the serve (`serve`)", CC.REGISTRATION: "the registration (`registration`)"}


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
    return LB.Stop(0, (f"each connection is charged its whole schedule: {why}", *more))


def _handed(kind: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """A verdict read as the calculator prints it: which of the four it is, what it is of and how it went, its run,
    serve, registration, verifier and proof as its record names them, and the mark."""
    return {"verdict": kind, **{k: record[CC.VERDICT].get(k) for k in CC.READ},
            **{k: record.get(k) for k in CC.FIELDS if k != CC.VERDICT}, "mark": LB.RUN_NOT_PROVED}


def _not_one(by_verdict: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    """What the four verdicts' records differ in, where each verdict has a record and no class that holds a row has
    all four on its head (the module docstring's "Not one class")."""
    named = [[{k: json.dumps(r[CC.VERDICT].get(k), sort_keys=True, default=repr) for k in PLACE} for r in by_verdict[kind]]
             for kind in CC.VERDICTS]

    def common(keys: Sequence[str]) -> set[tuple[str, ...]]:
        return set.intersection(*({tuple(p[k] for k in keys) for p in its} for its in named))

    if common(PLACE):
        return "they name one class on one head, which is no class that holds a row on a head read"
    differ = [what for keys, what in DIFFER if not common(keys)]
    return ("they name no one class of one execution phase on one head"
            + (f", and differ in {' and in '.join(differ)}" if differ else ""))


def stop(registrations: Iterable[Mapping[str, Any]], records: Iterable[Mapping[str, Any]],
         left: Sequence[str] = ()) -> LB.Stop:
    """How the stop class is charged (the module docstring's "The answer"), from registrations and verdict records
    as `read` returns them; ``left`` are lines on what was not read, which the answer carries."""
    registrations, records = list(registrations), list(records)
    if not registrations and not records:
        return LB.Stop(0, (*LB.NO_RECORDS.why, *left))
    if not registrations:
        return _whole(f"{len(records)} verdict record(s) were read and no registration (`{REGISTRATIONS_FILE}`), so "
                      "no class's count is known", *left)
    heads: dict[tuple[str, int], C.Head] = {}
    for doc in registrations:
        try:
            head = head_of(doc)
        except ValueError as e:
            return _whole(str(e), *left)
        if heads.setdefault((head.link, head.window), head) != head:
            return _whole(f"two heads are given for execution phase {head.window} of the link {head.link}", *left)
    by_verdict: dict[str, list[Mapping[str, Any]]] = {}
    unlisted = 0
    for record in records:
        if lack := CC.lacks(record):
            return _whole(f"{CC.A_RECORD}; of one read, {lack}: {record!r}", *left)
        verdict = record[CC.VERDICT]
        kind = FORMATS.get(verdict.get(FORMAT)) if isinstance(verdict.get(FORMAT), str) else None
        if kind is None:
            unlisted += 1
        else:
            by_verdict.setdefault(kind, []).append(record)
    try:
        phases = [(head, [CC.charge(head, cls, by_verdict) for cls in classes(head)])
                  for _, head in sorted(heads.items())]
    except ValueError as e:
        return _whole(str(e), *left)
    charges = [(head, c) for head, its in phases for c in its]
    lines = [f"the link {head.link}, execution phase {head.window}, {c.why()}" for head, c in charges]
    # an execution phase is held where a class of it holds a row and each class that holds one is charged below
    held = sum(1 for _, its in phases if any(c.count for c in its) and all(c.below for c in its if c.count))
    with_rows = [c for _, c in charges if c.count]
    # the classes that hold a row and have a record of each of the four verdicts on their head
    all_four = [c for c in with_rows if {kind for kind, _ in c.read} >= set(CC.VERDICTS)]
    counted = [_handed(kind, record) for _, c in charges for kind, record in c.read]
    of = f"{len(heads)} execution phase(s) read"
    if not with_rows:
        first = f"each connection is charged its whole schedule: no class holds a row in the {of}"
    elif all(by_verdict.get(kind) for kind in CC.VERDICTS) and not any(not c.differ for c in all_four):
        # each of the four was read, and no class has them as one class's: on its head, of one serve and registration
        if all_four:
            differ = [what for key, what in ONE.items() if any(key == k for c in all_four for k, _ in c.differ)]
            how = (f"has all of them on its head as one serve's under one registration: they differ in "
                   f"{' and in '.join(differ)}")
        else:
            how = f"has all of them on its head: {_not_one(by_verdict)}"
        first = (f"each connection is charged its whole schedule: {NOT_ONE_CLASS}: each of the {len(CC.VERDICTS)} "
                 f"verdicts was read, and no class that holds a row in the {of} {how}")
    elif not held:
        lacking = sum(1 for c in with_rows if not c.below)
        first = (f"each connection is charged its whole schedule: {lacking} of the {len(with_rows)} class(es) that "
                 f"hold a row in the {of} lack a verdict, one in each execution phase that holds a row")
    else:
        first = (f"{held} of the {of} are charged below their schedule: each class that holds a row in one of them "
                 f"has its {len(CC.VERDICTS)} verdicts, accepted and complete. Every other execution phase of a "
                 "monitoring period is charged its whole schedule, one no registration was read for among them")
    other = len(records) - unlisted - len(counted)
    return LB.Stop(held, (
        first, *lines,
        *([f"{other} verdict record(s) read are on no head read, or on no class of one that holds a row, and count "
           "for nothing"] if other else []),
        *([f"{unlisted} verdict record(s) read are of a format listed for none of the {len(CC.VERDICTS)} verdicts, "
           "and count for nothing"] if unlisted else []), *left), tuple(counted))


def _lines(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _adds(record: Any) -> bool:
    """Whether a record counts toward a charge below: a verdict record that is accepted and complete."""
    return (CC.reads(record) and record[CC.VERDICT].get(CC.ACCEPTED) is True
            and record[CC.VERDICT].get(CC.COMPLETE) is True)


def read(store: Any, runs: Iterable[str], superseded: Iterable[str] = ()) -> tuple[list[Any], list[Any], list[str]]:
    """The registrations and the verdict records in the run files of ``runs``, in the runs' order and each run's
    files by name, and a line for each run that is not in the store, kept no run files or holds no such file.
    ``superseded`` are the runs among them that another run replaces: a record of one that is accepted and complete
    is left out, with a line that counts them (the module docstring's "A superseded run").
    ``store`` is the evidence store as the harness reads it (`get_attempt`, `listing`, `fetch`)."""
    registrations: list[Any] = []
    records: list[Any] = []
    left: list[str] = []
    superseded = set(superseded)
    dropped = 0
    for run in runs:
        attempt = store.get_attempt(run)
        out = (attempt or {}).get("outputs") or {}
        files = out.get("run_files") or out.get("run_record")
        held = sorted(row["path"] for row in store.listing(files, [RECORDS_FILES, REGISTRATIONS_FILE])) if files else []
        if not held:
            left.append(f"{run} is not read: it " + ("is not in the store or on its remote" if attempt is None else
                                                      "kept no run files" if not files else
                                                      f"holds neither `{RECORDS_FILES}` nor `{REGISTRATIONS_FILE}`"))
            continue
        at = Path(store.fetch(files, paths=held))
        for path in held:
            if path == REGISTRATIONS_FILE:
                registrations.extend(_lines(at / path))
                continue
            for record in _lines(at / path):
                named = record.get(CC.RUN) if isinstance(record, Mapping) else None
                if (run in superseded or (isinstance(named, str) and named in superseded)) and _adds(record):
                    dropped += 1
                else:
                    records.append(record)
    if dropped:
        left.append(f"{dropped} verdict record(s) of the superseded run(s) {', '.join(sorted(superseded))} are "
                    "accepted and complete, and count for nothing: a superseded run adds no accept, and what it "
                    "refused stands")
    return registrations, records, left


def of_runs(store: Any, runs: Iterable[str], superseded: Iterable[str] = ()) -> LB.Stop:
    """How the stop class is charged by the records ``runs`` hold in ``store`` (`read`, `stop`): ``runs`` is every
    run that carries the record, and ``superseded`` those of them that another run replaces (the module docstring's
    "A superseded run"). With no run it is its whole schedule, and the line says no run carries the record; so it
    is with a superseded run that is not among ``runs``, whose refusals were not read."""
    runs, superseded = list(runs), list(superseded)
    if not runs:
        return LB.Stop(0, (*LB.NO_RECORDS.why, f"no run in the store carries `run_of_record={RECORD}`"))
    if unread := sorted(set(superseded) - set(runs)):
        return _whole(f"the superseded run(s) {', '.join(unread)} are not among the runs handed, so what they "
                      "refused was not read: a superseded run adds no accept and takes away no refusal")
    try:
        return stop(*read(store, runs, superseded))
    except (OSError, ValueError) as e:
        return _whole(f"the records of {', '.join(runs)} don't read: {e}")
