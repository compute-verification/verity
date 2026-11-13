"""The stop class's verdict records of a served run, and how they charge the rows that read them (`stop`).

`verity bench certifier_term`'s rows that hold the lengths by a stop block charge their stop class below its
schedule only as a served run's verdict records say. This module reads those records and the signed heads they
name, asks the network certifier's own rule (`class_charge.charge`) how each class of each head is charged, holds
what the rule charges below to two things the rule doesn't read (a negative control for each verifier that
accepted, and the rate the serve registered; both below), and hands the calculator the answer (`leak_bounds.Stop`).
Nothing but a run's files moves the answer.

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
  is read from the head;
- its control records, in every file named as `CONTROLS_FILES`: an object with the fields `CONTROL_FIELDS` ("A
  negative control", below);
- `SERVES_FILE`: each serve's registration that its verdict records name by digest (`digest_of`), which states the
  rate each sampled population is registered at ("The rate", below).

**A superseded run** adds no accept and takes away no refusal. The caller hands `read` every run that carries the
record and says which of them another run replaces. A superseded run's files are read as any run's, but for its
verdict records that are accepted and complete, and its control records that are refused, which count for nothing:
nothing retries a verdict, so what a run refused on a head stands whatever is labelled later, as does an accept of
its that is not complete, a control it did not refuse, a record of its that doesn't read and a registration it
kept. A record counts as a superseded run's where that run's files hold it or it names that run as the one that
produced it. So superseding a run never raises what is held. A superseded run that is not among the runs handed
holds everything at its whole schedule: what it refused was not read.

**A head** (`head_of`) is the registration's own: its link, execution phase, parameters digest and tables as listed,
held to the digest its certificate names. No signature is checked here: the certificate verifies under the network
certifier's key and session, which a calculator doesn't hold, so a head read here is as good as the run that kept it.
A verdict's own verifier checks the certificate before it accepts.

**The answer** (`stop`) counts the execution phases that are held: those whose head was read, in which some class
holds a count above 0 and every class with a count above 0 is charged below its schedule by the rule, at the
registered rate ("The rate"), each of its verdicts by a verifier with its control ("A negative control"). One
class at its whole schedule keeps its execution phase at its whole schedule, and each line says which verdict it
lacks.
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
records name (`ONE`).

**A negative control.** An execution phase is held only where each verifier that accepted a verdict counted in it
has, at the same commit, a control of record that it refused: a run on the same path with one thing changed, the
thing that verifier checks (`CONTROLS`). The guarantee rests on the verifier's code and not on a control; a control
is what shows that the verifier which ran refuses what it must, so that its accept is no pass for the wrong reason.
A control record names the run that produced it, its verifier's module and commit, what was changed (``control``)
and the verifier's own verdict, whose ``format`` says which of the four it is a control of. A verifier's refusal at
a commit is a fact of that code, so one control a verifier a commit serves every execution phase checked under that
commit, and a control names no head, class or serve. A control at another commit, by another verifier, or of a
thing its verifier doesn't owe counts for nothing. One its verifier did not refuse (``accepted`` anything but
false) leaves that verifier without its control at that commit, whatever other control it refused. An execution
phase whose verifier lacks its control is charged its whole schedule, with a line that names the verifier. A
control record that doesn't read holds no execution phase at all. Of a superseded run, a control that was refused
counts for nothing and one that was not refused stands, as its verdicts do.

**The rate.** The stop record's and the response tie's populations, and the rate each is proved at, are the serve's
to register before any coin, and a class is held only where every block of both is proved: the registration its
verdicts name states `EVERY_BLOCK` for both (`RATES`), and each stop record's and each response tie's record says
it ran at the registered rate (`RATE`). The rate is the registration's and never the verdict's own word: the
reader is handed the registration itself (`SERVES_FILE`), finds it by the digest the class's records name
(`digest_of`) and reads both rates from it. A registration that was not handed, one that states no rate for a
population, one that registers another rate, and a verdict that names no rate or another than the registered one
each leave the class at its whole schedule, with a line that names which; so a verdict that states `EVERY_BLOCK`
under a registration that says otherwise is refused by name. A verdict's ``complete`` says that every session ran
to a verdict and holds no class to a rate. Where a registration states the two rates is its writer's to name, and
`RATES` is the one place that holds it. A serve's registration that doesn't read holds no execution phase at all."""
from __future__ import annotations

import fnmatch
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import leak_bounds as LB
from verity.protocols.certifier.verifier import class_charge as CC
from verity.protocols.certifier.verifier import connection_rows as C
from verity.protocols.certifier.verifier import content_rows as CT
from verity.protocols.spot.verifier.one_stage import registration as R

#: the record, in the evidence store, of the runs that hold the stop class's verdict records
RECORD = "exfiltration-glm53-stop-verdicts"
#: a run's files that are read: its verdict records and its control records, in every file so named, the
#: registrations of connection rows its verdicts name, and the serves' registrations its records name
RECORDS_FILES, REGISTRATIONS_FILE = "out/*-verdicts.jsonl", "out/registrations.jsonl"
CONTROLS_FILES, SERVES_FILE = "out/*-controls.jsonl", "out/serve-registrations.jsonl"
FILES = (RECORDS_FILES, REGISTRATIONS_FILE, CONTROLS_FILES, SERVES_FILE)
#: which of the four verdicts a record is, or a control record is a control of, by the ``format`` of its verdict
FORMAT = "format"
FORMATS = {"verity-network-certifier/response-tie-verdict/v1": CC.RESPONSE_TIE}
#: a control record's fields: the run that produced it, the verifier that judged it, what was changed, and the
#: verifier's own verdict on it
CONTROL = "control"
CONTROL_FIELDS = (CC.RUN, CC.VERIFIER, CONTROL, CC.VERDICT)
#: the negative control each verdict's verifier owes, as a control record names it, and what it changes
CONTROLS = {CC.STOP_RECORD: ("stop_word", "a stop word changed, the count off by one"),
            CC.RESPONSE_TIE: ("id", "a released id changed"),
            CC.CERTIFIED_READ: ("max_tokens", "a request word changed, `max_tokens`"),
            CC.COMPUTE_AUDIT: ("output", "a committed output changed")}
#: the rate a record says its verdict ran at, and the one rate at which a class is held: every block proved
RATE, EVERY_BLOCK = "rate", "rate:1/1"
#: where a serve's registration states the rate each sampled population is registered at, by the verdict over that
#: population: the keys from the registration down to the rate. The names are the registration's writer's to give
RATES = {CC.STOP_RECORD: ("rates", "stop_record"), CC.RESPONSE_TIE: ("rates", "response_tie")}
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


def _named(x: object) -> bool:
    return type(x) is str and bool(x)


def control_lacks(record: object) -> str | None:
    """What keeps ``record`` from being a control record (the module docstring's "A negative control"), by the
    field's name, or None where it is one."""
    if not isinstance(record, Mapping):
        return "it is no object"
    if missing := [k for k in CONTROL_FIELDS if k not in record]:
        return f"it has no `{'`, `'.join(missing)}`"
    for key in (CC.RUN, CONTROL):
        if not _named(record[key]):
            return f"its `{key}` is no name"
    verifier = record[CC.VERIFIER]
    if unnamed := [k for k in CC.NAMES if not (isinstance(verifier, Mapping) and _named(verifier.get(k)))]:
        return f"its `{CC.VERIFIER}` names no `{'`, `'.join(unnamed)}`"
    if not isinstance(record[CC.VERDICT], Mapping):
        return f"its `{CC.VERDICT}` is no object"
    return None


#: what a control record is, as a refusal of one that is none says it
A_CONTROL = (f"a control record is an object with the fields {', '.join(CONTROL_FIELDS)}, its run and its control "
             f"named, its verifier by its {' and '.join(CC.NAMES)}, and its verdict an object")


def digest_of(registration: object) -> str:
    """The digest a verdict record names a serve's registration by: the registration record's own
    (`one_stage.registration.digest`). ValueError for one that is no object."""
    if not isinstance(registration, Mapping):
        raise ValueError(f"a serve's registration doesn't read: it is no object: {registration!r}")
    try:
        return R.digest(registration)
    except (TypeError, ValueError) as e:
        raise ValueError(f"a serve's registration doesn't read: {e!r}") from None


def _kind(verdict: Mapping[str, Any]) -> str | None:
    return FORMATS.get(verdict.get(FORMAT)) if isinstance(verdict.get(FORMAT), str) else None


def _rate_at(registration: Mapping[str, Any], path: Sequence[str]) -> str | None:
    at: Any = registration
    for key in path:
        at = at.get(key) if isinstance(at, Mapping) else None
    return at if _named(at) else None


def _unrated(c: CC.ClassCharge, served: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """What keeps a class the rule charges below from being held, by the rate (the module docstring's "The rate"):
    empty where the registration its records name was handed and registers `EVERY_BLOCK` for both populations, and
    each stop record and response tie counted says it ran at that."""
    name = c.read[0][1][CC.REGISTRATION]
    if name not in served:
        return [f"the registration its verdicts name, `{name}`, is not among the serves' registrations read "
                f"(`{SERVES_FILE}`), so the rate it registers is not known"]
    lacks = []
    for kind, path in RATES.items():
        registered = _rate_at(served[name], path)
        if registered is None:
            lacks.append(f"the registration `{name}` states no rate for the {kind} (`{'.'.join(path)}`)")
            continue
        if registered != EVERY_BLOCK:
            lacks.append(f"the registration registers the {kind} at `{registered}`, not `{EVERY_BLOCK}`")
        for of, record in c.read:
            if of == kind and record.get(RATE) != registered:
                ran = record.get(RATE)
                lacks.append(f"the {kind} of {record[CC.RUN]} "
                             + (f"states `{ran}`" if _named(ran) else f"names no rate (`{RATE}`)")
                             + f" under a registration that registers `{registered}`")
    return lacks


def _uncontrolled(kind: str, record: Mapping[str, Any],
                  owed: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]]) -> str | None:
    """What the verifier of a verdict counted lacks of its negative control (the module docstring's "A negative
    control"), or None where it has one of record at its commit and refused every one."""
    module, commit = (record[CC.VERIFIER][k] for k in CC.NAMES)
    name, what = CONTROLS[kind]
    who = f"the {kind}'s verifier {module} at {commit}"
    its = owed.get((kind, module, commit), ())
    if not its:
        return f"{who} has no negative control of record at that commit (`{name}`: {what})"
    if kept := sorted({r[CC.RUN] for r in its if r[CC.VERDICT].get(CC.ACCEPTED) is not False}):
        return f"{who} did not refuse its negative control (`{name}`: {what}) on {', '.join(kept)}"
    return None


def _handed(kind: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """A verdict read as the calculator prints it: which of the four it is, what it is of and how it went, its run,
    serve, registration, verifier and proof as its record names them, the rate it says it ran at, and the mark."""
    return {"verdict": kind, **{k: record[CC.VERDICT].get(k) for k in CC.READ},
            **{k: record.get(k) for k in CC.FIELDS if k != CC.VERDICT}, RATE: record.get(RATE),
            "mark": LB.RUN_NOT_PROVED}


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
         left: Sequence[str] = (), controls: Iterable[Mapping[str, Any]] = (),
         served: Iterable[Mapping[str, Any]] = ()) -> LB.Stop:
    """How the stop class is charged (the module docstring's "The answer"), from registrations, verdict records,
    control records and serves' registrations as `read` returns them; ``left`` are lines on what was not read,
    which the answer carries."""
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
        kind = _kind(record[CC.VERDICT])
        if kind is None:
            unlisted += 1
        else:
            by_verdict.setdefault(kind, []).append(record)
    # the controls each verdict's verifier owes, by verdict, module and commit
    owed: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    stray = 0
    for record in controls:
        if lack := control_lacks(record):
            return _whole(f"{A_CONTROL}; of one read, {lack}: {record!r}", *left)
        kind = _kind(record[CC.VERDICT])
        if kind is None or record[CONTROL] != CONTROLS[kind][0]:
            stray += 1
        else:
            owed.setdefault((kind, *(record[CC.VERIFIER][k] for k in CC.NAMES)), []).append(record)
    by_digest: dict[str, Mapping[str, Any]] = {}
    try:
        for doc in served:
            by_digest[digest_of(doc)] = doc
        phases = [(head, [CC.charge(head, cls, by_verdict) for cls in classes(head)])
                  for _, head in sorted(heads.items())]
    except ValueError as e:
        return _whole(str(e), *left)
    charges = [(head, c) for head, its in phases for c in its]
    lines: list[str] = []
    # an execution phase is ruled below where a class of it holds a row and each class that holds one is charged
    # below by the rule; it is held where it is also at the registered rate and each verifier has its control
    held = ruled = by_rate = by_control = 0
    for head, its in phases:
        at = f"the link {head.link}, execution phase {head.window}"
        lines += [f"{at}, {c.why()}" for c in its]
        rows = [c for c in its if c.count]
        if not rows or not all(c.below for c in rows):
            continue
        ruled += 1
        rates = [f"{at}, class {c.cls}: held to its whole schedule by the rate: {'; '.join(lacks)}"
                 for c in rows if (lacks := _unrated(c, by_digest))]
        wants = list(dict.fromkeys(lack for c in rows for kind, record in c.read
                                   if (lack := _uncontrolled(kind, record, owed))))
        lines += rates + ([f"{at}: held to its whole schedule by a control: {'; '.join(wants)}"] if wants else [])
        by_rate, by_control, held = by_rate + bool(rates), by_control + bool(wants), held + (not rates and not wants)
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
    elif not ruled:
        lacking = sum(1 for c in with_rows if not c.below)
        first = (f"each connection is charged its whole schedule: {lacking} of the {len(with_rows)} class(es) that "
                 f"hold a row in the {of} lack a verdict, one in each execution phase that holds a row")
    elif not held:
        back = [f"{n} by {what}" for n, what in (
            (by_rate, f"the rate, `{EVERY_BLOCK}` as registered and as run"),
            (by_control, "a negative control, one refused by each verifier that accepted, at its commit")) if n]
        first = (f"each connection is charged its whole schedule: {ruled} of the {of} have the {len(CC.VERDICTS)} "
                 f"verdicts of each class that holds a row, and none of them is held: {' and '.join(back)}")
    else:
        first = (f"{held} of the {of} are charged below their schedule: each class that holds a row in one of them "
                 f"has its {len(CC.VERDICTS)} verdicts, accepted and complete, at the registered rate "
                 f"`{EVERY_BLOCK}`, each by a verifier that refused its negative control. Every other execution "
                 "phase of a monitoring period is charged its whole schedule, one no registration was read for "
                 "among them")
    other = len(records) - unlisted - len(counted)
    return LB.Stop(held, (
        first, *lines,
        *([f"{other} verdict record(s) read are on no head read, or on no class of one that holds a row, and count "
           "for nothing"] if other else []),
        *([f"{unlisted} verdict record(s) read are of a format listed for none of the {len(CC.VERDICTS)} verdicts, "
           "and count for nothing"] if unlisted else []),
        *([f"{stray} control record(s) read are of a format listed for none of the {len(CC.VERDICTS)} verdicts, or "
           "are not the control their verdict's verifier owes, and count for nothing"] if stray else []),
        *left), tuple(counted))


def _lines(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _adds(record: Any) -> bool:
    """Whether a record counts toward a charge below: a verdict record that is accepted and complete."""
    return (CC.reads(record) and record[CC.VERDICT].get(CC.ACCEPTED) is True
            and record[CC.VERDICT].get(CC.COMPLETE) is True)


def _refuses(record: Any) -> bool:
    """Whether a record counts toward a verifier's control: a control record whose verdict is refused."""
    return control_lacks(record) is None and record[CC.VERDICT].get(CC.ACCEPTED) is False


def read(store: Any, runs: Iterable[str], superseded: Iterable[str] = ()
         ) -> tuple[list[Any], list[Any], list[str], list[Any], list[Any]]:
    """The registrations and the verdict records in the run files of ``runs``, in the runs' order and each run's
    files by name; a line for each run that is not in the store, kept no run files or holds no such file; and the
    runs' control records and serves' registrations: `stop`'s arguments, in its order.
    ``superseded`` are the runs among them that another run replaces: a verdict record of one that is accepted and
    complete is left out, and so is a control record of one that is refused, each with a line that counts them (the
    module docstring's "A superseded run" and "A negative control").
    ``store`` is the evidence store as the harness reads it (`get_attempt`, `listing`, `fetch`)."""
    registrations: list[Any] = []
    records: list[Any] = []
    controls: list[Any] = []
    served: list[Any] = []
    left: list[str] = []
    superseded = set(superseded)
    dropped = uncounted = 0
    for run in runs:
        attempt = store.get_attempt(run)
        out = (attempt or {}).get("outputs") or {}
        files = out.get("run_files") or out.get("run_record")
        held = sorted(row["path"] for row in store.listing(files, list(FILES))) if files else []
        if not held:
            left.append(f"{run} is not read: it " + ("is not in the store or on its remote" if attempt is None else
                                                      "kept no run files" if not files else
                                                      f"holds none of {', '.join(f'`{name}`' for name in FILES)}"))
            continue
        at = Path(store.fetch(files, paths=held))
        for path in held:
            if path in (REGISTRATIONS_FILE, SERVES_FILE):
                (registrations if path == REGISTRATIONS_FILE else served).extend(_lines(at / path))
                continue
            control = fnmatch.fnmatchcase(path, CONTROLS_FILES)
            for record in _lines(at / path):
                named = record.get(CC.RUN) if isinstance(record, Mapping) else None
                old = run in superseded or (isinstance(named, str) and named in superseded)
                if old and (_refuses(record) if control else _adds(record)):
                    dropped, uncounted = dropped + (not control), uncounted + control
                else:
                    (controls if control else records).append(record)
    of = f"of the superseded run(s) {', '.join(sorted(superseded))}"
    if dropped:
        left.append(f"{dropped} verdict record(s) {of} are accepted and complete, and count for nothing: a "
                    "superseded run adds no accept, and what it refused stands")
    if uncounted:
        left.append(f"{uncounted} control record(s) {of} are refused, and count for nothing: a superseded run adds "
                    "no control, and one it did not refuse stands")
    return registrations, records, left, controls, served


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
