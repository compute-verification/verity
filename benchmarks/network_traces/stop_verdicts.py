"""The stop class's verdict records of a served run, and how they charge the rows that read them (`stop`).

`verity bench certifier_term`'s rows that hold the lengths by a stop block charge their stop class below its
schedule only as a served run's verdict records say. This module reads those records and the signed heads they
name, asks the network certifier's own rule (`class_charge.charge`) how each class of each head is charged, holds
what the rule charges below to what the rule doesn't read (a negative control for each verifier that accepted,
the rate the serve registered, the one serve of an execution phase, a verdict under another head and the order of
runs; each below), and hands the calculator the answer (`leak_bounds.Stop`).
Nothing but a run's files moves the answer.

**No schedule.** The rule charges a class below its schedule only where it is handed the schedule the class's head
names, one that can be tied (`class_charge`'s "The schedule's word"). This reader is handed no schedule and asks
the rule without one. So every class that holds a row is at its whole schedule whatever its verdicts say, no
execution phase is held, each class's line says the schedule was not handed, and so does the answer's first line
(`NO_SCHEDULE`). What follows is what the reader holds of a class the rule charges below.

**Where they are.** In the evidence store, in the run files of the runs that carry `run_of_record=RECORD`, which the
caller names, every one of them, with those another run replaces (`read`'s ``runs`` and ``superseded``; the label's
readers are the harness's). A run holds, each one JSON object a line:

- its verdict records, in every file named as `RECORDS_FILES`: an object with the fields `class_charge.FIELDS`, as
  the verdict's own verifier writes it, naming its run, the served run its class came from, the digest of what that
  serve registered, and its verifier's module and commit (`class_charge.reads`; a record that lacks one doesn't
  read, whatever its format, and the line says which it lacks). Which of the four verdicts a record is, is read from
  its verdict's ``format`` (`FORMATS`). A verdict whose verifier's format is not listed there reads as missing, and
  a record of such a format is counted in a line and charges nothing. No format is listed, the response tie's
  among them, since no statement registers what the tie reads: so every verdict reads as missing, and no
  execution phase is charged below its schedule whatever a run's files hold;
- `REGISTRATIONS_FILE`: its first line, of the format `CHECKED`, what the offline check of the network certifier's
  run that wrote the file was handed to check it under (`CHECKED_UNDER`: the SHA-512 of the network certifier's
  public key and the session), then the registration of each execution phase's connection rows that its verdicts
  were decided under,
  as `connection_rows.registration` writes it and as that check kept it ("A head", below). A record names its head
  by digest alone, and a class's count is read from the head;
- its control records, in every file named as `CONTROLS_FILES`: an object with the fields `CONTROL_FIELDS` ("A
  negative control", below);
- `SERVES_FILE`: each registration record its verdict records name by digest (`digest_of`): the one a serve
  registered its class under, and the one of each verdict's own stream, which follows it and registers the rate
  that stream's population is sampled at ("The rate", below).

**A superseded run** adds no accept and takes away no refusal. The caller hands `read` every run that carries the
record and says which of them another run replaces. A superseded run's files are read as any run's, but for its
verdict records that are accepted and complete, which fill no verdict, and its control records that count for
their verifier ("A negative control"), which count for nothing: nothing retries a verdict, so what a run refused on
a head stands whatever is labelled later, as does an accept of its that is not complete, a control it did not
refuse or refused for another reason than the control's word, and a registration it kept; a record of its that
doesn't read stands for every serve it could be of ("The order of runs"). An accept of its is still read on its
head (`stop`'s ``spent``): the serve and
the registration it names, the rate it states and its verifier are held to what holds any record's, so it can keep
a class at its whole schedule, as it did before its run was superseded, and can never help hold one. A record
counts as a superseded run's where that run's files hold it or it names that run as the one that produced it; so
does a verdict record that names that run as the serve it is of. So superseding a run raises what is held only on
a serve launched after that run, and only where the run holds a record that doesn't read. A superseded run that is
not among the runs handed holds everything at its whole schedule: what it refused was not read.

**The order of runs** is read from two run ids (`launched_after`): an id states its launch's date and time, and a
name that is no run id is launched after nothing, and nothing after it. Two rules read it.
A run has no accept on a serve launched after it: a verdict record that names such a serve and is accepted and
complete fills no verdict, as a superseded run's does, and what such a record refused stands. A record's own run
is the one whose files hold it and the one it names as the run that produced it.
A record of a superseded run that doesn't read could have been a refusal, of any serve that run could have
judged. So an execution phase is held only where its serve was launched after every superseded run that holds a
record that doesn't read: after the newest of them, which the line of an execution phase so kept whole names, with
how many such records there are. A bad file is mended by a new serve with new verifier runs and never by a second
run over the old serve, since nothing retries a verdict. Such a record is any line of the run's files read here
that doesn't read, one rule for the four kinds of file and for a line that is no JSON, and a registrations' file
that does not say what it was checked under, which is one such record whatever it holds; a control record names no
serve, so one that doesn't read keeps whole the serves launched at or before its run, and a new control run mends
the serves launched after it. In a run that is not superseded a record that doesn't read holds no execution phase
at all, and a line that is no JSON names its line, its file and its run.
The order is read from run ids, which the team's tooling writes, and that is enough for runs of record here. In a
deployment it comes from the auditor's own ledger, from something the prover cannot write. A serve with no run id
counts as not later, which is the safe side.

**Another head.** A verdict that doesn't count, refused or not complete, counts against the execution phase it
names by its serve, its link and its number, whatever head it names: where the phase's head was read and its
records name that serve, the phase is charged its whole schedule, every class of it, with a line that names the
verdict, the class it names and its run. Refused under one head, registered again under another and accepted
would be a second try at a verdict. It holds every class and not the one it names because two heads need not sort
an execution phase's rows into the same classes, so a refusal on one class under the other head can speak for
rows the head read puts in another. Under the head read a verdict stays its class's, as the rule has it.

**Withheld, and the codes.** A verdict states two fields more, and both names are its verifier's own
(`class_charge.SAID`): ``withheld``, null or the one word for what the verifier held its accept back for, and
``codes``, a list of the verifier's own code words, empty on an accept. The charge reads neither: a verdict counts
only where ``accepted`` is exactly true, and a withheld one leaves its class at its whole schedule as a refused one
does, under the head read and under another. A line says "withheld" with the word of a verdict that is not accepted
and states one, and "refused" where ``withheld`` is null; ``codes`` is handed on as written with each verdict read.
Every verdict states both, whatever its format, a control's among them: a verifier with nothing to say writes
``withheld`` null and ``codes`` empty, so no withheld verdict reads as refused because a line was left out. A
record whose verdict leaves one out or states one wrongly doesn't read, and the line says which field to add
(`class_charge.misstates`): a ``withheld`` that is neither null nor a word, a ``codes`` that is no list of words,
and an accept that states a ``withheld`` word or a code.

**Stopped.** A verifier's command that stopped with an error after its stream's session opened reached no verdict
and was a try at one. Its record states ``accepted`` false, ``withheld`` null, ``codes`` empty and one field more,
``stopped``, the command's own word for the error (`class_charge.STOPPED`). The line says "stopped" with the word,
since no verifier refused; the charge is a refusal's, and it is final as a refusal is: it stands in a superseded
run, and under another head it counts against its execution phase ("Another head"). So a second run over the same
serve, which would fill the verdict the first never reached, holds nothing. A command that stopped before any
challenge was sent writes no record: nothing was tried. ``stopped`` may be absent, which reads as null; it is
handed on with each verdict read. A record doesn't read where its verdict's ``stopped`` is neither null nor a
word, where an accept states one, and where a verdict states a ``withheld`` word or a code too. A command that is
killed writes nothing, and a second run then fills its verdict unseen: only a record of the sessions opened, which
the command does not write itself, closes that, and the reader is handed none.

**A head** (`head_of`) is the registration's own: its link, execution phase, parameters digest and tables as listed,
held to the digest its certificate names. No signature is checked here: the certificate verifies under the network
certifier's key and session, which a calculator doesn't hold. The signature is checked where `REGISTRATIONS_FILE` is
written, by the offline check of the network certifier's run (`warden_serve.py check --registrations`,
`connection_rows_run.check`). The auditor hands that check the SHA-512 of the network certifier's public key and
the session; it holds the run's deployment record to them, verifies each registration under that key and session
(`connection_rows.check_registration`), lays its tables again from the run's own records, and writes the file only
where it found no problem: a first line, of a ``format`` of its own (`CHECKED`), that names the hash and the
session it was handed, then a line for each registration it held to both. The reader reads the first line
(`checked_under`), which is the file's line 1 and no later one. A file whose line 1 is blank, has no ``format`` or
has another, a registration's among them, gives no head: in a run that is not superseded nothing is held and the
line says the file does not say what it was checked under, and in a superseded run the file is one record that
doesn't read ("The order of runs"). The mend is to run that check again, handed the hash and the session. The
answer prints the hash and the session each file names, and the reader compares them with nothing: it is told no
key. The file is unsigned, and the reader can't tell who wrote the one it is handed: a head read here is as good as
the run of that check and as what that run was handed, and a file from any other writer is a head nobody checked.

**An execution phase is one serve's**: a serve, a link and the phase's number. A registration of connection rows
names no serve, so a head's serve is the one its verdict records name (those on its digest, counted or not), which
is a run's word and no proof, and the answer's second line says so (`SERVE_NAMED`). Two heads given for one number
of a link are two execution phases only where the records of each name a serve and no serve is named for both.
They are one execution phase given two heads, which holds no execution phase at all, where a serve is named for
both, where no record names a serve for one of them, and where a table of one name has one domain in both: a
table's domain binds the network certifier's session, the link, the phase's number and the table's rows, so the
two are one session's execution phase whatever serves their records name. A head whose records name two serves is
no one serve's execution phase and is charged its whole schedule.

**The answer** (`stop`) counts the execution phases that are held: those whose head was read, in which some class
holds a count above 0 and every class with a count above 0 is charged below its schedule by the rule, whose
records name one serve, at the registered rate ("The rate"), each of its verdicts by a verifier with its control
("A negative control"), with no verdict of its serve refused or not complete under another head ("Another head"),
and of a serve launched after each superseded run that holds a record that doesn't read ("The order of runs"). One
class at its whole schedule keeps its execution phase at its whole schedule, and each line says which verdict it
lacks.
Every other execution phase is charged its whole schedule, one whose registration no run kept among them: the
calculator counts a row's execution phases from its monitoring period (`leak_bounds.Stop.share`), never from the
registrations read here. So a run's files with an execution phase left out charge what they would with that
execution phase refused, and since nothing but verdicts on a head's digest holds an execution phase, a
registration no verdict names holds none. A verdict is held to a head by the digest it names, so an accept decided
under another head, another serve's among them, counts for nothing. Registrations or records that don't read, in a run
that is not superseded ("The order of runs"), and two
heads given for one execution phase ("An execution phase is one serve's"), hold no execution phase at all. Each
verdict that is on a head read is handed on under the mark `leak_bounds.RUN_NOT_PROVED`, whatever its
``proved_by`` says: a run is no proof, and what a record's ``proved_by`` is worth is its reader's to check.

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
A control record names the run that produced it, its verifier's module and commit, what was changed (``control``),
the verifier's own verdict, whose ``format`` says which of the four it is a control of, and the verifier's own word
for why it refused (``reason``). A control counts only where it is refused for the reason its change aims at: the
reader holds ``reason`` to that control's word (`REASONS`), which is declared here and is no field of the record.
A control without the field doesn't read, and neither does a refused one whose reason is no text; the reason a
verifier gave is printed in the line of each class its accept is counted in. A verifier's refusal at a commit is a
fact of that code, so one control a verifier a commit serves every execution phase checked under that commit, and
a control names no head, class or serve: of a control record the reader reads its five fields and no other. A
control at another commit, by another verifier, or of a thing its verifier doesn't owe counts for nothing. One its
verifier did not refuse (``accepted`` anything but false), and one it refused for another reason than the
control's word, leave that verifier without its control at that commit, whatever other control it refused. So
does one whose accept its verifier withheld, and one whose command stopped: a control counts only where the
verifier refused, and a verdict that states a ``withheld`` or a ``stopped`` word is not accepted and is no
refusal, whatever ``reason`` its record gives. A control's verdict is held to what any verdict states of those
fields ("Withheld, and the codes", "Stopped"), and one that states one wrongly is a control record that doesn't
read. An
execution phase whose verifier lacks its control is charged its whole schedule, with a line that names the
verifier and, for a control refused for another reason, the reason given and the control's word. A control record
that doesn't read holds no execution phase at all, in a run that is not superseded. Of a superseded run, a control
that counts is left out, and one that was not refused, or was refused for another reason, stands, as its verdicts
do.

**The rate.** The stop record's and the response tie's populations (`RATED`), and the rate each is proved at, are
registered before any challenge, and a class is held only where every block of both is proved. Each of the two
verdicts runs over a stream of its own, whose registration record follows the one the class's verdicts name. A
stop record's and a response tie's record names that record by digest (``stream_registration``, `STREAM`). The
reader is handed the record itself (`SERVES_FILE`), finds it by that digest (`digest_of`), holds it to follow the
registration the class's verdicts name (its ``previous`` is that digest) and reads the rate from what it
registers: K of its population, under a sampling design that is a subset of K (`_registered`). The class is held
only where that rate is `EVERY_BLOCK` for both, and each of the two records says it ran at the registered rate
(``rate``, `RATE`). So the rate is the registration's and never the verdict's own word. Each of these leaves the
class at its whole schedule, with a line that names which: the registration the class's verdicts name not handed;
a record that names no registration of its stream, or one not handed; a stream's registration that follows
another record or none; one whose sampling design is no subset of K of its population; another rate registered;
a stop record's population that is not the head's count of the class; and a verdict that names no rate or another
than the registered one. So a verdict that states `EVERY_BLOCK` under a registration that says otherwise is
refused by name.

The stop record's population is held to the head's count of the class (`class_charge.count`), with nothing taken
off: the stop record is of every row of the class and of no other, and every block of a population of another
size is `EVERY_BLOCK` all the same, so the rate alone doesn't say it. Equal numbers don't make the two one
population: what holds a block to its row is the response tie's read of that row, and the reader holds nothing of
it. The response tie's population is not held to the count here. A failure is final for what its verdict speaks
for, which at `EVERY_BLOCK` is one execution phase's class; a rate below `EVERY_BLOCK` widens the failure to the
whole registered population, for when the reader takes one. A verdict's ``complete`` says that every session ran to a
verdict and holds no class to a rate. The certified read is over no sampled population, and the rate of the
compute audit's second stage is not held here: a rate on either's record is not read. A registration record that
doesn't read holds no execution phase at all, in a run that is not superseded."""
from __future__ import annotations

import fnmatch
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import leak_bounds as LB
from verity.protocols.accounting.certifier.verifier import class_charge as CC
from verity.protocols.accounting.certifier.verifier import connection_rows as C
from verity.protocols.accounting.certifier.verifier import content_rows as CT
from verity.protocols.proofs.spot.verifier.one_stage import registration as R

#: the record, in the evidence store, of the runs that hold the stop class's verdict records
RECORD = "exfiltration-glm53-stop-verdicts"
#: a run's files that are read: its verdict records and its control records, in every file so named, the
#: registrations of connection rows its verdicts name, and the serves' registrations its records name
RECORDS_FILES, REGISTRATIONS_FILE = "out/*-verdicts.jsonl", "out/registrations.jsonl"
CONTROLS_FILES, SERVES_FILE = "out/*-controls.jsonl", "out/serve-registrations.jsonl"
FILES = (RECORDS_FILES, REGISTRATIONS_FILE, CONTROLS_FILES, SERVES_FILE)
#: the first line of a registrations' file, as the check that wrote it writes it (`connection_rows_run`'s "The
#: registrations' file"): its own ``format`` word, which no registration has, and what its registrations were
#: checked under, the SHA-512 of the network certifier's public key and the session, each in hex
CHECKED = "verity-network-certifier/checked-registrations/v1"
CHECKED_UNDER, KEY_SHA512, SESSION = "checked_under", "key_sha512", "session"
#: what the reader says of a registrations' file without that line
UNSAID = "does not say what it was checked under"
#: which of the four verdicts a record is, or a control record is a control of, by the ``format`` of its verdict.
#: No format is listed, the response tie's among them: no statement registers what the tie reads, so its verdict
#: holds no class yet. Every verdict reads as missing
FORMAT = "format"
FORMATS: dict[str, str] = {}
#: a control record's fields: the run that produced it, the verifier that judged it, what was changed, the
#: verifier's own verdict on it, and the verifier's own word for why it refused
CONTROL, REASON = "control", "reason"
CONTROL_FIELDS = (CC.RUN, CC.VERIFIER, CONTROL, CC.VERDICT, REASON)
#: the negative control each verdict's verifier owes, as a control record names it, and what it changes
CONTROLS = {CC.STOP_RECORD: ("stop_word", "a stop word changed, the count off by one"),
            CC.RESPONSE_TIE: ("id", "a released id changed"),
            CC.CERTIFIED_READ: ("max_tokens", "a request word changed, `max_tokens`"),
            CC.COMPUTE_AUDIT: ("output", "a committed output changed")}
#: the word each verdict's control is refused for where its verifier refuses it for what was changed: a control
#: refused for any other reason is none of its verifier's
REASONS = {CC.STOP_RECORD: "stop_word", CC.RESPONSE_TIE: "id", CC.CERTIFIED_READ: "max_tokens",
           CC.COMPUTE_AUDIT: "output"}
#: the rate a record says its verdict ran at, and the one rate at which a class is held: every block proved. A
#: rate's text is `NUM/DEN` in lowest terms, with no prefix
RATE, EVERY_BLOCK = "rate", "1/1"
#: the verdicts over a sampled population, whose rate is held; how a record of one names the registration record of
#: its own stream, by digest; and what the reader reads of that record: the record it follows, its population and
#: its sampling design, which registers a rate where it is a subset of K
RATED = (CC.STOP_RECORD, CC.RESPONSE_TIE)
STREAM, PREVIOUS, POPULATION, DESIGN, SUBSET = "stream_registration", "previous", "population", "law", "subset"
#: what the answer's second line says of the serve an execution phase is of
SERVE_NAMED = ("an execution phase is one serve's, and a head's serve is the one its verdict records name, which is "
               f"{LB.RUN_NOT_PROVED}")
#: what the answer's first line says where each of the four verdicts was read and they are not one class's
NOT_ONE_CLASS = "not one class: unchecked"
#: what the answer's first line ends with where the rule charged no class below its schedule for want of the
#: schedule (`class_charge.NOT_HANDED`)
NO_SCHEDULE = "this reader is handed no schedule and asks the rule without one"
#: where a verdict says it is, and what four verdicts that are not one class's can differ in, each with its fields:
#: of a verdict where no class has all four on its head, and of a record (`class_charge.ONE`) where one has
PLACE = (CC.LINK, CC.WINDOW, CC.CLASS, CC.HEAD)
DIFFER = (((CC.CLASS,), "the class (`class`)"),
          ((CC.LINK, CC.WINDOW), "the execution phase (`link`, `window`)"),
          ((CC.HEAD,), "the signed head (`head`) of the network certifier's rows"))
ONE = {CC.SERVE: "the serve (`serve`)", CC.REGISTRATION: "the registration (`registration`)"}
#: what holds back an execution phase that has the verdicts of each class that holds a row, as the answer's first
#: line counts it
BACK = {"serve": "its serve, one named by all of a head's verdict records",
        "rate": f"the rate, `{EVERY_BLOCK}` as registered and as run",
        "control": "a negative control, one refused by each verifier that accepted, at its commit, for the "
                   "control's word",
        "head": "a verdict refused or not complete under another head of the execution phase",
        "unread": "a superseded run's record that doesn't read, of a run its serve was not launched after"}
#: a run id: the date and the time of the run's launch, then four digits of its own
RUN_ID = re.compile(r"r(\d{8}-\d{6})-[0-9a-f]{4}")


def launched_after(a: object, b: object) -> bool:
    """Whether the run ``a`` was launched after the run ``b``, by their ids (the module docstring's "The order of
    runs"): never where one of them is no run id, or where both state one second."""
    stamps = [RUN_ID.fullmatch(x) if isinstance(x, str) else None for x in (a, b)]
    return all(stamps) and stamps[0][1] > stamps[1][1]


def head_of(doc: Mapping[str, Any]) -> C.Head:
    """The head a registration of an execution phase's connection rows lists, held to the digest its certificate
    names, or ValueError naming what doesn't read. Its signature is not checked here, but where the registrations'
    file is written (the module docstring's "A head")."""
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


def _hex(x: object, size: int | None = None) -> bool:
    """Whether ``x`` is the lowercase hex of at least one byte, of ``size`` bytes where it is given."""
    return type(x) is str and bool(re.fullmatch(r"(?:[0-9a-f]{2})+", x)) and size in (None, len(x) // 2)


def checked_under(first: str | None) -> tuple[str, str]:
    """What a registrations' file says it was checked under (the module docstring's "A head"), from its first
    line, the file's line 1, None for a file that holds no line: the SHA-512 of the network certifier's public key
    and the session, each the hex the line holds. ValueError, saying why the file does not say it: no line; a
    blank line 1; one that is no JSON or no object; one whose ``format`` is not `CHECKED`, by name where it has
    none, has a registration's or has another; one of that format that holds anything but the two."""
    if first is None:
        raise ValueError(f"{UNSAID}: it holds no line")
    if not first.strip():
        raise ValueError(f"{UNSAID}: its first line is blank")
    try:
        doc = json.loads(first)
    except ValueError as e:
        raise ValueError(f"{UNSAID}: its first line is no JSON: {e}") from None
    if not isinstance(doc, Mapping):
        raise ValueError(f"{UNSAID}: its first line is no object")
    if (form := doc.get(FORMAT)) != CHECKED:
        what = ("has no `format`" if FORMAT not in doc else "is a registration" if form == C.REGISTRATION else
                f"has the `format` {json.dumps(form)[:80]}")
        raise ValueError(f"{UNSAID}: its first line {what}, where a line of the `format` `{CHECKED}` is due")
    under = doc.get(CHECKED_UNDER) if set(doc) == {FORMAT, CHECKED_UNDER} else None
    if not isinstance(under, Mapping) or set(under) != {KEY_SHA512, SESSION}:
        raise ValueError(f"{UNSAID}: its first line is not its `format` and the one field `{CHECKED_UNDER}`, an "
                         f"object with the two fields `{KEY_SHA512}` and `{SESSION}`")
    if not _hex(under[KEY_SHA512], 64):
        raise ValueError(f"{UNSAID}: its first line's `{KEY_SHA512}` is not the hex of a SHA-512")
    if not _hex(under[SESSION]):
        raise ValueError(f"{UNSAID}: its first line's `{SESSION}` is not the hex of a session")
    return under[KEY_SHA512], under[SESSION]


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
    if lack := CC.misstates(record[CC.VERDICT]):
        return lack
    if record[CC.VERDICT].get(CC.ACCEPTED) is False and not _named(record[REASON]):
        return f"its `{REASON}` is no text, though its verdict is refused"
    return None


#: what a control record is, as a refusal of one that is none says it
A_CONTROL = (f"a control record is an object with the fields {', '.join(CONTROL_FIELDS)}, its run and its control "
             f"named, its verifier by its {' and '.join(CC.NAMES)}, its verdict an object, and its reason its "
             "verifier's word for a refusal")


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


def _registered(registration: Mapping[str, Any]) -> str | None:
    """The rate a stream's registration registers, as `NUM/DEN` in lowest terms: K of its population, under a
    sampling design that is a subset of K. None where it registers no such design."""
    design, n = registration.get(DESIGN), registration.get(POPULATION)
    k = design.get("k") if isinstance(design, Mapping) and design.get(DESIGN) == SUBSET else None
    if type(k) is not int or type(n) is not int or not 0 < k <= n:
        return None
    rate = Fraction(k, n)
    return f"{rate.numerator}/{rate.denominator}"


def _one_phase(a: C.Head, b: C.Head, of_a: set[str], of_b: set[str]) -> str | None:
    """Why two heads given for one number of a link are one execution phase's (the module docstring's "An execution
    phase is one serve's"), from the serves the records of each name, or None where they are two serves'."""
    if shared := sorted(of_a & of_b):
        return f"both of the serve {shared[0]}, as their verdict records name it"
    domains = {e.name: e.domain for e in a.tables}
    if same := sorted(e.name for e in b.tables if domains.get(e.name) == e.domain):
        return (f"their table `{same[0]}` has one domain, so they are one session's execution phase, whatever "
                "serves their verdict records name")
    if not of_a or not of_b:
        return "no verdict record names the serve of one of them, so they are not known to be two serves'"
    return None


Read = Sequence[tuple[str, Mapping[str, Any]]]


def _unrated(read: Read, served: Mapping[str, Mapping[str, Any]], count: int) -> list[str]:
    """What keeps a class the rule charges below from being held, by the rate (the module docstring's "The rate"),
    from its records on its head and the head's count of the class: empty where the registration they name was
    handed, and each stop record and response tie among them names its own stream's registration, which was
    handed, follows that one and registers `EVERY_BLOCK`, a stop record's of a population of ``count``, and says
    it ran at that."""
    name = read[0][1][CC.REGISTRATION]
    if name not in served:
        return [f"the registration its verdicts name, `{name}`, is not among the serves' registrations read "
                f"(`{SERVES_FILE}`), so what the serve registered is not known"]
    lacks = []
    for kind, record in read:
        if kind not in RATED:
            continue
        who, stream = f"the {kind} of {record[CC.RUN]}", record.get(STREAM)
        if not _named(stream):
            lacks.append(f"{who} names no registration of its own stream (`{STREAM}`), so the rate it is registered "
                         "at is not known")
        elif stream not in served:
            lacks.append(f"{who} names its stream's registration `{stream}`, which is not among the serves' "
                         f"registrations read (`{SERVES_FILE}`)")
        elif (previous := served[stream].get(PREVIOUS)) != name:
            lacks.append(f"{who} names its stream's registration `{stream}`, which follows "
                         + (f"`{previous}`" if _named(previous) else "no record")
                         + f" and not the registration its verdicts name, `{name}`")
        elif (registered := _registered(served[stream])) is None:
            lacks.append(f"{who} names its stream's registration `{stream}`, which registers no rate: its sampling "
                         f"design is no `{SUBSET}` of K of its population")
        else:
            if registered != EVERY_BLOCK:
                lacks.append(f"{who} is registered at `{registered}`, not `{EVERY_BLOCK}`")
            elif kind == CC.STOP_RECORD and (n := served[stream][POPULATION]) != count:
                lacks.append(f"{who} is registered over a population of {n}, and the head counts {count} row(s) of "
                             f"the class: `{EVERY_BLOCK}` of it is not every row of the class and no other")
            if record.get(RATE) != registered:
                ran = record.get(RATE)
                lacks.append(f"{who} " + (f"states `{ran}`" if _named(ran) else f"names no rate (`{RATE}`)")
                             + f" under a registration that registers `{registered}`")
    return lacks


def _uncontrolled(kind: str, record: Mapping[str, Any],
                  owed: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]]) -> str | None:
    """What the verifier of a verdict counted lacks of its negative control (the module docstring's "A negative
    control"), or None where it has one of record at its commit and refused every one for the control's word."""
    module, commit = (record[CC.VERIFIER][k] for k in CC.NAMES)
    name, what = CONTROLS[kind]
    who = f"the {kind}'s verifier {module} at {commit}"
    its = owed.get((kind, module, commit), ())
    if not its:
        return f"{who} has no negative control of record at that commit (`{name}`: {what})"
    if kept := sorted({r[CC.RUN] + _no_refusal(r[CC.VERDICT]) for r in its if not _refusal(r[CC.VERDICT])}):
        return f"{who} did not refuse its negative control (`{name}`: {what}) on {', '.join(kept)}"
    if other := sorted({(r[REASON], r[CC.RUN]) for r in its if r[REASON] != REASONS[kind]}):
        return f"{who}: " + "; ".join(f"control {name} refused for {reason!r}, not {REASONS[kind]!r}, on {run}"
                                      for reason, run in other)
    return None


def _refusal(verdict: Mapping[str, Any]) -> bool:
    """Whether a control's verdict is its verifier's refusal: ``accepted`` exactly false, and neither an accept
    withheld nor a command that stopped, which are not accepted and are no refusal (the module docstring's "A
    negative control")."""
    return verdict.get(CC.ACCEPTED) is False and CC.not_accepted(verdict) == CC.REFUSED


def _no_refusal(verdict: Mapping[str, Any]) -> str:
    """What a line adds of a control's verdict that is not accepted and is no refusal either: how it went."""
    return f" ({CC.not_accepted(verdict)}, which is no refusal)" if verdict.get(CC.ACCEPTED) is False else ""


def _counts(kind: str, record: Mapping[str, Any]) -> bool:
    """Whether a control record of the control ``kind``'s verifier owes counts for it: refused, for that control's
    word."""
    return _refusal(record[CC.VERDICT]) and record[REASON] == REASONS[kind]


def _refused(read: Read, owed: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]]) -> str:
    """What a class's line says of the negative controls the verifiers of its records refused: the reason each
    gave (`REASON`), which is its control's word. Nothing for a verifier that lacks its control."""
    said = []
    for kind, record in read:
        its = owed.get((kind, *(record[CC.VERIFIER][k] for k in CC.NAMES)), ())
        if its and all(_counts(kind, r) for r in its):
            said.append(f"the {kind}'s `{CONTROLS[kind][0]}` for {json.dumps(its[0][REASON])}")
    return f"; its verifiers refused their negative controls: {'; '.join(dict.fromkeys(said))}" if said else ""


def _elsewhere(failed: Read, serves: Sequence[str], head: C.Head, digest: str) -> Read:
    """The verdicts that don't count which are on the execution phase of ``head``, one of the ``serves`` its
    records name, its link and its number, under another head than its own, ``digest``, whatever class they name
    (the module docstring's "Another head")."""
    return [(kind, r) for kind, r in failed if r[CC.SERVE] in serves and r[CC.VERDICT].get(CC.HEAD) != digest
            and [r[CC.VERDICT].get(k) for k in (CC.LINK, CC.WINDOW)] == [head.link, head.window]]


def _handed(kind: str, record: Mapping[str, Any]) -> dict[str, Any]:
    """A verdict read as the calculator prints it: which of the four it is, what it is of and how it went, what it
    was withheld for (null where it was not) and its codes as written, the word its command stopped with (null
    where it states none), its run, serve, registration, verifier and proof as its record names them, the rate it
    says it ran at, and the mark."""
    verdict = record[CC.VERDICT]
    return {"verdict": kind, **{k: verdict.get(k) for k in CC.READ}, **{k: verdict[k] for k in CC.SAID},
            CC.STOPPED: verdict.get(CC.STOPPED),
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
         served: Iterable[Mapping[str, Any]] = (), spent: Iterable[Mapping[str, Any]] = (),
         unread: Iterable[str] = ()) -> LB.Stop:
    """How the stop class is charged (the module docstring's "The answer"), from registrations, verdict records,
    control records and serves' registrations as `read` returns them; ``left`` are lines on what the
    registrations' files were checked under and on what was not read, which the answer carries, and ``spent`` the
    accepts that fill no verdict, a superseded run's and those on a
    serve launched after their run, which are held to what holds any record on their head (the module docstring's
    "A superseded run"). ``unread`` is the run of each record of a superseded run that doesn't read: an execution
    phase is held only where its serve was launched after the newest of them ("The order of runs")."""
    registrations, records, spent, unread = list(registrations), list(records), list(spent), list(unread)
    if not registrations and not records:
        return LB.Stop(0, (*LB.NO_RECORDS.why, *left))
    if not registrations:
        return _whole(f"{len(records)} verdict record(s) were read and no registration (`{REGISTRATIONS_FILE}`), so "
                      "no class's count is known", *left)
    heads: dict[str, C.Head] = {}
    for doc in registrations:
        try:
            head = head_of(doc)
        except ValueError as e:
            return _whole(str(e), *left)
        heads[head.digest().hex()] = head
    by_verdict: dict[str, list[Mapping[str, Any]]] = {}
    spent_by: dict[str, list[Mapping[str, Any]]] = {}
    unlisted = 0
    for record in (*records, *spent):
        if lack := CC.lacks(record):
            return _whole(f"{CC.A_RECORD}; of one read, {lack}: {record!r}", *left)
    for record in records:
        kind = _kind(record[CC.VERDICT])
        if kind is None:
            unlisted += 1
        else:
            by_verdict.setdefault(kind, []).append(record)
    for record in spent:
        if (kind := _kind(record[CC.VERDICT])) is not None:
            spent_by.setdefault(kind, []).append(record)
    # the serves each head's verdict records name, and the heads of each number of a link: two of one number are
    # two execution phases only as two serves'
    serves: dict[str, set[str]] = {digest: set() for digest in heads}
    for record in (*records, *spent):
        if isinstance(name := record[CC.VERDICT].get(CC.HEAD), str) and name in serves:
            serves[name].add(record[CC.SERVE])
    numbered: dict[tuple[str, int], list[str]] = {}
    for digest, head in heads.items():
        for other in numbered.setdefault((head.link, head.window), []):
            if why := _one_phase(heads[other], head, serves[other], serves[digest]):
                return _whole(f"two heads are given for execution phase {head.window} of the link {head.link}: "
                              f"{why}", *left)
        numbered[head.link, head.window].append(digest)
    both = {kind: [*by_verdict.get(kind, ()), *spent_by.get(kind, ())] for kind in CC.VERDICTS}

    def charged(head: C.Head, cls: int) -> tuple[CC.ClassCharge, Read]:
        """How the rule charges a class by the records that count, held to one serve and one registration over
        the superseded runs' accepts on its head too; and its records on its head, those accepts among them."""
        live = CC.charge(head, cls, by_verdict)
        if not spent_by:
            return live, live.read
        whole = CC.charge(head, cls, both)
        return replace(live, below=live.below and not whole.differ, differ=whole.differ), whole.read
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
        phases = [(digest, head, [charged(head, cls) for cls in classes(head)]) for digest, head in sorted(
            heads.items(), key=lambda at: (at[1].link, at[1].window, sorted(serves[at[0]]), at[0]))]
    except ValueError as e:
        return _whole(str(e), *left)
    charges = [c for _, _, its in phases for c, _ in its]
    lines: list[str] = []
    # the verdicts that don't count, refused or not complete, each with which of the four it is; and the cut-off,
    # the newest superseded run that holds a record that doesn't read: a name that is no run id is after every id
    failed = [(kind, r) for kind, its in by_verdict.items() for r in its if not _adds(r)]
    cut = max(unread, key=lambda run: (RUN_ID.fullmatch(run) is None, run), default=None)
    # an execution phase is ruled below where a class of it holds a row and each class that holds one is charged
    # below by the rule; it is held where nothing of `BACK` holds it back
    held = ruled = elsewhere = 0
    by = dict.fromkeys(BACK, 0)
    for digest, head, its in phases:
        at = f"the link {head.link}, execution phase {head.window}"
        if len(numbered[head.link, head.window]) > 1:
            at += f" of the serve {', '.join(sorted(serves[digest]))}"
        lines += [f"{at}, {c.why()}{_refused(read, owed) if c.below else ''}" for c, read in its]
        rows = [(c, read) for c, read in its if c.count]
        named = sorted(serves[digest])
        tried = [f"the {kind} is "
                 f"{CC.NOT_COMPLETE if r[CC.VERDICT].get(CC.ACCEPTED) is True else CC.not_accepted(r[CC.VERDICT])} "
                 f"on class {r[CC.VERDICT].get(CC.CLASS)} ({r[CC.RUN]})"
                 for kind, r in _elsewhere(failed, named, head, digest)]
        again = [f"{at}: held to its whole schedule, every class of it, by a verdict under another head: "
                 f"{'; '.join(tried)}"] * bool(tried)
        elsewhere += len(tried)
        if not rows or not all(c.below for c, _ in rows):
            lines += again
            continue
        ruled += 1
        wants = list(dict.fromkeys(lack for _, read in rows for kind, record in read
                                   if (lack := _uncontrolled(kind, record, owed))))
        back = {
            "serve": [f"{at}: held to its whole schedule by its serve: its verdict records name {len(named)} serves, "
                      f"not one ({', '.join(named)}), and an execution phase is one serve's"] if len(named) > 1 else [],
            "rate": [f"{at}, class {c.cls}: held to its whole schedule by the rate: {'; '.join(lacks)}"
                     for c, read in rows if (lacks := _unrated(read, by_digest, c.count))],
            "control": [f"{at}: held to its whole schedule by a control: {'; '.join(wants)}"] if wants else [],
            "head": again,
            "unread": [f"{at}: held to its whole schedule by a record that doesn't read: its serve was not launched "
                       f"after {cut}, the newest of the superseded runs that hold one ({len(unread)} such "
                       "record(s))"] if cut is not None and not all(launched_after(s, cut) for s in named) else []}
        for why, its_lines in back.items():
            lines += its_lines
            by[why] += bool(its_lines)
        held += not any(back.values())
    with_rows = [c for c in charges if c.count]
    # the classes that hold a row and have a record of each of the four verdicts on their head
    all_four = [c for c in with_rows if {kind for kind, _ in c.read} >= set(CC.VERDICTS)]
    counted = [_handed(kind, record) for c in charges for kind, record in c.read]
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
    elif not ruled and (unasked := sum(1 for c in with_rows if c.untied == CC.NOT_HANDED)):
        short = sum(1 for c in with_rows if c.owed or c.differ)
        first = (f"each connection is charged its whole schedule: {CC.NOT_HANDED} for {unasked} of the "
                 f"{len(with_rows)} class(es) that hold a row in the {of}, so the rule charges none of those below "
                 f"it, and {short} of the {len(with_rows)} lack a verdict: {NO_SCHEDULE}")
    elif not ruled:
        lacking = sum(1 for c in with_rows if not c.below)
        first = (f"each connection is charged its whole schedule: {lacking} of the {len(with_rows)} class(es) that "
                 f"hold a row in the {of} lack a verdict, one in each execution phase that holds a row")
    elif not held:
        first = (f"each connection is charged its whole schedule: {ruled} of the {of} have the {len(CC.VERDICTS)} "
                 f"verdicts of each class that holds a row, and none of them is held: "
                 + " and ".join(f"{n} by {BACK[why]}" for why, n in by.items() if n))
    else:
        first = (f"{held} of the {of} are charged below their schedule: each class that holds a row in one of them "
                 f"has its {len(CC.VERDICTS)} verdicts, accepted and complete, at the registered rate "
                 f"`{EVERY_BLOCK}`, each by a verifier that refused its negative control. Every other execution "
                 "phase of a monitoring period is charged its whole schedule, one no registration was read for "
                 "among them")
    other = len(records) - unlisted - len(counted) - elsewhere
    return LB.Stop(held, (
        first, SERVE_NAMED, *lines,
        *([f"{other} verdict record(s) read are on no head read, or on no class of one that holds a row, and count "
           "for nothing"] if other else []),
        *([f"{unlisted} verdict record(s) read are of a format listed for none of the {len(CC.VERDICTS)} verdicts, "
           "and count for nothing"] if unlisted else []),
        *([f"{stray} control record(s) read are of a format listed for none of the {len(CC.VERDICTS)} verdicts, or "
           "are not the control their verdict's verifier owes, and count for nothing"] if stray else []),
        *left), tuple(counted))


def _lines(path: Path) -> list[tuple[int, str]]:
    return [(n, line) for n, line in enumerate(path.read_text().splitlines(), 1) if line.strip()]


def _said(key: str, session: str, runs: Sequence[str]) -> str:
    """The answer's line on what the registrations' files of ``runs`` say they were checked under."""
    return (f"`{REGISTRATIONS_FILE}` of {', '.join(runs)} says its registrations were checked under the network "
            f"certifier's key whose SHA-512 is {key}, in the session {session}: the reader holds neither, and "
            "compares them with nothing")


def _lack(path: str, record: Any) -> str | None:
    """What keeps a record of a run's file ``path`` from reading, as `stop` refuses it, or None where it reads."""
    if path in (REGISTRATIONS_FILE, SERVES_FILE):
        try:
            (head_of if path == REGISTRATIONS_FILE else digest_of)(record)
        except ValueError as e:
            return str(e)
        return None
    return (control_lacks if fnmatch.fnmatchcase(path, CONTROLS_FILES) else CC.lacks)(record)


def _adds(record: Any) -> bool:
    """Whether a record counts toward a charge below: a verdict record that is accepted and complete."""
    return (CC.reads(record) and record[CC.VERDICT].get(CC.ACCEPTED) is True
            and record[CC.VERDICT].get(CC.COMPLETE) is True)


def _backs(record: Any) -> bool:
    """Whether a record counts toward a verifier's control: a control record of the control its verdict's verifier
    owes, refused for that control's word."""
    if control_lacks(record) is not None or (kind := _kind(record[CC.VERDICT])) is None:
        return False
    return record[CONTROL] == CONTROLS[kind][0] and _counts(kind, record)


def read(store: Any, runs: Iterable[str], superseded: Iterable[str] = ()
         ) -> tuple[list[Any], list[Any], list[str], list[Any], list[Any], list[Any], list[str]]:
    """The registrations and the verdict records in the run files of ``runs``, in the runs' order and each run's
    files by name; a line on what the registrations' files say they were checked under (`checked_under`), then a
    line for each run that is not in the store, kept no run files or holds no such file; the runs' control records
    and serves' registrations; the accepts that fill no verdict; and the run of each record of a superseded run
    that doesn't read: `stop`'s arguments, in its order.
    A registrations' file is read past its first line, and one that does not say what it was checked under gives
    no registration: in a run that is not superseded it is a ValueError that names the file and the run, and in a
    superseded run the file is one record that doesn't read.
    ``superseded`` are the runs among them that another run replaces: a verdict record of one that is accepted and
    complete is handed apart, as one that fills no verdict, and a control record of one that counts for its
    verifier is left out, each with a line that counts them (the module docstring's "A superseded run" and "A
    negative control"). A record is a superseded run's where that run's files hold it or it names that run as its
    ``run``, or, a verdict record, as its ``serve``. A line of a superseded run's files that doesn't read, one that
    is no JSON among them, is handed as its run alone, with a line that names the first ("The order of runs").
    So is an accept handed apart that names a serve launched after its own run, the one whose files hold it or the
    one it names as its ``run``. A line that is no JSON in a run that is not superseded is a ValueError that names
    the line, the file and the run.
    ``store`` is the evidence store as the harness reads it (`get_attempt`, `listing`, `fetch`)."""
    registrations: list[Any] = []
    records: list[Any] = []
    controls: list[Any] = []
    served: list[Any] = []
    spent: list[Any] = []
    left: list[str] = []
    later: list[str] = []
    unread: list[tuple[str, str]] = []
    said: dict[tuple[str, str], list[str]] = {}
    superseded = set(superseded)
    uncounted = 0
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
            control = fnmatch.fnmatchcase(path, CONTROLS_FILES)
            lines = _lines(at / path)
            if path == REGISTRATIONS_FILE:
                # its line 1 is what it was checked under, and a file without one, a blank line 1 among them,
                # gives no head
                first = None if not lines else lines[0][1] if lines[0][0] == 1 else ""
                try:
                    said.setdefault(checked_under(first), []).append(run)
                except ValueError as e:
                    if run not in superseded:
                        raise ValueError(f"`{path}` of {run} {e}") from None
                    unread.append((run, f"`{path}` of {run} {e}"))
                    continue
                lines = lines[1:]
            for n, line in lines:
                where = f"line {n} of `{path}` of {run}"
                try:
                    record, lack = json.loads(line), None
                except ValueError as e:
                    if run not in superseded:
                        raise ValueError(f"{where} is no JSON: {e}") from None
                    record, lack = line, f"it is no JSON: {e}"
                if run in superseded and (lack := lack or _lack(path, record)):
                    unread.append((run, f"{where}: {lack}"))
                    continue
                if path in (REGISTRATIONS_FILE, SERVES_FILE):
                    (registrations if path == REGISTRATIONS_FILE else served).append(record)
                    continue
                keys = (CC.RUN,) if control else (CC.RUN, CC.SERVE)
                named = [record.get(key) for key in keys] if isinstance(record, Mapping) else []
                old = run in superseded or any(isinstance(name, str) and name in superseded for name in named)
                # the runs a verdict record is of that its serve was launched after
                own = [] if control or not named else [r for r in (run, named[0]) if launched_after(named[1], r)]
                if old and control and _backs(record):
                    uncounted += 1
                elif (old or own) and not control and _adds(record):
                    spent.append(record)
                    later += [] if old else own[:1]
                else:
                    (controls if control else records).append(record)
    left[:0] = [_said(key, session, runs) for (key, session), runs in said.items()]
    of = f"of the superseded run(s) {', '.join(sorted(superseded))}"
    if len(spent) > len(later):
        left.append(f"{len(spent) - len(later)} verdict record(s) {of} are accepted and complete, and fill no "
                    "verdict: a superseded run adds no accept, and what it refused stands")
    if later:
        left.append(f"{len(later)} verdict record(s) of {', '.join(sorted(set(later)))} name a serve launched after "
                    "their own run, are accepted and complete, and fill no verdict: a run has no accept on a serve "
                    "launched after it, and what such a record refused stands")
    if uncounted:
        left.append(f"{uncounted} control record(s) {of} are refused for their control's word, and count for "
                    "nothing: a superseded run adds no control, and one it did not refuse, or refused for another "
                    "reason, stands")
    if unread:
        left.append(f"{len(unread)} record(s) of the superseded run(s) {', '.join(sorted({r for r, _ in unread}))} "
                    "don't read, and keep at its whole schedule each execution phase of a serve that was not "
                    f"launched after their run: such a record could have been a refusal. The first is {unread[0][1]}")
    return registrations, records, left, controls, served, spent, [run for run, _ in unread]


def of_runs(store: Any, runs: Iterable[str], superseded: Iterable[str] = ()) -> LB.Stop:
    """How the stop class is charged by the records ``runs`` hold in ``store`` (`read`, `stop`): ``runs`` is every
    run that carries the record, and ``superseded`` those of them that another run replaces (the module docstring's
    "A superseded run"). With no run it is its whole schedule, and the line says no run carries the record; so it
    is with a superseded run that is not among ``runs``, whose refusals were not read, and with a file that can't
    be fetched or a line that is no JSON in a run that is not superseded."""
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
