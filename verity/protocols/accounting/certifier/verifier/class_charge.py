"""How an execution phase's class of connection rows is charged (`charge`): its whole schedule, or less.

A class is one of the schedule's (`content_rows`' "The classes"), and the head the network certifier signs for an
execution phase lists the class's three tables (`connection_rows.Head`). A connection of a class is charged its
schedule, what a response of any length the class admits can carry, while the length of its response is the server's
to choose. It is charged less only where four verdicts on the class in that execution phase (`VERDICTS`) say the
length is no one's choice:

- the stop record: the count of ids a response releases is the registered stop rule's, on the committed ids and the
  request's three words;
- the response tie: the ids the network certifier released are the committed ids below that count;
- the certified read: the three words the rule reads (the most ids asked for, the prompt's length and whether the end
  id is ignored) are the certified request's;
- the compute audit: its second stage over the class's sample, that the committed ids are the outputs of the
  request's Program.

**A verdict record** is an object with the six fields `FIELDS`: the run that produced the verdict (`run`, its id),
the served run the class came from (`serve`, its id), the digest of what that serve registered (`registration`),
the verifier that judged it (`verifier`: its `module` and the `commit` it ran at), the verdict (`verdict`) and what
proves the verifier (`proved_by`: a guarantee's name, or null while none does). A record that names no run, no
serve, no registration, or no verifier by its module and commit, is refused like one that is no object (`reads`;
`lacks` says which it is): a verdict that can't be traced to what produced it and to what it is of holds nothing.
The verdict is the verifier's own object, and whichever of the four it is it says what it is of and how it went,
in the fields `READ`: the execution phase and the class (`link`, `window`, `class`, as the head and the schedule
name them), the digest of the signed head it was decided under (`head`), and two bits, `accepted` and `complete`,
the second saying that what was accepted is the whole of what the verdict is of. `charge` decides by those six
and by the serve and the registration a record names: the run and the verifier are held to be named and are not
weighed, and `proved_by` moves no charge. Which of the four a record is is its caller's to say: `charge` takes the
records by verdict.

**Withheld, and the codes.** A verdict states two fields more, each its verifier's own (`SAID`): `withheld`,
null or the one word for what the verifier held its accept back for, and `codes`, a list of the verifier's own code
words, empty on an accept. Neither moves a charge: a verdict counts only where `accepted` is exactly true, and a
withheld one leaves its class at its whole schedule as a refused one does. Of a verdict that is not accepted the
line says "withheld" with the word where it states one, and "refused" where `withheld` is null (`not_accepted`).
Every verdict states both, whatever its format: one with nothing to say states `withheld` null and `codes` empty,
so no withheld verdict reads as refused because a line was left out. A record doesn't read where its verdict
leaves either out, where its `withheld` is neither null nor a word, where its `codes` is no list of words, and
where its verdict is accepted and states a `withheld` word or a code: an accept that also says why it is no
accept says two things (`misstates`).

**Stopped.** A verifier's command that stopped with an error after its stream's session opened reached no verdict,
and it was a try: nothing retries a verdict, so a second run that then fills the verdict would be a second try
nobody sees. Its record states `accepted` false, `withheld` null, `codes` empty and one field more, `stopped`, the
command's own word for the error (`STOPPED`). Such a verdict is no refusal by a verifier, so the line says
"stopped" with the word; it is charged as a refused one is, and stands as one does. A command that stopped before
any challenge was sent tried nothing and writes no record. `stopped` may be absent, which reads as null. A record
doesn't read where its verdict's `stopped` is neither null nor a word, where its verdict is accepted and states
one, and where it states a `withheld` word or a code too: a verifier that withheld its accept, or gave a code,
reached a verdict.

**All four or none.** A verdict counts where a record of it is of the class, in the head's execution phase, names
this head's digest, and every such record is accepted and complete. Naming the head is what holds a verdict to the
serve the head was signed on: a verdict decided under another head, another run's or another serve's, names another
digest. `complete` is a field of the verdict and has no default: a verdict without it reads as not complete. One
verdict that is missing, not on this head, refused or not complete leaves every connection of the class at its
whole schedule, and `charge` names each such verdict with what it lacks. No verdict of one class or head counts for
another.

**One serve, one registration.** The four are one class's only where they are of one serve under what that serve
registered: the class's records on this head, counted or not, name one serve and one registration (`ONE`). The head
holds the network certifier's rows to a serve; it says nothing of the values the stop record and the compute audit
judge, so two verdicts on one head can be of two serves' values, and these two names are what holds them to one.
Where the records name two of either, the class is at its whole schedule and `charge` says which (`differ`).

**The count.** A class's count is its three tables' rows as the head states them, one number. A class whose count is
0 has no connection to charge and owes no verdict. Where the schedule fixes a class's rows whatever the load, the
head states those rows, and the class owes its verdicts whatever it holds.

**The schedule's word.** A class is charged below its schedule only where `charge` is handed the schedule the head
names and that schedule can be tied (`content_rows.can_be_tied`). A class of a schedule that can't be tied is at
its whole schedule whatever its verdicts say, and the line says what keeps the schedule from it: no tie of such a
class has a statement, so a verdict that says one was accepted is of something else. A schedule that is not the
head's, by the parameters' digest the head holds, is refused. With no schedule handed the word was not asked, so
the class is at its whole schedule too, whatever its verdicts say, and the line says so (`NOT_HANDED`): the head
holds the schedule's digest and none of its classes' sizes or its response limit, so `charge` can't read the word
from the head alone, and no caller is charged below a schedule by leaving the schedule out.

`charge` checks no signature, and `head` is its caller's. The network certifier's signature on a registration is
checked by `connection_rows.check_registration`, under the key and the session the auditor hands the offline check
of a network certifier's run, which keeps each registration it held in a file that says what it was checked under;
a caller's heads are to be read from that file and from nowhere else."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from verity.protocols.accounting.certifier.verifier import content_rows as CT
from verity.protocols.accounting.certifier.verifier.connection_rows import Head
from verity.protocols.accounting.certifier.verifier.warden import WardenParams

#: the four verdicts a class owes before it is charged below its schedule
STOP_RECORD, RESPONSE_TIE, CERTIFIED_READ, COMPUTE_AUDIT = "stop record", "response tie", "certified read", "compute audit"
VERDICTS = (STOP_RECORD, RESPONSE_TIE, CERTIFIED_READ, COMPUTE_AUDIT)
#: a verdict record's fields, and the fields of its verdict that `charge` reads
RUN, SERVE, REGISTRATION, VERIFIER, VERDICT, PROVED_BY = FIELDS = (
    "run", "serve", "registration", "verifier", "verdict", "proved_by")
#: how a record names its verifier
MODULE, COMMIT = NAMES = ("module", "commit")
#: what a class's records on a head name one of: the served run the class came from, and what that serve registered
ONE = (SERVE, REGISTRATION)
LINK, WINDOW, CLASS, HEAD, ACCEPTED, COMPLETE = READ = ("link", "window", "class", "head", "accepted", "complete")
#: two fields more every verdict states, which move no charge: what its verifier held its accept back for, null or
#: a word, and the verifier's own code words, a list
WITHHELD, CODES = SAID = ("withheld", "codes")
#: one field more a verdict may state: the word of the error its verifier's command stopped with, after its
#: stream's session opened and before any verdict; null or absent where it did not
STOPPED = "stopped"
#: what a verdict that doesn't count lacks; one that is not accepted and states a `stopped` or a `withheld` word
#: is `STOPPED` with, or `WITHHELD` for, that word (`not_accepted`)
MISSING, ANOTHER_HEAD, REFUSED, NOT_COMPLETE = "missing", "not on this head", "refused", "not complete"
#: what keeps a class at its whole schedule where `charge` was handed no schedule: whether the schedule can be tied
#: was not asked
NOT_HANDED = "the schedule was not handed"


@dataclass(frozen=True, slots=True)
class ClassCharge:
    """How an execution phase's class is charged. `count`: its count in the head. `below`: whether its connections
    are charged below their schedule, which takes a count above 0, all four verdicts, one serve and one
    registration named by them, and the head's schedule, handed and one that can be tied. `owed`: each verdict
    that doesn't count, in `VERDICTS`' order, as (the verdict, what it lacks, the runs of the records that lack
    it), a verdict some records refused and others withheld or stopped once for each; empty where `below` and
    where the count is 0. `read`: the class's records on this head, counted or not, each with the verdict it is
    of, in `VERDICTS`' order. `differ`: each of `ONE` that those records name more than one of, with the names;
    empty where `below`. `untied`: what keeps the schedule handed from being tied (`content_rows.untied`), or
    `NOT_HANDED` where none was handed; empty where `below` and where the count is 0."""

    cls: int
    count: int
    below: bool
    owed: tuple[tuple[str, str, tuple[str, ...]], ...]
    read: tuple[tuple[str, Mapping[str, Any]], ...]
    differ: tuple[tuple[str, tuple[str, ...]], ...] = ()
    untied: str = ""

    @property
    def schedules(self) -> int:
        """How many of the class's connections are charged their whole schedule: all of its count, or none."""
        return 0 if self.below else self.count

    def why(self) -> str:
        """One line: how the class is charged and, at its whole schedule, each verdict that doesn't count."""
        if self.below:
            return (f"class {self.cls}: charged below its schedule, its {len(VERDICTS)} verdicts "
                    f"({', '.join(VERDICTS)}) each accepted and complete on this head")
        if not self.count:
            return f"class {self.cls}: a count of 0, so no connection is charged and no verdict is owed"
        lacks = [f"the {kind} is {lack}" + (f" ({', '.join(runs)})" if runs else "") for kind, lack, runs in self.owed]
        lacks += [f"its verdicts on this head name {len(names)} `{key}`s, not one ({', '.join(names)})"
                  for key, names in self.differ]
        if self.untied:
            lacks.append(self.untied if self.untied == NOT_HANDED else f"its schedule can't be tied: {self.untied}")
        return f"class {self.cls}: charged its whole schedule on each of its {self.count} rows: {'; '.join(lacks)}"


def count(head: Head, cls: int) -> int:
    """Class `cls`'s count in the head: its three tables' rows, which are one number. ValueError where the head
    doesn't list the class's tables or states two counts for it."""
    if type(cls) is not int or cls < 0:
        raise ValueError(f"a class is its index in the schedule, 0 or more, not {cls!r}")
    listed = {e.name: e.rows for e in head.tables}
    names = CT.names(cls)
    if lacking := [n for n in names if n not in listed]:
        raise ValueError(f"the head lists {sorted(listed)}, not class {cls}'s {', '.join(lacking)}")
    if len(rows := {listed[n] for n in names}) != 1:
        raise ValueError(f"the head states {sorted(rows)} rows for class {cls}'s three tables, which hold one count")
    return rows.pop()


def _is(verdict: Mapping[str, Any], key: str, value: object) -> bool:
    return type(verdict.get(key)) is type(value) and verdict[key] == value


def _runs(records: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted({str(r.get(RUN)) for r in records}))


def _named(x: object) -> bool:
    return type(x) is str and bool(x)


def lacks(record: object) -> str | None:
    """What keeps `record` from being a verdict record (the module docstring's "A verdict record"), by the field's
    name, or None where it is one: an object with each of `FIELDS`, its run, its serve and its registration named,
    its verifier named by each of `NAMES`, its verdict an object that states nothing wrongly (`misstates`), and
    `proved_by` a name or null."""
    if not isinstance(record, Mapping):
        return "it is no object"
    if missing := [k for k in FIELDS if k not in record]:
        return f"it has no `{'`, `'.join(missing)}`"
    for key in (RUN, SERVE, REGISTRATION):
        if not _named(record[key]):
            return f"its `{key}` is no name"
    verifier = record[VERIFIER]
    if unnamed := [k for k in NAMES if not (isinstance(verifier, Mapping) and _named(verifier.get(k)))]:
        return f"its `{VERIFIER}` names no `{'`, `'.join(unnamed)}`"
    if not isinstance(record[VERDICT], Mapping):
        return f"its `{VERDICT}` is no object"
    if not (record[PROVED_BY] is None or _named(record[PROVED_BY])):
        return f"its `{PROVED_BY}` is neither a name nor null"
    return misstates(record[VERDICT])


def misstates(verdict: Mapping[str, Any]) -> str | None:
    """What a verdict states wrongly of `SAID` (the module docstring's "Withheld, and the codes"), or None where it
    states nothing wrongly: one of the two left out, a `withheld` that is neither null nor a word, `codes` that
    are no list of words, a `stopped` that is neither null nor a word ("Stopped"), an accept that states a
    `withheld` word, a code or a `stopped` word, and a verdict that states a `stopped` word beside a `withheld`
    word or a code."""
    if missing := [k for k in SAID if k not in verdict]:
        return (f"its verdict states no `{'` and no `'.join(missing)}`: a verdict with nothing to say states "
                f"`{WITHHELD}` null and `{CODES}` empty")
    if not (verdict[WITHHELD] is None or _named(verdict[WITHHELD])):
        return f"its verdict's `{WITHHELD}` is neither a word nor null"
    if not (isinstance(verdict[CODES], list) and all(type(c) is str for c in verdict[CODES])):
        return f"its verdict's `{CODES}` is no list of words"
    if not (verdict.get(STOPPED) is None or _named(verdict[STOPPED])):
        return f"its verdict's `{STOPPED}` is neither a word nor null"
    if verdict.get(ACCEPTED) is True and (said := [k for k in (*SAID, STOPPED) if verdict.get(k)]):
        return (f"its verdict is accepted and states `{'` and `'.join(said)}`: an accept that also says why it is "
                "no accept says two things")
    if verdict.get(STOPPED) and (said := [k for k in SAID if verdict[k]]):
        return (f"its verdict states `{STOPPED}` and `{'` and `'.join(said)}`: a command that stopped reached no "
                f"verdict, so it states `{WITHHELD}` null and `{CODES}` empty")
    return None


def reads(record: object) -> bool:
    """Whether `record` is a verdict record: it lacks nothing (`lacks`)."""
    return lacks(record) is None


def not_accepted(verdict: Mapping[str, Any]) -> str:
    """How a line says a verdict that is not accepted went (the module docstring's "Withheld, and the codes" and
    "Stopped"): `STOPPED` with the word it states, `WITHHELD` for the word it states, or `REFUSED` where it
    states neither word."""
    if _named(word := verdict.get(STOPPED)):
        return f"{STOPPED} with `{word}`"
    word = verdict.get(WITHHELD)
    return f"{WITHHELD} for `{word}`" if _named(word) else REFUSED


#: what a verdict record is, as a refusal of one that is none says it
A_RECORD = (f"a verdict record is an object with the fields {', '.join(FIELDS)}, its verdict an object, its "
            f"{', '.join(FIELDS[:3])} named, its verifier by its {' and '.join(NAMES)}, and its proved_by a name or null")


def charge(head: Head, cls: int, records: Mapping[str, Iterable[Mapping[str, Any]]],
           schedule: WardenParams | None = None) -> ClassCharge:
    """How class `cls` of the execution phase `head` is signed for is charged, from the verdict records given, by
    verdict: a map from each of `VERDICTS` to its records, a verdict with none left out (the module docstring's
    "All four or none"). `schedule` is the schedule the head names: a class of one that can't be tied is at its
    whole schedule, and so is every class where none is handed ("The schedule's word"). ValueError for a head
    without the class, a key that is none of the four, a record that is no verdict record (`reads`), or a schedule
    that is not the head's."""
    if schedule is not None and schedule.digest != head.parameters:
        raise ValueError(f"the schedule handed has the parameters' digest {schedule.digest.hex()}, and the head "
                         f"names {head.parameters.hex()}: a class is charged by its own head's schedule")
    n = count(head, cls)
    other = [k for k in records if k not in VERDICTS] if isinstance(records, Mapping) else [type(records).__name__]
    if other:
        raise ValueError(f"the records are given by verdict, each one of {', '.join(VERDICTS)}, not {other[0]!r}")
    given = {kind: list(records.get(kind, ())) for kind in VERDICTS}
    for kind, its in given.items():
        if off := [r for r in its if not reads(r)]:
            raise ValueError(f"{A_RECORD}; of the {kind}'s, {lacks(off[0])}: {off[0]!r}")
    if not n:
        return ClassCharge(cls, 0, False, (), ())
    digest = head.digest().hex()
    owed: list[tuple[str, str, tuple[str, ...]]] = []
    read: list[tuple[str, Mapping[str, Any]]] = []
    for kind, its in given.items():
        here = [r for r in its if all(_is(r[VERDICT], key, value) for key, value in
                                      ((LINK, head.link), (WINDOW, head.window), (CLASS, cls)))]
        ours = [r for r in here if _is(r[VERDICT], HEAD, digest)]
        read += [(kind, r) for r in ours]
        if not ours:
            owed.append((kind, ANOTHER_HEAD if here else MISSING, _runs(here)))
        elif refused := [r for r in ours if r[VERDICT].get(ACCEPTED) is not True]:
            # refused first, then each word a command stopped with and each a verdict was withheld for: the charge
            # is the same, the line says which
            went = sorted({not_accepted(r[VERDICT]) for r in refused}, key=lambda how: (how != REFUSED, how))
            owed += [(kind, how, _runs(r for r in refused if not_accepted(r[VERDICT]) == how)) for how in went]
        elif partial := [r for r in ours if r[VERDICT].get(COMPLETE) is not True]:
            owed.append((kind, NOT_COMPLETE, _runs(partial)))
    differ = tuple((key, names) for key in ONE if len(names := tuple(sorted({r[key] for _, r in read}))) > 1)
    untied = NOT_HANDED if schedule is None else CT.untied(schedule) or ""
    return ClassCharge(cls, n, not owed and not differ and not untied, tuple(owed), tuple(read), differ, untied)
