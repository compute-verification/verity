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

**A verdict record** is an object with the four fields `FIELDS`: the run that produced the verdict (`run`), the
verifier that judged it (`verifier`), the verdict (`verdict`) and what proves the verifier (`proved_by`). The verdict
is the verifier's own object, and whichever of the four it is it says what it is of and how it went, in the fields
`READ`: the execution phase and the class (`link`, `window`, `class`, as the head and the schedule name them), the
digest of the signed head it was decided under (`head`), and two bits, `accepted` and `complete`, the second saying
that what was accepted is the whole of what the verdict is of. `charge` reads those six and the run, and nothing
else of a record. Which of the four a record is is its caller's to say: `charge` takes the records by verdict.

**All four or none.** A verdict counts where a record of it is of the class, in the head's execution phase, names
this head's digest, and every such record is accepted and complete. Naming the head is what holds a verdict to the
serve the head was signed on: a verdict decided under another head, another run's or another serve's, names another
digest. `complete` is a field of the verdict and has no default: a verdict without it reads as not complete. One
verdict that is missing, not on this head, refused or not complete leaves every connection of the class at its
whole schedule, and `charge` names each such verdict with what it lacks. No verdict of one class or head counts for
another.

**The count.** A class's count is its three tables' rows as the head states them, one number. A class whose count is
0 has no connection to charge and owes no verdict. Where the schedule fixes a class's rows whatever the load, the
head states those rows, and the class owes its verdicts whatever it holds.

`head` is the verifier's read of the registration (`connection_rows.check_registration`), which holds it to the
network certifier's certificate; `charge` checks no signature."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from verity.protocols.certifier.verifier import content_rows as CT
from verity.protocols.certifier.verifier.connection_rows import Head

#: the four verdicts a class owes before it is charged below its schedule
STOP_RECORD, RESPONSE_TIE, CERTIFIED_READ, COMPUTE_AUDIT = "stop record", "response tie", "certified read", "compute audit"
VERDICTS = (STOP_RECORD, RESPONSE_TIE, CERTIFIED_READ, COMPUTE_AUDIT)
#: a verdict record's fields, and the fields of its verdict that `charge` reads
RUN, VERIFIER, VERDICT, PROVED_BY = FIELDS = ("run", "verifier", "verdict", "proved_by")
LINK, WINDOW, CLASS, HEAD, ACCEPTED, COMPLETE = READ = ("link", "window", "class", "head", "accepted", "complete")
#: what a verdict that doesn't count lacks
MISSING, ANOTHER_HEAD, REFUSED, NOT_COMPLETE = "missing", "not on this head", "refused", "not complete"


@dataclass(frozen=True, slots=True)
class ClassCharge:
    """How an execution phase's class is charged. `count`: its count in the head. `below`: whether its connections
    are charged below their schedule, which takes a count above 0 and all four verdicts. `owed`: each verdict that
    doesn't count, in `VERDICTS`' order, as (the verdict, what it lacks, the runs of the records that lack it);
    empty where `below` and where the count is 0. `read`: the class's records on this head, counted or not, each
    with the verdict it is of, in `VERDICTS`' order."""

    cls: int
    count: int
    below: bool
    owed: tuple[tuple[str, str, tuple[str, ...]], ...]
    read: tuple[tuple[str, Mapping[str, Any]], ...]

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
        lacks = "; ".join(f"the {kind} is {lack}" + (f" ({', '.join(runs)})" if runs else "")
                          for kind, lack, runs in self.owed)
        return f"class {self.cls}: charged its whole schedule on each of its {self.count} rows: {lacks}"


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


def charge(head: Head, cls: int, records: Mapping[str, Iterable[Mapping[str, Any]]]) -> ClassCharge:
    """How class `cls` of the execution phase `head` is signed for is charged, from the verdict records given, by
    verdict: a map from each of `VERDICTS` to its records, a verdict with none left out (the module docstring's
    "All four or none"). ValueError for a head without the class, a key that is none of the four, or a record
    that is no object or whose verdict is none."""
    n = count(head, cls)
    other = [k for k in records if k not in VERDICTS] if isinstance(records, Mapping) else [type(records).__name__]
    if other:
        raise ValueError(f"the records are given by verdict, each one of {', '.join(VERDICTS)}, not {other[0]!r}")
    given = {kind: list(records.get(kind, ())) for kind in VERDICTS}
    for kind, its in given.items():
        if off := [r for r in its if not (isinstance(r, Mapping) and isinstance(r.get(VERDICT), Mapping))]:
            raise ValueError(f"a verdict record is an object with the fields {', '.join(FIELDS)}, its verdict an "
                             f"object with {', '.join(READ)}; the {kind}'s is {off[0]!r}")
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
            owed.append((kind, REFUSED, _runs(refused)))
        elif partial := [r for r in ours if r[VERDICT].get(COMPLETE) is not True]:
            owed.append((kind, NOT_COMPLETE, _runs(partial)))
    return ClassCharge(cls, n, not owed, tuple(owed), tuple(read))
