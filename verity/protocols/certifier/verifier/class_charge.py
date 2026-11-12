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

**A verdict record** is an object with the fields `FIELDS`: which of the four it is (`kind`), the serve it was run on
(`serve`), the execution phase and the class it is of (`link`, `window`, `class`, as the head and the schedule name
them), the run that produced it and the verifier that judged it (`run`, `verifier`), what the verifier said
(`verdict`, `ACCEPT` or anything else), whether what it accepted is the whole of what the verdict is of (`complete`)
and what proves the verifier (`proved_by`). `charge` reads the first eight but `run` and `verifier`, which it only
hands on with the record.

**All four or none.** A verdict counts where a record of its kind is of the class, in the head's execution phase, on
the serve the head is of, and every such record accepts and is complete. `complete` is a field of the record and no
default: a record without it reads as not complete. One verdict that is missing, refused, not complete or only of
another serve leaves every connection of the class at its whole schedule, and `charge` names each such verdict with
what it lacks. No verdict of one class, execution phase or serve counts for another.

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

#: the four verdicts a class owes before it is charged below its schedule, as a record's `kind` names them
STOP_RECORD, RESPONSE_TIE, CERTIFIED_READ, COMPUTE_AUDIT = "stop record", "response tie", "certified read", "compute audit"
VERDICTS = (STOP_RECORD, RESPONSE_TIE, CERTIFIED_READ, COMPUTE_AUDIT)
#: a verdict record's fields, and the verdict that accepts
KIND, SERVE, LINK, WINDOW, CLASS, RUN, VERIFIER, VERDICT, COMPLETE, PROVED_BY = FIELDS = (
    "kind", "serve", "link", "window", "class", "run", "verifier", "verdict", "complete", "proved_by")
ACCEPT = "accept"
#: what a verdict that doesn't count lacks
MISSING, ANOTHER_SERVE, REFUSED, NOT_COMPLETE = "missing", "of another serve", "refused", "not complete"


@dataclass(frozen=True, slots=True)
class ClassCharge:
    """How an execution phase's class is charged. `count`: its count in the head. `below`: whether its connections
    are charged below their schedule, which takes a count above 0 and all four verdicts. `owed`: each verdict that
    doesn't count, in `VERDICTS`' order, as (its kind, what it lacks, the runs of the records that lack it); empty
    where `below` and where the count is 0. `read`: the class's records on the head's serve and execution phase,
    by kind in `VERDICTS`' order, counted or not."""

    cls: int
    count: int
    below: bool
    owed: tuple[tuple[str, str, tuple[str, ...]], ...]
    read: tuple[Mapping[str, Any], ...]

    @property
    def schedules(self) -> int:
        """How many of the class's connections are charged their whole schedule: all of its count, or none."""
        return 0 if self.below else self.count

    def why(self) -> str:
        """One line: how the class is charged and, at its whole schedule, each verdict that doesn't count."""
        if self.below:
            return (f"class {self.cls}: charged below its schedule, its {len(VERDICTS)} verdicts "
                    f"({', '.join(VERDICTS)}) each accepted and complete on the head's serve")
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


def _is(record: Mapping[str, Any], key: str, value: object) -> bool:
    return type(record.get(key)) is type(value) and record[key] == value


def _runs(records: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted({str(r.get(RUN)) for r in records}))


def charge(head: Head, cls: int, serve: str, records: Iterable[Mapping[str, Any]]) -> ClassCharge:
    """How class `cls` of the execution phase `head` is signed for is charged, from the verdict records given
    (the module docstring's "All four or none"). `serve`: the serve the head is of. ValueError for a head without
    the class, a serve that is no name, or a record that is no object."""
    n = count(head, cls)
    if type(serve) is not str or not serve:
        raise ValueError(f"the head's serve is its name, not {serve!r}")
    records = list(records)
    if off := [r for r in records if not isinstance(r, Mapping)]:
        raise ValueError(f"a verdict record is an object with the fields {', '.join(FIELDS)}, not {off[0]!r}")
    if not n:
        return ClassCharge(cls, 0, False, (), ())
    here = [r for r in records if _is(r, LINK, head.link) and _is(r, WINDOW, head.window) and _is(r, CLASS, cls)]
    owed: list[tuple[str, str, tuple[str, ...]]] = []
    read: list[Mapping[str, Any]] = []
    for kind in VERDICTS:
        of_kind = [r for r in here if _is(r, KIND, kind)]
        ours = [r for r in of_kind if _is(r, SERVE, serve)]
        read += ours
        if not ours:
            owed.append((kind, ANOTHER_SERVE if of_kind else MISSING, _runs(of_kind)))
        elif refused := [r for r in ours if not _is(r, VERDICT, ACCEPT)]:
            owed.append((kind, REFUSED, _runs(refused)))
        elif partial := [r for r in ours if r.get(COMPLETE) is not True]:
            owed.append((kind, NOT_COMPLETE, _runs(partial)))
    return ClassCharge(cls, n, not owed, tuple(owed), tuple(read))
