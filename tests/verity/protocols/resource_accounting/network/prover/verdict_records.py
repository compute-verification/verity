"""The stop class's verdict records as the tests of `stop_verdicts`' reader and of the offline check's
registrations write them: the four verdicts' records on a class of a head and their negative controls, the serves'
registrations and their streams', a registrations' file with the line of what it was checked under, and the
evidence store as the reader reads it."""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path

from verity.protocols.resource_accounting.network.prover import stop_verdicts as SV
from verity.protocols.resource_accounting.network.verifier import class_charge as CC
from verity.protocols.resource_accounting.network.verifier import connection_rows as C
from verity.protocols.resource_accounting.network.verifier import content_rows as CT

TIE = "verity-network-certifier/response-tie-verdict/v1"
#: a format for each of the four verdicts: the response tie's own, and a stand-in for each verdict whose verifier's
#: format the reader doesn't list
FORMATS = {kind: TIE if kind == CC.RESPONSE_TIE else f"a format of the {kind}" for kind in CC.VERDICTS}
#: what the registrations' files of these tests say they were checked under: a key's SHA-512 and a session, in hex
UNDER = ("5a" * 64, "c3" * 32)


def checked(*registrations: object, under: tuple[object, object] = UNDER) -> str:
    """A registrations' file as the offline check of a network attester's run writes it: the line of what it was
    checked under, of its own format, then the registrations, one a line."""
    return "".join(json.dumps(x) + "\n" for x in (
        {"format": SV.CHECKED, SV.CHECKED_UNDER: {SV.KEY_SHA512: under[0], SV.SESSION: under[1]}}, *registrations))


def names_under(*runs: str, under: tuple[str, str] = UNDER) -> str:
    """The answer's line on what the registrations' files of `runs` say they were checked under."""
    return (f"`out/registrations.jsonl` of {', '.join(runs)} says its registrations were checked under the network "
            f"certifier's key whose SHA-512 is {under[0]}, in the session {under[1]}: the reader holds neither, and "
            "compares them with nothing")


#: the registration a serve registered its class under, and another serve's; each with the digest a record names
#: it by
SERVED = {"format": "a serve's registration"}
OTHERS = SERVED | {"window": "another serve's"}
REGISTERED, OTHER = SV.digest_of(SERVED), SV.digest_of(OTHERS)
#: the counts a class of the tests' heads holds, and the one `streams` registers the stop record's stream over
#: unless it is told another
COUNTS, ROWS = range(1, 9), 5


def streams(after: dict = SERVED, rows: int = ROWS, **over: object) -> dict[str, dict]:
    """The registration record of the stop record's stream and of the response tie's, by verdict: each follows
    ``after`` and registers every block of its population, the stop record's the ``rows`` its class holds and the
    response tie's 4, but for ``over``, the record's fields."""
    return {kind: {"format": "a stream's registration", "window": f"the {kind}'s stream", SV.POPULATION: n,
                   SV.DESIGN: {SV.DESIGN: SV.SUBSET, "k": n}, SV.PREVIOUS: SV.digest_of(after)} | over
            for kind, n in zip(SV.RATED, (rows, 4))}


def with_streams(*docs: dict, rows: tuple[int, ...] = (ROWS,)) -> list[dict]:
    """Each of the serves' registrations ``docs`` with the streams' that follow it (`streams`): the stop record's
    over each count of ``rows``, then the response tie's."""
    return [x for doc in docs
            for x in (doc, *(streams(doc, n)[CC.STOP_RECORD] for n in rows), streams(doc)[CC.RESPONSE_TIE])]


def rows_of(record: dict) -> int:
    """The count the stop record's stream a record of `four` names is registered over: its class's."""
    named = {SV.digest_of(streams(rows=rows)[CC.STOP_RECORD]): rows for rows in COUNTS}
    return named.get(record.get(SV.STREAM), ROWS)


def held_rows(of: C.Head, cls: int = 0) -> int:
    """The head's count of the class, or `ROWS` where it holds none or states no one count of it."""
    listed = {e.name: e.rows for e in of.tables}
    stated = {listed.get(name) for name in CT.names(cls)}
    return stated.pop() or ROWS if len(stated) == 1 else ROWS


def four(of: C.Head, cls: int = 0, proved_by: object = None, **over: object) -> list[dict]:
    """The four verdicts' records on a class of a head, as their verifiers write them: each accepted, complete and
    naming the head, but for `over`, the verdict's fields; of one serve under `SERVED`, the stop record and the
    response tie each naming its stream's registration (`streams`, the stop record's over the head's count of the
    class) and the rate it registers."""
    rows = held_rows(of, cls)
    return [{CC.RUN: f"run of the {kind}", CC.SERVE: "the serve", CC.REGISTRATION: REGISTERED,
             CC.VERIFIER: {"module": f"the {kind}'s verifier", "commit": "a commit"},
             CC.VERDICT: {SV.FORMAT: FORMATS[kind], CC.LINK: of.link, CC.WINDOW: of.window, CC.CLASS: cls,
                          CC.HEAD: of.digest().hex(), "registration": "a digest", CC.ACCEPTED: True,
                          CC.COMPLETE: True, "withheld": None, "codes": []} | over,
             CC.PROVED_BY: proved_by} | ({SV.RATE: "1/1", SV.STREAM: SV.digest_of(streams(rows=rows)[kind])}
                                         if kind in SV.RATED else {}) for kind in CC.VERDICTS]


def the_controls() -> list[dict]:
    """The negative control each of the four verifiers owes, as `four` names the verifiers: each refused, for the
    control's word."""
    return [{CC.RUN: f"control run of the {kind}",
             CC.VERIFIER: {"module": f"the {kind}'s verifier", "commit": "a commit"},
             SV.CONTROL: SV.CONTROLS[kind][0],
             CC.VERDICT: {SV.FORMAT: FORMATS[kind], CC.ACCEPTED: False, "withheld": None, "codes": ["a refusal"]},
             SV.REASON: SV.REASONS[kind]}
            for kind in CC.VERDICTS]


def under(doc: dict, records: list[dict]) -> list[dict]:
    """The records, naming ``doc`` as the registration their serve registered, and the stop record and the
    response tie the registration of its stream that follows ``doc`` (`streams`), the stop record's over the
    count its own names."""
    return [r | {CC.REGISTRATION: SV.digest_of(doc)} | (
        {SV.STREAM: SV.digest_of(streams(doc, rows_of(r))[kind])}
        if (kind := SV.FORMATS.get(r[CC.VERDICT][SV.FORMAT])) in SV.RATED else {})
        for r in records]


class Store:
    """The evidence store as `read` uses it: each run's files, by their path in the run."""

    def __init__(self, tmp: Path, runs: dict[str, dict[str, str] | None]):
        self.tmp, self.runs = tmp, runs

    def get_attempt(self, run: str) -> dict | None:
        if run not in self.runs:
            return None
        return {"outputs": {"run_files": run} if self.runs[run] is not None else {}}

    def listing(self, files: str, paths: list[str]) -> list[dict]:
        return [{"path": p} for p in self.runs[files] if any(fnmatch.fnmatchcase(p, pat) for pat in paths)]

    def fetch(self, files: str, paths: list[str]) -> str:
        at = self.tmp / files
        for name in paths:
            (at / name).parent.mkdir(parents=True, exist_ok=True)
            (at / name).write_text(self.runs[files][name])
        return str(at)


def jsonl(lines: list) -> str:
    return "".join(json.dumps(x) + "\n" for x in lines)
