"""The stop class's verdict records as the design's row reads them (`stop_verdicts`): a head is the registration's
own, held to the digest its certificate names; the answer is below the schedule only where every class that holds
a row, in every head read, has its four verdicts by the network certifier's rule, on one serve; anything less, a
head that doesn't read, or no record at all is the whole schedule with a line that says why; each verdict read is
handed on under the mark of a run that is no proof; and the run files are read from the store as the harness
reads it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import connection_rows_run as CR  # noqa: E402
import leak_bounds as LB  # noqa: E402
import stop_verdicts as SV  # noqa: E402
from verity.protocols.certifier.verifier import class_charge as CC  # noqa: E402
from verity.protocols.certifier.verifier import connection_rows as C  # noqa: E402
from verity.protocols.certifier.verifier import content_rows as CT  # noqa: E402
from verity.protocols.certifier.verifier.certificate import Certificate, Direction  # noqa: E402
from verity.protocols.certifier.verifier.warden import WardenParams  # noqa: E402

LINK, SERVE = "responses", "the serve"
ZEROS = bytes(64)


def schedule() -> WardenParams:
    """The lengths rows' cost runner's own schedule, with its content classes."""
    args = CR.parser().parse_args(["cost", "--out", "x"])
    return WardenParams(cell_bytes=args.cell_bytes, slots=args.slots, lanes=args.lanes, window=args.window,
                        every=args.every, delay=args.delay, ingress_buckets=args.delay,
                        ingress_cells=args.ingress_cells, egress_cells=args.egress_cells, lengths=1,
                        content=tuple(args.classes))


def line(window: int, *counts: int, serve: object = SERVE) -> dict:
    """A head line: a registration, in the verifier's format, of the lengths and fails tables and a class's three
    tables for each count, with a certificate that names the head's digest."""
    listed = [("lengths", 4, 8), ("fails", 4, 8)] + [(name, n, 16) for i, n in enumerate(counts) for name in CT.names(i)]
    head = C.Head(LINK, window, ZEROS, tuple(C.Entry(name, rows, size, ZEROS, ZEROS) for name, rows, size in listed))
    return {SV.SERVE: serve, SV.REGISTRATION: {
        "format": C.REGISTRATION, "link": LINK, "window": window, "parameters": ZEROS.hex(),
        "word_bits": C.WORD_BITS, "row_schema": C.SCHEMA,
        "tables": [{"name": e.name, "rows": e.rows, "words": e.row_bytes * 8 // C.WORD_BITS, "domain": e.domain.hex(),
                    "root": e.root.hex()} for e in head.tables],
        "certificate": {"key": "", "digest": head.digest().hex(), "signature": ""}}}


def four(window: int, cls: int = 0, **over: object) -> list[dict]:
    """The four verdicts' records on a class of an execution phase: each accepted, complete and on the serve."""
    return [{CC.KIND: kind, CC.SERVE: SERVE, CC.LINK: LINK, CC.WINDOW: window, CC.CLASS: cls, CC.RUN: f"run of the {kind}",
             CC.VERIFIER: "a verifier", CC.VERDICT: CC.ACCEPT, CC.COMPLETE: True, CC.PROVED_BY: None} | over
            for kind in CC.VERDICTS]


def test_a_head_is_the_registrations_own_and_held_to_the_digest_its_certificate_names():
    params = schedule()
    head = C.Head(LINK, 7, params.digest, tuple(C.Entry(name, rows, size, ZEROS, bytes([i]) * 64)
                                                for i, (name, rows, size) in enumerate(C.tables_of(params))))
    cert = Certificate(b"a key", LINK, Direction.EGRESS, 7, head.digest(), b"a signature")
    doc = json.loads(json.dumps(C.registration(head, cert, params)))
    assert SV.head_of(doc) == head and SV.classes(head) == list(range(len(CR.CLASSES)))
    assert SV.head_of(line(3, 5, 0)[SV.REGISTRATION]).table("request.0").rows == 5
    forged = doc | {"tables": [t | {"rows": t["rows"] + 1} if t["name"] == "request.0" else t for t in doc["tables"]]}
    with pytest.raises(ValueError, match="the registration's certificate doesn't name its head's digest"):
        SV.head_of(forged)
    for off in (doc | {"format": "another"}, {k: v for k, v in doc.items() if k != "certificate"},
                doc | {"parameters": "zz"}, doc | {"tables": [{}]}, [], None):
        with pytest.raises(ValueError, match="a registration of connection rows doesn't read"):
            SV.head_of(off)


def test_the_answer_is_below_the_schedule_only_with_every_held_class_verdicts_in_every_head_read():
    heads = [line(3, 5, 0), line(4, 2, 0)]
    got = SV.stop(heads, four(3) + four(4))
    assert (got.below, got.serve) == (True, SERVE)
    assert got.why == (
        "each connection is charged below its schedule: the 7 row(s) of 2 class(es) in the 2 execution phase(s) of "
        "the serve the serve read each have their 4 verdicts, accepted and complete",
        f"the link {LINK}, execution phase 3, {CC.charge(SV.head_of(heads[0][SV.REGISTRATION]), 0, SERVE, four(3)).why()}",
        f"the link {LINK}, execution phase 3, class 1: a count of 0, so no connection is charged and no verdict is owed",
        f"the link {LINK}, execution phase 4, {CC.charge(SV.head_of(heads[1][SV.REGISTRATION]), 0, SERVE, four(4)).why()}",
        f"the link {LINK}, execution phase 4, class 1: a count of 0, so no connection is charged and no verdict is owed")
    assert [v[CC.KIND] for v in got.verdicts] == list(CC.VERDICTS) * 2
    assert all(v["mark"] == LB.RUN_NOT_PROVED == "run, not proved" and set(v) == {*CC.FIELDS, "mark"}
               for v in got.verdicts)
    # a proof named in a record changes no mark here
    proved = SV.stop(heads[:1], four(3, proved_by="a theorem"))
    assert proved.below and {v["mark"] for v in proved.verdicts} == {LB.RUN_NOT_PROVED}
    # one execution phase without its tie keeps every connection at its whole schedule
    for records, lack in ((four(3) + four(4)[:1] + four(4)[2:], "the response tie is missing"),
                          (four(3) + four(4, complete=False)[1:2] + four(4)[:1] + four(4)[2:],
                           "the response tie is not complete (run of the response tie)"),
                          (four(3), "the stop record is missing; the response tie is missing; the certified read "
                                    "is missing; the compute audit is missing")):
        got = SV.stop(heads, records)
        assert not got.below and got.why[0] == (
            "each connection is charged its whole schedule: 1 of the 2 class(es) that hold a row in the 2 execution "
            "phase(s) of the serve the serve read lack a verdict")
        assert got.why[3] == (f"the link {LINK}, execution phase 4, class 0: charged its whole schedule on each of its "
                              f"2 rows: {lack}")
    # a second class that holds a row owes its own four
    assert not SV.stop([line(3, 5, 1)], four(3)).below and SV.stop([line(3, 5, 1)], four(3) + four(3, 1)).below
    # records of no head read are named and count for nothing
    stray = SV.stop(heads[:1], four(3) + four(9) + four(3, serve="another serve"))
    assert stray.below and stray.why[-1] == ("8 verdict record(s) read are of no class of a head read on that serve, "
                                             "and count for nothing")
    assert len(stray.verdicts) == 4
    # what was not read is carried
    assert SV.stop(heads[:1], four(3), ["r9 is not read"]).why[-1] == "r9 is not read"


def test_anything_less_is_the_whole_schedule_with_a_line_that_says_why():
    assert SV.stop([], []) == LB.NO_RECORDS
    assert SV.stop([], [], ["r9 is not read"]) == LB.Stop(False, (*LB.NO_RECORDS.why, "r9 is not read"))
    whole = "each connection is charged its whole schedule: "
    for heads, records, why in (
            ([], four(3), "4 verdict record(s) were read and no signed head (`stop-heads.jsonl`), so no class's "
                          "count is known"),
            ([line(3, 0, 0)], four(3), "no class holds a row in the 1 execution phase(s) of the serve the serve read"),
            ([line(3, 5), line(4, 5, serve="another serve")], four(3) + four(4),
             "the heads read are of 2 serves (another serve, the serve), and the verdicts are each from a run on one"),
            ([line(3, 5), line(3, 6)], four(3), f"two heads are given for execution phase 3 of the link {LINK}"),
            ([line(3, 5, serve="")], four(3), "a head line names no serve: ''"),
            ([line(3, 5, serve=7)], four(3), "a head line names no serve: 7"),
            ([{SV.SERVE: SERVE}], four(3), "a head line doesn't read: 'registration'"),
            ([line(3, 5)], [*four(3), "accept"],
             f"a verdict record is an object with the fields {', '.join(CC.FIELDS)}, not 'accept'")):
        got = SV.stop(heads, records)
        assert (got.below, got.verdicts, got.why[0]) == (False, (), whole + why), why
    tampered = line(3, 5)
    tampered[SV.REGISTRATION]["tables"][2]["rows"] = 0
    got = SV.stop([tampered], four(3))
    assert not got.below and got.why[0] == (whole + "a head line doesn't read: the registration's certificate "
                                            "doesn't name its head's digest")
    # the same head given twice is one head
    assert SV.stop([line(3, 5), line(3, 5)], four(3)).below


class Store:
    """The evidence store as `read` uses it: each run's files, by name."""

    def __init__(self, tmp: Path, runs: dict[str, dict[str, str] | None]):
        self.tmp, self.runs = tmp, runs

    def get_attempt(self, run: str) -> dict | None:
        if run not in self.runs:
            return None
        return {"outputs": {"run_files": run} if self.runs[run] is not None else {}}

    def listing(self, files: str, names: list[str]) -> list[str]:
        return [n for n in names if n in self.runs[files]]

    def fetch(self, files: str, paths: list[str]) -> str:
        at = self.tmp / files
        at.mkdir(exist_ok=True)
        for name in paths:
            (at / name).write_text(self.runs[files][name])
        return str(at)


def jsonl(lines: list) -> str:
    return "".join(json.dumps(x) + "\n" for x in lines)


def test_the_run_files_are_read_from_the_store_and_no_run_is_the_whole_schedule(tmp_path):
    assert (SV.RECORD, SV.RECORDS_FILE, SV.HEADS_FILE) == (
        "exfiltration-glm53-stop-verdicts", "stop-verdicts.jsonl", "stop-heads.jsonl")
    heads, records = [line(3, 5, 0)], four(3)
    store = Store(tmp_path, {
        "r1": {SV.RECORDS_FILE: jsonl(records[:3]) + "\n"},
        "r2": {SV.RECORDS_FILE: jsonl(records[3:]), SV.HEADS_FILE: jsonl(heads), "another.json": "{}"},
        "r3": {"another.json": "{}"}, "r4": None})
    assert SV.read(store, ["r1", "r2"]) == (heads, records, [])
    got = SV.of_runs(store, ["r1", "r2", "r3", "r4", "r5"])
    assert got.below and len(got.verdicts) == 4 and got == SV.stop(heads, records, [
        "r3 is not read: it holds neither `stop-verdicts.jsonl` nor `stop-heads.jsonl`",
        "r4 is not read: it kept no run files", "r5 is not read: it is not in the store or on its remote"])
    # the tie's run alone: three verdicts, so the whole schedule
    assert not SV.of_runs(store, ["r2"]).below
    assert SV.of_runs(store, ["r1"]).why[0].endswith("3 verdict record(s) were read and no signed head "
                                                      "(`stop-heads.jsonl`), so no class's count is known")
    none = SV.of_runs(store, [])
    assert none == LB.Stop(False, (*LB.NO_RECORDS.why, "no run in the store carries "
                                   "`run_of_record=exfiltration-glm53-stop-verdicts`"))
    assert SV.of_runs(store, ["r3"]) == LB.Stop(False, (*LB.NO_RECORDS.why, "r3 is not read: it holds neither "
                                                        "`stop-verdicts.jsonl` nor `stop-heads.jsonl`"))
    broken = Store(tmp_path, {"r6": {SV.RECORDS_FILE: "{not json\n"}})
    got = SV.of_runs(broken, ["r6"])
    assert not got.below and got.why[0].startswith("each connection is charged its whole schedule: the records of r6 "
                                                   "don't read: ")
    # the calculator's design row follows the answer
    doc = LB.document(LB.inputs(), stop=SV.of_runs(store, ["r1", "r2"]), only=["world"], periods=["year"])
    by = {r["row"]: r for r in doc["rows"]}
    assert doc["stop"]["below"] and by["design"]["schedules"]["bits"] == 0 < by["design, no stop block"]["schedules"]["bits"]
    doc = LB.document(LB.inputs(), stop=SV.of_runs(store, ["r2"]), only=["world"], periods=["year"])
    by = {r["row"]: r for r in doc["rows"]}
    assert by["design"]["term"] == by["design, no stop block"]["term"] and len(doc["stop"]["verdicts"]) == 1
