"""What a Commit makes of a Match record's view that a compared member is read through, by the view's shape, strides and dtype.

The Match record's reader (`verity_vllm.check.match_oracle.MatchOracle`) reads a compared member's bytes through one view of a
fold instance.  A view that leaves its shape or its dtype out, or states there or for its strides what is no list of numbers
and no dtype the recorder writes, is refused by name, and the Commit driver's `main` fails each run's compare on the refusal,
whether or not a committed value differs from the record's: v1 fails the run on it, and so do v1 and the rules on a run that
passes otherwise.  A view that states no strides is contiguous and its member is compared.  A member read through a view of a
dtype the recorder writes and the reader once had no size for is compared too: a committed byte of it that differs fails the
run, where the member was left out of the compare and the run passed.

Real and stood in: as `tests.pipeline.test_commit_refused_record` states, whose Match record, stand-ins and call of `main`
these tests take.  The view is the second output view of an instance of the norm, which the reader reads the norm's second
output through.
CPU only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

from tests.check.test_absent_commit_fields import ABSENT, put
from tests.check.test_absent_oracle_fields import (
    A_DTYPE,
    A_LIST,
    NORM,
    bound_in_dtype,
    in_dtype,
)
from tests.check.test_verdict import _base, _both
from tests.pipeline import test_commit_main as M
from tests.pipeline import test_commit_refused_record as R
from tests.pipeline.test_commit_main import Ended, Record, oracle_compares, run
from tests.pipeline.test_commit_refused_record import (
    DIFFERS,
    MEMBER,
    NO_C2,
    instrumented,
    match_record,
    refused,
)
from verity_vllm.records.fields import RecordError

#: The fixtures a test here takes: a Commit whose runs are compared against a Match record on disk, and the stand-ins under it.
compared, stand_ins = R.compared, M.stand_ins

#: The committed member the reader reads through the norm's second output view, and the verdict's first words where it differs.
THROUGH_THE_VIEW = (0, NORM + "/1")
ITS_BYTE_DIFFERS = f"C2 oracle mismatch: 1 identities differ from the Match oracle (first: step 0 {NORM}/1 inv 0 req r1 "
#: The dtypes the recorder writes that the reader had no size for.
ONCE_UNSIZED = ["complex64", "float8_e8m0fnu", "uint16", "uint32", "uint64"]


def restated(d: str, change: Any) -> list[Record]:
    """The fold's instances of the Match record at ``d`` after ``change`` edited the file's lines, written back."""
    path = Path(d) / "instances.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    change(lines)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return lines


def the_view(lines: list[Record]) -> Record:
    """The second output view of the first instance of the norm."""
    return next(m for m in lines[1:] if m["hint"].startswith(NORM + "/"))["output_views"][1]


def one_byte_off(committed: dict[tuple[int, str], bytes], member: tuple[int, str]) -> None:
    committed[member] = bytes([committed[member][0] ^ 1]) + committed[member][1:]


def holds(ended: Ended, why: str | None) -> None:
    """Each instrumented run's record carries ``why`` as its compare's reason, and v1 and the rules fail the run on it; with no
    reason the run fails only on what these Commits never establish, and the rules pass a run that passes otherwise."""
    for record in instrumented(ended):
        assert record["value_correspondence"]["result"] is (why is None)
        assert record["value_correspondence"]["oracle_compare"]["why"] == why
        assert (record["commit_pass"], record["commit_fail_reason"]) == (False, why or NO_C2)
        (ok, reason), outcome, checks = _both({**_base(), "oracle_compare": record["oracle_compare"],
                                               "value_correspondence": record["value_correspondence"]})
        [rule] = [c for c in checks if c.name == "runtime_match/oracle_compare"]
        assert ok is (why is None) and (ok or reason == why)
        assert (outcome, rule.outcome, rule.reason) == (("PASS", "PASS", None) if why is None else ("FAIL", "FAIL", why))
    assert ended.exit == 3 and ended.printed[-1] == M.FAIL_LINE.format(failed=2, n=2, pair=0, arm="instrumented", why=why or NO_C2)


@pytest.mark.parametrize("differs", [False, True], ids=["equal-values", "a-differing-value"])
@pytest.mark.parametrize(("field", "stated", "kind"), [
    ("shape", ABSENT, A_LIST), ("shape", None, A_LIST), ("shape", "thirteen", A_LIST), ("shape", ["two"], A_LIST),
    ("strides", "thirteen", A_LIST), ("strides", ["two"], A_LIST),
    ("dtype", ABSENT, A_DTYPE), ("dtype", None, A_DTYPE), ("dtype", "thirteen", A_DTYPE), ("dtype", ["two"], A_DTYPE),
], ids=["no-shape", "a-null-shape", "a-shape-of-text", "a-shape-that-lists-text", "strides-of-text", "strides-that-list-text",
        "no-dtype", "a-null-dtype", "a-dtype-the-recorder-does-not-write", "a-dtype-that-is-a-list"])
def test_a_members_view_that_states_no_shape_strides_or_dtype_the_reader_takes_fails_the_runs_compare_by_name(
        tmp_path, compared, capsys, field, stated, kind, differs):
    s = compared
    d, s.bmap, s.committed = match_record(tmp_path, view=False)
    view = the_view(restated(d, lambda lines: put(the_view(lines), (field,), stated)))
    if differs:
        one_byte_off(s.committed, MEMBER)
    ended = run(tmp_path, capsys, "--match-dir", d)
    refusal = RecordError("a fold instance's view", field, f"is {view.get(field)!r}, not {kind} (the view: {json.dumps(view)[:200]})")
    compares = oracle_compares(ended)
    assert compares == [refused(refusal)] * 2
    assert compares[0]["why"].startswith(f"oracle compare refused a record: a fold instance's view: field '{field}' is {view.get(field)!r}, not {kind}")
    holds(ended, compares[0]["why"])


@pytest.mark.parametrize("differs", [False, True], ids=["equal-values", "a-differing-value"])
@pytest.mark.parametrize("stated", [ABSENT, None], ids=["no-strides", "null-strides"])
def test_a_members_view_that_states_no_strides_is_contiguous_and_its_member_is_compared(tmp_path, compared, capsys, stated, differs):
    s = compared
    d, s.bmap, s.committed = match_record(tmp_path, view=False)
    restated(d, lambda lines: put(the_view(lines), ("strides",), stated))
    if differs:
        one_byte_off(s.committed, THROUGH_THE_VIEW)
    ended = run(tmp_path, capsys, "--match-dir", d)
    compares = oracle_compares(ended)
    assert [(c["result"], c["complete"], c["compared"], c["equal"], c["mismatch_n"]) for c in compares] == [
        (not differs, not differs, 14, 14 - differs, int(differs))] * 2
    why = compares[0]["why"]
    assert why.startswith(ITS_BYTE_DIFFERS) if differs else why is None
    holds(ended, why)


@pytest.mark.parametrize("differs", [None, MEMBER, THROUGH_THE_VIEW], ids=["equal-values", "another-members-byte-differs", "its-byte-differs"])
@pytest.mark.parametrize("dtype", ONCE_UNSIZED)
def test_a_member_read_through_a_view_of_a_dtype_the_reader_once_had_no_size_for_is_compared(tmp_path, compared, capsys, dtype, differs):
    """The norm's second output, restated in ``dtype`` over the bytes it has, in the fold's views and in the binding map."""
    s = compared
    d, s.bmap, s.committed = match_record(tmp_path, view=False)
    restated(d, in_dtype(dtype))
    bound_in_dtype(s.bmap, dtype)
    if differs:
        one_byte_off(s.committed, differs)
    ended = run(tmp_path, capsys, "--match-dir", d)
    compares = oracle_compares(ended)
    assert [(c["result"], c["complete"], c["compared"], c["equal"], c["mismatch_n"]) for c in compares] == [
        (not differs, not differs, 14, 14 - bool(differs), int(bool(differs)))] * 2
    why = compares[0]["why"]
    assert why is None if not differs else why.startswith(ITS_BYTE_DIFFERS if differs == THROUGH_THE_VIEW else DIFFERS)
    holds(ended, why)
