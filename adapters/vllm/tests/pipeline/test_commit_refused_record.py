"""`verity_vllm.pipeline.commit`: what `main` makes of a run's compare against the Match record where the compare raises.

A record the compare's reads refuse by name (`records.fields.RecordError`) fails the run's compare, and the verdict fails the
run on the refusal, which names the record and the field; anything else a compare raises leaves it not run, with no result.

Real: all that `tests.pipeline.test_commit_main` runs for real; the Match record's reader and the compare
(`check.match_oracle.MatchOracle`, `check.oracle_compare.oracle_compare`) on a record written to disk; and the run's verdict,
v1's from `main` (`check.commit_pass.commit_verdict`) and, on a run that passes otherwise, v1's and the rules'
(`check.verdict.run_checks`).

Stood in: that module's stand-ins; the binding record, which writes the Match record's binding map for `main` to read back;
the manifest's producers, which are those of the manifest the Match record's promoted member is named by; the producer
facts of the Programs of record, of which there are none; and the opened values' reader, which answers the bytes a test
commits.

The Match record is `tests.check.test_oracle_compare`'s with `tests.check.test_oracle_compare_promoted_key`'s promoted
member, whose producer the reader finds through the output views of every fold instance of the step.
CPU only."""
from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

from tests.check.test_oracle_compare import U0, U1, make_match
from tests.check.test_oracle_compare_promoted_key import (
    _add_promoted,
    _manifest,
    _promoted_bmap,
)
from tests.check.test_verdict import _base, _both
from tests.pipeline import test_commit_main as M
from tests.pipeline.test_commit_main import Ended, Record, oracle_compares, run
from verity_vllm.check import oracle_compare
from verity_vllm.pipeline import commit, commit_plan
from verity_vllm.records.fields import MissingField, RecordError

#: `test_commit_main`'s fixture, under its name in this module.
stand_ins = M.stand_ins

#: The real verdict's word on a run of these Commits that it finds nothing else against: they name no Program to replay.
NO_C2 = "C2 of record not established: sampled replay None -- not run (no --program-dir / --programs-root); the row's Commit is C1 + C3 only"
#: The opened values' record as far as `main` reads it.
OPENED: Record = {"reads": 3, "leaves": 3, "bytes": 320, "seconds": 0.5, "steps_bound": 3, "not_retained": 0, "failed": 0}
#: The output view of a fold instance of an op that no binding names: it states an allocation that is no number.
VIEW: Record = {"alloc": "thirteen", "offset": 0, "shape": [4], "strides": [1], "dtype": "bfloat16", "device": "cuda:0"}
#: What the Match record's reader raises on `VIEW`.
REFUSAL = RecordError("a fold instance's view", "alloc", f"is 'thirteen', not a number (the view: {json.dumps(VIEW)})")
#: The committed member a test changes a byte of, and the verdict's first words on it.
MEMBER = (0, "model.layers.0.self_attn.qkv_proj/0")
DIFFERS = "C2 oracle mismatch: 1 identities differ from the Match oracle (first: step 0 model.layers.0.self_attn.qkv_proj/0 inv 0 req r1 "


def refused(raised: Exception) -> Record:
    """The compare `main` records where a compare raised the refusal ``raised``."""
    return {"schema": "verity-sweep/oracle-compare/v1", "method": "oracle-compare", "result": False,
            "why": f"oracle compare refused a record: {raised}"[:300], "compared": 0, "mismatch_n": 0, "error": repr(raised)[:300]}


def not_run(raised: Exception) -> Record:
    """The compare `main` records where a compare raised anything else."""
    return {"schema": "verity-sweep/oracle-compare/v1", "method": "oracle-compare", "result": None,
            "why": f"oracle compare not run: {raised!r}"[:300], "compared": 0, "mismatch_n": 0, "error": repr(raised)[:300]}


def renamed(text: str) -> str:
    """``text`` of the Match record with its two requests under `M.WORKLOAD`'s ids."""
    return text.replace(U0, "a").replace(U1, "b")


def match_record(tmp_path: Path, *, view: bool) -> tuple[str, Record, dict[tuple[int, str], bytes]]:
    """The Match record of the module docstring, its two requests under `M.WORKLOAD`'s ids; with ``view``, one more fold instance
    whose one output view is `VIEW`.  Returns the record's directory, its binding map and the committed bytes that equal it."""
    d, values = make_match(str(tmp_path))
    bmap, committed = _promoted_bmap(values, _add_promoted(d))
    instances = Path(d) / "instances.jsonl"
    if view:
        request = json.loads(instances.read_text().splitlines()[1])["request"]
        with instances.open("a") as f:
            f.write(json.dumps({"node": 0, "step": 0, "event": 99, "pattern": "x", "spec": "X_v1{}", "row": 0, "rows": 8, "name": "t=0/other",
                                "hint": "model.other/model.other.weight", "request": request, "position": 0, "operand_views": [],
                                "output_views": [VIEW]}) + "\n")
    for path in (instances, Path(d) / "capture" / "tokens.json"):
        path.write_text(renamed(path.read_text()))
    log = Path(d) / "capture" / "log.jsonl.gz"
    with gzip.open(log, "rt") as f:
        text = f.read()
    with gzip.open(log, "wt") as f:
        f.write(renamed(text))
    return d, bmap, committed


@pytest.fixture
def compared(stand_ins, monkeypatch):
    """`stand_ins` for a Commit whose runs are compared against a Match record on disk, with the stand-ins of the module
    docstring and a committer that declares every class the verdict requires.  ``bmap`` is the binding map the binding record
    writes and ``committed`` the bytes the opened values' reader answers, each for a test to state."""
    s = stand_ins
    s.bmap, s.committed = {}, {}
    M.MAKE.update(declare=M.CLASSES)
    producers_of = oracle_compare.producers_of

    def binding_record(com, rc, reqs, sampling, args, pair, **kw):
        (Path(args.out) / f"binding_map_p{pair}.json").write_text(json.dumps(s.bmap))
        return copy.deepcopy(M.BINDING)

    class Reader:
        def __init__(self, com: Any, rc: Any) -> None:
            pass

        def __call__(self, step: int, name: str, lo: int, hi: int) -> bytes | None:
            held = s.committed.get((step, name))
            return None if held is None else held[lo:hi]

        def record(self) -> Record:
            return dict(OPENED)

    monkeypatch.setattr(commit, "binding_record", binding_record)
    monkeypatch.setattr(commit_plan, "derived", lambda args, manifest, reqs, sampling: (None, {"used": False, "why": "no Program of record"}))
    monkeypatch.setattr(oracle_compare, "producers_of", lambda manifest: producers_of(_manifest()))
    monkeypatch.setattr(oracle_compare, "OpenedReader", Reader)
    return s


def instrumented(ended: Ended) -> list[Record]:
    return [row["commit"] for row in ended.rows() if row["arm"] == "instrumented"]


@pytest.mark.parametrize(("view", "differs"), [(False, False), (False, True), (True, False), (True, True)],
                         ids=["no-such-view-and-equal-values", "no-such-view-and-a-differing-value", "the-view-and-equal-values",
                              "the-view-and-a-differing-value"])
def test_a_fold_view_that_states_no_number_fails_the_runs_compare_and_the_verdict_names_the_view(tmp_path, compared, capsys, view, differs):
    """A fold instance no binding names states an allocation that is no number, in a Match record whose promoted member makes
    the reader look at every instance's output views.  The reader refuses the view by name, and each run's compare is recorded
    as failed on that refusal, whether or not a committed value differs from the record's: v1 fails the run on it, and so do
    v1 and the rules on a run that passes otherwise.  Without the view the same record compares, equal or not."""
    s = compared
    d, s.bmap, s.committed = match_record(tmp_path, view=view)
    if differs:
        s.committed[MEMBER] = bytes([s.committed[MEMBER][0] ^ 1]) + s.committed[MEMBER][1:]
    ended = run(tmp_path, capsys, "--match-dir", d)
    compares = oracle_compares(ended)
    why = compares[0]["why"]
    if view:
        assert compares == [refused(REFUSAL)] * 2 and why.startswith("oracle compare refused a record: a fold instance's view: field 'alloc' is 'thirteen'")
    else:
        assert [(c["result"], c["complete"], c["compared"], c["equal"], c["mismatch_n"]) for c in compares] == [
            (not differs, not differs, 14, 14 - differs, int(differs))] * 2
        assert why.startswith(DIFFERS) if differs else why is None
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


@pytest.mark.parametrize(("raised", "record"), [
    (RecordError("a record", "its.field", "is a str, not a int"), refused),
    (MissingField("a record", "its.field"), refused),
    (ValueError("invalid literal for int() with base 10: 'thirteen'"), not_run),
    (KeyError("alloc"), not_run),
], ids=["a-refusal-by-name", "a-missing-field", "a-value-error-that-names-no-record", "a-key-error"])
def test_what_a_compare_raises_fails_the_run_only_where_it_refuses_a_record_by_name(tmp_path, compared, capsys, monkeypatch, raised, record):
    """A refusal by name is a `ValueError`, and only it fails the run's compare: any other error, a `ValueError` among them,
    leaves the compare not run, with no result, and the value correspondence as it was."""
    def raises(*args: Any, **kw: Any) -> Record:
        raise raised

    s = compared
    monkeypatch.setattr(oracle_compare, "oracle_compare", raises)
    d, s.bmap, s.committed = match_record(tmp_path, view=False)
    s.verdicts = [(True, None)]
    ended = run(tmp_path, capsys, "--match-dir", d)
    assert oracle_compares(ended) == [record(raised)] * 2
    assert [c["value_correspondence"]["result"] for c in instrumented(ended)] == [record(raised)["result"]] * 2
    assert ended.exit is None
