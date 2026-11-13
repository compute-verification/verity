"""What `verity_vllm.pipeline.c2_replay` reads of a block its record may leave out: a row's schedule of record, a loaded
Program's Build result, and the manifest's digests of record.  A block that is null or empty reads as one that is absent.
CPU only: the Programs are stand-ins and no Definition is checked (`tests/check/test_replay_definitions.py` loads real ones)."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import verity_vllm.pipeline.c2_replay as C2

Record = dict[str, Any]

REQS: list[Record] = [{"request_id": "a", "prompt_token_ids": [1, 9], "max_tokens": 4}, {"request_id": "b", "prompt_token_ids": [1], "arrive_step": 1}]
LABELS = {"a": "r0", "b": "r1"}
OBSERVED: Record = {"use": "observed", "observed_arrive_steps": {"r0": 0, "r1": 2}, "declared_arrive_steps": {"r0": 0, "r1": 1}}
#: a row that is not a sampling event: one step of its Program
ROWS: list[Record] = [{"i": 0, "spec": "NotReplayed_v1{request=0}"}]


def arrivals(row: Record) -> tuple[dict[str, int], bool]:
    return C2.arrivals(SimpleNamespace(reqs=REQS, row=row), LABELS)


@pytest.mark.parametrize("row", [
    {}, {"commit": None}, {"commit": {}}, {"commit": {"binding": None}}, {"commit": {"binding": {}}},
    {"commit": {"binding": {"coverage": None}}}, {"commit": {"binding": {"coverage": {}}}},
    {"commit": {"binding": {"coverage": {"schedule": None}}}}, {"commit": {"binding": {"coverage": {"schedule": {}}}}},
    {"commit": {"binding": {"coverage": {"schedule": OBSERVED | {"use": "declared"}}}}}])
def test_a_row_that_records_no_observed_schedule_leaves_each_request_at_its_declared_arrival(row: Record) -> None:
    assert arrivals(row) == ({"r0": 0, "r1": 1}, False)


def test_a_row_whose_schedule_of_record_is_observed_moves_the_arrivals_to_it() -> None:
    assert arrivals({"commit": {"binding": {"coverage": {"schedule": OBSERVED}}}}) == ({"r0": 0, "r1": 2}, True)


@pytest.mark.parametrize("result", [None, {}, {"T": None}])
def test_a_program_whose_build_result_names_no_t_takes_its_steps_from_its_rows_and_none_where_it_has_no_rows(result: Record | None) -> None:
    loaded: Record = {"r0": {"result": {"T": 3}}, "r1": {"result": result}, "r2": {}, "r3": {"result": result}, "r4": {"result": {"T": 0}}}
    progs: Record = {"r0": {"rows": ROWS}, "r1": {"rows": ROWS}, "r2": {"rows": ROWS, "checks": None}}
    assert C2.program_steps(loaded, progs) == {"r0": 4, "r1": 1, "r2": 1, "r3": None, "r4": 1}


def loads(monkeypatch: pytest.MonkeyPatch, program: Record, manifest: Any, digest: str | None = None) -> tuple[Record, dict[str, str]]:
    """`programs_of_record` for the one request `a`, whose Program dir loads as ``program``."""
    monkeypatch.setattr(C2.MF, "load_program", lambda d: program)
    args = SimpleNamespace(program_digest=digest, programs_root=None, program_dir="/programs/r0")
    return C2.programs_of_record(C2.ProgramSources(args, manifest, REQS[:1], {}), C2.request_labels(REQS[:1]), definitions=False)


BUILT: Record = {"dir": "/programs/r0", "result": {"program_digest": "d1"}, "instances": {"digest": "i1", "rows": ROWS}}


@pytest.mark.parametrize("manifest, digest", [
    ({"component_digests": ["d1"]}, None), ({"component_digests": ["d0", "d1"], "components": None}, None),
    ({"components": {"x": {"program_digest": "d1"}, "y": "no block", "z": {}}}, None),
    ({"component_digests": None, "components": {"x": {"program_digest": "d1"}}}, None),
    (None, "d1"), ({}, "d1"), ({"component_digests": None, "components": None}, "d1"), ({"component_digests": [], "components": {}}, "d1"),
    (["not a manifest"], "d1")])
def test_a_program_is_of_record_when_the_program_digest_or_the_manifest_names_its_digest(
        monkeypatch: pytest.MonkeyPatch, manifest: Any, digest: str | None) -> None:
    assert loads(monkeypatch, BUILT, manifest, digest) == ({"r0": BUILT}, {"r0": "d1"})


@pytest.mark.parametrize("manifest", [None, {}, {"component_digests": None, "components": None}, {"component_digests": [], "components": {}},
                                      {"components": {"x": {"program_digest": None}, "y": "no block"}}, ["not a manifest"]])
def test_a_replay_with_no_digest_of_record_is_refused(monkeypatch: pytest.MonkeyPatch, manifest: Any) -> None:
    with pytest.raises(RuntimeError, match="^no Program digest of record to check the replay Programs against"):
        loads(monkeypatch, BUILT, manifest)


@pytest.mark.parametrize("program, digest", [
    (BUILT | {"result": None}, "i1"), (BUILT | {"result": {}}, "i1"), (BUILT | {"result": {"program_digest": None}}, "i1"),
    ({"dir": "/programs/r0", "instances": {"digest": "i1"}}, "i1"), (BUILT | {"instances": None}, "d1"), ({"dir": "/programs/r0", "result": {"program_digest": "d1"}}, "d1")])
def test_a_program_s_digest_is_its_build_result_s_else_its_instances(monkeypatch: pytest.MonkeyPatch, program: Record, digest: str) -> None:
    assert loads(monkeypatch, program, {"component_digests": [digest]})[1] == {"r0": digest}


@pytest.mark.parametrize("program", [{"dir": "/programs/r0"}, {"dir": "/programs/r0", "result": None, "instances": None},
                                     {"dir": "/programs/r0", "result": {}, "instances": {}}])
def test_a_program_that_records_no_digest_is_not_of_record(monkeypatch: pytest.MonkeyPatch, program: Record) -> None:
    with pytest.raises(RuntimeError) as refused:
        loads(monkeypatch, program, {"component_digests": ["d1"]})
    assert str(refused.value) == "replay Program(s) are not the Program of record: r0:  (/programs/r0) vs of-record ['d1'] -- refused"


def test_a_program_the_manifest_does_not_name_is_refused_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError) as refused:
        loads(monkeypatch, BUILT, {"component_digests": ["d2"], "components": {"x": {"program_digest": "d3"}}}, "d4")
    assert str(refused.value) == "replay Program(s) are not the Program of record: r0: d1 (/programs/r0) vs of-record ['d2', 'd3', 'd4'] -- refused"
