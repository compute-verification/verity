"""The request a manifest is of (`query.required.request_manifest`): its label is `r<k>`, k the request's position in the workload
manifest, and k picks the workload id and the arrival step the manifest carries.  A position below zero is no request's, so the
manifest is refused, whether the position comes from the workload's first request or from the caller."""
from __future__ import annotations

import pytest

from tests.query.test_contested_members import RES, _program
from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query.manifest.format import request_label
from verity_vllm.query.required import request_manifest


def _workload(first_index: int = 0) -> dict:
    return {"sweep": {"batch": 3, "arrive_steps": [0, 0, 7]},
            "requests": [{"request_id": "first", "index": first_index}, {"request_id": "second", "index": 1},
                         {"request_id": "last", "index": 2}]}


def _manifest(request_id, workload: dict) -> dict:
    P, _ = _program()
    return request_manifest(P, Correspondence(P), RES, request_id=request_id, workload=workload).manifest


@pytest.mark.parametrize("request_id, label, workload_id, arrive",
                         [(None, "r0", "first", 0), (1, "r1", "second", 0), ("r2", "r2", "last", 7), ("last", "r2", "last", 7)])
def test_the_requests_position_picks_its_workload_id_and_arrival_step(request_id, label, workload_id, arrive):
    man = _manifest(request_id, _workload())
    assert {r["request_id"] for r in man["identities"]} == {label}
    assert (man["workload"]["workload_request_id"], man["workload"]["arrive_step"], man["engine_steps"]) == (workload_id, arrive, [arrive])


@pytest.mark.parametrize("request_id, first_index", [(None, -1), ("first", -1), (-1, 0)])
def test_a_position_below_zero_is_refused(request_id, first_index):
    """`request_label` answers `r-1` for it; read as a position, -1 would be the workload's last request."""
    with pytest.raises(ValueError, match="'r-1' is not `r<k>`"):
        _manifest(request_id, _workload(first_index))


def test_a_request_named_by_no_workload_is_its_position_or_the_first():
    assert [request_label(rid) for rid in (None, 0, 3, "3", "007", "r3")] == ["r0", "r0", "r3", "r3", "r7", "r3"]
    workloads: tuple[dict, ...] = ({}, {"requests": []}, {"requests": ["first"]}, {"requests": [{"index": 4}]})
    assert [request_label(None, w) for w in workloads] == ["r0", "r0", "r0", "r4"]
    P, _ = _program()
    man = request_manifest(P, Correspondence(P), RES).manifest
    assert {r["request_id"] for r in man["identities"]} == {"r0"} and (man["requests"], man["workload"]) == (["r0"], None)
    assert _manifest("2", _workload())["workload"] == {"row_id": None, "batch": 3, "request_id": "r2", "workload_request_id": "last", "arrive_step": 7}


@pytest.mark.parametrize("request_id, why", [
    (True, "^request id True: not a request index$"), (False, "^request id False: not a request index$"),
    ("third", r"^request id 'third' is not `r<k>` and is not a request of the workload manifest \(a workload UUID is not a request id\)$"),
    ("r 1", "^request id 'r 1' is not `r<k>` and is not a request"), (1.0, "^request id '1.0' is not `r<k>` and is not a request")])
def test_an_id_that_is_no_position_and_no_request_of_the_workload_is_refused(request_id, why):
    with pytest.raises(ValueError, match=why):
        request_label(request_id, _workload())
    with pytest.raises(ValueError, match=why):
        _manifest(request_id, _workload())
