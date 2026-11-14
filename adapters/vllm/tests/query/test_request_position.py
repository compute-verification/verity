"""The request a manifest is of (`query.required.request_manifest`): its label is `r<k>`, k the request's position in the workload
manifest, and k picks the workload id and the arrival step the manifest carries.  A position below zero is no request's, so the
manifest is refused, whether the position comes from the workload's first request or from the caller."""
from __future__ import annotations

import pytest

from tests.query.test_contested_members import RES, _program
from verity_vllm.correspondence.reader_for_query import Correspondence
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
