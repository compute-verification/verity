"""`verity_vllm.pipeline.tp.rank_worker`: the workload's label of an engine row id on a rank
(`TP2CaptureWorkerExtension._tp2_logical_request`).  The rank's Commit state maps the engine's ids to the workload's `r<i>`,
by the driver's map (`request_ids`) when it has one and else by the order the requests were submitted in (`request_order`).
An id is matched exactly, then with vLLM's `-<8 hex>` suffix stripped, then by the one mapped id it starts with; an id that
matches none, or more than one, stays raw.  A state with neither field maps nothing.  The swap negative relabels after that."""
from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("torch")

from verity_vllm.pipeline.tp.rank_worker import (
    TP2CaptureWorkerExtension,
    parse_swap_spec,
)

ABSENT: Any = object()


class _Rank(TP2CaptureWorkerExtension):
    def __init__(self, **state: Any) -> None:
        self._tp2_commit = {k: v for k, v in state.items() if v is not ABSENT}


@pytest.mark.parametrize("rid, label", [("q-a", "r0"), ("q-a-1a2b3c4d", "r0"), ("q-b-9f8e7d6c", "r1"), ("q-bx", "r1"), ("q-", "q-"), ("zzz", "zzz"),
                                        (17, "17")],
                         ids=["exact", "suffix stripped", "the other request", "the one id it starts with", "no id it starts with", "unmapped",
                              "not a string"])
def test_an_id_is_matched_exactly_then_without_its_suffix_then_by_the_one_id_it_starts_with(rid, label):
    assert _Rank(request_ids={"q-a": "r0", "q-b": "r1"})._tp2_logical_request(rid) == label


def test_an_id_that_starts_with_two_mapped_ids_stays_raw():
    w = _Rank(request_ids={"q-a": "r0", "q-ab": "r1"})
    assert [w._tp2_logical_request(r) for r in ("q-abc", "q-ab", "q-ax")] == ["q-abc", "r1", "r0"]


def test_the_id_without_its_suffix_is_tried_before_the_ids_it_starts_with():
    w = _Rank(request_ids={"q": "r0", "q-a": "r1"})
    assert [w._tp2_logical_request(r) for r in ("q-a-1a2b3c4d", "q-1a2b3c4d", "q-a-b-1a2b3c4d")] == ["r1", "r0", "q-a-b-1a2b3c4d"]


@pytest.mark.parametrize("ids", [ABSENT, None, {}], ids=["no map", "map none", "map empty"])
def test_without_the_drivers_map_a_request_is_labelled_by_its_place_in_the_order(ids):
    w = _Rank(request_ids=ids, request_order=["q-a", 7, "q-c"])
    assert [w._tp2_logical_request(r) for r in ("q-a", "7", "q-c-00000000", "q-d")] == ["r0", "r1", "r2", "q-d"]


def test_the_drivers_map_wins_over_the_order():
    w = _Rank(request_ids={"q-a": "r5"}, request_order=["q-b", "q-a"])
    assert (w._tp2_logical_request("q-a"), w._tp2_logical_request("q-b")) == ("r5", "q-b")


@pytest.mark.parametrize("ids", [ABSENT, None, {}], ids=["no map", "map none", "map empty"])
@pytest.mark.parametrize("order", [ABSENT, None, []], ids=["no order", "order none", "order empty"])
def test_a_state_with_neither_maps_nothing(ids, order):
    assert _Rank(request_ids=ids, request_order=order)._tp2_logical_request("q-a-1a2b3c4d") == "q-a-1a2b3c4d"


@pytest.mark.parametrize("spec, step, labels", [
    (None, 1, ["r0", "r1", "r2"]), (["r0", "r1"], None, ["r1", "r0", "r2"]), (["r0", "r1"], 9, ["r1", "r0", "r2"]),
    (["r0", "r1", 2, 3], 2, ["r1", "r0", "r2"]), (["r0", "r1", 2, 3], 4, ["r0", "r1", "r2"]), (["r0", "r1", 2, 3], None, ["r0", "r1", "r2"]),
], ids=["no swap", "every step, no step given", "every step", "a step inside the range", "a step outside it", "a range, no step given"])
def test_the_swap_negative_relabels_the_two_requests_on_the_steps_it_names(spec, step, labels):
    w = _Rank(request_order=["q-a", "q-b", "q-c"], swap_rule=parse_swap_spec(spec))
    assert [w._tp2_logical_request(r, step) for r in ("q-a", "q-b", "q-c")] == labels
