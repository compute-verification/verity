"""`edge`: a period's edge holds the requests it must leave to the next, in close order, each one's history rows after the ones
before it and padded to a power of two; the next period's carried requests read their slices only when their counters follow
from the close, and are refused by name when they don't."""
from __future__ import annotations

import pytest

from verity_vllm.pipeline import edge as E
from verity_vllm.pipeline import window_advice as WA


def _step(*rows: tuple[int, int, int]) -> WA.StepAdvice:
    return WA.StepAdvice(tuple(WA.Row(r, "decode" if n == 1 and s >= 3 else "prefill", s, n, 0, False, ()) for r, s, n in rows))


A, B, C = WA.RequestAdvice("a", 3, 4, "length"), WA.RequestAdvice("b", 3, 3, "length"), WA.RequestAdvice("c", 3, 1, "eos")
# period w admits a, b and c (prompts of 3); a keeps 2 outputs, b 1 and c its 1, in w's last step: a and b must carry, c may
W = WA.Advice((A, B, C), (_step((0, 0, 3), (1, 0, 3)), _step((0, 3, 1), (2, 0, 3))), (0, 0, 1, 1))
AT_A, AT_B = WA.Carried(0, 4, 2, ((0, 4),)), WA.Carried(0, 3, 1, ((0, 3),))
# period w + 1 carries both on and keeps an output of each
NEXT = WA.Advice((A, B), (_step((0, 4, 1)), _step((1, 3, 1))), (0, 1), (AT_A, AT_B))
SHAPE = E.Shape(layers=2, indexer_layers=1, latent=4, index_head_dim=4)


def test_the_edge_holds_the_requests_that_must_carry_in_close_order():
    assert [(c.request, c.must) for c in WA.close(W)] == [(0, True), (1, True), (2, False)]
    assert E.requests(W) == (WA.Close(0, AT_A, True), WA.Close(1, AT_B, True))
    assert E.requests(NEXT) == (WA.Close(0, WA.Carried(0, 5, 3, ((0, 5),)), True), WA.Close(1, WA.Carried(0, 4, 2, ((0, 4),)), True))


def test_each_value_holds_each_request_s_rows_after_the_ones_before_it_padded_to_a_power_of_two():
    assert E.layout(W, SHAPE) == {"edge.kv": ((0, 8), (8, 14)), "edge.ik": ((0, 4), (4, 7)), "edge.iks": ((0, 4), (4, 7)),
                                  "edge.last": ((0, 1), (1, 2))}
    assert E.sizes(W, SHAPE) == {"edge.kv": {"rows": 14, "leaves": 16, "words": 4, "word_bits": 16},
                                 "edge.ik": {"rows": 7, "leaves": 8, "words": 2, "word_bits": 16},
                                 "edge.iks": {"rows": 7, "leaves": 8, "words": 1, "word_bits": 32},
                                 "edge.last": {"rows": 2, "leaves": 2, "words": 1, "word_bits": 32}}
    alone = WA.Advice((C,), (_step((0, 0, 3)),), (0,))
    assert E.requests(alone) == () and E.layout(alone, SHAPE) == {v.name: () for v in E.VALUES}
    assert {v["rows"] for v in E.sizes(alone, SHAPE).values()} == {0} and {v["leaves"] for v in E.sizes(alone, SHAPE).values()} == {1}
    assert [E.padded(n) for n in (0, 1, 2, 3, 5, 8, 9)] == [1, 1, 2, 4, 8, 8, 16]
    assert [v.history for v in E.VALUES] == ["prefix", "prefix_ik", "prefix_iks", "last_token"]


def test_a_carried_request_reads_its_slice_of_the_last_period_s_edge():
    assert E.carried(W, NEXT, SHAPE) == (
        E.Slice(0, "a", 0, 4, {"edge.kv": (0, 8), "edge.ik": (0, 4), "edge.iks": (0, 4), "edge.last": (0, 1)}),
        E.Slice(1, "b", 1, 3, {"edge.kv": (8, 14), "edge.ik": (4, 7), "edge.iks": (4, 7), "edge.last": (1, 2)}))
    assert [(s.id, s.edge, s.computed, s.rows) for s in E.carried(W, NEXT)] == [("a", 0, 4, {}), ("b", 1, 3, {})]


def test_a_request_idle_in_a_period_stays_in_its_edge_at_the_state_it_was_carried_in():
    idle = WA.Advice((A, B), (_step((0, 4, 1)),), (0,), (AT_A, AT_B))
    assert E.requests(idle) == (WA.Close(0, WA.Carried(0, 5, 3, ((0, 5),)), True), WA.Close(1, AT_B, True))
    after = WA.Advice((A, B), (_step((1, 3, 1)),), (0,), (WA.Carried(0, 5, 3, ((0, 5),)), AT_B))
    assert [(s.id, s.edge, s.computed, s.rows["edge.kv"]) for s in E.carried(idle, after, SHAPE)] == [("a", 0, 5, (0, 10)),
                                                                                                       ("b", 1, 3, (10, 16))]


def _refused(last: WA.Advice, advice: WA.Advice) -> tuple[str, str]:
    with pytest.raises(WA.NotFollowing) as e:
        E.carried(last, advice, SHAPE)
    return e.value.code, e.value.request


def test_a_carried_request_whose_counters_dont_follow_from_the_close_is_refused_by_name():
    swapped = WA.Advice((B, A), (_step((0, 3, 1)), _step((1, 4, 1))), (0, 1), (AT_B, AT_A))
    assert _refused(W, swapped) == ("dropped", "a")
    assert _refused(W, WA.Advice((A,), (_step((0, 4, 1)),), (0,), (AT_A,))) == ("dropped", "b")
    for counter, b, release in (("C", WA.Carried(1, 3, 1, ((0, 3),)), (0, 1)), ("kept", WA.Carried(0, 3, 2, ((0, 3),)), (0,)),
                                ("made", WA.Carried(0, 3, 1, ((1, 3),)), (0, 1))):
        assert _refused(W, WA.Advice((A, B), NEXT.steps, release, (AT_A, b))) == (counter, "b")
    behind = WA.Advice((A, B), (_step((0, 4, 1)), _step((1, 2, 1)), _step((1, 3, 1))), (0, 1), (AT_A, WA.Carried(0, 2, 1, ((0, 2),))))
    assert _refused(W, behind) == ("computed", "b")


def test_a_shape_is_whole_numbers_and_whole_indexer_words():
    with pytest.raises(ValueError, match="layers is a whole number"):
        E.Shape(0, 1, 4, 4)
    with pytest.raises(ValueError, match="don't fill whole 16-bit words"):
        E.Shape(1, 1, 4, 3)
    with pytest.raises(ValueError, match="whole number"):
        E.padded(-1)
