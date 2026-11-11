"""`edge`: a window's edge holds the requests it must leave to the next that have computed a row, in close order, with a field in
`edge.kv` each, in `edge.ik` and `edge.iks` where its rows read past INDEX_TOPK and in `edge.last` once past its prompt; the next
window's carried requests read their fields only when their counters follow from the close, and are refused by name when they
don't."""
from __future__ import annotations

import pytest

from verity_vllm.pipeline import edge as E
from verity_vllm.pipeline import window_advice as WA


def _step(*rows: tuple[int, int, int]) -> WA.StepAdvice:
    return WA.StepAdvice(tuple(WA.Row(r, "decode" if n == 1 and s >= 3 else "prefill", s, n, 0, False, ()) for r, s, n in rows))


A, B, C = WA.RequestAdvice("a", 3, 4, "length"), WA.RequestAdvice("b", 3, 3, "length"), WA.RequestAdvice("c", 3, 1, "eos")
D = WA.RequestAdvice("d", 6, 2, "length")
# window w admits a, b and d (prompts of 3, 3 and 6) and then c; a keeps 2 outputs and b 1, d is 3 rows into its prompt, and c
# keeps its 1 in w's last step: a, b and d must carry, c may
W = WA.Advice((A, B, D, C), (_step((0, 0, 3), (1, 0, 3), (2, 0, 3)), _step((0, 3, 1), (3, 0, 3))), (0, 0, 1, 1))
AT_A, AT_B, AT_D = WA.Carried(0, 4, 2, ((0, 4),)), WA.Carried(0, 3, 1, ((0, 3),)), WA.Carried(0, 3, 0, ((0, 3),))
# window w + 1 carries all three on: a and b keep an output each, and d finishes its prompt and keeps its first
NEXT = WA.Advice((A, B, D), (_step((0, 4, 1), (2, 3, 3)), _step((1, 3, 1))), (0, 0, 1), (AT_A, AT_B, AT_D))


def test_the_edge_holds_the_requests_that_must_carry_in_close_order():
    assert [(c.request, c.must) for c in WA.close(W)] == [(0, True), (1, True), (2, True), (3, False)]
    assert E.members(W) == (WA.Close(0, AT_A, True), WA.Close(1, AT_B, True), WA.Close(2, AT_D, True))
    assert [c.request for c in E.members(NEXT)] == [0, 1, 2]
    assert E.members(WA.Advice((WA.RequestAdvice("f", 3, 2, "length"),), (), (), (WA.Carried(0, 0, 0, ()),))) == ()


def test_each_port_has_a_field_per_member_holding_it_and_edge_last_only_past_the_prompt():
    assert [(v.name, v.history) for v in E.VALUES] == [("edge.kv", "prefix"), ("edge.ik", "prefix_ik"), ("edge.iks", "prefix_iks"),
                                                       ("edge.last", "last_token")]
    assert E.fields(W) == {"edge.kv": (0, 1, 2), "edge.last": (0, 1)}
    assert E.fields(NEXT) == {"edge.kv": (0, 1, 2), "edge.last": (0, 1, 2)}
    assert E.fields(WA.Advice((C,), (_step((0, 0, 3)),), (0,))) == {"edge.kv": (), "edge.last": ()}


def test_a_carried_request_reads_its_fields_of_the_last_window_s_edge():
    assert E.carried(W, NEXT) == (E.Slice(0, "a", 4, {"edge.kv": 0, "edge.last": 0}),
                                  E.Slice(1, "b", 3, {"edge.kv": 1, "edge.last": 1}), E.Slice(2, "d", 3, {"edge.kv": 2}))


def test_the_indexer_s_keys_and_scales_are_carried_for_a_request_whose_rows_read_past_index_topk():
    # at INDEX_TOPK 5, a's last row reads 6 keys and d's 7, b's 5: a and d hold `edge.ik` and `edge.iks`, in every window
    held = {"edge.kv": (0, 1, 2), "edge.ik": (0, 2), "edge.iks": (0, 2)}
    assert E.fields(W, 5) == {**held, "edge.last": (0, 1)} and E.fields(NEXT, 5) == {**held, "edge.last": (0, 1, 2)}
    assert E.fields(W, 7) == {"edge.kv": (0, 1, 2), "edge.ik": (), "edge.iks": (), "edge.last": (0, 1)}
    assert [s.fields for s in E.carried(W, NEXT, 5)] == [{"edge.kv": 0, "edge.ik": 0, "edge.iks": 0, "edge.last": 0},
                                                         {"edge.kv": 1, "edge.last": 1}, {"edge.kv": 2, "edge.ik": 1, "edge.iks": 1}]


def test_a_request_idle_in_a_window_stays_in_its_edge_at_the_state_it_was_carried_in():
    idle = WA.Advice((A, B, D), (_step((0, 4, 1)), _step((1, 3, 1))), (0, 1), (AT_A, AT_B, AT_D))
    assert E.members(idle)[2] == WA.Close(2, AT_D, True) and E.fields(idle) == {"edge.kv": (0, 1, 2), "edge.last": (0, 1)}
    after = WA.Advice((A, B, D), (_step((2, 3, 3)),), (0,), tuple(c.state for c in E.members(idle)))
    assert [(s.id, s.computed, s.fields) for s in E.carried(idle, after)] == [
        ("a", 5, {"edge.kv": 0, "edge.last": 0}), ("b", 4, {"edge.kv": 1, "edge.last": 1}), ("d", 3, {"edge.kv": 2})]


def _refused(last: WA.Advice, advice: WA.Advice) -> tuple[str, str]:
    with pytest.raises(WA.NotFollowing) as e:
        E.carried(last, advice)
    return e.value.code, e.value.request


def test_a_carried_request_whose_counters_dont_follow_from_the_close_is_refused_by_name():
    swapped = WA.Advice((B, A, D), (_step((0, 3, 1), (2, 3, 3)), _step((1, 4, 1))), (0, 0, 1), (AT_B, AT_A, AT_D))
    assert _refused(W, swapped) == ("dropped", "a")
    assert _refused(W, WA.Advice((A, B), (_step((0, 4, 1)), _step((1, 3, 1))), (0, 1), (AT_A, AT_B))) == ("dropped", "d")
    for counter, b, release in (("C", WA.Carried(1, 3, 1, ((0, 3),)), (0, 0, 1)), ("kept", WA.Carried(0, 3, 2, ((0, 3),)), (0, 0)),
                                ("made", WA.Carried(0, 3, 1, ((1, 3),)), (0, 0, 1))):
        assert _refused(W, WA.Advice((A, B, D), NEXT.steps, release, (AT_A, b, AT_D))) == (counter, "b")
    behind = WA.Advice((A, B, D), (_step((0, 4, 1), (2, 3, 3)), _step((1, 2, 1)), _step((1, 3, 1))), (0, 0, 1),
                       (AT_A, WA.Carried(0, 2, 1, ((0, 2),)), AT_D))
    assert _refused(W, behind) == ("computed", "b")
