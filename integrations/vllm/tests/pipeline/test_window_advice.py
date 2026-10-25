"""`window_advice`: a Commit window that carries nothing is I1 v0's, byte for byte; one that carries a request across its open or
close is version 1, round-trips, and opens only where the last window closed."""
from __future__ import annotations

import pytest

from verity.core.protocols.certifier import advice as I1
from verity_vllm.pipeline import window_advice as WA


def _row(request: int, start: int, n: int, prompt_len: int = 3) -> WA.Row:
    return WA.Row(request, "decode" if n == 1 and start >= prompt_len else "prefill", start, n, 0, False, ())


def _steps(*rows: tuple[int, int, int]) -> tuple[WA.StepAdvice, ...]:
    return tuple(WA.StepAdvice((_row(*r),)) for r in rows)


A = WA.RequestAdvice("a", 3, 4, "length")
# window 1 admits a (prompt 3, 4 outputs) and keeps 2 of its outputs; window 2 carries it in and keeps the other 2
W1 = WA.Advice((A,), _steps((0, 0, 3), (0, 3, 1)), (0, 1))
AT_CLOSE = WA.Carried(0, 4, 2, ((0, 4),))
W2 = WA.Advice((A, WA.RequestAdvice("b", 3, 1, "eos")), _steps((0, 4, 1), (1, 0, 3), (0, 5, 1)), (0, 1, 2), (AT_CLOSE,))


def test_a_window_that_carries_nothing_is_i1_v0_byte_for_byte():
    a = WA.Advice((WA.RequestAdvice("a", 3, 2, "length"),), _steps((0, 0, 3), (0, 3, 1)), (0, 1))
    data = WA.encode(a)
    assert a.version == 0 and data == I1.encode(a.i1()) and data[0] == I1.VERSION
    assert WA.decode(data) == a and WA.to_json(a) == I1.to_json(a.i1()) and WA.from_json(WA.to_json(a)) == a


def test_a_request_split_across_two_windows_is_carried_at_the_state_the_first_leaves():
    assert WA.close(W1) == (WA.Close(0, AT_CLOSE, True),) and W1.version == 1
    assert [(o.request, o.index) for o in WA.outputs(W2)] == [(0, 2), (1, 0), (0, 3)] and W2.version == 1
    WA.follows(W1, W2)
    for a in (W1, W2):
        data = WA.encode(a)
        assert data[0] == WA.VERSION and WA.decode(data) == a and WA.encode(WA.decode(data)) == data
        assert WA.from_json(WA.to_json(a)) == a
        assert sum(WA.bits_by_window(a, 2).values()) == WA.total_bits(a)
    with pytest.raises(WA.MalformedAdvice, match="isn't I1 v0 advice"):
        W1.i1()


def test_a_carried_request_that_continues_anywhere_but_its_carried_position_is_refused():
    with pytest.raises(WA.MalformedAdvice, match="request 0 continues at 4, not 5"):
        WA.Advice((A,), _steps((0, 5, 1)), (0,), (AT_CLOSE,))
    with pytest.raises(WA.MalformedAdvice, match="first row starts at its cache hit"):
        WA.Advice((A,), _steps((0, 4, 1)), (0,))


def test_a_window_that_opens_anywhere_but_where_the_last_closed_is_refused():
    with pytest.raises(WA.MalformedAdvice, match="'a' ends the last window with outputs still to keep"):
        WA.follows(W1, WA.Advice((WA.RequestAdvice("b", 3, 1, "eos"),), _steps((0, 0, 3)), (0,)))
    moved = WA.Advice((A,), _steps((0, 3, 1), (0, 4, 1)), (0,), (WA.Carried(0, 3, 2, ((0, 3),)),))
    with pytest.raises(WA.MalformedAdvice, match="not as the last window left it"):
        WA.follows(W1, moved)
    longer = WA.RequestAdvice("a", 3, 5, "length")
    with pytest.raises(WA.MalformedAdvice, match="not as the last window left it"):
        WA.follows(W1, WA.Advice((longer,), _steps((0, 4, 1), (0, 5, 1), (0, 6, 1)), (0, 1, 2), (AT_CLOSE,)))
    stranger = WA.RequestAdvice("c", 3, 4, "length")
    with pytest.raises(WA.MalformedAdvice, match="isn't one the last window left"):
        WA.follows(W2, WA.Advice((stranger,), _steps((0, 4, 1)), (0,), (AT_CLOSE,)))


def test_decode_refuses_a_version_1_string_version_0_can_say(monkeypatch):
    a = WA.Advice((WA.RequestAdvice("a", 3, 2, "length"),), _steps((0, 0, 3), (0, 3, 1)), (0, 1))
    monkeypatch.setattr(WA.Advice, "version", property(lambda self: WA.VERSION))
    data = WA.encode(a)
    monkeypatch.undo()
    assert data[0] == WA.VERSION
    with pytest.raises(WA.MalformedAdvice, match="carries nothing in or out is I1 v0's"):
        WA.decode(data)
