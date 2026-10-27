"""`window_advice`: a Commit window that carries nothing is I1 v0's, byte for byte; one that carries a request across its open or
close is version 1, round-trips, and opens only where the last window closed; one that aborts a request is version 2, charged a
bit a request more, and every window has the one encoding."""
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
    with pytest.raises(WA.MalformedAdvice, match="carries nothing in or out and aborts nothing is I1 v0's"):
        WA.decode(data)


# a, aborted after 2 outputs: the endpoint frees it after the step in flight, whose row for it is the lag row past its last output
ROWS = _steps((0, 0, 3), (0, 3, 1), (0, 4, 1))
ABORTED = WA.Advice((WA.RequestAdvice("a", 3, 2, "abort"),), ROWS, (0, 1))
STOPPED = WA.Advice((WA.RequestAdvice("a", 3, 2, "eos"),), ROWS, (0, 1))


def test_an_aborted_request_is_version_2_round_trips_and_is_charged_its_finish_bits():
    assert STOPPED.version == 0 and ABORTED.version == WA.ABORTS
    data = WA.encode(ABORTED)
    assert data[0] == WA.ABORTS and WA.decode(data) == ABORTED and WA.encode(WA.decode(data)) == data
    assert WA.from_json(WA.to_json(ABORTED)) == ABORTED and WA.to_json(ABORTED)["requests"][0]["finish"] == "abort"
    # version 1's header (n_carried, a 6-bit width and no bits) and one finish bit more a request than I1 v0's
    assert WA.total_bits(ABORTED) == WA.total_bits(STOPPED) + 6 + 1
    assert sum(WA.bits_by_window(ABORTED, 1).values()) == WA.total_bits(ABORTED)
    with pytest.raises(WA.MalformedAdvice, match="aborts one, isn't I1 v0 advice"):
        ABORTED.i1()
    with pytest.raises(WA.MalformedAdvice, match="finish must be one of"):
        I1.RequestAdvice("a", 3, 2, "abort")


def test_an_aborted_request_is_freed_at_its_last_row_and_a_row_past_it_is_refused():
    assert WA.close(ABORTED) == ()
    with pytest.raises(WA.MalformedAdvice, match="past request 0's 5 known tokens"):
        WA.Advice((WA.RequestAdvice("a", 3, 2, "abort"),), _steps((0, 0, 3), (0, 3, 1), (0, 4, 1), (0, 5, 1)), (0, 1))


def test_a_carried_request_aborted_in_the_next_window_follows_its_record():
    aborted = WA.RequestAdvice("a", 3, 3, "abort")
    w1 = WA.Advice((aborted,), _steps((0, 0, 3), (0, 3, 1)), (0, 1))
    w2 = WA.Advice((aborted,), _steps((0, 4, 1), (0, 5, 1)), (0,), (AT_CLOSE,))
    assert w1.version == w2.version == WA.ABORTS and WA.close(w1) == (WA.Close(0, AT_CLOSE, True),) and WA.close(w2) == ()
    WA.follows(w1, w2)
    assert WA.decode(WA.encode(w2)) == w2
    with pytest.raises(WA.MalformedAdvice, match="not as the last window left it"):
        WA.follows(w1, WA.Advice((WA.RequestAdvice("a", 3, 3, "eos"),), w2.steps, w2.release, w2.carried))


def test_a_window_built_of_i1_records_is_the_window_of_its_own():
    assert WA.Advice((I1.RequestAdvice("a", 3, 2, "eos"),), ROWS, (0, 1)) == STOPPED
    assert type(WA.decode(WA.encode(STOPPED)).requests[0]) is WA.RequestAdvice


def test_decode_refuses_a_version_2_string_that_aborts_nothing_and_a_finish_of_3(monkeypatch):
    monkeypatch.setattr(WA.Advice, "version", property(lambda self: WA.ABORTS))
    alone, carried = WA.encode(STOPPED), WA.encode(W2)
    monkeypatch.undo()
    assert alone[0] == carried[0] == WA.ABORTS
    with pytest.raises(WA.MalformedAdvice, match="aborts nothing is I1 v0's"):
        WA.decode(alone)
    with pytest.raises(WA.MalformedAdvice, match="aborts no request is version 1, not 2"):
        WA.decode(carried)
    monkeypatch.setattr(WA, "FINISHES", ("eos", "length", "", "abort"))
    three = WA.encode(ABORTED)
    monkeypatch.undo()
    with pytest.raises(WA.MalformedAdvice, match="request 0's finish 3 names none of"):
        WA.decode(three)
