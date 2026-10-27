"""`window_chain.join`: Commit windows that carry a request across their close join into one I1 v0 advice, the carried
request the earlier window's, with its record and every step once, so its bits charge the carry once; a carried state
the earlier window didn't leave, a carry in from a window that isn't among them, a window left owing a carry and an
aborted request are refused."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from verity.core.protocols.certifier import advice as I1
from verity_vllm.pipeline import window_advice as WA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import window_chain as WC  # noqa: E402


def _row(request: int, start: int, n: int, prompt_len: int = 3) -> WA.Row:
    return WA.Row(request, "decode" if n == 1 and start >= prompt_len else "prefill", start, n, 0, False, ())


def _steps(*rows: tuple[int, int, int]) -> tuple[WA.StepAdvice, ...]:
    return tuple(WA.StepAdvice((_row(*r),)) for r in rows)


A, B = WA.RequestAdvice("a", 3, 4, "length"), WA.RequestAdvice("b", 3, 1, "eos")
# window 5 admits a (prompt 3, 4 outputs) and keeps 2 of its outputs; window 6 carries it in, admits b, keeps the rest
W5 = WA.Advice((A,), _steps((0, 0, 3), (0, 3, 1)), (0, 1))
AT_CLOSE = WA.Carried(0, 4, 2, ((0, 4),))
W6 = WA.Advice((A, B), _steps((0, 4, 1), (1, 0, 3), (0, 5, 1)), (0, 1, 2), (AT_CLOSE,))
V0 = WA.Advice((WA.RequestAdvice("c", 3, 1, "eos"),), _steps((0, 0, 3)), (0,))


def _at(offsets):
    return lambda k, rid, index, bucket: bucket + offsets[k]


def test_a_carried_request_is_the_earlier_windows_once_and_the_join_is_its_engines_steps():
    got = WC.join([(5, W5), (6, W6), (7, V0)], _at((0, 10, 20)))
    assert got == I1.Advice((A.i1(), B.i1(), V0.requests[0].i1()),
                            (*W5.steps, *_steps((0, 4, 1), (1, 0, 3), (0, 5, 1)), I1.StepAdvice((_row(2, 0, 3),))),
                            (0, 1, 10, 11, 12, 20))
    assert [(o.request, o.index) for o in I1.outputs(got)] == [(0, 0), (0, 1), (0, 2), (1, 0), (0, 3), (2, 0)]
    committed = sum(WA.total_bits(a) for a in (W5, W6, V0))
    one = I1.total_bits(I1.Advice(got.requests[:2], got.steps[:5], got.release[:5])) + I1.total_bits(V0.i1())
    assert I1.total_bits(got) < one < committed
    assert WC.join([(7, V0)], _at((0,))) == V0.i1()


def test_a_carried_state_the_earlier_window_didnt_leave_is_refused():
    lied = WA.Advice((A, B), W6.steps, W6.release, (WA.Carried(1, 4, 2, ((0, 4),)),))
    with pytest.raises(I1.MalformedAdvice, match="window 6 doesn't open where window 5 closed: .*not as the last"):
        WC.join([(5, W5), (6, lied)], _at((0, 0)))
    longer = WA.RequestAdvice("a", 3, 5, "length")
    more = WA.Advice((longer,), _steps((0, 4, 1), (0, 5, 1), (0, 6, 1)), (0, 1, 2), (AT_CLOSE,))
    with pytest.raises(I1.MalformedAdvice, match="not as the last window left it"):
        WC.join([(5, W5), (6, more)], _at((0, 0)))


def test_a_carry_from_a_window_not_among_them_and_a_window_left_owing_one_are_refused():
    with pytest.raises(I1.MalformedAdvice, match="window 6 carries 1 requests in from window 5, which isn't among"):
        WC.join([(6, W6)], _at((0,)))
    with pytest.raises(I1.MalformedAdvice, match="5 ends request 'a' with outputs still to keep, and window 6 isn't"):
        WC.join([(5, W5), (7, V0)], _at((0, 0)))
    with pytest.raises(I1.MalformedAdvice, match="window 5 ends request 'a' with outputs still to keep"):
        WC.join([(5, W5)], _at((0,)))
    with pytest.raises(I1.MalformedAdvice, match="'a' ends the last window with outputs still to keep"):
        WC.join([(5, W5), (6, V0)], _at((0, 0)))


def test_an_aborted_request_is_refused_since_i1_v0_cant_say_it():
    aborted = WA.Advice((WA.RequestAdvice("a", 3, 2, "abort"),), _steps((0, 0, 3), (0, 3, 1), (0, 4, 1)), (0, 1))
    with pytest.raises(I1.MalformedAdvice, match="window 3 aborts a request"):
        WC.join([(3, aborted)], _at((0,)))
