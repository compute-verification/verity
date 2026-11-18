"""The compile slice under the certified tick (`tick_slice.TickSlice_v1`): on the toy window, sync and async, the
honest proof block is accepted, its steps are the replay's under the tick and its T `Replay.clock`; the circuit agrees with the
reference on every case. An arrival's bucket changed in ``cert`` is refused by ``rows``, and accepted with the prover's
own rows in ``srv``, which only the b‖c of ``cert`` and ``srv`` refuse; a T other than the chain's is refused by
``load`` at the window's first state row and by ``steps`` at a later one; a decision in ``adv`` is refused by ``load``."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.compiler.test_network_attester_compile_slice import BOOK
from verity_vllm.compiler import compile_slice as CS
from verity_vllm.pipeline import tick_slice as TL
from verity_vllm.pipeline.advice_code import replay as RP
from verity_vllm.pipeline.advice_code import schedule as SC


@pytest.fixture(scope="module", params=[False, True], ids=["sync", "async"])
def x(request):
    st, sl = TL.demo(request.param)
    S = SimpleNamespace(**st)
    return SimpleNamespace(st=st, S=S, sl=sl, honest=TL.case(S, sl, BOOK), asy=request.param)


def _rows(steps) -> list[list[tuple[int, int, int]]]:
    return [[(i, start, n) for _, i, _, start, n in r] for r in steps if r]


def test_the_honest_window_is_the_replays(x):
    """Accepted (0x3F) at the length terms' charge and refused by ``charge`` alone one bit below it; its steps are the
    replay's of the same requests under the rule with the slice's buckets, and its T after each step with rows is
    `Replay.clock`, carried in the witness's state rows."""
    assert x.honest.verdict == TL.ACCEPT
    low = TL.case(x.S, x.sl, BOOK, charge=x.honest.length - 1)
    assert low.verdict == TL.ACCEPT & ~(1 << 3)
    e = SC.Engine(**{**CS.DEMO_ENGINE, "max_num_seqs": 2, "enable_prefix_caching": False, "async_scheduling": x.asy})
    ios = [SC.RequestIO(prompt_len=P, max_tokens=m, output_len=O, eos=O < m)
           for (_i, P, O, *_c), m in zip(x.sl.arrivals, x.sl.max_tokens)]
    rp = RP.replay(e, ios, tick=TL.rule(x.S), released=list(x.sl.buckets))
    sc = TL.scan(x.S, x.sl)
    assert _rows(sc.steps) == [s.rows for s in rp.steps] and len(sc.steps) == x.sl.steps
    assert [r.clock for r in sc.states[1:x.sl.steps + 1]] == rp.clock
    assert TL.evaluate(x.st, [x.honest, low]) == [x.honest.verdict, low.verdict]


def _bucket_moved(x):
    return TL.arrivals(TL.Slice(x.sl.arrivals, (), x.sl.steps, (), x.sl.max_tokens, (0, 0, 3)))


def test_a_changed_bucket_is_refused_by_the_rows(x):
    """Request 1 released in bucket 0 in place of 1: it is given at the first step, so the scan's rows aren't the
    served ones and ``rows`` refuses (0x2F)."""
    moved = TL.case(x.S, x.sl, BOOK, cert=_bucket_moved(x), served=x.honest.steps)
    assert moved.verdict == TL.ACCEPT & ~(1 << 4) and _rows(moved.steps) != _rows(x.honest.steps)
    assert TL.evaluate(x.st, [moved]) == [moved.verdict]


def test_a_changed_bucket_passes_with_the_provers_own_served_rows(x):
    """The same bucket with the prover's own rows in ``srv``: the circuit accepts. The bucket is ``cert``'s, which its
    b‖c to the certified record holds, and the rows are ``srv``'s, which theirs holds."""
    own = TL.case(x.S, x.sl, BOOK, cert=_bucket_moved(x))
    assert own.verdict == TL.ACCEPT and own.rows["cert"] != x.honest.rows["cert"]
    assert own.rows["srv"] != x.honest.rows["srv"]
    assert TL.evaluate(x.st, [own]) == [own.verdict]


def test_a_t_other_than_the_chains_is_refused(x):
    """T = 1 in place of 0 in the window's first state row: the steps go on from it (the first step's jump to bucket 0's
    close at 10 hides it from the rows), and ``load`` refuses, as it copies T from the empty row. T one past the step's
    at step 2: every later T is one later, and ``steps`` refuses the step that wrote it."""
    first = TL.case(x.S, x.sl, BOOK, clock={0: 1})
    assert first.flags["load"] is False and first.flags["rows"] is True and first.verdict == TL.ACCEPT & ~1
    sc = TL.scan(x.S, x.sl)
    later = TL.case(x.S, x.sl, BOOK, clock={2: sc.states[2].clock + 1}, served=x.honest.steps)
    assert later.flags["steps"] is False and later.flags["load"] is True and later.verdict & 0b11 == 0b01
    assert TL.evaluate(x.st, [first, later]) == [first.verdict, later.verdict]


def test_a_decision_is_refused_by_the_load(x):
    """TickStep_v1 reads no decision: ``adv`` with one, the window row holding it, is refused by ``load``."""
    S, h = x.S, x.honest
    rows = dict(h.rows)
    rows["adv"] = CS._pack(S, "adv", {"dec": [1] + [0] * (S.D - 1), "count": [1], "steps": [x.sl.steps]}, TL.CODE)
    flags = TL.check(S, rows)
    assert flags["load"] is False
    got = TL.evaluate(x.st, [SimpleNamespace(rows=rows)])[0]
    assert got & 1 == 0 and got == CS.verdict(flags)


def test_the_statics_refuse_a_prefix_cache():
    st, _ = TL.demo()
    with pytest.raises(ValueError, match="PC = 0"):
        TL.statics(2, 1, 3, {**CS.DEMO_ENGINE, "max_num_seqs": 2}, 3, Q=3, D=1, P=6, K=3, tau=10, sat=6, free=4,
                   token=1, carry=3)
    assert TL.unit(**st).id.startswith("TickSlice_v1{")


def test_the_statics_refuse_the_catch_up_while_requests_run():
    st, _ = TL.demo()
    with pytest.raises(ValueError, match="the tick's slice has no catch-up while requests run: TCU = 0"):
        TL._sig(SimpleNamespace(**{**st, "TCU": 1}))
