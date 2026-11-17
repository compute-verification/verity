from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from tests.pipeline import test_tick_step as TT
from verity_vllm.compiler import replay_step as RS
from verity_vllm.compiler import tick_step as TK
from verity_vllm.pipeline.advice_code import schedule as SC
from verity_vllm.pipeline.advice_code import tick_step as TS

TODAY = TS.Rule(TT.TICK.bucket_us, TT.TICK.sat_us, TT.TICK.free_us)
CHOSEN = TS.Rule(10**8, 6_303_000, 6_303_000, 18_126)
RULES = {f"{name}-{'quiet' if quiet else 'unqueued'}": dataclasses.replace(rule, quiet=quiet)
         for name, rule in (("today", TODAY), ("chosen", CHOSEN)) for quiet in (True, False)}

Step = tuple[TS.State, TS.Inputs, TS.Out]


def binding(e: SC.Engine, n: int, R: int, W: int, A: int, rule: TS.Rule) -> dict:
    return TK.statics(R, W, A, dataclasses.asdict(e), max(n - 1, 1).bit_length(), tau=rule.bucket, sat=rule.sat,
                      free=rule.free, token=rule.token, quiet=rule.quiet)


def scan(S: SimpleNamespace, rule: TS.Rule, ios: list[SC.RequestIO], released: list[int]) -> list[Step]:
    st, cursor, out = TS.empty(S), 0, []
    while cursor < len(ios) or st.step.running or st.step.waiting:
        assert len(out) < 1000
        inp = TT.inputs(S, st.step, cursor, ios, released)
        got = TS.reference(S, rule, st, inp)
        assert got.ok and got.state is not None
        out.append((st, inp, got))
        st, cursor = got.state, cursor + got.taken
    return out


def held(st_: dict, steps: list[Step]) -> None:
    cases = [(st.step, st.clock, RS.Inputs(inp.arrivals, (), inp.hit, inp.evict, inp.ret, inp.row_pfx), inp.released)
             for st, inp, _ in steps]
    for (st, _, want), (got, clock) in zip(steps, TK.evaluate(st_, cases), strict=True):
        assert want.state is not None and got.ok and got.state is not None
        assert (got.rows, got.state, got.taken, got.looked, clock) == (
            want.rows, want.state.step, want.taken, want.looked, want.state.clock), (st, want)


def seen(steps: list[Step], rule: TS.Rule) -> set[str]:
    out = set()
    for st, inp, o in steps:
        assert o.state is not None
        jump = (not st.step.waiting and not (rule.quiet and st.step.running) and bool(inp.arrivals)
                and st.clock < rule.close(inp.released[0]))
        if jump:
            out.add("jumps while running" if st.step.running else "jumps")
        elif o.rows and st.clock // rule.bucket != o.state.clock // rule.bucket:
            out.add("carries")
        moved = o.state.clock - st.clock - rule.token * sum(n for *_, n in o.rows)
        if o.rows and not jump and moved == rule.sat != rule.free:
            out.add("saturates")
    return out


SMALL = [(False, 2, [(15, 10, True), (10, 8, True)], [0, 0]), (True, 2, [(15, 4, True), (10, 2, True)], [0, 0]),
         (True, 2, [(10, 7, False), (5, 9, True)], [0, 0]),
         (False, 15, [(4, 3, True), (4, 6, True), (4, 2, True)], [0, 1, 2]),
         (True, 15, [(4, 2, True), (4, 6, True), (4, 2, True)], [0, 1, 2])]


@pytest.mark.parametrize("name", RULES)
@TT.BOTH
def test_the_circuit_is_the_reference_on_the_plan(name: str, asy: bool) -> None:
    rule = RULES[name]
    e = dataclasses.replace(TT.engine(asy), enable_prefix_caching=False)
    ios, released = TT.plan()
    st_ = binding(e, TT.N, 4, 4, 5, rule)
    steps = scan(SimpleNamespace(**st_), rule, ios, released)
    held(st_, steps)
    want = {"carries", "jumps"} | ({"saturates"} if rule.sat != rule.free else set()) | (
        set() if rule.quiet else {"jumps while running"})
    assert want <= seen(steps, rule)


@pytest.mark.parametrize("name", RULES)
@pytest.mark.parametrize("asy, blocks, requests, released", SMALL)
def test_the_circuit_is_the_reference_on_the_small_engines(name: str, asy: bool, blocks: int,
                                                         requests: list[tuple[int, int, bool]], released: list[int]) -> None:
    rule = RULES[name]
    e = SC.Engine(max_num_seqs=2, max_num_batched_tokens=64, max_model_len=64, num_gpu_blocks=blocks + 1, block_size=16,
                  async_scheduling=asy)
    ios = [SC.RequestIO(prompt_len=p, max_tokens=o + 4 if eos else o, output_len=o, eos=eos) for p, o, eos in requests]
    st_ = binding(e, len(ios), 2, 2, 3, rule)
    held(st_, scan(SimpleNamespace(**st_), rule, ios, released))


@TT.BOTH
def test_the_circuit_is_the_reference_on_the_replays_prefix_hits(asy: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    e = TT.engine(asy)
    ios, released = TT.plan()
    _, units = TT.record(monkeypatch, e, ios, released)
    st_ = binding(e, TT.N, 4, 4, 5, TODAY)
    S = SimpleNamespace(**st_)
    st, cursor, steps = TS.empty(S), 0, []
    for u in (u for u in units if not u.idle):
        inp = TT.inputs(S, st.step, cursor, ios, released, u)
        out = TS.reference(S, TT.TICK, st, inp)
        assert out.ok and out.state is not None
        steps.append((st, inp, out))
        st, cursor = out.state, cursor + out.taken
    assert cursor == TT.N and any(any(inp.hit) for _, inp, _ in steps)
    held(st_, steps)


def test_a_bucket_count_past_32_bits_is_refused() -> None:
    e = SC.Engine(max_num_seqs=2, max_num_batched_tokens=64, max_model_len=64, num_gpu_blocks=16, block_size=16)
    st_ = binding(e, 2, 2, 2, 3, CHOSEN)
    top = (1 << TK.QW) - 1
    one, empty = RS.Req(0, 4, 3), RS.empty(SimpleNamespace(**st_))
    running = RS.State((dataclasses.replace(one, out=1, comp=4),), (), empty.free - 1)
    cases = [(empty, 0, RS.Inputs((one,)), (top,)), (empty, 0, RS.Inputs((one,)), (top - 1,)),
             (running, top * CHOSEN.bucket + CHOSEN.bucket - 1, RS.Inputs(()), ()),
             (running, (top - 1) * CHOSEN.bucket + CHOSEN.bucket - 1, RS.Inputs(()), ())]
    assert [o.ok for o, _ in TK.evaluate(st_, cases)] == [False, True, False, True]


def test_the_binding_refuses_an_advance_that_reaches_a_bucket() -> None:
    e = SC.Engine(max_num_seqs=8, max_num_batched_tokens=2048, max_model_len=512, num_gpu_blocks=64, block_size=16)
    binding(e, 2, 8, 1, 9, CHOSEN)
    with pytest.raises(ValueError, match=r"max\(CSAT, CFREE\) \+ C1·T = 154791192 reaches the bucket TAU = 100000000"):
        binding(dataclasses.replace(e, max_num_batched_tokens=8192), 2, 8, 1, 9, CHOSEN)
    with pytest.raises(ValueError, match=r"A ≥ min\(M, R\) \+ 1 = 9"):
        binding(e, 2, 8, 1, 8, CHOSEN)
