"""verity_vllm.program.predict.steps under the advice code (`pipeline.advice_code.codec`): a registration that selects it has the
compiler read the advice v1 string, decode it against the certified record into the I1 v0 it carries, and bound the string's bits
by ℓ; a corrupted string, one over ℓ, and any other advice are refused by name. A registration that selects none reads I1's.
"""
from __future__ import annotations

import pytest
from verity_vllm.pipeline.advice_code import codec as V1
from verity_vllm.pipeline.advice_code import schedule as SC
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.steps import Registered, compile_steps, plan_steps

from tests.program.predict.test_steps import GREEDY, _prompt, _registered
from verity.core.protocols.certifier import advice as I1

EOS = 2
#: two sequences and an 8-token budget: the second request's prefill is chunked beside the first's, the third waits for a slot
ENGINE = SC.Engine(max_num_seqs=2, max_num_batched_tokens=8, max_model_len=64, num_gpu_blocks=32, block_size=4)
TICK = SC.Tick(bucket_us=1000, sat_us=1000, free_us=1000)
CAP = 1 << 13


def _v1_session(tick: bool = True) -> tuple[V1.Record, V1.Registered, I1.Advice, bytes, dict]:
    """Three greedy requests of 6 prompt tokens and 4 outputs each with no prefix cache. Under the certified tick the replay
    fixes every step and v1 sends nothing: the honest advice is the one the empty string decodes to. Without the tick
    (`tick=False`) the same advice's string sends the replay's decisions. With the record, the code's registration, the
    advice's v1 string under it, and the certified bodies."""
    reqs = tuple(V1.Request(f"q{k}", tuple(_prompt(k, 6)), 4, (5, 6, 7, 8), end, (end + 2, end + 3, end + 4, end + 5))
                 for k, end in enumerate((0, 0, 2)))
    record, ticked = V1.Record(reqs), V1.Registered(ENGINE, (EOS,), T=3, tick=TICK)
    advice = V1.decode_bits([], record, ticked)
    code = ticked if tick else V1.Registered(ENGINE, (EOS,), T=3)
    ingress = {q.id: {**GREEDY, "prompt": list(q.prompt), "model": "m", "user": q.id} for q in reqs}
    return record, code, advice, V1.encode(advice, record, code), ingress


def _coded(code: V1.Registered, cap_bits: int = CAP) -> Registered:
    return Registered.of(_registered().phi, 7, cap_bits=cap_bits, code=code)


def test_a_v1_string_compiles_to_the_compilation_of_the_i1_it_carries():
    """The string and the I1 v0 bytes of one session compile alike, every request's Program and every step's rows: under the
    certified tick the string is empty, and without it, where it sends the decisions, it is a tenth of I1's bits."""
    for tick in (True, False):
        record, code, advice, data, ingress = _v1_session(tick)
        assert [[(r.request, r.start, r.n) for r in st.rows] for st in advice.steps][:2] == [[(0, 0, 2), (1, 0, 6)], [(1, 6, 1), (0, 2, 4)]]
        coded = compile_steps(_coded(code), data, ingress, record=record)
        plain = compile_steps(_registered(cap_bits=CAP), I1.encode(advice), ingress)
        assert coded.summary() == plain.summary() and len(coded.programs) == 1 and coded.batch_rule is None
        assert plan_steps(_coded(code), data, ingress, record=record).steps == plan_steps(_registered(), advice, ingress).steps
        assert data == b"" if tick else 0 < 10 * 8 * len(data) < 8 * len(I1.encode(advice))


def test_the_cap_bounds_the_v1_strings_bits():
    """ℓ is held to the string's bits before any of it is read: at its length the string compiles, a bit below it is refused, and
    at that ℓ a registration that selects no code refuses the session's I1 bytes."""
    record, code, advice, data, ingress = _v1_session(tick=False)
    n = 8 * len(data)
    assert compile_steps(_coded(code, n), data, ingress, record=record).cap_bits == n
    with pytest.raises(Unexpressible, match=f"the advice is {n} bits, over the registered cap ℓ = {n - 1} bits"):
        compile_steps(_coded(code, n - 1), data, ingress, record=record)
    with pytest.raises(Unexpressible, match=f"the advice is {8 * len(I1.encode(advice))} bits, over the registered cap ℓ = {n} bits"):
        compile_steps(_registered(cap_bits=n), I1.encode(advice), ingress)


def test_a_corrupted_v1_string_is_refused_by_name():
    """Under the certified tick the record leaves one string, the empty one: every string of one byte, and two zero bytes, v1
    refuses, the same way twice, and the compiler names it ill-formed v1 advice."""
    record, code, _, data, ingress = _v1_session()
    assert data == b""
    for x in [bytes([b]) for b in range(256)] + [b"\0\0"]:
        why = []
        for _ in range(2):
            with pytest.raises(Unexpressible, match="^ill-formed v1 advice: ") as e:
                plan_steps(_coded(code), x, ingress, record=record)
            why.append(str(e.value))
        assert why[0] == why[1]


def test_a_registration_that_selects_the_code_reads_nothing_else():
    """Under the code: I1's bytes are an ill-formed v1 string, an Advice isn't a string, and the record is required. Without it,
    a record is refused, and the registration's code is a codec registration."""
    record, code, advice, data, ingress = _v1_session()
    with pytest.raises(Unexpressible, match="^ill-formed v1 advice: "):
        compile_steps(_coded(code), I1.encode(advice), ingress, record=record)
    with pytest.raises(Unexpressible, match="its advice is a v1 string's bytes, not a Advice"):
        compile_steps(_coded(code), advice, ingress, record=record)
    with pytest.raises(Unexpressible, match="decodes against the session's certified record, and none was given"):
        compile_steps(_coded(code), data, ingress)
    with pytest.raises(Unexpressible, match="^ill-formed v1 advice: v1 decodes against a certified Record, not a dict"):
        compile_steps(_coded(code), data, ingress, record={"requests": []})
    with pytest.raises(Unexpressible, match="the registration selects none: its advice is I1's"):
        compile_steps(_registered(), I1.encode(advice), ingress, record=record)
    with pytest.raises(ValueError, match="and a dict has no decode"):
        Registered.of(_registered().phi, 7, code={"T": 3})
