"""verity_vllm.program.predict.steps under the advice code (`pipeline.advice_code.codec`): a registration that selects it has the
compiler read the advice v1 string, decode it against the certified record into the I1 v0 it carries, and bound the string's bits
by ℓ; a corrupted string, one over ℓ, and any other advice are refused by name. A registration that selects none reads I1's.
While the registration sends lengths (`codec.Registered.send_lengths`, `advice_code.lengths`), each request's output_len and
finish are sent and charged in its body's window, and held to the certified ids and the stop rule.
"""
from __future__ import annotations

import math
from dataclasses import replace

import pytest
from verity_vllm.pipeline.advice_code import codec as V1
from verity_vllm.pipeline.advice_code import lengths as LN
from verity_vllm.pipeline.advice_code import schedule as SC
from verity_vllm.pipeline.advice_code import window_code as CV
from verity_vllm.program.predict.build import Unexpressible
from verity_vllm.program.predict.steps import Registered, compile_steps, plan_steps

from tests.program.predict.test_steps import GREEDY, _prompt, _registered
from verity.core.protocols.certifier import advice as I1

EOS = 2
#: two sequences and an 8-token budget: the second request's prefill is chunked beside the first's, the third waits for a slot
ENGINE = SC.Engine(max_num_seqs=2, max_num_batched_tokens=8, max_model_len=64, num_gpu_blocks=32, block_size=4)
TICK = SC.Tick(bucket_us=1000, sat_us=1000, free_us=1000)
CAP = 1 << 13


def _v1_session() -> tuple[V1.Record, V1.Registered, I1.Advice, bytes, dict]:
    """Three greedy requests of 6 prompt tokens and 4 outputs each under the certified tick with no prefix cache, so the replay
    fixes every step and v1 sends nothing but its windows' frames and its length terms: the honest advice is the one W empty
    windows decode to with no length terms sent, and its v1 string is that advice's encoding under the registration, which
    sends them. With the record, the code's registration, and the certified bodies."""
    reqs = tuple(V1.Request(f"q{k}", tuple(_prompt(k, 6)), 4, (5, 6, 7, 8), end, (end + 2, end + 3, end + 4, end + 5))
                 for k, end in enumerate((0, 0, 2)))
    record, code = V1.Record(reqs), V1.Registered(ENGINE, (EOS,), T=3, tick=TICK)
    empty = CV.encode([CV.Window()] * V1.windows(record, code.T), CV.Book.uniform(V1.TICK_CAPS)).bitlist()
    advice = V1.decode_bits(empty, record, replace(code, send_lengths=False))
    ingress = {q.id: {**GREEDY, "prompt": list(q.prompt), "model": "m", "user": q.id} for q in reqs}
    return record, code, advice, V1.encode(advice, record, code), ingress


def _coded(code: V1.Registered, cap_bits: int = CAP) -> Registered:
    return Registered.of(_registered().phi, 7, cap_bits=cap_bits, code=code)


def test_a_v1_string_compiles_to_the_compilation_of_the_i1_it_carries():
    """The string and the I1 v0 bytes of one session compile alike, every request's Program and every step's rows, while the
    string, its length terms included, is under a seventh of I1's bits."""
    record, code, advice, data, ingress = _v1_session()
    assert [[(r.request, r.start, r.n) for r in st.rows] for st in advice.steps][:2] == [[(0, 0, 2), (1, 0, 6)], [(1, 6, 1), (0, 2, 4)]]
    coded = compile_steps(_coded(code), data, ingress, record=record)
    plain = compile_steps(_registered(cap_bits=CAP), I1.encode(advice), ingress)
    assert coded.summary() == plain.summary() and len(coded.programs) == 1 and coded.batch_rule is None
    assert plan_steps(_coded(code), data, ingress, record=record).steps == plan_steps(_registered(), advice, ingress).steps
    assert 7 * 8 * len(data) < 8 * len(I1.encode(advice))


def test_the_cap_bounds_the_v1_strings_bits():
    """ℓ is held to the string's bits before any of it is read: at its length the string compiles, a bit below it is refused, and
    at that ℓ a registration that selects no code refuses the session's I1 bytes."""
    record, code, advice, data, ingress = _v1_session()
    n = 8 * len(data)
    assert compile_steps(_coded(code, n), data, ingress, record=record).cap_bits == n
    with pytest.raises(Unexpressible, match=f"the advice is {n} bits, over the registered cap ℓ = {n - 1} bits"):
        compile_steps(_coded(code, n - 1), data, ingress, record=record)
    with pytest.raises(Unexpressible, match=f"the advice is {8 * len(I1.encode(advice))} bits, over the registered cap ℓ = {n} bits"):
        compile_steps(_registered(cap_bits=n), I1.encode(advice), ingress)


def test_a_corrupted_v1_string_is_refused_by_name():
    """Every prefix, every one-bit change and an extension of the string: v1 refuses each (the tick and the record leave one
    string), the same way twice, and the compiler names it ill-formed v1 advice."""
    record, code, _, data, ingress = _v1_session()
    flips = [bytes(b ^ (1 << (7 - j % 8)) if i == j // 8 else b for i, b in enumerate(data)) for j in range(8 * len(data))]
    for x in [data[:k] for k in range(len(data))] + flips + [data + b"\0"]:
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


def _lengths_session() -> tuple[V1.Record, V1.Registered, I1.Advice]:
    """Three requests under the certified tick at T = 2, whose bodies land in windows 0, 0 and 1: one cut at its
    max_tokens of 4, one ended by EOS under a max_tokens of 9, and one cut at 7. With the registration, which sends
    lengths, and the honest advice."""
    outs = {"q0": (4, (5, 6, 7, 8)), "q1": (9, (5, 6, EOS)), "q2": (7, (5, 6, 7, 8, 9, 10, 11))}
    reqs = tuple(V1.Request(i, tuple(_prompt(k, 6)), m, o, end, tuple(range(end + 2, end + 2 + len(o))))
                 for k, ((i, (m, o)), end) in enumerate(zip(outs.items(), (0, 0, 2))))
    record, code = V1.Record(reqs), V1.Registered(ENGINE, (EOS,), T=2, tick=TICK)
    empty = CV.encode([CV.Window()] * V1.windows(record, code.T), CV.Book.uniform(V1.TICK_CAPS)).bitlist()
    return record, code, V1.decode_bits(empty, record, replace(code, send_lengths=False))


def test_a_length_term_is_charged_its_formula_in_its_bodys_window():
    """Each request adds ⌈log2(max_tokens + 1)⌉ + 1 bits to the window its body lands in, and nothing to any other: the
    switch's charge less the old one's, window by window."""
    record, code, advice = _lengths_session()
    on = V1.bits_by_window(V1.encode(advice, record, code), record, code)
    off_code = replace(code, send_lengths=False)
    off = V1.bits_by_window(V1.encode(advice, record, off_code), record, off_code)
    want = {w: 0 for w in off}
    for q in record.requests:
        want[q.end // code.T] += math.ceil(math.log2(q.max_tokens + 1)) + 1
    assert {w: on[w] - off[w] for w in off} == want == {0: 3 + 1 + 4 + 1, 1: 3 + 1, 2: 0, 3: 0, 4: 0, 5: 0}
    assert V1.length_widths(record, code)[:2] == [[3, 1, 4, 1], [3, 1]]


def test_a_wrong_sent_length_or_finish_is_refused_by_name():
    """The tail holds the record's values: a string sending another output_len, an output_len past max_tokens, or another
    finish decodes to no advice, and the compiler names the request and the field."""
    record, code, advice = _lengths_session()
    wins = V1.residual(advice, record, code)
    ks = V1.lengths(record, code)
    for window, slot, value, why in ((0, 2, 2, "request q1: the string sends output_len 2, and its certified output ids number 3"),
                                     (1, 0, 0, "request q2: the string sends output_len 0, and its certified output ids number 7"),
                                     (0, 1, 0, "request q0: the string sends finish 'eos', and the stop rule's is 'length'"),
                                     (0, 3, 1, "request q1: the string sends finish 'length', and the stop rule's is 'eos'")):
        tails = LN.tails(record.requests, ks, code.config(code.engine))
        tails[window][slot] = (value, tails[window][slot][1])
        bits = V1.string_of(wins, code, tails)[0]
        with pytest.raises(V1.AdviceCodeError, match=f"^{why}$"):
            V1.decode_bits(bits, record, code)
        with pytest.raises(Unexpressible, match=f"^ill-formed v1 advice: {why}$"):
            plan_steps(_coded(code), CV.pack(bits)[0], _ingress(record), record=record)


def test_with_the_switch_off_the_string_is_the_old_one():
    """Off, the string is byte for byte what the code wrote before it sent lengths (pinned from that code), its charge
    is the window code's alone, and it decodes to the advice; each switch's string is refused under the other."""
    record, code, advice = _lengths_session()
    off = replace(code, send_lengths=False)
    data = V1.encode(advice, record, off)
    assert data == bytes.fromhex("00080040020010008004") and V1.lengths(record, off) == [[]] * 6
    old = CV.encode(V1.residual(advice, record, off), V1.book_of(off))
    assert (data, V1.bits_by_window(data, record, off)) == (CV.pack(old.bitlist())[0], dict(enumerate(old.lens)))
    assert V1.decode(data, record, off) == advice
    with pytest.raises(V1.AdviceCodeError, match="^no 6 windows end the string"):
        V1.decode(data, record, code)
    with pytest.raises(V1.AdviceCodeError):
        V1.decode(V1.encode(advice, record, code), record, off)


def test_a_small_record_round_trips_with_its_lengths_sent():
    """On, encode and decode invert each other, the charge sums to the string's bits less padding, and the string
    compiles as the I1 it carries does."""
    record, code, advice = _lengths_session()
    data = V1.encode(advice, record, code)
    assert data == bytes.fromhex("000c98003e00100080040020")
    got = V1.decode_full(data, record, code)
    assert got.advice == advice and got.streams == V1.streams(advice, record)
    assert 0 <= 8 * len(data) - sum(V1.bits_by_window(data, record, code).values()) < 8
    ingress = _ingress(record)
    assert (compile_steps(_coded(code), data, ingress, record=record).summary()
            == compile_steps(_registered(cap_bits=CAP), I1.encode(advice), ingress).summary())


def test_a_length_is_never_charged_after_its_admission_window():
    """A replay that admits a request at a step whose window precedes its body's is refused: its length would be
    charged after the first window whose rows it changes."""
    reqs = [V1.Request("q0", (3,), 1, (5,), 0, (1,)), V1.Request("q1", (3,), 1, (5,), 4, (6,))]
    LN.check_admitted(reqs, [0, 1], [0, 2], lambda s: 2 if s >= 2 else 0, 2, V1.AdviceCodeError)
    with pytest.raises(V1.AdviceCodeError, match="^request q1: admitted at step 1, whose window 0 precedes window 2, where its"):
        LN.check_admitted(reqs, [0, 1], [0, 1], lambda s: 2 if s >= 2 else 0, 2, V1.AdviceCodeError)
    with pytest.raises(V1.AdviceCodeError, match="send_lengths is a bool"):
        V1.Registered(ENGINE, (EOS,), send_lengths=1)


def _ingress(record: V1.Record) -> dict:
    return {q.id: {**GREEDY, "prompt": list(q.prompt), "model": "m", "user": q.id, "max_tokens": q.max_tokens}
            for q in record.requests}
