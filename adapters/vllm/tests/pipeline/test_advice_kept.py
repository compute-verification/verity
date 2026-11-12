"""Advice v1 under a draft (`pipeline.advice_code.kept`, `schedule_code`): the drafts each emitting row kept are sent in
their request's body's window as its second coded segment and charged there, the replay runs on them, and the string
goes to and from the replay's schedule; I1 v0, whose rows emit one output each, is refused by name. Without a draft
every string is the one it was.

The schedules here are the advice code's own replay's, run on counts a hash picks row by row (`_served`): they hold
the string and its charge to the replay. That the replay is vLLM's scheduler under a draft is
`tests/properties/test_specdec_replay.py`'s.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from itertools import product

import pytest

from verity_vllm.pipeline.advice_code import by_class as BC
from verity_vllm.pipeline.advice_code import codec as V
from verity_vllm.pipeline.advice_code import kept as KP
from verity_vllm.pipeline.advice_code import schedule as S
from verity_vllm.pipeline.advice_code import schedule_code as SC
from verity_vllm.pipeline.advice_code import window_code as CV

EOS = 2
TICK = S.Tick(bucket_us=1000, sat_us=1000, free_us=1000)
T = 4
#: each request's max_tokens, output ids and last ingress bucket: one cut at its max_tokens, one ended by EOS, one cut;
#: the first two bodies land in window 0 and the third in window 1
REQUESTS = ((9, (5, 6, 7, 8, 9, 10, 11, 12, 13), 0), (12, (5, 6, 7, EOS), 0), (7, (5, 6, 7, 8, 9, 10, 11), 5))


def _record(shared: int = 0) -> V.Record:
    """Three requests of 6 prompt tokens, the first `shared` of them the same in each."""
    return V.Record(tuple(V.Request(f"q{k}", (50,) * shared + tuple(100 + 10 * k + j for j in range(6 - shared)), m, o, end,
                                    tuple(range(end + 2, end + 2 + len(o)))) for k, (m, o, end) in enumerate(REQUESTS)))


def _engine(K: int, **kw) -> S.Engine:
    """Two sequences and a 16-token budget on 4-token blocks; `K` drafts a row, or no draft."""
    kw = dict(max_num_seqs=2, max_num_batched_tokens=16, max_model_len=64, num_gpu_blocks=64, block_size=4, **kw)
    return S.DraftEngine(**kw, num_speculative_tokens=K) if K else S.Engine(**kw)


def _pick(k: int, row: int, most: int) -> int:
    """A stand-in for the model's accept rule: a hash of the request and its emitting row, 0 to `most`."""
    return hashlib.sha512(f"kept {k} {row}".encode()).digest()[0] % (most + 1)


def _served(record: V.Record, reg: V.Registered, order: list[int]) -> SC.Schedule:
    """The schedule of a serve under `reg` that took the requests in `order`, each as soon as the scheduler could: the
    replay's, on counts decided one emitting row at a time in the steps' order. A row keeps `_pick` of the drafts it
    carries and its request still lacks, and the rows not yet decided keep none, which every row can."""
    K, decided = reg.engine.num_speculative_tokens, [[] for _ in record.requests]
    while True:
        kept = [tuple(d) + (0,) * (len(q.output) - len(d) - sum(d)) for d, q in zip(decided, record.requests)]
        if K:
            with KP.emitting() as emitted:
                rp, _, _ = V.replay_record(record, reg, reg.engine, order, lambda i, k: True, kept)
        else:
            rp, _, _ = V.replay_record(record, reg, reg.engine, order, lambda i, k: True)
            emitted = []
        seen, undecided = [0] * len(order), None
        for st, out in zip(rp.steps, emitted):
            for i in sorted(out):
                k = order[i]
                if undecided is None and seen[i] == len(decided[k]):
                    undecided = (k, min(st.drafts.get(i, 0), len(record.requests[k].output) - out[i] - 1))
                seen[i] += 1
        if undecided is None:
            return SC.Schedule(tuple(order), tuple(tuple((order[i], a, n) for i, a, n in st.rows) for st in rp.steps),
                               tuple(kept))
        decided[undecided[0]].append(_pick(undecided[0], len(decided[undecided[0]]), undecided[1]))


def _case(K: int = 3, *, asy: bool = False, tick: bool = False, order: tuple[int, ...] = (0, 1, 2), shared: int = 0,
          **kw) -> tuple[V.Record, V.Registered, SC.Schedule]:
    record = _record(shared)
    reg = V.Registered(_engine(K, async_scheduling=asy, enable_prefix_caching=bool(shared)), (EOS,), T=T,
                       tick=TICK if tick else None, **kw)
    return record, reg, _served(record, reg, list(order))


def _bits(data: bytes, record: V.Record, reg: V.Registered) -> list[int]:
    return CV.unpack(data, sum(V.bits_by_window(data, record, reg).values()))


def test_a_drafted_schedule_encodes_decodes_and_replays_with_every_step_equal():
    """Under 1, 3 and 5 drafts, sync and async, with and without the certified tick, with a prefix hit, and with an
    arrival order that isn't release order: the string decodes to the schedule it encodes, every step's rows and every
    row's kept count, its windows' charges are its bits, and the same registration refuses I1 v0 by name."""
    carried = kept = 0
    for K, asy, (tick, order, shared) in product((1, 3, 5), (False, True), ((False, (0, 1, 2), 0), (True, (0, 1, 2), 0),
                                                                           (False, (1, 0, 2), 0), (False, (0, 1, 2), 4))):
        record, reg, served = _case(K, asy=asy, tick=tick, order=order, shared=shared)
        data = SC.encode(served, record, reg)
        assert SC.decode(data, record, reg) == served
        charge = V.bits_by_window(data, record, reg)
        assert len(charge) == V.windows(record, T) and 0 <= 8 * len(data) - sum(charge.values()) < 8
        assert all(len(c) + sum(c) == len(q.output) and max(c) <= K for c, q in zip(served.kept, record.requests))
        kept += sum(map(sum, served.kept))
        carried += sum(n - 1 for rows in served.steps for _, start, n in rows if start >= 6)
        with pytest.raises(V.AdviceCodeError, match="^the registered engine drafts, and I1 v0's rows emit one output each"):
            V.decode(data, record, reg)
    assert 0 < kept < carried


def test_the_kept_counts_are_charged_in_their_requests_body_window():
    """The counts add to a window exactly its second segment: the bits a coder spends on the counts of the requests
    whose body lands there, in release order, with the 2 that end a segment. A request whose every row can keep only
    none codes nothing."""
    record, reg, served = _case(3, tick=True)
    assert served.kept == ((0, 0, 2, 1, 1), (0, 2), (0, 0, 2, 0, 0))
    data = SC.encode(served, record, reg)
    charge = V.bits_by_window(data, record, reg)
    lengths = [sum(w) for w in V.length_widths(record, reg)]
    assert V.kept_terms(record, reg) == [[0, 1], [2], [], []] and lengths == [4 + 1 + 4 + 1, 3 + 1, 0, 0]
    alone = CV.Encoder(V.book_of(reg))
    for ks in V.kept_terms(record, reg):
        alone.window()
        for k in ks:
            KP.put(alone, 3, len(record.requests[k].output), served.kept[k])
    second = list(alone.finish().lens)
    assert [charge[w] - lengths[w] for w in charge] == second == [12, 6, 0, 0]
    one = replace(record, requests=tuple(replace(q, max_tokens=1, output=q.output[:1], release=q.release[:1])
                                         for q in record.requests))
    served = _served(one, reg, [0, 1, 2])
    assert served.kept == ((0,), (0,), (0,))
    assert V.bits_by_window(SC.encode(served, one, reg), one, reg) == {0: 2 * (1 + 1), 1: 1 + 1}


def test_without_a_draft_every_string_is_the_one_it_was():
    """With no draft registered the schedule's string is `codec.encode`'s of the I1 v0 it decodes to, with no kept
    term in any window, and its bytes are the ones this code gave such a record before it carried kept counts."""
    was = {False: "15a68127804020", True: "9a3c"}
    for tick in (False, True):
        record, reg, served = _case(0, tick=tick)
        data = SC.encode(served, record, reg)
        advice = V.decode(data, record, reg)
        assert data == V.encode(advice, record, reg) and data.hex() == was[tick]
        ordinal = V.streams(advice, record)
        assert [sorted((ordinal[r.request], r.start, r.n) for r in st.rows) for st in advice.steps] == [
            sorted(rows) for rows in served.steps]
        assert SC.decode(data, record, reg) == served and served.kept == tuple((0,) * len(q.output) for q in record.requests)
        assert V.kept_terms(record, reg) == []
        with pytest.raises(V.AdviceCodeError, match="^without a draft every row keeps none"):
            SC.encode(served._replace(kept=((1, 0, 0, 0, 0, 0, 0, 0), *served.kept[1:])), record, reg)


def test_decode_takes_only_the_string_encode_gives_its_schedule():
    """Every prefix of a drafted string, every one-bit change and an extension: each is refused by name, the same way
    twice, or decodes to another schedule, whose string it is, bit for bit. A changed count the replay's rows can have
    is another schedule, so the counts are the prover's to choose and their bits are charged."""
    record, reg, served = _case(3, tick=True)
    data = SC.encode(served, record, reg)
    flips = [bytes(b ^ (1 << (7 - j % 8)) if i == j // 8 else b for i, b in enumerate(data)) for j in range(8 * len(data))]
    refused = others = 0
    for x in [data[:k] for k in range(len(data))] + flips + [data + b"\0"]:
        try:
            got = SC.decode(x, record, reg)
        except V.AdviceCodeError as e:
            with pytest.raises(V.AdviceCodeError) as again:
                SC.decode(x, record, reg)
            assert str(again.value) == str(e) and str(e)
            refused += 1
            continue
        assert got != served and got.kept != served.kept and SC.encode(got, record, reg) == x
        others += 1
    assert refused and others


def test_counts_no_row_can_have_are_refused_by_name():
    """A count over the drafts its row carries, counts that don't add up to the request's outputs or aren't a row's,
    and a schedule that isn't the replay's under its counts; and every reader of a drafted string or replay that would
    read it as one without a draft."""
    record, reg, served = _case(3)
    data = SC.encode(served, record, reg)

    def with_kept(k: int, counts: tuple[int, ...]) -> SC.Schedule:
        return served._replace(kept=tuple(counts if j == k else c for j, c in enumerate(served.kept)))

    assert served.kept[0] == (0, 0, 2, 1, 1)
    needs = "request 0: the engine drafts 3 tokens a row, so the replay needs the drafts each of its rows kept"
    for counts, why in (((3, 3, 0), "request 0: emitting row 0 kept 3 drafts, but the row the replay schedules there carries 0"),
                        ((0, 3, 1, 3), needs), ((0, 3, 0, 1), needs), ((0, 0, 2, 1.0, 1), needs), ((0, 4, 3), needs)):
        with pytest.raises(V.AdviceCodeError, match=f"^{why}"):
            SC.encode(with_kept(0, counts), record, reg)
    for counts, why in (((0, 3, 1, 3), "a row kept 3 drafts where it can keep 0 to 1: 7 of its request's 9 outputs are out"),
                        ((0, 3, 0, 1), "a request's rows and the drafts they kept add up to 8 outputs, not its 9"),
                        ((0, 3, 3, 0), "a request's rows before its last add up to its 9 outputs"),
                        ((0, 0, 2, 1.0, 1), "a row kept 1.0 drafts where it can keep 0 to 3"),
                        ((0, 4, 3), "a row kept 4 drafts where it can keep 0 to 3")):
        e = CV.Encoder(CV.Book.uniform())
        e.window()
        with pytest.raises(ValueError, match=f"^{why}"):
            KP.put(e, 3, 9, counts)
    with pytest.raises(V.AdviceCodeError, match=r"^the schedule isn't the replay's from step 2 \(6/7 steps match\)$"):
        SC.encode(with_kept(0, (0, 2, 0, 1, 1)), record, reg)
    with pytest.raises(V.AdviceCodeError, match="^the schedule's requests aren't the record's$"):
        SC.encode(served._replace(arrival=(0, 1, 1)), record, reg)
    with pytest.raises(V.AdviceCodeError, match="^request q2 has no row in the schedule$"):
        SC.encode(served._replace(steps=tuple(tuple(r for r in rows if r[0] != 2) for rows in served.steps)), record, reg)
    with pytest.raises(V.AdviceCodeError, match="^v1 encodes a Schedule here, not a tuple$"):
        SC.encode(tuple(served), record, reg)
    order = list(served.arrival)
    with pytest.raises(ValueError, match="so the replay needs the drafts each of its rows kept"):
        V.replay_record(record, reg, reg.engine, order, lambda i, k: True)
    rp, decisions, looks = V.replay_record(record, reg, reg.engine, order, lambda i, k: True, served.kept)
    with pytest.raises(V.AdviceCodeError, match="^under a draft, and only then, each step's outputs are read off the replay"):
        V.charged(record, reg, order, rp, V.windows(record, T))
    wins, tails, kept = V._parse(data, reg, record)
    assert kept == list(served.kept)
    with pytest.raises(V.AdviceCodeError, match="^under a draft, and only then, the string carries each window's kept terms"):
        V.string_of(wins, reg, V.length_tails(record, reg))
    with pytest.raises(V.AdviceCodeError, match="^under a draft a window's second segment is its kept counts'"):
        BC.bits_by_class(data, record, reg)
    plain_record, plain, _ = _case(0)
    rp, _, _ = V.replay_record(plain_record, plain, plain.engine, order, lambda i, k: True)
    with pytest.raises(V.AdviceCodeError, match="^under a draft, and only then, each step's outputs are read off the replay"):
        V.charged(plain_record, plain, order, rp, V.windows(plain_record, T), [{} for _ in rp.steps])


def test_a_request_admitted_before_its_bodys_window_is_refused_under_a_draft():
    """A count changes no row before its request's admission, and the charge is in the body's window: a schedule
    that admits a request at a step whose window precedes it is refused whether or not lengths are sent."""
    record, reg, _ = _case(3)
    late = replace(record, requests=(*record.requests[:2], replace(record.requests[2], end=9, release=tuple(range(11, 18)))))
    for lengths in (True, False):
        r = replace(reg, send_lengths=lengths)
        with pytest.raises(V.AdviceCodeError, match="^request q2: admitted at step 2, whose window 1 precedes window 2, "):
            SC.encode(_served(late, r, [0, 1, 2]), late, r)


def test_put_and_get_mirror_each_other():
    """Random counts of several requests under each K go through the coder and come back, as one segment charged to
    its window, and a segment cut short is refused."""
    rng = random.Random(0)
    for K in (1, 2, 3, 5):
        requests = []
        for _ in range(8):
            outputs, counts, done = rng.randrange(1, 40), [], 0
            while done < outputs:
                counts.append(rng.randrange(min(K, outputs - done - 1) + 1))
                done += 1 + counts[-1]
            requests.append((outputs, tuple(counts)))
        e = CV.Encoder(CV.Book.uniform())
        e.window()
        e.segment()
        for outputs, counts in requests:
            KP.put(e, K, outputs, counts)
        enc = e.finish()
        assert enc.nbits > 2 and enc.lens == (enc.nbits,)
        d = CV.Decoder(enc.bits, enc.nbits, CV.Book.uniform())
        d.window()
        d.segment()
        assert [KP.get(d, K, outputs) for outputs, _ in requests] == [counts for _, counts in requests]
        d.finish()
        assert d.lens == [enc.nbits]
        short = CV.Decoder(*CV.pack(enc.bitlist()[:-1]), CV.Book.uniform())
        short.window()
        short.segment()
        with pytest.raises(ValueError, match="window 0: "):
            for outputs, _ in requests:
                KP.get(short, K, outputs)
            short.finish()
    assert [KP.context(None, 0), KP.context(0, 2), KP.context(5, 4)] == ["first.0", "0.2", "5.4"]


def test_a_second_segment_is_charged_to_its_window_and_none_leaves_the_string_as_it_was():
    """`window_code.encode`'s ``coded`` adds each window's second segment to that window's charge, after its tail, and
    `decode_tailed` reads it back; without it the string is bit for bit the one with no second segment, and a second
    segment outside any window is refused."""
    book = CV.Book.uniform()
    wins = [CV.Window(arrived=(True, False)), CV.Window(), CV.Window(arrived=(False,))]
    tails = [[(5, 3)], [], [(1, 1), (2, 2)]]
    flags = [[True, False, True], [], [False]]
    plain = CV.encode(wins, book, tails)
    assert CV.encode(wins, book, tails, None) == plain
    both = CV.encode(wins, book, tails, lambda e, w: [e.bit("kept", b, w) for b in flags[w]])
    alone = CV.Encoder(book)
    for fs in flags:
        alone.window()
        for b in fs:
            alone.bit("kept", b, flags.index(fs))
    extra = alone.finish().lens
    assert extra == (6, 0, 3) and both.lens == tuple(a + b for a, b in zip(plain.lens, extra))
    got: list[list[bool]] = []
    out = CV.decode_tailed(both.bits, both.nbits, book, [[n for _, n in t] for t in tails],
                           lambda d, w: got.append([d.bit("kept", w) for _ in flags[w]]))
    assert out == (wins, [[v for v, _ in t] for t in tails]) and got == flags
    for coder in (CV.Encoder(book), CV.Decoder(b"", 0, book)):
        with pytest.raises(ValueError, match="^a second segment outside any window: call window\\(\\) first$"):
            coder.segment()
