"""Advice v1 under a draft (`pipeline.advice_code.kept`, `schedule_code`): the drafts each emitting row kept are sent in
their step's window as its second coded segment and charged there, in the order the replay reads them as it runs, and
the string goes to and from the replay's schedule; I1 v0, whose rows emit one output each, is refused by name. Without
a draft every string is the one it was.

The schedules here are the advice code's own replay's, run on counts a hash picks row by row (`_served`): they hold
the string and its charge to the replay. That the replay is vLLM's scheduler under a draft is
`tests/properties/test_specdec_replay.py`'s.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import pytest
from verity.protocols.accounting.certifier.verifier.arith_encode import CW

from verity_vllm.pipeline.advice_code import by_class as BC
from verity_vllm.pipeline.advice_code import codec as V
from verity_vllm.pipeline.advice_code import kept as KP
from verity_vllm.pipeline.advice_code import replay as R
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
    K = reg.engine.num_speculative_tokens
    decided: list[list[int]] = [[] for _ in record.requests]
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


def _terms(data: bytes, record: V.Record, reg: V.Registered) -> Any:
    """The kept terms the string carries (`kept.Terms`), as its decoder's replay read them; None without a draft."""
    return V._replayed(*V._parse(data, reg, record), record, reg)[3]


def _second(reg: V.Registered, terms: KP.Terms) -> list[int]:
    """The bits a coder alone spends on each window's counts, with the 2 that end a segment."""
    alone, counts = CV.Encoder(V.book_of(reg)), KP.Counts(reg.engine.num_speculative_tokens, terms.outputs)
    for ks in terms.windows:
        alone.window()
        for i, c in ks:
            counts.put(alone, i, c)
    return list(alone.finish().lens)


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


def test_the_kept_counts_are_charged_in_their_steps_window():
    """Each emitting row's count is in its step's window, the earliest that releases an output at that step or later, in
    the order the replay reads them, and adds to that window exactly its second segment: the bits a coder spends on
    those counts, with the 2 that end a segment. Request 0's last row emits outputs 7 and 8, released in window 2, and
    its count is charged in window 1, where request 2's first output, a later step's, is released. A request whose
    every row can keep only none codes nothing."""
    record, reg, served = _case(3, tick=True)
    assert served.kept == ((0, 0, 2, 1, 1), (0, 2), (0, 0, 2, 0, 0))
    assert [[b // T for b in q.release] for q in record.requests] == [[0, 0, 1, 1, 1, 1, 2, 2, 2], [0, 0, 1, 1],
                                                                      [1, 2, 2, 2, 2, 3, 3]]
    data = SC.encode(served, record, reg)
    charge = V.bits_by_window(data, record, reg)
    lengths = [sum(w) for w in V.length_widths(record, reg)]
    terms = KP.Terms((9, 4, 7), (((0, 0), (1, 0), (0, 0), (1, 2)), ((0, 2), (0, 1), (0, 1), (2, 0)), ((2, 0), (2, 2)),
                                 ((2, 0), (2, 0))))
    assert _terms(data, record, reg) == terms and terms.by_request() == list(served.kept)
    assert lengths == [4 + 1 + 4 + 1, 3 + 1, 0, 0]
    assert [charge[w] - lengths[w] for w in charge] == _second(reg, terms) == [7, 7, 4, 4]
    one = replace(record, requests=tuple(replace(q, max_tokens=1, output=q.output[:1], release=q.release[:1])
                                         for q in record.requests))
    served = _served(one, reg, [0, 1, 2])
    assert served.kept == ((0,), (0,), (0,))
    assert V.bits_by_window(SC.encode(served, one, reg), one, reg) == {0: 2 * (1 + 1), 1: 1 + 1}


@pytest.mark.parametrize("asy, tick", ((False, False), (False, True), (True, False), (True, True)))
def test_the_replay_asks_each_emitting_rows_count_as_it_emits(asy: bool, tick: bool):
    """`replay.replay`'s ``kept`` is asked once for each row that emits, the steps in the order they execute and a
    step's rows in the scheduler's, and gives the replay `RequestIO.kept` gives; a what-if run on a fork asks
    nothing; and a count over the drafts its row carries is refused as `RequestIO.kept`'s is."""
    record, reg, served = _case(3, asy=asy, tick=tick)
    order = list(served.arrival)
    with KP.emitting() as emitted:
        rp, _, _ = V.replay_record(record, reg, reg.engine, order, lambda i, k: True, served.kept)
    rows, want = [0] * len(order), []
    for out in emitted:
        for i in out:
            want.append((i, rows[i]))
            rows[i] += 1
    asked: list[tuple[int, int]] = []
    forks: list[R.Emitted] = []

    def ask(i: int, row: int) -> int:
        asked.append((i, row))
        return served.kept[order[i]][row]

    def decide(i: int, k: int, o: R.Open) -> bool:
        forks.append(o.whatif(True, 2))
        return True

    got, _, _ = V.replay_record(record, reg, reg.engine, order, lambda i, k: True, ask)
    assert got.steps == rp.steps and asked == want and len(want) == sum(map(len, served.kept))
    with pytest.raises(ValueError, match="^request 0: emitting row 0 kept 3 drafts, but the row the replay schedules there "
                                         "carries 0$"):
        V.replay_record(record, reg, reg.engine, order, lambda i, k: True, lambda i, row: 3)
    if not tick:
        del asked[:]
        ios = V._ios(record, reg.engine, order, reg.config(reg.engine).stops)
        assert R.replay(reg.engine, ios, decide=decide, kept=ask).steps == rp.steps and asked == want
        assert any(out for fork in forks for out in fork)


def test_the_kept_segments_follow_the_last_windows_tail():
    """A drafted string is its windows and tails, as a string with no count has them, then each window's kept segment
    in the windows' order: the decoder has every decision and order entry before any count, and a count moved to
    another window's segment than its step's is refused."""
    for tick, order in ((False, (1, 0, 2)), (True, (0, 1, 2))):
        record, reg, served = _case(3, tick=tick, order=order)
        data = SC.encode(served, record, reg)
        wins, tails, d = V._parse(data, reg, record)
        first = list(d.lens)
        terms = V._replayed(wins, tails, d, record, reg)[3]
        assert any(w.arrived or w.order for w in wins) != tick
        sent = V.length_tails(record, reg)
        bits, charge = V.string_of(wins, reg, sent, terms)
        assert CV.pack(bits)[0] == data and charge == list(V.bits_by_window(data, record, reg).values())
        assert bits[:sum(first)] == CV.encode(wins, V.book_of(reg), sent).bitlist()
        assert [c - a for c, a in zip(charge, first)] == _second(reg, terms)
        w = next(w for w, ks in enumerate(terms.windows) if w and ks)
        moved = terms._replace(windows=(*terms.windows[:w - 1], (*terms.windows[w - 1], terms.windows[w][0]),
                                        terms.windows[w][1:], *terms.windows[w + 1:]))
        assert moved.by_request() == terms.by_request()
        with pytest.raises(V.AdviceCodeError, match=f"^window {w - 1}: "):
            SC.decode(CV.pack(V.string_of(wins, reg, sent, moved)[0])[0], record, reg)


def test_the_decoder_reads_each_count_from_the_window_the_replay_charges_it_to():
    """The window the decoder takes a step's counts from, the earliest that releases an output no row had emitted when
    the step began, is the one `codec.charged` gives that step off the replay that ran: the terms a string decodes to
    are `codec.kept_terms`' of its replay, window by window. That rests on each request's release buckets being in
    order, and a request whose buckets decrease is refused by name."""
    for K, asy, (tick, order, shared) in product((1, 3, 5), (False, True), ((False, (0, 1, 2), 0), (True, (0, 1, 2), 0),
                                                                           (False, (1, 0, 2), 0), (False, (0, 1, 2), 4))):
        record, reg, served = _case(K, asy=asy, tick=tick, order=order, shared=shared)
        with KP.emitting() as emitted:
            rp, _, _ = V.replay_record(record, reg, reg.engine, list(order), lambda i, k: True, served.kept)
        terms = V.kept_terms(record, reg, list(order), rp, emitted)
        assert _terms(SC.encode(served, record, reg), record, reg) == terms and len(terms.windows) > 2
    q = record.requests[0]
    assert replace(q, release=(q.release[0],) * len(q.output)).release == (q.release[0],) * len(q.output)
    with pytest.raises(V.AdviceCodeError, match="^request q0: its outputs' release buckets decrease$"):
        replace(q, release=(*q.release[:-2], q.release[-1] + 1, q.release[-1]))


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
        assert _terms(data, record, reg) is None
        with pytest.raises(V.AdviceCodeError, match="^without a draft every row keeps none"):
            SC.encode(served._replace(kept=((1, 0, 0, 0, 0, 0, 0, 0), *served.kept[1:])), record, reg)


def test_decode_takes_only_the_string_encode_gives_its_schedule():
    """Every prefix of a drafted string, every one-bit change and an extension: each is refused by name, the same way
    twice, or decodes to another schedule, whose string it is, bit for bit. A changed count the replay's rows can have
    is another schedule, so the counts are the prover's to choose and their bits are charged."""
    record, reg, served = _case(5, tick=True)
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

    def with_kept(k: int, counts: tuple[Any, ...]) -> SC.Schedule:
        return served._replace(kept=tuple(counts if j == k else c for j, c in enumerate(served.kept)))

    def coded(rows: Any) -> None:
        """One request of 9 outputs under 3 drafts a row: its window's counts `rows` through the coder."""
        e = CV.Encoder(CV.Book.uniform())
        e.window()
        e.segment(0)
        KP.coder(3, KP.Terms((9,), (tuple(rows),)))(e, 0)

    assert served.kept[0] == (0, 0, 2, 1, 1)
    needs = "request 0: the engine drafts 3 tokens a row, so the replay needs the drafts each of its rows kept"
    for counts, why in (((3, 3, 0), "request 0: emitting row 0 kept 3 drafts, but the row the replay schedules there carries 0"),
                        ((0, 3, 1, 3), needs), ((0, 3, 0, 1), needs), ((0, 0, 2, 1.0, 1), needs), ((0, 4, 3), needs)):
        with pytest.raises(V.AdviceCodeError, match=f"^{why}"):
            SC.encode(with_kept(0, counts), record, reg)
    for counts, why in (((0, 3, 1, 3), "request 0: a row kept 3 drafts where it can keep 0 to 1: 7 of its 9 outputs are out"),
                        ((0, 3, 0, 1), "request 0: its rows and the drafts they kept add up to 8 outputs, not its 9"),
                        ((0, 3, 3, 0), "request 0: a row emits after its 9 outputs"),
                        ((0, 0, 2, 1.0, 1), "request 0: a row kept 1.0 drafts where it can keep 0 to 3"),
                        ((0, 4, 3), "request 0: a row kept 4 drafts where it can keep 0 to 3")):
        with pytest.raises(ValueError, match=f"^{why}"):
            coded([(0, c) for c in counts])
    with pytest.raises(ValueError, match="^window 0: a kept count of request 1, and there are 1$"):
        coded([(0, 0), (1, 0)])
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
    wins, tails, d = V._parse(data, reg, record)
    assert V._replayed(wins, tails, d, record, reg)[3].by_request() == list(served.kept)
    with pytest.raises(V.AdviceCodeError, match="^under a draft, and only then, the string carries each window's kept terms"):
        V.string_of(wins, reg, V.length_tails(record, reg))
    plain_record, plain, _ = _case(0)
    rp, _, _ = V.replay_record(plain_record, plain, plain.engine, order, lambda i, k: True)
    with pytest.raises(V.AdviceCodeError, match="^under a draft, and only then, each step's outputs are read off the replay"):
        V.charged(plain_record, plain, order, rp, V.windows(plain_record, T), [{} for _ in rp.steps])


def test_the_kept_counts_are_a_class_of_their_own_in_the_charge_by_class():
    """Under a draft each window's charge splits into the fields it had and its second segment's two, the kept counts'
    bits and what ends them, which are the kept drafts' class: a hand-worked case under the tick, where the length
    terms are all else a window is charged; then, over drafts, scheduling and the tick, the parts sum to the window's
    charge and the class is the bits a coder alone spends on the window's counts. Beside the bits, the rows whose
    counts each window sends, those that kept a draft and the drafts kept. Without a draft there is no such field,
    class or row."""
    record, reg, served = _case(3, tick=True)
    data = SC.encode(served, record, reg)
    zero = dict.fromkeys(("decisions", "order", "hits", "resumes", "lag", "segment_end"), 0)
    assert BC.bits_by_class(data, record, reg) == {
        0: {**zero, "output_len": 8, "finish": 2, "kept": 5, "kept_end": 2},
        1: {**zero, "output_len": 3, "finish": 1, "kept": 5, "kept_end": 2},
        2: {**zero, "output_len": 0, "finish": 0, "kept": 2, "kept_end": 2},
        3: {**zero, "output_len": 0, "finish": 0, "kept": 2, "kept_end": 2}}
    assert BC.kept_rows(data, record, reg) == {0: (4, 1, 2), 1: (4, 3, 4), 2: (2, 1, 2), 3: (2, 0, 0)}
    assert BC.fields(reg) == BC.FIELDS + BC.KEPT and BC.classes(reg) == {**BC.CLASSES, "kept drafts": ("kept", "kept_end")}
    assert not set(BC.KEPT) & set(BC.closed(reg))
    file, line = BC.where("kept count")
    assert file == "kept.py" and 'e.bit("kept", c > j' in (Path(BC.__file__).parent / file).read_text().splitlines()[line - 1]
    for K, asy, tick in product((1, 3, 5), (False, True), (False, True)):
        record, reg, served = _case(K, asy=asy, tick=tick)
        data = SC.encode(served, record, reg)
        parts, terms = BC.bits_by_class(data, record, reg), _terms(data, record, reg)
        assert all(tuple(p) == BC.fields(reg) and min(p.values()) >= 0 for p in parts.values())
        assert {w: sum(p.values()) for w, p in parts.items()} == V.bits_by_window(data, record, reg)
        assert [sum(p[f] for f in BC.KEPT) for p in parts.values()] == _second(reg, terms)
        assert sorted(f for fs in BC.classes(reg).values() for f in fs) == sorted(BC.fields(reg))
        rows = BC.kept_rows(data, record, reg)
        assert [r.rows for r in rows.values()] == [len(w) for w in terms.windows]
        assert sum(r.rows for r in rows.values()) == sum(map(len, served.kept))
        assert sum(r.kept for r in rows.values()) == sum(c > 0 for cs in served.kept for c in cs)
        assert sum(r.rows + r.drafts for r in rows.values()) == sum(len(q.output) for q in record.requests)
    record, reg, served = _case(0)
    data = SC.encode(served, record, reg)
    assert BC.fields(reg) == BC.FIELDS and BC.classes(reg) == BC.CLASSES
    assert all(tuple(p) == BC.FIELDS for p in BC.bits_by_class(data, record, reg).values())
    with pytest.raises(V.AdviceCodeError, match="^the registered engine serves with no draft, so no row keeps one$"):
        BC.kept_rows(data, record, reg)


def test_a_request_admitted_before_its_bodys_window_is_carried_where_no_length_is_sent():
    """A count is charged in its own step's window, so a draft adds no rule on admission: a schedule that admits a
    request at a step whose window precedes its body's is refused where its length is sent there, as without a draft,
    and where no length is sent its string decodes to it, the counts of the rows before the body's window among them."""
    record, reg, _ = _case(3)
    late = replace(record, requests=(*record.requests[:2], replace(record.requests[2], end=9, release=tuple(range(11, 18)))))
    with pytest.raises(V.AdviceCodeError, match="^request q2: admitted at step 2, whose window 1 precedes window 2, "):
        SC.encode(_served(late, reg, [0, 1, 2]), late, reg)
    unsent = replace(reg, send_lengths=False)
    served = _served(late, unsent, [0, 1, 2])
    data = SC.encode(served, late, unsent)
    assert SC.decode(data, late, unsent) == served and late.requests[2].end // T == 2
    assert any(i == 2 for ks in _terms(data, late, unsent).windows[:2] for i, _ in ks)


def _with_l(reg: V.Registered, bits: int) -> V.Registered:
    """`reg` with ℓ = `bits`."""
    return V.Registered(reg.engine, reg.stop_ids, reg.row_order, reg.T, reg.book, bits, reg.tick, reg.send_lengths)


def test_a_drafted_string_over_the_registered_length_is_refused_by_name_at_encode_and_unread_at_decode():
    """A string over the registered ℓ is neither written nor read. With ℓ one bit under a drafted string,
    `schedule_code.encode` refuses it itself, with or without its decode back: by name, with the string's bits, ℓ and
    how many of its windows' charges fit. `schedule_code.decode` and `bits_by_window` refuse it before they read a
    bit, so other bytes of its length get the same refusal. With ℓ its bits, all three take it. Without a draft both
    encoders refuse the same way."""
    record, reg, served = _case(3, tick=True)
    data = SC.encode(served, record, reg)
    charge, n = V.bits_by_window(data, record, reg), 8 * len(data)
    assert list(charge.values()) == [17, 11, 4, 4] and n == 40 and _with_l(reg, 1 << 23) == reg
    short, exact = _with_l(reg, n - 1), _with_l(reg, n)
    written = ("^the string is 40 bits, over the registered ℓ = 39: its 4 windows are charged 36 bits, and the first 3 of them "
               "32, the most that fit under ℓ; end the session by then, or register a larger ℓ$")
    for check in (True, False):
        with pytest.raises(V.AdviceCodeError, match=written):
            SC.encode(served, record, short, check=check)
    for string in (data, b"\xff" * len(data)):
        for read in (SC.decode, V.bits_by_window):
            with pytest.raises(V.AdviceCodeError, match="^40 bits, over .* ℓ = 39$"):
                read(string, record, short)
    assert SC.encode(served, record, exact) == data and SC.encode(served, record, exact, check=False) == data
    assert SC.decode(data, record, exact) == served and V.bits_by_window(data, record, exact) == charge
    with pytest.raises(V.AdviceCodeError, match="^the string is 40 bits, over the registered ℓ = 7: .* the first 0 of them 0, "):
        SC.encode(served, record, _with_l(reg, 7), check=False)
    record, reg, served = _case(0)
    data = SC.encode(served, record, reg)
    advice, n = V.decode(data, record, reg), 8 * len(data)
    short, exact = _with_l(reg, n - 1), _with_l(reg, n)
    for check in (True, False):
        for encode, said in ((V.encode, advice), (SC.encode, served)):
            with pytest.raises(V.AdviceCodeError, match=f"^the string is {n} bits, over the registered ℓ = {n - 1}: its 4 "):
                encode(said, record, short, check=check)
            assert encode(said, record, exact, check=check) == data


def test_put_and_get_mirror_each_other():
    """Random counts of several requests under each K, their rows interleaved as steps interleave them, go through the
    coder and come back, as one segment charged to its window, and a segment cut short is refused."""
    rng = random.Random(0)
    for K in (1, 2, 3, 5):
        outputs = [rng.randrange(1, 40) for _ in range(8)]
        done, rows = [0] * 8, []
        while done != outputs:
            i = rng.choice([i for i in range(8) if done[i] < outputs[i]])
            rows.append((i, rng.randrange(min(K, outputs[i] - done[i] - 1) + 1)))
            done[i] += 1 + rows[-1][1]
        e, put = CV.Encoder(CV.Book.uniform()), KP.Counts(K, outputs)
        e.window()
        e.segment(0)
        for i, c in rows:
            put.put(e, i, c)
        put.end()
        enc = e.finish()
        assert enc.nbits > 2 and enc.lens == (enc.nbits,)
        d, get = CV.Decoder(enc.bits, enc.nbits, CV.Book.uniform()), KP.Counts(K, outputs)
        d.window()
        d.segment(0)
        assert [(i, get.get(d, i)) for i, _ in rows] == rows
        d.finish()
        get.end()
        assert d.lens == [enc.nbits]
        short, get = CV.Decoder(*CV.pack(enc.bitlist()[:-1]), CV.Book.uniform()), KP.Counts(K, outputs)
        short.window()
        short.segment(0)
        with pytest.raises(ValueError, match="window 0: "):
            for i, _ in rows:
                get.get(short, i)
            short.finish()
    assert [KP.context(None, 0), KP.context(0, 2), KP.context(5, 4)] == ["first.0", "0.2", "5.4"]


def test_the_terms_are_the_steps_rows_counts_in_each_steps_window():
    """`kept.terms` reads each row's count off what the steps emitted: the outputs from its first to its request's next
    row's first, less its own token, in its step's window and the steps' order; `Terms.by_request` gives each request's
    back in its rows' order."""
    emitted = [{0: 0}, {1: 0, 0: 1}, {}, {0: 4, 1: 1}, {1: 2}]
    terms = KP.terms(emitted, [6, 5], [0, 0, 1, 1, 3], 4)
    assert terms == KP.Terms((6, 5), (((0, 0), (1, 0), (0, 2)), ((0, 1), (1, 0)), (), ((1, 2),)))
    assert terms.by_request() == [(0, 2, 1), (0, 0, 2)]


def test_a_second_segment_is_charged_to_its_window_and_none_leaves_the_string_as_it_was():
    """`window_code.encode`'s ``coded`` adds each window's second segment to that window's charge, the segments after
    the last window's tail, and `Decoder.segment` reads each back after `decode_windows`; without it the string is bit
    for bit the one with no second segment, which is the other's start, and a second segment of a window not yet coded
    is refused."""
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
    assert both.bitlist()[:plain.nbits] == plain.bitlist() and both.nbits == plain.nbits + sum(extra)
    widths = [[n for _, n in t] for t in tails]
    d = CV.Decoder(both.bits, both.nbits, book)
    out = CV.decode_windows(d, widths)
    assert out == (wins, [[v for v, _ in t] for t in tails]) and d.lens == list(plain.lens)
    got: list[list[bool]] = []
    for w, fs in enumerate(flags):
        d.segment(w)
        got.append([d.bit("kept", w) for _ in fs])
    d.finish()
    assert got == flags and d.lens == list(both.lens)
    assert CV.decode_tailed(plain.bits, plain.nbits, book, widths) == out
    with pytest.raises(ValueError, match=f"^{sum(extra)} bits after the last window$"):
        CV.decode_tailed(both.bits, both.nbits, book, widths)
    for coder in (CV.Encoder(book), CV.Decoder(b"", 0, book)):
        with pytest.raises(ValueError, match="^a second segment of window 0, and 0 windows are coded: call window\\(\\) first$"):
            coder.segment(0)


def test_a_windows_charge_of_2_to_the_20_with_its_kept_segment_is_refused_by_name(monkeypatch):
    """The certifier counts a window's charge in `arith_encode.CW` = 20 bits, so a charge is under 2^20, its second
    segment counted. One window's kept counts take 24 bits. Where its first segment and length terms come to
    2^20 − 25 the window is charged 2^20 − 1 and its string decodes; where they come to 2^20 − 24, which alone is taken,
    `window_code.encode` refuses the window by name, and so does the decoder, on the string a coder with a wider count
    writes, where it ends the kept segment. A tail that alone reaches 2^20 is refused at the field that reaches it."""
    assert CW == 20 and CV.CW is CW
    book, K = CV.Book.uniform(), 3
    cs = [(7 * j) % (K + 1) for j in range(40)]
    terms = KP.Terms((sum(cs) + len(cs),), (tuple((0, c) for c in cs),))

    def tail(last: int) -> list[tuple[int, int]]:
        return [(0, 32)] * 32766 + [(0, last)]

    def read(string: CV.Encoded, last: int) -> tuple[list[int], list[int]]:
        d, counts = CV.Decoder(string.bits, string.nbits, book), KP.Counts(K, terms.outputs)
        assert CV.decode_windows(d, [[n for _, n in tail(last)]])[0] == [CV.Window()]
        d.segment(0)
        got = [counts.get(d, 0) for _ in cs]
        d.finish()
        return got, d.lens

    under = CV.encode([CV.Window()], book, [tail(18)], KP.coder(K, terms))
    assert CV.encode([CV.Window()], book, [tail(18)]).lens == ((1 << CW) - 25,) and under.lens == ((1 << CW) - 1,)
    assert read(under, 18) == (cs, [(1 << CW) - 1])
    assert CV.encode([CV.Window()], book, [tail(19)]).lens == ((1 << CW) - 24,)
    said = (f"^window 0: its charge is {1 << CW} bits, and the certifier counts a window's charge in 20 bits, so it is under "
            f"{1 << CW}; register a shorter window, so that less is charged to one$")
    with pytest.raises(ValueError, match=said):
        CV.encode([CV.Window()], book, [tail(19)], KP.coder(K, terms))
    with monkeypatch.context() as wider:
        wider.setattr(CV, "CW", CW + 1)
        at = CV.encode([CV.Window()], book, [tail(19)], KP.coder(K, terms))
        assert at.lens == (1 << CW,) and read(at, 19) == (cs, [1 << CW])
    with pytest.raises(ValueError, match=said):
        read(at, 19)
    with pytest.raises(ValueError, match=said):
        CV.encode([CV.Window()], book, [tail(19 + 24)])
    with pytest.raises(ValueError, match=f"^window 0: its charge is {(1 << CW) + 13} bits, "):
        CV.encode([CV.Window()], book, [tail(19 + 24 + 13)])
