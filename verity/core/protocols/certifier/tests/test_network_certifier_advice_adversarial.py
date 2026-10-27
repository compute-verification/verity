"""Adversarial cases against advice v0 (`advice`), the content advice whose bit length is the accepted-runs bound's
A_c (PROTOCOL.md, "Advice for arbitrary serving: what it can carry"). Each case is named for the attack or assumption
it pins and is expected to be refused (`MalformedAdvice`) or charged (`total_bits`, `bits_by_window`):

- A5, the charge is a canonical encoding: every string `decode` accepts is the `encode` of its advice, so at most
  2^(c+1) − 1 advices have c bits or fewer; wider fields, longer length prefixes, padding and trailing bytes refused.
- A5, the charge is causal: windows 0 … m − 1 are charged every record the frames released in them read.
- A2, one writer per value read: no position the next row reads is skipped, recomputed without a resume, or read
  from KV no request computed by that step; each request at most once a step; no free bit (`kind`) and no row past
  the one a stop allows.
- Attacks 2 and 7: schedule choices that leave the outputs unchanged are distinct advices, each paying its own bits.
"""

import random

import pytest

from verity.core.protocols.certifier import advice as A
from verity.core.protocols.certifier.advice import Advice, MalformedAdvice, RequestAdvice, Row, Source, StepAdvice


def _kind(start: int, n: int, prompt_len: int) -> str:
    return "decode" if n == 1 and start >= prompt_len else "prefill"


def schedule(seed: int, R: int = 5, max_prompt: int = 24, max_out: int = 6, T_gap: int = 3) -> Advice:
    """A random valid advice from a toy v1 scheduler: chunked prefills, decodes, prefix-cache hits on earlier requests'
    KV (one or two sources), preemptions resumed from their own KV or from scratch, the one row past a stop, and
    release buckets that follow the steps with jitter, non-decreasing per request."""
    rng = random.Random(seed)
    reqs = [RequestAdvice(f"u{seed}-{r}", rng.randint(1, max_prompt), rng.randint(1, max_out), rng.choice(A.FINISHES))
            for r in range(R)]
    computed, known, kept = [0] * R, [q.prompt_len for q in reqs], [0] * R
    made: list[int] = [0] * R  # each request computed positions h0 … made − 1, h0 its first row's hit
    first_hit: list[tuple[Source, ...]] = [()] * R
    trailed, admitted = [False] * R, 0
    steps: list[StepAdvice] = []
    release: list[int] = []
    last = [0] * R
    while admitted < R or any(kept[r] < reqs[r].output_len for r in range(R)):
        running, admits = [], []
        for r in range(admitted):
            q = reqs[r]
            if kept[r] == q.output_len:
                if not trailed[r] and computed[r] == known[r] - 1 and rng.random() < 0.3:
                    trailed[r] = True
                    running.append(Row(r, _kind(computed[r], 1, q.prompt_len), computed[r], 1))
                continue
            if rng.random() < 0.2:
                continue
            if rng.random() < 0.1 and computed[r] > 0:
                h = min(made[r], known[r] - 1) if rng.random() < 0.6 else 0
                lead = first_hit[r]
                own = h - sum(s.n for s in lead)
                if h and own < 0:
                    h, own = 0, 0
                sources = (lead + ((Source(r, own),) if own else ())) if h else ()
                n = rng.randint(1, known[r] - h)
                running.append(Row(r, _kind(h, n, q.prompt_len), h, n, h, True, sources))
                continue
            n = rng.randint(1, known[r] - computed[r])
            running.append(Row(r, _kind(computed[r], n, q.prompt_len), computed[r], n))
        rng.shuffle(running)
        while admitted < R and (not running and not admits or rng.random() < 0.4):
            r, q = admitted, reqs[admitted]
            sources: tuple[Source, ...] = ()
            donors = [x for x in range(admitted) if made[x] >= 1 and not first_hit[x]]
            if donors and q.prompt_len > 1 and rng.random() < 0.4:
                x = rng.choice(donors)
                h1 = rng.randint(1, min(made[x], q.prompt_len - 1))
                sources = (Source(x, h1),)
                more = [y for y in donors if y != x and made[y] > h1]
                if more and h1 < q.prompt_len - 1 and rng.random() < 0.5:
                    y = rng.choice(more)
                    sources += (Source(y, rng.randint(1, min(made[y], q.prompt_len - 1) - h1)),)
            h = sum(s.n for s in sources)
            n = rng.randint(1, known[r] - h)
            admits.append(Row(r, _kind(h, n, q.prompt_len), h, n, h, False, sources))
            first_hit[r] = sources
            admitted += 1
        rows = running + admits
        if not rows:
            continue
        s = len(steps)
        for row in rows:
            r = row.request
            end = row.start + row.n
            computed[r] = end
            if row in admits:
                made[r] = end
            elif row.start <= made[r]:
                made[r] = max(made[r], end)
            if end == known[r] and kept[r] < reqs[r].output_len:
                b = max(last[r], s * T_gap + rng.randint(0, 2 * T_gap))
                release.append(b)
                last[r] = b
                kept[r] += 1
                known[r] += 1
        steps.append(StepAdvice(tuple(rows)))
    # `release` is in outputs' (step, row) order, the order the loop appended them
    return Advice(tuple(reqs), tuple(steps), tuple(release))


SEEDS = range(40)


@pytest.fixture(scope="module")
def advices() -> list[Advice]:
    return [schedule(seed) for seed in SEEDS]


def test_toy_scheduler_exercises_every_field(advices):
    rows = [row for a in advices for st in a.steps for row in st.rows]
    assert any(len(row.sources) == 2 for row in rows)
    assert any(row.preempted and row.cache_hit for row in rows)
    assert any(row.preempted and not row.cache_hit for row in rows)
    assert any(row.kind == "prefill" and not row.cache_hit and row.start for row in rows)
    assert any(len(st.rows) > 2 for a in advices for st in a.steps)


# --- A5: the charge is a canonical encoding --------------------------------------------------------------------------

def test_round_trip(advices):
    for a in advices:
        data = A.encode(a)
        assert A.decode(data) == a
        assert A.from_json(A.to_json(a)) == a
        assert len(data) == -(-A.total_bits(a) // 8)


def test_every_accepted_mutation_is_its_own_canonical_encoding(advices):
    """Flip each bit, cut each byte, append one: whatever `decode` accepts, `encode` gives back byte for byte. So the
    encoding is injective and #advices with total_bits ≤ c is at most 2^(c+1) − 1, the count A_c needs."""
    accepted = 0
    for a in advices[:12]:
        data = A.encode(a)
        mutants = [data[:i] for i in range(len(data))] + [data + b"\0", data + b"\x80", data + b"\x01"]
        mutants += [bytes(data[:i >> 3]) + bytes([data[i >> 3] ^ 0x80 >> (i & 7)]) + data[(i >> 3) + 1:]
                    for i in range(8 * len(data))]
        for m in mutants:
            try:
                b = A.decode(m)
            except MalformedAdvice:
                continue
            accepted += 1
            assert b != a
            assert A.encode(b) == m
    assert accepted > 0  # some flips land on another valid advice: the property is exercised, not vacuous


def _encode_wide(a: Advice, dlen: int = 0, drel: int = 0, dsrc: int = 0, dcount: int = 0, pad: int = 0) -> bytes:
    """`advice._layout` with each width raised by its d, the request count's length prefix by dcount, and `pad`'s bits
    in the padding: a non-minimal or non-zero-padded encoding of the same advice."""
    w = A._widths(a)
    wl, wr, ws = w.len + dlen, w.rel + drel, w.src + dsrc
    out = A._Writer()
    out.put(A.VERSION, 8)
    n = len(a.requests)
    out.put(n.bit_length() + dcount, A.WIDTH_BITS)
    out.put(n, n.bit_length() + dcount)
    out.uvar(len(a.steps))
    out.put(wl, A.WIDTH_BITS)
    out.put(wr, A.WIDTH_BITS)
    out.put(ws, A.WIDTH_BITS)
    for q in a.requests:
        raw = q.id.encode("utf-8")
        out.uvar(len(raw))
        for byte in raw:
            out.put(byte, 8)
        out.put(q.prompt_len, wl)
        out.put(q.output_len, wl)
        out.put(A.FINISHES.index(q.finish), 1)
    for st in a.steps:
        out.put(len(st.rows) - 1, w.req)
        for row in st.rows:
            out.put(row.request, w.req)
            out.put(A.KINDS.index(row.kind), 1)
            for x in (row.start, row.n, row.cache_hit):
                out.put(x, wl)
            out.put(int(row.preempted), 1)
            if row.cache_hit:
                out.put(len(row.sources) - 1, ws)
                for src in row.sources:
                    out.put(src.request, w.req)
                    out.put(src.n, wl)
    for b in a.release:
        out.put(b, wr)
    data = bytearray(out.finish())
    if pad:
        assert out.held, "no padding bits to set"
        data[-1] |= 1
    return bytes(data)


def test_the_wide_encoder_is_the_encoder_at_zero(advices):
    for a in advices:
        assert _encode_wide(a) == A.encode(a)


@pytest.mark.parametrize("field", ["dlen", "drel", "dsrc", "dcount"])
def test_a_wider_field_is_refused(advices, field):
    """A non-minimal width would give one advice 2^k encodings: k free bits beside the charge."""
    tried = 0
    for a in advices:
        if field == "dsrc" and not any(row.cache_hit for st in a.steps for row in st.rows):
            continue
        tried += 1
        with pytest.raises(MalformedAdvice, match="least that hold|width isn't its bit length"):
            A.decode(_encode_wide(a, **{field: 1}))
    assert tried


def test_padding_and_trailing_bytes_are_refused(advices):
    for a in advices:
        data = A.encode(a)
        with pytest.raises(MalformedAdvice):
            A.decode(data + b"\0")
        if A.total_bits(a) % 8:
            with pytest.raises(MalformedAdvice):
                A.decode(_encode_wide(a, pad=1))


def test_a_version_or_type_other_than_v0_is_refused(advices):
    data = A.encode(advices[0])
    for bad in (bytes([1]) + data[1:], b"", list(data), data.hex()):
        with pytest.raises(MalformedAdvice):
            A.decode(bad)


# --- A5: the charge is causal ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("T", [1, 4, 16])
def test_every_window_prefix_is_charged_the_records_its_frames_read(advices, T):
    """For each m, Σ_{w<m} bits_w covers the header, every step up to the last that samples a frame released before
    window m, every request first scheduled by then, and those frames' release fields, measured from the layout."""
    for a in advices:
        charge = A.bits_by_window(a, T)
        assert sum(charge.values()) == A.total_bits(a)
        lay, outs = A._layout(a), A.outputs(a)
        first = {}
        for s, st in enumerate(a.steps):
            for row in st.rows:
                first.setdefault(row.request, s)
        windows = [b // T for b in a.release]
        for m in range(1, max(windows) + 2):
            shown = [o for o, w in zip(outs, windows) if w < m]
            if not shown:
                continue
            last = max(o.step for o in shown)
            need = (lay.header + sum(lay.steps[:last + 1]) + lay.release * len(shown)
                    + sum(n for r, n in enumerate(lay.requests) if first[r] <= last))
            assert sum(v for w, v in charge.items() if w < m) >= need


def test_an_empty_session_is_charged_its_header():
    a = Advice((), (), ())
    assert A.decode(A.encode(a)) == a
    assert A.bits_by_window(a, 4) == {0: A.total_bits(a)}


# --- A2: one writer per value read -----------------------------------------------------------------------------------

def _one(prompt: int = 8, out: int = 2, rid: str = "a") -> RequestAdvice:
    return RequestAdvice(rid, prompt, out, "eos")


def _decodes(r: int, prompt: int, out: int, step0: int) -> list[StepAdvice]:
    """Request r's decode steps after a whole prefill: one row each, positions prompt … prompt + out − 2."""
    return [StepAdvice((Row(r, "decode", prompt + i, 1),)) for i in range(out - 1)]


def test_the_honest_two_request_schedule_is_accepted():
    a = Advice((_one(rid="a"), _one(rid="b")),
               (StepAdvice((Row(0, "prefill", 0, 8),)),
                StepAdvice((Row(0, "decode", 8, 1), Row(1, "prefill", 4, 4, 4, False, (Source(0, 4),)))),
                StepAdvice((Row(1, "decode", 8, 1),))),
               (0, 1, 1, 2))
    assert len(A.outputs(a)) == 4


@pytest.mark.parametrize("name, steps, why", [
    ("skipped positions", lambda: [StepAdvice((Row(0, "prefill", 0, 4),)), StepAdvice((Row(0, "prefill", 5, 3),))],
     "continues at 4, not 5"),
    ("recomputed without a resume", lambda: [StepAdvice((Row(0, "prefill", 0, 4),)),
                                             StepAdvice((Row(0, "prefill", 2, 6),))], "continues at 4, not 2"),
    ("two rows of one request in one step", lambda: [StepAdvice((Row(0, "prefill", 0, 4), Row(0, "prefill", 4, 4)))],
     "at most once"),
    ("a first row past position 0 without a hit", lambda: [StepAdvice((Row(0, "prefill", 2, 6),))],
     "first row starts at its cache hit"),
    ("a hit on a running row", lambda: [StepAdvice((Row(0, "prefill", 0, 4),)),
                                        StepAdvice((Row(0, "prefill", 4, 4, 4, False, (Source(0, 4),)),))],
     "only on a row that admits"),
    ("a hit on its own KV without a resume", lambda: [StepAdvice((Row(0, "prefill", 0, 8),)),
                                                      StepAdvice((Row(0, "decode", 8, 1),)),
                                                      StepAdvice((Row(0, "prefill", 4, 5, 4, False, (Source(0, 4),)),))],
     "only on a row that admits"),
    ("rows past the known tokens", lambda: [StepAdvice((Row(0, "prefill", 0, 9),))], "past request 0's 8 known"),
])
def test_a_row_that_reads_unwritten_or_rewritten_positions_is_refused(name, steps, why):
    with pytest.raises(MalformedAdvice, match=why):
        Advice((_one(),), tuple(steps()), (0, 1))


@pytest.mark.parametrize("name, hit_row, why", [
    ("KV past what the donor computed", lambda: Row(1, "prefill", 6, 2, 6, False, (Source(0, 6),)), "computed no KV"),
    ("KV of a request not yet scheduled", lambda: Row(1, "prefill", 4, 4, 4, False, (Source(2, 4),)), "computed no KV"),
    ("sources that miss the hit", lambda: Row(1, "prefill", 4, 4, 4, False, (Source(0, 3),)), "sources cover 3"),
    ("one run split in two", lambda: Row(1, "prefill", 4, 4, 4, False, (Source(0, 2), Source(0, 2))), "one run"),
])
def test_a_cache_hit_on_kv_no_request_wrote_is_refused(name, hit_row, why):
    """Attack 6's dangerous form: a hit whose KV has no checked writer would give the circuit an unconstrained value."""
    reqs = (_one(rid="a", out=1), _one(rid="b", out=1), _one(rid="c", out=1))
    with pytest.raises(MalformedAdvice, match=why):
        steps = (StepAdvice((Row(0, "prefill", 0, 4),)), StepAdvice((hit_row(),)), StepAdvice((Row(2, "prefill", 0, 8),)))
        Advice(reqs, steps, (0, 1, 2))


def test_a_resume_reads_only_kv_its_request_computed():
    reqs = (_one(out=2),)
    chunk = [StepAdvice((Row(0, "prefill", 0, 4),))]
    tail = [StepAdvice((Row(0, "decode", 8, 1),))]
    ok = chunk + [StepAdvice((Row(0, "prefill", 4, 4, 4, True, (Source(0, 4),)),))] + tail
    assert len(A.outputs(Advice(reqs, tuple(ok), (0, 1)))) == 2
    for bad in (Row(0, "prefill", 6, 2, 6, True, (Source(0, 6),)),  # positions 4 and 5 were never computed
                Row(0, "prefill", 4, 4, 4, False, (Source(0, 4),))):  # not a resume
        with pytest.raises(MalformedAdvice, match="computed no KV|only on a row that admits"):
            Advice(reqs, tuple(chunk + [StepAdvice((bad,))] + tail), (0, 1))


def test_a_finished_request_has_at_most_its_one_trailing_row_and_no_resume():
    reqs = (_one(out=2),)
    done = [StepAdvice((Row(0, "prefill", 0, 8),)), StepAdvice((Row(0, "decode", 8, 1),))]
    assert len(A.outputs(Advice(reqs, tuple(done + [StepAdvice((Row(0, "decode", 9, 1),))]), (0, 1)))) == 2
    for tail, why in (([StepAdvice((Row(0, "decode", 9, 1),)), StepAdvice((Row(0, "decode", 10, 1),))], "past request 0"),
                      ([StepAdvice((Row(0, "prefill", 0, 9, 0, True),))], "finished and can't be preempted")):
        with pytest.raises(MalformedAdvice, match=why):
            Advice(reqs, tuple(done + tail), (0, 1))


def test_kind_is_not_a_free_bit(advices):
    for a in advices[:8]:
        st = a.steps[0]
        row = st.rows[0]
        flipped = Row(row.request, A.KINDS[1 - A.KINDS.index(row.kind)], row.start, row.n, row.cache_hit,
                      row.preempted, row.sources)
        with pytest.raises(MalformedAdvice, match="kind"):
            Advice(a.requests, (StepAdvice((flipped,) + st.rows[1:]),) + a.steps[1:], a.release)


@pytest.mark.parametrize("name, requests, release, why", [
    ("an output the steps don't sample", (_one(out=3),), (0, 1, 2), "kept 2 of its 3"),
    ("a release bucket too many", (_one(out=2),), (0, 1, 2), "3 release buckets for 2"),
    ("an output released before the one before it", (_one(out=2),), (5, 4), "before its output 0's"),
])
def test_the_outputs_and_their_releases_are_exactly_the_steps(name, requests, release, why):
    steps = (StepAdvice((Row(0, "prefill", 0, 8),)), StepAdvice((Row(0, "decode", 8, 1),)))
    with pytest.raises(MalformedAdvice, match=why):
        Advice(requests, steps, release)


def test_requests_are_in_order_of_first_row_and_ids_are_unique():
    """Attack 14's shared `user`: v0 keys a request by its body's `user`, so two honest requests that share one, or
    carry none, have no advice (a forced refusal), and the order is the first rows', not the ingress's."""
    steps = (StepAdvice((Row(1, "prefill", 0, 8), Row(0, "prefill", 0, 8))),)
    with pytest.raises(MalformedAdvice, match="order of first row"):
        Advice((_one(out=1, rid="a"), _one(out=1, rid="b")), steps, (0, 0))
    with pytest.raises(MalformedAdvice, match="share an id"):
        Advice((_one(out=1, rid="a"), _one(out=1, rid="a")),
               (StepAdvice((Row(0, "prefill", 0, 8), Row(1, "prefill", 0, 8))),), (0, 0))
    with pytest.raises(MalformedAdvice, match="non-empty"):
        RequestAdvice("", 8, 1, "eos")


# --- attacks 2 and 7: schedule choices with one egress ---------------------------------------------------------------

def test_schedule_choices_that_leave_the_outputs_unchanged_are_distinct_and_charged():
    """Chunking a prefill, recomputing a prefix instead of hitting the cache, and a resume that changes nothing all
    give the same outputs and releases; each is its own advice, so each choice is paid for in total_bits, and none is
    "the" serving's schedule (attack 7)."""
    reqs = (_one(rid="a", out=2), _one(rid="b", out=2))
    tail = (StepAdvice((Row(0, "decode", 8, 1), Row(1, "decode", 8, 1))),)
    variants = {
        "whole prefills": (StepAdvice((Row(0, "prefill", 0, 8), Row(1, "prefill", 0, 8))),) + tail,
        "b hits a's prefix": (StepAdvice((Row(0, "prefill", 0, 8), Row(1, "prefill", 4, 4, 4, False, (Source(0, 4),)))),)
                              + tail,
        "a chunked": (StepAdvice((Row(0, "prefill", 0, 5),)), StepAdvice((Row(0, "prefill", 5, 3), Row(1, "prefill", 0, 8))))
                     + tail,
        "a resumed in place": (StepAdvice((Row(0, "prefill", 0, 8), Row(1, "prefill", 0, 8))),
                               StepAdvice((Row(0, "decode", 8, 1, 8, True, (Source(0, 8),)), Row(1, "decode", 8, 1)))),
    }
    release = (0, 0, 1, 1)
    advs = {k: Advice(reqs, v, release) for k, v in variants.items()}
    shown = {k: [(o.request, o.index) for o in A.outputs(a)] for k, a in advs.items()}
    assert len({tuple(sorted(v)) for v in shown.values()}) == 1
    assert len({A.encode(a) for a in advs.values()}) == len(advs)
    assert all(A.total_bits(a) > 0 for a in advs.values())
