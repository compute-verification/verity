"""Advice v0 (`advice`): the round trip, the bit layout rebuilt field by field from the module's documented one and
pinned on a fixture, the split by window summing to the total, and the refusals of malformed advice
and of every byte string that isn't a canonical encoding.

The random advices come from `simulate`, a small scheduler in vLLM v1's manner: running requests first under a token
budget, then FCFS admission up to a sequence cap, chunked prefill, prefix-cache hits from requests sharing a prompt
prefix (in the same step, as vLLM caches a block when it schedules it), recompute preemption with a resume that hits its
own and others' KV, EOS before the cap, the one row async scheduling runs past an EOS, and rows in a shuffled batch
order."""

import json
import random

import pytest

from verity.core.protocols.certifier.advice import (FINISHES, KINDS, SCHEMA, Advice, MalformedAdvice, Output, RequestAdvice,
                                                    Row, Source, StepAdvice, bits, bits_by_window, decode, encode, from_json,
                                                    outputs, to_json, total_bits)

P, D = "prefill", "decode"

#: two requests: a's 5-token prompt in two chunks (the first chunk's sample discarded), b admitted in a's second step
#: with a 2-token hit on a's KV computed the step before; b stops at its cap of 2, a by EOS after 3; each then gets the
#: one row async scheduling runs past its last output
FIXTURE = Advice(
    (RequestAdvice("a", 5, 3, "eos"), RequestAdvice("b", 4, 2, "length")),
    (StepAdvice((Row(0, P, 0, 3),)),
     StepAdvice((Row(0, P, 3, 2), Row(1, P, 2, 2, 2, False, (Source(0, 2),)))),
     StepAdvice((Row(0, D, 5, 1), Row(1, D, 4, 1))),
     StepAdvice((Row(0, D, 6, 1), Row(1, D, 5, 1))),
     StepAdvice((Row(0, D, 7, 1),))),
    (1, 1, 2, 2, 4))
FIXTURE_HEX = "000a0e861800d86b03628a0308d049215a43842e21d207902528"


def u(value: int, width: int) -> str:
    assert 0 <= value < (1 << width) or (width == 0 and value == 0)
    return format(value, f"0{width}b") if width else ""


def uvar(value: int, width: int | None = None) -> str:
    width = value.bit_length() if width is None else width
    return u(width, 6) + u(value, width)


def records(a: Advice, *, w_len: int | None = None, n_requests_width: int | None = None):
    """The documented layout, field by field: the header's bits, each request's, each step's and each release's
    (widths overridable to build non-canonical ones)."""
    w_req = max(len(a.requests) - 1, 0).bit_length()
    lens = [x for q in a.requests for x in (q.prompt_len, q.output_len)]
    lens += [x for st in a.steps for r in st.rows for x in (r.start, r.n, r.cache_hit, *(s.n for s in r.sources))]
    w_len = max(lens, default=0).bit_length() if w_len is None else w_len
    w_rel = max(a.release, default=0).bit_length()
    w_src = max((len(r.sources) - 1 for st in a.steps for r in st.rows if r.cache_hit), default=0).bit_length()
    header = u(0, 8) + uvar(len(a.requests), n_requests_width) + uvar(len(a.steps)) + u(w_len, 6) + u(w_rel, 6)
    header += u(w_src, 6)
    reqs = []
    for q in a.requests:
        raw = q.id.encode()
        s = uvar(len(raw)) + "".join(u(b, 8) for b in raw) + u(q.prompt_len, w_len) + u(q.output_len, w_len)
        reqs.append(s + u(FINISHES.index(q.finish), 1))
    steps = []
    for st in a.steps:
        s = u(len(st.rows) - 1, w_req)
        for r in st.rows:
            s += u(r.request, w_req) + u(KINDS.index(r.kind), 1) + u(r.start, w_len) + u(r.n, w_len) + u(r.cache_hit, w_len)
            s += u(int(r.preempted), 1)
            if r.cache_hit:
                s += u(len(r.sources) - 1, w_src) + "".join(u(x.request, w_req) + u(x.n, w_len) for x in r.sources)
        steps.append(s)
    return header, reqs, steps, [u(b, w_rel) for b in a.release]


def rebuild(a: Advice, **widths) -> str:
    header, reqs, steps, release = records(a, **widths)
    return header + "".join(reqs) + "".join(steps) + "".join(release)


def packed(s: str) -> bytes:
    s += "0" * (-len(s) % 8)
    return int(s, 2).to_bytes(len(s) // 8, "big") if s else b""


def simulate(rng: random.Random, *, requests: int = 8, budget: int = 10, seqs: int = 4, prefix: int = 6,
             preempt: float = 0.1) -> Advice:
    """A random schedule as vLLM v1's makes them (the module docstring); requests numbered in order of first row."""
    spec = []
    for i in range(requests):
        cap = rng.randint(1, 6)
        out = rng.randint(1, cap)
        spec.append({"id": f"user-{i}", "group": rng.randrange(2), "P": rng.randint(1, 14), "out": out,
                     "finish": "length" if out == cap else "eos", "arrive": rng.randrange(12)})
    spec.sort(key=lambda q: q["arrive"])
    computed = [0] * requests
    known = [q["P"] for q in spec]
    kept = [0] * requests
    made: list[set[int]] = [set() for _ in range(requests)]
    admitted, preempted, extra = set(), set(), set()
    waiting, running, done = list(range(requests)), [], set()
    steps: list[list[tuple[Row, bool]]] = []

    def shares(i: int, j: int, p: int) -> bool:
        """Do i's and j's tokens agree at position p (one group's prompts share their first `prefix` tokens)?"""
        return i == j or (spec[i]["group"] == spec[j]["group"] and p < min(prefix, spec[i]["P"], spec[j]["P"]))

    def hit(i: int) -> list[list[int]]:
        if rng.random() < 0.3:
            return []
        runs: list[list[int]] = []
        for p in range(known[i] - 1):
            owners = [j for j in sorted(admitted | {i}) if p in made[j] and shares(i, j, p)]
            if not owners:
                break
            j = runs[-1][0] if runs and runs[-1][0] in owners else rng.choice(owners)
            if runs and runs[-1][0] == j:
                runs[-1][1] += 1
            else:
                runs.append([j, 1])
        return runs

    def add(i: int, start: int, n: int, hit_runs=()) -> None:
        end = start + n
        is_kept = end == known[i] and kept[i] < spec[i]["out"]
        kind = D if n == 1 and start >= spec[i]["P"] else P
        cache = sum(m for _, m in hit_runs)
        row = Row(i, kind, start, n, cache, i in preempted and i in admitted,
                  tuple(Source(j, m) for j, m in hit_runs))
        rows.append((row, is_kept))
        computed[i] = end
        made[i].update(range(start, end))
        if is_kept:
            kept[i] += 1
            known[i] += 1
            if kept[i] == spec[i]["out"]:
                done.add(i)
                if spec[i]["finish"] == "eos" or rng.random() < 0.3:
                    extra.add(i)

    for t in range(500):
        rows: list[tuple[Row, bool]] = []
        tokens = budget
        if len(running) > 1 and rng.random() < preempt and running[-1] not in done:
            i = running.pop()
            preempted.add(i)
            waiting.insert(0, i)
        for i in list(running):
            if i in done:
                if i in extra:
                    add(i, computed[i], 1)
                    extra.discard(i)
                running.remove(i)
                continue
            n = min(known[i] - computed[i], tokens)
            if n:
                add(i, computed[i], n)
                tokens -= n
        while waiting and len(running) < seqs and tokens > 0 and spec[waiting[0]]["arrive"] <= t:
            i = waiting.pop(0)
            runs = hit(i)
            start = sum(m for _, m in runs)
            n = min(known[i] - start, tokens)
            add(i, start, n, runs)
            admitted.add(i)
            tokens -= n
            running.append(i)
        for row, _ in rows:
            if row.preempted:
                preempted.discard(row.request)
        if rows:
            rng.shuffle(rows)
            steps.append(rows)
        if len(done) == requests and not extra:
            break
    order: dict[int, int] = {}
    for rows in steps:
        for row, _ in rows:
            order.setdefault(row.request, len(order))
    release, last = [], {}
    for s, rows in enumerate(steps):
        for row, is_kept in rows:
            if is_kept:
                b = max(last.get(row.request, 0), s // 3 + rng.choice((0, 0, 1)))
                last[row.request] = b
                release.append(b)
    remap = lambda row: Row(order[row.request], row.kind, row.start, row.n, row.cache_hit, row.preempted,  # noqa: E731
                            tuple(Source(order[x.request], x.n) for x in row.sources))
    reqs = [None] * requests
    for i, k in order.items():
        q = spec[i]
        reqs[k] = RequestAdvice(q["id"], q["P"], q["out"], q["finish"])
    return Advice(tuple(reqs), tuple(StepAdvice(tuple(remap(r) for r, _ in rows)) for rows in steps), tuple(release))


RANDOM = [simulate(random.Random(seed)) for seed in range(200)]


# --- the round trip and the layout ---------------------------------------------------------------------------------


def test_fixture_bytes_are_pinned_and_the_documented_layout():
    assert encode(FIXTURE).hex() == FIXTURE_HEX
    assert encode(FIXTURE) == packed(rebuild(FIXTURE))
    assert total_bits(FIXTURE) == len(rebuild(FIXTURE)) == 207
    assert decode(bytes.fromhex(FIXTURE_HEX)) == FIXTURE


def test_fixture_outputs_follow_the_schedule():
    """A partial chunk keeps no sample, a completing chunk and each decode keep one, and the rows past the last outputs
    keep none."""
    assert outputs(FIXTURE) == (Output(1, 0, 0, 0), Output(1, 1, 1, 0), Output(2, 0, 0, 1), Output(2, 1, 1, 1),
                                Output(3, 0, 0, 2))


@pytest.mark.parametrize("a", RANDOM[:50] + [FIXTURE, Advice((), (), ())])
def test_round_trip(a):
    data = encode(a)
    assert decode(data) == a
    assert encode(decode(data)) == data
    assert from_json(json.loads(json.dumps(to_json(a)))) == a
    assert data == packed(rebuild(a))
    assert len(data) == -(-total_bits(a) // 8)


def test_random_schedules_exercise_every_decision():
    """The simulator's advices hold chunked prefill, cache hits (multi-run and same-step ones among them), preemptions,
    rows past the last output, and both finishes: what the round trip above covers."""
    rows = [(a, s, r) for a in RANDOM for s, st in enumerate(a.steps) for r in st.rows]
    assert any(r.kind == P and r.start + r.n < a.requests[r.request].prompt_len for a, _, r in rows)
    assert any(len(r.sources) > 1 for _, _, r in rows)
    assert any(r.preempted for _, _, r in rows)
    assert any(r.preempted and any(x.request == r.request for x in r.sources) for _, _, r in rows)
    assert any(any(x.request in {o.request for o in a.steps[s].rows} for x in r.sources) for a, s, r in rows)
    assert {q.finish for a in RANDOM for q in a.requests} == set(FINISHES)
    past = 0
    for a in RANDOM:
        last = {o.request: (o.step, o.row) for o in outputs(a)}
        past += sum(1 for s, st in enumerate(a.steps) for j, r in enumerate(st.rows) if (s, j) > last[r.request])
    assert past > 0


def test_lists_are_taken_as_tuples():
    a = Advice(list(FIXTURE.requests), [StepAdvice(list(st.rows)) for st in FIXTURE.steps], list(FIXTURE.release))
    assert a == FIXTURE and hash(a) == hash(FIXTURE)


# --- by window -----------------------------------------------------------------------------------------------------


def test_fixture_charge_per_window():
    """At T = 2 the outputs are released in windows 0, 0, 1, 1, 2. Window 0: the header (43 bits), both requests (22
    each), steps 0 and 1 (13 and 29) and two releases (3 each); window 1: step 2 (25) and two releases; window 2: step 3
    (25), step 4, which no output follows (13), and one release."""
    assert bits_by_window(FIXTURE, 2) == {0: 135, 1: 31, 2: 41}
    assert [bits(FIXTURE, w, 2) for w in range(4)] == [135, 31, 41, 0]
    assert bits_by_window(FIXTURE, 1000) == {0: 207}


@pytest.mark.parametrize("T", [1, 2, 3, 7, 1000])
def test_charge_sums_to_the_total(T):
    for a in RANDOM:
        charge = bits_by_window(a, T)
        assert sum(charge.values()) == total_bits(a)
        assert all(n > 0 for n in charge.values())
        assert sum(bits(a, w, T) for w in range(max(charge) + 1)) == total_bits(a)


@pytest.mark.parametrize("T", [1, 3])
def test_charge_is_the_rule_and_early_enough(T):
    """Recomputed from the documented records: each record in the earliest window it can show in. Then windows 0 … w
    are charged at least what their outputs read: the header, every step up to the last of their outputs' steps, the
    requests those steps admit, and their releases."""
    for a in RANDOM[:60]:
        header, reqs, steps, release = (len(x) if isinstance(x, str) else [len(y) for y in x] for x in records(a))
        out = outputs(a)
        windows = [b // T for b in a.release]
        first = {}
        for s, st in enumerate(a.steps):
            for r in st.rows:
                first.setdefault(r.request, s)
        later = [[w for o, w in zip(out, windows) if o.step >= s] for s in range(len(a.steps))]
        earliest = [min(ws) if ws else max(windows) for ws in later]
        expect: dict[int, int] = {}
        for w, n in [(earliest[0], header), *((earliest[first[r]], n) for r, n in enumerate(reqs)),
                     *zip(earliest, steps), *zip(windows, release)]:
            expect[w] = expect.get(w, 0) + n
        charge = bits_by_window(a, T)
        assert charge == {w: n for w, n in expect.items() if n}
        for w in range(max(windows) + 1):
            mine = [o for o, x in zip(out, windows) if x <= w]
            if not mine:
                continue
            last = max(o.step for o in mine)
            needed = header + sum(steps[:last + 1]) + sum(n for r, n in enumerate(reqs) if first[r] <= last)
            needed += sum(n for n, x in zip(release, windows) if x <= w)
            assert sum(n for x, n in charge.items() if x <= w) >= needed


def test_empty_advice_is_its_header():
    a = Advice((), (), ())
    assert encode(a) == bytes(5) and total_bits(a) == 38
    assert bits_by_window(a, 600) == {0: 38}


# --- refusals ------------------------------------------------------------------------------------------------------


def with_steps(*steps, requests=FIXTURE.requests, release=None):
    a_steps = tuple(StepAdvice(tuple(rows)) for rows in steps)
    return lambda: Advice(requests, a_steps, FIXTURE.release if release is None else release)


S0, S1, S2, S3, S4 = (st.rows for st in FIXTURE.steps)

REFUSED = {
    "empty id": lambda: RequestAdvice("", 1, 1, "eos"),
    "id not str": lambda: RequestAdvice(b"a", 1, 1, "eos"),
    "id not UTF-8": lambda: RequestAdvice("\ud800", 1, 1, "eos"),
    "empty prompt": lambda: RequestAdvice("a", 0, 1, "eos"),
    "no output": lambda: RequestAdvice("a", 1, 0, "eos"),
    "finish stop": lambda: RequestAdvice("a", 1, 1, "stop"),
    "bool as int": lambda: RequestAdvice("a", True, 1, "eos"),
    "2^63": lambda: RequestAdvice("a", 1 << 63, 1, "eos"),
    "kind chunk": lambda: Row(0, "chunk", 0, 1),
    "n 0": lambda: Row(0, P, 0, 0),
    "negative start": lambda: Row(0, P, -1, 1),
    "preempted int": lambda: Row(0, P, 0, 1, 0, 1),
    "sources short of the hit": lambda: Row(0, P, 2, 1, 2, False, (Source(1, 1),)),
    "adjacent sources of one request": lambda: Row(0, P, 2, 1, 2, False, (Source(1, 1), Source(1, 1))),
    "hit not the start": lambda: Row(0, P, 3, 1, 2, False, (Source(1, 2),)),
    "a source of 0 tokens": lambda: Source(1, 0),
    "sources a list of tuples": lambda: Row(0, P, 2, 1, 2, False, ((1, 2),)),
    "empty step": lambda: StepAdvice(()),
    "a request twice in a step": lambda: StepAdvice((Row(0, P, 0, 1), Row(0, P, 1, 1))),
    "duplicate ids": lambda: Advice((RequestAdvice("a", 1, 1, "eos"), RequestAdvice("a", 1, 1, "eos")),
                                    (StepAdvice((Row(0, P, 0, 1), Row(1, P, 0, 1))),), (0, 0)),
    "release short": with_steps(S0, S1, S2, S3, S4, release=(1, 1, 2, 2)),
    "release before the request's last": with_steps(S0, S1, S2, S3, S4, release=(1, 1, 2, 0, 4)),
    "release negative": with_steps(S0, S1, S2, S3, S4, release=(-1, 1, 2, 2, 4)),
    "a request index past the requests": with_steps(S0, S1, S2, S3, (Row(2, D, 7, 1),)),
    "requests out of first-row order": with_steps((Row(1, P, 0, 3),), S1, S2, S3, S4),
    "a request with no row": with_steps(S0, S1, S2, S3, S4, requests=FIXTURE.requests + (RequestAdvice("c", 1, 1, "eos"),)),
    "a request short of its outputs": with_steps(S0, S1, S2, S3, release=(1, 1, 2, 2), requests=(
        RequestAdvice("a", 5, 4, "eos"), FIXTURE.requests[1])),
    "a chunk past the known tokens": with_steps((Row(0, P, 0, 6),), S1, S2, S3, S4),
    "a gap": with_steps(S0, (Row(0, P, 4, 1), S1[1]), S2, S3, S4),
    "a decode before the prompt is in": with_steps(S0, (Row(0, D, 3, 1), S1[1]), S2, S3, S4),
    "kind mismatch": with_steps(S0, S1, (Row(0, P, 5, 1), S2[1]), S3, S4),
    "a hit on a row that doesn't admit": with_steps(S0, S1, (S2[0], Row(1, P, 4, 1, 4, False, (Source(0, 4),))), S3, S4),
    "a first row that resumes": with_steps((Row(0, P, 0, 3, 0, True),), S1, S2, S3, S4),
    "a first row past its hit": with_steps(S0, (S1[0], Row(1, P, 3, 1)), S2, S3, S4),
    "a source that computed nothing there": lambda: Advice(
        (FIXTURE.requests[0], RequestAdvice("b", 6, 1, "length")),
        (StepAdvice((Row(0, P, 0, 3), Row(1, P, 4, 2, 4, False, (Source(0, 4),)))),), (0,)),
    "a source not yet admitted": with_steps((Row(0, P, 0, 3),), (S1[0], Row(1, P, 2, 2, 2, False, (Source(1, 2),))), S2, S3,
                                            S4),
    "a resume of a finished request": with_steps(S0, S1, S2, (S3[0], Row(1, P, 0, 6, 0, True)), S4),
    "a second row past the last output": with_steps(S0, S1, S2, S3, (S4[0], Row(1, D, 6, 1))),
}


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_malformed_advice_is_refused(name):
    with pytest.raises(MalformedAdvice):
        REFUSED[name]()


def test_a_self_source_needs_a_resume():
    """a resumes after a preemption at step 2 with a 4-token hit on its own KV; without the resume the same hit is
    refused (a request reuses its own KV only when it was preempted)."""
    reqs = (RequestAdvice("a", 5, 2, "length"),)
    steps = lambda pre: (StepAdvice((Row(0, P, 0, 5),)), StepAdvice((Row(0, D, 5, 1),)),  # noqa: E731
                         StepAdvice((Row(0, P, 4, 3, 4, pre, (Source(0, 4),)),)))
    reqs3 = (RequestAdvice("a", 5, 3, "length"),)
    assert outputs(Advice(reqs3, steps(True), (0, 0, 1)))[-1] == Output(2, 0, 0, 2)
    with pytest.raises(MalformedAdvice):
        Advice(reqs3, steps(False), (0, 0, 1))
    with pytest.raises(MalformedAdvice):
        Advice(reqs, steps(True), (0, 0))


@pytest.mark.parametrize("seed", range(20))
def test_mutations_of_valid_advice_are_refused(seed):
    """A kind flipped, a start moved and two requests of different lengths swapped each break the schedule's rules."""
    rng = random.Random(seed)
    a = RANDOM[seed]
    s = rng.randrange(len(a.steps))
    j = rng.randrange(len(a.steps[s].rows))
    row = a.steps[s].rows[j]

    def put(kind, start):
        new = Row(row.request, kind, start, row.n, row.cache_hit, row.preempted, row.sources)
        rows = a.steps[s].rows[:j] + (new,) + a.steps[s].rows[j + 1:]
        return Advice(a.requests, a.steps[:s] + (StepAdvice(rows),) + a.steps[s + 1:], a.release)

    with pytest.raises(MalformedAdvice):
        put(KINDS[1 - KINDS.index(row.kind)], row.start)
    with pytest.raises(MalformedAdvice):
        put(row.kind, row.start + 1)
    x, y = a.requests[:2] if len(a.requests) > 1 else (None, None)
    if x and (x.prompt_len, x.output_len) != (y.prompt_len, y.output_len):
        with pytest.raises(MalformedAdvice):
            Advice((y, x) + a.requests[2:], a.steps, a.release)


# --- decode refuses every non-canonical byte string ----------------------------------------------------------------


@pytest.mark.parametrize("data, why", [
    (b"", "empty"),
    (bytes.fromhex("01" + FIXTURE_HEX[2:]), "version 1"),
    (bytes.fromhex(FIXTURE_HEX + "00"), "a byte past the advice"),
    (bytes.fromhex(FIXTURE_HEX[:-2] + "29"), "nonzero padding"),
    (bytes.fromhex(FIXTURE_HEX[:-4]), "truncated"),
    (packed(rebuild(FIXTURE, w_len=4)), "a token width wider than its largest value"),
    (packed(rebuild(FIXTURE, n_requests_width=3)), "a length prefix wider than its value"),
])
def test_decode_refuses(data, why):
    with pytest.raises(MalformedAdvice):
        decode(data)


def test_decode_refuses_an_id_that_isnt_utf8():
    s = rebuild(FIXTURE)
    at = 8 + 8 + 9 + 18 + 7  # the header, then request a's id length: its one byte
    with pytest.raises(MalformedAdvice):
        decode(packed(s[:at] + u(0xFF, 8) + s[at + 8:]))


def test_every_bit_flip_is_refused_or_canonical():
    """Fail closed: each of the fixture's single-bit flips is refused, or decodes to an advice whose encoding is exactly the
    flipped bytes. Nothing else escapes decode (no IndexError, no repaired advice)."""
    data = bytes.fromhex(FIXTURE_HEX)
    accepted = 0
    for bit in range(8 * len(data)):
        flipped = bytearray(data)
        flipped[bit // 8] ^= 0x80 >> (bit % 8)
        try:
            a = decode(bytes(flipped))
        except MalformedAdvice:
            continue
        accepted += 1
        assert encode(a) == bytes(flipped)
    assert accepted < 8 * len(data) // 4


@pytest.mark.parametrize("seed", range(10))
def test_random_corruption_is_refused_or_canonical(seed):
    rng = random.Random(seed)
    data = bytearray(encode(RANDOM[seed]))
    for _ in range(50):
        bad = bytearray(data)
        for _ in range(rng.randint(1, 3)):
            bad[rng.randrange(len(bad))] = rng.randrange(256)
        bad = bad[:rng.randint(len(bad) - 2, len(bad))] if rng.random() < 0.3 else bad
        try:
            a = decode(bytes(bad))
        except MalformedAdvice:
            continue
        assert encode(a) == bytes(bad)


def test_decode_takes_bytes_only():
    with pytest.raises(MalformedAdvice):
        decode(FIXTURE_HEX)
    with pytest.raises(MalformedAdvice):
        encode(to_json(FIXTURE))


# --- the JSON form -------------------------------------------------------------------------------------------------


def test_json_form_is_strict():
    doc = to_json(FIXTURE)
    assert doc["schema"] == SCHEMA
    for bad in ({**doc, "extra": 1}, {**doc, "schema": "verity/advice/v1"}, {k: v for k, v in doc.items() if k != "release"},
                {**doc, "requests": [{**doc["requests"][0], "user": "a"}, doc["requests"][1]]},
                {**doc, "steps": [{"rows": [{**doc["steps"][0]["rows"][0], "sources": None}]}] + doc["steps"][1:]},
                {**doc, "release": "11224"}):
        with pytest.raises(MalformedAdvice):
            from_json(bad)
