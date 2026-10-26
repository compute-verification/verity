"""Advice v0's validity circuit (`advice_check`, `AdviceCheck_v1{L}`) against `advice.decode`: the circuit computes
`reference`, `reference` takes what decode takes, and where they differ decode refuses with one of `DEFERRED`'s messages.

The strings: the advice tests' fixture and simulated vLLM schedules, every bit flip of the fixture, random byte
corruptions, unchecked mutations of the schedules (a row moved, its kind or resume flipped, a step dropped or repeated, a
release or a request's lengths changed, a source retargeted) encoded field by field, non-minimal widths, trailing bytes,
nonzero padding and random strings."""
from __future__ import annotations

import itertools
import random

import pytest

from test_network_certifier_advice import FIXTURE, RANDOM, packed, rebuild

from verity.core.protocols.certifier import advice as V
from verity.ml.protocols.certifier.advice_check import (DEFERRED, _sort_net, ands, arguments, comparators, decodes,
                                                          evaluate, evaluate_bits, reference, size, unit)

P, D = "prefill", "decode"


def _raw(cls, *values):
    """An instance of a frozen advice class that skips its checks."""
    obj = object.__new__(cls)
    for name, v in zip(cls.__dataclass_fields__, values):
        object.__setattr__(obj, name, v)
    return obj


def _row(r, kind, start, n, hit=0, pre=False, sources=()):
    return _raw(V.Row, r, kind, start, n, hit, pre, tuple(_raw(V.Source, q, m) for q, m in sources))


def _unchecked(requests, steps, release) -> bytes:
    """The documented layout of an advice that may break any rule."""
    return packed(rebuild(_raw(V.Advice, tuple(requests), tuple(_raw(V.StepAdvice, tuple(s)) for s in steps),
                               tuple(release))))


def _mutant(a: V.Advice, rng: random.Random) -> bytes:
    reqs, steps, rel = list(a.requests), [list(st.rows) for st in a.steps], list(a.release)
    s = rng.randrange(len(steps))
    j = rng.randrange(len(steps[s]))
    row = steps[s][j]
    src = [(x.request, x.n) for x in row.sources]
    k = rng.randrange(10)
    if k == 0:
        start = row.start + rng.choice((-1, 1)) if row.start else 1
        steps[s][j] = _row(row.request, row.kind, start, row.n, row.cache_hit, row.preempted, src)
    elif k == 1:
        steps[s][j] = _row(row.request, V.KINDS[1 - V.KINDS.index(row.kind)], row.start, row.n, row.cache_hit,
                           row.preempted, src)
    elif k == 2:
        steps[s][j] = _row(row.request, row.kind, row.start, max(1, row.n + rng.choice((-1, 1))), row.cache_hit,
                           row.preempted, src)
    elif k == 3:
        steps[s][j] = _row(row.request, row.kind, row.start, row.n, row.cache_hit, not row.preempted, src)
    elif k == 4:
        steps[s][j] = _row(rng.randrange(len(reqs)), row.kind, row.start, row.n, row.cache_hit, row.preempted, src)
    elif k == 5 and len(steps) > 1:
        t = rng.randrange(len(steps) - 1)
        steps[t], steps[t + 1] = steps[t + 1], steps[t]
    elif k == 6:
        if len(steps[s]) > 1:
            del steps[s][j]
        else:
            del steps[s]
    elif k == 7:
        steps.insert(s, list(steps[s]))
    elif k == 8:
        i = rng.randrange(len(rel))
        rel[i] = max(0, rel[i] + rng.choice((-1, 1, 2)))
        i = rng.randrange(len(reqs))
        q = reqs[i]
        reqs[i] = _raw(V.RequestAdvice, q.id, max(1, q.prompt_len + rng.choice((-1, 1))),
                       max(1, q.output_len + rng.choice((-1, 0, 1))), q.finish)
    else:
        if src:
            t = rng.randrange(len(src))
            src[t] = (rng.randrange(len(reqs)), src[t][1]) if rng.random() < 0.5 else (src[t][0], src[t][1] + 1)
        else:
            src = [(rng.randrange(len(reqs)), 1)]
        hit = sum(m for _, m in src)
        steps[s][j] = _row(row.request, row.kind, hit, row.n, hit, row.preempted, src)
    return _unchecked(reqs, steps, rel)


def _strings(advices, seed: int, cap: int) -> list[bytes]:
    rng = random.Random(seed)
    out = [V.encode(V.Advice((), (), ()))]
    for a in advices:
        e = V.encode(a)
        out += [e, e + b"\0", e[:-1] + bytes([e[-1] | 1]), packed(rebuild(a, w_len=V._widths(a).len + 1)),
                packed(rebuild(a, n_requests_width=len(a.requests).bit_length() + 1))]
        for _ in range(4):
            bit = rng.randrange(8 * len(e))
            out.append(e[:bit // 8] + bytes([e[bit // 8] ^ 0x80 >> bit % 8]) + e[bit // 8 + 1:])
            bad = bytearray(e)
            for _ in range(rng.randint(1, 3)):
                bad[rng.randrange(len(bad))] = rng.randrange(256)
            out.append(bytes(bad[:rng.randint(len(bad) - 2, len(bad))]))
            out.append(_mutant(a, rng))
    out += [bytes(rng.randrange(256) for _ in range(rng.randint(0, cap // 8))) for _ in range(100)]
    out += [b"\0" + bytes(rng.randrange(256) for _ in range(rng.randint(0, 8))) for _ in range(50)]
    return [s for s in out if 8 * len(s) <= cap]


FX = V.encode(FIXTURE)
FLIPS = [FX[:b // 8] + bytes([FX[b // 8] ^ 0x80 >> b % 8]) + FX[b // 8 + 1:] for b in range(8 * len(FX))]


def _deferred(why: str) -> bool:
    return any(m in why for m in DEFERRED.values())


@pytest.mark.parametrize("seed", range(3))
def test_reference_is_decode_less_the_deferred_refusals(seed):
    """decode takes a string ⇒ `reference` does; `reference` refuses ⇒ decode does; and where `reference` takes what
    decode refuses, decode's reason is one `DEFERRED` names."""
    datas = _strings(RANDOM[seed::3], seed, 1520) + FLIPS
    seen = {True: 0, False: 0}
    for d in datas:
        ok, why = decodes(d)
        ref = reference(d, 1520)
        assert ref or not ok, (d.hex(), why)
        assert ok or not ref or _deferred(why), (d.hex(), why)
        seen[ok] += 1
    assert min(seen.values()) > 50, seen


#: one string for each deferred refusal, which decode refuses and the circuit takes
GAPS = {
    "unique ids": _unchecked((V.RequestAdvice("a", 1, 1, "eos"), V.RequestAdvice("a", 1, 1, "eos")),
                             [[_row(0, P, 0, 1), _row(1, P, 0, 1)]], (0, 0)),
    "a source's KV computed there": _unchecked(  # a has computed 0 … 2 when b's hit reads its 0 … 3
        (V.RequestAdvice("a", 5, 1, "length"), V.RequestAdvice("b", 6, 1, "length")),
        [[_row(0, P, 0, 3), _row(1, P, 4, 2, 4, False, ((0, 4),))], [_row(0, P, 3, 2)]], (0, 0)),
    "release buckets non-decreasing per request": _unchecked(FIXTURE.requests, [s.rows for s in FIXTURE.steps],
                                                             (1, 1, 2, 0, 4)),
}


def test_each_deferred_refusal_is_a_gap_the_circuit_leaves():
    assert set(GAPS) == set(DEFERRED)
    assert evaluate(240, list(GAPS.values())) == [1, 1, 1]
    for name, d in GAPS.items():
        ok, why = decodes(d)
        assert not ok and DEFERRED[name] in why and reference(d, 240), (name, why)


def test_the_circuit_is_the_reference_on_the_fixture():
    """At L = 240: the fixture, the empty advice, every bit flip and truncation of the fixture, the fixture with garbage
    past its length (bits past 8·length are no part of the string), and lengths past L / 8."""
    rng = random.Random(7)
    datas = [FX, V.encode(V.Advice((), (), ())), b"", *FLIPS, *(FX[:k] for k in range(len(FX))), FX + b"\0"]
    datas += [bytes(rng.randrange(256) for _ in range(rng.randint(0, 30))) for _ in range(60)]
    assert evaluate(240, datas) == [int(reference(d, 240)) for d in datas]
    bits, length = arguments(FX, 240)
    junk = bits[:8 * len(FX)] + [rng.randrange(2) for _ in range(240 - 8 * len(FX))]
    too_long = [(240 // 8 + 1) >> j & 1 for j in range(len(length))]
    out = evaluate_bits(unit(240), [bits, junk, bits], [length, length, too_long]).reshape(-1).tolist()
    assert [int(b) for b in out] == [1, 1, 0]
    with pytest.raises(ValueError):
        arguments(bytes(31), 240)


@pytest.mark.slow
def test_the_circuit_is_the_reference_on_schedules_and_corruptions():
    """At L = 1520 (the simulated schedules are at most 187 bytes): about 3000 strings, a third of them taken by decode,
    in one bit-sliced evaluation; the fixture's flips and corruptions find refusals the random strings can't reach."""
    datas = _strings(RANDOM, 11, 1520) + FLIPS
    got = evaluate(1520, datas)
    assert got == [int(reference(d, 1520)) for d in datas]
    assert sum(got) > 250


def test_the_sort_sorts():
    """`_sort_net(nb)` sorts every 0-1 input (so every input) for nb ≤ 12, and `comparators` counts it."""
    for nb in range(2, 13):
        net = _sort_net(nb)
        for bits in itertools.product((0, 1), repeat=nb):
            v = list(bits)
            for layer in net:
                for i, j in layer:
                    if v[j] < v[i]:
                        v[i], v[j] = v[j], v[i]
            assert v == sorted(v), (nb, bits)
    for nb in [*range(2, 200), 1000, 1023, 1024, 1025, 4097]:
        assert comparators(nb) == sum(map(len, _sort_net(nb))), nb


def test_size_counts_the_unit():
    """`size` is the unit's AND count, and at the 60 s llama8b-shape window (10.6 Mbit, PROTOCOL.md "Advice v0's
    validity") the count PROTOCOL.md cites."""
    for L in (40, 240):
        assert size(L)["ands"] == ands(unit(L)), L
    assert size(240)["ands"] == 470116
    assert size(10_600_000)["ands"] == 86_704_366_964
