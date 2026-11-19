"""charge_chain: `ChargeCount_v1`'s carry chained across windows under hiding commitments. Held to `code_v1`'s charges
over whole streams (the arith fixture's decisions cut into the compile slice's P·A = 14 slots, each window's charge
`code_v1`'s with the carry from every window before it), refusing a restarted carry wherever the prover puts it, and
the circuit to its reference."""
from __future__ import annotations

import json
import random
from pathlib import Path


from verity.protocols.interface.charge import arith_encode as AE, charge_chain as CC
from verity.commitments.gates.bits import ands

FIXTURE = Path(__file__).parent / "fixtures" / "arith_encode_vectors.json"
PROGRAM, REGISTRATION = "charge-chain-test", "ab" * 64


def _ints(s):
    return [int(c) for c in s]


def _streams():
    """Each fixture stream with decisions: (source, N, book JSON, [(en, bits)], code_v1's charges)."""
    for s in json.loads(FIXTURE.read_text())["streams"]:
        if "decisions" in s:
            dec = s["decisions"]
            wins = [(_ints(en), _ints(bits)) for en, bits, _l in dec["cut"]["windows"]]
            yield s["source"], dec["cut"]["N"], dec["book"], wins, [length for *_x, length in dec["cut"]["windows"]]


def _two_windows():
    source, N, book, wins, lens = next(s for s in _streams() if s[0].startswith("async-kv-pressure"))
    assert len(wins) == 2
    return N, book, wins, lens


def test_a_two_window_charge_is_code_v1s():
    """async-kv-pressure's two windows: the second's charge is `code_v1`'s 23 bits only from the first's carry-out;
    from the priors it would be 20."""
    N, book, wins, lens = _two_windows()
    (l0, in0, out0), (l1, in1, out1) = CC.session(N, book, wins)
    cb, prior = AE.charge_book(book, N)
    assert (l0, l1) == tuple(lens) == (23, 23) and in0 == CC.encode(prior) and in1 == out0 != in0
    assert AE.reference_charge(N, cb, prior, *wins[1])[:2] == (1, 20)
    assert CC.evaluate_links(N, [(cb, in0, out0, *wins[0]), (cb, in1, out1, *wins[1])]) == [(1, 23), (1, 23)]


def test_a_window_charged_over_its_own_decisions_is_code_v1s():
    """The load proof block's form: each window's decisions in the first slots of `ChargeLink_v1{32}` (the engine
    class's D), en_i = i < count, are `code_v1`'s charge and carry-out from the same carry-in as its 14 asks, since a
    slot with en 0 changes nothing."""
    D, windows = 32, 0
    for source, N, book, wins, _lens in _streams():
        cb, _prior = AE.charge_book(book, D)
        for (en, bits), (length, c_in, c_out) in zip(wins, CC.session(N, book, wins)):
            dec = [b for e, b in zip(en, bits) if e]
            packed = [1] * len(dec) + [0] * (D - len(dec)), dec + [1] * (D - len(dec))
            assert AE.reference_charge(D, cb, CC.carry_of(c_in), *packed) == (1, length, CC.carry_of(c_out)), source
            assert CC.reference_link(D, cb, c_in, c_out, *packed) == (1, length), source
            windows += 1
    assert windows == 41


def _random_link(rng, N):
    """A random `ChargeLink_v1{N}` case: a book with skewed bins; a carry-in at the halving edge, now and then run 3
    or nonzero padding; the carry-out its charge's carry' or one bit off it, the padding's included."""
    nb = N.bit_length() + 1
    arrived = [rng.choice([1, rng.randrange(1, 64), rng.randrange(1, 1 << 12)]) for _ in range(nb)]
    total = min(sum(arrived) + rng.choice([0, 0, rng.randrange(1 << 12)]), AE.MAX_TOTAL)
    zeros = []
    for _ in AE.ZERO_COUNTS:
        t = rng.choice([2, 1000, AE.MAX_TOTAL, rng.randrange(1, AE.MAX_TOTAL + 1)])
        zeros.append((rng.choice([1, t, rng.randrange(1, t + 1)]), t))
    book = AE.ChargeBook(tuple(arrived), total, tuple(zeros))
    counts = []
    for _ in range(AE.RUNS):
        s = rng.choice([(1 << 15) - 1, (1 << 15) - 2, rng.randrange(1 << 15), rng.randrange(40)])
        a = rng.randrange(s + 1)
        counts.append((a, s - a))
    carry = AE.Carry(tuple(counts), rng.choice([0, 1, 2, 3]) if rng.random() < 0.1 else rng.randrange(3))
    row_in = list(CC.encode(carry))
    if rng.random() < 0.1:
        row_in[rng.randrange(92, CC.ROW_BITS)] = 1
    p = rng.choice([0.0, 0.08, 0.3, 0.8, 1.0])
    en, bits = [int(rng.random() < p) for _ in range(N)], [int(rng.random() < 0.3) for _ in range(N)]
    ok, _length, nxt = AE.reference_charge(N, book, carry, en, bits)
    row_out = list(CC.encode(nxt) if ok else CC.encode(carry))
    if rng.random() < 0.3:
        row_out[rng.randrange(CC.ROW_BITS)] ^= 1
    return book, row_in, row_out, en, bits


def _same_links(N, cases):
    got = CC.evaluate_links(N, cases)
    for c, g in zip(cases, got):
        want = CC.reference_link(N, *c)
        assert (g[0] == 0) if want[0] == 0 else g == want, (c, g, want)
    return got


def test_the_circuit_is_the_reference():
    """`ChargeLink_v1{14}` and `{2}` against `reference_link` on random cases: accepted ones, carry-outs one bit off
    (the padding's too), refused charges, and carry-ins with nonzero padding, which it ignores."""
    rng = random.Random(96)
    for N, n in ((14, 300), (2, 200)):
        cases = [_random_link(rng, N) for _ in range(n)]
        want = [CC.reference_link(N, *c) for c in cases]
        charged = [AE.reference_charge(N, c[0], CC.carry_of(c[1]), c[3], c[4])[0] for c in cases]
        assert sum(w[0] for w in want) > n // 3 and sum(not w[0] for w in want) > n // 5
        assert sum(ch and not w[0] for w, ch in zip(want, charged)) > n // 10
        assert any(w[0] and any(c[1][92:]) for w, c in zip(want, cases))
        _same_links(N, cases)


def test_build_link_inline():
    """`build_link`, the call a caller's circuit makes, is the proof block: its book split by `book_fields`."""
    from verity.circuits.boolean.forms import Circuit
    N, rng = 3, random.Random(3)
    cases = [_random_link(rng, N) for _ in range(40)]
    C = Circuit(cse=True)
    fields = CC.book_fields(N)
    pub = C.input("pub", sum(n * w for _f, n, w in fields))
    row_in, row_out = C.input("in", CC.ROW_BITS), C.input("out", CC.ROW_BITS)
    en, bits = C.input("en", N), C.input("bits", N)
    elems, at = [], 0
    for _f, n, w in fields:
        elems += [pub[at + k * w:at + (k + 1) * w] for k in range(n)]
        at += n * w
    ok, count = CC.build_link(C, N, elems, row_in, row_out, en, bits)
    C.output("res", [ok] + count)

    def sliced(rows):
        return [sum(r[i] << t for t, r in enumerate(rows)) for i in range(len(rows[0]))]

    pubs = [[b for f in AE._args(fields, CC.book_values(c[0])) for b in f] for c in cases]
    outs, _fail = C.evaluate({"pub": sliced(pubs), "in": sliced([c[1] for c in cases]),
                              "out": sliced([c[2] for c in cases]), "en": sliced([c[3] for c in cases]),
                              "bits": sliced([c[4] for c in cases])}, len(cases))
    for t, c in enumerate(cases):
        res = [(v >> t) & 1 for v in outs["res"]]
        g, want = (res[0], sum(b << j for j, b in enumerate(res[1:]))), CC.reference_link(N, *c)
        assert (g[0] == 0) if want[0] == 0 else g == want, (c, g, want)


def test_size_counts_the_gates():
    for N in (2, 14):
        size = CC.size(N)
        assert size["ands"] == AE.size("charge", N=N)["ands"] + 96 and size["link"] == 96
    assert CC.size(2)["ands"] == ands(CC.link_unit(2)) == 19632
    assert CC.size(14)["ands"] == 50760
