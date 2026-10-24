"""charge_chain: `ChargeCount_v1`'s carry chained across windows under hiding commitments. Held to `code_v1`'s charges
over whole streams (the arith fixture's decisions cut into the compile slice's P·A = 14 slots, each window's charge
`code_v1`'s with the carry from every window before it), refusing a restarted carry wherever the prover puts it, and
the circuit to its reference."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from verity.core.protocols.certifier import arith_encode as AE, charge_chain as CC
from verity.core.protocols.certifier.head_check import ands
from verity.core.protocols.sampled_proofs.one_stage import produced
from verity.ml.commitments import hm96

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


def _chain(book, rows_out, start=None):
    """The session's committed values, window w's carry-out row `rows_out[w]`; `start`, window 0's (row, salt) in
    place of the priors under their public salt."""
    values = []
    for w, out in enumerate(rows_out):
        if w == 0 and start is not None:
            values.append(produced.commit(PROGRAM, CC.VALUE, [CC.row(start[0]), CC.row(out)], REGISTRATION,
                                          [start[1], hm96.SHA512.fresh_salt()], CC.WORD_BITS, CC.SCHEMA))
        else:
            values.append(CC.commit_window(PROGRAM, book, CC.carry_of(out), REGISTRATION, values[-1] if values else None))
    return values


def test_the_chain_is_code_v1s_over_the_whole_stream():
    """Each stream's windows chained: the carry each window reads is committed by the last, the verifier's check
    passes, and every window's `ChargeLink_v1{14}` on its committed rows is ok at `code_v1`'s charge over the whole
    stream, which the priors alone (a restart every window) would undercharge on 6 of the 41 windows."""
    windows = carried = restarted = 0
    for source, N, book, wins, lens in _streams():
        ses = CC.session(N, book, wins)
        assert [s[0] for s in ses] == lens, source
        values = _chain(book, [s[2] for s in ses])
        assert CC.check_chain([CC.openings(v) for v in values], PROGRAM, book) is None
        cb, prior = AE.charge_book(book, N)
        cases = [(cb, CC.unrow(v.rows[0]), CC.unrow(v.rows[1]), en, bits) for v, (en, bits) in zip(values, wins)]
        assert [c[1] for c in cases] == [s[1] for s in ses]
        assert CC.evaluate_links(N, cases) == [(1, length) for length in lens], source
        windows += len(wins)
        carried += sum(s[1] != CC.encode(prior) for s in ses)
        restarted += sum(AE.reference_charge(N, cb, prior, en, bits)[1] < length
                         for (en, bits), length in zip(wins, lens))
    assert (windows, carried, restarted) == (41, 35, 6)


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
    """The load unit's form: each window's decisions in the first slots of `ChargeLink_v1{32}` (the engine class's D),
    en_i = i < count, are `code_v1`'s charge and carry-out from the same carry-in as its 14 asks, since a slot with en 0
    changes nothing."""
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


def test_a_reset_carry_is_refused():
    """A prover restarting the counts before window 1 (to the priors, 20 bits in place of 23, or to counts fitted to
    the window, 16): window 1's unit alone accepts any carry-in, so the chain refuses it, whether window 1's carry-in
    is committed under a fresh salt, under the public start salt, or window 0's carry-out claims the reset."""
    N, book, wins, _lens = _two_windows()
    (_l0, in0, out0), (_l1, _in1, out1) = CC.session(N, book, wins)
    cb, prior = AE.charge_book(book, N)
    honest = _chain(book, [out0, out1])
    assert CC.check_chain([CC.openings(v) for v in honest], PROGRAM, book) is None
    for reset, cheap in ((prior, 20), (AE.Carry(((0, 18000), (0, 0), (0, 0)), 0), 16)):
        ok, length, nxt = AE.reference_charge(N, cb, reset, *wins[1])
        assert (ok, length) == (1, cheap)
        assert CC.evaluate_links(N, [(cb, CC.encode(reset), CC.encode(nxt), *wins[1])]) == [(1, cheap)]
        for salt in (hm96.SHA512.fresh_salt(), CC.start_salt(PROGRAM)):
            forged = produced.commit(PROGRAM, CC.VALUE, [CC.row(CC.encode(reset)), CC.row(CC.encode(nxt))],
                                     REGISTRATION, [salt, hm96.SHA512.fresh_salt()], CC.WORD_BITS, CC.SCHEMA)
            got = CC.check_chain([CC.openings(honest[0]), CC.openings(forged)], PROGRAM, book)
            assert got == "CH-carry: window 1's carry-in isn't window 0's carry-out"
        assert CC.evaluate_links(N, [(cb, in0, CC.encode(reset), *wins[0])])[0][0] == 0
    same_rows = produced.commit(PROGRAM, CC.VALUE, [honest[0].rows[1], CC.row(out1)], REGISTRATION,
                                [hm96.SHA512.fresh_salt(), hm96.SHA512.fresh_salt()], CC.WORD_BITS, CC.SCHEMA)
    assert CC.check_chain([CC.openings(honest[0]), CC.openings(same_rows)], PROGRAM, book).startswith("CH-carry")


def test_check_chain_refusals():
    """A window that isn't two rows, a session that doesn't start at the book's priors under their public salt (a
    fresh salt, another program's, another book's priors, a carry other than the priors), and an empty session."""
    N, book, wins, _lens = _two_windows()
    ses = CC.session(N, book, wins)
    values = _chain(book, [s[2] for s in ses])
    ops = [CC.openings(v) for v in values]
    assert CC.check_chain(ops, PROGRAM, book) is None and CC.check_chain(ops[:1], PROGRAM, book) is None
    assert CC.check_chain([], PROGRAM, book) is None
    assert CC.check_chain([ops[0], ops[1] + ops[1][:1]], PROGRAM, book).startswith("CH-rows: window 1 has 3")
    assert CC.check_chain([ops[0][:1]], PROGRAM, book).startswith("CH-rows: window 0 has 1")
    assert CC.check_chain(ops, "another-program", book).startswith("CH-first")
    other = {**book, "priors": {**book["priors"], "arrived/0": [1, 1]}}
    assert CC.check_chain(ops, PROGRAM, other).startswith("CH-first")
    prior = CC.encode(AE.first_carry(book))
    for start in ((prior, hm96.SHA512.fresh_salt()), (CC.encode(AE.Carry()), CC.start_salt(PROGRAM)),
                  (prior, CC.start_salt("another-program"))):
        assert CC.check_chain([CC.openings(v) for v in _chain(book, [s[2] for s in ses], start)], PROGRAM,
                              book).startswith("CH-first")


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
    """`build_link`, the call a caller's circuit makes, is the unit: its book split by `book_fields`."""
    from verity.ml.circuits.boolean.forms import Circuit
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


def test_rows():
    """The row's layout, its words, and its commitment: window 0's row 0 is the priors' under the public salt, whose
    opening the verifier computes; a carry-out is under a fresh salt that the next window's carry-in reuses."""
    carry = AE.Carry(((1, (1 << 15) - 1), (7, 0), (0, 3)), 2)
    bits = CC.encode(carry)
    assert len(bits) == CC.ROW_BITS == 96 and not any(bits[92:]) and CC.carry_of(bits) == carry
    assert bits[:15] == tuple((1 >> j) & 1 for j in range(15)) and bits[90:92] == (0, 1)
    assert CC.unrow(CC.row(bits)) == bits and len(CC.row(bits)) == 6
    assert CC.carry_of(bits[:92] + (1, 0, 1, 1)) == carry
    assert CC.book_fields(14) + AE._CARRY == AE.charge_fields(14)
    for bad in (AE.Carry(((1 << 15, 0), (0, 0), (0, 0)), 0), AE.Carry(((0, 0),) * 3, 4), AE.Carry(((0, 0),) * 2, 0)):
        with pytest.raises(ValueError):
            CC.encode(bad)
    with pytest.raises(ValueError):
        CC.carry_of(bits[:95])
    N, book, wins, _lens = _two_windows()
    ses = CC.session(N, book, wins)
    v0, v1 = _chain(book, [s[2] for s in ses])
    assert v0.salts[0] == CC.start_salt(PROGRAM) and v0.bcs[0] == CC.start_opening(PROGRAM, book)
    assert v0.rows[0] == CC.row(CC.encode(AE.first_carry(book)))
    assert (v1.rows[0], v1.salts[0], v1.bcs[0]) == (v0.rows[1], v0.salts[1], v0.bcs[1])
    assert v0.salts[1] != v1.salts[1] and CC.reads() == {CC.VALUE: {"carry_in": 0, "carry_out": 1}}


def test_size_counts_the_gates():
    for N in (2, 14):
        size = CC.size(N)
        assert size["ands"] == AE.size("charge", N=N)["ands"] + 96 and size["link"] == 96
    assert CC.size(2)["ands"] == ands(CC.link_unit(2)) == 19632
    assert CC.size(14)["ands"] == 50760
