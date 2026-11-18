"""charge_rows: `ChargeLink_v1`'s carry rows committed window to window and the verifier's chain check over their
openings (`network.verifier.charge_check`): code_v1's charges over whole streams, and a restarted carry refused wherever
the prover puts it."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity.commitments import hm96
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
from verity.protocols.interface.charge import arith_encode as AE, charge_chain as CC
from verity.protocols.proofs.spot.verifier.one_stage import produced
from verity.protocols.resource_accounting.network.verifier import charge_check as CK
from verity.protocols.resource_accounting.network.prover import charge_rows as CR

FIXTURE = Path(__file__).parents[1] / "verifier" / "fixtures" / "arith_encode_vectors.json"
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


def _chain(book, rows_out, start=None):
    """The session's committed values, window w's carry-out row `rows_out[w]`; `start`, window 0's (row, salt) in
    place of the priors under their public salt."""
    values = []
    for w, out in enumerate(rows_out):
        if w == 0 and start is not None:
            values.append(produced.commit(PROGRAM, CC.VALUE, [CC.row(start[0]), CC.row(out)], REGISTRATION,
                                          [start[1], hm96.SHA512.fresh_salt()], CC.WORD_BITS, SCHEMA))
        else:
            values.append(CR.commit_window(PROGRAM, book, CC.carry_of(out), REGISTRATION, values[-1] if values else None))
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
        assert CK.check_chain([CR.openings(v) for v in values], PROGRAM, book) is None
        cb, prior = AE.charge_book(book, N)
        cases = [(cb, CC.unrow(v.rows[0]), CC.unrow(v.rows[1]), en, bits) for v, (en, bits) in zip(values, wins)]
        assert [c[1] for c in cases] == [s[1] for s in ses]
        assert CC.evaluate_links(N, cases) == [(1, length) for length in lens], source
        windows += len(wins)
        carried += sum(s[1] != CC.encode(prior) for s in ses)
        restarted += sum(AE.reference_charge(N, cb, prior, en, bits)[1] < length
                         for (en, bits), length in zip(wins, lens))
    assert (windows, carried, restarted) == (41, 35, 6)


def test_a_reset_carry_is_refused():
    """A prover restarting the counts before window 1 (to the priors, 20 bits in place of 23, or to counts fitted to
    the window, 16): window 1's proof block alone accepts any carry-in, so the chain refuses it, whether window 1's
    carry-in is committed under a fresh salt, under the public start salt, or window 0's carry-out claims the reset."""
    N, book, wins, _lens = _two_windows()
    (_l0, in0, out0), (_l1, _in1, out1) = CC.session(N, book, wins)
    cb, prior = AE.charge_book(book, N)
    honest = _chain(book, [out0, out1])
    assert CK.check_chain([CR.openings(v) for v in honest], PROGRAM, book) is None
    for reset, cheap in ((prior, 20), (AE.Carry(((0, 18000), (0, 0), (0, 0)), 0), 16)):
        ok, length, nxt = AE.reference_charge(N, cb, reset, *wins[1])
        assert (ok, length) == (1, cheap)
        assert CC.evaluate_links(N, [(cb, CC.encode(reset), CC.encode(nxt), *wins[1])]) == [(1, cheap)]
        for salt in (hm96.SHA512.fresh_salt(), CR.start_salt(PROGRAM)):
            forged = produced.commit(PROGRAM, CC.VALUE, [CC.row(CC.encode(reset)), CC.row(CC.encode(nxt))],
                                     REGISTRATION, [salt, hm96.SHA512.fresh_salt()], CC.WORD_BITS, SCHEMA)
            got = CK.check_chain([CR.openings(honest[0]), CR.openings(forged)], PROGRAM, book)
            assert got == "CH-carry: window 1's carry-in isn't window 0's carry-out"
        assert CC.evaluate_links(N, [(cb, in0, CC.encode(reset), *wins[0])])[0][0] == 0
    same_rows = produced.commit(PROGRAM, CC.VALUE, [honest[0].rows[1], CC.row(out1)], REGISTRATION,
                                [hm96.SHA512.fresh_salt(), hm96.SHA512.fresh_salt()], CC.WORD_BITS, SCHEMA)
    assert CK.check_chain([CR.openings(honest[0]), CR.openings(same_rows)], PROGRAM, book).startswith("CH-carry")


def test_check_chain_refusals():
    """A window that isn't two rows, a session that doesn't start at the book's priors under their public salt (a
    fresh salt, another program's, another book's priors, a carry other than the priors), and an empty session."""
    N, book, wins, _lens = _two_windows()
    ses = CC.session(N, book, wins)
    values = _chain(book, [s[2] for s in ses])
    ops = [CR.openings(v) for v in values]
    assert CK.check_chain(ops, PROGRAM, book) is None and CK.check_chain(ops[:1], PROGRAM, book) is None
    assert CK.check_chain([], PROGRAM, book) is None
    assert CK.check_chain([ops[0], ops[1] + ops[1][:1]], PROGRAM, book).startswith("CH-rows: window 1 has 3")
    assert CK.check_chain([ops[0][:1]], PROGRAM, book).startswith("CH-rows: window 0 has 1")
    assert CK.check_chain(ops, "another-program", book).startswith("CH-first")
    other = {**book, "priors": {**book["priors"], "arrived/0": [1, 1]}}
    assert CK.check_chain(ops, PROGRAM, other).startswith("CH-first")
    prior = CC.encode(AE.first_carry(book))
    for start in ((prior, hm96.SHA512.fresh_salt()), (CC.encode(AE.Carry()), CR.start_salt(PROGRAM)),
                  (prior, CR.start_salt("another-program"))):
        assert CK.check_chain([CR.openings(v) for v in _chain(book, [s[2] for s in ses], start)], PROGRAM,
                              book).startswith("CH-first")


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
    assert v0.salts[0] == CR.start_salt(PROGRAM) and v0.bcs[0] == CK.start_opening(PROGRAM, book)
    assert v0.rows[0] == CC.row(CC.encode(AE.first_carry(book)))
    assert (v1.rows[0], v1.salts[0], v1.bcs[0]) == (v0.rows[1], v0.salts[1], v0.bcs[1])
    assert v0.salts[1] != v1.salts[1] and CC.reads() == {CC.VALUE: {"carry_in": 0, "carry_out": 1}}

