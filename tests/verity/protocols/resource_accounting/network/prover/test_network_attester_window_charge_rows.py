"""charge_rows: `ChargeLink_v2`'s carry rows committed window to window under `window_charge.VALUE`, and the verifier's
chain check over their openings (`network.verifier.charge_check.window_check_chain`)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity.commitments import hm96
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
from verity.protocols.interface.charge import window_charge as WC
from verity.protocols.proofs.spot.prover import produced as PP
from verity.protocols.resource_accounting.network.verifier import charge_check as CK
from verity.protocols.resource_accounting.network.prover import charge_rows as CR

FIXTURE = Path(__file__).parents[1] / "verifier" / "fixtures" / "arith_encode_vectors.json"
PROGRAM, REGISTRATION = "window-charge-test", "ab" * 64


def _ints(s):
    return [int(c) for c in s]


def _streams():
    return [s for s in json.loads(FIXTURE.read_text())["streams"] if "window" in s]


def _cut(stream):
    win = stream["window"]
    slots = [(_ints(en), _ints(bits), _ints(hen), short, _ints(oen), p, r, _ints(qen), mt, olen)
             for en, bits, hen, short, oen, p, r, qen, mt, olen, _l in win["cut"]]
    return win["N"], win["H"], win["O"], win["L"], stream["decisions"]["book"], slots, [w[-1] for w in win["cut"]]


@pytest.mark.slow
def test_the_chain_on_the_cut_is_code_v1s():
    """Each fixture stream's cut windows chained under hiding commitments: the verifier's check passes, and
    `ChargeLink_v2{4, 2, 2, 2}` on each window's committed rows is ok at `code_v1`'s charge. A carry-in other than the last
    carry-out is refused by the chain (CH-carry), and the session must start at the book's first carry (CH-first)."""
    for s in _streams():
        N, H, O, L, book, slots, lens = _cut(s)
        ses = WC.session(N, H, O, L, book, slots)
        values = []
        for _length, _c_in, c_out in ses:
            values.append(CR.window_commit(PROGRAM, book, WC.carry_of(c_out), REGISTRATION,
                                           values[-1] if values else None))
        ops = [list(v.bcs) for v in values]
        assert CK.window_check_chain(ops, PROGRAM, book) is None
        assert CK.window_check_chain(ops, "another-program", book).startswith("CH-first")
        if len(ops) > 2:
            assert CK.window_check_chain([ops[0], ops[2]], PROGRAM, book).startswith("CH-carry")
        wb = WC.window_book(book, N, H, O)[0]
        cases = [(wb, WC.CC.unrow(v.rows[0]), WC.CC.unrow(v.rows[1]), *sl) for v, sl in zip(values, slots)]
        assert WC.evaluate_links(N, H, O, L, cases) == [(1, length) for length in lens], s["source"]
    v = values[0]
    assert v.salts[0] == CR.start_salt(PROGRAM, WC.VALUE) and v.bcs[0] == CK.window_start_opening(PROGRAM, book)


def test_rows():
    """The 160-bit row: `ChargeCount_v1`'s 92 bits, then pos and rank, then zeros; its commitment under this module's
    own value, the session's first row the book's first carry under the value's public salt."""
    carry = WC.WindowCarry(((1, (1 << 15) - 1), (7, 0), (0, 3)), 2, (1 << 32) - 1, (1 << 24) + 1)
    bits = WC.encode(carry)
    assert len(bits) == WC.ROW_BITS == 160 and not any(bits[149:]) and WC.carry_of(bits) == carry
    assert bits[92:124] == (1,) * 32 and bits[124:149] == tuple(((1 << 24) + 1 >> j) & 1 for j in range(25))
    assert WC.carry_of(bits[:149] + (1,) * 11) == carry
    for bad in (WC.WindowCarry(pos=1 << 32), WC.WindowCarry(rank=1 << 25), WC.WindowCarry(((0, 0),) * 2)):
        with pytest.raises(ValueError):
            WC.encode(bad)
    book = _streams()[0]["decisions"]["book"]
    v0 = CR.window_commit(PROGRAM, book, carry, REGISTRATION)
    v1 = CR.window_commit(PROGRAM, book, carry, REGISTRATION, v0)
    assert v0.rows[0] == WC.CC.row(WC.encode(WC.first_carry(book))) and v0.bcs[0] == CK.window_start_opening(PROGRAM, book)
    assert v0.salts[0] != CR.start_salt(PROGRAM) and v0.bcs[0] != CK.start_opening(PROGRAM, book)
    assert (v1.rows[0], v1.salts[0], v1.bcs[0]) == (v0.rows[1], v0.salts[1], v0.bcs[1])
    assert WC.reads() == {WC.VALUE: {"carry_in": 0, "carry_out": 1}}
    forged = PP.commit(PROGRAM, WC.VALUE, [WC.CC.row(WC.encode(WC.WindowCarry())), v0.rows[1]], REGISTRATION,
                             [hm96.SHA512.fresh_salt(), hm96.SHA512.fresh_salt()], WC.CC.WORD_BITS, SCHEMA)
    assert CK.window_check_chain([list(forged.bcs)], PROGRAM, book).startswith("CH-first")

