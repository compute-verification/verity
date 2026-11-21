"""The charges' carry rows under hiding commitments: each window's produced value of its carry-in and carry-out rows,
chained from window to window (`protocol/charge-chain.md`), which `network.prover.charge_check` checks.
`commit_window` is `charge_chain.ChargeLink_v1`'s (`VALUE` "replay-charge"); `window_commit` is
`window_charge.ChargeLink_v2`'s."""
from __future__ import annotations

from collections.abc import Sequence

from verity.commitments import hm96
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
from verity.protocols.proof.spot.prover import produced as PP
from verity.commitments import registered_value as RV
from verity.protocols.interface.charge import window_charge as WC
from verity.protocols.interface.charge import arith_encode as AE, charge_chain as CC

_INST = hm96.SHA512


def start_salt(program: str, value: str = CC.VALUE) -> bytes:
    """The public salt of the session's first carry row (`registered_value.supplied_salts`' row 0 of `value`)."""
    return RV.supplied_salts(program, value, 1)[0]


def commit_rows(program: str, value: str, first: Sequence[int], out: Sequence[int], registration: str,
                carry: RV.Registered | None = None, salt: bytes | None = None) -> RV.Registered:
    """A window's produced `value` of two carry rows, given as bits: row 0 `first` under `start_salt` for the session's
    first window (`carry` None), else the previous window's row 1 under its salt; row 1 `out` under `salt` (default
    fresh, `hm96`'s `fresh_salt`)."""
    if carry is None:
        rows0, salt0 = CC.row(first), start_salt(program, value)
    else:
        rows0, salt0 = carry.rows[-1], carry.salts[-1]
    return PP.commit(program, value, [rows0, CC.row(out)], registration,
                           [salt0, salt if salt is not None else _INST.fresh_salt()], CC.WORD_BITS, SCHEMA)


def commit_window(program: str, book: dict, carry_out: AE.Carry, registration: str, carry: RV.Registered | None = None,
                  salt: bytes | None = None) -> RV.Registered:
    """A window's produced value: its carry-in row and `carry_out`'s. `carry`: the previous window's value, whose row 1
    is this window's row 0 under the same salt; for the session's first window (None), row 0 is the priors' row under
    `start_salt`. `salt`: row 1's (default fresh, `hm96`'s `fresh_salt`)."""
    return commit_rows(program, CC.VALUE, CC.encode(AE.first_carry(book)), CC.encode(carry_out), registration, carry,
                       salt)


def openings(value: RV.Registered) -> list[bytes]:
    """Each carry row's b‖c, as the produced record's `openings` give them."""
    return list(value.bcs)


def window_commit(program: str, book: dict, carry_out: WC.WindowCarry, registration: str,
                  carry: RV.Registered | None = None, salt: bytes | None = None) -> RV.Registered:
    """A window's produced `window_charge.VALUE`: its carry-in row and `carry_out`'s (`commit_rows`), the session's
    first carry-in the book's first carry."""
    return commit_rows(program, WC.VALUE, WC.encode(WC.first_carry(book)), WC.encode(carry_out), registration, carry,
                       salt)


__all__ = ["commit_rows", "commit_window", "openings", "start_salt", "window_commit"]
