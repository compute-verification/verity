"""The charges' carry rows under hiding commitments: each window's produced value of its carry-in and carry-out rows,
chained from window to window, and the check of how the windows join (`protocol/charge-chain.md`). `commit_window`
and `check_chain` are `charge_chain.ChargeLink_v1`'s (`VALUE` "replay-charge"); the `window_` ones are
`window_charge.ChargeLink_v2`'s."""
from __future__ import annotations

from collections.abc import Sequence

from verity.commitments import hm96
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
from verity.protocols.proofs.spot.verifier.one_stage import produced
from verity.protocols.proofs.spot.verifier.one_stage import registered as G
from verity.protocols.resource_accounting.network.prover import window_charge as WC
from verity.protocols.resource_accounting.network.verifier import arith_encode as AE, charge_chain as CC

_INST = hm96.SHA512


def start_salt(program: str, value: str = CC.VALUE) -> bytes:
    """The public salt of the session's first carry row (`registered.supplied_salts`' row 0 of `value`)."""
    return G.supplied_salts(program, value, 1)[0]


def opening(words: Sequence[int], salt: bytes) -> bytes:
    return _INST.commit_string(_INST.default_key, G.row_digest(words, CC.WORD_BITS, SCHEMA), salt)


def start_opening(program: str, book: dict) -> bytes:
    """b‖c of the priors' row under `start_salt`, which the verifier computes from the registered book."""
    return opening(CC.row(CC.encode(AE.first_carry(book))), start_salt(program))


def commit_rows(program: str, value: str, first: Sequence[int], out: Sequence[int], registration: str,
                carry: G.Registered | None = None, salt: bytes | None = None) -> G.Registered:
    """A window's produced `value` of two carry rows, given as bits: row 0 `first` under `start_salt` for the session's
    first window (`carry` None), else the previous window's row 1 under its salt; row 1 `out` under `salt` (default
    fresh, `hm96`'s `fresh_salt`)."""
    if carry is None:
        rows0, salt0 = CC.row(first), start_salt(program, value)
    else:
        rows0, salt0 = carry.rows[-1], carry.salts[-1]
    return produced.commit(program, value, [rows0, CC.row(out)], registration,
                           [salt0, salt if salt is not None else _INST.fresh_salt()], CC.WORD_BITS, SCHEMA)


def commit_window(program: str, book: dict, carry_out: AE.Carry, registration: str, carry: G.Registered | None = None,
                  salt: bytes | None = None) -> G.Registered:
    """A window's produced value: its carry-in row and `carry_out`'s. `carry`: the previous window's value, whose row 1
    is this window's row 0 under the same salt; for the session's first window (None), row 0 is the priors' row under
    `start_salt`. `salt`: row 1's (default fresh, `hm96`'s `fresh_salt`)."""
    return commit_rows(program, CC.VALUE, CC.encode(AE.first_carry(book)), CC.encode(carry_out), registration, carry,
                       salt)


def openings(value: G.Registered) -> list[bytes]:
    """Each carry row's b‖c, as the produced record's `openings` give them."""
    return list(value.bcs)


def check_chain(windows: Sequence[Sequence[bytes]], program: str, book: dict) -> str | None:
    """The verifier's check of a session's carry rows, given each window's two openings b‖c in order: None, or why not.
    `CH-rows`: a window isn't 2 carry rows; `CH-first`: the session doesn't start at the book's priors under their
    public salt; `CH-carry`: a window's carry-in isn't the previous window's carry-out."""
    return check_rows(windows, start_opening(program, book))


def check_rows(windows: Sequence[Sequence[bytes]], start: bytes) -> str | None:
    """`check_chain` given the session's first carry row's opening `start`."""
    for w, rows in enumerate(windows):
        if len(rows) != 2:
            return f"CH-rows: window {w} has {len(rows)} carry rows, not 2"
    if windows and bytes(windows[0][0]) != start:
        return "CH-first: the session's first carry isn't the book's priors under their public salt"
    for w in range(1, len(windows)):
        if bytes(windows[w][0]) != bytes(windows[w - 1][1]):
            return f"CH-carry: window {w}'s carry-in isn't window {w - 1}'s carry-out"
    return None


def window_start_opening(program: str, book: dict) -> bytes:
    """b‖c of `window_charge`'s first carry's row under its public salt (`start_salt` of its `VALUE`)."""
    return opening(CC.row(WC.encode(WC.first_carry(book))), start_salt(program, WC.VALUE))


def window_commit(program: str, book: dict, carry_out: WC.WindowCarry, registration: str,
                  carry: G.Registered | None = None, salt: bytes | None = None) -> G.Registered:
    """A window's produced `window_charge.VALUE`: its carry-in row and `carry_out`'s (`commit_rows`), the session's
    first carry-in the book's first carry."""
    return commit_rows(program, WC.VALUE, WC.encode(WC.first_carry(book)), WC.encode(carry_out), registration, carry,
                       salt)


def window_check_chain(windows: Sequence[Sequence[bytes]], program: str, book: dict) -> str | None:
    """`check_chain`'s CH-rows, CH-first and CH-carry over `window_charge.VALUE`'s rows."""
    return check_rows(windows, window_start_opening(program, book))


__all__ = ["check_chain", "check_rows", "commit_rows", "commit_window", "opening", "openings", "start_opening",
           "start_salt", "window_check_chain", "window_commit", "window_start_opening"]
