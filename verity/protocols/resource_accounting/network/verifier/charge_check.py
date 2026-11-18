"""The verifier's check of how the charges' carry rows join from window to window (`protocol/charge-chain.md`): the
session's first carry-in is the book's first carry under its public salt, and each later one the last carry-out.
`check_chain` is `charge_chain.ChargeLink_v1`'s (`VALUE` "replay-charge"); `window_check_chain` is
`window_charge.ChargeLink_v2`'s."""
from __future__ import annotations

from collections.abc import Sequence

from verity.commitments import hm96
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
from verity.protocols.interface.charge import arith_encode as AE, charge_chain as CC, window_charge as WC
from verity.protocols.proofs.spot.verifier.one_stage import registered as G

_INST = hm96.SHA512


def opening(words: Sequence[int], salt: bytes) -> bytes:
    """A carry row's b‖c: its u16 words under `salt`."""
    return _INST.commit_string(_INST.default_key, G.row_digest(words, CC.WORD_BITS, SCHEMA), salt)


def start_opening(program: str, book: dict) -> bytes:
    """b‖c of the priors' row under its public salt (`registered.supplied_salts`' row 0 of `VALUE`)."""
    return opening(CC.row(CC.encode(AE.first_carry(book))), G.supplied_salts(program, CC.VALUE, 1)[0])


def window_start_opening(program: str, book: dict) -> bytes:
    """b‖c of `window_charge`'s first carry's row under its public salt."""
    return opening(CC.row(WC.encode(WC.first_carry(book))), G.supplied_salts(program, WC.VALUE, 1)[0])


def check_chain(windows: Sequence[Sequence[bytes]], program: str, book: dict) -> str | None:
    """A session's carry rows, given each window's two openings b‖c in order: None, or why not (`charge_chain.check_rows`)."""
    return CC.check_rows(windows, start_opening(program, book))


def window_check_chain(windows: Sequence[Sequence[bytes]], program: str, book: dict) -> str | None:
    """`check_chain` over `window_charge.VALUE`'s rows."""
    return CC.check_rows(windows, window_start_opening(program, book))


__all__ = ["check_chain", "opening", "start_opening", "window_check_chain", "window_start_opening"]
