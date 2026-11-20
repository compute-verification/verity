"""The registration's hiding commitment (Lean of record: `Spot.RegistrationCommitment`), and a challenger's coins taken
after it. No verifier or benchmark reads either yet."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Mapping

from verity.circuits.codec import canonical_json
from verity.commitments import docket_digest as DD
from verity.commitments.hm96 import SHA512 as HM

TAG = "verity/registration-commitment/v1"
COIN_TAG = "verity/registration-commitment/coin/v1"
OPENING = "registration-commitment-opening"
DIGEST = "registration-commitment-digest"
COMPILER = ("registration-commitment-compiler: the record names a compiler (window.compiler.digest) and the verifier read "
            "no registration log's compiler, or a log that registers none: null commits only a log that registers no "
            "compiler")
#: the compiler of a verifier that read no registration log's compiler (Lean's none; None is a log that registers none)
NO_LOG = "no-log"
COIN_BYTES = 64
KEY = HM.default_key

Compiler = tuple[str, int] | None | str


class Refused(ValueError):
    """A commitment `whole` or `opens` refuses; its message is the refusal (`OPENING`, `DIGEST` or `COMPILER`)."""


def _get(j: Any, k: str) -> Any:
    return j.get(k) if isinstance(j, Mapping) else None


def names_compiler(rec: Mapping[str, Any]) -> bool:
    """The record names a compiler: its ``window.compiler.digest`` is not null."""
    return _get(_get(_get(rec, "window"), "compiler"), "digest") is not None


def whole(rec: Mapping[str, Any], compiler: Compiler) -> bytes:
    """The whole registration digest: the docket's digest and the registration log's compiler, ``(digest, cap)``; null
    for None (a log that registers none) or `NO_LOG` only where the record names no compiler, else `COMPILER`."""
    if isinstance(compiler, tuple):
        c = {"cap": int(compiler[1]), "digest": compiler[0]}
    elif names_compiler(rec):
        raise Refused(COMPILER)
    else:
        c = None
    return DD.record_digest(TAG, canonical_json({"compiler": c, "record": DD.digest(rec)}))


def commitment(d: bytes, salt: bytes) -> bytes:
    """The hm96-sha512 leaf of the whole digest ``d`` under ``salt``."""
    return HM.tree_leaf(KEY, HM.commit_string(KEY, d, salt))


def commit(rec: Mapping[str, Any], compiler: Compiler) -> tuple[bytes, bytes, bytes]:
    """The prover's side: ``(commitment, whole digest, salt)``, the salt fresh from the operating system."""
    d, salt = whole(rec, compiler), HM.fresh_salt()
    return commitment(d, salt), d, salt


def opens(c: bytes, d: bytes, salt: bytes, rec: Mapping[str, Any], compiler: Compiler) -> None:
    """`Spot.RegistrationCommitment.opens`: refuses `COMPILER` where `whole` does, `DIGEST` when ``d`` is not the whole
    digest of the registration the verifier reads (``rec``, ``compiler``), `OPENING` when ``salt`` and that digest give
    another leaf than ``c``."""
    mine = whole(rec, compiler)
    if bytes(d) != mine:
        raise Refused(DIGEST)
    if len(salt) != HM.salt_bytes or commitment(mine, salt) != bytes(c):
        raise Refused(OPENING)


def bound_coin(coin: bytes, c: bytes) -> bytes:
    """SHA-512(COIN_TAG ‖ 0 ‖ coin ‖ c): a new derivation of a coin from a challenger's coin released after the
    commitment ``c``, so the coin is a function of ``c``. No verifier reads it yet, and the coins any audit draws today
    are the challenger's raw coins."""
    return hashlib.sha512(COIN_TAG.encode() + b"\0" + bytes(coin) + bytes(c)).digest()


class Bound:
    """A challenger's coins (`coins.ChallengerCoins`, or anything with its ``commit(after)`` and ``__call__(n)``) taken
    after the registration commitment: ``commit(c)`` hands ``c`` to the challenger as the 64 bytes its coins follow, and
    each read is the stream of `bound_coin` over the raw coins, read whole, in order. The coins underneath are unchanged."""

    def __init__(self, coins: Any) -> None:
        self._coins, self._c, self._stream, self._at = coins, None, b"", 0

    def commit(self, c: bytes) -> Any:
        c = bytes(c)
        if len(c) != COIN_BYTES:
            raise ValueError("a registration commitment is 64 bytes")
        segment = self._coins.commit(c)
        self._c, self._stream, self._at = c, b"", 0
        return segment

    def __call__(self, n: int) -> bytes:
        if self._c is None:
            raise ValueError("bound coins are read only after the registration commitment (Bound.commit)")
        read: Callable[[int], bytes] = self._coins
        while len(self._stream) < self._at + n:
            self._stream += bound_coin(read(COIN_BYTES), self._c)
        self._at += n
        return self._stream[self._at - n:self._at]


__all__ = ["COIN_TAG", "COMPILER", "DIGEST", "NO_LOG", "OPENING", "TAG", "Bound", "Compiler", "Refused", "bound_coin",
           "commit", "commitment", "names_compiler", "opens", "whole"]
