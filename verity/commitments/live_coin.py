"""The challenger's live coin formats: a coin's opening, the challenger's key, the signature check's type and the
refusal of either form."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


_HEX = re.compile(r"(?:[0-9a-f]{2})*")


OPENING_FORM = ("a live coin opening is {after, coin, epoch, index, label, path, salt}: integers, a string, lowercase hex "
                "and a path of SHA-512 digests")


#: an LMS public key over SHA-512: I (16 bytes) || T[1] (64)
PUBLIC_BYTES = 80


KEY_FORM = ("a challenger key is {H, W, public}: LMS over SHA-512 (N = 64) with W in 1, 2, 4, 8 and 1 <= H <= 30, and its "
            "80-byte public key in lowercase hex")


class CoinError(ValueError):
    """A layout, a coin commitment or an opening the challenger refuses, or a coin its commitment doesn't hold."""


def _hex(s: Any) -> bytes | None:
    return bytes.fromhex(s) if type(s) is str and _HEX.fullmatch(s) else None


@dataclass(frozen=True, slots=True)
class LiveOpening:
    """Coin (`label`, `index`) of epoch `epoch`, opened by the challenger after the verifier recorded `after`: the coin, its
    hm96 salt and its path."""

    epoch: int
    label: str
    index: int
    after: bytes
    coin: bytes
    salt: bytes
    path: tuple[bytes, ...]

    def to_json(self) -> dict:
        return {"after": self.after.hex(), "coin": self.coin.hex(), "epoch": self.epoch, "index": self.index,
                "label": self.label, "path": [p.hex() for p in self.path], "salt": self.salt.hex()}

    @classmethod
    def from_json(cls, doc: Any) -> LiveOpening:
        if not isinstance(doc, dict) or sorted(doc) != ["after", "coin", "epoch", "index", "label", "path", "salt"]:
            raise CoinError(OPENING_FORM)
        hexes = [_hex(doc[k]) for k in ("after", "coin", "salt")]
        path = [_hex(p) for p in doc["path"]] if type(doc["path"]) is list else None
        if (type(doc["epoch"]) is not int or type(doc["index"]) is not int or type(doc["label"]) is not str
                or None in hexes or path is None or any(p is None or len(p) != 64 for p in path)):
            raise CoinError(OPENING_FORM)
        return cls(doc["epoch"], doc["label"], doc["index"], *hexes, tuple(path))


@dataclass(frozen=True, slots=True)
class ChallengerKey:
    """The verifier's key for its own challenger: LMS over SHA-512 at (64, `W`, `H`), and the public key `public`."""

    W: int
    H: int
    public: bytes

    def __post_init__(self) -> None:
        if (type(self.W) is not int or self.W not in (1, 2, 4, 8) or type(self.H) is not int or not 1 <= self.H <= 30
                or type(self.public) is not bytes or len(self.public) != PUBLIC_BYTES):
            raise CoinError(KEY_FORM)

    def to_json(self) -> dict:
        return {"H": self.H, "W": self.W, "public": self.public.hex()}

    @classmethod
    def from_json(cls, doc: Any) -> ChallengerKey:
        if not isinstance(doc, dict) or sorted(doc) != ["H", "W", "public"] or _hex(doc["public"]) is None:
            raise CoinError(KEY_FORM)
        return cls(doc["W"], doc["H"], _hex(doc["public"]))


#: `verify(key, message, signature)`: whether `signature` signs `message` under LMS at `key`'s parameters
#: (`verity.commitments.crypto.lms.verify` over `Params(64, key.W, key.H)`), false for a signature that doesn't decode.
Verify = Callable[[ChallengerKey, bytes, bytes], bool]
