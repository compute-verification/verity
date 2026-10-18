"""The proof service's challenger: its live coins, the one live-coin source of every protocol (Lean of record:
`SampledProofs.Challenger`, `verity/core/service/sampled_proofs`, which `tests/test_difftest.py` there holds this to).

Coins are indexed by label and index, and named by no protocol.  A layout is a scheme, an epoch (the client's session the
coins are for) and its families ``(label, count)`` in order: coin ``(label, i)`` sits at its family's offset plus i, at
most `MAX_COINS` coins.  Before the epoch the challenger draws every coin and its hm96 salt from its own generator and
publishes the coin commitment:

    x_p     = SHA-512(scheme || 0x00 || coin_p)
    value_p = hm96-sha512 tree_leaf(key, commit_string(key, x_p, salt_p))           salt_p: 192 fresh bytes, key pinned
    root    = frame-v3-sha512 tree over value_0 … value_(K−1) at positions 0 … K−1, leaf schema the scheme, domain
              binding identity_digest_sha512("verity/challenger/coins/v1", {epoch, families, scheme}), owner -1

The hm96 leaves keep every unopened coin statistically hidden; binding rests on SHA-512's collision resistance.  The
challenger opens coin ``(label, i)`` once, after the verifier recorded the root the coin follows (`LiveOpening.after`),
and never twice: a retry never redraws.  There is no beacon and no Fiat–Shamir: a coin is the challenger's own bytes, and
a client derives what it draws from the opened coin (`verity.ml.randomness.derive`).  The coins are live and the verifier
designated: an outcome convinces the auditor, who trusts its own challenger, and no third party.

`live_reason` is the verifier's check of one opening and `order_reason` of a family's openings against the roots it
recorded, in order: coin i opened after root i, each once, none before its root is recorded, and every recorded root's
coin opened.  `LiveRecord` is the verifier's side of one family: it records a root and only then asks for that root's
coin.

The first client is PoUW's live window (`verity.core.protocols.pouw.window`, "Live windows"): its salt, its draw and
its round coins, round r's opened after round r's root.  The module reads only the standard library and
`verity.ml.commitments`, so any client calls it as it is.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from verity.ml.commitments import (Commitment, CommitmentDomain, MerkleTree, Opening, RangeIndexedDomain,
                                   VerificationLimits, verify_opening)
from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512

BINDING = "verity/challenger/coins/v1"
COIN_BYTES = 32
HM96 = hm96.SHA512
KEY = HM96.default_key
#: the most coins one commitment holds
MAX_COINS = 1 << 20
_NAME = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-/")
_HEX = re.compile(r"(?:[0-9a-f]{2})*")
COMMITMENT_FORM = ("a live coin commitment is {epoch, families, root, scheme}: families a list of [label, count], root "
                   "lowercase hex")
OPENING_FORM = ("a live coin opening is {after, coin, epoch, index, label, path, salt}: integers, a string, lowercase hex "
                "and a path of SHA-512 digests")


class CoinError(ValueError):
    """A layout, a coin commitment or an opening the challenger refuses, or a coin its commitment doesn't hold."""


def _name(s: Any) -> bool:
    """A scheme or a label: nonempty, of lowercase ASCII letters, digits, '-' and '/'."""
    return type(s) is str and s != "" and all(c in _NAME for c in s)


def layout_problem(scheme: Any, epoch: Any, families: Any) -> str | None:
    """Why `scheme`, `epoch` and `families` are not a live coin layout; None when they are."""
    if not _name(scheme):
        return "a live coin scheme is a nonempty name of lowercase letters, digits, '-' and '/'"
    if type(epoch) is not int or epoch < 0:
        return "an epoch is a nonnegative integer"
    if type(families) is not tuple or not families or not all(
            type(f) is tuple and len(f) == 2 and _name(f[0]) and type(f[1]) is int and f[1] >= 1 for f in families):
        return "a live coin layout is a nonempty sequence of (label, count), a name and a positive integer"
    if len({label for label, _ in families}) != len(families):
        return "a live coin layout's labels are distinct"
    if sum(n for _, n in families) > MAX_COINS:
        return f"a live coin layout holds at most {MAX_COINS} coins"
    return None


def live_domain(scheme: str, epoch: int, families: Sequence[tuple[str, int]]) -> CommitmentDomain:
    """The layout's coin tree: positions 0 … K−1 under the layout's binding, owner -1, frame-v3-sha512."""
    binding = identity_digest_sha512(BINDING, {"epoch": epoch, "families": [[label, n] for label, n in families],
                                               "scheme": scheme})
    return CommitmentDomain(binding, -1, RangeIndexedDomain(sum(n for _, n in families)), hash="sha512")


def live_leaf(scheme: str, coin: bytes, salt: bytes) -> bytes:
    """The committed value of one coin: the hm96-sha512 leaf over SHA-512(scheme ‖ 0x00 ‖ coin) under `salt`."""
    return HM96.tree_leaf(KEY, HM96.commit_string(KEY, hashlib.sha512(scheme.encode() + b"\0" + coin).digest(), salt))


def _hex(s: Any) -> bytes | None:
    return bytes.fromhex(s) if type(s) is str and _HEX.fullmatch(s) else None


@dataclass(frozen=True, slots=True)
class LiveCommitment:
    """What the challenger publishes before epoch `epoch`: its coins' layout under `scheme`, families `(label, count)` in
    order, and the 64-byte root over them."""

    scheme: str
    epoch: int
    families: tuple[tuple[str, int], ...]
    root: bytes

    def __post_init__(self) -> None:
        why = layout_problem(self.scheme, self.epoch, self.families)
        if why is not None:
            raise CoinError(why)
        if type(self.root) is not bytes or len(self.root) != 64:
            raise CoinError("a coin commitment's root is 64 bytes")

    @property
    def count(self) -> int:
        return sum(n for _, n in self.families)

    def family(self, label: str) -> int | None:
        """How many coins family `label` holds, or None when the layout has no such family."""
        return next((n for name, n in self.families if name == label), None)

    def position(self, label: str, index: int) -> int | None:
        """Coin (`label`, `index`)'s position in the tree, or None when the layout has no such coin."""
        offset = 0
        for name, n in self.families:
            if name == label:
                return offset + index if type(index) is int and 0 <= index < n else None
            offset += n
        return None

    def to_json(self) -> dict:
        return {"epoch": self.epoch, "families": [[label, n] for label, n in self.families], "root": self.root.hex(),
                "scheme": self.scheme}

    @classmethod
    def from_json(cls, doc: Any) -> LiveCommitment:
        if not isinstance(doc, dict) or sorted(doc) != ["epoch", "families", "root", "scheme"]:
            raise CoinError(COMMITMENT_FORM)
        families, root = doc["families"], _hex(doc["root"])
        if type(families) is not list or not all(type(f) is list for f in families) or root is None:
            raise CoinError(COMMITMENT_FORM)
        return cls(doc["scheme"], doc["epoch"], tuple(tuple(f) for f in families), root)


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


def live_reason(commitment: LiveCommitment, opening: LiveOpening, label: str, index: int, after: bytes) -> str | None:
    """Why `opening` is not coin (`label`, `index`) of `commitment`, opened after `after`; None when it is."""
    c, o = commitment, opening
    position = c.position(label, index)
    if position is None:
        raise CoinError(f"the epoch's coins have no {label} coin {index}")
    if o.epoch != c.epoch:
        return f"coin is epoch {o.epoch}'s, not epoch {c.epoch}'s"
    if (o.label, o.index) != (label, index):
        return f"coin is the epoch's {o.label} coin {o.index}, not its {label} coin {index}"
    if len(o.coin) != COIN_BYTES or len(o.salt) != HM96.salt_bytes:
        return "coin opening is malformed"
    if not verify_opening(live_domain(c.scheme, c.epoch, c.families), Commitment(c.root, c.count),
                          Opening(position, live_leaf(c.scheme, o.coin, o.salt), o.path), c.scheme, VerificationLimits()):
        return "coin does not open against the epoch's coin commitment"
    if o.after != after:
        return "coin was not opened after what the verifier recorded"
    return None


def order_reason(commitment: LiveCommitment, label: str, roots: Sequence[bytes], openings: Sequence[LiveOpening]
                 ) -> str | None:
    """Why `openings` are not family `label`'s coins in order, coin i opened once after `roots[i]`, the i-th root the
    verifier recorded, with every recorded root's coin opened; None when they are."""
    count = commitment.family(label)
    if count is None:
        raise CoinError(f"the epoch's coins have no {label} family")
    if len(roots) > count:
        return f"{len(roots)} roots recorded, more than the epoch's {count} {label} coins"
    seen: set[tuple[str, int]] = set()
    for i, o in enumerate(openings):
        if (o.label, o.index) in seen:
            return f"the epoch's {o.label} coin {o.index} is opened twice: a retry never redraws"
        seen.add((o.label, o.index))
        if i >= len(roots):
            return f"{label} coin {i} was opened before its root was recorded"
        why = live_reason(commitment, o, label, i, roots[i])
        if why is not None:
            return f"{label} coin {i}: {why}"
    if len(openings) < len(roots):
        return f"{label} coin {len(openings)} was never opened"
    return None


class LiveChallenger:
    """The challenger's side of one epoch's coins.  It draws every coin and hm96 salt of the layout from `entropy` (the
    operating system's generator by default) and commits to them at construction, before the epoch; `open` releases coin
    (`label`, `index`) once, after the verifier recorded `after`."""

    def __init__(self, scheme: str, epoch: int, families: Sequence[tuple[str, int]],
                 entropy: Callable[[int], bytes] = os.urandom) -> None:
        families = tuple(tuple(f) for f in families)
        why = layout_problem(scheme, epoch, families)
        if why is not None:
            raise CoinError(why)

        def draw(n: int) -> bytes:
            out = entropy(n)
            if type(out) is not bytes or len(out) != n:
                raise ValueError(f"entropy({n}) returned other than {n} bytes")
            return out

        self._coins = tuple((draw(COIN_BYTES), draw(HM96.salt_bytes)) for _ in range(sum(n for _, n in families)))
        self._tree = MerkleTree(live_domain(scheme, epoch, families),
                                {p: live_leaf(scheme, *c) for p, c in enumerate(self._coins)}, lambda _: scheme)
        self.commitment = LiveCommitment(scheme, epoch, families, self._tree.commitment.root)
        self._opened: set[tuple[str, int]] = set()

    def open(self, label: str, index: int, after: bytes) -> LiveOpening:
        position = self.commitment.position(label, index)
        if position is None:
            raise CoinError(f"the epoch's coins have no {label} coin {index}")
        if (label, index) in self._opened:
            raise CoinError(f"epoch {self.commitment.epoch}'s {label} coin {index} is already open: a retry never redraws")
        self._opened.add((label, index))
        coin, salt = self._coins[position]
        return LiveOpening(self.commitment.epoch, label, index, bytes(after), coin, salt, self._tree.open(position).path)


class LiveRecord:
    """The verifier's record of one family of an epoch's coins, in order: `record(root)` records the family's next root and
    only then asks the challenger for its coin, after that root; `reason` is `order_reason` over the record."""

    def __init__(self, challenger: LiveChallenger, label: str) -> None:
        if challenger.commitment.family(label) is None:
            raise CoinError(f"the epoch's coins have no {label} family")
        self.challenger, self.label = challenger, label
        self.roots: list[bytes] = []
        self.openings: list[LiveOpening] = []

    def record(self, root: bytes) -> LiveOpening:
        if len(self.roots) == self.challenger.commitment.family(self.label):
            raise CoinError(f"every {self.label} coin of the epoch is open")
        self.roots.append(bytes(root))
        opening = self.challenger.open(self.label, len(self.roots) - 1, root)
        self.openings.append(opening)
        return opening

    def reason(self) -> str | None:
        return order_reason(self.challenger.commitment, self.label, self.roots, self.openings)


__all__ = ["BINDING", "COIN_BYTES", "COMMITMENT_FORM", "MAX_COINS", "OPENING_FORM", "CoinError", "LiveChallenger",
           "LiveCommitment", "LiveOpening", "LiveRecord", "layout_problem", "live_domain", "live_leaf", "live_reason",
           "order_reason"]
