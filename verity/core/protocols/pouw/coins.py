"""PoUW's epoch coins: the challenger's live coins for one epoch, committed before the epoch and each opened once, live,
against that commitment (``PROTOCOL.md``, "Epoch coins").

The challenger is the auditor's service that issues coins.  Before epoch e it draws one 32-byte coin per label of `LABELS`
from its own generator and publishes a coin commitment:

    x_l     = SHA-512("verity/pouw/epoch-coin/v1\\0" || coin_l)                      the coin's unsalted digest
    value_l = hm96-sha512 tree_leaf(key, commit_string(key, x_l, salt_l))           salt_l: 192 fresh bytes, key pinned
    root    = frame-v3-sha512 tree over (value_salt, value_draw) at positions 0, 1,
              domain binding tagged("verity/pouw/epoch-coins/v1", e), owner -1, leaf schema "verity/pouw/epoch-coins/v1"

The hm96 leaves keep every unopened coin statistically hidden (``hm96/PROTOCOL.md``); binding rests on SHA-512's collision
resistance.  The challenger opens a coin only once the verifier has recorded what the coin must follow (`CoinOpening.after`:
the weights' root for the salt coin, the transcript's commitment for the draw coin), and never twice, so a retry never
redraws.  `coin_reason` is the verifier's check of an opening: the epoch, the label, the record it followed, and the opening
against the published root.  A coin that doesn't open against the epoch's commitment is refused.

The coins are live and the verifier designated: an outcome convinces the auditor, who trusts its own challenger, and no
third party.  There is no public beacon and no Fiat–Shamir over a prover's commitment.

**Live coins** (`LiveCommitment`, `LiveChallenger`), the proof service's one live-coin source, indexed by label and index
and named by no protocol: a layout is a scheme and its families ``(label, count)`` in order, coin ``(label, i)`` sitting
at its family's offset plus i.  Before the epoch the challenger draws every coin and publishes

    x_p     = SHA-512(scheme || 0x00 || coin_p)
    value_p = hm96-sha512 tree_leaf(key, commit_string(key, x_p, salt_p))           salt_p: 192 fresh bytes, key pinned
    root    = frame-v3-sha512 tree over value_0 … value_(K−1), domain binding tagged(scheme, e, label_0, count_0, …),
              owner -1, leaf schema the scheme

and opens coin ``(label, i)`` once, after the root the verifier recorded for it (`LiveOpening.after`).  `live_reason` checks
one opening; `order_reason` checks a family's openings against the roots the verifier recorded, in order: coin i opened
after root i, each once, none before its root is recorded, and every recorded root's coin opened.

**PoUW's live epoch** (`live_challenger`, its first client) has the layout `live_families(R)` under `LIVE_SCHEME`: its
salt coin, its draw coin and R round coins, R declared in the commitment as the most rounds a window of the epoch takes.
Round r's coin opens after the verifier recorded round r's root, the root of the values round r commits, and round r's
noise is keyed by `round_key` from that coin and that root, not by Fiat–Shamir over the prover's roots: the deferred
noised products (decision 36).  `rounds_reason` is the verifier's check of a window's round coins.  The commitment's scheme
is the epoch's coin mode (`mode`): `SCHEME`'s epochs are seeded, their noise Fiat–Shamir inside the scheme, and
`LIVE_SCHEME`'s live.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from verity.ml.commitments import (Commitment, CommitmentDomain, MerkleTree, Opening, RangeIndexedDomain,
                                           VerificationLimits, verify_opening)
from verity.ml.commitments import hm96
from verity.ml.randomness import Key, derive

from .protocol import DomainError, tagged

SCHEME = "verity/pouw/epoch-coins/v1"
COIN_TAG = b"verity/pouw/epoch-coin/v1\0"
LABELS = ("salt", "draw")
COIN_BYTES = 32
HM96 = hm96.SHA512
KEY = HM96.default_key


def domain(epoch: int) -> CommitmentDomain:
    return CommitmentDomain(tagged(SCHEME, epoch), -1, RangeIndexedDomain(len(LABELS)), hash="sha512")


def leaf_value(coin: bytes, salt: bytes) -> bytes:
    """The committed value of one coin: the hm96-sha512 leaf over its unsalted digest under `salt`."""
    return HM96.tree_leaf(KEY, HM96.commit_string(KEY, hashlib.sha512(COIN_TAG + coin).digest(), salt))


@dataclass(frozen=True, slots=True)
class CoinCommitment:
    """What the challenger publishes before epoch `epoch`: the 64-byte root over the epoch's coins."""

    epoch: int
    root: bytes

    def __post_init__(self) -> None:
        if type(self.epoch) is not int or self.epoch < 0:
            raise DomainError("an epoch is a nonnegative integer")
        if type(self.root) is not bytes or len(self.root) != 64:
            raise DomainError("a coin commitment's root is 64 bytes")

    def to_json(self) -> dict:
        return {"scheme": SCHEME, "epoch": self.epoch, "root": self.root.hex()}

    @classmethod
    def from_json(cls, doc: dict) -> CoinCommitment:
        if doc.get("scheme") != SCHEME:
            raise DomainError(f"a coin commitment's scheme is {SCHEME!r}")
        return cls(doc["epoch"], bytes.fromhex(doc["root"]))


@dataclass(frozen=True, slots=True)
class CoinOpening:
    """One coin of epoch `epoch`, opened by the challenger after it recorded `after`: the coin, its hm96 salt and its path."""

    epoch: int
    label: str
    after: bytes
    coin: bytes
    salt: bytes
    path: tuple[bytes, ...]

    def to_json(self) -> dict:
        return {"epoch": self.epoch, "label": self.label, "after": self.after.hex(), "coin": self.coin.hex(),
                "salt": self.salt.hex(), "path": [p.hex() for p in self.path]}

    @classmethod
    def from_json(cls, doc: dict) -> CoinOpening:
        if type(doc["epoch"]) is not int:
            raise DomainError("an opening's epoch is an integer")
        if type(doc["label"]) is not str:
            raise DomainError("an opening's label is a string")
        return cls(doc["epoch"], doc["label"], bytes.fromhex(doc["after"]), bytes.fromhex(doc["coin"]),
                   bytes.fromhex(doc["salt"]), tuple(bytes.fromhex(p) for p in doc["path"]))


def coin_reason(commitment: CoinCommitment, opening: CoinOpening, label: str, after: bytes) -> str | None:
    """Why `opening` is not epoch `commitment.epoch`'s `label` coin, opened after `after` and opening against
    `commitment`; None when it is."""
    if label not in LABELS:
        raise DomainError(f"an epoch's coins are {LABELS}")
    if opening.epoch != commitment.epoch:
        return f"coin is epoch {opening.epoch}'s, not epoch {commitment.epoch}'s"
    if opening.label != label:
        return f"coin is the epoch's {opening.label!r} coin, not its {label!r} coin"
    if len(opening.coin) != COIN_BYTES or len(opening.salt) != HM96.salt_bytes:
        return "coin opening is malformed"
    value = leaf_value(opening.coin, opening.salt)
    if not verify_opening(domain(commitment.epoch), Commitment(commitment.root, len(LABELS)),
                          Opening(LABELS.index(label), value, opening.path), SCHEME, VerificationLimits()):
        return "coin does not open against the epoch's coin commitment"
    if opening.after != after:
        return "coin was not opened after what the verifier recorded"
    return None


class Challenger:
    """The challenger's side of one epoch.  It draws every coin and hm96 salt of the epoch from `entropy` (the operating
    system's generator by default) and commits to them at construction, before the epoch starts; `open` releases one
    coin, once, after the verifier has recorded `after`."""

    def __init__(self, epoch: int, entropy: Callable[[int], bytes] = os.urandom):
        def draw(n: int) -> bytes:
            out = entropy(n)
            if type(out) is not bytes or len(out) != n:
                raise ValueError(f"entropy({n}) returned {len(out)} bytes")
            return out

        self._coins = {label: (draw(COIN_BYTES), draw(HM96.salt_bytes)) for label in LABELS}
        self._tree = MerkleTree(domain(epoch), {i: leaf_value(*self._coins[label]) for i, label in enumerate(LABELS)},
                                lambda _: SCHEME)
        self.commitment = CoinCommitment(epoch, self._tree.commitment.root)
        self._opened: set[str] = set()

    @property
    def epoch(self) -> int:
        return self.commitment.epoch

    def open(self, label: str, after: bytes) -> CoinOpening:
        if label not in LABELS:
            raise DomainError(f"an epoch's coins are {LABELS}")
        if label in self._opened:
            raise DomainError(f"epoch {self.epoch}'s {label!r} coin is already open: a retry never redraws")
        self._opened.add(label)
        coin, salt = self._coins[label]
        return CoinOpening(self.epoch, label, bytes(after), coin, salt, self._tree.open(LABELS.index(label)).path)


# ---- live coins, by label and index ------------------------------------------------------------------------------------

LIVE_SCHEME = "verity/pouw/epoch-coins/live/v1"
ROUND = "round"
ROUND_NOISE_DOMAIN = "verity/pouw/round-noise/v1"
#: an epoch's coin mode, Compose's `CoinMode`: its scheme's (`mode`)
MODES = ("live", "seeded")
#: the most coins one live commitment holds
MAX_COINS = 1 << 20
_NAME = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")


def _name(s: Any, scheme: bool) -> bool:
    """A scheme or a label: nonempty, of lowercase ASCII letters, digits and '-', and '/' in a scheme."""
    return type(s) is str and s != "" and all(c in _NAME or (scheme and c == "/") for c in s)


def layout_problem(scheme: Any, epoch: Any, families: Any) -> str | None:
    """Why `scheme`, `epoch` and `families` are not a live coin layout; None when they are."""
    if not _name(scheme, True):
        return "a live coin scheme is a nonempty name of lowercase letters, digits, '-' and '/'"
    if type(epoch) is not int or epoch < 0:
        return "an epoch is a nonnegative integer"
    if type(families) is not tuple or not families or not all(
            type(f) is tuple and len(f) == 2 and _name(f[0], False) and type(f[1]) is int and f[1] >= 1 for f in families):
        return "a live coin layout is a nonempty sequence of (label, count), a name and a positive integer"
    if len({label for label, _ in families}) != len(families):
        return "a live coin layout's labels are distinct"
    if sum(n for _, n in families) > MAX_COINS:
        return f"a live coin layout holds at most {MAX_COINS} coins"
    return None


def live_domain(scheme: str, epoch: int, families: Sequence[tuple[str, int]]) -> CommitmentDomain:
    """The layout's coin tree: `CommitmentDomain(tagged(scheme, epoch, label_0, count_0, …), -1, positions, "sha512")`."""
    return CommitmentDomain(tagged(scheme, epoch, *(p for f in families for p in f)), -1,
                            RangeIndexedDomain(sum(n for _, n in families)), hash="sha512")


def live_leaf(scheme: str, coin: bytes, salt: bytes) -> bytes:
    """The committed value of one live coin: the hm96-sha512 leaf over SHA-512(scheme ‖ 0x00 ‖ coin) under `salt`."""
    return HM96.tree_leaf(KEY, HM96.commit_string(KEY, hashlib.sha512(scheme.encode() + b"\0" + coin).digest(), salt))


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
            raise DomainError(why)
        if type(self.root) is not bytes or len(self.root) != 64:
            raise DomainError("a coin commitment's root is 64 bytes")

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
    def from_json(cls, doc: dict) -> LiveCommitment:
        families = doc["families"]
        if type(families) is not list or not all(type(f) is list for f in families):
            raise DomainError("a live coin layout is a list of [label, count]")
        return cls(doc["scheme"], doc["epoch"], tuple(tuple(f) for f in families), bytes.fromhex(doc["root"]))


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
    def from_json(cls, doc: dict) -> LiveOpening:
        if type(doc["epoch"]) is not int or type(doc["index"]) is not int:
            raise DomainError("an opening's epoch and index are integers")
        if type(doc["label"]) is not str:
            raise DomainError("an opening's label is a string")
        return cls(doc["epoch"], doc["label"], doc["index"], bytes.fromhex(doc["after"]), bytes.fromhex(doc["coin"]),
                   bytes.fromhex(doc["salt"]), tuple(bytes.fromhex(p) for p in doc["path"]))


def live_reason(commitment: LiveCommitment, opening: LiveOpening, label: str, index: int, after: bytes) -> str | None:
    """Why `opening` is not coin (`label`, `index`) of `commitment`, opened after `after`; None when it is."""
    position = commitment.position(label, index)
    if position is None:
        raise DomainError(f"the epoch's coins have no {label!r} coin {index}")
    if opening.epoch != commitment.epoch:
        return f"coin is epoch {opening.epoch}'s, not epoch {commitment.epoch}'s"
    if (opening.label, opening.index) != (label, index):
        return f"coin is the epoch's {opening.label!r} coin {opening.index}, not its {label!r} coin {index}"
    if len(opening.coin) != COIN_BYTES or len(opening.salt) != HM96.salt_bytes:
        return "coin opening is malformed"
    c = commitment
    value = live_leaf(c.scheme, opening.coin, opening.salt)
    if not verify_opening(live_domain(c.scheme, c.epoch, c.families), Commitment(c.root, c.count),
                          Opening(position, value, opening.path), c.scheme, VerificationLimits()):
        return "coin does not open against the epoch's coin commitment"
    if opening.after != after:
        return "coin was not opened after what the verifier recorded"
    return None


def order_reason(commitment: LiveCommitment, label: str, roots: Sequence[bytes], openings: Sequence[LiveOpening]
                 ) -> str | None:
    """Why `openings` are not family `label`'s coins in order, coin i opened once after `roots[i]`, the i-th root the
    verifier recorded, with every recorded root's coin opened; None when they are."""
    count = commitment.family(label)
    if count is None:
        raise DomainError(f"the epoch's coins have no {label!r} family")
    if len(roots) > count:
        return f"{len(roots)} roots recorded, more than the epoch's {count} {label!r} coins"
    seen: set[tuple[str, int]] = set()
    for i, o in enumerate(openings):
        if (o.label, o.index) in seen:
            return f"the epoch's {o.label!r} coin {o.index} is opened twice: a retry never redraws"
        seen.add((o.label, o.index))
        if i >= len(roots):
            return f"{label!r} coin {i} was opened before its root was recorded"
        why = live_reason(commitment, o, label, i, roots[i])
        if why is not None:
            return f"{label!r} coin {i}: {why}"
    if len(openings) < len(roots):
        return f"{label!r} coin {len(openings)} was never opened"
    return None


class LiveChallenger:
    """The challenger's side of one epoch's live coins.  It draws every coin and hm96 salt of the layout from `entropy` (the
    operating system's generator by default) and commits to them at construction, before the epoch; `open` releases coin
    (`label`, `index`) once, after the verifier recorded `after`."""

    def __init__(self, scheme: str, epoch: int, families: Sequence[tuple[str, int]],
                 entropy: Callable[[int], bytes] = os.urandom) -> None:
        families = tuple(tuple(f) for f in families)
        why = layout_problem(scheme, epoch, families)
        if why is not None:
            raise DomainError(why)

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
            raise DomainError(f"the epoch's coins have no {label!r} coin {index}")
        if (label, index) in self._opened:
            raise DomainError(f"epoch {self.commitment.epoch}'s {label!r} coin {index} is already open: a retry never "
                              "redraws")
        self._opened.add((label, index))
        coin, salt = self._coins[position]
        return LiveOpening(self.commitment.epoch, label, index, bytes(after), coin, salt, self._tree.open(position).path)


class LiveRecord:
    """The verifier's record of one family of a live epoch's coins, in order: `record(root)` records the family's next root
    and only then asks the challenger for its coin, after that root; `reason` is `order_reason` over the record."""

    def __init__(self, challenger: LiveChallenger, label: str) -> None:
        if challenger.commitment.family(label) is None:
            raise DomainError(f"the epoch's coins have no {label!r} family")
        self.challenger, self.label = challenger, label
        self.roots: list[bytes] = []
        self.openings: list[LiveOpening] = []

    def record(self, root: bytes) -> LiveOpening:
        if len(self.roots) == self.challenger.commitment.family(self.label):
            raise DomainError(f"every {self.label!r} coin of the epoch is open")
        self.roots.append(bytes(root))
        opening = self.challenger.open(self.label, len(self.roots) - 1, root)
        self.openings.append(opening)
        return opening

    def reason(self) -> str | None:
        return order_reason(self.challenger.commitment, self.label, self.roots, self.openings)


# ---- PoUW's live epoch -------------------------------------------------------------------------------------------------

def mode(scheme: str) -> str:
    """The coin mode of an epoch whose coin commitment is of `scheme` (`MODES`)."""
    if scheme == SCHEME:
        return "seeded"
    if scheme == LIVE_SCHEME:
        return "live"
    raise DomainError(f"a PoUW epoch's coin scheme is {SCHEME!r} (seeded) or {LIVE_SCHEME!r} (live)")


def live_families(rounds: int) -> tuple[tuple[str, int], ...]:
    """A live epoch's layout: its salt coin, its draw coin and `rounds` round coins, the most rounds a window takes."""
    return (("salt", 1), ("draw", 1), (ROUND, rounds))


def live_challenger(epoch: int, rounds: int, entropy: Callable[[int], bytes] = os.urandom) -> LiveChallenger:
    return LiveChallenger(LIVE_SCHEME, epoch, live_families(rounds), entropy)


def live_epoch_reason(commitment: LiveCommitment) -> str | None:
    """Why `commitment` is not a live PoUW epoch's (`LIVE_SCHEME`, `live_families`); None when it is."""
    if commitment.scheme != LIVE_SCHEME:
        return f"the coin commitment's scheme is {commitment.scheme!r}, not {LIVE_SCHEME!r}"
    f = commitment.families
    if len(f) != 3 or f[0] != ("salt", 1) or f[1] != ("draw", 1) or f[2][0] != ROUND:
        return "the coin commitment's coins are not a live epoch's: one salt, one draw, then the rounds'"
    return None


def rounds_reason(commitment: LiveCommitment, roots: Sequence[bytes], openings: Sequence[LiveOpening]) -> str | None:
    """Why `openings` are not a window's round coins over the round roots the verifier recorded, `roots` in order: round
    r's coin opened once, after round r's root; None when they are."""
    return live_epoch_reason(commitment) or order_reason(commitment, ROUND, roots, openings)


def round_key(commitment: LiveCommitment, coin: bytes, r: int, root: bytes) -> Key:
    """Round r's noise key, from its opened coin and the round's root: no Fiat–Shamir over the prover's roots."""
    return derive(coin, ROUND_NOISE_DOMAIN, {"coins": commitment.root, "epoch": commitment.epoch, "root": root, "round": r})


__all__ = ["COIN_BYTES", "LABELS", "LIVE_SCHEME", "MAX_COINS", "MODES", "ROUND", "ROUND_NOISE_DOMAIN", "SCHEME",
           "Challenger", "CoinCommitment", "CoinOpening", "LiveChallenger", "LiveCommitment", "LiveOpening", "LiveRecord",
           "coin_reason", "domain", "layout_problem", "leaf_value", "live_challenger", "live_domain", "live_epoch_reason",
           "live_families", "live_leaf", "live_reason", "mode", "order_reason", "round_key", "rounds_reason"]
