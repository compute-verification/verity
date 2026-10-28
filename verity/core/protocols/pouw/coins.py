"""PoUW's epoch coins: the challenger's live coins for one epoch, committed before the epoch and each opened once, live,
against that commitment.

The challenger is the auditor's service that issues coins.  Before epoch e it draws one 32-byte coin per label of `LABELS`
from its entropy (a served one's: the Lean verifier of record's, `verifier.coins`) and publishes a coin commitment:

    x_l     = SHA-512("verity/pouw/epoch-coin/v1\\0" || coin_l)                      the coin's unsalted digest
    value_l = hm96-sha512 tree_leaf(key, commit_string(key, x_l, salt_l))           salt_l: 192 fresh bytes, key pinned
    root    = frame-v3-sha512 tree over (value_salt, value_draw) at positions 0, 1,
              domain binding tagged("verity/pouw/epoch-coins/v1", e), owner -1, leaf schema "verity/pouw/epoch-coins/v1"

The hm96 leaves keep every unopened coin statistically hidden (``hm96/PROTOCOL.md``); binding rests on SHA-512's collision
resistance.  The challenger opens a coin only once the verifier has recorded what the coin must follow (`CoinOpening.after`:
the weights' root for the salt coin, the transcript's commitment for the draw coin), and never twice, so a retry never
redraws.  These are a seeded epoch's coins; a live window's salt coin follows its own open root instead
(`salt.window_open_root`, `core_check.salt`), since the weights' root is the same in every window of a model.
`coin_reason` is the verifier's check of an opening: the epoch, the label, the record it followed, and the opening against
the published root.  A coin that doesn't open against the epoch's commitment is refused.

The coins are live and the verifier designated: an outcome convinces the auditor, who trusts its own challenger, and no
third party.  There is no public beacon and no Fiat–Shamir over a prover's commitment.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

from verity.ml.commitments import (Commitment, CommitmentDomain, MerkleTree, Opening, RangeIndexedDomain,
                                           VerificationLimits, verify_opening)
from verity.ml.commitments import domains, hm96
from verity.core.protocols.sampled_proofs.challenger import coin_claim

from .protocol import DomainError, tagged

SCHEME = domains.POUW_EPOCH_COINS
COIN_TAG = domains.POUW_EPOCH_COIN.encode() + b"\0"
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
    """The challenger's side of one epoch.  It draws every coin and hm96 salt of the epoch from `entropy` and commits to
    them at construction, before the epoch starts; `open` releases one coin, once, after the verifier has recorded
    `after`.  `entropy` has no default, so nothing here reads the operating system: a served challenger's is the Lean
    verifier of record's read of it (`verifier.coins`, `IO.getRandomBytes`, A3).  `claim` is the claim instance naming
    `entropy` (the proof service's `challenger.coin_claim`), None for an unnamed source, which a served challenger
    refuses (`challenger.served_claim`)."""

    def __init__(self, epoch: int, entropy: Callable[[int], bytes]):
        self.claim = coin_claim(entropy)

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


__all__ = ["COIN_BYTES", "LABELS", "SCHEME", "Challenger", "CoinCommitment", "CoinOpening", "coin_reason", "domain",
           "leaf_value"]
