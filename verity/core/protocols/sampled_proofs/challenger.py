"""The proof service's challenger: its live coins, the one live-coin source of every protocol (Lean of record:
`SampledProofs.Challenger`, `verity/core/service/sampled_proofs`, which `tests/test_difftest.py` there holds this to).

Coins are indexed by label and index, and named by no protocol.  A layout is a scheme, an epoch (the client's session the
coins are for) and its families ``(label, count)`` in order: coin ``(label, i)`` sits at its family's offset plus i, at
most `MAX_COINS` coins.  Before the epoch the challenger draws every coin and its hm96 salt from its own generator and
publishes the coin commitment:

    x_p     = SHA-512(scheme || 0x00 || coin_p)                                     coin_p: 64 fresh bytes (`COIN_BYTES`)
    value_p = hm96-sha512 tree_leaf(key, commit_string(key, x_p, salt_p))           salt_p: 192 fresh bytes, key pinned
    root    = frame-v3-sha512 tree over value_0 … value_(K−1) at positions 0 … K−1, leaf schema the scheme, domain
              binding identity_digest_sha512("verity/challenger/coins/v1", {epoch, families, scheme}), owner -1

The hm96 leaves keep every unopened coin statistically hidden; binding rests on SHA-512's collision resistance.  The
challenger opens coin ``(label, i)`` once, after the verifier recorded the root the coin follows (`LiveOpening.after`),
and never twice: a retry never redraws.  There is no beacon and no Fiat–Shamir: a coin is the challenger's own bytes, and
a client derives what it draws from the opened coin (`verity.ml.randomness.derive`).  The coins are live and the verifier
designated: an outcome convinces the auditor, who trusts its own challenger, and no third party.

Timing is a trust assumption, not a check.  `after` binds an opening to the root the verifier recorded, but can't show
the coin was released after it: that rests on the challenger releasing coin i only once it has received root i.  The
challenger is auditor-trusted, the auditor's device on a machine the prover doesn't control, under the named assumption
`trusted-nebius-host`.  The record keeps the release order with its own time (`LiveRecord.log`).

`live_reason` is the verifier's check of one opening and `order_reason` of a family's openings against the roots it
recorded, in order: coin i opened after root i, each once, none before its root is recorded, and every recorded root's
coin opened.  `LiveRecord` is the verifier's side of one family: it records a root and only then asks for that root's
coin, and logs both on its clock.

**The challenger's record** is the verifier's trusted coin commitment.  Before any coin opens, the challenger signs its
commitment with its LMS key over SHA-512 (`LiveChallenger.sign_record`) and writes the record on its own host
(`write_record`):

    record  = {commitment, host, signature}                       host: the auditor machine the challenger runs on
    message = SHA-512([RECORD_ROLE] || I || [[host]] || [identity_digest_sha512(COMMITMENT_TAG, commitment)])

each [x] a u64 big-endian length then x, and I the key's identifier (its public key's first 16 bytes): the LMS role
digest's form (`verity.ml.crypto.lms`).  The verifier reads the record from the challenger's file, never from the prover's
bundle, and checks it under its own challenger's key (`ChallengerKey`: the LMS parameters and public key):
`trusted_commitment` returns the commitment, or raises naming why the record is missing, malformed or unsigned, so a client
that trusts no other commitment fails closed.  `record_reason` is that check and `trusted_reason` adds that a commitment the
prover presents is the record's.  This module verifies no signature itself (a verifier imports no `verity.ml` code beyond
its frozen edges): the caller hands in `verify`, LMS at the key's parameters, and the Lean of record verifies on its own
(`Lms.verify`).

The first client is PoUW's live window (`verity.core.protocols.pouw.window`, "Live windows"): its salt, its draw and
its round coins, round r's opened after round r's root.  The module reads only the standard library and
`verity.ml.commitments`, so any client calls it as it is.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import string
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.ml.commitments import (Commitment, CommitmentDomain, MerkleTree, Opening, RangeIndexedDomain,
                                   VerificationLimits, verify_opening)
from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512

BINDING = "verity/challenger/coins/v1"
COIN_BYTES = 64
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
RECORD_ROLE = "verity/challenger-record/v1"
COMMITMENT_TAG = "verity/challenger/commitment/v1"
#: an LMS public key over SHA-512: I (16 bytes) || T[1] (64)
PUBLIC_BYTES = 80
_HOST = frozenset(string.ascii_letters + string.digits + "-._")
RECORD_FORM = ("a challenger record is {commitment, host, signature}: a live coin commitment, a host of ASCII letters, "
               "digits, '-', '.' and '_', and lowercase hex")
KEY_FORM = ("a challenger key is {H, W, public}: LMS over SHA-512 (N = 64) with W in 1, 2, 4, 8 and 1 <= H <= 30, and its "
            "80-byte public key in lowercase hex")
NO_RECORD = "no challenger record: the verifier holds no coin commitment its challenger recorded and signed"


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


def _host(s: Any) -> bool:
    return type(s) is str and s != "" and all(c in _HOST for c in s)


def _field(b: bytes) -> bytes:
    return len(b).to_bytes(8, "big") + b


def record_message(I: bytes, host: str, commitment: LiveCommitment) -> bytes:
    """What a challenger's record signs (module docstring), under the key whose identifier is `I`."""
    if len(I) != 16:
        raise CoinError(f"an LMS identifier is 16 bytes, not {len(I)}")
    digest = identity_digest_sha512(COMMITMENT_TAG, commitment.to_json())
    return hashlib.sha512(_field(RECORD_ROLE.encode()) + I + _field(_field(host.encode())) + _field(digest)).digest()


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


@dataclass(frozen=True, slots=True)
class ChallengerRecord:
    """The challenger's record of an epoch's coin commitment, signed on host `host` before any coin opened."""

    commitment: LiveCommitment
    host: str
    signature: bytes

    def to_json(self) -> dict:
        return {"commitment": self.commitment.to_json(), "host": self.host, "signature": self.signature.hex()}

    @classmethod
    def from_json(cls, doc: Any) -> ChallengerRecord:
        if not isinstance(doc, dict) or sorted(doc) != ["commitment", "host", "signature"]:
            raise CoinError(RECORD_FORM)
        signature = _hex(doc["signature"])
        if not _host(doc["host"]) or signature is None:
            raise CoinError(RECORD_FORM)
        return cls(LiveCommitment.from_json(doc["commitment"]), doc["host"], signature)


#: `verify(key, message, signature)`: whether `signature` signs `message` under LMS at `key`'s parameters
#: (`verity.ml.crypto.lms.verify` over `Params(64, key.W, key.H)`), false for a signature that doesn't decode.
Verify = Callable[[ChallengerKey, bytes, bytes], bool]


def record_reason(key: ChallengerKey, record: Any, verify: Verify) -> str | None:
    """Why `record` (its JSON, None when the verifier has none) is not a challenger record signed under `key`; None when
    it is."""
    if record is None:
        return NO_RECORD
    try:
        r = ChallengerRecord.from_json(record)
    except CoinError as e:
        return f"the challenger record is malformed: {e}"
    try:
        ok = verify(key, record_message(key.public[:16], r.host, r.commitment), r.signature) is True
    except ValueError:
        ok = False
    if not ok:
        return "the challenger record's signature does not verify under the verifier's challenger key"
    return None


def trusted_reason(key: ChallengerKey, record: Any, commitment: LiveCommitment, verify: Verify) -> str | None:
    """Why `commitment`, a coin commitment the prover presents, is not the one in the challenger's record (`record_reason`
    under `key`); None when it is."""
    why = record_reason(key, record, verify)
    if why is not None:
        return why
    trusted = ChallengerRecord.from_json(record).commitment
    if commitment != trusted:
        return (f"coin commitment {commitment.root.hex()[:16]} is not the one the verifier's challenger recorded "
                f"({trusted.root.hex()[:16]})")
    return None


def write_record(path: str | os.PathLike, record: ChallengerRecord) -> None:
    """Write the challenger's record to `path` on its own host, once: a record that exists is refused."""
    with open(path, "x", encoding="utf-8") as f:
        f.write(json.dumps(record.to_json(), sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def trusted_commitment(path: str | os.PathLike, key: ChallengerKey, verify: Verify) -> LiveCommitment:
    """The verifier's trusted coin commitment: the record at `path`, the file the challenger wrote on its own host (never
    the prover's bundle), checked under the verifier's challenger key (`record_reason`).  A missing, malformed or unsigned
    record raises `CoinError` naming why."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        doc = None
    except (OSError, ValueError) as e:
        raise CoinError(f"the challenger record at {path} is unreadable: {e}") from e
    why = record_reason(key, doc, verify)
    if why is not None:
        raise CoinError(why if doc is not None else f"{why} (none at {path})")
    return ChallengerRecord.from_json(doc).commitment


class LiveChallenger:
    """The challenger's side of one epoch's coins.  It draws every coin and hm96 salt of the layout from `entropy` and
    commits to them at construction, before the epoch; `open` releases coin (`label`, `index`) once, after the verifier
    recorded `after`.  `entropy` has no default, so nothing here reads the operating system: a served challenger's is the
    Lean verifier of record's read of it (`IO.getRandomBytes`, A3)."""

    def __init__(self, scheme: str, epoch: int, families: Sequence[tuple[str, int]],
                 entropy: Callable[[int], bytes]) -> None:
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
        self.record: ChallengerRecord | None = None

    def sign_record(self, host: str, public: bytes, sign: Callable[[bytes], bytes]) -> ChallengerRecord:
        """The challenger's record of its commitment, signed once and before any coin opens: `sign(message)` signs with
        the LMS key whose public key is `public`, on host `host`."""
        if self._opened:
            raise CoinError("a challenger signs its record before any coin opens, and a coin is open")
        if self.record is not None:
            raise CoinError("the challenger's record is signed once")
        if not _host(host) or type(public) is not bytes or len(public) != PUBLIC_BYTES:
            raise CoinError(RECORD_FORM)
        self.record = ChallengerRecord(self.commitment, host, bytes(sign(record_message(public[:16], host, self.commitment))))
        return self.record

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
    only then asks the challenger for its coin, after that root; `reason` is `order_reason` over the record.  `clock` is
    the record's clock in seconds, and `log` the release order on it."""

    def __init__(self, challenger: LiveChallenger, label: str, clock: Callable[[], float] = time.monotonic) -> None:
        if challenger.commitment.family(label) is None:
            raise CoinError(f"the epoch's coins have no {label} family")
        self.challenger, self.label, self._clock = challenger, label, clock
        self.roots: list[bytes] = []
        self.openings: list[LiveOpening] = []
        self._log: list[dict[str, Any]] = []

    @property
    def log(self) -> tuple[dict[str, Any], ...]:
        """The release order with the record's time, two events a coin: root i's receipt, `{"event": "root", "label",
        "index", "root": <hex>, "received_at"}`, then coin i's release after it, `{"event": "release", "label", "index",
        "after": <hex>, "released_at"}`.  It records the order; that the challenger kept it is `trusted-nebius-host`'s."""
        return tuple(self._log)

    def record(self, root: bytes) -> LiveOpening:
        if len(self.roots) == self.challenger.commitment.family(self.label):
            raise CoinError(f"every {self.label} coin of the epoch is open")
        i = len(self.roots)
        self.roots.append(bytes(root))
        self._log.append({"event": "root", "label": self.label, "index": i, "root": bytes(root).hex(),
                          "received_at": self._clock()})
        opening = self.challenger.open(self.label, i, root)
        self.openings.append(opening)
        self._log.append({"event": "release", "label": self.label, "index": i, "after": opening.after.hex(),
                          "released_at": self._clock()})
        return opening

    def reason(self) -> str | None:
        return order_reason(self.challenger.commitment, self.label, self.roots, self.openings)


__all__ = ["BINDING", "COIN_BYTES", "COMMITMENT_FORM", "COMMITMENT_TAG", "KEY_FORM", "MAX_COINS", "NO_RECORD", "OPENING_FORM",
           "PUBLIC_BYTES", "RECORD_FORM", "RECORD_ROLE", "ChallengerKey", "ChallengerRecord", "CoinError", "LiveChallenger",
           "LiveCommitment", "LiveOpening", "LiveRecord", "Verify", "layout_problem", "live_domain", "live_leaf", "live_reason",
           "order_reason", "record_message", "record_reason", "trusted_commitment", "trusted_reason", "write_record"]
