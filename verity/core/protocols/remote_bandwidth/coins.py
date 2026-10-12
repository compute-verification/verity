"""The challenger's coin stream for one session (`verity/remote-bandwidth/coins/v0`): one live coin a pass and one for
each window's openings, every coin drawn and committed before the session and each opened once, on the schedule.

~~~text
K          W·(P + 1) coins for W windows of P passes: pass k's coin at k < W·P, window t's openings coin at W·P + t
c_k        32 bytes from the challenger's generator
x_k        SHA512("verity/remote-bandwidth/coin/v0\0" ‖ c_k)
value_k    hm96-sha512 tree_leaf(key, commit_string(key, x_k, y_k)), y_k a fresh 192-byte salt, key hm96-sha512's pinned key
root       frame-v3-sha512 over value_0 … value_(K−1): CommitmentDomain(SHA512("verity/remote-bandwidth/coins/v0\0" ‖
           registration digest ‖ u64be session), −1, positions 0 … K − 1, "sha512"), schema "verity/remote-bandwidth/coins/v0"
~~~

The prover has the root before the first coin and checks every coin it is sent against it (`coin_reason`), so the
challenger can't choose a coin after seeing a root.  hm96 keeps every unopened coin statistically hidden
(`hm96/PROTOCOL.md`), so the root says nothing about a pass's seed before its coin is released.  The construction is
PoUW's epoch coins' (`verity.core.protocols.pouw.coins`) over a stream: the coins are live and the verifier designated,
and nothing is derived from the prover's messages.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass

from verity.ml.commitments import Commitment, CommitmentDomain, MerkleTree, Opening, RangeIndexedDomain, VerificationLimits
from verity.ml.commitments import hm96, verify_opening

SCHEME = "verity/remote-bandwidth/coins/v0"
COIN_DOMAIN = b"verity/remote-bandwidth/coin/v0\0"
BINDING_DOMAIN = b"verity/remote-bandwidth/coins/v0\0"
COIN_BYTES = 32
HM96 = hm96.SHA512
KEY = HM96.default_key
OWNER = -1
LIMITS = VerificationLimits()


def domain(registration: bytes, session: int, count: int) -> CommitmentDomain:
    if type(registration) is not bytes or len(registration) != 64:
        raise ValueError("a registration digest is 64 bytes")
    if type(session) is not int or not 0 <= session < 1 << 64 or type(count) is not int or count < 1:
        raise ValueError("a session is a u64 and its coins at least one")
    binding = hashlib.sha512(BINDING_DOMAIN + registration + session.to_bytes(8, "big")).digest()
    return CommitmentDomain(binding, OWNER, RangeIndexedDomain(0, count), hash="sha512")


def leaf_value(coin: bytes, salt: bytes) -> bytes:
    """`value_k`: the hm96-sha512 leaf over the coin's digest under `salt`."""
    return HM96.tree_leaf(KEY, HM96.commit_string(KEY, hashlib.sha512(COIN_DOMAIN + coin).digest(), salt))


@dataclass(frozen=True, slots=True)
class CoinCommitment:
    """What the challenger sends before the session's first coin: the session, its `count` coins and their root."""

    registration: bytes
    session: int
    count: int
    root: bytes

    def to_json(self) -> dict:
        return {"count": self.count, "registration": self.registration.hex(), "root": self.root.hex(),
                "scheme": SCHEME, "session": self.session}


@dataclass(frozen=True, slots=True)
class CoinOpening:
    """Coin `index` of a session, released: the coin, its hm96 salt and its leaf's path to the root."""

    index: int
    coin: bytes
    salt: bytes
    path: tuple[bytes, ...]


def coin_reason(commitment: CoinCommitment, opening: CoinOpening, index: int) -> str | None:
    """Why `opening` is not coin `index` of `commitment`'s session; None when it is."""
    if type(index) is not int or type(opening) is not CoinOpening or opening.index != index:
        return f"not coin {index}"
    if (type(opening.coin) is not bytes or len(opening.coin) != COIN_BYTES or type(opening.salt) is not bytes
            or len(opening.salt) != HM96.salt_bytes or type(opening.path) is not tuple
            or any(type(x) is not bytes for x in opening.path) or not 0 <= index < commitment.count):
        return "a malformed coin opening"
    d = domain(commitment.registration, commitment.session, commitment.count)
    if not verify_opening(d, Commitment(commitment.root, commitment.count),
                          Opening(index, leaf_value(opening.coin, opening.salt), opening.path), SCHEME, LIMITS):
        return "the coin does not open against the session's coin commitment"
    return None


class Challenger:
    """The challenger's side of one session's coins: it draws every coin and salt from `entropy` (the operating system's
    generator by default) and commits to them at construction, before the session; `open` releases a coin, once."""

    def __init__(self, registration: bytes, session: int, count: int,
                 entropy: Callable[[int], bytes] = os.urandom) -> None:
        def draw(n: int) -> bytes:
            out = entropy(n)
            if type(out) is not bytes or len(out) != n:
                raise ValueError(f"entropy({n}) returned other than {n} bytes")
            return out

        d = domain(registration, session, count)
        self._coins = tuple((draw(COIN_BYTES), draw(HM96.salt_bytes)) for _ in range(count))
        self._tree = MerkleTree(d, {k: leaf_value(*c) for k, c in enumerate(self._coins)}, lambda _k: SCHEME)
        self.commitment = CoinCommitment(registration, session, count, self._tree.commitment.root)
        self._opened: set[int] = set()

    def open(self, index: int) -> CoinOpening:
        if type(index) is not int or not 0 <= index < self.commitment.count:
            raise ValueError(f"the session has coins 0 … {self.commitment.count - 1}")
        if index in self._opened:
            raise ValueError(f"coin {index} is already open: a retry never redraws")
        self._opened.add(index)
        coin, salt = self._coins[index]
        return CoinOpening(index, coin, salt, self._tree.open(index).path)
