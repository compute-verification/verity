"""The keyed coin tree, ``coin-tree/hm96-sha512/v2`` (``verity/ml/flock/live/src/coin_tree.rs``, which this mirrors byte
for byte): the challenger's commitment, at ``Hello``, to every coin of a session, and the check of each round's opening
against it. The challenger's device samples and opens it (`verity.core.protocols.sampled_proofs.challenger`,
`LiveChallenger.sample_tree` and `open_tree`), and a prover's side checks each opening (`check`).

At ``Hello`` the challenger samples a fresh hm96 key ``K`` and every coin of the session, a block of ``block`` coins for
each (stream, round), and commits to all of them under one SHA-512 Merkle root: one keyed hm96 leaf a (stream, round), each
stream a subtree of ``2^rounds_log`` leaves, the streams' tree padded to a power of two. Every SHA-512 input is prefixed by
the session's nonce ``ν``. A leaf commits to its block through a hash chain, ``T_block = H(TAIL)``, ``T_i = H(CHAIN ‖ c_i ‖
T_{i+1})``, the round digest ``x = H(ROUND ‖ u32be(stream) ‖ u32be(round) ‖ T_0)`` and the leaf ``H(LEAF_PREFIX(K) ‖ x ⊕
M(K)·y ‖ H(SALT_PREFIX ‖ y))`` for the leaf's salt ``y``. ``Hello``'s answer is the root's 4 words, then the key's 16; an
opening of a round's first ``n`` coins is the coins, ``T_n``, the salt, then the path (the stream's subtree, then the
streams' tree), as GF(2^128) words. A word is an int, ``lo | hi << 64``, 16 bytes little-endian on the wire.

Sampling reads ``rand`` in Rust's order: the key (its bit 2047 cleared), then per stream and round its block, then its
salt.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable

SCHEME = "coin-tree/hm96-sha512/v2"
SALT_BYTES, KEY_BYTES, NONCE_BYTES = 192, 256, 32
#: the session's own stream: round 0 is the link points sampled at ``Commit``
SESSION = "session"
SALT_TAG = b"verity/hm96-sha512/salt/v1\0"
LEAF_TAG = b"verity/hm96-sha512/leaf/v1\0"
_WINDOW = (1 << 512) - 1


@dataclass(frozen=True)
class Spec:
    """The committed schedule's shape: every stream the session may open, in leaf order; ``2^rounds_log`` rounds a stream;
    at most ``block`` coins a round."""

    streams: tuple[str, ...]
    rounds_log: int
    block: int

    def to_json(self) -> dict:
        return {"block": self.block, "rounds_log": self.rounds_log, "scheme": SCHEME, "streams": list(self.streams)}

    @classmethod
    def of_json(cls, v) -> Spec:
        """The spec a ``coins`` object names (its ``nonce`` aside); ValueError for anything else."""
        if not isinstance(v, dict) or {k: x for k, x in v.items() if k != "nonce"}.keys() != {"block", "rounds_log", "scheme", "streams"}:
            raise ValueError("a coin-tree spec is exactly block, rounds_log, scheme and streams")
        s, b, r = v["streams"], v["block"], v["rounds_log"]
        if v["scheme"] != SCHEME or not (isinstance(s, list) and s and all(isinstance(x, str) for x in s)) \
                or not all(type(x) is int and x >= 0 for x in (b, r)):
            raise ValueError(f"not a {SCHEME} spec")
        return cls(tuple(s), r, b)

    @property
    def streams_log(self) -> int:
        return (len(self.streams) - 1).bit_length()

    def opening_words(self, n: int) -> int:
        """Words of an opening of ``n`` coins: the coins, ``T_n`` (4), the salt (12), the path (4 a node)."""
        return n + 4 + SALT_BYTES // 16 + 4 * (self.rounds_log + self.streams_log)


def _padded128(*parts: bytes) -> bytes:
    b = b"".join(parts)
    return b + bytes(128 - len(b))


def f128_bytes(w: int) -> bytes:
    return w.to_bytes(16, "little")


def words(b: bytes) -> list[int]:
    return [int.from_bytes(b[i:i + 16], "little") for i in range(0, len(b), 16)]


def unwords(ws: list[int]) -> bytes:
    return b"".join(f128_bytes(w) for w in ws)


class Rules:
    """The tree's hashes under a session's nonce and key: every SHA-512 input prefixed by the nonce, v2's tags, and hm96
    leaves under the session key."""

    def __init__(self, nonce: bytes, key: bytes):
        if len(nonce) != NONCE_BYTES or len(key) != KEY_BYTES:
            raise ValueError(f"a coin-tree nonce is {NONCE_BYTES} bytes and a key {KEY_BYTES}")
        self.nonce, self.key = bytes(nonce), bytes(key)
        self._key_int = int.from_bytes(key, "little")
        self._windows: list[int] | None = None
        self._base = hashlib.sha512(self.nonce)
        def tag(name: str) -> bytes:
            return f"verity/coin-tree/{name}/v2\0".encode()

        self.tail, self.chain_tag, self.round_tag, self.pad_tag = tag("tail"), tag("chain"), tag("round"), tag("pad")
        self.salt_prefix = _padded128(SALT_TAG)
        self.leaf_prefix = _padded128(LEAF_TAG, hashlib.sha512(key).digest())
        self._chain = self._prefixed(self.chain_tag)

    def _prefixed(self, *parts: bytes):
        h = self._base.copy()
        for p in parts:
            h.update(p)
        return h

    def h(self, *parts: bytes) -> bytes:
        return self._prefixed(*parts).digest()

    def mask(self, salt: bytes) -> bytes:
        """hm96's ``M(K)·y``: bit ``i`` is ``⊕_j K_{i+j} y_j`` (``i < 512``, ``j < 1536``), bit ``t`` of a byte string bit
        ``t % 8`` of byte ``t / 8``: each set bit ``j`` of ``y`` adds the key's 512-bit window at ``j``."""
        if self._windows is None:
            self._windows = [(self._key_int >> j) & _WINDOW for j in range(8 * SALT_BYTES)]
        w, acc = self._windows, 0
        for i, v in enumerate(salt):
            for b in range(8):
                if v >> b & 1:
                    acc ^= w[8 * i + b]
        return acc.to_bytes(64, "little")

    def link(self, coin: bytes, t: bytes) -> bytes:
        """``T_i`` from ``c_i`` (its 16 bytes) and ``T_{i+1}``."""
        h = self._chain.copy()
        h.update(coin)
        h.update(t)
        return h.digest()

    def chain_end(self) -> bytes:
        return self.h(self.tail)

    def leaf(self, stream: int, rnd: int, t0: bytes, salt: bytes) -> bytes:
        x = self.h(self.round_tag, stream.to_bytes(4, "big"), rnd.to_bytes(4, "big"), t0)
        c = self.h(self.salt_prefix, salt)
        b = bytes(p ^ q for p, q in zip(x, self.mask(salt)))
        return self.h(self.leaf_prefix, b, c)

    def node(self, left: bytes, right: bytes) -> bytes:
        return self.h(left, right)

    def pad(self) -> bytes:
        return self.h(self.pad_tag)


def _chain(rules: Rules, block: bytes, lo: int, hi: int, t: bytes) -> bytes:
    """``T_lo`` of a block whose ``T_hi`` is ``t``."""
    mv = memoryview(block)
    for i in range(hi - 1, lo - 1, -1):
        t = rules.link(mv[16 * i:16 * i + 16], t)
    return t


def _levels(rules: Rules, leaves: list[bytes]) -> list[list[bytes]]:
    out = [leaves]
    while len(out[-1]) > 1:
        lv = out[-1]
        out.append([rules.node(lv[2 * i], lv[2 * i + 1]) for i in range(len(lv) // 2)])
    return out


def _path(levels: list[list[bytes]], j: int) -> list[bytes]:
    out = []
    for lv in levels[:-1]:
        out.append(lv[j ^ 1])
        j //= 2
    return out


class CoinTree:
    """The challenger's side: every coin of the session, their commitment, and the openings. Each block is kept as
    sampled; an opening's ``T_n`` is recomputed from it."""

    def __init__(self, spec: Spec, rules: Rules, blocks: list[list[bytes]], salts: list[list[bytes]],
                 trees: list[list[list[bytes]]], top: list[list[bytes]]):
        self.spec, self.rules, self.blocks, self.salts, self.trees, self.top = spec, rules, blocks, salts, trees, top

    @classmethod
    def sample(cls, spec: Spec, nonce: bytes, rand: Callable[[int], bytes]) -> CoinTree:
        """Sample the session key, then every coin and salt, from ``rand``, and commit under ``nonce``."""
        key = bytearray(rand(KEY_BYTES))
        key[-1] &= 0x7F
        rules = Rules(nonce, bytes(key))
        rounds, end = 1 << spec.rounds_log, rules.chain_end()
        blocks, salts, trees = [], [], []
        for s in range(len(spec.streams)):
            bs, ss, leaves = [], [], []
            for k in range(rounds):
                b, y = rand(16 * spec.block), rand(SALT_BYTES)
                t = _chain(rules, b, 0, spec.block, end)
                bs.append(b)
                ss.append(y)
                leaves.append(rules.leaf(s, k, t, y))
            blocks.append(bs)
            salts.append(ss)
            trees.append(_levels(rules, leaves))
        tops = [t[-1][0] for t in trees]
        tops += [rules.pad()] * ((1 << spec.streams_log) - len(tops))
        return cls(spec, rules, blocks, salts, trees, _levels(rules, tops))

    def root(self) -> bytes:
        return self.top[-1][0]

    @property
    def key(self) -> bytes:
        return self.rules.key

    @property
    def nonce(self) -> bytes:
        return self.rules.nonce

    def answer_words(self) -> list[int]:
        """``Hello``'s answer: the root's 4 words, then the session key's 16."""
        return words(self.root()) + words(self.key)

    def record(self) -> dict:
        """The session record's ``coin_tree`` (`Flock.CoinTree.checkRecord`): the key, the nonce, the root and the spec."""
        return {"key": self.key.hex(), "nonce": self.nonce.hex(), "root": self.root().hex(), "spec": self.spec.to_json()}

    def open(self, stream: int, rnd: int, n: int) -> tuple[list[int], list[int]]:
        """Round ``rnd`` of stream ``stream``'s first ``n`` coins, and their opening words (``Spec.opening_words(n)``)."""
        sp = self.spec
        if not (0 <= stream < len(sp.streams) and 0 <= rnd < 1 << sp.rounds_log and 0 <= n <= sp.block):
            raise ValueError(f"coin tree: round {rnd} of stream {stream} ({n} coins) is outside the committed schedule")
        b = self.blocks[stream][rnd]
        coins = words(b[:16 * n])
        t = _chain(self.rules, b, n, sp.block, self.rules.chain_end())
        path = _path(self.trees[stream], rnd) + _path(self.top, stream)
        return coins, coins + words(t) + words(self.salts[stream][rnd]) + [w for h in path for w in words(h)]


def check(spec: Spec, rules: Rules, root: bytes, stream: int, rnd: int, n: int, ws: list[int]) -> list[int]:
    """The prover's check: ``ws`` opens ``n`` coins of stream ``stream``'s round ``rnd`` under ``root`` and ``rules``; the
    coins, or ValueError."""
    name = spec.streams[stream] if 0 <= stream < len(spec.streams) else "?"
    if len(ws) != spec.opening_words(n) or not 0 <= rnd < 1 << spec.rounds_log or n > spec.block:
        raise ValueError(f"coin opening of {name} round {rnd}: {len(ws)} words, want {spec.opening_words(n)}")
    coins = list(ws[:n])
    t = _chain(rules, unwords(coins), 0, n, unwords(ws[n:n + 4]))
    at = n + 4 + SALT_BYTES // 16
    h = rules.leaf(stream, rnd, t, unwords(ws[n + 4:at]))
    for idx, depth in ((rnd, spec.rounds_log), (stream, spec.streams_log)):
        for _ in range(depth):
            sib = unwords(ws[at:at + 4])
            h = rules.node(h, sib) if idx & 1 == 0 else rules.node(sib, h)
            idx //= 2
            at += 4
    if h != root:
        raise ValueError(f"the challenger's coins for {name} round {rnd} do not open its coin tree")
    return coins


def hello_nonce(hello: str) -> bytes:
    """The nonce a ``Hello`` names (``coins.nonce``: exactly 64 lowercase hex digits), or ValueError."""
    try:
        n = json.loads(hello)["coins"]["nonce"]
    except (ValueError, KeyError, TypeError):
        raise ValueError("Hello carries no coin-tree nonce (coins.nonce)") from None
    if not isinstance(n, str) or len(n) != 2 * NONCE_BYTES or any(c not in "0123456789abcdef" for c in n):
        raise ValueError(f"Hello's coin-tree nonce is not {2 * NONCE_BYTES} lowercase hex digits")
    return bytes.fromhex(n)


def answer_of(ws: list[int]) -> tuple[bytes, bytes]:
    """``Hello``'s answer read as the prover reads it: the root and the session key; ValueError unless exactly 20 words with
    the key's bit 2047 zero."""
    if len(ws) != 4 + KEY_BYTES // 16:
        raise ValueError(f"Hello's answer has {len(ws)} words, not the coin tree's root and session key (20)")
    key = unwords(ws[4:])
    if key[-1] & 0x80:
        raise ValueError("the challenger's session key has bit 2047 set")
    return unwords(ws[:4]), key
