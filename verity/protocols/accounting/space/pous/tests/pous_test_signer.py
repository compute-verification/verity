"""A TEST signer for the memory challenger's sessions: XMSS-style, Winternitz one-time keys (w = 16) under a SHA-512
Merkle tree, stateful.  It stands in for S1's scheme in tests only.  Its scheme id starts with
`memory_challenger.TEST_PREFIX`, which `memory_challenger.register` refuses, so a record it signs is accepted only by a
test's own mapping (`SCHEMES`), never by `memory_challenger.SCHEMES`.

~~~text
d      = SHA-512(MSG ‖ pub_seed ‖ u32 leaf ‖ message)            128 base-16 digits, then a 3-digit checksum: 131 chains
sk_c   = SHA-512(SK ‖ secret ‖ u32 leaf ‖ u32 c)
F(x)   = SHA-512(CHAIN ‖ pub_seed ‖ u32 leaf ‖ u32 c ‖ u32 step ‖ x)   one step of chain c
leaf   = SHA-512(LEAF ‖ pub_seed ‖ u32 leaf ‖ every chain's end)
node   = SHA-512(NODE ‖ pub_seed ‖ u32 level ‖ u32 index ‖ left ‖ right)
pk     = pub_seed ‖ root          signature = u32 leaf ‖ 131 chain values ‖ the leaf's path
~~~
"""
from __future__ import annotations

import hashlib

from verity.protocols.accounting.space.pous.memory_challenger import TEST_PREFIX

SCHEME = TEST_PREFIX + "xmss-wots16-sha512/v0"
N, W, DIGITS, CHECKSUM = 64, 16, 128, 3
CHAINS = DIGITS + CHECKSUM
MSG, SK, CHAIN, LEAF, NODE = (f"verity-pous/test-signer/{t}/v0\0".encode() for t in ("msg", "sk", "chain", "leaf", "node"))


def _u32(x: int) -> bytes:
    return x.to_bytes(4, "big")


def _h(*parts: bytes) -> bytes:
    return hashlib.sha512(b"".join(parts)).digest()


def _digits(pub_seed: bytes, leaf: int, message: bytes) -> list[int]:
    d = _h(MSG, pub_seed, _u32(leaf), message)
    digits = [x for b in d for x in (b >> 4, b & 15)]
    total = sum(W - 1 - x for x in digits)
    return digits + [(total >> (4 * i)) & 15 for i in reversed(range(CHECKSUM))]


def _chain(pub_seed: bytes, leaf: int, c: int, x: bytes, start: int, steps: int) -> bytes:
    for s in range(start, start + steps):
        x = _h(CHAIN, pub_seed, _u32(leaf), _u32(c), _u32(s), x)
    return x


def _node(pub_seed: bytes, level: int, index: int, left: bytes, right: bytes) -> bytes:
    return _h(NODE, pub_seed, _u32(level), _u32(index), left, right)


class TestSigner:
    """2^`height` one-time keys from `secret`; each `sign` uses the next, and one past the last is refused."""

    __test__ = False
    scheme = SCHEME

    def __init__(self, secret: bytes, height: int = 3, pub_seed: bytes | None = None) -> None:
        self.secret, self.height = secret, height
        self.pub_seed = pub_seed or _h(b"pub", secret)
        self.next = 0
        level = [self._leaf(i) for i in range(1 << height)]
        self.levels = [level]
        for lv in range(height):
            level = [_node(self.pub_seed, lv, i // 2, level[i], level[i + 1]) for i in range(0, len(level), 2)]
            self.levels.append(level)
        self.public_key = self.pub_seed + self.levels[-1][0]

    def _sk(self, leaf: int, c: int) -> bytes:
        return _h(SK, self.secret, _u32(leaf), _u32(c))

    def _leaf(self, leaf: int) -> bytes:
        ends = b"".join(_chain(self.pub_seed, leaf, c, self._sk(leaf, c), 0, W - 1) for c in range(CHAINS))
        return _h(LEAF, self.pub_seed, _u32(leaf), ends)

    def sign(self, message: bytes) -> bytes:
        leaf = self.next
        if leaf >= 1 << self.height:
            raise RuntimeError("every one-time key is used")
        self.next += 1
        sig = b"".join(_chain(self.pub_seed, leaf, c, self._sk(leaf, c), 0, d)
                       for c, d in enumerate(_digits(self.pub_seed, leaf, message)))
        path, i = b"", leaf
        for level in self.levels[:-1]:
            path += level[i ^ 1]
            i >>= 1
        return _u32(leaf) + sig + path


def verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
    if len(public_key) != 2 * N or (len(signature) - 4 - CHAINS * N) % N:
        return False
    pub_seed, root = public_key[:N], public_key[N:]
    leaf = int.from_bytes(signature[:4], "big")
    height = (len(signature) - 4 - CHAINS * N) // N
    if leaf >= 1 << height:
        return False
    body = signature[4:4 + CHAINS * N]
    ends = b"".join(_chain(pub_seed, leaf, c, body[c * N:(c + 1) * N], d, W - 1 - d)
                    for c, d in enumerate(_digits(pub_seed, leaf, message)))
    node, i = _h(LEAF, pub_seed, _u32(leaf), ends), leaf
    for lv in range(height):
        sib = signature[4 + CHAINS * N + lv * N:4 + CHAINS * N + (lv + 1) * N]
        node = _node(pub_seed, lv, i // 2, *((node, sib) if i % 2 == 0 else (sib, node)))
        i >>= 1
    return node == root


SCHEMES = {SCHEME: verify}
"""The mapping a test passes to `memory_challenger.verdict`: the test signer accepted, in that test alone."""
