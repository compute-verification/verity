"""A test-only stateful hash-based signer for network certificates; the wheel leaves tests out, so nothing can import it
from the package.

Each one-time key is a Lamport key over 128 bits of SHA-512(I ‖ q ‖ message), and the 2^h one-time keys are the leaves
of a SHA-512 Merkle tree whose root, with I, is the public key. The state is a file holding the next unused leaf q,
and the leaves used are logged apart. The signer refuses when the state file is missing, when q is at or below a
logged leaf (the state is behind, as after a restore from an old copy), or when the leaves run out. It records q as
used before it signs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from verity.experimental.protocols.certifier.certificate import SignerRefused

BITS = 128


def _h(*parts: bytes) -> bytes:
    return hashlib.sha512(b"".join(parts)).digest()


def _sk(seed: bytes, q: int, i: int, b: int) -> bytes:
    return _h(b"toy-sk", seed, q.to_bytes(4, "big"), i.to_bytes(2, "big"), bytes([b]))[:32]


def _pk(sk: bytes) -> bytes:
    return _h(b"toy-pk", sk)[:32]


def _leaf(q: int, pks: list[bytes]) -> bytes:
    return _h(b"toy-leaf", q.to_bytes(4, "big"), *pks)


def _node(left: bytes, right: bytes) -> bytes:
    return _h(b"toy-node", left, right)


def _bits(key_id: bytes, q: int, message: bytes) -> list[int]:
    d = _h(b"toy-msg", key_id, q.to_bytes(4, "big"), message)
    return [(d[i // 8] >> (7 - i % 8)) & 1 for i in range(BITS)]


class ToyKey:
    def __init__(self, key_id: bytes, root: bytes, height: int) -> None:
        self._key_id, self.root, self.height = key_id, root, height

    @property
    def key_id(self) -> bytes:
        return self._key_id

    def verify(self, message: bytes, signature: bytes) -> bool:
        if len(signature) != 4 + 2 * BITS * 32 + 64 * self.height:
            return False
        q = int.from_bytes(signature[:4], "big")
        if q >= 1 << self.height:
            return False
        revealed = [signature[4 + 32 * i: 36 + 32 * i] for i in range(BITS)]
        others = [signature[4 + 32 * (BITS + i): 36 + 32 * (BITS + i)] for i in range(BITS)]
        pks: list[bytes] = []
        for bit, sk, other in zip(_bits(self.key_id, q, message), revealed, others, strict=True):
            pks += [_pk(sk), other] if bit == 0 else [other, _pk(sk)]
        digest, r = _leaf(q, pks), q
        path = signature[4 + 2 * BITS * 32:]
        for lv in range(self.height):
            sibling = path[64 * lv: 64 * (lv + 1)]
            digest = _node(digest, sibling) if r % 2 == 0 else _node(sibling, digest)
            r >>= 1
        return digest == self.root


class ToySigner:
    def __init__(self, seed: bytes, height: int, state: Path, log: Path) -> None:
        self.seed, self.height, self.state, self.log = seed, height, state, log
        self._key_id = _h(b"toy-I", seed)[:16]
        self._levels = [[_leaf(q, self._pks(q)) for q in range(1 << height)]]
        while len(self._levels[-1]) > 1:
            lv = self._levels[-1]
            self._levels.append([_node(lv[i], lv[i + 1]) for i in range(0, len(lv), 2)])

    @classmethod
    def create(cls, directory: Path, seed: bytes = b"toy", height: int = 5) -> ToySigner:
        """A signer with fresh state (next leaf 0) under `directory`."""
        state = directory / "signer-state.json"
        state.write_text(json.dumps({"next": 0}))
        return cls(seed, height, state, directory / "signer-used.log")

    def _pks(self, q: int) -> list[bytes]:
        return [_pk(_sk(self.seed, q, i, b)) for i in range(BITS) for b in (0, 1)]

    @property
    def key_id(self) -> bytes:
        return self._key_id

    def public_key(self) -> ToyKey:
        return ToyKey(self._key_id, self._levels[-1][0], self.height)

    def used(self) -> list[int]:
        return [int(line) for line in self.log.read_text().split()] if self.log.exists() else []

    def sign(self, message: bytes) -> bytes:
        if not self.state.exists():
            raise SignerRefused("the one-time-key state is missing")
        q = json.loads(self.state.read_text())["next"]
        used = self.used()
        if used and q <= max(used):
            raise SignerRefused(f"the one-time-key state is behind: leaf {q}, but leaf {max(used)} was used")
        if q >= 1 << self.height:
            raise SignerRefused("every one-time key is used")
        self.state.write_text(json.dumps({"next": q + 1}))
        with self.log.open("a") as fh:
            fh.write(f"{q}\n")
        bits = _bits(self._key_id, q, message)
        revealed = [_sk(self.seed, q, i, b) for i, b in enumerate(bits)]
        others = [_pk(_sk(self.seed, q, i, 1 - b)) for i, b in enumerate(bits)]
        path, r = [], q
        for lv in self._levels[:-1]:
            path.append(lv[r ^ 1])
            r >>= 1
        return q.to_bytes(4, "big") + b"".join(revealed) + b"".join(others) + b"".join(path)
