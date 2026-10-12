"""The receipts of one pass, the seed its live coin gives it, the plan of what it reads, and the tree that commits them.

~~~text
Z(x)        x zero-padded to whole 128-byte blocks (one SHA-512 block here)
seed_k      SHA512(SEED_DOMAIN ‖ c_k ‖ u64be k),  SEED_DOMAIN = "verity/remote-bandwidth/seed/v0\0", c_k pass k's live
            coin (`coins.py`), k the pass's index in the session
plan_k      R runs (first, count) in read order: count ≥ 1, inside the store, pairwise disjoint, and no run starting
            where the one before it ends (it would be the same run).  Position p is the p-th block of the runs
            concatenated, L_k positions in all.
            digest SHA512(PLAN_DOMAIN ‖ u32be R ‖ (u64be first ‖ u64be count) × R),  PLAN_DOMAIN = "…/plan/v0\0"
spread(z)   z laid over b = 4096 bytes: byte u of z XORed into byte u mod b (a z of b bytes or more folds), a shorter
            z repeated
y_(k,p)     SHA512(Z(RECEIPT_DOMAIN ‖ frame(seed_k)) ‖ u64be j ‖ (E_j ⊕ spread(z_(k,j)))),  j = plan_k[p],
            RECEIPT_DOMAIN = "verity/remote-bandwidth/receipt/v0\0"                                     64 bytes
leaf c      v_c = y_(16c) ‖ … ‖ y_(16c+15), positions in plan order, the last leaf short; committed as an
            hm96-sha512/row/v2 leaf: x_c = sha512/row/v2(v_c), value hm96-sha512 tree_leaf(key, commit_string(key, x_c,
            salt_c)), salt_c 192 fresh bytes, key hm96-sha512's pinned key
root_k      frame-v3-sha512 over the leaves: CommitmentDomain(binding = SHA512(TREE_DOMAIN ‖ seed_k ‖ plan digest),
            owner 0, positions 0 … ⌈L_k/16⌉ − 1), schema "hm96-sha512/row/v2",  TREE_DOMAIN = "…/receipt-tree/v0\0"
~~~

`frame` is `verity.ml.randomness.frame`.  A receipt is seed-first: the seed fills the first SHA-512 block, so no state
over a block's bytes exists before the seed, and the seed exists only once its coin has been released.  `z_(k,j)` is
PoUW's noised output of the tile before block j's read in pass k; it enters elementwise, before any compression, so a
relay that meets the block and z over a link carries `min(|z|, b)` bytes per block, not a digest of one side.

A pass's plan is public: the prover sends it with the root, and the root binds its digest.  Its runs are disjoint so
that one read of a block serves at most one receipt of the pass; a plan that repeats a block is refused.  Its order is
the tree's, not necessarily the reads': listed in store order, a plan is one run per contiguous range it reads.
"""

from __future__ import annotations

import bisect
import hashlib
import itertools
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from verity.ml.commitments import Commitment, CommitmentDomain, MerkleTree, Opening, RangeIndexedDomain, VerificationLimits
from verity.ml.commitments import hm96, verify_opening
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2, sha512_row_v2_digest
from verity.ml.randomness import frame

from .codec import BLOCK_BYTES

SEED_DOMAIN = b"verity/remote-bandwidth/seed/v0\0"
PLAN_DOMAIN = b"verity/remote-bandwidth/plan/v0\0"
RECEIPT_DOMAIN = b"verity/remote-bandwidth/receipt/v0\0"
TREE_DOMAIN = b"verity/remote-bandwidth/receipt-tree/v0\0"
SCHEMA = SCHEMA_HM96_SHA512_ROW_V2
COIN_BYTES = 32
DIGEST_BYTES = 64
PER_LEAF = 16
OWNER = 0
HM96 = hm96.SHA512
KEY = HM96.default_key
SALT_BYTES = HM96.salt_bytes
LIMITS = VerificationLimits()


def _digest(value: bytes, name: str) -> bytes:
    if type(value) is not bytes or len(value) != DIGEST_BYTES:
        raise ValueError(f"{name} is {DIGEST_BYTES} bytes")
    return value


def seed(coin: bytes, k: int) -> bytes:
    """`seed_k`: pass `k`'s seed under its live coin."""
    if type(coin) is not bytes or len(coin) != COIN_BYTES:
        raise ValueError(f"a coin is {COIN_BYTES} bytes")
    if type(k) is not int or not 0 <= k < 1 << 64:
        raise ValueError("a pass index is a u64")
    return hashlib.sha512(SEED_DOMAIN + coin + k.to_bytes(8, "big")).digest()


@dataclass(frozen=True, slots=True)
class Plan:
    """What one pass reads, in read order, in a store of `blocks` blocks: `runs` of `(first, count)` consecutive blocks.
    Construction refuses any plan but a canonical one of distinct blocks (the format above), so equal read sequences
    have equal digests and no block is read twice."""

    runs: tuple[tuple[int, int], ...]
    blocks: int
    ends: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.blocks) is not int or self.blocks < 1:
            raise ValueError("a store has at least one block")
        runs = tuple(self.runs) if isinstance(self.runs, Sequence) else None
        if runs is None or not 1 <= len(runs) < 1 << 32:
            raise ValueError("a plan is 1 to 2^32 − 1 runs")
        for r in runs:
            if (type(r) is not tuple or len(r) != 2 or any(type(x) is not int for x in r) or r[0] < 0 or r[1] < 1
                    or r[0] + r[1] > self.blocks):
                raise ValueError("a run is (first, count) with count ≥ 1, inside the store")
        if any(a[0] + a[1] == b[0] for a, b in zip(runs, runs[1:])):
            raise ValueError("a run that starts where the one before it ends is part of it")
        ordered = sorted(runs)
        if any(a[0] + a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
            raise ValueError("a plan's runs are disjoint: no block is read twice in a pass")
        object.__setattr__(self, "runs", runs)
        object.__setattr__(self, "ends", tuple(itertools.accumulate(c for _, c in runs)))

    @classmethod
    def whole(cls, blocks: int) -> Plan:
        """Every block, in store order."""
        return cls(((0, blocks),), blocks)

    @classmethod
    def of(cls, order: Sequence[int], blocks: int) -> Plan:
        """The plan that reads `order`, block indices in read order (the prover-side registry's form)."""
        runs: list[tuple[int, int]] = []
        for j in order:
            if runs and type(j) is int and runs[-1][0] + runs[-1][1] == j:
                runs[-1] = (runs[-1][0], runs[-1][1] + 1)
            else:
                runs.append((j, 1))
        return cls(tuple(runs), blocks)

    @property
    def length(self) -> int:
        """L, the pass's positions."""
        return self.ends[-1]

    def block(self, p: int) -> int:
        """The block at position `p`."""
        if type(p) is not int or not 0 <= p < self.length:
            raise IndexError(f"a plan of {self.length} positions has no position {p!r}")
        r = bisect.bisect_right(self.ends, p)
        return self.runs[r][0] + p - (self.ends[r - 1] if r else 0)

    def __iter__(self) -> Iterator[int]:
        for first, count in self.runs:
            yield from range(first, first + count)

    def encode(self) -> bytes:
        return len(self.runs).to_bytes(4, "big") + b"".join(f.to_bytes(8, "big") + c.to_bytes(8, "big")
                                                            for f, c in self.runs)

    @property
    def digest(self) -> bytes:
        return hashlib.sha512(PLAN_DOMAIN + self.encode()).digest()


def prefix(seed: bytes) -> bytes:
    """`Z(RECEIPT_DOMAIN ‖ frame(seed))`, one SHA-512 block."""
    body = RECEIPT_DOMAIN + frame(_digest(seed, "a pass seed"))
    return body + bytes(-len(body) % 128)


def spread(z: bytes, b: int = BLOCK_BYTES) -> bytes:
    """`z` laid over `b` bytes: folded by XOR when it is `b` bytes or longer, repeated when it is shorter."""
    if type(z) is not bytes or not z:
        raise ValueError("a tile's output is nonempty bytes")
    if len(z) < b:
        return (z * -(-b // len(z)))[:b]
    out = int.from_bytes(z[:b], "little")
    for u in range(b, len(z), b):
        out ^= int.from_bytes(z[u:u + b], "little")
    return out.to_bytes(b, "little")


def _xor(a: bytes, b: bytes) -> bytes:
    return (int.from_bytes(a, "little") ^ int.from_bytes(b, "little")).to_bytes(len(a), "little")


class Receipter:
    """Receipts under one pass's seed: `receipter(j, block, z)` is `y_(k,j)`."""

    def __init__(self, seed: bytes) -> None:
        self.head = hashlib.sha512(prefix(seed))

    def __call__(self, j: int, block: bytes, z: bytes) -> bytes:
        if type(j) is not int or not 0 <= j < 1 << 64:
            raise ValueError("a block index is a u64")
        if type(block) is not bytes or len(block) != BLOCK_BYTES:
            raise ValueError(f"a block is {BLOCK_BYTES} bytes")
        h = self.head.copy()
        h.update(j.to_bytes(8, "big"))
        h.update(_xor(block, spread(z)))
        return h.digest()


def leaves(positions: int) -> int:
    return math.ceil(positions / PER_LEAF)


def binding(seed: bytes, plan: Plan) -> bytes:
    """The receipt tree's binding: the pass's seed and its plan's digest."""
    return hashlib.sha512(TREE_DOMAIN + _digest(seed, "a pass seed") + plan.digest).digest()


def domain(seed: bytes, plan: Plan) -> CommitmentDomain:
    """The domain of the receipt tree of the pass with seed `seed` and plan `plan`."""
    return CommitmentDomain(binding(seed, plan), OWNER, RangeIndexedDomain(0, leaves(plan.length)), hash="sha512")


def hidden(row: bytes, salt: bytes) -> bytes:
    """`row`'s committed value as an hm96-sha512/row/v2 leaf: the hm96-sha512 leaf over its sha512/row/v2 digest under
    `salt` (a receipt leaf's receipts, or a payload block)."""
    return HM96.tree_leaf(KEY, HM96.commit_string(KEY, sha512_row_v2_digest(row, 8 * len(row)), salt))


def tree(seed: bytes, plan: Plan, ys: Sequence[bytes], salts: Sequence[bytes]) -> MerkleTree:
    """The receipt tree of one pass: `ys` in plan order, sixteen to a leaf, leaf c hidden under `salts[c]`."""
    if len(ys) != plan.length or len(salts) != leaves(plan.length):
        raise ValueError(f"the plan has {plan.length} positions in {leaves(plan.length)} leaves, not {len(ys)} in "
                         f"{len(salts)}")
    values = {c: hidden(b"".join(ys[PER_LEAF * c:PER_LEAF * (c + 1)]), salts[c]) for c in range(len(salts))}
    return MerkleTree(domain(seed, plan), values, lambda _c: SCHEMA)


def leaf_receipt(leaf: bytes, p: int, positions: int) -> bytes | None:
    """Position `p`'s receipt inside its leaf's receipts, or None if they are not that leaf's size."""
    c = p // PER_LEAF
    if len(leaf) != DIGEST_BYTES * min(PER_LEAF, positions - PER_LEAF * c):
        return None
    k = p - PER_LEAF * c
    return leaf[DIGEST_BYTES * k:DIGEST_BYTES * (k + 1)]


def verify_leaf(seed: bytes, plan: Plan, root: bytes, p: int, leaf: bytes, salt: bytes, path: tuple[bytes, ...]) -> bool:
    """Whether `leaf`, the receipts of position `p`'s leaf, under `salt` and with `path`, is that leaf of the tree whose
    root is `root`."""
    d = domain(seed, plan)
    return verify_opening(d, Commitment(root, d.count), Opening(p // PER_LEAF, hidden(leaf, salt), path), SCHEMA, LIMITS)
