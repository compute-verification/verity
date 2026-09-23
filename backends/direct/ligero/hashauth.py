"""``authentication = included-hash``: the committed-object side of the self-hashing relation (``hashchain.py``).

Three indexed Merkle trees of :mod:`verity.commitments` (the framing, domains and multiproofs of ``auth.py``, lane
auth-included, unchanged), with the operand trees now holding one **row** per leaf (``leaf/v2h``,
:mod:`verity.commitments.rowleaf`): tree ``a`` commits the activation rows (one x row of K words = one leaf: the row's
Poseidon2 digest, 8 BabyBear lanes as 32 big-endian bytes, schema ``poseidon2-babybear-w24/row/v2h``), tree ``b`` the
weight columns (one W column of K words = one leaf), tree ``y`` the output words (``auth.py``'s word leaves, 2 or 4 bytes
big-endian by the relation's ``y_bits``).  The verifier never hashes an operand word: the relation proved that the
published digests are Poseidon2 of the rows it computed on, and this module checks that those digests are the leaves
at the claimed ranks under the roots (SHA-256 multiproofs, ~depth siblings per distinct row).

Statement block (:class:`HashAuth`): per VU the ``y`` leaf rank (the VU index), the ``a`` leaf rank (which x row) and the
``b`` leaf rank (which W column) -- a 64 x 64 GEMM tile is 4096 VUs over 64 + 64 distinct rows -- and the three tree
references.  ``digest_bytes`` of the block is absorbed into the Ligero statement digest through the relation hooks' tag
(``relchain.HashedRelationRunner``), so every challenge is bound to "these rows under these roots".
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from verity.commitments import identity_digest, verify_multiproof
from verity.commitments.rowleaf import SCHEMA_ROW, row_leaf_value

from . import poseidon2 as h2
from .auth import LIMITS, OWNER, TREE_NAMES, TreeRef, WordTree, _chunks, _node_chunk, _pad_chunk, domain_for, multiproof_siblings
from .auth import build_word_tree as _build_word_tree_u16
from verity.errors import ResourceLimit

__all__ = ["BINDING_TAG", "HASH_AUTH_VERSION", "POSEIDON2", "HashAuth", "LeafParams", "binding_digest", "build_row_tree",
           "build_trees", "digest_bytes", "leaf_params", "load_or_build_trees", "open_trees", "verify_hash_auth", "word_schema"]

HASH_AUTH_VERSION = "ligero-b/auth/v2h"
BINDING_TAG = "verity/ligero-b/auth-binding/v2h"


@dataclass(frozen=True)
class LeafParams:
    """What the tree side needs of a row-digest scheme: the leaf schema string, the 32-byte parameter digest the auth
    block carries, the digest width, and the two functions (rows -> digests; digest -> leaf value bytes)."""

    schema: str
    params_sha256: str          # hex, 32 bytes
    lanes: int                  # BabyBear elements per row digest
    native: object              # (rows (R, n_words) int64, word_bits, role) -> (R, lanes) int64
    leaf_bytes: object          # digest row -> bytes

    @property
    def n_digest(self) -> int:
        return 2 * self.lanes


POSEIDON2 = LeafParams(SCHEMA_ROW, h2.CONSTANTS_SHA256, h2.OUT, h2.hash_rows_np, row_leaf_value)


def leaf_params(leaf=None, device=None) -> LeafParams:
    """``None`` -> Poseidon2 (today's tree, byte for byte); a ``leaf/`` scheme object -> its parameters.  ``device``: where
    the scheme's ``native`` digests the committed rows (a torch device string; the tree itself is host SHA-256)."""
    if leaf is None:
        return POSEIDON2
    if isinstance(leaf, LeafParams):
        return leaf

    def native(rows, word_bits, role):
        if device is not None and str(device) != "cpu":
            import torch
            t = torch.as_tensor(np.ascontiguousarray(rows, dtype=np.int64), device=device)
            return leaf.native(t, word_bits=word_bits, role=role).cpu().numpy().astype(np.int64)
        return np.asarray(leaf.native(rows, word_bits=word_bits, role=role), dtype=np.int64)

    return LeafParams(leaf.schema, leaf.params_sha256, leaf.digest_elems, native, leaf.leaf_bytes)


def word_schema(y_bits: int) -> tuple[str, int]:
    if y_bits <= 16:
        return "u16", 2
    if y_bits <= 32:
        return "u32", 4
    raise ValueError("output words wider than 32 bits")


def binding_digest(*, dataset: str, tier: str, manifest_sha256: str, lo: int, hi: int, K: int, tree: str, schema: str) -> bytes:
    """Identity of the committed object: the frozen instance set (by manifest digest), which rows of it, the leaf schema."""
    if tree not in TREE_NAMES:
        raise ValueError(f"unknown tree {tree!r}")
    h = identity_digest(BINDING_TAG, {"dataset": dataset, "tier": tier, "manifest_sha256": manifest_sha256,
                                      "lo": int(lo), "hi": int(hi), "K": int(K), "tree": tree, "schema": schema})
    return bytes.fromhex(h)


# ---------------------------------------------------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------------------------------------------------

def _levels(domain, leaves: list[bytes], count: int) -> list[np.ndarray]:
    from verity.commitments import merkle_depth
    depth = merkle_depth(count)
    width = 1 << depth
    pads = b"".join(_pad_chunk((domain.domain_id, s + count, e + count)) for s, e in _chunks(width - count, 1 << 16))
    level = np.frombuffer(b"".join(leaves) + pads, dtype=np.uint8).reshape(width, 32).copy()
    levels = [level]
    for lv in range(depth):
        pairs = level.reshape(-1, 64)
        level = np.frombuffer(b"".join(_node_chunk((domain, lv, s, pairs[s:e])) for s, e in _chunks(pairs.shape[0], 1 << 16)),
                              dtype=np.uint8).reshape(-1, 32).copy()
        levels.append(level)
    return levels


def build_row_tree(name: str, binding: bytes, rows: np.ndarray, word_bits: int, role: int,
                   leaf: LeafParams = POSEIDON2) -> tuple[WordTree, np.ndarray]:
    """Commit ``rows`` (count, n_words) of ``word_bits``-bit words as ``leaf/v2h`` leaves: the ``leaf`` digest (Poseidon2
    by default) of each row (``role``: ``h2.ROLE_X`` / ``ROLE_W``, the sponge's domain tag) under the SHA-256 tree.
    Returns the tree and the digests (count, leaf.lanes)."""
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    count = int(rows.shape[0])
    LIMITS.enforce("max_positions", count)
    domain = domain_for(binding, OWNER[name], count)
    digests = leaf.native(rows, word_bits, role)
    leaves = [domain.leaf(r, r, leaf.schema, leaf.leaf_bytes(digests[r])) for r in range(count)]
    levels = _levels(domain, leaves, count)
    ref = TreeRef(name, bytes(binding), OWNER[name], count, levels[-1][0].tobytes())
    return WordTree(ref, levels), digests


def build_word_tree(name: str, binding: bytes, words: np.ndarray, y_bits: int) -> WordTree:
    """The output-word tree: one word per leaf, 2 or 4 big-endian bytes by ``y_bits`` (``auth.py``'s tree for 16 bits)."""
    schema, nbytes = word_schema(y_bits)
    if nbytes == 2:
        return _build_word_tree_u16(name, binding, words, procs=1)
    words = np.ascontiguousarray(words, dtype=np.int64).ravel()
    count = int(words.shape[0])
    domain = domain_for(binding, OWNER[name], count)
    leaves = [domain.leaf(r, r, schema, int(w).to_bytes(nbytes, "big")) for r, w in enumerate(words.tolist())]
    levels = _levels(domain, leaves, count)
    return WordTree(TreeRef(name, bytes(binding), OWNER[name], count, levels[-1][0].tobytes()), levels)


def build_trees(*, x_rows: np.ndarray, w_cols: np.ndarray, y_words: np.ndarray, word_bits: int, y_bits: int,
                bindings: Mapping[str, bytes], leaf: LeafParams = POSEIDON2) -> tuple[dict[str, WordTree], dict[str, np.ndarray]]:
    """The three trees of a committed instance set; also the operand digests (what the honest prover will publish)."""
    ta, da = build_row_tree("a", bindings["a"], x_rows, word_bits, h2.ROLE_X, leaf)
    tb, db = build_row_tree("b", bindings["b"], w_cols, word_bits, h2.ROLE_W, leaf)
    ty = build_word_tree("y", bindings["y"], y_words, y_bits)
    return {"a": ta, "b": tb, "y": ty}, {"a": da, "b": db}


def load_or_build_trees(cache: Path | None, **kw) -> tuple[dict[str, WordTree], dict[str, np.ndarray], bool]:
    bindings = kw["bindings"]
    lanes = kw.get("leaf", POSEIDON2).lanes
    if cache is not None:
        cache = Path(cache)
        try:
            trees = {n: WordTree.load(cache, n) for n in TREE_NAMES}
            if all(trees[n].ref.binding == bytes(bindings[n]) for n in TREE_NAMES):
                dig = {n: np.load(cache / f"{n}.digests.npy") for n in ("a", "b")}
                if all(dig[n].ndim == 2 and dig[n].shape[1] == lanes for n in ("a", "b")):
                    return trees, dig, True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    trees, dig = build_trees(**kw)
    if cache is not None:
        for t in trees.values():
            t.save(cache)
        for n in ("a", "b"):
            np.save(cache / f"{n}.digests.npy", dig[n])
    return trees, dig, False


# ---------------------------------------------------------------------------------------------------------------------
# Statement block, openings, verification
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class HashAuth:
    """Per VU: its ``y`` leaf (the VU index, strictly increasing), the ``a`` leaf (x row) and ``b`` leaf (W column) it used."""

    vu_index: tuple[int, ...]
    x_index: tuple[int, ...]
    w_index: tuple[int, ...]
    trees: tuple[TreeRef, TreeRef, TreeRef]

    def __post_init__(self):
        if tuple(t.name for t in self.trees) != TREE_NAMES:
            raise ValueError("trees must be given in (a, b, y) order")
        n = len(self.vu_index)
        if len(self.x_index) != n or len(self.w_index) != n:
            raise ValueError("one (x, W, y) leaf triple per VU")
        idx = self.vu_index
        if any(idx[i] >= idx[i + 1] for i in range(n - 1)) or any(v < 0 for v in idx):
            raise ValueError("vu_index must be strictly increasing and non-negative")
        if any(v < 0 for v in self.x_index) or any(v < 0 for v in self.w_index):
            raise ValueError("leaf ranks must be non-negative")

    @property
    def n_vus(self) -> int:
        return len(self.vu_index)

    def tree(self, name: str) -> TreeRef:
        return self.trees[TREE_NAMES.index(name)]

    def to_json(self) -> dict:
        return {"version": HASH_AUTH_VERSION, "schema": SCHEMA_ROW, "vu_index": list(self.vu_index),
                "x_index": list(self.x_index), "w_index": list(self.w_index), "trees": [t.to_json() for t in self.trees]}

    @classmethod
    def from_json(cls, d: Mapping) -> HashAuth:
        if d.get("version", HASH_AUTH_VERSION) != HASH_AUTH_VERSION:
            raise ValueError(f"unknown authentication version {d.get('version')!r}")
        return cls(tuple(int(v) for v in d["vu_index"]), tuple(int(v) for v in d["x_index"]),
                   tuple(int(v) for v in d["w_index"]), tuple(TreeRef.from_json(t) for t in d["trees"]))

    @classmethod
    def for_vus(cls, trees: Mapping[str, WordTree | TreeRef], vu_index: Iterable[int], x_index: Iterable[int],
                w_index: Iterable[int]) -> HashAuth:
        refs = tuple(trees[n].ref if isinstance(trees[n], WordTree) else trees[n] for n in TREE_NAMES)
        return cls(tuple(int(v) for v in vu_index), tuple(int(v) for v in x_index), tuple(int(v) for v in w_index), refs)

    def indices(self) -> np.ndarray:
        """(n_vus, 3) ``<u4``: what the Ligero statement's ``b`` slot carries (``hashchain.hash_public_vectors``)."""
        return np.stack([np.asarray(self.vu_index), np.asarray(self.x_index), np.asarray(self.w_index)], 1).astype(np.int64)


def digest_bytes(auth: HashAuth, leaf: LeafParams = POSEIDON2) -> bytes:
    """The auth block's contribution to the statement digest (the relation hooks' tag).  Poseidon2's bytes are those of
    the lane hash-relation format; another leaf changes the schema string and the parameter digest after ``|p2=``."""
    out = bytearray(b"|auth=" + HASH_AUTH_VERSION.encode() + b"|schema=" + leaf.schema.encode() + b"|p2=" +
                    bytes.fromhex(leaf.params_sha256) + b"|idx=")
    out += auth.indices().astype("<u4").tobytes()
    for t in auth.trees:
        out += b"|" + t.name.encode() + b":binding=" + t.binding + b"|owner=" + str(t.owner).encode()
        out += b"|count=" + str(t.count).encode() + b"|root=" + t.root
    return bytes(out)


def _ranks(auth: HashAuth) -> dict[str, np.ndarray]:
    return {"a": np.unique(np.asarray(auth.x_index, dtype=np.int64)), "b": np.unique(np.asarray(auth.w_index, dtype=np.int64)),
            "y": np.asarray(auth.vu_index, dtype=np.int64)}


def open_trees(trees: Mapping[str, WordTree], auth: HashAuth) -> list[list[bytes]]:
    """The prover's three multiproofs (in TREE_NAMES order): the distinct x rows, the distinct W columns, the y words."""
    r = _ranks(auth)
    return [multiproof_siblings(trees[n], r[n]) for n in TREE_NAMES]


def verify_hash_auth(auth: HashAuth, siblings: Sequence[Sequence[bytes]], *, digests: np.ndarray, y_words: np.ndarray,
                     y_bits: int, leaf: LeafParams = POSEIDON2) -> tuple[bool, str]:
    """``digests`` (n_vus, 2 * leaf.lanes): the x digest lanes then the W digest lanes every VU published (the relation
    proved them); ``y_words`` (n_vus,).  Every VU's leaves must open under the roots; two VUs claiming the same leaf must
    have published the same digest.  Every failure is a rejection (no exceptions)."""
    digests = np.asarray(digests, dtype=np.int64)
    y_words = np.asarray(y_words, dtype=np.int64).ravel()
    n = auth.n_vus
    lanes = leaf.lanes
    if digests.shape != (n, 2 * lanes) or y_words.shape != (n,):
        return False, "auth: digest / word table does not match the VU count"
    if len(siblings) != len(TREE_NAMES):
        return False, "auth: expected one multiproof per tree"
    if ((digests < 0) | (digests >= h2.P)).any():
        return False, "auth: digest lane out of the field"
    yschema, ybytes = word_schema(y_bits)
    if ((y_words < 0) | (y_words >= 2**y_bits)).any():
        return False, "auth: y word out of range"
    per_vu = {"a": (np.asarray(auth.x_index, dtype=np.int64), digests[:, :lanes]),
              "b": (np.asarray(auth.w_index, dtype=np.int64), digests[:, lanes:])}
    for name, sibs in zip(TREE_NAMES, siblings):
        ref = auth.tree(name)
        domain = ref.domain()
        leaves: dict[int, bytes] = {}
        if name == "y":
            ranks = np.asarray(auth.vu_index, dtype=np.int64)
            if ranks.size and ranks.max() >= ref.count:
                return False, "auth: y: leaf index out of range"
            for r, w in zip(ranks.tolist(), y_words.tolist()):
                leaves[r] = domain.leaf(r, r, yschema, w.to_bytes(ybytes, "big"))
        else:
            ranks, dg = per_vu[name]
            if ranks.size and ranks.max() >= ref.count:
                return False, f"auth: {name}: leaf index out of range"
            vals: dict[int, bytes] = {}
            for v in range(n):
                r = int(ranks[v])
                val = leaf.leaf_bytes(dg[v])
                if vals.setdefault(r, val) != val:
                    return False, f"auth: {name}: VUs claiming leaf {r} published different digests"
            leaves = {r: domain.leaf(r, r, leaf.schema, val) for r, val in vals.items()}
        try:
            LIMITS.enforce("max_positions", int(ref.count))
            LIMITS.enforce("max_openings", len(leaves))
            LIMITS.enforce("max_proof_bytes", 32 * len(sibs))
        except ResourceLimit as e:
            return False, f"auth: {name}: {e}"
        if any(len(s) != 32 for s in sibs):
            return False, f"auth: {name}: malformed sibling"
        if not verify_multiproof(ref.root, ref.count, domain.domain_id, leaves, [bytes(s) for s in sibs]):
            return False, f"auth: {name}: multiproof rejected"
    return True, ""
