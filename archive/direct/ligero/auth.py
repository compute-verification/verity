"""Authentication of the boundary words against committed object roots (``authentication = included``).

The B-Ligero statement (PROTOCOL.md) publishes every VU's operand words ``a``, ``b`` and output word ``y16`` and binds
the prover's challenges to them through ``statement_digest``.  On its own that proves "*some* dot products were
computed".  This module supplies the missing half of the claim -- "the dot products were computed on *these committed*
weights and activations, and produced *these committed* outputs" -- by

  1. committing the input set's words as three indexed Merkle trees of :mod:`verity.ml.commitments` (one 16-bit word per
     leaf, ``schema="u16"``, big-endian value bytes; SHA-256 with the ``veritor/protocol/merkle/frame/v3`` framing;
     binary tree, ``merkle_depth(count)`` levels):

         tree ``a``  activations   owner -1   leaf rank  vu*K + k        for the K words of VU ``vu``
         tree ``b``  weights       owner -2   leaf rank  vu*K + k
         tree ``y``  outputs       owner -1   leaf rank  vu              (the 16-bit public output word)

     The commitment domain of each tree is ``CommitmentDomain(binding, owner, RangeIndexedDomain(0, count))``; the
     *binding* is the identity digest of the frozen input set's manifest (dataset, tier, ``manifest_sha256``, the VU
     range, K, the tree name), so the same words under a different manifest are a different object.

  2. extending the public statement with an :class:`AuthStatement` -- the three ``(binding, owner, count, root)``
     references plus the claimed VU indices ``vu_index`` (one per chain, strictly increasing) -- and absorbing
     :func:`digest_bytes` of it into ``statement_digest`` so every challenge of the proof is bound to "the words at
     these leaf indices under these roots".

  3. having the prover ship one Merkle *multiproof* per tree (shared authentication paths, canonical order of
     :func:`verity.ml.commitments.multiproof`) and having *both* verifiers recompute the leaf digests from the published
     words and fold them to the root before accepting.  This is the verifier-side-openings option of the
     ``auth-integration`` lane: the Ligero proof is unchanged; the authentication cost lands in the verifier's time
     and in the transcript bytes.

Everything here is torch-free.  The trees are built with a process pool (SHA-256 in :mod:`hashlib` releases the GIL
only per call; the per-leaf framing is the Python-level cost) and cached on disk as one ``.npy`` per level.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from verity.ml.commitments import (
    CommitmentDomain,
    RangeIndexedDomain,
    VerificationLimits,
    identity_digest,
    merkle_depth,
)
from verity.ml.commitments.multiproof import verify_multiproof
from verity.ml.commitments.merkle import (  # the pad-leaf frame has no public wrapper; pinned by tests
    _FRAME,
    _LEAF,
    _NODE,
    _PAD,
    _uint,
)
from verity.errors import ResourceLimit

__all__ = [
    "AUTH_VERSION",
    "BINDING_TAG",
    "SCHEMA",
    "TREE_NAMES",
    "AuthStatement",
    "TreeRef",
    "WordTree",
    "auth_positions",
    "binding_digest",
    "build_word_tree",
    "digest_bytes",
    "load_or_build_trees",
    "multiproof_siblings",
    "open_trees",
    "verify_auth",
]

AUTH_VERSION = "ligero-b/auth/v1"
BINDING_TAG = "verity/ligero-b/auth-binding/v1"
SCHEMA = "u16"
TREE_NAMES = ("a", "b", "y")
OWNER = {"a": -1, "b": -2, "y": -1}
LIMITS = VerificationLimits()

_CHUNK = 1 << 16


_POOLS: dict[int, ProcessPoolExecutor] = {}


def _pool(workers: int) -> ProcessPoolExecutor:
    """A hashing pool of ``workers`` processes, forked once per process and reused (never shut down; the workers only
    hash, no torch/CUDA).  Forking a large prover process anew for every verification write-protects its pages and the
    *next* sub-batch's prover pays the copy-on-write faults (measured on the A100 pod: 0.061 -> 0.10-0.16 s per sub-batch
    when a 64-worker pool was created per verification), so the fork happens once, in the warm-up."""
    import multiprocessing as mp
    ex = _POOLS.get(workers)
    if ex is None:
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
        ex = _POOLS[workers] = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
    return ex


# ---------------------------------------------------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------------------------------------------------

def binding_digest(*, dataset: str, tier: str, manifest_sha256: str, lo: int, hi: int, K: int, tree: str) -> bytes:
    """Identity of the committed object: the frozen input set (by manifest digest) and which words of it."""
    if tree not in TREE_NAMES:
        raise ValueError(f"unknown tree {tree!r}")
    h = identity_digest(BINDING_TAG, {"dataset": dataset, "tier": tier, "manifest_sha256": manifest_sha256,
                                      "vu_lo": int(lo), "vu_hi": int(hi), "K": int(K), "tree": tree,
                                      "schema": SCHEMA})
    return bytes.fromhex(h)


def domain_for(binding: bytes, owner: int, count: int) -> CommitmentDomain:
    return CommitmentDomain(bytes(binding), int(owner), RangeIndexedDomain(0, int(count)))


# Batch forms of CommitmentDomain.leaf / .node and the pad leaf (verity.ml.commitments.merkle): the frame, tag and domain id
# are one hashed prefix copied per hash, the parts one update -- the same bytes into SHA-256 (pinned by hashchain_test.py).
_L32 = (32).to_bytes(8, "big")


def _part(b: bytes) -> bytes:
    return len(b).to_bytes(8, "big") + b


def _uint_part(v: int) -> bytes:
    return _part(_uint(v))


def _prefix(tag: bytes, domain_id: bytes):
    h = hashlib.sha256(_FRAME)
    h.update(len(tag).to_bytes(4, "big") + tag + _part(domain_id))
    return h


def leaf_hashes(domain_id: bytes, start: int, schema: str, values: Sequence[bytes]) -> bytes:
    """``domain.leaf(r, r, schema, values[r - start])`` for consecutive ranks from ``start``, concatenated."""
    h0, sp = _prefix(_LEAF, domain_id), _part(schema.encode("utf-8"))
    out = bytearray()
    for i, v in enumerate(values):
        r = _uint_part(start + i)
        h = h0.copy()
        h.update(r + r + sp + _part(v))
        out += h.digest()
    return bytes(out)


def _leaf_chunk(args) -> bytes:
    domain, start, words = args
    return leaf_hashes(domain.domain_id, start, SCHEMA, [w.to_bytes(2, "big") for w in words.tolist()])


def _pad_chunk(args) -> bytes:
    domain_id, start, stop = args
    h0 = _prefix(_PAD, domain_id)
    out = bytearray()
    for r in range(start, stop):
        h = h0.copy()
        h.update(_uint_part(r))
        out += h.digest()
    return bytes(out)


def _node_chunk(args) -> bytes:
    domain, level, start, pairs = args
    h0, lv, buf = _prefix(_NODE, domain.domain_id), _uint_part(level), np.ascontiguousarray(pairs).tobytes()
    out = bytearray()
    for i in range(len(buf) // 64):
        h = h0.copy()
        h.update(lv + _uint_part(start + i) + _L32 + buf[64 * i:64 * i + 32] + _L32 + buf[64 * i + 32:64 * i + 64])
        out += h.digest()
    return bytes(out)


def _chunks(n: int, size: int) -> list[tuple[int, int]]:
    return [(s, min(n, s + size)) for s in range(0, n, size)]


@dataclass(frozen=True)
class TreeRef:
    """Public reference to one committed tree: what the verifier needs to rebuild the domain and check the root."""

    name: str
    binding: bytes
    owner: int
    count: int
    root: bytes

    def domain(self) -> CommitmentDomain:
        return domain_for(self.binding, self.owner, self.count)

    def to_json(self) -> dict:
        return {"name": self.name, "binding": self.binding.hex(), "owner": self.owner, "count": self.count,
                "root": self.root.hex()}

    @classmethod
    def from_json(cls, d: Mapping) -> TreeRef:
        return cls(str(d["name"]), bytes.fromhex(d["binding"]), int(d["owner"]), int(d["count"]),
                   bytes.fromhex(d["root"]))


class WordTree:
    """An indexed Merkle tree over ``count`` 16-bit words; ``levels[0]`` are the leaves (padded to a power of two)."""

    def __init__(self, ref: TreeRef, levels: Sequence[np.ndarray]):
        self.ref = ref
        self._dev = None
        self._levels = list(levels)
        if len(self._levels) != merkle_depth(ref.count) + 1 or self._levels[-1].shape[0] != 1:
            raise ValueError("levels do not form a tree of the declared count")
        if self.root != ref.root:
            raise ValueError("stored root does not match the reference")

    @classmethod
    def on_device(cls, ref: TreeRef, tree) -> WordTree:
        """A tree built on the device (``frame_gpu``: a ``hash_gpu.frame_v3.Tree``); its levels come to the host once,
        when first read (openings, saving), unless the tree was small enough to be copied at build time."""
        self = cls.__new__(cls)
        self.ref, self._dev, self._levels = ref, tree, None
        if tree.depth != merkle_depth(ref.count) or tree.root != ref.root:
            raise ValueError("device tree does not match the reference")
        return self

    @property
    def levels(self) -> list[np.ndarray]:
        if self._levels is None:
            self._levels = self._dev.levels
        return self._levels

    @property
    def root(self) -> bytes:
        return self._dev.root if self._levels is None else self._levels[-1][0].tobytes()

    @property
    def count(self) -> int:
        return self.ref.count

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for lv, arr in enumerate(self.levels):
            np.save(directory / f"{self.ref.name}.L{lv:02d}.npy", arr)
        (directory / f"{self.ref.name}.json").write_text(json.dumps(self.ref.to_json(), indent=1))

    @classmethod
    def load(cls, directory: Path, name: str) -> WordTree:
        directory = Path(directory)
        ref = TreeRef.from_json(json.loads((directory / f"{name}.json").read_text()))
        levels = [np.load(directory / f"{name}.L{lv:02d}.npy", mmap_mode="r") for lv in range(merkle_depth(ref.count) + 1)]
        return cls(ref, levels)


def build_word_tree(name: str, binding: bytes, words: np.ndarray, *, procs: int | None = None) -> WordTree:
    """Commit ``words`` (uint16, one per leaf, leaf rank == position == array index) in the tree's domain; on the GPU
    (``frame_gpu``) when it is available."""
    from . import frame_gpu
    if frame_gpu.available():
        return frame_gpu.word_tree(name, binding, words, nbytes=2, schema=SCHEMA)
    words = np.ascontiguousarray(words, dtype=np.uint16).ravel()
    count = int(words.shape[0])
    LIMITS.enforce("max_positions", count)
    owner = OWNER[name]
    domain = domain_for(binding, owner, count)
    depth = merkle_depth(count)
    width = 1 << depth
    procs = procs or (os.cpu_count() or 1)

    def run(fn, tasks):
        if procs <= 1 or len(tasks) <= 1:
            return [fn(t) for t in tasks]
        return list(_pool(procs).map(fn, tasks, chunksize=1))

    parts = run(_leaf_chunk, [(domain, s, words[s:e]) for s, e in _chunks(count, _CHUNK)])
    parts += run(_pad_chunk, [(domain.domain_id, count + s, count + e) for s, e in _chunks(width - count, _CHUNK)])
    level = np.frombuffer(b"".join(parts), dtype=np.uint8).reshape(width, 32).copy()
    levels = [level]
    for lv in range(depth):
        pairs = level.reshape(-1, 64)
        parts = run(_node_chunk, [(domain, lv, s, pairs[s:e]) for s, e in _chunks(pairs.shape[0], _CHUNK)])
        level = np.frombuffer(b"".join(parts), dtype=np.uint8).reshape(-1, 32).copy()
        levels.append(level)
    ref = TreeRef(name, bytes(binding), owner, count, levels[-1][0].tobytes())
    return WordTree(ref, levels)


def load_or_build_trees(cache: Path | None, *, a_words: np.ndarray, b_words: np.ndarray, y_words: np.ndarray,
                        bindings: Mapping[str, bytes], procs: int | None = None) -> tuple[dict[str, WordTree], bool]:
    """The three trees of an input set, from ``cache`` when its bindings match, else built (and cached)."""
    words = {"a": a_words, "b": b_words, "y": y_words}
    if cache is not None:
        cache = Path(cache)
        try:
            trees = {n: WordTree.load(cache, n) for n in TREE_NAMES}
            if all(trees[n].ref.binding == bytes(bindings[n]) and trees[n].count == int(np.size(words[n]))
                   for n in TREE_NAMES):
                return trees, True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    trees = {n: build_word_tree(n, bindings[n], words[n], procs=procs) for n in TREE_NAMES}
    if cache is not None:
        for t in trees.values():
            t.save(cache)
    return trees, False


# ---------------------------------------------------------------------------------------------------------------------
# Statement block, positions, openings
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AuthStatement:
    """The public authentication block: which VUs (by index) under which committed trees."""

    vu_index: tuple[int, ...]
    trees: tuple[TreeRef, TreeRef, TreeRef]   # in TREE_NAMES order

    def __post_init__(self):
        if tuple(t.name for t in self.trees) != TREE_NAMES:
            raise ValueError("trees must be given in (a, b, y) order")
        idx = self.vu_index
        if any(idx[i] >= idx[i + 1] for i in range(len(idx) - 1)) or any(v < 0 for v in idx):
            raise ValueError("vu_index must be strictly increasing and non-negative")

    def tree(self, name: str) -> TreeRef:
        return self.trees[TREE_NAMES.index(name)]

    def to_json(self) -> dict:
        return {"version": AUTH_VERSION, "schema": SCHEMA, "vu_index": list(self.vu_index),
                "trees": [t.to_json() for t in self.trees]}

    @classmethod
    def from_json(cls, d: Mapping) -> AuthStatement:
        return cls(tuple(int(v) for v in d["vu_index"]), tuple(TreeRef.from_json(t) for t in d["trees"]))

    @classmethod
    def for_vus(cls, trees: Mapping[str, WordTree | TreeRef], vu_index: Iterable[int]) -> AuthStatement:
        refs = tuple(trees[n].ref if isinstance(trees[n], WordTree) else trees[n] for n in TREE_NAMES)
        return cls(tuple(int(v) for v in vu_index), refs)


def digest_bytes(auth: AuthStatement) -> bytes:
    """What ``statement_digest`` absorbs: version, schema, the VU indices and each tree's reference (root included)."""
    out = bytearray(b"|auth=" + AUTH_VERSION.encode() + b"|schema=" + SCHEMA.encode() + b"|vus=")
    out += np.asarray(auth.vu_index, dtype="<u4").tobytes()
    for t in auth.trees:
        out += b"|" + t.name.encode() + b":binding=" + t.binding + b"|owner=" + str(t.owner).encode()
        out += b"|count=" + str(t.count).encode() + b"|root=" + t.root
    return bytes(out)


def auth_positions(auth: AuthStatement, *, K: int, steps: int, words: int = 16) -> dict[str, np.ndarray]:
    """Leaf ranks of the published words, by tree, in the column-major order of the statement's ``a``/``b``/``y16``.

    Column ``j`` of the statement is step ``j % steps`` of chain ``j // steps``; its ``words`` A-words are leaves
    ``vu*K + i*words + o`` (likewise B); the chain's output word is leaf ``vu`` of tree ``y`` (one per chain).
    """
    if steps * words != K:
        raise ValueError("steps*words must equal K")
    vus = np.asarray(auth.vu_index, dtype=np.int64)
    base = (vus[:, None] * K + (np.arange(steps, dtype=np.int64) * words)[None, :]).reshape(-1)  # per column
    ab = (base[:, None] + np.arange(words, dtype=np.int64)[None, :]).reshape(-1)               # per column, per word
    return {"a": ab, "b": ab.copy(), "y": vus.copy()}


def _ranges(idx: np.ndarray) -> list[tuple[int, int]]:
    """Sorted unique ranks -> maximal runs ``[lo, hi)`` of consecutive ranks (so no two runs are adjacent)."""
    if idx.size == 0:
        return []
    cut = np.flatnonzero(idx[1:] != idx[:-1] + 1) + 1
    starts = np.concatenate(([0], cut))
    ends = np.concatenate((cut, [idx.size]))
    return [(int(idx[s]), int(idx[e - 1]) + 1) for s, e in zip(starts.tolist(), ends.tolist())]


def _merge(runs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge sorted runs that overlap or touch (``hi == lo_next``)."""
    out: list[tuple[int, int]] = []
    for lo, hi in runs:
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def multiproof_siblings(tree: WordTree, ranks: np.ndarray) -> list[bytes]:
    """The canonical multiproof of :mod:`verity.ml.commitments.multiproof` for ``ranks`` (a set): per level, ascending,
    the sibling of every known node whose sibling is not itself known.

    Computed on the *runs* of known nodes rather than node by node: inside a run every sibling is known, so a run
    ``[lo, hi)`` needs ``lo - 1`` iff ``lo`` is odd and ``hi`` iff ``hi - 1`` is even (neither is known, the runs being
    maximal); its parents are the run ``[lo >> 1, ((hi - 1) >> 1) + 1)`` and touching parent runs merge.  Emission in run
    order is ascending: ``lo_i - 1 < hi_i < lo_{i+1} - 1`` whenever both ends are emitted (they differ in parity).  A
    sub-batch's ``170 x 1536`` contiguous leaves are one run, so the cost is ``O(depth x runs)`` -- this is prover-side
    work the transcript carries, timed as ``auth_openings``."""
    idx = np.asarray(ranks, dtype=np.int64).ravel()
    if idx.size > 1 and not bool(np.all(idx[1:] > idx[:-1])):   # already a sorted set (the statement's order) -> no sort
        idx = np.unique(idx)
    if idx.size and (idx[0] < 0 or idx[-1] >= tree.count):
        raise ValueError("rank out of range")
    runs = _ranges(idx)
    out: list[bytes] = []
    for lv in range(len(tree.levels) - 1):
        level = tree.levels[lv]
        for lo, hi in runs:
            if lo & 1:
                out.append(level[lo - 1].tobytes())
            if not ((hi - 1) & 1):
                out.append(level[hi].tobytes())
        runs = _merge([(lo >> 1, ((hi - 1) >> 1) + 1) for lo, hi in runs])
    return out


def open_trees(trees: Mapping[str, WordTree], auth: AuthStatement, *, K: int, steps: int) -> list[list[bytes]]:
    """The prover's three multiproofs (in TREE_NAMES order) for the VUs of ``auth``."""
    pos = auth_positions(auth, K=K, steps=steps)
    return [multiproof_siblings(trees[n], pos[n]) for n in TREE_NAMES]


# ---------------------------------------------------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------------------------------------------------

def _leaf_digests(domain: CommitmentDomain, ranks: np.ndarray, words: np.ndarray, procs: int) -> dict[int, bytes]:
    ranks = np.asarray(ranks, dtype=np.int64).ravel()
    words = np.asarray(words, dtype=np.uint16).ravel()
    if ranks.shape != words.shape:
        raise ValueError("ranks/words shape mismatch")
    if ranks.size == 0:
        return {}
    order = np.argsort(ranks, kind="stable")
    ranks, words = ranks[order], words[order]
    # The domain is a range domain (position == rank): the pool workers use the same leaf frame as the tree builder.
    chunk = max(4096, -(-ranks.size // max(procs, 1)))   # spread one sub-batch's ~261k leaves over the workers
    tasks = [(domain, int(ranks[s]), words[s:e]) for s, e in _chunks(ranks.size, chunk)
             if np.array_equal(ranks[s:e], np.arange(ranks[s], ranks[s] + (e - s)))]
    if len(tasks) == len(_chunks(ranks.size, chunk)) and procs > 1 and len(tasks) > 1:
        blob = b"".join(_pool(procs).map(_leaf_chunk, tasks, chunksize=1))
        return {int(r): blob[32 * i:32 * i + 32] for i, r in enumerate(ranks.tolist())}
    leaf = domain.leaf
    return {int(r): leaf(int(r), int(r), SCHEMA, int(w).to_bytes(2, "big")) for r, w in zip(ranks.tolist(), words.tolist())}


def verify_auth(auth: AuthStatement, siblings: Sequence[Sequence[bytes]], *, a_words: np.ndarray, b_words: np.ndarray,
                y_words: np.ndarray, K: int, steps: int, procs: int | None = None) -> tuple[bool, str]:
    """Check the three multiproofs against the roots of ``auth`` for the words the statement publishes.

    ``a_words``/``b_words`` are the statement's operand words in column-major order (``(n_cols, 16)`` or flat),
    ``y_words`` the per-chain output words.  Returns ``(ok, reason)``; every failure is a rejection (no exceptions).
    """
    procs = procs or (os.cpu_count() or 1)
    try:
        pos = auth_positions(auth, K=K, steps=steps)
    except ValueError as e:
        return False, f"auth: {e}"
    if len(siblings) != len(TREE_NAMES):
        return False, "auth: expected one multiproof per tree"
    words = {"a": np.asarray(a_words, dtype=np.uint16).ravel(), "b": np.asarray(b_words, dtype=np.uint16).ravel(),
             "y": np.asarray(y_words, dtype=np.uint16).ravel()}
    for name, sibs in zip(TREE_NAMES, siblings):
        ref = auth.tree(name)
        ranks, w = pos[name], words[name]
        if ranks.shape != w.shape:
            return False, f"auth: {name}: {ranks.size} leaves claimed, {w.size} words published"
        if ranks.size and (ranks.min() < 0 or ranks.max() >= ref.count):
            return False, f"auth: {name}: leaf index out of range"
        if np.unique(ranks).size != ranks.size:
            return False, f"auth: {name}: duplicate leaf index"
        try:
            LIMITS.enforce("max_positions", int(ref.count))
            LIMITS.enforce("max_openings", int(ranks.size))
            LIMITS.enforce("max_proof_bytes", 32 * len(sibs))
        except ResourceLimit as e:
            return False, f"auth: {name}: {e}"
        if any(len(s) != 32 for s in sibs):
            return False, f"auth: {name}: malformed sibling"
        domain = ref.domain()
        leaves = _leaf_digests(domain, ranks, w, procs)
        if not verify_multiproof(ref.root, ref.count, domain.domain_id, leaves, [bytes(s) for s in sibs]):
            return False, f"auth: {name}: multiproof rejected"
    return True, ""
