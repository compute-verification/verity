"""``authentication = included-hash``: the committed-object side of the self-hashing relation (``hashchain.py``).

Three indexed Merkle trees of :mod:`verity.ml.commitments` (the framing, domains and multiproofs of ``auth.py``, lane
auth-included, unchanged), with the operand trees now holding one **row** per leaf (``leaf/v2h``,
:mod:`verity.ml.commitments.rowleaf`): tree ``a`` commits the activation rows (one x row of K words = one leaf: the row's
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
from verity.ml.commitments.multiproof import verify_multiproof
from verity.ml.commitments import identity_digest
from verity.ml.commitments.rowleaf import SCHEMA_ROW

from . import frame_gpu
from . import poseidon2 as h2
from .auth import (LIMITS, OWNER, TREE_NAMES, TreeRef, WordTree, _chunks, _node_chunk, _pad_chunk, domain_for, leaf_hashes,
                   multiproof_siblings)
from .auth import build_word_tree as _build_word_tree_u16
from .leaf import registry as leaf_registry
from verity.errors import ResourceLimit

__all__ = ["BINDING_TAG", "HASH_AUTH_VERSION", "HashAuth", "binding_digest", "build_row_tree", "build_trees", "digest_bytes",
           "layout_error", "load_or_build_trees", "open_trees", "verify_hash_auth", "word_schema"]

HASH_AUTH_VERSION = "ligero-b/auth/v2h"
BINDING_TAG = "verity/ligero-b/auth-binding/v2h"


def _leaf(leaf):
    """The scheme behind ``leaf`` (a registry name, a scheme, or ``None`` = the default Poseidon2 leaf/v2h)."""
    return leaf_registry.get(leaf) if leaf is None or isinstance(leaf, str) else leaf


def word_schema(y_bits: int) -> tuple[str, int]:
    if y_bits <= 16:
        return "u16", 2
    if y_bits <= 32:
        return "u32", 4
    raise ValueError("output words wider than 32 bits")


def binding_digest(*, dataset: str, tier: str, manifest_sha256: str, lo: int, hi: int, K: int, tree: str, schema: str) -> bytes:
    """Identity of the committed object: the frozen input set (by manifest digest), which rows of it, the leaf schema."""
    if tree not in TREE_NAMES:
        raise ValueError(f"unknown tree {tree!r}")
    h = identity_digest(BINDING_TAG, {"dataset": dataset, "tier": tier, "manifest_sha256": manifest_sha256,
                                      "lo": int(lo), "hi": int(hi), "K": int(K), "tree": tree, "schema": schema})
    return bytes.fromhex(h)


# ---------------------------------------------------------------------------------------------------------------------
# Trees
# ---------------------------------------------------------------------------------------------------------------------

def _levels(domain, leaves: list[bytes], count: int) -> list[np.ndarray]:
    from verity.ml.commitments import merkle_depth
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


def build_row_tree(name: str, binding: bytes, rows: np.ndarray, word_bits: int, role: int, leaf=None,
                   digests: np.ndarray | None = None, rows_dev=None) -> tuple[WordTree, np.ndarray]:
    """Commit ``rows`` (count, n_words) of ``word_bits``-bit words as row leaves of the scheme ``leaf`` (default: Poseidon2,
    ``leaf/v2h``): the digest of each row (``role``: ``h2.ROLE_X`` / ``ROLE_W``, the scheme's domain tag), framed by
    ``leaf.leaf_bytes`` under schema ``leaf.schema``, under the SHA-256 tree.  Returns the tree and the digests
    (count, digest_elems).  ``digests``: the rows' ``scheme.native`` digests already computed (the committer's device
    sponges, ``relchain.HashedRelationRunner.commit``) instead of hashing ``rows`` here.  A scheme with a GPU row digest
    (``frame_gpu.ROW_SCHEMES``) is committed on the device from ``rows_dev`` (the rows already there) or ``rows``, and
    ``digests`` (host or device) must then equal the device's."""
    scheme = _leaf(leaf)
    count = int(np.shape(rows)[0])
    LIMITS.enforce("max_positions", count)
    if frame_gpu.available() and scheme.name in frame_gpu.ROW_SCHEMES:
        return frame_gpu.row_tree(name, binding, rows if rows_dev is None else rows_dev, word_bits=word_bits, role=role,
                                  scheme=scheme, expect=digests)
    domain = domain_for(binding, OWNER[name], count)
    if digests is None:
        digests = scheme.native(np.ascontiguousarray(rows, dtype=np.int64), word_bits=word_bits, role=role)
    digests = np.asarray(digests, dtype=np.int64)
    if digests.shape != (count, scheme.digest_elems) or ((digests < 0) | (digests >= h2.P)).any():
        raise ValueError(f"leaf {scheme.name}: native() must return (count, {scheme.digest_elems}) elements in [0, p)")
    many = getattr(scheme, "leaf_bytes_many", None)
    values = many(digests) if many is not None else [scheme.leaf_bytes(digests[r]) for r in range(count)]
    levels = _levels(domain, [leaf_hashes(domain.domain_id, 0, scheme.schema, values)], count)
    ref = TreeRef(name, bytes(binding), OWNER[name], count, levels[-1][0].tobytes())
    return WordTree(ref, levels), digests


def build_word_tree(name: str, binding: bytes, words: np.ndarray, y_bits: int) -> WordTree:
    """The output-word tree: one word per leaf, 2 or 4 big-endian bytes by ``y_bits`` (``auth.py``'s tree for 16 bits)."""
    schema, nbytes = word_schema(y_bits)
    if frame_gpu.available():
        return frame_gpu.word_tree(name, binding, words, nbytes=nbytes, schema=schema)
    if nbytes == 2:
        return _build_word_tree_u16(name, binding, words, procs=1)
    words = np.ascontiguousarray(words, dtype=np.int64).ravel()
    count = int(words.shape[0])
    domain = domain_for(binding, OWNER[name], count)
    values = [int(w).to_bytes(nbytes, "big") for w in words.tolist()]
    levels = _levels(domain, [leaf_hashes(domain.domain_id, 0, schema, values)], count)
    return WordTree(TreeRef(name, bytes(binding), OWNER[name], count, levels[-1][0].tobytes()), levels)


def build_trees(*, x_rows: np.ndarray, w_cols: np.ndarray, y_words: np.ndarray, word_bits: int, y_bits: int,
                bindings: Mapping[str, bytes], leaf=None, digests: Mapping[str, np.ndarray] | None = None,
                rows_dev: Mapping | None = None) -> tuple[dict[str, WordTree], dict[str, np.ndarray]]:
    """The three trees of a committed input set; also the operand digests (what the honest prover will publish).
    ``digests``: ``{"a": x digests, "b": W digests}`` computed already (:func:`build_row_tree`); ``rows_dev``: ``{"a",
    "b"}`` the rows on the device."""
    scheme = _leaf(leaf)
    digests, rows_dev = digests or {}, rows_dev or {}
    if is_vllm(scheme):
        return _vllm_trees(x_rows=x_rows, w_cols=w_cols, y_words=y_words, word_bits=word_bits, y_bits=y_bits, bindings=bindings,
                           scheme=scheme, digests=digests, rows_dev=rows_dev)
    ta, da = build_row_tree("a", bindings["a"], x_rows, word_bits, h2.ROLE_X, scheme, digests=digests.get("a"),
                            rows_dev=rows_dev.get("a"))
    tb, db = build_row_tree("b", bindings["b"], w_cols, word_bits, h2.ROLE_W, scheme, digests=digests.get("b"),
                            rows_dev=rows_dev.get("b"))
    ty = build_word_tree("y", bindings["y"], y_words, y_bits)
    return {"a": ta, "b": tb, "y": ty}, {"a": da, "b": db}


def load_or_build_trees(cache: Path | None, **kw) -> tuple[dict[str, WordTree], dict[str, np.ndarray], bool]:
    bindings = kw["bindings"]
    if is_vllm(_leaf(kw.get("leaf"))):
        cache = None                    # vllm-v1 trees are always built (their levels are not WordTree files)
    if cache is not None:
        cache = Path(cache)
        try:
            trees = {n: WordTree.load(cache, n) for n in TREE_NAMES}
            if all(trees[n].ref.binding == bytes(bindings[n]) for n in TREE_NAMES):
                # the binding names the leaf schema, so a cache built under another scheme does not match
                dig = {n: np.load(cache / f"{n}.digests.npy") for n in ("a", "b")}
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


def is_vllm(scheme) -> bool:
    """Whether the scheme's statements commit under ``vllm-v1`` trees (``vllm_tree.py``) instead of frame-v3 indexed trees."""
    return getattr(scheme, "tree", "frame-v3") == "vllm-v1"


def vllm_shape(K: int, word_bits: int, y_bits: int) -> tuple[int, int, int]:
    """``(K, word_bits, y_bytes)``: what a vllm-v1 port's program / geo / layout digests are derived from."""
    from . import vllm_tree
    return int(K), int(word_bits), vllm_tree.y_bytes_of(y_bits)


def vllm_domain(name: str, ref, vshape: tuple[int, int, int]):
    """The verifier's step domain of tree reference ``ref``: its own program / geo / layout, the ref's ctx and count."""
    from . import vllm_tree
    K, wb, yb = vshape
    return vllm_tree.domain(name, ref.binding, ref.count, K=K, word_bits=wb, y_bytes=yb)


def _vllm_trees(*, x_rows, w_cols, y_words, word_bits: int, y_bits: int, bindings: Mapping[str, bytes], scheme, digests, rows_dev):
    """The three ``vllm-v1`` port trees: x rows / W columns as position leaves (the published digests), output words as
    position leaves of their ``y_bytes`` little-endian bytes; each root the step-root binding of its domain."""
    from . import vllm_tree
    from .leaf import vllm_v1 as lv
    K = int(np.shape(x_rows)[1])
    vshape = vllm_shape(K, word_bits, y_bits)
    trees, out = {}, {}
    gpu = frame_gpu.available()
    for name, rows, role in (("a", x_rows, h2.ROLE_X), ("b", w_cols, h2.ROLE_W)):
        count = int(np.shape(rows)[0])
        LIMITS.enforce("max_positions", count)
        dom = vllm_tree.domain(name, bindings[name], count, K=K, word_bits=word_bits, y_bytes=vshape[2])
        if gpu:
            cp = frame_gpu.cp
            with frame_gpu._torch_stream():
                rb = frame_gpu.row_bytes(rows if rows_dev.get(name) is None else rows_dev[name], word_bits)
                leaves = frame_gpu.fv3.sha256_rows(rb, lv.prefix(int(rb.shape[1])))
                words = leaves.reshape(count, 8, 4).astype(cp.int64)
                be = (words[:, :, 0] << 24) | (words[:, :, 1] << 16) | (words[:, :, 2] << 8) | words[:, :, 3]
                dg = cp.stack([be & 0xFFFF, be >> 16], -1).reshape(count, 16)
                if digests.get(name) is not None:
                    e = cp.asarray(digests[name]).astype(cp.int64, copy=False)
                    if e.shape != dg.shape or not bool((e == dg).all()):
                        raise ValueError(f"leaf {scheme.name}: the chain digests are not the committed rows' position leaves")
                trees[name] = vllm_tree.build(name, bindings[name], leaves, dom)
                out[name] = dg.get()
        else:
            dg = digests.get(name)
            dg = scheme.native(np.ascontiguousarray(rows, dtype=np.int64), word_bits=word_bits, role=role) if dg is None else np.asarray(dg, dtype=np.int64)
            if dg.shape != (count, scheme.digest_elems):
                raise ValueError(f"leaf {scheme.name}: native() must return (count, {scheme.digest_elems})")
            leaves = np.frombuffer(b"".join(scheme.leaf_bytes_many(dg)), dtype=np.uint8).reshape(count, 32)
            trees[name] = vllm_tree.build(name, bindings[name], leaves, dom)
            out[name] = dg
    yb = vshape[2]
    yw = np.ascontiguousarray(np.asarray(y_words, dtype=np.int64).ravel())
    count = int(yw.shape[0])
    LIMITS.enforce("max_positions", count)
    ybytes = yw.astype({2: "<u2", 4: "<u4"}[yb]).view(np.uint8).reshape(count, yb)
    dom = vllm_tree.domain("y", bindings["y"], count, K=K, word_bits=word_bits, y_bytes=yb)
    if gpu:
        with frame_gpu._torch_stream():
            leaves = frame_gpu.fv3.sha256_rows(frame_gpu.cp.asarray(ybytes), lv.prefix(yb))
            trees["y"] = vllm_tree.build("y", bindings["y"], leaves, dom)
    else:
        from verity.ml.commitments.vllm_v1 import pos_leaf
        leaves = np.frombuffer(b"".join(pos_leaf(bytes(r)) for r in ybytes), dtype=np.uint8).reshape(count, 32)
        trees["y"] = vllm_tree.build("y", bindings["y"], leaves, dom)
    return trees, out


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
        refs = tuple(trees[n].ref if hasattr(trees[n], "ref") else trees[n] for n in TREE_NAMES)
        return cls(tuple(int(v) for v in vu_index), tuple(int(v) for v in x_index), tuple(int(v) for v in w_index), refs)

    def indices(self) -> np.ndarray:
        """(n_vus, 3) ``<u4``: what the Ligero statement's ``b`` slot carries (``hashchain.hash_public_vectors``)."""
        return np.stack([np.asarray(self.vu_index), np.asarray(self.x_index), np.asarray(self.w_index)], 1).astype(np.int64)


def digest_bytes(auth: HashAuth, leaf=None, vshape: tuple[int, int, int] | None = None) -> bytes:
    """What the statement digest absorbs (through the relation hooks' tag): version, the scheme's leaf schema and
    parameters digest, the index rows, the tree references.  (The ``|p2=`` label predates the registry and stays: the
    Rust verifier's ``hash_digest_bytes`` mirrors these bytes for every scheme.)"""
    scheme = _leaf(leaf)
    out = bytearray(b"|auth=" + HASH_AUTH_VERSION.encode() + b"|schema=" + scheme.schema.encode() + b"|p2=" +
                    bytes.fromhex(scheme.params_sha256) + b"|idx=")
    out += auth.indices().astype("<u4").tobytes()
    for t in auth.trees:
        out += b"|" + t.name.encode() + b":binding=" + t.binding + b"|owner=" + str(t.owner).encode()
        out += b"|count=" + str(t.count).encode() + b"|root=" + t.root
        if is_vllm(scheme):
            # PROTOCOL.md (vllm-v1) section 9.1: the verifier-held domain of every port, beside its bound root
            if vshape is None:
                raise ValueError("a vllm-v1 statement's digest needs the relation's (K, word_bits, y_bytes)")
            from verity.ml.commitments.vllm_v1 import domain_digest
            out += b"|domain=" + domain_digest(vllm_domain(t.name, t, vshape))
    return bytes(out)


def layout_error(auth: HashAuth) -> str | None:
    """The x row and W column of a VU are fixed by its ``y`` leaf and the committed layout, never chosen by the prover:
    one leaf per VU in all three trees (unshared) is x = W = vu; an ``nx`` x ``nw`` tile (x tree ``nx``, W tree ``nw``,
    y tree ``nx * nw`` leaves, :func:`relchain.tile_indices`) is x = vu // nw, W = vu % nw.  Any other tree shape or
    claim is refused (red-team-standard-hash R1: a statement pairing an output with another VU's rows)."""
    na, nb, ny = (int(t.count) for t in auth.trees)
    vu = np.asarray(auth.vu_index, dtype=np.int64)
    if vu.size and int(vu.max()) >= ny:
        return "auth: y: leaf index out of range"
    if na == nb == ny:
        ex, ew = vu, vu
    elif na * nb == ny:
        ex, ew = vu // nb, vu % nb
    else:
        return f"auth: tree counts a={na} b={nb} y={ny} are neither one leaf per VU nor an nx x nw tile"
    if not (np.array_equal(np.asarray(auth.x_index, dtype=np.int64), ex) and np.array_equal(np.asarray(auth.w_index, dtype=np.int64), ew)):
        return "auth: a VU's x row / W column is not the one its index fixes in the committed layout"
    return None


def _ranks(auth: HashAuth) -> dict[str, np.ndarray]:
    return {"a": np.unique(np.asarray(auth.x_index, dtype=np.int64)), "b": np.unique(np.asarray(auth.w_index, dtype=np.int64)),
            "y": np.asarray(auth.vu_index, dtype=np.int64)}


def open_trees(trees: Mapping[str, WordTree], auth: HashAuth) -> list[list[bytes]]:
    """The prover's three multiproofs (in TREE_NAMES order): the distinct x rows, the distinct W columns, the y words."""
    r = _ranks(auth)
    from . import vllm_tree
    return [vllm_tree.multiproof(trees[n], r[n]) if isinstance(trees[n], vllm_tree.VllmTree) else multiproof_siblings(trees[n], r[n])
            for n in TREE_NAMES]


def verify_hash_auth(auth: HashAuth, siblings: Sequence[Sequence[bytes]], *, digests: np.ndarray, y_words: np.ndarray,
                     y_bits: int, leaf=None, vshape: tuple[int, int, int] | None = None) -> tuple[bool, str]:
    """``digests`` (n_vus, 2 x digest_elems): the x digest elements then the W digest elements every VU published (the
    relation proved them); ``y_words`` (n_vus,).  Every VU's leaves must open under the roots (framed by the scheme's
    ``leaf_bytes`` under its ``schema``); two VUs claiming the same leaf must have published the same digest.  Every
    failure is a rejection (no exceptions)."""
    scheme = _leaf(leaf)
    de = scheme.digest_elems
    digests = np.asarray(digests, dtype=np.int64)
    y_words = np.asarray(y_words, dtype=np.int64).ravel()
    n = auth.n_vus
    if digests.shape != (n, 2 * de) or y_words.shape != (n,):
        return False, "auth: digest / word table does not match the VU count"
    if len(siblings) != len(TREE_NAMES):
        return False, "auth: expected one multiproof per tree"
    if (why := layout_error(auth)) is not None:
        return False, why
    if ((digests < 0) | (digests >= h2.P)).any():
        return False, "auth: digest lane out of the field"
    yschema, ybytes = word_schema(y_bits)
    if ((y_words < 0) | (y_words >= 2**y_bits)).any():
        return False, "auth: y word out of range"
    if is_vllm(scheme):
        return _verify_vllm(auth, siblings, digests=digests, y_words=y_words, y_bits=y_bits, scheme=scheme, vshape=vshape)
    per_vu = {"a": (np.asarray(auth.x_index, dtype=np.int64), digests[:, :de]),
              "b": (np.asarray(auth.w_index, dtype=np.int64), digests[:, de:])}
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
                try:
                    val = scheme.leaf_bytes(dg[v])
                except ValueError as e:
                    return False, f"auth: {name}: digest of VU {v} does not frame as a {scheme.name} leaf: {e}"
                if vals.setdefault(r, val) != val:
                    return False, f"auth: {name}: VUs claiming leaf {r} published different digests"
            leaves = {r: domain.leaf(r, r, scheme.schema, val) for r, val in vals.items()}
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


def _verify_vllm(auth: HashAuth, siblings, *, digests, y_words, y_bits: int, scheme, vshape) -> tuple[bool, str]:
    """``verify_hash_auth`` under ``vllm-v1`` (PROTOCOL.md section 9): per port, the leaves at the verifier-fixed positions
    (the published digests for x / W, ``pos_leaf`` of each output word's little-endian bytes for y) fold through the
    port's own multiproof to a tree root whose step-root binding under the verifier's domain is the statement's root."""
    from verity.ml.commitments.vllm_v1 import pos_leaf

    from . import vllm_tree
    if vshape is None:
        return False, "auth: a vllm-v1 statement needs the relation's (K, word_bits, y_bytes)"
    de, n = scheme.digest_elems, auth.n_vus
    yb = vshape[2]
    if yb != ybytes_of(y_bits):
        return False, "auth: y value length is not the relation's"
    per_vu = {"a": (np.asarray(auth.x_index, dtype=np.int64), digests[:, :de]),
              "b": (np.asarray(auth.w_index, dtype=np.int64), digests[:, de:])}
    for name, sibs in zip(TREE_NAMES, siblings):
        ref = auth.tree(name)
        leaves: dict[int, bytes] = {}
        if name == "y":
            for r, w in zip(np.asarray(auth.vu_index, dtype=np.int64).tolist(), y_words.tolist()):
                leaves[int(r)] = pos_leaf(int(w).to_bytes(yb, "little"))
        else:
            ranks, dg = per_vu[name]
            for v in range(n):
                val = scheme.leaf_bytes(dg[v])
                if leaves.setdefault(int(ranks[v]), val) != val:
                    return False, f"auth: {name}: VUs claiming leaf {int(ranks[v])} published different digests"
        try:
            LIMITS.enforce("max_positions", int(ref.count))
            LIMITS.enforce("max_openings", len(leaves))
            LIMITS.enforce("max_proof_bytes", 32 * len(sibs))
        except ResourceLimit as e:
            return False, f"auth: {name}: {e}"
        if leaves and max(leaves) >= ref.count:
            return False, f"auth: {name}: leaf index out of range"
        try:
            tree_root = vllm_tree.fold_multiproof(int(ref.count), leaves, [bytes(s) for s in sibs])
            dom = vllm_domain(name, ref, vshape)
        except ValueError as e:
            return False, f"auth: {name}: multiproof rejected ({e})"
        if dom.bind(tree_root) != ref.root:
            return False, f"auth: {name}: multiproof rejected (step root mismatch)"
    return True, ""


def ybytes_of(y_bits: int) -> int:
    return word_schema(y_bits)[1]
