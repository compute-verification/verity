"""The committed-object side of a ``vllm-v1`` statement (``leaf = vllm-v1``, lane b-ligero-vllm-v1): each port (``a`` x rows,
``b`` W columns, ``y`` output words) is a ``vllm-v1`` position-leaf tree (``verity.ml.commitments.vllm_v1``: pairs ``node``, an
odd trailing child ``lift``, no padding) whose published root is its **step-root binding** over a verifier-held
:class:`~verity.ml.commitments.vllm_v1.StepDomain` (PROTOCOL.md §4, §5), never the bare tree root.

The domain of a port (PROTOCOL.md §9.1) is built from the verifier's own data only:

* ``program``: the subcircuit (a GEMM coordinate of ``K`` words) and the scheme;
* ``ctx``: the input set (dataset, tier, manifest digest) and the committed range ``[lo, hi)``, per port -- the
  :class:`~.auth.TreeRef` ``binding`` field, recomputed from the input set by every re-verifier (``reverify.py``) as
  the frame-v3 binding is;
* ``geo``: ``K`` and the value widths;
* ``layout``: the port, its leaf rule (``pos-leaf``, §9.3), what a value is and its exact length (§9.4);
* ``N`` = the tree's ``count``.

``domain_digest`` of every port is absorbed into the statement digest (``hashauth.digest_bytes``) beside the bound root.
Positions are the verifier's: a VU's x / W / y leaves are fixed by its index (``hashauth.layout_error``, §9.5), and a
multiproof's shape is derived from ``(N, ranks)`` alone.

**Multiproof** (statement-defined, PROTOCOL.md §8.2: vllm-v1 has no core multiproof encoding): level by level, the known
nodes in ascending index order; a known even node whose right neighbour is known is paired with it, the last node of an
odd-width level is lifted, any other node takes the next proof sibling.  One leaf's multiproof is its §3 opening path with
the ``null`` (lift) entries dropped.  Siblings of one tree never enter another's fold (§9.7).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from verity.ml.commitments import identity_digest
from verity.ml.commitments import vllm_v1 as core

__all__ = ["TAG", "VALUES", "VllmTree", "build", "ctx_digest", "domain", "fold_multiproof", "multiproof", "port_digests",
           "value_len", "widths", "y_bytes_of"]

TAG = "verity/ligero-b/vllm-v1"
VALUES = {"a": "x-row", "b": "w-column", "y": "y-word-le"}


def _d(kind: str, doc: Mapping) -> bytes:
    return bytes.fromhex(identity_digest(f"{TAG}/{kind}/v1", dict(doc)))


def y_bytes_of(y_bits: int) -> int:
    if y_bits <= 16:
        return 2
    if y_bits <= 32:
        return 4
    raise ValueError("output words wider than 32 bits")


def value_len(tree: str, *, K: int, word_bits: int, y_bytes: int) -> int:
    """Every position of a port holds a value of exactly this many bytes (PROTOCOL.md §9.4)."""
    return y_bytes if tree == "y" else K * word_bits // 8


def ctx_digest(*, dataset: str, tier: str, manifest_sha256: str, lo: int, hi: int, tree: str) -> bytes:
    """The port's context: the input set and range it commits (the tree reference's ``binding``)."""
    if tree not in VALUES:
        raise ValueError(f"unknown tree {tree!r}")
    return _d("ctx", {"dataset": dataset, "tier": tier, "manifest_sha256": manifest_sha256, "lo": int(lo), "hi": int(hi),
                      "tree": tree})


def port_digests(tree: str, *, K: int, word_bits: int, y_bytes: int) -> tuple[bytes, bytes, bytes]:
    """``(program, geo, layout)`` of a port: fixed by the subcircuit and the port, never by the prover."""
    program = _d("program", {"scheme": core.NAME, "subcircuit": "gemm-coordinate", "K": int(K)})
    geo = _d("geo", {"K": int(K), "word_bits": int(word_bits), "row_bytes": K * word_bits // 8, "y_bytes": int(y_bytes)})
    layout = _d("layout", {"tree": tree, "leaf": "pos-leaf", "value": VALUES[tree],
                           "value_len": value_len(tree, K=K, word_bits=word_bits, y_bytes=y_bytes),
                           "word_bits": int(8 * y_bytes if tree == "y" else word_bits)})
    return program, geo, layout


def domain(tree: str, ctx: bytes, count: int, *, K: int, word_bits: int, y_bytes: int) -> core.StepDomain:
    program, geo, layout = port_digests(tree, K=K, word_bits=word_bits, y_bytes=y_bytes)
    return core.StepDomain(program, bytes(ctx), geo, layout, int(count))


def widths(count: int) -> list[int]:
    if count < 1:
        raise ValueError("a vllm-v1 port commits at least one position")
    w = [int(count)]
    while w[-1] > 1:
        w.append((w[-1] + 1) // 2)
    return w


def _levels_host(leaves: np.ndarray) -> list[np.ndarray]:
    lv = [np.ascontiguousarray(leaves, dtype=np.uint8).reshape(-1, 32)]
    nh, lh = hashlib.sha256(_NODE_PREFIX), hashlib.sha256(_LIFT_PREFIX)
    while lv[-1].shape[0] > 1:
        cur = lv[-1]
        n = cur.shape[0]
        out = np.empty(((n + 1) // 2, 32), dtype=np.uint8)
        raw = cur.tobytes()
        for i in range(n // 2):
            h = nh.copy()
            h.update(raw[64 * i:64 * i + 32] + _L32 + raw[64 * i + 32:64 * i + 64])
            out[i] = np.frombuffer(h.digest(), dtype=np.uint8)
        if n % 2:
            h = lh.copy()
            h.update(raw[32 * (n - 1):])
            out[-1] = np.frombuffer(h.digest(), dtype=np.uint8)
        lv.append(out)
    return lv


_L32 = (32).to_bytes(8, "big")
_NODE_PREFIX = len(b"verity-vllm/node/v1").to_bytes(4, "big") + b"verity-vllm/node/v1" + _L32
_LIFT_PREFIX = len(b"verity-vllm/lift/v1").to_bytes(4, "big") + b"verity-vllm/lift/v1" + _L32


class VllmTree:
    """One port's tree: ``ref`` (:class:`~.auth.TreeRef`: binding = ctx digest, count = N, root = the step-root binding)
    and ``levels`` (level 0 = the leaf digests, widths N, ceil(N / 2), ..., 1)."""

    def __init__(self, ref, levels: Sequence[np.ndarray], dom: core.StepDomain):
        self.ref, self._levels, self.dom = ref, list(levels), dom
        if [lv.shape[0] for lv in self._levels] != widths(ref.count):
            raise ValueError("levels do not form a vllm-v1 tree of the declared count")
        if dom.leaf_count != ref.count or dom.ctx_digest != ref.binding or dom.bind(self.tree_root) != ref.root:
            raise ValueError("stored root is not the step-root binding of the levels under the domain")

    @property
    def levels(self) -> list[np.ndarray]:
        return self._levels

    @property
    def tree_root(self) -> bytes:
        return bytes(self._levels[-1][0])

    @property
    def root(self) -> bytes:
        return self.ref.root

    @property
    def count(self) -> int:
        return self.ref.count

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for lv, arr in enumerate(self._levels):
            np.save(directory / f"{self.ref.name}.V{lv:02d}.npy", arr)
        (directory / f"{self.ref.name}.json").write_text(json.dumps({**self.ref.to_json(), "tree": core.NAME}, indent=1))

    @classmethod
    def load(cls, directory: Path, name: str, dom_of) -> VllmTree:
        from .auth import TreeRef
        directory = Path(directory)
        d = json.loads((directory / f"{name}.json").read_text())
        if d.get("tree") != core.NAME:
            raise ValueError("not a vllm-v1 tree")
        ref = TreeRef.from_json(d)
        levels = [np.load(directory / f"{name}.V{lv:02d}.npy") for lv in range(len(widths(ref.count)))]
        return cls(ref, levels, dom_of(name, ref.binding, ref.count))


def build(name: str, ctx: bytes, leaves, dom: core.StepDomain) -> VllmTree:
    """The tree over ``leaves`` ((N, 32) uint8 digests in position order; host numpy or a device array) under ``dom``."""
    from . import frame_gpu
    from .auth import OWNER, TreeRef
    count = int(np.shape(leaves)[0])
    if dom.leaf_count != count or dom.ctx_digest != bytes(ctx):
        raise ValueError("the domain does not name this tree's context and size")
    if frame_gpu.available() and count > 1:
        with frame_gpu._torch_stream():
            levels = frame_gpu.fv3.vllm_tree(frame_gpu.cp.ascontiguousarray(frame_gpu.cp.asarray(leaves), dtype=frame_gpu.cp.uint8)).levels
    else:
        levels = _levels_host(np.asarray(leaves))
    tree_root = bytes(levels[-1][0])
    ref = TreeRef(name, bytes(ctx), OWNER[name], count, dom.bind(tree_root))
    return VllmTree(ref, levels, dom)


def multiproof(tree: VllmTree, ranks) -> list[bytes]:
    """The prover's siblings for the distinct ``ranks`` (the rule in the module doc)."""
    known = sorted({int(r) for r in np.asarray(ranks, dtype=np.int64).ravel().tolist()})
    if known and (known[0] < 0 or known[-1] >= tree.count):
        raise ValueError("rank out of range")
    out: list[bytes] = []
    for lv, w in enumerate(widths(tree.count)[:-1]):
        level = tree.levels[lv]
        ks = set(known)
        nxt = []
        for i in known:
            if i % 2 == 1 and (i - 1) in ks:
                continue
            if i % 2 == 0 and (i + 1) in ks:
                pass
            elif i == w - 1 and w % 2:
                pass
            else:
                out.append(bytes(level[i ^ 1]))
            nxt.append(i >> 1)
        known = nxt
    return out


def fold_multiproof(count: int, leaves: Mapping[int, bytes], siblings: Sequence[bytes]) -> bytes:
    """The tree root from ``leaves`` ({rank: 32-byte leaf}) and the siblings; ``ValueError`` on any malformed input (a rank
    out of range, a short or long proof, a sibling that is not 32 bytes)."""
    if not leaves:
        raise ValueError("no leaves")
    nodes = sorted((int(r), bytes(v)) for r, v in leaves.items())
    if nodes[0][0] < 0 or nodes[-1][0] >= count or any(len(v) != 32 for _, v in nodes):
        raise ValueError("leaf rank out of range or malformed leaf")
    if any(len(s) != 32 for s in siblings):
        raise ValueError("malformed sibling")
    cur = 0
    for w in widths(count)[:-1]:
        nxt: list[tuple[int, bytes]] = []
        at = 0
        while at < len(nodes):
            i, v = nodes[at]
            if i % 2 == 0 and at + 1 < len(nodes) and nodes[at + 1][0] == i + 1:
                nxt.append((i >> 1, hashlib.sha256(_NODE_PREFIX + v + _L32 + nodes[at + 1][1]).digest()))
                at += 2
                continue
            if i == w - 1 and w % 2:
                nxt.append((i >> 1, hashlib.sha256(_LIFT_PREFIX + v).digest()))
            else:
                if cur >= len(siblings):
                    raise ValueError("multiproof too short")
                s = bytes(siblings[cur])
                cur += 1
                pair = v + _L32 + s if i % 2 == 0 else s + _L32 + v
                nxt.append((i >> 1, hashlib.sha256(_NODE_PREFIX + pair).digest()))
            at += 1
        nodes = nxt
    if cur != len(siblings):
        raise ValueError("multiproof too long")
    return nodes[0][1]
