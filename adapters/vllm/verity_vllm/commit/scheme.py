"""The commitment scheme of the Commit path: ``vllm-v1`` (:mod:`verity.ml.commitments.vllm_v1`, spec ``PROTOCOL.md`` beside it).

Every leaf and root rule the integration hashes comes from here.  Headers, root bindings, tags and the tree walk are core's.
The two rules that run once per leaf keep a prefix-once form over core's tags -- the position leaf (one hash per
``chunk``-byte piece) and the level fold -- and take a hashlib-shaped ``new`` so a benchmark can swap the hash; under
SHA-256 they are core's bytes.  ``tests/commit/test_production_vectors.py`` pins this module to core's conformance vectors
and ``tests/commit/test_scheme_cuda.py`` pins the CUDA copies (``native_tree.cu``, ``hidden_gpu_tree.cu``, the FA2 tap's
``Verity_sha_ctx``), which must hash the same bytes.

The adapters take the integration's argument shapes (a layout object, positional header fields, numpy integers, buffer
digests) and hand core plain ints and bytes.
"""
from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Callable, Sequence
from typing import Any

from verity.ml.commitments import leaves as core_tree
from verity.ml.commitments import vllm_v1 as V

__all__ = ["FA2H_KIND_CHUNK", "FA2H_TAG", "FA2H_VERSION", "LIFT_PREFIX", "NAME", "NODE_PREFIX", "POS_LEAF_TAG", "U64_32", "chunk_header", "chunk_leaf",
           "empty_root", "fold", "fold_levels", "fold_path", "names_digest", "path_in_levels", "path_shape", "pos_leaf", "program_field", "run_root",
           "semantic_root", "step_root", "stream_root", "thread_header", "thread_leaf", "thread_root", "weights_root"]

NAME = V.NAME
POS_LEAF_TAG = V.POS_LEAF_TAG
FA2H_TAG = V.FA2H_TAG
FA2H_VERSION = V.FA2H_VERSION
FA2H_KIND_CHUNK = V.FA2H_KIND_CHUNK

empty_root = core_tree.empty_root
path_shape = core_tree.path_shape
fold_path = core_tree.fold_path
path_in_levels = core_tree.open_path
chunk_leaf = V.chunk_leaf
thread_leaf = V.thread_leaf

Hash = Callable[..., Any]
U64_32 = (32).to_bytes(8, "big")


def _tag_prefix(tag: str) -> bytes:
    t = tag.encode("utf-8")
    return len(t).to_bytes(4, "big") + t


NODE_PREFIX = _tag_prefix(core_tree.NODE_TAG) + U64_32
LIFT_PREFIX = _tag_prefix(core_tree.LIFT_TAG) + U64_32


# --- per-leaf rules (hot paths) --------------------------------------------------------------------------------------------

def pos_leaf(value: bytes | memoryview, new: Hash = hashlib.sha256) -> bytes:
    """Position leaf: ``SHA-256("verity/pos-leaf/v0" || u64be(len(value)) || value)``; a memoryview must be byte-cast."""
    h = new(POS_LEAF_TAG)
    h.update(len(value).to_bytes(8, "big"))
    h.update(value)
    return h.digest()


def fold_levels(level: list[bytes], new: Hash = hashlib.sha256) -> list[list[bytes]]:
    """Level 0 (leaf digests in tree order) up to the root: neighbours pair into ``node``, an odd trailing child is ``lift``ed."""
    levels = [level]
    while len(level) > 1:
        nxt = [new(NODE_PREFIX + level[i] + U64_32 + level[i + 1]).digest() for i in range(0, len(level) - 1, 2)]
        if len(level) % 2 == 1:
            nxt.append(new(LIFT_PREFIX + level[-1]).digest())
        level = nxt
        levels.append(level)
    return levels


def fold(level: list[bytes], new: Hash = hashlib.sha256) -> bytes:
    """The tree root over leaf digests in tree order; ``empty_root()`` for none."""
    if not level:
        return empty_root()
    while len(level) > 1:
        nxt = [new(NODE_PREFIX + level[i] + U64_32 + level[i + 1]).digest() for i in range(0, len(level) - 1, 2)]
        if len(level) % 2 == 1:
            nxt.append(new(LIFT_PREFIX + level[-1]).digest())
        level = nxt
    return level[0]


# --- fa2h headers ----------------------------------------------------------------------------------------------------------

def chunk_header(layout: Any, launch_tag: int, chunk_index: int, chunk_words: int, src_mask: int) -> bytes:
    """The ``fa2h`` chunk-leaf header of ``layout`` (``HB``/``M``/``NB``).  A layout without ``legacy_tile`` (the committers'
    block layouts) is the legacy tile, whose header words 8/9 are 0: the CUDA chunk kernels write nothing else there."""
    ix = operator.index
    tile = {} if getattr(layout, "legacy_tile", True) else {"BN": ix(layout.BN), "D": ix(layout.D)}
    # The kernels write u32 M (M mod 2**32 for a step of 2**32 words or more); the length is bound by the step root's leaf count n.
    return V.chunk_header(launch_tag=ix(launch_tag), chunk_index=ix(chunk_index), chunk_words=ix(chunk_words), src_mask=ix(src_mask),
                          HB=ix(layout.HB), M=ix(layout.M) & 0xFFFFFFFF, NB=ix(layout.NB), **tile)


def thread_header(src_mask: int, slab: int, m_block: int, n_block: int, tidx: int, seqlen_q: int, seqlen_k: int, launch_tag: int,
                  n_block_max: int, ok0: bool, ok1: bool) -> bytes:
    """The 64-byte header the H1 kernel puts in front of every thread leaf (``Verity_sha_ctx::header``)."""
    ix = operator.index
    return V.thread_header(src_mask=ix(src_mask), slab=ix(slab), m_block=ix(m_block), n_block=ix(n_block), tidx=ix(tidx),
                           seqlen_q=ix(seqlen_q), seqlen_k=ix(seqlen_k), launch_tag=ix(launch_tag), n_block_max=ix(n_block_max),
                           ok0=bool(ok0), ok1=bool(ok1))


# --- root bindings ---------------------------------------------------------------------------------------------------------

def program_field(program_digest: str) -> bytes:
    """The 32-byte ``program`` field the roots bind for a Program digest (hex): its first 32 bytes, so SHA-512 truncated to
    256 bits for a Program digest (``codec.program_digest``), and the whole digest for a 32-byte one."""
    d = bytes.fromhex(program_digest)
    if len(d) < 32:
        raise ValueError(f"a Program digest has at least 32 bytes, not {len(d)}")
    return d[:32]


def step_root(program_digest: bytes, ctx_digest: bytes, geo_digest: bytes, layout_digest: bytes, n: int, tree_root: bytes) -> bytes:
    return V.step_root(bytes(program_digest), bytes(ctx_digest), bytes(geo_digest), bytes(layout_digest), operator.index(n), bytes(tree_root))


def run_root(program_digest: bytes, geo_digest: bytes, step_roots: Sequence[bytes]) -> bytes:
    return V.run_root(bytes(program_digest), bytes(geo_digest), [bytes(r) for r in step_roots])


def names_digest(names: Sequence[Sequence[Any]]) -> bytes:
    """The weights root's names digest: SHA-256 of the compact JSON of ``[name, dtype, shape, nbytes]`` per tensor, in order."""
    return hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).digest()


def weights_root(geo_digest: bytes, names: bytes, tensor_roots: Sequence[bytes]) -> bytes:
    return V.weights_root(bytes(geo_digest), bytes(names), [bytes(r) for r in tensor_roots])


def semantic_root(program_digest: bytes, query_id: bytes, template_digest: bytes, ctx_digest: bytes, epoch: int, n: int,
                  tree_root: bytes) -> bytes:
    return V.semantic_root(bytes(program_digest), bytes(query_id), bytes(template_digest), bytes(ctx_digest), operator.index(epoch),
                           operator.index(n), bytes(tree_root))


def stream_root(layout_digest: bytes, launch_tag: int, src_mask: int, chunk_words: int, leaf_count: int, tree_root: bytes) -> bytes:
    ctx = V.stream_ctx_digest(launch_tag=operator.index(launch_tag), src_mask=operator.index(src_mask), chunk_words=operator.index(chunk_words),
                              leaf_count=operator.index(leaf_count))
    return V.stream_root(bytes(layout_digest), ctx, bytes(tree_root))


def thread_root(geo_digest: bytes, launch_tag: int, src_mask: int, tree_root: bytes) -> bytes:
    ctx = V.thread_ctx_digest(launch_tag=operator.index(launch_tag), src_mask=operator.index(src_mask))
    return V.thread_root(bytes(geo_digest), ctx, bytes(tree_root))
