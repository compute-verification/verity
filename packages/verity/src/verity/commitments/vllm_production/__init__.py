"""``vllm-production/v1``: the commitment framing vLLM's production Commit path emits today, as a named scheme.

Spec: ``PROTOCOL.md`` beside this file.  Conformance vectors: ``vectors.json`` beside it, written by
``python -m verity.commitments.vllm_production.vectors``.  This module is the plain reference: stdlib ``hashlib``
only, no fast paths.  It changes no root; every byte below is what the integration and its CUDA kernels hash.

The tree rule (pairs ``node``, an odd trailing child ``lift``, zero leaves ``empty``) is the one :mod:`..leaves`
already carries for the ``verity-vllm/leaf/v1`` id leaf, and is reused from there.  What this scheme adds is the
production leaves -- the position leaf and the two ``fa2h`` headered leaves -- and the raw-SHA-256 root bindings
that tie a tree root to its program, context, geometry and size.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

from verity.errors import InvalidArtifact

from ..leaves import DIGEST_BYTES, empty_root, fold_path, levels, lift_hash, node_hash, open_path, path_shape
from ..scheme import Path

__all__ = [
    "FA2H_KIND_CHUNK",
    "FA2H_TAG",
    "FA2H_VERSION",
    "NAME",
    "POS_LEAF_TAG",
    "RUN_ROOT_TAG",
    "SCHEME",
    "SEMANTIC_ROOT_TAG",
    "STEP_ROOT_TAG",
    "STREAM_ROOT_TAG",
    "THREAD_ROOT_TAG",
    "WEIGHTS_ROOT_TAG",
    "VllmProduction",
    "chunk_header",
    "chunk_leaf",
    "fold",
    "pos_leaf",
    "run_root",
    "semantic_root",
    "step_root",
    "stream_ctx_digest",
    "stream_root",
    "thread_ctx_digest",
    "thread_header",
    "thread_leaf",
    "thread_root",
    "verify_opening",
    "weights_root",
]

NAME = "vllm-production/v1"

POS_LEAF_TAG = b"verity/pos-leaf/v0"
FA2H_TAG = 0x68326166  # b"fa2h" read as a little-endian u32
FA2H_VERSION = 7
FA2H_KIND_CHUNK = 1

SEMANTIC_ROOT_TAG = b"verity/semantic-root/v0"
STEP_ROOT_TAG = b"verity/fa2c/root/v0"
RUN_ROOT_TAG = b"verity/cmt-integ/run-root/v0"
WEIGHTS_ROOT_TAG = b"verity/cmt-integ/weights-root/v0"
STREAM_ROOT_TAG = b"verity/fa2-hidden/root/v7"
THREAD_ROOT_TAG = b"verity/fa2-hidden/thread-root/v7"

_U32 = 1 << 32
_U64 = 1 << 64


def _u32(value: int, name: str, bound: int = _U32) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < bound:
        raise InvalidArtifact(f"{name} must be an integer in [0, {bound}), got {value!r}")
    return value


def _u64be(value: int, name: str) -> bytes:
    return _u32(value, name, _U64).to_bytes(8, "big")


def _digest(value: bytes, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != DIGEST_BYTES:
        raise InvalidArtifact(f"{name} must be a 32-byte digest")
    return value


def _words(words: Sequence[int]) -> bytes:
    return b"".join(w.to_bytes(4, "little") for w in words)


# --- leaves ------------------------------------------------------------------------------------------------------


def pos_leaf(value: bytes) -> bytes:
    """``SHA-256("verity/pos-leaf/v0" || u64be(len(value)) || value)``: a position leaf, identified by its index."""

    value = bytes(value)
    return hashlib.sha256(POS_LEAF_TAG + _u64be(len(value), "value length") + value).digest()


def chunk_header(
    *, launch_tag: int, chunk_index: int, chunk_words: int, src_mask: int, HB: int, M: int, NB: int, BN: int = 0, D: int = 0
) -> bytes:
    """The 64-byte ``fa2h`` chunk-leaf header: sixteen little-endian u32 words.

    ``BN``/``D`` are 0 for the legacy (kBlockN 128, head dim 64) tile, the kernel tile otherwise.
    """

    return _words(
        [
            FA2H_TAG,
            (FA2H_VERSION << 16) | (FA2H_KIND_CHUNK << 8) | _u32(src_mask, "src_mask", 1 << 8),
            _u32(launch_tag, "launch_tag"),
            _u32(chunk_index, "chunk_index"),
            _u32(chunk_words, "chunk_words"),
            _u32(HB, "HB"),
            _u32(M, "M"),
            _u32(NB, "NB"),
            _u32(BN, "BN"),
            _u32(D, "D"),
            0, 0, 0, 0, 0, 0,
        ]
    )


def chunk_leaf(header: bytes, chunk: bytes) -> bytes:
    """``SHA-256(header || chunk)``; the chunk is ``4 * chunk_words`` bytes of little-endian u32 stream words."""

    if len(header) != 64:
        raise InvalidArtifact("fa2h header must be 64 bytes")
    return hashlib.sha256(bytes(header) + bytes(chunk)).digest()


def thread_header(
    *,
    src_mask: int,
    slab: int,
    m_block: int,
    n_block: int,
    tidx: int,
    seqlen_q: int,
    seqlen_k: int,
    launch_tag: int,
    n_block_max: int,
    ok0: bool,
    ok1: bool,
) -> bytes:
    """The 64-byte ``fa2h`` header an FA2 kernel thread puts before its words (word 1 carries no kind byte)."""

    return _words(
        [
            FA2H_TAG,
            (FA2H_VERSION << 16) | _u32(src_mask, "src_mask", 1 << 16),
            0,
            _u32(slab, "slab"),
            _u32(m_block, "m_block"),
            _u32(n_block, "n_block"),
            _u32(tidx, "tidx"),
            _u32(seqlen_q, "seqlen_q"),
            _u32(seqlen_k, "seqlen_k"),
            _u32(launch_tag, "launch_tag"),
            _u32(n_block_max, "n_block_max"),
            (1 if ok0 else 0) | (2 if ok1 else 0),
            0, 0, 0, 0,
        ]
    )


def thread_leaf(message: bytes) -> bytes:
    """``SHA-256(message)``, where ``message`` is a thread header followed by the thread's words."""

    if len(message) < 64:
        raise InvalidArtifact("thread leaf message is shorter than its header")
    return hashlib.sha256(bytes(message)).digest()


# --- tree --------------------------------------------------------------------------------------------------------


def fold(leaves: Sequence[bytes]) -> bytes:
    """The tree root over leaf digests in tree order; ``H("verity-vllm/empty/v1")`` for none."""

    if not leaves:
        return empty_root()
    return levels([_digest(leaf, "leaf") for leaf in leaves])[-1][0]


def verify_opening(
    root: bytes,
    leaf_count: int,
    index: int,
    leaf: bytes,
    path: Sequence[bytes | None],
    bind: Callable[[bytes], bytes] | None = None,
) -> bool:
    """Authenticate ``leaf`` at ``index``: the path shape comes from ``(leaf_count, index)``, the fold from the
    path, and ``bind`` (the verifier's own root binding over the tree root, identity when omitted) must give
    ``root``.  Returns ``False`` for any malformed input."""

    try:
        expected = path_shape(leaf_count, index)
        if len(path) != len(expected):
            return False
        for sibling, has_sibling in zip(path, expected, strict=True):
            if has_sibling != (sibling is not None):
                return False
            if sibling is not None:
                _digest(sibling, "sibling")
        tree_root = fold_path(_digest(leaf, "leaf"), index, path)
        return (bind(tree_root) if bind else tree_root) == root
    except (InvalidArtifact, TypeError, ValueError):
        return False


class VllmProduction:
    """The :class:`~verity.commitments.scheme.CommitmentScheme` of ``vllm-production/v1`` (tree rule only)."""

    name = NAME

    def root(self, leaves: Sequence[bytes]) -> bytes:
        return fold(leaves)

    def path(self, leaves: Sequence[bytes], index: int) -> Path:
        path_shape(len(leaves), index)
        return open_path(levels([_digest(leaf, "leaf") for leaf in leaves]), index)

    def verify_path(self, root: bytes, leaf_count: int, index: int, leaf: bytes, path: Sequence[bytes | None]) -> bool:
        return verify_opening(root, leaf_count, index, leaf, path)


SCHEME = VllmProduction()


# --- root bindings -----------------------------------------------------------------------------------------------


def semantic_root(
    program_digest: bytes, query_id: bytes, template_digest: bytes, ctx_digest: bytes, epoch: int, n: int, tree_root: bytes
) -> bytes:
    """``SHA-256(tag || program || query_id || template || ctx || u64be epoch || u64be N || tree_root)``.

    ``query_id`` is the only variable-length field; everything else is fixed width, so the preimage parses one way.
    """

    return hashlib.sha256(
        SEMANTIC_ROOT_TAG
        + _digest(program_digest, "program_digest")
        + bytes(query_id)
        + _digest(template_digest, "template_digest")
        + _digest(ctx_digest, "ctx_digest")
        + _u64be(epoch, "epoch")
        + _u64be(n, "N")
        + _digest(tree_root, "tree_root")
    ).digest()


def step_root(program_digest: bytes, ctx_digest: bytes, geo_digest: bytes, layout_digest: bytes, n: int, tree_root: bytes) -> bytes:
    """``SHA-256("verity/fa2c/root/v0" || program || ctx || geo || layout || u64be N || tree_root)``."""

    return hashlib.sha256(
        STEP_ROOT_TAG
        + _digest(program_digest, "program_digest")
        + _digest(ctx_digest, "ctx_digest")
        + _digest(geo_digest, "geo_digest")
        + _digest(layout_digest, "layout_digest")
        + _u64be(n, "N")
        + _digest(tree_root, "tree_root")
    ).digest()


def run_root(program_digest: bytes, geo_digest: bytes, step_roots: Sequence[bytes]) -> bytes:
    """``SHA-256("verity/cmt-integ/run-root/v0" || program || geo || u64be S || fold(step_roots))``."""

    return hashlib.sha256(
        RUN_ROOT_TAG
        + _digest(program_digest, "program_digest")
        + _digest(geo_digest, "geo_digest")
        + _u64be(len(step_roots), "S")
        + fold(step_roots)
    ).digest()


def weights_root(geo_digest: bytes, names_digest: bytes, tensor_roots: Sequence[bytes]) -> bytes:
    """``SHA-256("verity/cmt-integ/weights-root/v0" || geo || names || u64be T || fold(tensor_roots))``."""

    return hashlib.sha256(
        WEIGHTS_ROOT_TAG
        + _digest(geo_digest, "geo_digest")
        + _digest(names_digest, "names_digest")
        + _u64be(len(tensor_roots), "T")
        + fold(tensor_roots)
    ).digest()


def stream_ctx_digest(*, launch_tag: int, src_mask: int, chunk_words: int, leaf_count: int) -> bytes:
    """SHA-256 of the ASCII text ``{"chunk_words": C, "launch_tag": T, "leaf_count": N, "src_mask": S}`` (decimal)."""

    text = '{"chunk_words": %d, "launch_tag": %d, "leaf_count": %d, "src_mask": %d}' % (
        _u32(chunk_words, "chunk_words"), _u32(launch_tag, "launch_tag"), _u32(leaf_count, "leaf_count", _U64), _u32(src_mask, "src_mask", 1 << 8)
    )
    return hashlib.sha256(text.encode("ascii")).digest()


def thread_ctx_digest(*, launch_tag: int, src_mask: int) -> bytes:
    """SHA-256 of the ASCII text ``{"launch_tag": T, "src_mask": S}`` (decimal)."""

    text = '{"launch_tag": %d, "src_mask": %d}' % (_u32(launch_tag, "launch_tag"), _u32(src_mask, "src_mask", 1 << 16))
    return hashlib.sha256(text.encode("ascii")).digest()


def stream_root(layout_digest: bytes, ctx_digest: bytes, tree_root: bytes) -> bytes:
    """``SHA-256("verity/fa2-hidden/root/v7" || layout || ctx || tree_root)`` for an ``fa2h`` chunk tree."""

    return hashlib.sha256(
        STREAM_ROOT_TAG + _digest(layout_digest, "layout_digest") + _digest(ctx_digest, "ctx_digest") + _digest(tree_root, "tree_root")
    ).digest()


def thread_root(geo_digest: bytes, ctx_digest: bytes, tree_root: bytes) -> bytes:
    """``SHA-256("verity/fa2-hidden/thread-root/v7" || geo || ctx || tree_root)`` for an ``fa2h`` thread tree."""

    return hashlib.sha256(
        THREAD_ROOT_TAG + _digest(geo_digest, "geo_digest") + _digest(ctx_digest, "ctx_digest") + _digest(tree_root, "tree_root")
    ).digest()
