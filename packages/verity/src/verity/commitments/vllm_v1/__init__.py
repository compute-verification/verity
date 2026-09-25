"""``vllm-v1``: the commitment framing vLLM's production Commit path emits, as a named scheme.

A permanent scheme beside frame-v3 (:mod:`..merkle`), selectable by configured backends.  Spec: ``PROTOCOL.md``
beside this file.  Conformance vectors: ``vectors.json`` beside it, written by
``python -m verity.commitments.vllm_v1.vectors``.  This module is the plain reference: stdlib ``hashlib``
only, no fast paths; every byte below is what the integration and its CUDA kernels hash.

The tree rule (pairs ``node``, an odd trailing child ``lift``, zero leaves ``empty``) and the id leaf are the ones
:mod:`..leaves` carries, reused from there.  No root binds which leaf rule a tree uses, nor (for position leaves)
which position a leaf sits at: the verifier fixes both.  The domain classes at the end of this module
(:class:`SemanticDomain`, :class:`StepDomain`, :class:`StreamDomain`, :class:`ThreadDomain`) carry everything a
verifier must hold itself -- context, size, leaf rule, position arithmetic -- and :func:`verify` takes one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from verity.errors import InvalidArtifact

from ..leaves import DIGEST_BYTES, empty_root, fold_path, levels, open_path, path_shape
from ..leaves import ids_digest as ids_digest
from ..leaves import leaf_hash as id_leaf
from ..leaves import LEAF_TAG, shape_text
from ..scheme import LeafLayout, Path

__all__ = [
    "FA2H_KIND_CHUNK",
    "FA2H_TAG",
    "FA2H_VERSION",
    "NAME",
    "POS_LEAF_TAG",
    "RUN_ROOT_TAG",
    "TREE",
    "Tree",
    "VllmV1",
    "id_leaf_layout",
    "pos_leaf_layout",
    "SEMANTIC_ROOT_TAG",
    "SemanticDomain",
    "StepDomain",
    "StreamDomain",
    "ThreadDomain",
    "THREADS_PER_BLOCK",
    "id_leaf",
    "ids_digest",
    "thread_leaf_index",
    "verify",
    "STEP_ROOT_TAG",
    "STREAM_ROOT_TAG",
    "THREAD_ROOT_TAG",
    "WEIGHTS_ROOT_TAG",
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

NAME = "vllm-v1"

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
    """The 64-byte ``fa2h`` header an FA2 kernel thread puts before its words.

    Word 1's kind byte (bits 8-15) is 0; ``src_mask`` < 2**8 keeps it so, and a thread header distinct from every
    chunk header (kind 1).
    """

    return _words(
        [
            FA2H_TAG,
            (FA2H_VERSION << 16) | _u32(src_mask, "src_mask", 1 << 8),
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


class Tree:
    """The tree rule alone, over leaf digests in tree order (no root binding): what :class:`VllmV1` folds with."""

    def root(self, leaves: Sequence[bytes]) -> bytes:
        return fold(leaves)

    def path(self, leaves: Sequence[bytes], index: int) -> Path:
        path_shape(len(leaves), index)
        return open_path(levels([_digest(leaf, "leaf") for leaf in leaves]), index)

    def verify_path(self, root: bytes, leaf_count: int, index: int, leaf: bytes, path: Sequence[bytes | None]) -> bool:
        return verify_opening(root, leaf_count, index, leaf, path)


TREE = Tree()


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

    text = '{"launch_tag": %d, "src_mask": %d}' % (_u32(launch_tag, "launch_tag"), _u32(src_mask, "src_mask", 1 << 8))
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


# --- verifier-held domains ---------------------------------------------------------------------------------------
#
# A domain is what the verifier derives from its own trusted data and never takes from the prover: the root
# binding's context fields, the tree size, the leaf rule, and (for headered leaves) the position arithmetic.
# ``verify`` authenticates a leaf *digest* the caller recomputed through the domain's own ``leaf`` method.

THREADS_PER_BLOCK = 128


@dataclass(frozen=True, slots=True)
class SemanticDomain:
    """A C0 position-leaf tree bound by the semantic root.  Leaf rule: :func:`pos_leaf`."""

    program_digest: bytes
    query_id: bytes
    template_digest: bytes
    ctx_digest: bytes
    epoch: int
    leaf_count: int

    def leaf(self, index: int, value: bytes) -> bytes:
        return pos_leaf(value)

    def leaf_layout(self, index: int, nbytes: int) -> LeafLayout:
        return pos_leaf_layout(nbytes)

    def bind(self, tree_root: bytes) -> bytes:
        return semantic_root(self.program_digest, self.query_id, self.template_digest, self.ctx_digest, self.epoch, self.leaf_count, tree_root)


@dataclass(frozen=True, slots=True)
class StepDomain:
    """A native-host step tree bound by the step root.

    ``chunk`` fixes the leaf rule: ``None`` for position leaves (host path), else the ``fa2h`` chunk-header fields
    of the GPU tree except ``chunk_index``, which is the leaf's index.  A step whose leaves use two header field sets
    (an executed block followed by a padding block) needs the caller to recompute each leaf with its block's fields.
    """

    program_digest: bytes
    ctx_digest: bytes
    geo_digest: bytes
    layout_digest: bytes
    leaf_count: int
    chunk: dict[str, int] | None = None

    def leaf(self, index: int, value: bytes) -> bytes:
        if self.chunk is None:
            return pos_leaf(value)
        if len(value) != 4 * self.chunk["chunk_words"]:
            raise InvalidArtifact("chunk leaf value must be 4 * chunk_words bytes")
        return chunk_leaf(chunk_header(chunk_index=index, **self.chunk), value)

    def leaf_layout(self, index: int, nbytes: int | None = None) -> LeafLayout:
        """``nbytes`` is the position leaf's value length (host path); the chunk rule fixes its own."""

        if self.chunk is None:
            if nbytes is None:
                raise InvalidArtifact("a position leaf's layout needs its value length")
            return pos_leaf_layout(nbytes)
        return LeafLayout("sha256", chunk_header(chunk_index=index, **self.chunk), 4 * self.chunk["chunk_words"])

    def bind(self, tree_root: bytes) -> bytes:
        return step_root(self.program_digest, self.ctx_digest, self.geo_digest, self.layout_digest, self.leaf_count, tree_root)


@dataclass(frozen=True, slots=True)
class StreamDomain:
    """An ``fa2h`` M1 chunk tree bound by the stream root; the header fixes each leaf's chunk index."""

    layout_digest: bytes
    launch_tag: int
    src_mask: int
    chunk_words: int
    leaf_count: int
    HB: int
    M: int
    NB: int
    BN: int = 0
    D: int = 0

    def leaf(self, index: int, value: bytes) -> bytes:
        if len(value) != 4 * self.chunk_words:
            raise InvalidArtifact("chunk leaf value must be 4 * chunk_words bytes")
        header = chunk_header(launch_tag=self.launch_tag, chunk_index=index, chunk_words=self.chunk_words, src_mask=self.src_mask,
                              HB=self.HB, M=self.M, NB=self.NB, BN=self.BN, D=self.D)
        return chunk_leaf(header, value)

    def leaf_layout(self, index: int) -> LeafLayout:
        header = chunk_header(launch_tag=self.launch_tag, chunk_index=index, chunk_words=self.chunk_words, src_mask=self.src_mask,
                              HB=self.HB, M=self.M, NB=self.NB, BN=self.BN, D=self.D)
        return LeafLayout("sha256", header, 4 * self.chunk_words)

    def bind(self, tree_root: bytes) -> bytes:
        ctx = stream_ctx_digest(launch_tag=self.launch_tag, src_mask=self.src_mask, chunk_words=self.chunk_words, leaf_count=self.leaf_count)
        return stream_root(self.layout_digest, ctx, tree_root)


def thread_leaf_index(slab: int, m_block: int, n_block: int, tidx: int, *, num_m_block: int, NB: int) -> int:
    """``((slab * num_m_block + m_block) * NB + n_block) * 128 + tidx``: a thread leaf's position in its tree."""

    if not (0 <= m_block < num_m_block and 0 <= n_block < NB and 0 <= tidx < THREADS_PER_BLOCK and slab >= 0):
        raise InvalidArtifact("thread position outside the geometry")
    return ((slab * num_m_block + m_block) * NB + n_block) * THREADS_PER_BLOCK + tidx


@dataclass(frozen=True, slots=True)
class ThreadDomain:
    """An ``fa2h`` H1 thread tree bound by the thread root.

    ``N = HB * num_m_block * NB * 128`` is not in the root: it is bound only through ``geo_digest``, which the
    verifier must derive from the same geometry these fields come from.  Unvisited slots hold 32 zero bytes, which
    no message hashes to: the verifier derives the visited set from the geometry and never opens those slots.
    """

    geo_digest: bytes
    launch_tag: int
    src_mask: int
    HB: int
    num_m_block: int
    NB: int

    @property
    def leaf_count(self) -> int:
        return self.HB * self.num_m_block * self.NB * THREADS_PER_BLOCK

    def index(self, slab: int, m_block: int, n_block: int, tidx: int) -> int:
        if not 0 <= slab < self.HB:
            raise InvalidArtifact("slab outside the geometry")
        return thread_leaf_index(slab, m_block, n_block, tidx, num_m_block=self.num_m_block, NB=self.NB)

    def leaf(self, message: bytes, *, slab: int, m_block: int, n_block: int, tidx: int, seqlen_q: int, seqlen_k: int,
             n_block_max: int, ok0: bool, ok1: bool) -> bytes:
        """The leaf of the thread at the claimed position; refuses a message whose header is not the derived one."""

        header = thread_header(src_mask=self.src_mask, slab=slab, m_block=m_block, n_block=n_block, tidx=tidx, seqlen_q=seqlen_q,
                               seqlen_k=seqlen_k, launch_tag=self.launch_tag, n_block_max=n_block_max, ok0=ok0, ok1=ok1)
        if bytes(message[:64]) != header:
            raise InvalidArtifact("thread message header is not the one derived from the claimed position")
        return thread_leaf(message)

    def leaf_layout(self, n_words: int, *, slab: int, m_block: int, n_block: int, tidx: int, seqlen_q: int, seqlen_k: int,
                    n_block_max: int, ok0: bool, ok1: bool) -> LeafLayout:
        """The thread's header is the prefix; its ``n_words`` register words (fixed by ``src_mask``) are the value."""

        header = thread_header(src_mask=self.src_mask, slab=slab, m_block=m_block, n_block=n_block, tidx=tidx, seqlen_q=seqlen_q,
                               seqlen_k=seqlen_k, launch_tag=self.launch_tag, n_block_max=n_block_max, ok0=ok0, ok1=ok1)
        return LeafLayout("sha256", header, 4 * n_words)

    def bind(self, tree_root: bytes) -> bytes:
        return thread_root(self.geo_digest, thread_ctx_digest(launch_tag=self.launch_tag, src_mask=self.src_mask), tree_root)


def verify(
    domain: SemanticDomain | StepDomain | StreamDomain | ThreadDomain,
    root: bytes,
    index: int,
    leaf: bytes,
    path: Sequence[bytes | None],
) -> bool:
    """Authenticate a recomputed ``leaf`` at ``index`` against ``root`` under the verifier's ``domain``."""

    return verify_opening(root, domain.leaf_count, index, leaf, path, domain.bind)


# --- leaf layouts and the configured scheme ----------------------------------------------------------------------


def pos_leaf_layout(nbytes: int) -> LeafLayout:
    """``pos_leaf``: prefix ``"verity/pos-leaf/v0" || u64be(nbytes)`` (26 bytes, not block aligned), no suffix."""

    return LeafLayout("sha256", POS_LEAF_TAG + _u64be(nbytes, "value length"), nbytes)


def id_leaf_layout(canonical_id: str, dtype: str, shape: Sequence[int], nbytes: int) -> LeafLayout:
    """The capture-v1 id leaf ``H(leaf/v1, id, dtype, shape text, value)``: everything before the value is public."""

    tag = LEAF_TAG.encode("utf-8")
    out = len(tag).to_bytes(4, "big") + tag
    for part in (canonical_id.encode("utf-8"), dtype.encode("utf-8"), shape_text(shape).encode("utf-8")):
        out += len(part).to_bytes(8, "big") + part
    return LeafLayout("sha256", out + _u64be(nbytes, "value length"), nbytes)


Domain = SemanticDomain | StepDomain | StreamDomain | ThreadDomain


class VllmV1:
    """``vllm-v1`` over a statement's ports, each a verifier-held domain; ``root`` is the published (bound) root.

    Position-leaf ports need ``value_nbytes[port](position)``: the value length the layout (template or layout digest)
    assigns to that position.  Thread ports have no position-only layout (their header needs the geometry's row
    validity); use :meth:`ThreadDomain.leaf_layout`.  The tree leaf is the proven digest itself.
    """

    name = NAME
    hashes = ("sha256",)

    def __init__(self, ports: Mapping[str, Domain], value_nbytes: Mapping[str, Callable[[int], int]] | None = None) -> None:
        self.ports = dict(ports)
        self.value_nbytes = dict(value_nbytes or {})

    def _domain(self, port: str) -> Domain:
        try:
            return self.ports[port]
        except KeyError as error:
            raise KeyError(f"{port!r} is not a port of this configuration") from error

    def leaf_layout(self, port: str, position: int) -> LeafLayout:
        domain = self._domain(port)
        if not 0 <= position < domain.leaf_count:
            raise InvalidArtifact(f"position {position} outside the {domain.leaf_count}-leaf tree of {port!r}")
        if isinstance(domain, ThreadDomain):
            raise InvalidArtifact("thread leaves need their geometry: use ThreadDomain.leaf_layout(n_words, **position)")
        if isinstance(domain, StreamDomain) or (isinstance(domain, StepDomain) and domain.chunk is not None):
            return domain.leaf_layout(position)
        if port not in self.value_nbytes:
            raise InvalidArtifact(f"{port!r} has position leaves: configure value_nbytes[{port!r}]")
        return pos_leaf_layout(self.value_nbytes[port](position))

    def tree_leaf(self, port: str, position: int, digest: bytes) -> bytes:
        self._domain(port)
        return _digest(bytes(digest), "leaf digest")

    def root(self, port: str, leaves: Sequence[bytes]) -> bytes:
        domain = self._domain(port)
        if len(leaves) != domain.leaf_count:
            raise InvalidArtifact(f"{port!r} has {domain.leaf_count} leaves, got {len(leaves)}")
        return domain.bind(fold(leaves))

    def path(self, port: str, leaves: Sequence[bytes], index: int) -> Path:
        self._domain(port)
        return TREE.path(leaves, index)

    def verify_path(self, port: str, root: bytes, leaf_count: int, index: int, leaf: bytes, path: Sequence[bytes | None]) -> bool:
        domain = self.ports.get(port)
        if domain is None or leaf_count != domain.leaf_count:
            return False
        return verify(domain, root, index, leaf, path)
