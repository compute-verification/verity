"""load-relay: the network certifier's relay of the weights' load to the H200 and its commit of what crosses, in the exact
form the units read it. The certifier reads the load from a source (a file, a pipe, another host's stream), writes every
byte on to the sink (the H200's) as it reads it, and commits every tensor of CMP-95 interface 1's manifest as a
registered value, as the reading program's units read it (interface 1's registration for a program named by its
digest): an f32 scale one 32-bit leaf a row, every other tensor 16-bit leaves, a linear's row ``row_codes`` codes, in
``hm96-sha512/row/v2`` rows but ``tok``'s, one token's embedding a ``hm96-sha512/row-seg/v1`` row (`EmbeddingRead_v1`'s);
each row under its own salt, in a frame-v3-sha512 tree under the value's domain, the root `Registered.commit`'s
(`transfer_rows.Transfer`). It then signs
each value's head (`transfer_rows.TransferHead`: kind ``load``, ingress, window 0, the server session's) and writes the
registrations the verifier pins each load value from (`blocks.pins`).

    python benchmarks/network_traces/load_relay.py relay --program HEX --out DIR [--shape h200|tiny] [--format plain|p2]
      [--source PATH|-] [--sink PATH|-|none] [--salt HEX] [--witness PATH] [--key DIR] [--session HEX] [--workers N]
    python benchmarks/network_traces/load_relay.py bench --out DIR [--shape h200] [--bytes N] [--workers 1,16]

* **The link** (``--format``):
  - ``plain``: interface 1's weights stream, every tensor's bytes in manifest order (139,736,443,904 B at H200);
  - ``p2``: CMP-95 interface 2's store, C* in block order (``p2-16448/v4``: 1,038 segments of 65,536 blocks of 2,056
    bytes, 139,862,212,608 B; the segment files back to back), then every tensor the store doesn't hold (the scales,
    norms and embeddings, 36,760,576 B) back to back in manifest order. The store's tensors are interface 1's s8
    matrices: a layer's ``qkv_proj`` is ``wq ‖ wk ‖ wv`` and its ``o_proj``, ``up_proj`` and ``down_proj`` are ``wo``,
    ``wup`` and ``wdown``; ``lm_head`` is ``head`` (`store_tensors`). The certifier decodes each block a value's rows
    fall in (`pous.schemes.p2`'s keys after ``--salt``, the store's, and ``m = sq(σ_K(sq(c))) − t mod p``), refusing
    a block that isn't a codeword and a nonzero byte past a store tensor's end, and commits the decoded bytes. The
    store's blocks past its last tensor are relayed and not decoded: no unit reads them.
* **Streaming over cores** (`relay`): the main process reads the link in order, and each value's aligned chunks of
  2^b rows (at most `CHUNK_BYTES`; a value smaller than a chunk is one, its pads included) go to ``workers`` processes as
  their bytes arrive. Each returns its chunk's subtree root, and the main process folds them as `tree_levels` does. A P2
  chunk takes the blocks its bytes fall in, so a block two chunks share is decoded twice.
* **The salts** are the operating system's, 192 bytes a row, drawn by the worker that commits the row (`os_salts`):
  every hm96 salt is the operating system's (Daniel, 9 Oct, decisions 4(a) and 4(c); A3 in
  ``Security/Proofs/Flock/Soundness/Assumptions.lean``), so nothing rests on a stream generator. ``--witness`` is
  where they go, every value's in manifest order, its rows in order (each value's `Transfer.witness`): the prover's
  alone, 104.8 GB at H200. `keyed_salts` is interface 1's prover-side derivation (C-Flock's ``hm96_salt``), here only to
  hold this commit to interface 1's fixture roots.
* **The domain** is a parameter, by default the one the verifier derives (`transfer_rows.domain`: the reading program's
  digest ``--program``, the value's leaf width, a value held across windows dropping only the window), interface 1's.
* **The certifier's key** is ``--key``'s (`certifier_serve.load_key`'s layout), one LMS leaf a value (733 at H200), in
  session ``--session``: the verifier pins a load and the windows' transfers under one key and session, so the load's
  session is the serving certifier's.

`bench` times the relay over the first ``--bytes`` of the H200 manifest's plain link, from random bytes (the commit's cost
doesn't depend on them) to no sink, at each ``--workers``, and extrapolates the whole load; with ``load_commit.p2_decode``'s
per-block decode for the ``p2`` link.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import os
import platform
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from verity.ml.commitments import hm96  # noqa: E402
from verity.ml.commitments.hm96 import batch as HB  # noqa: E402
from verity.ml.commitments.merkle import _FRAME, _LEAF, _NODE, CommitmentDomain, _uint, merkle_depth  # noqa: E402
from verity.ml.commitments.rowleaf import sha512_row_seg_digest, sha512_row_v2_prefix  # noqa: E402
from verity.ml.crypto import chacha20  # noqa: E402
from verity.core.protocols.certifier import transfer_rows as TR  # noqa: E402
from verity.core.protocols.certifier.certificate import Direction, Issuer  # noqa: E402
from verity.core.protocols.pous.schemes import p2 as P2  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import registered as G  # noqa: E402

FORMAT = "verity-network-certifier/load/v1"
LINK, KIND, WINDOW = "s0/uplink", "load", 0
SCHEMA = G.SCHEMA_V2
_INST = hm96.SHA512
_KEY = _INST.default_key
SALT_BYTES = _INST.salt_bytes
#: a chunk's bytes are at most this many
CHUNK_BYTES = 1 << 23
READ_BYTES = 1 << 23
#: CMP-95 interface 1's seeds (`verity_torch_loop.weights`): the demo's public one and its fixture's
DEMO_SEED, TEST_SEED = b"verity/cmp92/h200/weights/v1", b"verity/seeded-weights/test-seed"


# --- interface 1's manifest ----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    """Interface 1's `Shape` (``verity_torch_loop.weights``)."""

    name: str
    width: int
    layers: int
    heads: int
    head_dim: int
    kv_heads: int
    mlp: int
    vocab: int
    context: int
    row_codes: int


H200 = Shape("cmp92-h200-mqa-w16384-l52", 16384, 52, 256, 64, 1, 65536, 256, 32, 256)
TINY = Shape("tiny-h32-l1", 32, 1, 2, 16, 1, 64, 256, 8, 16)
SHAPES = {"h200": H200, "tiny": TINY}


@dataclass(frozen=True)
class Value:
    """One registered value of the load: interface 1's tensor ``name``, its bytes ``[offset, offset + nbytes)`` of the
    weights stream, in ``leaves`` rows of ``words`` u16 words, each row ``schema``'s bit row of `reg_words` leaves of
    `word_bits`; and how interface 1 generates it, ``rows`` generation rows of ``kind`` elements (``s8``, ``scale`` with
    exponent ``exponent``, ``gain``, ``embed``)."""

    name: str
    offset: int
    nbytes: int
    words: int
    kind: str = "s8"
    rows: int = 1
    exponent: int = 0
    schema: str = SCHEMA

    @property
    def row_bytes(self) -> int:
        return 2 * self.words

    @property
    def leaves(self) -> int:
        return self.nbytes // self.row_bytes

    @property
    def word_bits(self) -> int:
        """Its leaves' width read by a program named by its digest: its elements' rounded up to 16 or 32 bits
        (interface 1's `Tensor.word_bits`)."""
        return 32 if self.kind == "scale" else 16

    @property
    def reg_words(self) -> int:
        """Its leaves of `word_bits` a registered row (the same row bytes)."""
        return 16 * self.words // self.word_bits


def manifest(shape: Shape) -> tuple[Value, ...]:
    """Interface 1's tensors in stream order, each a value: ``tok`` (a token's embedding, H words, a row-seg row),
    ``pos`` (1 word), then per layer ``attn_norm``, ``wq``, ``wk``, ``wv``, ``wo``, ``mlp_norm``, ``wup``, ``wdown``,
    each linear its s8 rows of ``row_codes`` codes then its f32 ``.scale`` (one 32-bit leaf a row), a gain 1 word a
    row; then ``final_norm``, ``head`` and ``head.scale``."""
    s, out, off = shape, [], 0

    def add(name: str, nbytes: int, words: int, kind: str, rows: int = 1, exponent: int = 0,
            schema: str = SCHEMA) -> None:
        nonlocal off
        if nbytes % (2 * words):
            raise ValueError(f"{name}: {nbytes} bytes are no whole rows of {words} words")
        out.append(Value(name, off, nbytes, words, kind, rows, exponent, schema))
        off += nbytes

    def linear(name: str, n: int, k: int, shift: int = 0) -> None:
        add(name, n * k, s.row_codes // 2, "s8", n)
        add(name + ".scale", 4 * n, 2, "scale", 1, -(7 + _log2(k) // 2) + shift)

    add("tok", 2 * s.width * s.vocab, s.width, "embed", s.vocab, schema=G.SCHEMA_SEG)
    add("pos", 2 * s.width * s.context, 1, "embed", s.width)
    q, kv = s.heads * s.head_dim, s.kv_heads * s.head_dim
    for i in range(s.layers):
        p = f"l{i}."
        add(p + "attn_norm", 2 * s.width, 1, "gain")
        linear(p + "wq", q, s.width, -(_log2(s.head_dim) // 2))
        linear(p + "wk", kv, s.width)
        linear(p + "wv", kv, s.width)
        linear(p + "wo", s.width, q)
        add(p + "mlp_norm", 2 * s.width, 1, "gain")
        linear(p + "wup", s.mlp, s.width)
        linear(p + "wdown", s.width, s.mlp)
    add("final_norm", 2 * s.width, 1, "gain")
    linear("head", s.vocab, s.width)
    return tuple(out)


def _log2(n: int) -> int:
    return n.bit_length() - 1


def seeded(seed: bytes, name: str, row: int, n: int) -> bytes:
    """Interface 1's generation row `row` of tensor `name`, its first `n` bytes: the SHA-512 counter stream
    ``SHA-512(K_t ‖ u64le row ‖ u64le i)``, ``K_t = SHA-512("verity/seeded-weights/v1" 0x00 seed 0x00 name)``."""
    key = hashlib.sha512(b"verity/seeded-weights/v1\0" + seed + b"\0" + name.encode()).digest()
    g = row.to_bytes(8, "little")
    return b"".join(hashlib.sha512(key + g + i.to_bytes(8, "little")).digest() for i in range(-(-n // 64)))[:n]


def generate(seed: bytes, v: Value, first: int = 0, rows: int | None = None) -> bytes:
    """Generation rows ``first ..`` of `v` (default all) as interface 1 generates them, little-endian: an s8 code a byte,
    a scale ``(127 + E) << 23 | u32 >> 9``, a gain ``0x3F80 | u16 >> 9``, an embedding ``u16 & 0x8000 | 0x3F00 |
    u16 & 0x7F``."""
    import numpy as np
    rows = v.rows - first if rows is None else rows
    rb, out = v.nbytes // v.rows, []
    for g in range(first, first + rows):
        raw = seeded(seed, v.name, g, rb)
        if v.kind == "scale":
            u = np.frombuffer(raw, "<u4")
            raw = (np.uint32((127 + v.exponent) << 23) | (u >> np.uint32(9))).astype("<u4").tobytes()
        elif v.kind in ("gain", "embed"):
            u = np.frombuffer(raw, "<u2")
            e = np.uint16(0x3F80) | (u >> np.uint16(9)) if v.kind == "gain" else \
                (u & np.uint16(0x8000)) | np.uint16(0x3F00) | (u & np.uint16(0x7F))
            raw = e.astype("<u2").tobytes()
        out.append(raw)
    return b"".join(out)


def write_stream(seed: bytes, values: Sequence[Value], sink: BinaryIO, rows_at_once: int = 256) -> int:
    """Interface 1's weights stream of `values`, in order, to `sink`: the plain link's bytes."""
    n = 0
    for v in values:
        for g in range(0, v.rows, rows_at_once):
            data = generate(seed, v, g, min(rows_at_once, v.rows - g))
            sink.write(data)
            n += len(data)
    return n


def read_manifest(doc: dict) -> tuple[Value, ...]:
    """Interface 1's manifest JSON (``weights.py manifest``'s ``verity/seeded-weights/manifest/v1``) as values."""
    if doc.get("format") != "verity/seeded-weights/manifest/v1":
        raise ValueError(f"not interface 1's manifest: {doc.get('format')!r}")
    out = tuple(Value(t["name"], t["offset"], t["nbytes"], t["words"], t["kind"], t["shape"][0] if len(t["shape"]) == 2
                      else 1, t.get("exponent", 0), t.get("schema", SCHEMA)) for t in doc["tensors"])
    if any(v.leaves != t["leaves"] for v, t in zip(out, doc["tensors"])):
        raise ValueError("a tensor's leaves are not its bytes in rows of its words")
    if bad := [v.name for v in out if v.schema not in TR.SCHEMAS]:
        raise ValueError(f"{bad}: a load value's rows are {' or '.join(TR.SCHEMAS)} rows")
    return out


# --- interface 2's store --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class StoreTensor:
    """One tensor of interface 2's store: its blocks ``[first_block, first_block + blocks)``, and the values whose bytes
    it holds back to back."""

    name: str
    first_block: int
    blocks: int
    values: tuple[str, ...]
    size: int


def store_tensors(shape: Shape, params: P2.P2Params) -> tuple[StoreTensor, ...]:
    """Interface 2's layout (``benchmarks/pous/p2_v1/store.py``'s `build`) over interface 1's values: per layer
    ``qkv_proj`` (``wq ‖ wk ‖ wv``), ``o_proj``, ``up_proj`` and ``down_proj``, then ``lm_head`` as layer L, each from a
    block boundary and ⌈size / payload⌉ blocks."""
    by = {v.name: v for v in manifest(shape)}
    groups = []
    for i in range(shape.layers):
        at, p = f"model.layers.{i}.", f"l{i}."
        groups += [(at + "self_attn.qkv_proj.weight", (p + "wq", p + "wk", p + "wv")),
                   (at + "self_attn.o_proj.weight", (p + "wo",)), (at + "mlp.up_proj.weight", (p + "wup",)),
                   (at + "mlp.down_proj.weight", (p + "wdown",))]
    groups.append(("lm_head.weight", ("head",)))
    out, block, pb = [], 0, params.payload_bytes
    for name, vs in groups:
        size = sum(by[v].nbytes for v in vs)
        out.append(StoreTensor(name, block, -(-size // pb), vs, size))
        block += out[-1].blocks
    return tuple(out)


def store_params(shape: Shape, w: int = P2.POINT[0], c0: int = P2.POINT[1],
                 blocks_per_segment: int = P2.POINT[2]) -> P2.P2Params:
    """The store's P2 params at its registered point: whole segments over its tensors' blocks."""
    one = P2.P2Params(w, c0, blocks_per_segment)
    last = store_tensors(shape, one)[-1]
    return one.with_segments(-(-(last.first_block + last.blocks) // blocks_per_segment))


# --- the plan: which link bytes each chunk's commit takes ---------------------------------------------------------------

@dataclass(frozen=True)
class Job:
    """Value ``value``'s chunk ``chunk`` of 2^``bits`` rows from row ``j0``, ``n`` of them present (the rest pads), its
    bytes from link bytes ``[lo, hi)``: the rows themselves, or for a P2 chunk the blocks from ``block`` they fall in,
    the rows from byte ``skip`` of the decoded payload, and ``tail`` decoded bytes past them that must be zero."""

    value: int
    chunk: int
    bits: int
    j0: int
    n: int
    lo: int
    hi: int
    block: int | None = None
    skip: int = 0
    tail: int = 0


def chunk_bits(v: Value, chunk_bytes: int = CHUNK_BYTES) -> int:
    return max(0, min(merkle_depth(v.leaves), (max(1, chunk_bytes // v.row_bytes)).bit_length() - 1))


def _chunks(v: Value, chunk_bytes: int) -> Iterable[tuple[int, int, int, int]]:
    """``(chunk, bits, j0, n)`` of each of `v`'s chunks."""
    bits = chunk_bits(v, chunk_bytes)
    for c in range((1 << merkle_depth(v.leaves)) >> bits):
        j0 = c << bits
        yield c, bits, j0, max(0, min(1 << bits, v.leaves - j0))


def plan(values: Sequence[Value], fmt: str = "plain", shape: Shape | None = None, params: P2.P2Params | None = None,
         chunk_bytes: int = CHUNK_BYTES) -> tuple[list[Job], int]:
    """Every chunk's job in link order, and the link's length."""
    jobs: list[Job] = []
    if fmt == "plain":
        at = 0
        for i, v in enumerate(values):
            if v.offset != at:
                raise ValueError(f"{v.name} is at stream byte {v.offset}, not {at}: the stream is its tensors back to back")
            for c, bits, j0, n in _chunks(v, chunk_bytes):
                lo = v.offset + min(j0, v.leaves) * v.row_bytes
                jobs.append(Job(i, c, bits, j0, n, lo, lo + n * v.row_bytes))
            at += v.nbytes
        return jobs, at
    if fmt != "p2" or shape is None or params is None:
        raise ValueError("the link is plain, or p2 with its shape and store params")
    index = {v.name: i for i, v in enumerate(values)}
    nb, pb = params.block_bytes, params.payload_bytes
    stored, ours = set(), {v.name: v.nbytes for v in manifest(shape)}
    if any(ours.get(v.name) != v.nbytes for v in values):
        raise ValueError(f"the values are not {shape.name}'s manifest, which the store's layout is of")
    for t in store_tensors(shape, params):
        w0 = t.first_block * pb
        for name in t.values:
            i, v = index[name], values[index[name]]
            stored.add(name)
            for c, bits, j0, n in _chunks(v, chunk_bytes):
                lo, hi = w0 + j0 * v.row_bytes, w0 + (j0 + n) * v.row_bytes
                if not n:
                    jobs.append(Job(i, c, bits, j0, n, 0, 0))
                    continue
                b0, b1 = lo // pb, -(-hi // pb)
                end = t.first_block * pb + t.size
                tail = b1 * pb - hi if hi == end else 0
                jobs.append(Job(i, c, bits, j0, n, b0 * nb, b1 * nb, b0, lo - b0 * pb, tail))
            w0 += v.nbytes
    at = params.blocks * nb
    for i, v in enumerate(values):
        if v.name in stored:
            continue
        for c, bits, j0, n in _chunks(v, chunk_bytes):
            lo = at + min(j0, v.leaves) * v.row_bytes
            jobs.append(Job(i, c, bits, j0, n, lo, lo + n * v.row_bytes))
        at += v.nbytes
    # a chunk's blocks start no earlier than an earlier chunk's, so in order of their last byte each job's first byte is
    # the least any job left needs (a pad-only chunk needs none, and goes first)
    jobs.sort(key=lambda j: j.hi)
    return jobs, at


# --- one chunk's commit ---------------------------------------------------------------------------------------------------

def os_salts(value: int, first: int, n: int) -> bytes:
    """The certifier's salts: `n` rows' 192 bytes each, fresh from the operating system."""
    return os.urandom(SALT_BYTES * n)


@dataclass(frozen=True)
class keyed_salts:  # noqa: N801
    """Interface 1's prover-side salts (C-Flock's ``hm96_salt``): row r of the value at manifest index i, ChaCha20 blocks
    3r .. 3r + 2 under `key` with nonce i. Only to hold this commit to interface 1's fixture roots."""

    key: bytes

    def __call__(self, value: int, first: int, n: int) -> bytes:
        k = SALT_BYTES // chacha20.BLOCK_BYTES
        return chacha20.keystream(self.key, value, k * first, k * n)


class Frame:
    """A frame-v3-sha512 domain's leaf and node hashes with their framing before the varying fields precomputed: each
    `CommitmentDomain.leaf`'s and `node`'s."""

    def __init__(self, domain_id: bytes, schema: str = SCHEMA) -> None:
        d = len(domain_id).to_bytes(8, "big") + domain_id
        s = schema.encode()
        self.leaf_head = _FRAME + len(_LEAF).to_bytes(4, "big") + _LEAF + d
        self.leaf_tail = len(s).to_bytes(8, "big") + s + (64).to_bytes(8, "big")
        self.node_head = _FRAME + len(_NODE).to_bytes(4, "big") + _NODE + d
        self.domain_id = domain_id

    def leaf(self, rank: int, value: bytes) -> bytes:
        r = _uint(rank)
        rr = len(r).to_bytes(8, "big") + r
        return hashlib.sha512(self.leaf_head + rr + rr + self.leaf_tail + value).digest()

    def node(self, level: int, index: int, left: bytes, right: bytes) -> bytes:
        lv, ix = _uint(level), _uint(index)
        return hashlib.sha512(self.node_head + len(lv).to_bytes(8, "big") + lv + len(ix).to_bytes(8, "big") + ix
                              + b"\0" * 7 + b"\x40" + left + b"\0" * 7 + b"\x40" + right).digest()


def subtree(dom: CommitmentDomain, row_bytes: int, data: bytes, salts: bytes, j0: int, bits: int,
            schema: str = SCHEMA) -> bytes:
    """The subtree root over ranks ``[j0, j0 + 2^bits)`` of a registered value's tree: `data`'s whole rows from rank
    `j0`, each `schema`'s bit row of its bytes (whatever its leaves' width) under its salt of `salts`, then pads
    (`tree_levels`' levels of `Registered.commit`)."""
    n = len(data) // row_bytes
    if len(data) != n * row_bytes or len(salts) != n * SALT_BYTES or n > 1 << bits:
        raise ValueError(f"{len(data)} bytes and {len(salts)} of salts are not as many rows of {row_bytes} bytes, "
                         f"at most 2^{bits}")
    f, xs = Frame(dom.domain_id, schema), bytearray()
    if schema == G.SCHEMA_SEG:
        for k in range(n):
            xs += sha512_row_seg_digest(data[row_bytes * k:row_bytes * (k + 1)], 8 * row_bytes)
    else:
        xh = hashlib.sha512(sha512_row_v2_prefix(8 * row_bytes))
        for k in range(n):
            h = xh.copy()
            h.update(data[row_bytes * k:row_bytes * (k + 1)])
            xs += h.digest()
    level = [f.leaf(j0 + k, t) for k, t in enumerate(HB.tree_leaves(_INST, _KEY, bytes(xs), salts))]
    level += [dom.pad(j) for j in range(j0 + n, j0 + (1 << bits))]
    for d in range(bits):
        base = j0 >> (d + 1)
        level = [f.node(d, base + k // 2, level[k], level[k + 1]) for k in range(0, len(level), 2)]
    return level[0]


def decode_blocks(params: P2.P2Params, salt: bytes, first: int, cstar: bytes) -> bytes:
    """Blocks ``first ..`` of C* decoded (`p2.decode_segment`'s arithmetic, at any block): their payloads back to back.
    Raises ValueError, as the store's decode refuses, on a block that isn't below p or decodes past the payload."""
    f, nb, pb, bps = P2._field(params), params.block_bytes, params.payload_bytes, params.blocks_per_segment
    out, keys, seg = bytearray(), None, -1
    for k in range(len(cstar) // nb):
        i = first + k
        s, v = divmod(i, bps)
        if s != seg:
            keys, seg = P2.segment_keys(params, salt, s), s
        c = int.from_bytes(cstar[nb * k:nb * (k + 1)], "big")
        if c >= f.p:
            raise ValueError(f"block {i} is not below p: not a codeword")
        kk, t = keys(v)
        m = (f.sq(f.sigma(f.sq(c), kk)) - t) % f.p
        if m >> (8 * pb):
            raise ValueError(f"block {i} decodes to more than {8 * pb} bits: not a codeword")
        out += m.to_bytes(pb, "big")
    return bytes(out)


@dataclass(frozen=True)
class Task:
    """What a worker needs for one job beside its bytes."""

    job: Job
    row_bytes: int
    schema: str
    domain: CommitmentDomain
    witness: str | None
    witness_at: int
    salts: Callable[[int, int, int], bytes]
    params: P2.P2Params | None = None
    salt: bytes | None = None


def run_task(task: Task, data: bytes) -> tuple[int, int, bytes]:
    """One chunk's subtree root: its bytes decoded first when they are P2 blocks, its salts drawn and written at their
    witness offset."""
    j = task.job
    if j.block is not None:
        w = decode_blocks(task.params, task.salt, j.block, data)
        if j.tail and any(w[len(w) - j.tail:]):
            raise ValueError(f"block {j.block + len(w) // task.params.payload_bytes - 1} is nonzero past its tensor's end")
        data = w[j.skip:j.skip + j.n * task.row_bytes]
    salts = task.salts(j.value, j.j0, j.n)
    if task.witness is not None and j.n:
        fd = os.open(task.witness, os.O_WRONLY)
        try:
            os.pwrite(fd, salts, task.witness_at + SALT_BYTES * j.j0)
        finally:
            os.close(fd)
    return j.value, j.chunk, subtree(task.domain, task.row_bytes, data, salts, j.j0, j.bits, task.schema)


def _work(args: tuple[Task, bytes]) -> tuple[int, int, bytes]:
    return run_task(*args)


# --- the relay ----------------------------------------------------------------------------------------------------------

class Refused(Exception):
    """The certifier refuses the load: it signs nothing."""


@dataclass
class Relayed:
    roots: dict[str, bytes]
    link_bytes: int
    seconds: float
    jobs: int


def relay(values: Sequence[Value], source: BinaryIO, sink: BinaryIO | None, domain: Callable[[Value], CommitmentDomain], *,
          fmt: str = "plain", shape: Shape | None = None, params: P2.P2Params | None = None, salt: bytes | None = None,
          salts: Callable[[int, int, int], bytes] = os_salts, witness: str | Path | None = None, workers: int = 1,
          chunk_bytes: int = CHUNK_BYTES, read_bytes: int = READ_BYTES, progress: Callable[[int, int], None] | None = None,
          ) -> Relayed:
    """Read the link from `source` to its end, writing each read on to `sink` before its rows are committed, and commit
    every value over `workers` processes: each value's root, equal to `Registered.commit`'s of its rows under the same
    salts and `domain(value)`. Refused when the link isn't the load's length or a P2 block doesn't decode."""
    jobs, length = plan(values, fmt, shape, params, chunk_bytes)
    doms = [domain(v) for v in values]
    offsets, at = [], 0
    for v in values:
        offsets.append(at)
        at += SALT_BYTES * v.leaves
    wpath = None if witness is None else str(witness)
    if wpath is not None:
        with open(wpath, "wb") as fh:
            fh.truncate(at)
    got: dict[int, dict[int, bytes]] = {i: {} for i in range(len(values))}
    errors: list[BaseException] = []
    slots = threading.BoundedSemaphore(max(2, 2 * workers))
    pool = mp.get_context("fork").Pool(workers) if workers > 1 else None
    t0 = time.perf_counter()

    def done(r: tuple[int, int, bytes]) -> None:
        got[r[0]][r[1]] = r[2]
        slots.release()

    def failed(e: BaseException) -> None:
        errors.append(e)
        slots.release()

    def submit(j: Job, data: bytes) -> None:
        v = values[j.value]
        task = Task(j, v.row_bytes, v.schema, doms[j.value], wpath, offsets[j.value], salts, params, salt)
        if pool is None:
            try:
                r = run_task(task, data)
                got[r[0]][r[1]] = r[2]
            except ValueError as e:
                errors.append(e)
            return
        slots.acquire()
        pool.apply_async(_work, ((task, data),), callback=done, error_callback=failed)

    buf, buf_lo, pos, k = bytearray(), 0, 0, 0
    try:
        while True:
            while k < len(jobs) and jobs[k].hi <= pos:
                j = jobs[k]
                submit(j, bytes(buf[j.lo - buf_lo:j.hi - buf_lo]))
                k += 1
            if errors:
                break
            keep = min(jobs[k].lo, pos) if k < len(jobs) else pos
            if keep - buf_lo >= read_bytes:
                del buf[:keep - buf_lo]
                buf_lo = keep
            data = source.read(read_bytes)
            if not data:
                break
            if sink is not None:
                sink.write(data)
            buf += data
            pos += len(data)
            if progress is not None:
                progress(pos, length)
        if pool is not None:
            pool.close()
            pool.join()
    except BaseException:
        if pool is not None:
            pool.terminate()
        raise
    if errors:
        if not all(isinstance(e, ValueError) for e in errors):
            raise next(e for e in errors if not isinstance(e, ValueError))
        raise Refused(f"the link doesn't decode: {errors[0]}") from errors[0]
    if pos != length:
        raise Refused(f"the link carried {pos} bytes, and the load is {length}")
    roots = {}
    for i, v in enumerate(values):
        level, d = [got[i][c] for c in range(len(got[i]))], chunk_bits(v, chunk_bytes)
        f = Frame(doms[i].domain_id, v.schema)
        while len(level) > 1:
            level = [f.node(d, q // 2, level[q], level[q + 1]) for q in range(0, len(level), 2)]
            d += 1
        roots[v.name] = level[0]
    return Relayed(roots, pos, time.perf_counter() - t0, len(jobs))


def verifier_domain(program: str) -> Callable[[Value], CommitmentDomain]:
    """The domain the verifier derives for a load value read by the program of digest `program`."""
    return lambda v: TR.domain(program, v.name, v.reg_words, v.leaves, v.word_bits)


def sign(issuer: Issuer, link: str, program: str, values: Sequence[Value], roots: dict[str, bytes],
         domain: Callable[[Value], CommitmentDomain]) -> list[dict[str, Any]]:
    """Each value's head signed for the load's link-window: its registration (`transfer_rows.registration`)."""
    out = []
    for v in values:
        head = TR.TransferHead(link, Direction.INGRESS, WINDOW, KIND, v.name, bytes.fromhex(program), v.reg_words,
                               v.leaves, domain(v).domain_id, roots[v.name], v.word_bits, v.schema)
        cert = issuer.sign(link, Direction.INGRESS, WINDOW, head.digest())
        if cert is None:
            raise Refused(f"{v.name}: the network certifier's signer refused: {issuer.refused}")
        out.append(TR.registration(head, cert))
    return out


# --- the command --------------------------------------------------------------------------------------------------------

class RandomSource(io.RawIOBase):
    """`n` bytes of the operating system's generator, read as a stream."""

    def __init__(self, n: int) -> None:
        self.left = n

    def read(self, size: int = -1) -> bytes:
        size = self.left if size < 0 else min(size, self.left)
        self.left -= size
        return os.urandom(size)


def first_values(values: Sequence[Value], nbytes: int) -> tuple[Value, ...]:
    """The values that end within the first `nbytes` of the weights stream."""
    return tuple(v for v in values if v.offset + v.nbytes <= nbytes)


def bench(shape: Shape, nbytes: int, workers: Sequence[int], witness_dir: Path | None) -> dict[str, Any]:
    import load_commit as LC
    values = first_values(manifest(shape), nbytes)
    rows = sum(v.leaves for v in values)
    total_rows, total_bytes = sum(v.leaves for v in manifest(shape)), sum(v.nbytes for v in manifest(shape))
    dom = verifier_domain("00" * 64)
    out: dict[str, Any] = {"values": len(values), "bytes": sum(v.nbytes for v in values), "rows": rows, "runs": []}
    for n in workers:
        wit = None if witness_dir is None else witness_dir / f"witness-{n}.bin"
        r = relay(values, RandomSource(out["bytes"]), None, dom, workers=n, witness=wit)
        if wit is not None:
            wit.unlink()
        us = r.seconds / rows * 1e6
        out["runs"].append({"workers": n, "seconds": r.seconds, "row_us_wall": us, "mb_s": out["bytes"] / r.seconds / 1e6,
                            "load_s": total_rows * us / 1e6, "witness": wit is not None})
    params = store_params(shape)
    t0, blocks = time.perf_counter(), 64
    LC.p2_decode(os.urandom(blocks * params.block_bytes), os.urandom(24))
    dec = (time.perf_counter() - t0) / blocks
    out["p2"] = {"decode_block_us": dec * 1e6, "blocks": params.blocks, "decode_core_s": params.blocks * dec,
                 "by_workers": {n: params.blocks * dec / n for n in workers}}
    out["shape"] = {"rows": total_rows, "bytes": total_bytes, "witness_bytes": SALT_BYTES * total_rows}
    return out


def _open(path: str, mode: str) -> BinaryIO | None:
    if path == "none":
        return None
    if path == "-":
        return (sys.stdin if "r" in mode else sys.stdout).buffer
    return open(path, mode)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="load_relay.py", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("relay")
    r.add_argument("--program", required=True, help="the reading program's digest, lowercase hex of 64 bytes")
    r.add_argument("--out", required=True)
    r.add_argument("--shape", choices=sorted(SHAPES), default="h200")
    r.add_argument("--manifest", help="interface 1's manifest JSON, in place of --shape's")
    r.add_argument("--bytes", type=int, help="only the values that end within the weights stream's first BYTES (plain)")
    r.add_argument("--format", choices=("plain", "p2"), default="plain")
    r.add_argument("--source", default="-")
    r.add_argument("--sink", default="none")
    r.add_argument("--salt", help="the store's P2 salt, hex (--format p2)")
    r.add_argument("--witness", help="where the salts go (default OUT/load-witness.bin)")
    r.add_argument("--key", help="the network certifier's key directory (default OUT/certifier)")
    r.add_argument("--height", type=int, default=12, help="a new key's LMS height")
    r.add_argument("--session", help="the certifier's session, hex (default fresh)")
    r.add_argument("--link", default=LINK)
    r.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    b = sub.add_parser("bench")
    b.add_argument("--out", required=True)
    b.add_argument("--shape", choices=sorted(SHAPES), default="h200")
    b.add_argument("--bytes", type=int, default=1 << 31)
    b.add_argument("--workers", default="1")
    b.add_argument("--witness-dir", help="write each run's salts there (default: none)")
    s = sub.add_parser("stream", help="interface 1's weights stream to stdout: a plain link's source")
    s.add_argument("--shape", choices=sorted(SHAPES), default="h200")
    s.add_argument("--seed", default=DEMO_SEED.decode(), help="interface 1's seed, as text")
    s.add_argument("--bytes", type=int, help="only the values that end within the stream's first BYTES")
    a = ap.parse_args(argv)
    if a.cmd == "stream":
        values = manifest(SHAPES[a.shape])
        write_stream(a.seed.encode(), first_values(values, a.bytes) if a.bytes else values, sys.stdout.buffer)
        sys.stdout.buffer.flush()
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    host = {"node": platform.node(), "cpus": os.cpu_count(), "affinity": len(os.sched_getaffinity(0)),
            "python": sys.version.split()[0]}
    if a.cmd == "bench":
        res = bench(SHAPES[a.shape], a.bytes, [int(x) for x in a.workers.split(",")],
                    None if a.witness_dir is None else Path(a.witness_dir)) | {"host": host}
        (out / "load-relay-bench.json").write_text(json.dumps(res, indent=1, sort_keys=True))
        for run in res["runs"]:
            print(f"LOADRELAY workers {run['workers']}: {res['bytes'] / 2**30:.2f} GiB, {res['rows']:,} rows in "
                  f"{run['seconds']:.1f} s ({run['mb_s']:.0f} MB/s), whole load {run['load_s']:,.0f} s", flush=True)
        p = res["p2"]
        print(f"LOADRELAY p2 decode {p['decode_block_us']:.0f} µs/block, {p['decode_core_s']:,.0f} core-s; "
              + ", ".join(f"{n} workers {s:,.0f} s" for n, s in p["by_workers"].items()), flush=True)
        return 0
    shape = SHAPES[a.shape]
    values = read_manifest(json.loads(Path(a.manifest).read_text())) if a.manifest else manifest(shape)
    params = store_params(shape) if a.format == "p2" else None
    if a.format == "p2" and (a.salt is None or a.bytes):
        ap.error("--format p2 needs the store's --salt, and takes the whole load")
    if a.bytes:
        values = first_values(values, a.bytes)
    import certifier_serve as CS
    key_dir = Path(a.key) if a.key else out / "certifier"
    signer = CS.signer(key_dir, a.height)
    session = bytes.fromhex(a.session) if a.session else os.urandom(32)
    dom = verifier_domain(a.program)
    witness = a.witness or str(out / "load-witness.bin")
    src, snk = _open(a.source, "rb"), _open(a.sink, "wb")

    said = [0]

    def say(done: int, total: int) -> None:
        if done == total or done - said[0] >= 1 << 33:
            said[0] = done
            print(f"LOADRELAY {done:,}/{total:,} bytes", file=sys.stderr, flush=True)

    try:
        got = relay(values, src, snk, dom, fmt=a.format, shape=shape, params=params,
                    salt=None if a.salt is None else bytes.fromhex(a.salt), witness=witness, workers=a.workers,
                    progress=say)
    except Refused as e:
        print(f"LOADRELAY refused: {e}", flush=True)
        return 1
    finally:
        if snk is not None:
            snk.flush()
    regs = sign(Issuer(signer, session), a.link, a.program, values, got.roots, dom)
    doc = {"format": FORMAT, "link": a.link, "window": WINDOW, "program": a.program, "link_format": a.format,
           "key": CS.public_json(signer.public_key()), "session": session.hex(), "values": regs}
    (out / "load.json").write_text(json.dumps(doc, indent=1))
    rep = {"host": host, "link_bytes": got.link_bytes, "seconds": got.seconds, "jobs": got.jobs, "workers": a.workers,
           "values": len(values), "rows": sum(v.leaves for v in values), "witness": witness}
    (out / "load-relay.json").write_text(json.dumps(rep, indent=1, sort_keys=True))
    print(f"LOADRELAY {len(values)} values, {got.link_bytes:,} bytes in {got.seconds:.1f} s "
          f"({got.link_bytes / got.seconds / 1e6:.0f} MB/s) over {a.workers} workers; signed {len(regs)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
