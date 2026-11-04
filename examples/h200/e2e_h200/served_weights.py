"""The demo's seeded weights and their registration root: one deterministic generator from a seed to
every tensor of a decoder of any `Shape` (the H200 demo's frozen shape `H200`, Phase 1's CPU model `PHASE1`), and the
registration of those tensors as core's registered values (``one_stage.registered``), computed in parallel.

No torch: memory-accounting's encoder and the H200 demo run it with `examples/h200` on their path
(``PYTHONPATH=examples/h200 python -m e2e_h200.served_weights ...``).

* **The model** (`Shape`): a pre-norm decoder with learned absolute positions, ``heads`` query heads of ``head_dim`` and
  ``kv_heads`` KV heads (1: MQA), a ReLU MLP of width ``mlp``, a byte vocabulary, W8A8 linears (s8 codes, an f32 scale per
  output channel) and bf16 embeddings and norm gains.
* **The tensors** (`tensors`), in this order: ``tok`` (H, V), ``pos`` (H, C), then per layer ``l<i>.`` ``attn_norm`` (H),
  ``wq`` (heads·d, H), ``wk``, ``wv`` (kv_heads·d, H), ``wo`` (H, heads·d), ``mlp_norm`` (H), ``wup`` (F, H), ``wdown``
  (H, F), each linear followed by its ``.scale`` (N), then ``final_norm`` (H), ``head`` (V, H) and ``head.scale``. A 2-D
  tensor is row-major: a linear's row r is output channel r's elements in ascending input index, ``tok`` and ``pos``
  are stored transposed (row j is coordinate j over the tokens, or over the positions: the row the embed-add unit of
  coordinate j reads whole, selecting the token inside it; `Shape.tok_rows` ``"token"`` stores ``tok`` (V, H) instead,
  row r token r's embedding). A tensor's bytes are its elements little-endian (s8 codes, f32 scales, bf16 words).
  **The weights stream** is every tensor's bytes in this order, no padding (`Tensor.offset`): 130.10 GiB at `H200`, of
  which the 52 layers' s8 matrices are 139,695,489,024 bytes.
* **The generator** (`generate`, `row_bytes`): tensor ``t``'s generation row ``g`` (a 2-D tensor's row; a 1-D tensor is one
  row) is the SHA-512 counter stream ``SHA-512(K_t || u64le(g) || u64le(i))``, i = 0, 1, ..., truncated to the row's bytes,
  ``K_t = SHA-512("verity/seeded-weights/v1" 0x00 seed 0x00 name)``: one 80-byte message (one compression) per 64 bytes, a
  function of (seed, tensor, row) alone, so any range generates on its own, on any core or GPU. Elements from the stream:
  an s8 code is a byte; an f32 scale is ``(127 + E) << 23 | u32 >> 9`` (u32 little-endian; uniform in [2^E, 2^(E+1)),
  ``E = -(7 + floor(log2 K / 2))``, so a linear keeps its input's scale, and ``wq`` takes the score's 1/sqrt(d) as
  ``-floor(log2 d / 2)`` more); a norm gain ``0x3F80 | u16 >> 9`` (bf16 in [1, 2)); an embedding ``u16 & 0x8000 | 0x3F00 |
  u16 & 0x7F`` (bf16 of magnitude in [0.5, 1)).
* **The registration** (`root`, `entries`), one per reading program: every tensor is the registered value of the program's
  root parameter of its name (core's ``Registered.commit``), its bytes cut into rows of `Tensor.words` u16 words
  (little-endian, the reading port's row: an s8 linear ``row_codes`` codes, at `H200` the 8,192 codes of the part
  ``ScaledMmS8CoordinateRows_v2``'s ``wrow{r}`` reads on circuits' FAMILIES map (``quant_chunk``); a scale 2 words; ``tok`` one coordinate's V words; a gain and ``pos`` 1 word), in
  ``hm96-sha512/row/v2`` (``tok`` in `Shape.tok_schema`, row/v2 by default; ``EmbeddingRead_v1``'s
  ``hm96-sha512/row-seg/v1`` with ``tok_rows="token"``), under the
  domain the verifier derives (``registered.registered_entries``, the value held across windows):
  ``registered.domain(program_digest(program), name, words, leaves, word_bits)``, the word width the parameter's leaf
  width rounded up to 16 or 32 bits (an f32 scale read as one 32-bit leaf is 1 word of 32 bits, the same row bytes; a
  program named by its digest alone reads each tensor at `Tensor.word_bits`, 16: a Boolean program's one-bit leaves, as
  the served step's Program reads them). Row
  i's salt is ChaCha20 blocks 3i .. 3i+2 under the prover's 32-byte salt key at the tensor's index as nonce (Flock's
  ``hm96_salt``; the key salts these rows only). At `H200` every tensor's row count is a power of two, so no tree has a
  pad. **The registration digest** (`registration_digest`) is SHA-512 of ``verity/seeded-weights/registration/v1`` and
  ``{program, values}`` (the program's digest, the entries of the tensors it reads in tensor order): the one root the
  weights' store ties to.
  The generator never reads the program, so a new program re-roots the weights and leaves their bytes as they were.
* **In parallel** (`root`): a value's rows are cut into aligned chunks of a power of two rows; each worker generates its
  chunk's bytes itself and returns the chunk's subtree root, and the caller folds them as ``tree_levels`` does. Only the
  seed, the salt key, the `Reader` (the program's digest and widths) and chunk indices cross processes.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import multiprocessing
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from verity.core.protocols.sampled_proofs.one_stage import registered as REG
from verity.ml.commitments import hm96
from verity.ml.commitments.hm96 import batch as hm96_batch
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.merkle import merkle_depth
from verity.ml.circuits.codec import program_digest
from verity.ml.circuits.program import Program
from verity.ml.commitments.rowleaf import sha512_row_seg_digest, sha512_row_v2_prefix
from verity.ml.crypto import chacha20

FORMAT = "verity/seeded-weights/v1"
MANIFEST = "verity/seeded-weights/manifest/v1"
REGISTRATION = "verity/seeded-weights/registration/v1"
SCHEMA = REG.SCHEMA_V2
_INST = hm96.SHA512
_SALT_BLOCKS = _INST.salt_bytes // chacha20.BLOCK_BYTES
#: a chunk's bytes are at most this many (`root`)
CHUNK_BYTES = 1 << 23
_DTYPES = {"s8": (1, np.int8), "f32": (4, np.float32), "bf16": (2, np.uint16)}


@dataclass(frozen=True)
class Shape:
    name: str
    width: int
    layers: int
    heads: int
    head_dim: int
    kv_heads: int
    mlp: int
    vocab: int
    context: int
    #: s8 codes in one registered row of a linear: the K of the part a unit reads
    row_codes: int
    #: ``tok``'s row format: ``hm96-sha512/row/v2``, the rows the demo's proofs cover;
    #: ``hm96-sha512/row-seg/v1`` is ``EmbeddingRead_v1``'s
    tok_schema: str = REG.SCHEMA_V2
    #: ``tok``'s rows: ``"coordinate"`` (H, V), a coordinate over the tokens (the default),
    #: or ``"token"`` (V, H), a token's embedding
    tok_rows: str = "coordinate"


#: the H200 demo's frozen shape; a linear's row the 8,192 codes a FAMILIES ``wrow``
#: part reads
H200 = Shape("cmp92-h200-mqa-w16384-l52", 16384, 52, 256, 64, 1, 65536, 256, 32, 8192)
#: Phase 1's CPU model (`transformer`): its linears read whole 128-code rows, its down projection in K = 128 parts
PHASE1 = Shape("cmp92-phase1-h128-l2", 128, 2, 2, 64, 2, 512, 256, 32, 128)
#: the smallest shape the tests commit whole
TINY = Shape("tiny-h32-l1", 32, 1, 2, 16, 1, 64, 256, 8, 16)
SHAPES = {s.name: s for s in (H200, PHASE1, TINY)}
ALIASES = {"h200": H200, "phase1": PHASE1, "tiny": TINY}


@dataclass(frozen=True)
class Tensor:
    name: str
    #: ``s8`` codes, f32 ``scale``, bf16 ``gain`` or ``embed``
    kind: str
    #: generation rows (output channels; 1 for a 1-D tensor) and elements a row
    rows: int
    cols: int
    #: u16 words in one registered row
    words: int
    #: the tensor's byte offset in the weights stream
    offset: int
    #: a scale's exponent E
    exponent: int = 0
    #: its registered rows' format
    schema: str = SCHEMA

    @property
    def dtype(self) -> str:
        return {"s8": "s8", "scale": "f32"}.get(self.kind, "bf16")

    @property
    def word_bits(self) -> int:
        """Its leaves' width when its reader is named by digest alone: 16, a Boolean program's one-bit leaves rounded up
        (the served step's Program reads every tensor so, an f32 scale as two 16-bit words)."""
        return 16

    @property
    def row_bytes(self) -> int:
        return self.cols * _DTYPES[self.dtype][0]

    @property
    def nbytes(self) -> int:
        return self.rows * self.row_bytes

    @property
    def leaves(self) -> int:
        """Its registered rows."""
        return self.nbytes // (2 * self.words)

    @property
    def shape(self) -> list[int]:
        return [self.cols] if self.rows == 1 else [self.rows, self.cols]


def _log2(n: int) -> int:
    return n.bit_length() - 1


def scale_exponent(k: int, shift: int = 0) -> int:
    return -(7 + _log2(k) // 2) + shift


@functools.lru_cache(maxsize=8)
def tensors(shape: Shape) -> tuple[Tensor, ...]:
    """The shape's tensors in stream order (module docstring)."""
    s, out, off = shape, [], 0
    if s.row_codes % 2 or s.width % s.row_codes or s.mlp % s.row_codes or (s.heads * s.head_dim) % s.row_codes:
        raise ValueError(f"{s.name}: every linear's K must be a multiple of row_codes = {s.row_codes}, an even number")

    def add(name: str, kind: str, rows: int, cols: int, words: int, exponent: int = 0, schema: str = SCHEMA) -> None:
        nonlocal off
        t = Tensor(name, kind, rows, cols, words, off, exponent, schema)
        if t.nbytes % (2 * words):
            raise ValueError(f"{name}: {t.nbytes} bytes are no whole rows of {words} words")
        out.append(t)
        off += t.nbytes

    def linear(name: str, n: int, k: int, shift: int = 0) -> None:
        add(name, "s8", n, k, s.row_codes // 2)
        add(f"{name}.scale", "scale", 1, n, 2, scale_exponent(k, shift))

    if s.tok_schema not in (REG.SCHEMA_V2, REG.SCHEMA_SEG):
        raise ValueError(f"{s.name}: tok's rows are {REG.SCHEMA_V2} or {REG.SCHEMA_SEG}, not {s.tok_schema!r}")
    if s.tok_rows == "coordinate":
        add("tok", "embed", s.width, s.vocab, s.vocab, schema=s.tok_schema)
    elif s.tok_rows == "token":
        add("tok", "embed", s.vocab, s.width, s.width, schema=s.tok_schema)
    else:
        raise ValueError(f"{s.name}: tok's rows are 'coordinate' or 'token', not {s.tok_rows!r}")
    add("pos", "embed", s.width, s.context, 1)
    q, kv = s.heads * s.head_dim, s.kv_heads * s.head_dim
    for i in range(s.layers):
        p = f"l{i}."
        add(p + "attn_norm", "gain", 1, s.width, 1)
        linear(p + "wq", q, s.width, -(_log2(s.head_dim) // 2))
        linear(p + "wk", kv, s.width)
        linear(p + "wv", kv, s.width)
        linear(p + "wo", s.width, q)
        add(p + "mlp_norm", "gain", 1, s.width, 1)
        linear(p + "wup", s.mlp, s.width)
        linear(p + "wdown", s.width, s.mlp)
    add("final_norm", "gain", 1, s.width, 1)
    linear("head", s.vocab, s.width)
    return tuple(out)


def tensor(shape: Shape, name: str) -> Tensor:
    for t in tensors(shape):
        if t.name == name:
            return t
    raise ValueError(f"{shape.name} has no tensor {name!r}")


def stream_bytes(shape: Shape) -> int:
    t = tensors(shape)[-1]
    return t.offset + t.nbytes


def _seed(seed: bytes) -> bytes:
    if not isinstance(seed, bytes) or not seed:
        raise ValueError("a seed is nonempty bytes")
    return seed


@functools.lru_cache(maxsize=1024)
def tensor_key(seed: bytes, name: str) -> bytes:
    return hashlib.sha512(FORMAT.encode() + b"\0" + _seed(seed) + b"\0" + name.encode()).digest()


def row_bytes(seed: bytes, name: str, row: int, n: int) -> bytes:
    """The first `n` bytes of generation row `row`'s SHA-512 counter stream of tensor `name` (module docstring)."""
    key, g = tensor_key(seed, name), row.to_bytes(8, "little")
    sha = hashlib.sha512
    return b"".join(sha(key + g + i.to_bytes(8, "little")).digest() for i in range(-(-n // 64)))[:n]


def _elements(t: Tensor, raw: bytes) -> np.ndarray:
    if t.kind == "s8":
        return np.frombuffer(raw, np.int8)
    if t.kind == "scale":
        u = np.frombuffer(raw, "<u4").astype(np.uint32)
        return ((np.uint32((127 + t.exponent) << 23)) | (u >> np.uint32(9))).view(np.float32)
    u = np.frombuffer(raw, "<u2").astype(np.uint16)
    if t.kind == "gain":
        return np.uint16(0x3F80) | (u >> np.uint16(9))
    return (u & np.uint16(0x8000)) | np.uint16(0x3F00) | (u & np.uint16(0x7F))


def generate(shape: Shape, seed: bytes, name: str, rows: Iterable[int] | None = None) -> np.ndarray:
    """Tensor `name`'s generation rows `rows` (default all), ``(len(rows), cols)``: s8 codes as int8, scales as float32,
    bf16 as their uint16 words."""
    t = tensor(shape, name)
    rows = range(t.rows) if rows is None else list(rows)
    if any(not 0 <= r < t.rows for r in rows):
        raise ValueError(f"{name} has rows [0, {t.rows})")
    n = t.cols * _DTYPES[t.dtype][0]
    out = np.empty((len(rows), t.cols), _DTYPES[t.dtype][1])
    for j, r in enumerate(rows):
        out[j] = _elements(t, row_bytes(seed, name, r, n))
    return out


def tensor_bytes(shape: Shape, seed: bytes, name: str, start: int = 0, stop: int | None = None) -> bytes:
    """Bytes ``[start, stop)`` of tensor `name` (little-endian elements), generating only the rows they touch."""
    t = tensor(shape, name)
    stop = t.nbytes if stop is None else stop
    if not 0 <= start <= stop <= t.nbytes:
        raise ValueError(f"{name} has bytes [0, {t.nbytes})")
    if start == stop:
        return b""
    rb = t.row_bytes
    g0, g1 = start // rb, -(-stop // rb)
    data = generate(shape, seed, name, range(g0, g1)).astype(_LE[t.dtype], copy=False).tobytes()
    return data[start - g0 * rb:stop - g0 * rb]


_LE = {"s8": "i1", "f32": "<f4", "bf16": "<u2"}


def stream(shape: Shape, seed: bytes, start: int, stop: int) -> bytes:
    """Bytes ``[start, stop)`` of the weights stream."""
    out = []
    for t in tensors(shape):
        lo, hi = max(start, t.offset), min(stop, t.offset + t.nbytes)
        if lo < hi:
            out.append(tensor_bytes(shape, seed, t.name, lo - t.offset, hi - t.offset))
    return b"".join(out)


def model_id(shape: Shape, seed: bytes) -> str:
    """The manifest's name of the weights: the format, the shape and the seed's id (not the seed). No domain binds it."""
    sid = hashlib.sha512(FORMAT.encode() + b"/seed-id\0" + _seed(seed)).hexdigest()[:32]
    return f"{FORMAT}/{shape.name}/{sid}"


@dataclass(frozen=True)
class Reader:
    """The program a registration is for, as the verifier derives its domains (``registered.registered_entries``): its
    digest and the leaf width of each tensor it reads."""
    digest: str
    widths: tuple[tuple[str, int], ...]

    def word_bits(self, name: str) -> int:
        for n, b in self.widths:
            if n == name:
                return b
        raise ValueError(f"the program {self.digest[:16]}... has no parameter {name!r}")


def registered_words(t: Tensor, word_bits: int) -> int:
    """`t`'s leaves of `word_bits` bits in one registered row (the same row bytes)."""
    if 16 * t.words % word_bits:
        raise ValueError(f"{t.name}: a row of {t.words} u16 words is no whole number of {word_bits}-bit leaves")
    return 16 * t.words // word_bits


def reader(shape: Shape, program: Program | str | Reader) -> Reader:
    """`program`'s `Reader`: a `Program` reads the tensors its root parameters name, each at the parameter's width
    (``registered.width``), and must cut each into the tensor's rows; a digest alone (128 hex digits) reads every tensor
    at `Tensor.word_bits`."""
    if isinstance(program, Reader):
        return program
    if isinstance(program, str):
        if len(program) != 128 or any(c not in "0123456789abcdef" for c in program):
            raise ValueError(f"a program digest is lowercase hex of 64 bytes, not {program!r}")
        return Reader(program, tuple((t.name, t.word_bits) for t in tensors(shape)))
    if not isinstance(program, Program):
        raise TypeError(f"a reader is a Program or its digest, not {type(program).__name__}")
    params = {name for name, _ in program.fn.signature()[0]}
    widths = []
    for t in tensors(shape):
        if t.name not in params:
            continue
        b = REG.width(program, t.name)
        n = REG.shape(program, t.name, registered_words(t, b), t.schema)
        if n != t.leaves:
            raise ValueError(f"{t.name}: the program reads {n} rows of it, not its {t.leaves}")
        widths.append((t.name, b))
    if not widths:
        raise ValueError(f"the program reads no tensor of {shape.name}")
    return Reader(program_digest(program), tuple(widths))


def _salt_key(key: bytes) -> bytes:
    if not isinstance(key, bytes) or len(key) != 32:
        raise ValueError("a salt key is 32 bytes")
    return key


def salts(salt_key: bytes, index: int, first: int, n: int) -> bytes:
    """Rows ``first .. first + n - 1``'s 192-byte salts of the tensor at `index` (Flock's ``hm96_salt``)."""
    return chacha20.keystream(_salt_key(salt_key), index, _SALT_BLOCKS * first, _SALT_BLOCKS * n)


def domain(t: Tensor, r: Reader):
    """`t`'s domain under reader `r`, held across windows (no window root)."""
    b = r.word_bits(t.name)
    return REG.domain(r.digest, t.name, registered_words(t, b), t.leaves, b)


def commit(shape: Shape, seed: bytes, salt_key: bytes, program: Program | str | Reader, name: str) -> REG.Registered:
    """Tensor `name` committed whole by core's ``Registered.commit`` (small shapes; `root` is the same root in parallel)."""
    i, t = next((i, t) for i, t in enumerate(tensors(shape)) if t.name == name)
    r = reader(shape, program)
    b = r.word_bits(name)
    rows = np.frombuffer(tensor_bytes(shape, seed, name), "<u2" if b == 16 else "<u4")
    s = salts(salt_key, i, 0, t.leaves)
    return REG.Registered.commit(r.digest, name, rows.reshape(t.leaves, registered_words(t, b)).tolist(),
                                 [s[k * _INST.salt_bytes:(k + 1) * _INST.salt_bytes] for k in range(t.leaves)],
                                 word_bits=b, schema=t.schema)


def chunk_bits(t: Tensor, chunk_bytes: int = CHUNK_BYTES) -> int:
    rb = 2 * t.words
    return max(0, min(merkle_depth(t.leaves), _log2(max(1, chunk_bytes // rb))))


def _row_digests(t: Tensor, data: bytes) -> bytes:
    """Each row's ``x`` (core's ``registered.row_digest``: a bit row's digest is its bytes', whatever its leaves' width)."""
    rb, out = 2 * t.words, bytearray()
    if t.schema == REG.SCHEMA_SEG:
        for k in range(len(data) // rb):
            out += sha512_row_seg_digest(data[rb * k:rb * (k + 1)], 8 * rb)
        return bytes(out)
    row_h = hashlib.sha512(sha512_row_v2_prefix(8 * rb))
    for k in range(len(data) // rb):
        h = row_h.copy()
        h.update(data[rb * k:rb * (k + 1)])
        out += h.digest()
    return bytes(out)


def chunk_levels(t: Tensor, dom: Any, data: bytes, row_salts: bytes, j0: int, bits: int) -> list[list[bytes]]:
    """The levels of `t`'s subtree over rows ``[j0, j0 + 2^bits)`` under domain `dom`, leaves first, from those rows'
    bytes `data` and salts `row_salts` (rows past the tensor's are pads); its root is the last level's one node."""
    rb = 2 * t.words
    n = len(data) // rb
    if len(data) != n * rb or len(row_salts) != n * _INST.salt_bytes or n > 1 << bits:
        raise ValueError(f"{t.name}: {len(data)} bytes and {len(row_salts)} of salts are not as many rows of {rb} bytes")
    level = []
    if n:
        tl = hm96_batch.tree_leaves(_INST, _INST.default_key, _row_digests(t, data), row_salts)
        level = [dom.leaf(j0 + k, j0 + k, t.schema, tl[k]) for k in range(n)]
    level += [dom.pad(j) for j in range(j0 + n, j0 + (1 << bits))]
    levels, g = [level], j0
    for d in range(bits):
        level = [dom.node(d, (g >> 1) + k // 2, level[k], level[k + 1]) for k in range(0, len(level), 2)]
        levels.append(level)
        g >>= 1
    return levels


def subtree(shape: Shape, seed: bytes, salt_key: bytes, r: Reader, index: int, chunk: int, bits: int) -> bytes:
    """The subtree root over rows ``[chunk << bits, (chunk + 1) << bits)`` of the tensor at `index`, pads included."""
    t = tensors(shape)[index]
    rb, j0 = 2 * t.words, chunk << bits
    n = max(0, min(1 << bits, t.leaves - j0))
    data = tensor_bytes(shape, seed, t.name, j0 * rb, (j0 + n) * rb) if n else b""
    return chunk_levels(t, domain(t, r), data, salts(salt_key, index, j0, n) if n else b"", j0, bits)[-1][0]


def _task(args: tuple) -> tuple[int, int, bytes]:
    shape, seed, key, r, index, chunk, bits = args
    return index, chunk, subtree(shape, seed, key, r, index, chunk, bits)


def root(shape: Shape, seed: bytes, salt_key: bytes, program: Program | str | Reader,
         names: Sequence[str] | None = None, *, workers: int = 1, chunk_bytes: int = CHUNK_BYTES,
         progress: Any = None) -> dict[str, bytes]:
    """Each tensor's registered root for `program` (`names`, default every tensor it reads), equal to `commit`'s, computed
    over `workers` processes in chunks of at most `chunk_bytes`."""
    ts, r = tensors(shape), reader(shape, program)
    read = {n for n, _ in r.widths}
    for n in names or ():
        r.word_bits(tensor(shape, n).name)
    want = [i for i, t in enumerate(ts) if (t.name in names if names is not None else t.name in read)]
    plan, bits_of = [], {}
    for i in want:
        t = ts[i]
        bits = chunk_bits(t, chunk_bytes)
        bits_of[i] = bits
        plan += [(shape, seed, _salt_key(salt_key), r, i, c, bits) for c in range((1 << merkle_depth(t.leaves)) >> bits)]
    got: dict[int, dict[int, bytes]] = {i: {} for i in want}
    if workers == 1:
        results: Iterable = map(_task, plan)
        pool = None
    else:
        pool = multiprocessing.get_context("fork").Pool(workers)
        results = pool.imap_unordered(_task, plan, chunksize=1)
    try:
        for done, (i, c, sub) in enumerate(results, 1):
            got[i][c] = sub
            if progress is not None:
                progress(done, len(plan))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    out = {}
    for i in want:
        t, level, d = ts[i], [got[i][c] for c in range(len(got[i]))], bits_of[i]
        dom = domain(t, r)
        while len(level) > 1:
            level = [dom.node(d, k // 2, level[k], level[k + 1]) for k in range(0, len(level), 2)]
            d += 1
        out[t.name] = level[0]
    return out


def entry(shape: Shape, program: Program | str | Reader, name: str, value_root: bytes) -> dict[str, Any]:
    """The registration's entry of tensor `name` for `program` (core's ``Registered.entry``)."""
    t, r = tensor(shape, name), reader(shape, program)
    b = r.word_bits(name)
    return {"value": name, "root": value_root.hex(), "leaves": t.leaves, "words": registered_words(t, b),
            **REG.width_entry(b), "domain": domain(t, r).domain_id.hex()}


def entries(shape: Shape, program: Program | str | Reader, roots: dict[str, bytes]) -> list[dict[str, Any]]:
    """The entries of the tensors `roots` names, in tensor order."""
    r = reader(shape, program)
    return [entry(shape, r, t.name, roots[t.name]) for t in tensors(shape) if t.name in roots]


def registration_digest(shape: Shape, program: Program | str | Reader, roots: dict[str, bytes]) -> str:
    """SHA-512 of the registration of `roots` for `program`: ``{program: its digest, values: entries}``."""
    r = reader(shape, program)
    return identity_digest_sha512(REGISTRATION, {"program": r.digest, "values": entries(shape, r, roots)}).hex()


def _sha_task(args: tuple) -> tuple[str, str]:
    shape, seed, name = args
    return name, tensor_sha256(shape, seed, name)


def manifest(shape: Shape, seed: bytes | None = None, *, digests: bool = False, workers: int = 1,
             progress: Any = None) -> dict[str, Any]:
    """The tensor list: each tensor's name, kind, dtype, shape, stream offset and bytes, registered row words and rows,
    and with `digests` the SHA-256 of its bytes (each tensor generated whole by one of `workers` processes, the largest
    first: about 15 s per GiB on one core)."""
    sha: dict[str, str] = {}
    if digests:
        if seed is None:
            raise ValueError("a manifest's digests need the seed")
        plan = [(shape, seed, t.name) for t in sorted(tensors(shape), key=lambda t: -t.nbytes)]
        pool = multiprocessing.get_context("fork").Pool(workers) if workers > 1 else None
        try:
            it = pool.imap_unordered(_sha_task, plan, chunksize=1) if pool is not None else map(_sha_task, plan)
            for done, (name, d) in enumerate(it, 1):
                sha[name] = d
                if progress is not None:
                    progress(done, len(plan))
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    rows = []
    for t in tensors(shape):
        r = {"name": t.name, "kind": t.kind, "dtype": t.dtype, "shape": t.shape, "offset": t.offset, "nbytes": t.nbytes,
             "words": t.words, "leaves": t.leaves, "schema": t.schema,
             **({"exponent": t.exponent} if t.kind == "scale" else {}), **({"sha256": sha[t.name]} if digests else {})}
        rows.append(r)
    return {"format": MANIFEST, "generator": FORMAT, "shape": asdict(shape), "stream_bytes": stream_bytes(shape),
            **({"model": model_id(shape, seed)} if seed is not None else {}), "tensors": rows}


def tensor_sha256(shape: Shape, seed: bytes, name: str) -> str:
    t, h = tensor(shape, name), hashlib.sha256()
    step = max(1, CHUNK_BYTES // t.row_bytes) * t.row_bytes
    for lo in range(0, t.nbytes, step):
        h.update(tensor_bytes(shape, seed, name, lo, min(t.nbytes, lo + step)))
    return h.hexdigest()


def _write_task(args: tuple) -> int:
    shape, seed, path, lo, hi = args
    data = stream(shape, seed, lo, hi)
    fd = os.open(path, os.O_WRONLY)
    try:
        os.pwrite(fd, data, lo)
    finally:
        os.close(fd)
    return hi - lo


def write(shape: Shape, seed: bytes, path: str | Path, *, workers: int = 1, start: int = 0, stop: int | None = None,
          chunk_bytes: int = 1 << 26, progress: Any = None) -> int:
    """Write the weights stream's bytes ``[start, stop)`` to `path` at their stream offsets, `workers` processes each
    generating its own chunks."""
    stop = stream_bytes(shape) if stop is None else stop
    path = str(path)
    with open(path, "ab"):
        pass
    plan = [(shape, seed, path, lo, min(stop, lo + chunk_bytes)) for lo in range(start, stop, chunk_bytes)]
    done = 0
    if workers == 1:
        it: Iterable = map(_write_task, plan)
        pool = None
    else:
        pool = multiprocessing.get_context("fork").Pool(workers)
        it = pool.imap_unordered(_write_task, plan, chunksize=1)
    try:
        for n in it:
            done += n
            if progress is not None:
                progress(done, stop - start)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return done


#: the H200 demo's seed, frozen (its weights' registration roots depend on these bytes): public, so its weights are
#: reproducible; the demo's salt key is the prover's, sampled from the OS at registration
DEMO_SEED = b"verity/cmp92/h200/weights/v1"
#: the fixture's public seed and dummy salt key: test values, never the demo's
TEST_SEED = b"verity/seeded-weights/test-seed"
TEST_SALT_KEY = hashlib.sha512(b"not-a-secret: the seeded weights' test salt key").digest()[:32]
#: the fixture's reader, a digest of no program: every tensor at `Tensor.word_bits`
TEST_PROGRAM = hashlib.sha512(b"not-a-program: the seeded weights' test reader").hexdigest()


def first_rows(shape: Shape, seed: bytes) -> list[list[str]]:
    """Each tensor's ``[name, SHA-256 of generation row 0, SHA-256 of its last generation row]`` (its bytes)."""
    out = []
    for t in tensors(shape):
        rb = t.row_bytes
        out.append([t.name, hashlib.sha256(tensor_bytes(shape, seed, t.name, 0, rb)).hexdigest(),
                    hashlib.sha256(tensor_bytes(shape, seed, t.name, t.nbytes - rb, t.nbytes)).hexdigest()])
    return out


def fixture() -> dict[str, Any]:
    """The interface's fixture (``tests/fixtures/weights-v1.json``): under `TEST_SEED` and `TEST_SALT_KEY`, `TINY`'s and
    `PHASE1`'s manifests with digests, and their entries and registration digests for `TEST_PROGRAM`; at `H200` the
    stream's length and first 64 bytes, the first and last rows of layer 0's tensors and of those outside the layers, and
    SHA-256 of every tensor's
    (`first_rows`, as compact JSON). `H200`'s whole manifest, with each tensor's SHA-256 under `DEMO_SEED`, is an artifact
    of the evidence store (``examples/h200/README.md``)."""
    out: dict[str, Any] = {"format": "verity/seeded-weights/fixture/v1", "seed": TEST_SEED.hex(),
                           "salt_key": TEST_SALT_KEY.hex(), "program": TEST_PROGRAM, "shapes": {}}
    for s in (TINY, PHASE1):
        roots = root(s, TEST_SEED, TEST_SALT_KEY, TEST_PROGRAM)
        out["shapes"][s.name] = {"manifest": manifest(s, TEST_SEED, digests=True),
                                 "values": entries(s, TEST_PROGRAM, roots),
                                 "registration": registration_digest(s, TEST_PROGRAM, roots)}
    rows = first_rows(H200, TEST_SEED)
    out["shapes"][H200.name] = {
        "stream_bytes": stream_bytes(H200), "stream_head": stream(H200, TEST_SEED, 0, 64).hex(),
        "rows": [r for r in rows if not r[0].startswith("l") or r[0].startswith("l0.")],
        "rows_sha256": hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()}
    return out


def dumps(doc: Any) -> str:
    """JSON with every list of objects or lists one item a line."""
    def go(x: Any, ind: str) -> str:
        if isinstance(x, dict) and x:
            inner = ind + " "
            return "{\n" + ",\n".join(f"{inner}{json.dumps(k)}: {go(v, inner)}" for k, v in sorted(x.items())) + f"\n{ind}}}"
        if isinstance(x, list) and x and all(isinstance(v, (dict, list)) for v in x):
            inner = ind + " "
            return "[\n" + ",\n".join(inner + json.dumps(v, sort_keys=True, separators=(",", ":")) for v in x) + f"\n{ind}]"
        return json.dumps(x, sort_keys=True, separators=(",", ":"))
    return go(doc, "") + "\n"


def _shape_arg(name: str) -> Shape:
    s = ALIASES.get(name) or SHAPES.get(name)
    if s is None:
        raise argparse.ArgumentTypeError(f"a shape is one of {sorted(ALIASES) + sorted(SHAPES)}")
    return s


def _bytes_arg(text: str) -> bytes:
    """``hex:...``, ``file:PATH`` (its bytes, a trailing newline stripped) or ``text:...``."""
    kind, _, v = text.partition(":")
    if kind == "hex":
        return bytes.fromhex(v)
    if kind == "file":
        return Path(v).read_bytes().rstrip(b"\n")
    if kind == "text":
        return v.encode()
    raise argparse.ArgumentTypeError("hex:..., file:PATH or text:...")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("manifest", "write", "root", "fixture"):
        p = sub.add_parser(name)
        if name != "fixture":
            p.add_argument("--shape", type=_shape_arg, default=H200)
            p.add_argument("--seed", type=_bytes_arg, help="hex:..., file:PATH or text:...")
        p.add_argument("--out", required=True)
        p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    sub.choices["manifest"].add_argument("--digests", action="store_true")
    sub.choices["write"].add_argument("--start", type=int, default=0)
    sub.choices["write"].add_argument("--stop", type=int)
    sub.choices["root"].add_argument("--salt-key", type=_bytes_arg, required=True,
                                     help="the prover's 32-byte salt key: file:PATH, never on a command line in the demo")
    sub.choices["root"].add_argument("--program", required=True,
                                     help="the reading program's digest (128 hex digits); each tensor read at "
                                          "Tensor.word_bits, 16")
    sub.choices["root"].add_argument("--tensors", help="comma-separated tensor names (default all)")
    a = ap.parse_args(argv)

    def say(done: int, total: int) -> None:
        if done == total or done % max(1, total // 100) == 0:
            print(f"{a.cmd}: {done}/{total}", file=sys.stderr, flush=True)

    if a.cmd == "fixture":
        Path(a.out).write_text(dumps(fixture()))
        return 0
    if a.seed is None:
        ap.error("--seed is required")
    if a.cmd == "manifest":
        Path(a.out).write_text(dumps(manifest(a.shape, a.seed, digests=a.digests, workers=a.workers, progress=say)))
    elif a.cmd == "write":
        write(a.shape, a.seed, a.out, workers=a.workers, start=a.start, stop=a.stop, progress=say)
    else:
        names = a.tensors.split(",") if a.tensors else None
        roots = root(a.shape, a.seed, a.salt_key, a.program, names, workers=a.workers, progress=say)
        doc: dict[str, Any] = {"format": REGISTRATION, "program": a.program, "model": model_id(a.shape, a.seed),
                               "shape": a.shape.name, "values": entries(a.shape, a.program, roots)}
        if names is None:
            doc["registration"] = registration_digest(a.shape, a.program, roots)
        Path(a.out).write_text(dumps(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
