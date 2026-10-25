"""The demo's seeded weights and their registration root (CMP-95 interface 1): one deterministic generator from a seed to
every tensor of a decoder of any `Shape` (the H200 demo's frozen shape `H200`, Phase 1's CPU model `PHASE1`), and the
registration of those tensors as core's registered values (``one_stage.registered``), computed in parallel.

No torch: memory-accounting's encoder, the loop and the all-protocols runner run it with this directory on their path
(``PYTHONPATH=integrations/torch_loop python -m verity_torch_loop.weights ...``).

* **The model** (`Shape`): a pre-norm decoder with learned absolute positions, ``heads`` query heads of ``head_dim`` and
  ``kv_heads`` KV heads (1: MQA), a ReLU MLP of width ``mlp``, a byte vocabulary, W8A8 linears (s8 codes, an f32 scale per
  output channel) and bf16 embeddings and norm gains.
* **The tensors** (`tensors`), in this order: ``tok`` (H, V), ``pos`` (H, C), then per layer ``l<i>.`` ``attn_norm`` (H),
  ``wq`` (heads·d, H), ``wk``, ``wv`` (kv_heads·d, H), ``wo`` (H, heads·d), ``mlp_norm`` (H), ``wup`` (F, H), ``wdown``
  (H, F), each linear followed by its ``.scale`` (N), then ``final_norm`` (H), ``head`` (V, H) and ``head.scale``. A 2-D
  tensor is (output channel, input) row-major: row r is output channel r's elements in ascending input index, so the
  embeddings are stored transposed (row j is coordinate j over the vocabulary, or over the positions), the row an
  embedding coordinate's unit reads. A tensor's bytes are its elements little-endian (s8 codes, f32 scales, bf16 words).
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
* **The registration** (`root`, `entries`): every tensor is one registered value of its name (core's
  ``Registered.commit``): its bytes cut into rows of `Tensor.words` u16 words (little-endian, the reading port's row, as
  CIR-100's units read them: an s8 linear ``row_codes`` codes, the K-part ``S8AccumulatorPart{K=row_codes}`` reads; a scale
  2 words; ``tok`` a column of V words; a gain and ``pos`` 1 word), in ``hm96-sha512/row/v2``, under
  ``registered.domain(program, name, words, leaves)`` (held across windows; ``program`` = `program`, the model's name as
  ALLP-WIRE's loop names its weights), row i's salt ChaCha20 blocks 3i .. 3i+2 under the prover's 32-byte salt key at the
  tensor's index as nonce (C-Flock's ``hm96_salt``; the key salts these rows only). At `H200` every tensor's row count is
  a power of two, so no tree has a pad. **The registration digest** (`registration_digest`) is SHA-512 of
  ``verity/seeded-weights/registration/v1`` and ``{program, values}`` (every entry in tensor order): the one root the
  weights' store ties to.
* **In parallel** (`root`): a value's rows are cut into aligned chunks of a power of two rows; each worker generates its
  chunk's bytes itself and returns the chunk's subtree root, and the caller folds them as ``tree_levels`` does. Only the
  seed, the salt key and chunk indices cross processes.
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
from verity.ml.commitments.rowleaf import sha512_row_v2_prefix
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


#: the H200 demo's frozen shape (compute-accounting, 25 Oct 23:00Z)
H200 = Shape("cmp92-h200-mqa-w16384-l52", 16384, 52, 256, 64, 1, 65536, 256, 32, 256)
#: Phase 1's CPU model (ALLP-XFMR, CMP-92): its linears read whole 128-code rows, its down projection in K = 128 parts
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

    @property
    def dtype(self) -> str:
        return {"s8": "s8", "scale": "f32"}.get(self.kind, "bf16")

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

    def add(name: str, kind: str, rows: int, cols: int, words: int, exponent: int = 0) -> None:
        nonlocal off
        t = Tensor(name, kind, rows, cols, words, off, exponent)
        if t.nbytes % (2 * words):
            raise ValueError(f"{name}: {t.nbytes} bytes are no whole rows of {words} words")
        out.append(t)
        off += t.nbytes

    def linear(name: str, n: int, k: int, shift: int = 0) -> None:
        add(name, "s8", n, k, s.row_codes // 2)
        add(f"{name}.scale", "scale", 1, n, 2, scale_exponent(k, shift))

    add("tok", "embed", s.width, s.vocab, s.vocab)
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


def program(shape: Shape, seed: bytes) -> str:
    """What every value's domain binds as its program: the format, the shape and the seed's id (not the seed)."""
    sid = hashlib.sha512(FORMAT.encode() + b"/seed-id\0" + _seed(seed)).hexdigest()[:32]
    return f"{FORMAT}/{shape.name}/{sid}"


def _salt_key(key: bytes) -> bytes:
    if not isinstance(key, bytes) or len(key) != 32:
        raise ValueError("a salt key is 32 bytes")
    return key


def salts(salt_key: bytes, index: int, first: int, n: int) -> bytes:
    """Rows ``first .. first + n - 1``'s 192-byte salts of the tensor at `index` (C-Flock's ``hm96_salt``)."""
    return chacha20.keystream(_salt_key(salt_key), index, _SALT_BLOCKS * first, _SALT_BLOCKS * n)


def domain(shape: Shape, seed: bytes, t: Tensor):
    return REG.domain(program(shape, seed), t.name, t.words, t.leaves)


def commit(shape: Shape, seed: bytes, salt_key: bytes, name: str) -> REG.Registered:
    """Tensor `name` committed whole by core's ``Registered.commit`` (small shapes; `root` is the same root in parallel)."""
    i, t = next((i, t) for i, t in enumerate(tensors(shape)) if t.name == name)
    words = np.frombuffer(tensor_bytes(shape, seed, name), "<u2").reshape(t.leaves, t.words)
    s = salts(salt_key, i, 0, t.leaves)
    return REG.Registered.commit(program(shape, seed), name, words.tolist(),
                                 [s[k * _INST.salt_bytes:(k + 1) * _INST.salt_bytes] for k in range(t.leaves)],
                                 schema=SCHEMA)


def chunk_bits(t: Tensor, chunk_bytes: int = CHUNK_BYTES) -> int:
    rb = 2 * t.words
    return max(0, min(merkle_depth(t.leaves), _log2(max(1, chunk_bytes // rb))))


def subtree(shape: Shape, seed: bytes, salt_key: bytes, index: int, chunk: int, bits: int) -> bytes:
    """The subtree root over rows ``[chunk << bits, (chunk + 1) << bits)`` of the tensor at `index`, pads included."""
    t = tensors(shape)[index]
    dom, rb = domain(shape, seed, t), 2 * t.words
    j0 = chunk << bits
    n = max(0, min(1 << bits, t.leaves - j0))
    level = []
    if n:
        data = tensor_bytes(shape, seed, t.name, j0 * rb, (j0 + n) * rb)
        row_h = hashlib.sha512(sha512_row_v2_prefix(8 * rb))
        digests = bytearray()
        for k in range(n):
            h = row_h.copy()
            h.update(data[rb * k:rb * (k + 1)])
            digests += h.digest()
        tl = hm96_batch.tree_leaves(_INST, _INST.default_key, bytes(digests), salts(salt_key, index, j0, n))
        level = [dom.leaf(j0 + k, j0 + k, SCHEMA, tl[k]) for k in range(n)]
    level += [dom.pad(j) for j in range(j0 + n, j0 + (1 << bits))]
    g = j0
    for d in range(bits):
        level = [dom.node(d, (g >> 1) + k // 2, level[k], level[k + 1]) for k in range(0, len(level), 2)]
        g >>= 1
    return level[0]


def _task(args: tuple) -> tuple[int, int, bytes]:
    shape, seed, key, index, chunk, bits = args
    return index, chunk, subtree(shape, seed, key, index, chunk, bits)


def root(shape: Shape, seed: bytes, salt_key: bytes, names: Sequence[str] | None = None, *, workers: int = 1,
         chunk_bytes: int = CHUNK_BYTES, progress: Any = None) -> dict[str, bytes]:
    """Each tensor's registered root (`names`, default every tensor), equal to `commit`'s, computed over `workers`
    processes in chunks of at most `chunk_bytes`."""
    ts = tensors(shape)
    want = [i for i, t in enumerate(ts) if names is None or t.name in names]
    plan, bits_of = [], {}
    for i in want:
        t = ts[i]
        bits = chunk_bits(t, chunk_bytes)
        bits_of[i] = bits
        plan += [(shape, seed, _salt_key(salt_key), i, c, bits) for c in range((1 << merkle_depth(t.leaves)) >> bits)]
    got: dict[int, dict[int, bytes]] = {i: {} for i in want}
    if workers == 1:
        results: Iterable = map(_task, plan)
        pool = None
    else:
        pool = multiprocessing.get_context("fork").Pool(workers)
        results = pool.imap_unordered(_task, plan, chunksize=1)
    try:
        for done, (i, c, r) in enumerate(results, 1):
            got[i][c] = r
            if progress is not None:
                progress(done, len(plan))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    out = {}
    for i in want:
        t, level, d = ts[i], [got[i][c] for c in range(len(got[i]))], bits_of[i]
        dom = domain(shape, seed, t)
        while len(level) > 1:
            level = [dom.node(d, k // 2, level[k], level[k + 1]) for k in range(0, len(level), 2)]
            d += 1
        out[t.name] = level[0]
    return out


def entry(shape: Shape, seed: bytes, name: str, value_root: bytes) -> dict[str, Any]:
    """The registration's entry of tensor `name` (core's ``Registered.entry``)."""
    t = tensor(shape, name)
    return {"value": name, "root": value_root.hex(), "leaves": t.leaves, "words": t.words,
            "domain": domain(shape, seed, t).domain_id.hex()}


def entries(shape: Shape, seed: bytes, roots: dict[str, bytes]) -> list[dict[str, Any]]:
    return [entry(shape, seed, t.name, roots[t.name]) for t in tensors(shape)]


def registration_digest(shape: Shape, seed: bytes, roots: dict[str, bytes]) -> str:
    return identity_digest_sha512(REGISTRATION, {"program": program(shape, seed),
                                                 "values": entries(shape, seed, roots)}).hex()


def manifest(shape: Shape, seed: bytes | None = None, *, digests: bool = False) -> dict[str, Any]:
    """The tensor list: each tensor's name, kind, dtype, shape, stream offset and bytes, registered row words and rows,
    and with `digests` the SHA-256 of its bytes (generated whole: minutes per GiB on one core)."""
    rows = []
    for t in tensors(shape):
        r = {"name": t.name, "kind": t.kind, "dtype": t.dtype, "shape": t.shape, "offset": t.offset, "nbytes": t.nbytes,
             "words": t.words, "leaves": t.leaves, **({"exponent": t.exponent} if t.kind == "scale" else {})}
        if digests:
            if seed is None:
                raise ValueError("a manifest's digests need the seed")
            r["sha256"] = tensor_sha256(shape, seed, t.name)
        rows.append(r)
    return {"format": MANIFEST, "generator": FORMAT, "shape": asdict(shape), "stream_bytes": stream_bytes(shape),
            **({"program": program(shape, seed)} if seed is not None else {}), "tensors": rows}


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


#: the fixture's public seed and dummy salt key: test values, never the demo's
TEST_SEED = b"verity/seeded-weights/test-seed"
TEST_SALT_KEY = hashlib.sha512(b"not-a-secret: the seeded weights' test salt key").digest()[:32]


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
    `PHASE1`'s manifests with digests, entries and registration digests, `H200`'s manifest and each tensor's first and
    last rows, and the first 64 bytes of `H200`'s stream."""
    out: dict[str, Any] = {"format": "verity/seeded-weights/fixture/v1", "seed": TEST_SEED.hex(),
                           "salt_key": TEST_SALT_KEY.hex(), "shapes": {}}
    for s in (TINY, PHASE1):
        roots = root(s, TEST_SEED, TEST_SALT_KEY)
        out["shapes"][s.name] = {"manifest": manifest(s, TEST_SEED, digests=True),
                                 "values": entries(s, TEST_SEED, roots),
                                 "registration": registration_digest(s, TEST_SEED, roots)}
    m = manifest(H200, TEST_SEED)
    out["shapes"][H200.name] = {"manifest": m, "rows": first_rows(H200, TEST_SEED),
                                "stream_head": stream(H200, TEST_SEED, 0, 64).hex()}
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
        Path(a.out).write_text(dumps(manifest(a.shape, a.seed, digests=a.digests)))
    elif a.cmd == "write":
        write(a.shape, a.seed, a.out, workers=a.workers, start=a.start, stop=a.stop, progress=say)
    else:
        names = a.tensors.split(",") if a.tensors else None
        roots = root(a.shape, a.seed, a.salt_key, names, workers=a.workers, progress=say)
        doc: dict[str, Any] = {"format": REGISTRATION, "program": program(a.shape, a.seed), "shape": a.shape.name,
                               "values": [entry(a.shape, a.seed, n, r) for n, r in roots.items()]}
        if names is None:
            doc["registration"] = registration_digest(a.shape, a.seed, roots)
        Path(a.out).write_text(dumps(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
