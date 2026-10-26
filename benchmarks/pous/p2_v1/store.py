"""The encoded weights store (CMP-95 item 2): the frozen MQA model's INT8 weights as `W` under the registered P2
(`p2-16448/v4`), laid out so that one layer, or one matrix, decodes alone into a staging buffer, and the reference
decode a kernel's `decode_layer` matches byte for byte.

    store.py layout [--out store-layout.json]
    store.py toy [--out store-toy] [--width 64] [--layers 2] [--ff 64] [--per-segment 11] [--workers N]
    store.py toy --plain [--out store-toy]
    store.py commit --store DIR [--out $OUT] --seed text:... --manifest MANIFEST.json [--generator G] [--shape h200]
        [--workers N]

`layout` writes the real store's layout, which Git doesn't track (`LAYOUT_SHA512` pins it). `toy` writes the toy
fixture (`store-toy/`): `toy.json`, `spot.json`, `spot-c.bin` and `c-star.bin`, which are tracked, and `w.bin` and
`staging.bin`, which are not (`toy.json` pins each by its digest). `toy --plain` writes those two in under a second with
no encoding.

`commit` is `setup.commit` of the real `W` under the layout's manifest, before any salt exists, into `setup.json` in
`--store` and `--out` (p2_v1/p2-encode's `commit`, with `W` laid out here). `W` is regenerated from the seeded generator
(`Generated`: `verity_torch_loop.weights`, a module name or its `.py` file) by `--workers` forked processes, and each of
the generator's s8 tensors is held to `--manifest`'s SHA-256 (the generator's `manifest(digests=True)`) as the pass reads
it.

**Layout.** The tensors, all s8 `[N, K]` row-major as `ScaledMmS8Coordinate` reads them, in this order: for each layer
`l`, `model.layers.{l}.self_attn.qkv_proj.weight` (rows: the query heads, then K, then V), `...self_attn.o_proj.weight`,
`...mlp.up_proj.weight`, `...mlp.down_proj.weight`; then `lm_head.weight` as layer `layers`. Embeddings, norms and
scales are not in `W`. Each tensor starts on a block: its `W` offset is `first_block · 2054`, it takes `⌈size / 2054⌉`
blocks, and its last block's payload is zero past the tensor, so no block holds bytes of two tensors. `W` is zero to
whole segments of `blocks_per_segment` blocks (`Layout.of_ranges`, which `layout.decode_tensor` reads as is).
`Layout.of` doesn't fit: packed, blocks straddle tensors; `align=True` starts each tensor on a segment, 7 GiB of
padding at the real shape.

**C*.** Block `i = blocks_per_segment · s + v` is 2,056 bytes, big-endian, at byte `2056 · i` of one buffer in HBM; on
disk, one file per segment, blocks in `v` order. Its leaf is `SHA512(0x00 ‖ tag_s ‖ u16le(v) ‖ R)`, `tag_s = salt ‖
u64le(s)` (32 bytes) and `R` its `sha512/row/v2` digest, under nodes `SHA512(0x01 ‖ l ‖ r)` of RFC 6962's shape
(`verifier.VerifierKey`). The commitment is `setup.commit(params, W)` and the salt `setup.draw_salt(source,
commitment)`.

**Decode.** Layer `l`'s staging buffer holds its tensors back to back from offset 0, each at a multiple of
`STAGING_ALIGN`: `staging[t.staging_offset + b] = W[t.offset + b]` for `b < t.size`, which is payload byte `b mod 2054`
of block `t.first_block + ⌊b / 2054⌋`. Block `i`'s payload is `m = sq(σ_K(sq(c))) − t mod p` (`reference.decode`),
its `K` and `t` from `reference.keys_after(prefix, …, s, v)` with `prefix = schemes.p2.expand_prefix(params, salt)`,
128 bytes. `decode_range` refuses what a kernel flags: a block not below `p` or decoding to more than 16,432 bits (flag
bit 2), and a nonzero byte past a tensor's end in its last block (bit 3). A kernel that takes `t`'s first candidate
flags bit 0 when it is not below `p` (about `2^-16434` a block), where the reference draws the next.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import importlib.util
import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from p2_v1 import reference
else:
    from . import reference

from verity.core.protocols.pous import setup
from verity.core.protocols.pous.layout import Layout
from verity.core.protocols.pous.schemes.p2 import P2, POINT, P2Params, encode_segment, expand_prefix, stream

HERE = Path(__file__).resolve().parent
LAYOUT_JSON = HERE / "store-layout.json"
LAYOUT_SHA512 = "45436c2a341cfef20219a071b9b27bd9013535b1de9001ae22d0bf3bd25a825e6aa7c82a60734047d1de59c38e572e126e3c7c64c78f1176e258cb8cb833b783"
"""`store.py layout`'s output: the test regenerates it and holds it to this."""
TOY_DIR = HERE / "store-toy"
SCHEMA = "verity-pous/p2-store-layout/v1"
TOY_SCHEMA = "verity-pous/p2-store-toy/v1"
SPOT_SCHEMA = "verity-pous/p2-store-spot/v1"
W_BITS, C0, PER_SEGMENT = POINT
STAGING_ALIGN = 256
"""Every tensor's staging offset is a multiple of this (TMA wants 16 bytes; a 2,054-multiple would give 2)."""
TOY_SOURCE = hashlib.sha512(b"verity-pous/p2-store/toy/setup-source").digest()
TOY_W = b"verity-pous/p2-store/toy/w/v1\0"
SETUP_FORMAT = "verity-pous/p2/store/v1"
GENERATOR = "verity_torch_loop.weights"
PARTS = {"self_attn.qkv_proj.weight": ("wq", "wk", "wv"), "self_attn.o_proj.weight": ("wo",),
         "mlp.up_proj.weight": ("wup",), "mlp.down_proj.weight": ("wdown",)}
"""A layer's tensor's rows: the generator's `l<layer>.<part>` tensors', in order (`lm_head.weight` is `head`)."""
TENSOR_COLUMNS = ("name", "layer", "n", "k", "offset", "size", "first_block", "blocks", "staging_offset")
LAYER_COLUMNS = ("layer", "first_block", "blocks", "staging_bytes")
FRAMING = {
    "block": "i = blocks_per_segment * s + v; W[2054 i : 2054 (i + 1)] is block i's payload m_i, big-endian",
    "c_star": "block i is 2056 bytes, big-endian, at byte 2056 i; on disk one file per segment s, blocks in v order",
    "keys": "prefix = schemes.p2.expand_prefix(params, salt) (128 B); base = prefix || u64be(s) || u64be(v); "
            "K = stream(base, 'K', 2056); t = the first 2056-byte word of stream(base, 't') below p",
    "decode": "m = sq(sigma_K(sq(c))) - t mod p; sq(a) = a^2 mod p, negated for odd a; sigma_K(r) = r xor K if below p, "
              "else r; refused: c >= p, m >= 2^16432, nonzero padding past a tensor's end",
    "staging": "layer l's tensors back to back from 0; staging[staging_offset + b] = W[offset + b] for b < size",
    "leaf": "SHA512(0x00 || salt || u64le(s) || u16le(v) || sha512/row/v2(c_i)); nodes SHA512(0x01 || l || r), RFC 6962",
}


@dataclass(frozen=True)
class Shape:
    """A transformer's INT8 matrices: `layers` layers of attention (`heads` query heads and `kv_heads` KV heads of
    `head_dim`), an `ff`-wide MLP (up, then down, not gated), and a `vocab`-row head."""

    name: str
    d: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    ff: int
    vocab: int

    def tensors(self) -> list[tuple[str, int, int, int]]:
        """`(name, layer, N, K)` in `W`'s order."""
        out = []
        for l in range(self.layers):
            at = f"model.layers.{l}."
            out += [(at + "self_attn.qkv_proj.weight", l, (self.heads + 2 * self.kv_heads) * self.head_dim, self.d),
                    (at + "self_attn.o_proj.weight", l, self.d, self.heads * self.head_dim),
                    (at + "mlp.up_proj.weight", l, self.ff, self.d),
                    (at + "mlp.down_proj.weight", l, self.d, self.ff)]
        return out + [("lm_head.weight", self.layers, self.vocab, self.d)]


MQA16K = Shape("mqa16k-int8", 16_384, 52, 256, 1, 64, 65_536, 256)


def toy_shape(width: int = 64, layers: int = 2, ff: int = 64) -> Shape:
    return Shape(f"toy-mqa-d{width}", width, layers, width // 64, 1, 64, ff, 256)


@dataclass(frozen=True)
class Tensor:
    name: str
    layer: int
    n: int
    k: int
    offset: int
    first_block: int
    blocks: int
    staging_offset: int

    @property
    def size(self) -> int:
        return self.n * self.k


@dataclass(frozen=True)
class Store:
    shape: Shape
    layout: Layout
    tensors: tuple[Tensor, ...]

    @property
    def params(self) -> P2Params:
        return self.layout.params

    def layer(self, layer: int) -> tuple[Tensor, ...]:
        out = tuple(t for t in self.tensors if t.layer == layer)
        if not out:
            raise KeyError(f"no layer {layer} in the store")
        return out

    def staging_bytes(self, layer: int) -> int:
        last = self.layer(layer)[-1]
        return last.staging_offset + last.size

    def to_json(self) -> dict:
        p, layers = self.params, sorted({t.layer for t in self.tensors})
        rows = []
        for l in layers:
            ts = self.layer(l)
            rows.append([l, ts[0].first_block, sum(t.blocks for t in ts), self.staging_bytes(l)])
        return {"schema": SCHEMA, "scheme": p.codec.name, "model": asdict(self.shape) | {"dtype": "int8",
                "order": "[N, K] row-major"}, "params": p.manifest(), "block_bytes": p.block_bytes,
                "payload_bytes": p.payload_bytes, "blocks": p.blocks, "used_blocks": sum(t.blocks for t in self.tensors),
                "w_bytes": p.w_bytes, "c_star_bytes": p.blocks * p.block_bytes,
                "segment_file_bytes": p.blocks_per_segment * p.block_bytes,
                "tensor_bytes": sum(t.size for t in self.tensors), "staging_align": STAGING_ALIGN,
                "staging_bytes": max(r[3] for r in rows), "framing": FRAMING, "tensor_columns": list(TENSOR_COLUMNS),
                "tensors": [[getattr(t, c) for c in TENSOR_COLUMNS] for t in self.tensors],
                "layer_columns": list(LAYER_COLUMNS), "layers": rows}


def build(shape: Shape, blocks_per_segment: int = PER_SEGMENT) -> Store:
    """The store's layout: each tensor from a block boundary, in `shape.tensors()`'s order."""
    params = P2Params(W_BITS, C0, blocks_per_segment)
    pb = params.payload_bytes
    tensors, block, layer, staging = [], 0, None, 0
    for name, l, n, k in shape.tensors():
        if l != layer:
            layer, staging = l, 0
        blocks = -(-n * k // pb)
        tensors.append(Tensor(name, l, n, k, block * pb, block, blocks, staging))
        block += blocks
        staging += -(-n * k // STAGING_ALIGN) * STAGING_ALIGN
    layout = Layout.of_ranges(params, [(t.name, t.offset, t.size) for t in tensors])
    return Store(shape, layout, tuple(tensors))


def toy_store(doc: dict | None = None) -> Store:
    """The store `toy.json` describes (the tracked one by default)."""
    doc = json.loads((TOY_DIR / "toy.json").read_text()) if doc is None else doc
    model = doc["layout"]["model"]
    shape = Shape(**{k: model[k] for k in Shape.__dataclass_fields__})
    return build(shape, doc["layout"]["params"]["blocks_per_segment"])


def _ref(params: P2Params) -> reference.Params:
    return reference.Params(params.w, params.c0, params.payload_bytes, params.blocks_per_segment, params.segments)


def decode_block(params: P2Params, prefix: bytes, i: int, block: bytes) -> bytes:
    """Block `i`'s payload, `params.payload_bytes` bytes: `reference.decode` under `reference.keys_after(prefix, …)`."""
    ref, nb, pb = _ref(params), params.block_bytes, params.payload_bytes
    if len(block) != nb:
        raise ValueError(f"block {i} is {len(block)} bytes, not {nb}")
    c = int.from_bytes(block, "big")
    if c >= ref.p:
        raise ValueError(f"block {i} is not below p: not a codeword (flag bit 2)")
    s, v = divmod(i, params.blocks_per_segment)
    k, t, _ = reference.keys_after(prefix, ref, s, v)
    m = reference.decode(ref, c, k, t)
    if m >> (8 * pb):
        raise ValueError(f"block {i} decodes to more than {8 * pb} bits: not a codeword (flag bit 2)")
    return m.to_bytes(pb, "big")


def decode_range(params: P2Params, c_star: bytes | memoryview, first_block: int, size: int, prefix: bytes) -> bytes:
    """`size` bytes of `W` from block `first_block` on, decoded from `c_star` (block `i` at byte `2056 · i`); the rest of
    the last block's payload must be zero (flag bit 3)."""
    nb, pb = params.block_bytes, params.payload_bytes
    out = bytearray()
    for i in range(first_block, first_block + -(-size // pb)):
        out += decode_block(params, prefix, i, bytes(c_star[nb * i:nb * (i + 1)]))
    if any(out[size:]):
        raise ValueError(f"block {first_block + len(out) // pb - 1} is nonzero past the tensor's end (flag bit 3)")
    return bytes(out[:size])


def decode_layer(store: Store, c_star: bytes | memoryview, salt: bytes, layer: int) -> bytes:
    """Layer `layer`'s staging buffer, `store.staging_bytes(layer)` bytes (zero between tensors): what a kernel's
    `decode_layer(c_star, layout, layer, prefix, staging)` writes."""
    prefix = expand_prefix(store.params, salt)
    out = bytearray(store.staging_bytes(layer))
    for t in store.layer(layer):
        out[t.staging_offset:t.staging_offset + t.size] = decode_range(store.params, c_star, t.first_block, t.size, prefix)
    return bytes(out)


def refusal(store: Store, c_star: bytes, salt: bytes, layer: int) -> int | None:
    """The flag bit `decode_layer` refuses layer `layer` with, or None when it decodes."""
    try:
        decode_layer(store, c_star, salt, layer)
    except ValueError as e:
        return int(str(e).rsplit("flag bit ", 1)[1].rstrip(")"))
    return None


def patch(c_star: bytes, block: int, c: bytes) -> bytes:
    """`c_star` with block `block` replaced by `c`."""
    return c_star[:len(c) * block] + c + c_star[len(c) * (block + 1):]


def negative(neg: dict, c_star: bytes, params: P2Params) -> bytes:
    """`c_star` with a negative's block in place: its `c` in hex, `"p"` for `p` itself, or `C*`'s own block with byte
    `xor[0]` xored by `xor[1]`."""
    nb, i = params.block_bytes, neg["block"]
    if "xor" in neg:
        c = bytearray(c_star[nb * i:nb * (i + 1)])
        c[neg["xor"][0]] ^= neg["xor"][1]
    else:
        c = params.p.to_bytes(nb, "big") if neg["c"] == "p" else bytes.fromhex(neg["c"])
    return patch(c_star, i, bytes(c))


def staging_of(store: Store, w: bytes, layer: int) -> bytes:
    """Layer `layer`'s staging buffer from the plaintext `W`: what `decode_layer` must give."""
    out = bytearray(store.staging_bytes(layer))
    for t in store.layer(layer):
        out[t.staging_offset:t.staging_offset + t.size] = w[t.offset:t.offset + t.size]
    return bytes(out)


def toy_tensors(store: Store) -> dict[str, bytes]:
    """The toy's weights: each tensor's bytes from a SHA-512 counter stream under its name."""
    return {t.name: stream(TOY_W, t.name.encode(), t.size) for t in store.tensors}


def toy_plain(store: Store) -> dict[str, bytes]:
    """The toy's `w.bin` (W) and `staging.bin` (every layer's staging buffer, in layer order), from the generator alone."""
    w = store.layout.pack(toy_tensors(store))
    return {"w.bin": w, "staging.bin": b"".join(staging_of(store, w, l) for l in sorted({t.layer for t in store.tensors}))}


def spot_payload(i: int, used: int, payload_bytes: int) -> bytes:
    """A spot vector's payload: `used` bytes of a SHA-512 counter stream under its index, then zeros."""
    return stream(TOY_W + b"spot", i.to_bytes(8, "big"), used) + bytes(payload_bytes - used)


def w_range(store: Store, tensor: Callable[[str, int, int], bytes], lo: int, hi: int) -> bytes:
    """Bytes `[lo, hi)` of `W`: tensor `t`'s bytes `[a, b)`, `tensor(t.name, a, b)`, at `t.offset + a`, zero elsewhere."""
    if not 0 <= lo <= hi <= store.params.w_bytes:
        raise ValueError(f"W has bytes [0, {store.params.w_bytes}), not [{lo}, {hi})")
    out = bytearray(hi - lo)
    view = memoryview(out)
    for t in store.tensors:
        a, b = max(lo, t.offset), min(hi, t.offset + t.size)
        if a < b:
            view[a - lo:b - lo] = tensor(t.name, a - t.offset, b - t.offset)
    return bytes(out)


def frozen(shape: Shape) -> Shape:
    """`MQA16K` itself for a shape of its dimensions, so that its store is the frozen one (`LAYOUT_SHA512`)."""
    return MQA16K if replace(shape, name=MQA16K.name) == MQA16K else shape


def generator_parts(name: str) -> tuple[str, ...]:
    """The generator's tensors whose rows, in order, are the layout's tensor `name`'s (`PARTS`)."""
    if name == "lm_head.weight":
        return ("head",)
    _, _, layer, rest = name.split(".", 3)
    return tuple(f"l{layer}.{p}" for p in PARTS[rest])


def load_generator(spec: str) -> Any:
    """`spec`: the generator's module file (`.py`), or its module name."""
    if not spec.endswith(".py"):
        return importlib.import_module(spec)
    name = "_p2_store_generator"
    if name not in sys.modules:
        found = importlib.util.spec_from_file_location(name, spec)
        if found is None or found.loader is None:
            raise ImportError(f"no generator module at {spec}")
        mod = importlib.util.module_from_spec(found)
        sys.modules[name] = mod
        found.loader.exec_module(mod)
    return sys.modules[name]


def git_blob(path: str) -> str | None:
    """A module file's git blob id, which names it at any commit that has it."""
    if not path.endswith(".py"):
        return None
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


class Generated:
    """`W` from a seeded weights generator (`verity_torch_loop.weights`: `tensors`, `tensor_bytes`, `stream_bytes`,
    `program`) at a shape and seed, laid out by `build`: each of the layout's tensors is its `generator_parts`' bytes
    back to back, each part an s8 tensor of the layout tensor's `K` columns."""

    def __init__(self, generator: str, shape: str, seed: bytes) -> None:
        self.generator, self.shape_name, self.seed = generator, shape, seed
        self.mod = load_generator(generator)
        self.shape = g = getattr(self.mod, "ALIASES", {}).get(shape) or self.mod.SHAPES[shape]
        self.store = build(frozen(Shape(g.name, g.width, g.layers, g.heads, g.kv_heads, g.head_dim, g.mlp, g.vocab)))
        gen = {t.name: t for t in self.mod.tensors(g)}
        self.parts: dict[str, list[tuple[str, int]]] = {}
        for t in self.store.tensors:
            ps = [gen[p] for p in generator_parts(t.name)]
            if any(p.kind != "s8" or p.cols != t.k for p in ps) or sum(p.rows for p in ps) != t.n:
                raise ValueError(f"{t.name} [{t.n}, {t.k}] is not the generator's s8 {generator_parts(t.name)} at {g.name}")
            self.parts[t.name] = [(p.name, p.nbytes) for p in ps]
        self.size = self.mod.stream_bytes(g)
        self.program = self.mod.program(g, seed)

    def tensor(self, name: str, a: int, b: int) -> bytes:
        """Bytes `[a, b)` of the layout's tensor `name`."""
        out, at = [], 0
        for part, n in self.parts[name]:
            lo, hi = max(a, at), min(b, at + n)
            if lo < hi:
                out.append(self.mod.tensor_bytes(self.shape, self.seed, part, lo - at, hi - at))
            at += n
        return b"".join(out)

    def read(self, lo: int, hi: int) -> bytes:
        return w_range(self.store, self.tensor, lo, hi)

    def pieces(self, manifest: dict) -> list[tuple[str, int, int, str]]:
        """Each of the generator's s8 tensors in `W`, in `W`'s order: its name, `W` offset, bytes and the SHA-256 that
        `manifest` (the generator's at this shape and seed) gives it. Every s8 tensor the manifest lists is one."""
        if (manifest.get("program"), manifest.get("stream_bytes")) != (self.program, self.size):
            raise ValueError("the manifest is another shape's or seed's: its program or stream_bytes differs")
        rows = {r["name"]: r for r in manifest["tensors"]}
        out = []
        for t in self.store.tensors:
            at = t.offset
            for name, n in self.parts[t.name]:
                r = rows[name]
                if (r["kind"], r["nbytes"]) != ("s8", n) or not r.get("sha256"):
                    raise ValueError(f"the manifest's {name} is not {n} s8 bytes with a sha256")
                out.append((name, at, n, r["sha256"]))
                at += n
        left = {r["name"] for r in manifest["tensors"] if r["kind"] == "s8"} - {p[0] for p in out}
        if left:
            raise ValueError(f"W leaves out the manifest's s8 tensors {sorted(left)}")
        return out


_READ: Callable[[int, int], bytes] | None = None
"""`commit_w`'s reader, which its forked workers inherit."""


def _read(job: tuple[int, int]) -> bytes:
    return _READ(*job)


def _sha256_pieces(q: queue.Queue, pieces: Sequence[tuple[str, int, int, str]], state: dict) -> None:
    """`pieces`' SHA-256 over the `(W offset, segment)` items `q` brings, in `W`'s order, until `None`: each piece whose
    last byte has passed is checked (a mismatch's name into `state["bad"]`) and counted (`state["done"]`); an error
    lands in `state["error"]` and the queue is still drained."""
    i, h = 0, hashlib.sha256()
    try:
        while (item := q.get()) is not None:
            lo, seg = item
            hi, view = lo + len(seg), memoryview(seg)
            while i < len(pieces) and pieces[i][1] < hi:
                name, off, size, want = pieces[i]
                h.update(view[max(lo, off) - lo:min(hi, off + size) - lo])
                if off + size > hi:
                    break
                if h.hexdigest() != want:
                    state["bad"].append(name)
                i, h = i + 1, hashlib.sha256()
        state["done"] = i
    except BaseException as e:  # noqa: BLE001
        state["error"] = e
        while q.get() is not None:
            pass


def commit_w(store: Store, read: Callable[[int, int], bytes], workers: int,
             pieces: Sequence[tuple[str, int, int, str]] = ()) -> bytes:
    """`setup.commit(store.params, W)`, `W`'s segments `read(lo, hi)` by `workers` forked processes (at most `workers + 4`
    held at once), and each of `pieces` (name, `W` offset, bytes, SHA-256 hex, in `W`'s order) held to its digest, hashed
    on a thread as the segments pass, beside the commitment's SHA-512 (hashlib drops the GIL)."""
    global _READ
    params, sb = store.params, store.params.segment_bytes
    jobs = iter([(s * sb, (s + 1) * sb) for s in range(params.segments)])
    _READ = read
    pool = mp.get_context("fork").Pool(workers) if workers > 1 else None
    q: queue.Queue = queue.Queue(maxsize=4)
    state: dict[str, Any] = {"bad": []}
    hasher = threading.Thread(target=_sha256_pieces, args=(q, pieces, state), daemon=True)
    hasher.start()

    def results() -> Iterator[bytes]:
        if pool is None:
            yield from map(_read, jobs)
            return
        first = [next(jobs) for _ in range(min(workers + 4, params.segments))]
        pending = collections.deque(pool.apply_async(_read, (j,)) for j in first)
        while pending:
            seg = pending.popleft().get()
            nxt = next(jobs, None)
            if nxt is not None:
                pending.append(pool.apply_async(_read, (nxt,)))
            yield seg

    def segments() -> Iterator[bytes]:
        for s, seg in enumerate(results()):
            q.put((s * sb, seg))
            if state["bad"] or "error" in state:
                break
            yield seg

    try:
        commitment = setup.commit(params, segments())
    except ValueError:
        if not (state["bad"] or "error" in state):
            raise
    finally:
        q.put(None)
        hasher.join()
        if pool is not None:
            pool.terminate()
            pool.join()
    if "error" in state:
        raise state["error"]
    if state["bad"]:
        raise ValueError(f"the generator's {state['bad']} are not the manifest's: their SHA-256 differs")
    if state["done"] != len(pieces):
        raise ValueError(f"W holds {state['done']} of the {len(pieces)} tensors to check")
    return commitment


def _bytes_arg(text: str) -> bytes:
    """``hex:...``, ``file:PATH`` (its bytes, a trailing newline stripped), ``hexfile:PATH`` or ``text:...``."""
    kind, _, v = text.partition(":")
    if kind == "hex":
        return bytes.fromhex(v)
    if kind == "file":
        return Path(v).read_bytes().rstrip(b"\n")
    if kind == "hexfile":
        return bytes.fromhex(Path(v).read_text().strip())
    if kind == "text":
        return v.encode()
    raise argparse.ArgumentTypeError("hex:..., hexfile:PATH, file:PATH or text:...")


def dump(path: Path, doc: Any) -> None:
    """`setup.json`'s form (p2-encode's): sorted keys, a one-space indent, replaced atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def commit_cmd(a: argparse.Namespace) -> dict:
    """`setup.json` with the commitment to `W` in this layout: p2-encode's `commit` fields for its stream, plus the
    layout's digest and the checked tensors; reused when `--store` already has it for the same `W`."""
    store_dir, out = Path(a.store), Path(a.out)
    w = Generated(a.generator, a.shape, a.seed)
    manifest = Path(a.manifest).read_bytes()
    pieces = w.pieces(json.loads(manifest))
    params = w.store.params
    doc = {"format": SETUP_FORMAT, "manifest": params.manifest(), "scheme": P2(params).name, "generator": a.generator,
           "generator_blob": git_blob(a.generator), "shape": a.shape, "program": w.program, "stream_bytes": w.size,
           "w_bytes": params.w_bytes, "c_bytes": params.blocks * params.block_bytes, "layout": SCHEMA,
           "layout_sha512": _sha(_dump(w.store.to_json()).encode()), "tensor_bytes": sum(t.size for t in w.store.tensors),
           "weights_manifest_sha256": hashlib.sha256(manifest).hexdigest(), "checked_tensors": len(pieces)}
    path = store_dir / "setup.json"
    if path.exists():
        old = json.loads(path.read_text())
        if {k: old.get(k) for k in doc if k != "generator"} != {k: v for k, v in doc.items() if k != "generator"}:
            raise SystemExit(f"{path} is another W's or another layout's setup: use a fresh --store")
        if old.get("commitment"):
            print(f"commit: reusing {path}'s commitment", file=sys.stderr)
            dump(out / "setup.json", old)
            return old
    t0 = time.perf_counter()
    doc["commitment"] = commit_w(w.store, w.read, a.workers, pieces).hex()
    doc["commit_s"] = time.perf_counter() - t0
    for d in (store_dir, out):
        dump(d / "setup.json", doc)
    print(json.dumps({"commitment": doc["commitment"], "segments": params.segments, "commit_s": doc["commit_s"]}))
    return doc


def _encode_segment(args: tuple[P2Params, bytes, int, bytes]) -> list[bytes]:
    params, salt, s, w_seg = args
    return encode_segment(params, None, salt, s, w_seg)


def _encode_block(args: tuple[P2Params, bytes, int, bytes]) -> bytes:
    """One block at any index, by the spec's reference encoder (two roots by Python's `pow`, seconds)."""
    params, prefix, i, m = args
    ref = _ref(params)
    k, t, _ = reference.keys_after(prefix, ref, *divmod(i, params.blocks_per_segment))
    return reference.encode(ref, int.from_bytes(m, "big"), k, t).to_bytes(params.block_bytes, "big")


def _sha(data: bytes) -> str:
    return hashlib.sha512(data).hexdigest()


def _dump(doc: dict) -> str:
    """JSON with an object's keys one a line and a table's rows one a line, each row compact."""
    def fmt(v, ind: int) -> str:
        pad = " " * ind
        if isinstance(v, dict) and v:
            return "{\n" + ",\n".join(f"{pad} {json.dumps(k)}: {fmt(x, ind + 1)}" for k, x in v.items()) + f"\n{pad}}}"
        if isinstance(v, list) and v and all(isinstance(x, (dict, list)) for x in v):
            return "[\n" + ",\n".join(f"{pad} {fmt(x, ind + 1) if isinstance(x, dict) else json.dumps(x, ensure_ascii=False)}"
                                     for x in v) + f"\n{pad}]"
        return json.dumps(v, ensure_ascii=False)
    return fmt(doc, 0) + "\n"


def toy(shape: Shape, blocks_per_segment: int, workers: int | None = None) -> dict[str, bytes]:
    """The toy fixture's files: `toy.json` (the store at `shape`, its salt, root, digests and negative blocks),
    `spot.json` and `spot-c.bin` (blocks at the real store's indices under the toy's salt), `c-star.bin`, and
    `toy_plain`'s two; encoded by the registered codec in a process pool, and every file checked against the reference
    decode before it is written."""
    store = build(shape, blocks_per_segment)
    params = store.params
    plain = toy_plain(store)
    w, staging = plain["w.bin"], plain["staging.bin"]
    commitment = setup.commit(params, w)
    salt = setup.draw_salt(TOY_SOURCE, commitment)
    prefix = expand_prefix(params, salt)
    real = build(MQA16K).params
    real_prefix = expand_prefix(real, salt)
    spots = _spots(build(MQA16K), salt)
    seg, pb = params.segment_bytes, params.payload_bytes
    padded = store.layer(0)[0]
    last = padded.first_block + padded.blocks - 1
    bad_payload = bytearray(w[pb * last:pb * (last + 1)])
    bad_payload[-1] ^= 1
    with ProcessPoolExecutor(workers) as pool:
        segments = pool.map(_encode_segment, [(params, salt, s, w[s * seg:(s + 1) * seg]) for s in range(params.segments)])
        blocks = pool.map(_encode_block, [(real, real_prefix, sp["i"], spot_payload(sp["i"], sp["used"], pb)) for sp in spots]
                          + [(params, prefix, last, bytes(bad_payload))])
        segments, blocks = list(segments), list(blocks)
    c_star = b"".join(b for blocks_s in segments for b in blocks_s)
    for sp, c in zip(spots, blocks):
        if _sha(decode_block(real, real_prefix, sp["i"], c)) != sp["m_sha512"]:
            raise RuntimeError(f"spot vector {sp['id']!r} doesn't decode back")
    spot_c = b"".join(blocks[:len(spots)])
    vk = setup.verifier_key_from_encoding(params, None, commitment, salt, segments)
    layers, start = [], 0
    for l in sorted({t.layer for t in store.tensors}):
        buf = decode_layer(store, c_star, salt, l)
        if buf != staging[start:start + len(buf)]:
            raise RuntimeError(f"layer {l} doesn't decode to its tensors")
        layers.append([l, start, len(buf), _sha(buf)])
        start += len(buf)
    negatives = [{"id": "padding", "layer": 0, "block": last, "flag": 3, "c": blocks[-1].hex(),
                  "why": "the true payload with its last byte, past the tensor's end, xor 1"},
                 {"id": "not-below-p", "layer": 0, "block": 0, "flag": 2, "c": "p", "why": "c = p"},
                 {"id": "flipped", "layer": 0, "block": 8, "flag": 2, "xor": [1000, 1],
                  "why": "C*'s block 8 with byte 1000 xor 1: decodes to more than 16,432 bits"}]
    for neg in negatives:
        if refusal(store, negative(neg, c_star, params), salt, neg["layer"]) != neg["flag"]:
            raise RuntimeError(f"negative block {neg['id']!r} doesn't raise flag bit {neg['flag']}")
    doc = {"schema": TOY_SCHEMA, "layout": store.to_json(), "source": TOY_SOURCE.hex(), "commitment": commitment.hex(),
           "salt": salt.hex(), "prefix": prefix.hex(), "root": vk.root.hex(), "w_sha512": _sha(w),
           "c_star_sha512": _sha(c_star), "staging_sha512": _sha(staging), "staging_columns": ["layer", "start", "bytes", "sha512"],
           "staging": layers, "negatives": negatives}
    spot = {"schema": SPOT_SCHEMA, "params": real.manifest(), "salt": salt.hex(), "prefix": real_prefix.hex(),
            "c": "spot-c.bin: vector j's block at byte 2056 j", "c_sha512": _sha(spot_c), "vectors": spots}
    return {"toy.json": _dump(doc).encode(), "spot.json": _dump(spot).encode(), "spot-c.bin": spot_c,
            "c-star.bin": c_star} | plain


def _spots(real: Store, salt: bytes) -> list[dict]:
    """Blocks of the real store under the toy's salt: W's first, layer 0's qkv's last (268 bytes used), layer 51's
    down_proj's first, the head's last (36 used) and W's last (all padding, at v = 65,535)."""
    params = real.params
    prefix, ref, pb = expand_prefix(params, salt), _ref(params), params.payload_bytes
    qkv, down, head = real.layer(0)[0], real.layer(real.shape.layers - 1)[3], real.layer(real.shape.layers)[0]
    picks = [("w-first", qkv, qkv.first_block), ("qkv0-last", qkv, qkv.first_block + qkv.blocks - 1),
             ("down51-first", down, down.first_block), ("head-last", head, head.first_block + head.blocks - 1),
             ("w-last", None, params.blocks - 1)]
    out, nb = [], params.block_bytes
    for name, t, i in picks:
        used = 0 if t is None else min(pb, t.offset + t.size - pb * i)
        s, v = divmod(i, params.blocks_per_segment)
        k, tw, attempt = reference.keys_after(prefix, ref, s, v)
        out.append({"id": name, "i": i, "s": s, "v": v, "tensor": None if t is None else t.name, "used": used,
                    "K_sha512": _sha(k.to_bytes(nb, "big")), "t_sha512": _sha(tw.to_bytes(nb, "big")),
                    "t_attempt": attempt, "m_sha512": _sha(spot_payload(i, used, pb))})
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lay = sub.add_parser("layout", help="the real store's layout JSON")
    lay.add_argument("--out", type=Path, default=LAYOUT_JSON)
    tp = sub.add_parser("toy", help="the toy fixture: W, salt, C*, staging, spot vectors")
    tp.add_argument("--out", type=Path, default=TOY_DIR)
    tp.add_argument("--plain", action="store_true", help="only w.bin and staging.bin, for the tracked toy.json; no encoding")
    tp.add_argument("--width", type=int, default=64)
    tp.add_argument("--layers", type=int, default=2)
    tp.add_argument("--ff", type=int, default=64)
    tp.add_argument("--per-segment", type=int, default=11)
    tp.add_argument("--workers", type=int, default=os.cpu_count())
    cp = sub.add_parser("commit", help="setup.commit of the real W from the seeded generator, into setup.json")
    cp.add_argument("--store", required=True, help="the store's directory (scratch, outside the run directory)")
    cp.add_argument("--out", default=os.environ.get("OUT", "."), help="the public outputs' directory ($OUT)")
    cp.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="processes generating W")
    cp.add_argument("--generator", default=GENERATOR, help="the weights generator: a module name or its .py file")
    cp.add_argument("--seed", type=_bytes_arg, required=True, help="hex:..., hexfile:PATH, file:PATH or text:...")
    cp.add_argument("--shape", default="h200")
    cp.add_argument("--manifest", required=True, help="the generator's manifest with every tensor's sha256")
    args = ap.parse_args(argv)
    if args.cmd == "commit":
        commit_cmd(args)
        return 0
    if args.cmd == "layout":
        args.out.write_text(_dump(build(MQA16K).to_json()))
        print(args.out)
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    if args.plain:
        files = toy_plain(toy_store())
    else:
        files = toy(toy_shape(args.width, args.layers, args.ff), args.per_segment, args.workers)
    for name, data in files.items():
        (args.out / name).write_bytes(data)
        print(args.out / name, len(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
