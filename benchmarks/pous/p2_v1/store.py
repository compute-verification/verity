"""The encoded weights store (CMP-95 item 2): the frozen MQA model's INT8 weights as `W` under the registered P2
(`p2-16448/v4`), laid out so that one layer, or one matrix, decodes alone into a staging buffer, and the reference
decode a kernel's `decode_layer` matches byte for byte.

    store.py layout [--out store-layout.json]
    store.py toy [--out store-toy] [--width 64] [--layers 2] [--per-segment 12] [--workers N]

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
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from p2_v1 import reference
else:
    from . import reference

from verity.core.protocols.pous import setup
from verity.core.protocols.pous.layout import Layout
from verity.core.protocols.pous.schemes.p2 import POINT, P2Params, encode_segment, expand_prefix, stream

HERE = Path(__file__).resolve().parent
LAYOUT_JSON = HERE / "store-layout.json"
TOY_DIR = HERE / "store-toy"
SCHEMA = "verity-pous/p2-store-layout/v1"
TOY_SCHEMA = "verity-pous/p2-store-toy/v1"
SPOT_SCHEMA = "verity-pous/p2-store-spot/v1"
W_BITS, C0, PER_SEGMENT = POINT
STAGING_ALIGN = 256
"""Every tensor's staging offset is a multiple of this (TMA wants 16 bytes; a 2,054-multiple would give 2)."""
TOY_SOURCE = hashlib.sha512(b"verity-pous/p2-store/toy/setup-source").digest()
TOY_W = b"verity-pous/p2-store/toy/w/v1\0"
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


def toy_shape(width: int = 64, layers: int = 2) -> Shape:
    return Shape(f"toy-mqa-d{width}", width, layers, width // 64, 1, 64, 4 * width, 256)


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


def toy_tensors(store: Store) -> dict[str, bytes]:
    """The toy's weights: each tensor's bytes from a SHA-512 counter stream under its name."""
    return {t.name: stream(TOY_W, t.name.encode(), t.size) for t in store.tensors}


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
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def toy(shape: Shape, blocks_per_segment: int, workers: int | None = None) -> dict[str, bytes]:
    """The toy fixture's files: the store at `shape`, its `W`, salt, `C*`, every layer's staging buffer and the spot
    vectors at the real store's indices, plus negative blocks; encoded by the registered codec in a process pool."""
    store = build(shape, blocks_per_segment)
    params = store.params
    w = store.layout.pack(toy_tensors(store))
    commitment = setup.commit(params, w)
    salt = setup.draw_salt(TOY_SOURCE, commitment)
    prefix = expand_prefix(params, salt)
    real = build(MQA16K).params
    real_prefix = expand_prefix(real, salt)
    spots = _spots(build(MQA16K), salt)
    seg = params.segment_bytes
    padded = store.layer(0)[0]
    last = padded.first_block + padded.blocks - 1
    bad_payload = bytearray(w[params.payload_bytes * last:params.payload_bytes * (last + 1)])
    bad_payload[-1] ^= 1
    with ProcessPoolExecutor(workers) as pool:
        segments = pool.map(_encode_segment, [(params, salt, s, w[s * seg:(s + 1) * seg]) for s in range(params.segments)])
        blocks = pool.map(_encode_block, [(real, real_prefix, sp["i"], bytes.fromhex(sp["m"])) for sp in spots]
                          + [(params, prefix, last, bytes(bad_payload))])
        segments, blocks = list(segments), list(blocks)
    c_star = b"".join(b for blocks_s in segments for b in blocks_s)
    for sp, c in zip(spots, blocks):
        if decode_block(real, real_prefix, sp["i"], c) != bytes.fromhex(sp["m"]):
            raise RuntimeError(f"spot vector {sp['id']!r} doesn't decode back")
        sp["c"] = c.hex()
    vk = setup.verifier_key_from_encoding(params, None, commitment, salt, segments)
    staging, layers = b"", []
    for l in sorted({t.layer for t in store.tensors}):
        buf = decode_layer(store, c_star, salt, l)
        layers.append([l, len(staging), len(buf), _sha(buf)])
        staging += buf
    nb = params.block_bytes
    flipped = bytearray(c_star[nb * 8:nb * 9])
    flipped[1000] ^= 1
    negatives = [{"id": "padding", "layer": 0, "block": last, "flag": 3, "c": blocks[-1].hex(),
                  "why": "the true payload with its last byte, past the tensor's end, xor 1"},
                 {"id": "not-below-p", "layer": 0, "block": 0, "flag": 2,
                  "c": reference.DEPLOYMENT.p.to_bytes(nb, "big").hex(), "why": "c = p"},
                 {"id": "flipped", "layer": 0, "block": 8, "flag": 2, "c": bytes(flipped).hex(),
                  "why": "C*'s block 8 with byte 1000 xor 1: decodes to more than 16,432 bits"}]
    for neg in negatives:
        if refusal(store, patch(c_star, neg["block"], bytes.fromhex(neg["c"])), salt, neg["layer"]) != neg["flag"]:
            raise RuntimeError(f"negative block {neg['id']!r} doesn't raise flag bit {neg['flag']}")
    doc = {"schema": TOY_SCHEMA, "layout": store.to_json(), "source": TOY_SOURCE.hex(), "commitment": commitment.hex(),
           "salt": salt.hex(), "prefix": prefix.hex(), "root": vk.root.hex(), "w_sha512": _sha(w),
           "c_star_sha512": _sha(c_star), "staging_sha512": _sha(staging), "staging_columns": ["layer", "start", "bytes", "sha512"],
           "staging": layers, "negatives": negatives}
    spot = {"schema": SPOT_SCHEMA, "params": real.manifest(), "salt": salt.hex(), "prefix": real_prefix.hex(),
            "vectors": spots}
    return {"toy.json": _dump(doc).encode(), "spot.json": _dump(spot).encode(), "w.bin": w, "c-star.bin": c_star,
            "staging.bin": staging}


def _spots(real: Store, salt: bytes) -> list[dict]:
    """Blocks of the real store under the toy's salt: W's first, layer 0's qkv's last (268 bytes used), layer 51's
    down_proj's first, the head's last (36 used) and W's last (all padding, at v = 65,535)."""
    params = real.params
    prefix, ref, pb = expand_prefix(params, salt), _ref(params), params.payload_bytes
    qkv, down, head = real.layer(0)[0], real.layer(real.shape.layers - 1)[3], real.layer(real.shape.layers)[0]
    picks = [("w-first", qkv, qkv.first_block), ("qkv0-last", qkv, qkv.first_block + qkv.blocks - 1),
             ("down51-first", down, down.first_block), ("head-last", head, head.first_block + head.blocks - 1),
             ("w-last", None, params.blocks - 1)]
    out = []
    for name, t, i in picks:
        used = 0 if t is None else min(pb, t.offset + t.size - pb * i)
        s, v = divmod(i, params.blocks_per_segment)
        k, tw, attempt = reference.keys_after(prefix, ref, s, v)
        m = stream(TOY_W + b"spot", i.to_bytes(8, "big"), used) + bytes(pb - used)
        out.append({"id": name, "i": i, "s": s, "v": v, "tensor": None if t is None else t.name, "used": used,
                    "K": k.to_bytes(params.block_bytes, "big").hex(), "t": tw.to_bytes(params.block_bytes, "big").hex(),
                    "t_attempt": attempt, "m": m.hex()})
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lay = sub.add_parser("layout", help="the real store's layout JSON")
    lay.add_argument("--out", type=Path, default=LAYOUT_JSON)
    tp = sub.add_parser("toy", help="the toy fixture: W, salt, C*, staging, spot vectors")
    tp.add_argument("--out", type=Path, default=TOY_DIR)
    tp.add_argument("--width", type=int, default=64)
    tp.add_argument("--layers", type=int, default=2)
    tp.add_argument("--per-segment", type=int, default=12)
    tp.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args(argv)
    if args.cmd == "layout":
        args.out.write_text(_dump(build(MQA16K).to_json()))
        print(args.out)
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in toy(toy_shape(args.width, args.layers), args.per_segment, args.workers).items():
        (args.out / name).write_bytes(data)
        print(args.out / name, len(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
