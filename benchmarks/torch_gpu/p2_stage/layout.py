"""Where a layer's matrices lie in the P2-encoded weights store, where their plaintext comes from, and P2's decode of one
block on the CPU (the check's reference). No torch.

The store's layout is CMP-95 item 2's fixture, memory-accounting's (`benchmarks/pous/p2_v1/store.py`, schema
`verity-pous/p2-store-layout/v1`): `store_layout` reads its JSON, the real store's (`store.py layout`, pinned by its
`LAYOUT_SHA512`) or the toy's (`store-toy/toy.json`'s `layout`), with the key prefix:

- the registered codec `p2-16448/v4` (`verity/core/protocols/pous/schemes/p2.py`): 2,056-byte big-endian blocks, block
  i = blocks_per_segment · s + v being block v of segment s;
- block i's payload is W bytes [2054 i, 2054 i + 2054), bytes 2 .. 2055 of its decoded row, whose bytes 0 and 1 are zero;
- each tensor (s8 [N, K] row-major) starts on a block, `first_block`, takes ⌈N K / 2054⌉ blocks, and its last block's
  payload is zero past the tensor; in its layer's staging buffer it starts at `staging_offset`;
- K and t from counter-mode SHA-512 after `schemes.p2.expand_prefix` of the protocol's 24-byte salt (`registered_prefix`).

A layer's matrices (`MATRICES`) are its four tensors: QKV (the query heads, then K, then V: 16,512 x 16,384 at the frozen
shape), O, up and down. Their scales are not in W.

`Weights` is where the plaintext comes from: the seeded weights stream (`verity_torch_loop.weights manifest`), every tensor
at its stream offset, QKV being wq, wk and wv one after another, each with its f32 scales. `payloads` lays it out as the
store's W.

`decode_block` is the reference's decode of one block (`p2_v1.reference.decode` under `reference.keys_after`), refusing
what `store.decode_block` refuses: a row not below p, or a decode wider than the payload.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from verity.core.protocols.pous.schemes import p2 as S

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "pous"))
from p2_v1 import reference as R  # noqa: E402

PARAMS = S.P2Params(*S.POINT)
SPEC = R.Params(PARAMS.w, PARAMS.c0, PARAMS.payload_bytes, PARAMS.blocks_per_segment)
ROW = PARAMS.block_bytes          # 2,056
PAYLOAD = PARAMS.payload_bytes    # 2,054
LEAD = ROW - PAYLOAD              # a decoded row's zero bytes before its payload
SCHEMA = "verity-pous/p2-store-layout/v1"
SCHEME = "p2-16448/v4"
COLUMNS = ("name", "layer", "n", "k", "offset", "size", "first_block", "blocks", "staging_offset")

MATRICES = {"qkv": "self_attn.qkv_proj", "o": "self_attn.o_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
PARTS = {"qkv": ("wq", "wk", "wv"), "o": ("wo",), "up": ("wup",), "down": ("wdown",)}

Read = Callable[[int, int], bytes]


def registered_prefix(salt: bytes) -> bytes:
    """The registered scheme's key prefix for the protocol's 24-byte `salt`: whole 128-byte blocks, one midstate."""
    return S.expand_prefix(PARAMS, salt)


def matrix_name(layer: int, matrix: str) -> str:
    return f"model.layers.{layer}.{MATRICES[matrix]}.weight"


def blocks_of(offset: int, size: int) -> range:
    """The blocks holding W bytes [offset, offset + size)."""
    if offset < 0 or size < 0:
        raise ValueError("a byte range has a nonnegative offset and size")
    return range(offset // PAYLOAD, -(-(offset + size) // PAYLOAD)) if size else range(0)


@dataclass(frozen=True)
class Tensor:
    """One row of the fixture's tensor table (`COLUMNS`)."""

    name: str
    layer: int
    n: int
    k: int
    offset: int
    size: int
    first_block: int
    blocks: int
    staging_offset: int

    @property
    def block_range(self) -> range:
        return range(self.first_block, self.first_block + self.blocks)

    @property
    def matrix(self) -> str | None:
        """Its key in `MATRICES`, or None (the head)."""
        return next((m for m in MATRICES if self.name == matrix_name(self.layer, m)), None)


@dataclass(frozen=True)
class StoreLayout:
    """The store's P2 geometry (`params`), its tensors in block order, and the prefix every block's keys follow."""

    params: S.P2Params
    tensors: tuple[Tensor, ...]
    prefix: bytes

    def __post_init__(self) -> None:
        p = self.params
        if (p.w, p.c0) != S.POINT[:2] or p.block_bytes != ROW:
            raise ValueError(f"the store is P2 at w = {S.POINT[0]}, c0 = {S.POINT[1]}, the point libp2v1.so is built for")
        if not self.prefix or len(self.prefix) % 128:
            raise ValueError("the key prefix is whole 128-byte SHA-512 blocks (p2v1_keys compresses it to one midstate)")
        at, staged = 0, {}
        for t in self.tensors:
            if t.size != t.n * t.k or t.offset != t.first_block * PAYLOAD or t.blocks != -(-t.size // PAYLOAD):
                raise ValueError(f"{t.name} is not N K bytes from a block boundary over ⌈N K / {PAYLOAD}⌉ blocks")
            if t.first_block < at:
                raise ValueError(f"{t.name} shares a block with the tensor before it")
            at = t.first_block + t.blocks
            if t.staging_offset < staged.get(t.layer, 0):
                raise ValueError(f"{t.name} overlaps the tensor before it in layer {t.layer}'s staging buffer")
            staged[t.layer] = t.staging_offset + t.size
        if at > p.blocks:
            raise ValueError(f"the tensors run to block {at}, past the store's {p.blocks}")

    @property
    def spec(self) -> R.Params:
        p = self.params
        return R.Params(p.w, p.c0, p.payload_bytes, p.blocks_per_segment, p.segments)

    def layer(self, layer: int) -> tuple[Tensor, ...]:
        out = tuple(t for t in self.tensors if t.layer == layer)
        if not out:
            raise KeyError(f"no layer {layer} in the store")
        return out

    def tensor(self, layer: int, name: str) -> Tensor:
        """Layer `layer`'s tensor `name`: a key of `MATRICES` or the fixture's name."""
        full = matrix_name(layer, name) if name in MATRICES else name
        for t in self.layer(layer):
            if t.name == full:
                return t
        raise KeyError(f"no tensor {full!r} in layer {layer} of the store")

    def blocks(self, layer: int) -> range:
        ts = self.layer(layer)
        return range(ts[0].first_block, ts[-1].first_block + ts[-1].blocks)

    def staging_bytes(self, layer: int) -> int:
        return max(t.staging_offset + t.size for t in self.layer(layer))

    def manifest(self) -> dict:
        return {"params": self.params.manifest(), "tensor_columns": list(COLUMNS),
                "tensors": [[getattr(t, c) for c in COLUMNS] for t in self.tensors], "prefix": self.prefix.hex()}


def store_layout(doc: dict, prefix: bytes) -> StoreLayout:
    """The layout of the fixture's JSON `doc` (the layout itself, or the toy's `toy.json`, which holds it as `layout`)."""
    doc = doc.get("layout", doc)
    if doc.get("schema") != SCHEMA or doc.get("scheme") != SCHEME or tuple(doc.get("tensor_columns", ())) != COLUMNS:
        raise ValueError(f"not a {SCHEMA} layout of {SCHEME} with columns {COLUMNS}")
    q = doc["params"]
    params = S.P2Params(q["w"], q["c0"], q["blocks_per_segment"], q["segments"])
    if params.manifest() != q or (doc["block_bytes"], doc["payload_bytes"], doc["blocks"]) != (
            params.block_bytes, params.payload_bytes, params.blocks):
        raise ValueError(f"the layout's params {q} are not the registered codec's")
    return StoreLayout(params, tuple(Tensor(*row) for row in doc["tensors"]), prefix)


@dataclass(frozen=True)
class Weights:
    """The weights stream's tensors (`verity_torch_loop.weights manifest`: rows with ``name``, ``offset``, ``nbytes`` and
    ``shape``); layer l's tensor t is ``l{l}.{t}``, its f32 scales ``l{l}.{t}.scale``."""

    rows: dict[str, dict]

    @classmethod
    def of(cls, manifest: dict) -> Weights:
        return cls({t["name"]: t for t in manifest["tensors"]})

    def parts(self, layer: int, matrix: str, scales: bool = False) -> list[tuple[int, int]]:
        """`(offset, nbytes)` in the stream of `matrix`'s tensors (or their scales), in the order its rows are stacked."""
        if matrix not in PARTS:
            raise KeyError(f"a matrix is one of {sorted(PARTS)}, not {matrix!r}")
        rows = [self.rows[f"l{layer}.{p}{'.scale' if scales else ''}"] for p in PARTS[matrix]]
        return [(r["offset"], r["nbytes"]) for r in rows]

    def shape(self, layer: int, matrix: str) -> tuple[int, int]:
        """`(N, K)`: its tensors' rows stacked, their common row length."""
        ss = [self.rows[f"l{layer}.{p}"]["shape"] for p in PARTS[matrix]]
        if len({s[1] for s in ss}) != 1:
            raise ValueError(f"{matrix}'s tensors {PARTS[matrix]} have different row lengths")
        return sum(s[0] for s in ss), ss[0][1]

    def span(self, layer: int) -> tuple[int, int]:
        """The stream bytes [lo, hi) holding the layer's matrices and their scales."""
        ps = [p for m in PARTS for s in (False, True) for p in self.parts(layer, m, s)]
        return min(o for o, _ in ps), max(o + n for o, n in ps)

    @staticmethod
    def read(parts: list[tuple[int, int]], lo: int, hi: int, read: Read) -> bytes:
        """Bytes [lo, hi) of `parts` one after another, each read from the stream by `read(offset, size)`."""
        out, at = bytearray(), 0
        for off, n in parts:
            a, b = max(lo, at), min(hi, at + n)
            if a < b:
                out += read(off + a - at, b - a)
            at += n
        if len(out) != hi - lo:
            raise ValueError(f"bytes [{lo}, {hi}) run past the {at} the parts hold")
        return bytes(out)


def payloads(lay: StoreLayout, src: Weights, read: Read, j0: int, n: int) -> bytes:
    """The payloads of blocks `j0 .. j0 + n - 1` (2,054 bytes each): the store's W, each tensor's bytes from the stream,
    zero past every tensor's end."""
    lo, hi = j0 * PAYLOAD, (j0 + n) * PAYLOAD
    out = bytearray(hi - lo)
    for t in lay.tensors:
        a, b = max(lo, t.offset), min(hi, t.offset + t.size)
        if a >= b:
            continue
        if t.matrix is None:
            raise KeyError(f"{t.name} has no tensor in the weights stream")
        if src.shape(t.layer, t.matrix) != (t.n, t.k):
            raise ValueError(f"{t.name} is [{t.n}, {t.k}] in the store, {list(src.shape(t.layer, t.matrix))} in the stream")
        out[a - lo:b - lo] = src.read(src.parts(t.layer, t.matrix), a - t.offset, b - t.offset, read)
    return bytes(out)


def decode_block(spec: R.Params, prefix: bytes, j: int, c: bytes) -> bytes:
    """Block `j`'s payload from its 2,056-byte row `c`, by the P2 spec's reference; `ValueError` for a row that is not a
    codeword."""
    if len(c) != ROW:
        raise ValueError(f"block {j}: a row is {ROW} bytes, not {len(c)}")
    v = int.from_bytes(c, "big")
    if v >= spec.p:
        raise ValueError(f"block {j} is not below p")
    k, t, _ = R.keys_after(prefix, spec, *divmod(j, spec.blocks_per_segment))
    m = R.decode(spec, v, k, t)
    if m >> (8 * spec.payload_bytes):
        raise ValueError(f"block {j} decodes to more than {8 * spec.payload_bytes} bits: not a codeword")
    return m.to_bytes(spec.payload_bytes, "big")


def check_blocks(spec: R.Params, prefix: bytes, first: int, rows: bytes | memoryview,
                 payloads: bytes | memoryview) -> list[tuple[int, str]]:
    """The blocks `first ..` of `rows` (2,056 bytes each) whose reference decode is refused or differs from their 2,054
    bytes of `payloads`, with why."""
    n = len(rows) // ROW
    if len(rows) != n * ROW or len(payloads) != n * PAYLOAD:
        raise ValueError(f"{len(rows)} bytes of rows and {len(payloads)} of payload are not the same whole blocks")
    bad = []
    for i in range(n):
        try:
            got = decode_block(spec, prefix, first + i, bytes(rows[i * ROW:(i + 1) * ROW]))
        except ValueError as e:
            bad.append((first + i, str(e)))
            continue
        if got != bytes(payloads[i * PAYLOAD:(i + 1) * PAYLOAD]):
            bad.append((first + i, "decodes to other bytes than W's"))
    return bad
