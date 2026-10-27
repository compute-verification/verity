"""Where a layer's matrices lie in the P2-encoded weights store, where their plaintext comes from, and P2's decode of one
block on the CPU (the check's reference). No torch.

The store's layout is p2_v1's fixture (`benchmarks/pous/p2_v1/store.py`, schema
`verity-pous/p2-store-layout/v1`): `store_layout` reads its JSON, the real store's (`store.py layout`, pinned by its
`LAYOUT_SHA512`) or the toy's (`store-toy/toy.json`'s `layout`), with the key prefix:

- the registered codec `p2-16448/v4` (`verity/core/protocols/pous/schemes/p2.py`): 2,056-byte big-endian blocks, block
  i = blocks_per_segment · s + v being block v of segment s;
- block i's payload is W bytes [2054 i, 2054 i + 2054), bytes 2 .. 2055 of its decoded row, whose bytes 0 and 1 are zero;
- each tensor (s8 [N, K] row-major) starts on a block, `first_block`, takes ⌈N K / 2054⌉ blocks, and its last block's
  payload is zero past the tensor; in its layer's staging buffer it starts at `staging_offset`;
- K and t from counter-mode SHA-512 after `schemes.p2.expand_prefix` of the protocol's 24-byte salt (`registered_prefix`).

A layer's matrices (`MATRICES`) are its four tensors: QKV (the query heads, then K, then V: 16,512 x 16,384 at the frozen
shape), O, up and down; after the last layer the head (`HEAD`, 256 x 16,384) is layer `layers`. Their scales are not in W.

`Weights` is where the plaintext comes from: the seeded weights stream (`e2e_h200.served_weights manifest`), every tensor
at its stream offset, QKV being wq, wk and wv one after another, each with its f32 scales, and the head the stream's
`head` (`Tensor.stream_names`). `payloads` lays it out as the store's W.

`check_blocks` holds rows to W's payloads by memory-accounting's reference decode of one block (`p2_v1/store.py`'s
`decode_block`: `p2_v1.reference.decode` under `reference.keys_after`), which refuses a row not below p, or a decode
wider than the payload.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
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
#: the store's last tensor, after the layers (as layer `layers`), and its tensor in the weights stream
HEAD, STREAM_HEAD = "lm_head.weight", "head"

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

    @property
    def stream_names(self) -> tuple[str, ...]:
        """The weights stream's tensors it stacks, in order: a matrix's `PARTS` of its layer, or the head."""
        if self.matrix is not None:
            return tuple(f"l{self.layer}.{p}" for p in PARTS[self.matrix])
        if self.name == HEAD:
            return (STREAM_HEAD,)
        raise KeyError(f"{self.name} has no tensor in the weights stream")


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
    """The weights stream's tensors (`e2e_h200.served_weights manifest`: rows with ``name``, ``offset``, ``nbytes`` and
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

    def tensor_parts(self, t: Tensor) -> list[tuple[int, int]]:
        """`(offset, nbytes)` in the stream of store tensor `t`'s stream tensors (`Tensor.stream_names`), stacked; refused
        unless they stack to `t`'s [N, K]."""
        rows = [self.rows[n] for n in t.stream_names]
        if len({r["shape"][1] for r in rows}) != 1 or (sum(r["shape"][0] for r in rows), rows[0]["shape"][1]) != (t.n, t.k):
            raise ValueError(f"{t.name} is [{t.n}, {t.k}] in the store, {[r['shape'] for r in rows]} in the stream")
        return [(r["offset"], r["nbytes"]) for r in rows]

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


def reader(path: Path) -> Read:
    """The stream's bytes from a file holding them at their stream offsets (`weights write`)."""
    def read(offset: int, size: int) -> bytes:
        with open(path, "rb") as f:
            data = os.pread(f.fileno(), size, offset)
        if len(data) != size:
            raise ValueError(f"{path} ends before byte {offset + size}")
        return data
    return read


def seed_bytes(text: str) -> bytes:
    """The generator's ``--seed`` form: ``text:...`` or ``hex:...``."""
    kind, _, v = text.partition(":")
    if kind == "text":
        return v.encode()
    if kind == "hex":
        return bytes.fromhex(v)
    raise ValueError(f"a seed is text:... or hex:..., not {text!r}")


def generator_reader(module: str, shape: str, seed: str) -> Read:
    """The stream regenerated by the seeded generator (`e2e_h200.served_weights.stream`), when no stream file is at
    hand."""
    w = importlib.import_module(module)
    sh, sd = w.ALIASES.get(shape) or w.SHAPES[shape], seed_bytes(seed)

    def read(offset: int, size: int) -> bytes:
        data = w.stream(sh, sd, offset, offset + size)
        if len(data) != size:
            raise ValueError(f"the stream ends before byte {offset + size}")
        return data
    return read


def payloads(lay: StoreLayout, src: Weights, read: Read, j0: int, n: int) -> bytes:
    """The payloads of blocks `j0 .. j0 + n - 1` (2,054 bytes each): the store's W, each tensor's bytes from the stream,
    zero past every tensor's end."""
    lo, hi = j0 * PAYLOAD, (j0 + n) * PAYLOAD
    out = bytearray(hi - lo)
    for t in lay.tensors:
        a, b = max(lo, t.offset), min(hi, t.offset + t.size)
        if a >= b:
            continue
        out[a - lo:b - lo] = src.read(src.tensor_parts(t), a - t.offset, b - t.offset, read)
    return bytes(out)


def store_salt(setup: dict, layout_doc: dict, layout_sha512: str, source: bytes, commitment: bytes) -> bytes:
    """The salt an encoded store is under (`p2_v1/store.py encode`'s directory, whose `setup.json` is `setup`):
    `setup.draw_salt(source, commitment)`, the auditor's coin `source` drawn after the store's commitment. Refused unless
    the store commits to `commitment` in this layout (`layout_sha512`, its JSON's SHA-512) under its parameters, and
    under this coin and salt where `setup.json` records them."""
    from verity.core.protocols.pous import setup as P

    salt = P.draw_salt(source, commitment)
    want = {"commitment": commitment.hex(), "layout_sha512": layout_sha512, "manifest": layout_doc.get("params"),
            "source": source.hex(), "salt": salt.hex()}
    bad = sorted(k for k, v in want.items() if (k in setup or k not in ("source", "salt")) and setup.get(k) != v)
    if bad:
        raise ValueError(f"the store's setup.json is not this layout's, coin's and commitment's: {bad} differ")
    return salt


def open_store(layout_path: Path, manifest: dict, encoded: Path | None, source: str | None, commitment: str | None,
               salt: str | None) -> tuple[StoreLayout, dict | None]:
    """The layout at `layout_path` under its store's prefix, and the store's `setup.json`: an encoded store's
    (`encoded`, under `store_salt` of the hex `source` and `commitment`, holding the weights manifest's program), or a
    development store's under the public test salt `salt` (hex), with no setup."""
    doc = json.loads(layout_path.read_text())
    setup = None
    if encoded is not None:
        if not (source and commitment):
            raise ValueError("an encoded store needs its coin and commitment")
        setup = json.loads((encoded / "setup.json").read_text())
        if "program" in manifest and setup.get("program") != manifest["program"]:
            raise ValueError(f"the store holds {setup.get('program')}, the weights manifest is {manifest['program']}")
        key = store_salt(setup, doc, hashlib.sha512(layout_path.read_bytes()).hexdigest(), bytes.fromhex(source),
                         bytes.fromhex(commitment))
    elif salt:
        key = bytes.fromhex(salt)
    else:
        raise ValueError("a development store needs its test salt")
    return store_layout(doc, registered_prefix(key)), setup


def segment_path(root: Path, s: int, ext: str) -> Path:
    """Segment `s`'s `c` (its blocks' rows in order) or `json` (written last: its digests) in an encoded store."""
    return root / "seg" / f"{s:05d}.{ext}"


def segment_rows(root: Path, spec: R.Params, j0: int, n: int) -> bytes:
    """The rows of blocks `j0 .. j0 + n - 1` (2,056 bytes each) from an encoded store's segment files."""
    out, bps = bytearray(), spec.blocks_per_segment
    j, stop = j0, j0 + n
    while j < stop:
        s, v = divmod(j, bps)
        k = min(stop - j, bps - v)
        with open(segment_path(root, s, "c"), "rb") as f:
            data = os.pread(f.fileno(), k * ROW, v * ROW)
        if len(data) != k * ROW:
            raise ValueError(f"segment {s} ends before block {j + k}")
        out += data
        j += k
    return bytes(out)


def check_blocks(params: S.P2Params, prefix: bytes, first: int, rows: bytes | memoryview,
                 payloads: bytes | memoryview) -> list[tuple[int, str]]:
    """The blocks `first ..` of `rows` (2,056 bytes each) whose reference decode (`p2_v1/store.py`'s `decode_block`) is
    refused or differs from their 2,054 bytes of `payloads`, with why."""
    from p2_v1.store import decode_block

    n = len(rows) // ROW
    if len(rows) != n * ROW or len(payloads) != n * PAYLOAD:
        raise ValueError(f"{len(rows)} bytes of rows and {len(payloads)} of payload are not the same whole blocks")
    bad = []
    for i in range(n):
        try:
            got = decode_block(params, prefix, first + i, bytes(rows[i * ROW:(i + 1) * ROW]))
        except ValueError as e:
            bad.append((first + i, str(e)))
            continue
        if got != bytes(payloads[i * PAYLOAD:(i + 1) * PAYLOAD]):
            bad.append((first + i, "decodes to other bytes than W's"))
    return bad
