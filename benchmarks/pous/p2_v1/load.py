"""The store's load: where tensors lie in the P2-encoded weights store, and their bytes decoded from its rows into plaintext
load buffers on the GPU by `libp2v1.so` (`gpu.Kernels`), with the refusals the store's reference decode (`store.py`'s
`decode_range`) makes that the library's fast decode leaves to its caller. No torch until a decode is built.

The store's layout is this package's fixture (`store.py`, schema `verity-pous/p2-store-layout/v1`):

- the registered codec `p2-16448/v4` (`verity/protocols/pous/verifier/schemes/p2.py`): 2,056-byte big-endian blocks, block
  i = blocks_per_segment · s + v being block v of segment s;
- block i's payload is W bytes [2054 i, 2054 i + 2054), bytes 2 .. 2055 of its decoded row, whose bytes 0 and 1 are zero;
- each tensor (s8 [N, K] row-major) starts on a block, `first_block`, takes ⌈N K / 2054⌉ blocks, and its last block's
  payload is zero past the tensor; in its layer's load buffer it starts at `staging_offset`;
- K and t from counter-mode SHA-512 after `schemes.p2.expand_prefix` of the protocol's 24-byte salt.

A layer's matrices (`MATRICES`) are its four tensors: QKV (the query heads, then K, then V: 16,512 x 16,384 at the frozen
shape), O, up and down; after the last layer the head (`HEAD`, 256 x 16,384) is layer `layers`. Their scales are not in W.

`EncodedRows` is the encoded rows of a range of the store's blocks resident on the device (`rows`, [n, 2056] uint8, block
`first` first) under a `StoreLayout`. `decode_into` decodes a range of W `chunk` blocks at a time: `Kernels.decode` expands
each block's K and t on the device (`p2v1_keys`, v4) and decodes (`p2v1_decode`), and `scatter` copies the payload bytes
2 .. 2055 of each decoded row that fall in the range to their place in the buffer. Nothing is re-laid out.

Every chunk is screened on the device and nothing is fixed silently. `EncodedRows.check` raises `StoreError`, with the fixture's
flag bits, for anything decoded since the last check that the reference refuses: bit 2 for a decoded row whose two lead
bytes aren't zero (wider than the payload: a corrupt row or other keys) or a row at or above p (a row whose first eight
bytes are all 0xFF is compared with p exactly), bit 3 for a tensor's last block nonzero past the tensor's end, and the
library's own flags (bit 0: a tweak's first candidate at or above p, where the spec samples the next).

With a pass of PoUB attached (`EncodedRows.sketch`, an object with `read(rows, first, j0, j1)`), `decode_rows` sketches
the units each chunk's blocks lie in from the same rows on the device before the decode reads them; the rows then hold
whole units.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.protocols.pous.verifier.schemes import p2 as S

from . import gpu
from . import reference as R

PARAMS = S.P2Params(*S.POINT)
SPEC = R.DEPLOYMENT
ROW = PARAMS.block_bytes          # 2,056
PAYLOAD = PARAMS.payload_bytes    # 2,054
LEAD = ROW - PAYLOAD              # a decoded row's zero bytes before its payload
COLUMNS = ("name", "layer", "n", "k", "offset", "size", "first_block", "blocks", "staging_offset")

MATRICES = {"qkv": "self_attn.qkv_proj", "o": "self_attn.o_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
PARTS = {"qkv": ("wq", "wk", "wv"), "o": ("wo",), "up": ("wup",), "down": ("wdown",)}
#: the store's last tensor, after the layers (as layer `layers`), and its tensor in the weights stream
HEAD, STREAM_HEAD = "lm_head.weight", "head"

P_BYTES = SPEC.p.to_bytes(ROW, "big")
WIDE, PADDING = 1 << 2, 1 << 3


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
                raise ValueError(f"{t.name} overlaps the tensor before it in layer {t.layer}'s load buffer")
            staged[t.layer] = t.staging_offset + t.size
        if at > p.blocks:
            raise ValueError(f"the tensors run to block {at}, past the store's {p.blocks}")

    @property
    def spec(self) -> R.Params:
        return R.Params.of(self.params)

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


class StoreError(RuntimeError):
    """The store held something the decode refuses; `flags` holds the fixture's flag bits."""

    def __init__(self, message: str, flags: int) -> None:
        super().__init__(message)
        self.flags = flags


def scatter(m: Any, j0: int, out: Any, offset: int, size: int) -> int:
    """Copy the payload bytes of decoded rows `m` ([n, 2056] uint8, blocks `j0 ..`) that fall in W bytes
    [offset, offset + size) to `out` (that range's bytes, flat uint8); the number of bytes copied."""
    n = m.shape[0]
    lo, hi = max(offset, j0 * PAYLOAD), min(offset + size, (j0 + n) * PAYLOAD)
    done = 0
    while lo < hi:
        i, col = divmod(lo - j0 * PAYLOAD, PAYLOAD)
        if col or hi - lo < PAYLOAD:
            k = min(PAYLOAD - col, hi - lo)
            out[lo - offset:lo - offset + k].copy_(m[i, LEAD + col:LEAD + col + k])
        else:
            rows = (hi - lo) // PAYLOAD
            k = rows * PAYLOAD
            out[lo - offset:lo - offset + k].view(rows, PAYLOAD).copy_(m[i:i + rows, LEAD:])
        lo, done = lo + k, done + k
    return done


class EncodedRows:
    """Encoded rows of the store's blocks `first .. first + rows.shape[0] - 1`, on the decode's device."""

    def __init__(self, layout: StoreLayout, rows: Any, first: int, lib_dir: str | Path, chunk: int = 1 << 16) -> None:
        import torch

        if rows.dtype != torch.uint8 or rows.dim() != 2 or rows.shape[1] != ROW or not rows.is_contiguous():
            raise ValueError(f"the store's rows are a contiguous [n, {ROW}] uint8 tensor")
        if first < 0 or chunk < 1:
            raise ValueError("the first block is nonnegative and a chunk at least one block")
        self.layout, self.rows, self.first, self.chunk = layout, rows, first, chunk
        self.kernels = gpu.Kernels(str(lib_dir), "", layout.spec, layout.prefix)
        if rows.device.type != self.kernels.device.type:
            raise ValueError(f"the store's rows are on {rows.device}, the decode on {self.kernels.device}")
        self.kernels.raised()
        self._counts = torch.zeros(3, dtype=torch.int64, device=rows.device)
        self._high: list[tuple[int, Any]] = []
        self.blocks_decoded = 0
        self.sketch: Any = None

    @property
    def blocks(self) -> range:
        return range(self.first, self.first + self.rows.shape[0])

    def decode_rows(self, j0: int, j1: int) -> Any:
        """Decoded rows of blocks `j0 .. j1 - 1`, screened (`check` reports what the screen found)."""
        import torch

        if not (self.first <= j0 <= j1 <= self.blocks.stop):
            raise ValueError(f"blocks [{j0}, {j1}) are not all in the store's [{self.first}, {self.blocks.stop})")
        c = self.rows[j0 - self.first:j1 - self.first]
        if self.sketch is not None:
            self.sketch.read(self.rows, self.first, j0, j1)
        m = self.kernels.decode(c, j0)
        high = (c[:, :8] == 0xFF).all(dim=1)
        self._counts[:2] += torch.stack(((m[:, :LEAD] != 0).any(dim=1).sum(), high.sum()))
        self._high.append((j0, high))
        self.blocks_decoded += j1 - j0
        return m

    def decode_into(self, offset: int, size: int, out: Any, *, end: bool = False) -> None:
        """W bytes [offset, offset + size) into `out` (`size` bytes, flat uint8). With `end`, the range ends a tensor,
        and the rest of its last block's payload must be zero."""
        import torch

        if out.numel() != size:
            raise ValueError(f"{size} bytes don't fit a {out.numel()}-byte buffer")
        r = blocks_of(offset, size)
        used = offset + size - (r.stop - 1) * PAYLOAD
        for j0 in range(r.start, r.stop, self.chunk):
            j1 = min(r.stop, j0 + self.chunk)
            m = self.decode_rows(j0, j1)
            scatter(m, j0, out, offset, size)
            if end and j1 == r.stop and used < PAYLOAD:
                self._counts[2] += (m[-1, LEAD + used:] != 0).any().to(torch.int64)

    def check(self) -> None:
        """Raise `StoreError` for anything decoded since the last check that the decode refuses."""
        wide, high, padded = (int(v) for v in self._counts.tolist())
        lib = self.kernels.raised()
        pending, self._high = self._high, []
        self._counts.zero_()
        bad = []
        if high:
            for j0, mask in pending:
                for i in mask.nonzero().flatten().tolist():
                    if bytes(self.rows[j0 - self.first + i].cpu().numpy()) >= P_BYTES:
                        bad.append(j0 + i)
        why = []
        if wide:
            why.append(f"{wide} rows decode wider than the {PAYLOAD}-byte payload (not codewords: a corrupt row or "
                       "other keys)")
        if bad:
            why.append(f"{len(bad)} rows at or above p (blocks {bad[:8]})")
        if padded:
            why.append(f"{padded} tensors' last blocks are nonzero past the tensor's end")
        if lib & 1:
            why.append("a tweak's first candidate at or above p (the spec derives the next; the library doesn't)")
        if lib & ~1:
            why.append(f"library flags {lib:#x}")
        if why:
            raise StoreError("; ".join(why), lib | (WIDE if wide or bad else 0) | (PADDING if padded else 0))
