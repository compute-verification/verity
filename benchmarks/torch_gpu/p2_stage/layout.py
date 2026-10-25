"""Where a layer's matrices lie in the P2-encoded weights store, and P2's decode of one block on the CPU (the check's
reference). No torch.

The layout is a parameter until CMP-95 item 2's fixture fixes it. `StoreLayout` is PoUS's `Layout` (each tensor's byte
range in W, and W's P2 geometry), the prefix every block's K and t are expanded under, and the format that names a layer's
tensors. Until the fixture lands, `stream_layout` is p2_v1's own block format over the seeded weights stream
(`verity_torch_loop.weights`: every tensor back to back, no padding):

- the registered codec `p2-16448/v4` (`verity/core/protocols/pous/schemes/p2.py`): 2,056-byte big-endian blocks, 65,536 a
  segment, global block j being block j mod 65,536 of segment j div 65,536;
- block j carries W bytes [2054 j, 2054 j + 2054) as bytes 2 .. 2055 of its decoded row, whose bytes 0 and 1 are zero
  (`decode_segment`'s payload);
- K and t from counter-mode SHA-512 after `schemes.p2.expand_prefix` of the protocol's 24-byte salt (`registered_prefix`),
  or any other prefix of whole 128-byte blocks (`reference.expand_prefix`'s, the spec's).

A matrix is what one GEMM reads (`MATRICES`): QKV is wq, wk and wv one after another (16,512 x 16,384 at the frozen
shape), and O, up and down are one tensor each. Each tensor's per-channel f32 scales are `<tensor>.scale`.

`decode_block` is the reference's decode of one block (`p2_v1.reference.decode` under `reference.keys_after`), refusing
what `schemes.p2.decode_segment` refuses: a row not below p, or a decode wider than the payload (no codeword).
"""
from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from verity.core.protocols.pous.layout import Layout, Placement
from verity.core.protocols.pous.schemes import p2 as S

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks" / "pous"))
from p2_v1 import reference as R  # noqa: E402

PARAMS = S.P2Params(*S.POINT)
SPEC = R.Params(PARAMS.w, PARAMS.c0, PARAMS.payload_bytes, PARAMS.blocks_per_segment)
ROW = PARAMS.block_bytes          # 2,056
PAYLOAD = PARAMS.payload_bytes    # 2,054
LEAD = ROW - PAYLOAD              # a decoded row's zero bytes before its payload

MATRICES = {"qkv": ("wq", "wk", "wv"), "o": ("wo",), "up": ("wup",), "down": ("wdown",)}


def registered_prefix(salt: bytes) -> bytes:
    """The registered scheme's key prefix for the protocol's 24-byte `salt`: whole 128-byte blocks, one midstate."""
    return S.expand_prefix(PARAMS, salt)


def blocks_of(offset: int, size: int) -> range:
    """The blocks holding W bytes [offset, offset + size)."""
    if offset < 0 or size < 0:
        raise ValueError("a byte range has a nonnegative offset and size")
    return range(offset // PAYLOAD, -(-(offset + size) // PAYLOAD)) if size else range(0)


@dataclass(frozen=True)
class StoreLayout:
    """`layout`'s tensors in W at the registered P2 point, keys after `prefix`; a layer's tensor `t` is named
    `names.format(layer=layer, tensor=t)`."""

    layout: Layout
    prefix: bytes
    names: str = "l{layer}.{tensor}"

    def __post_init__(self) -> None:
        p = self.layout.params
        if not isinstance(p, S.P2Params) or (p.w, p.c0, p.blocks_per_segment) != S.POINT:
            raise ValueError(f"the store is P2 at {S.POINT} (w, c0, blocks a segment), the point libp2v1.so is built for")
        if not self.prefix or len(self.prefix) % 128:
            raise ValueError("the key prefix is whole 128-byte SHA-512 blocks (p2v1_keys compresses it to one midstate)")

    def tensor(self, layer: int, name: str) -> Placement:
        return self.layout.placement(self.names.format(layer=layer, tensor=name))

    def parts(self, layer: int, matrix: str, scales: bool = False) -> tuple[Placement, ...]:
        """The stream tensors of `matrix` (or their scales) in the order its rows are staged."""
        if matrix not in MATRICES:
            raise KeyError(f"a matrix is one of {sorted(MATRICES)}, not {matrix!r}")
        return tuple(self.tensor(layer, f"{t}.scale" if scales else t) for t in MATRICES[matrix])

    def nbytes(self, layer: int, matrix: str, scales: bool = False) -> int:
        return sum(p.size for p in self.parts(layer, matrix, scales))

    def span(self, layer: int) -> range:
        """The blocks holding every matrix of `layer` and its scales."""
        ranges = [blocks_of(p.offset, p.size) for m in MATRICES for s in (False, True) for p in self.parts(layer, m, s)]
        return range(min(r.start for r in ranges), max(r.stop for r in ranges))

    def manifest(self) -> dict:
        return {"layout": self.layout.manifest(), "prefix": self.prefix.hex(), "names": self.names}


def stream_layout(sizes: Iterable[tuple[str, int]], prefix: bytes) -> StoreLayout:
    """The tensors `(name, nbytes)` back to back from W's first byte, no padding (`Layout.of`, align=False)."""
    return StoreLayout(Layout.of(PARAMS, sizes), prefix)


def layout_of(manifest: dict, prefix: bytes) -> StoreLayout:
    """Each tensor at the W offset a manifest gives: PoUS's `Layout.manifest` (``[name, offset, size]`` rows) or the
    weights stream's (`verity_torch_loop.weights.manifest`: rows with ``name``, ``offset`` and ``nbytes``)."""
    rows = [(t["name"], t["offset"], t["nbytes"]) if isinstance(t, dict) else tuple(t) for t in manifest["tensors"]]
    return StoreLayout(Layout.of_ranges(PARAMS, rows), prefix)


def decode_block(prefix: bytes, j: int, c: bytes) -> bytes:
    """Global block `j`'s 2,054 payload bytes from its 2,056-byte row `c`, by the P2 spec's reference; `ValueError` for a
    row that is not a codeword."""
    if len(c) != ROW:
        raise ValueError(f"block {j}: a row is {ROW} bytes, not {len(c)}")
    v = int.from_bytes(c, "big")
    if v >= SPEC.p:
        raise ValueError(f"block {j} is not below p")
    s, b = divmod(j, PARAMS.blocks_per_segment)
    k, t, _ = R.keys_after(prefix, SPEC, s, b)
    m = R.decode(SPEC, v, k, t)
    if m >> (8 * PAYLOAD):
        raise ValueError(f"block {j} decodes to more than {8 * PAYLOAD} bits: not a codeword")
    return m.to_bytes(PAYLOAD, "big")


def check_blocks(prefix: bytes, first: int, rows: bytes | memoryview, payloads: bytes | memoryview) -> list[tuple[int, str]]:
    """The blocks `first ..` of `rows` (2,056 bytes each) whose reference decode is refused or differs from their 2,054
    bytes of `payloads`, with why."""
    n = len(rows) // ROW
    if len(rows) != n * ROW or len(payloads) != n * PAYLOAD:
        raise ValueError(f"{len(rows)} bytes of rows and {len(payloads)} of payload are not the same whole blocks")
    bad = []
    for i in range(n):
        try:
            got = decode_block(prefix, first + i, bytes(rows[i * ROW:(i + 1) * ROW]))
        except ValueError as e:
            bad.append((first + i, str(e)))
            continue
        if got != bytes(payloads[i * PAYLOAD:(i + 1) * PAYLOAD]):
            bad.append((first + i, "decodes to other bytes than W's"))
    return bad
