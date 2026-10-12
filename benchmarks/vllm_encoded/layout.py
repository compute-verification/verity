"""The word codec's layouts on torch tensors: what `benchmarks/word_codec/codec.py` computes byte by byte, done whole-matrix on the
device, so a model's linears can be laid out, packed and unpacked at load time.

~~~text
fragment stream   W (N × K, row-major bytes) as each (task, lane)'s 256 bytes: task = (16-row block nb, 512-column chunk kc),
                  lane l = 4g + q, byte 16·tt + 4·r + b = W[16·nb + g + 8·(r & 1), 512·kc + 32·tt + 4·q + 16·(r >> 1) + b]
plain fragments   per task its 16 k-tiles, each tile's 32 lanes × 16 bytes (the plain kernel's operand, KR = 0)
payloads          a lane's 256 bytes as L words of b payload bits (b = 56: L = 37, seven bytes a word; b = 60: L = 35,
                  fifteen bytes in two words), stored at L·32·task + 32·w + l
~~~

`codec.fragment_stream`, `codec.plain_fragments` and `codec.fragment_payloads` are the references (`tests/test_layout.py`).
Every function takes and returns uint8 or int64 tensors on any device; int64 holds a payload or codec word (< 2^61) as is.
"""
from __future__ import annotations

import torch

TILE_ROWS, TASK_COLS, TASK_TILES, LANES, LANE_BYTES = 16, 512, 16, 32, 256
PACKINGS = (56, 60)


def lane_words(bits: int) -> int:
    if bits not in PACKINGS:
        raise ValueError(f"a word carries one of {PACKINGS} payload bits")
    return -(-8 * LANE_BYTES // bits)


def check_shape(n: int, k: int) -> None:
    if n <= 0 or k <= 0 or n % TILE_ROWS or k % TASK_COLS:
        raise ValueError(f"the fragment layout takes N a multiple of {TILE_ROWS} and K of {TASK_COLS}, not {n} × {k}")


def tasks(n: int, k: int) -> int:
    check_shape(n, k)
    return (n // TILE_ROWS) * (k // TASK_COLS)


def _bytes(w: torch.Tensor) -> torch.Tensor:
    if w.dim() != 2:
        raise ValueError("W is a matrix")
    w = w.contiguous()
    return w if w.dtype == torch.uint8 else w.view(torch.uint8)


def fragment_stream(w: torch.Tensor) -> torch.Tensor:
    """(tasks, 32, 256) uint8: each (task, lane)'s stream, tasks in (row block, column chunk) order."""
    w = _bytes(w)
    n, k = w.shape
    check_shape(n, k)
    v = w.view(n // 16, 2, 8, k // 512, 16, 2, 4, 4)       # nb, r&1, g, kc, tt, r>>1, q, b
    return v.permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(-1, LANES, LANE_BYTES)


def unfragment(s: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """W (n × k uint8) from its fragment stream."""
    check_shape(n, k)
    v = s.reshape(n // 16, k // 512, 8, 4, 16, 2, 2, 4)      # nb, kc, g, q, tt, r>>1, r&1, b
    return v.permute(0, 6, 2, 1, 4, 5, 3, 7).reshape(n, k)


def plain_fragments(w: torch.Tensor) -> torch.Tensor:
    """The plain kernel's operand, flat uint8: per task its 16 tiles, each 32 lanes × 16 bytes."""
    s = fragment_stream(w).view(-1, LANES, TASK_TILES, 16)
    return s.permute(0, 2, 1, 3).reshape(-1)


def unplain(f: torch.Tensor, n: int, k: int) -> torch.Tensor:
    s = f.view(-1, TASK_TILES, LANES, 16).permute(0, 2, 1, 3).reshape(-1, LANES, LANE_BYTES)
    return unfragment(s, n, k)


def _shifts(device: torch.device, *values: int) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device=device)


def pack(s: torch.Tensor, bits: int) -> torch.Tensor:
    """(..., 256) uint8 lane streams -> (..., L) int64 payloads, `codec.pack`'s little-endian bit order."""
    lw, lead = lane_words(bits), s.shape[:-1]
    if bits == 56:
        pad = torch.zeros(*lead, 7 * lw - LANE_BYTES, dtype=torch.uint8, device=s.device)
        b = torch.cat([s, pad], -1).view(*lead, lw, 7)
        b = torch.cat([b, torch.zeros(*lead, lw, 1, dtype=torch.uint8, device=s.device)], -1)
        return b.contiguous().view(torch.int64).view(*lead, lw)
    groups = -(-LANE_BYTES // 15)
    pad = torch.zeros(*lead, 15 * groups - LANE_BYTES, dtype=torch.uint8, device=s.device)
    g = torch.cat([s, pad], -1).view(*lead, groups, 15).to(torch.int64)
    lo = (g[..., :7] << _shifts(s.device, 0, 8, 16, 24, 32, 40, 48)).sum(-1) | ((g[..., 7] & 0xF) << 56)
    hi = (g[..., 7] >> 4) | (g[..., 8:] << _shifts(s.device, 4, 12, 20, 28, 36, 44, 52)).sum(-1)
    return torch.stack([lo, hi], -1).view(*lead, 2 * groups)[..., :lw].contiguous()


def unpack(p: torch.Tensor, bits: int) -> torch.Tensor:
    """`pack`'s inverse: (..., L) int64 payloads -> (..., 256) uint8."""
    lw, lead = lane_words(bits), p.shape[:-1]
    if p.shape[-1] != lw:
        raise ValueError(f"a lane is {lw} words at {bits} bits")
    if bits == 56:
        return p.contiguous().view(torch.uint8).view(*lead, lw, 8)[..., :7].reshape(*lead, 7 * lw)[..., :LANE_BYTES]
    groups = -(-lw // 2)
    if lw % 2:
        p = torch.cat([p, torch.zeros(*lead, 1, dtype=torch.int64, device=p.device)], -1)
    w = p.view(*lead, groups, 2)
    lo, hi = w[..., 0:1], w[..., 1:2]
    b0 = (lo >> _shifts(p.device, 0, 8, 16, 24, 32, 40, 48)) & 0xFF
    b7 = ((lo >> 56) & 0xF) | ((hi & 0xF) << 4)
    b8 = (hi >> _shifts(p.device, 4, 12, 20, 28, 36, 44, 52)) & 0xFF
    return torch.cat([b0, b7, b8], -1).to(torch.uint8).reshape(*lead, 15 * groups)[..., :LANE_BYTES]


def payloads(w: torch.Tensor, bits: int) -> torch.Tensor:
    """W's payloads in store order (word w of lane l of a task at L·32·task + 32·w + l), flat int64."""
    return pack(fragment_stream(w), bits).permute(0, 2, 1).reshape(-1)


def unpayload(p: torch.Tensor, n: int, k: int, bits: int) -> torch.Tensor:
    """W (n × k uint8) from its payloads in store order."""
    lw = lane_words(bits)
    s = unpack(p.view(-1, lw, LANES).permute(0, 2, 1), bits)
    return unfragment(s, n, k)


def store_words(n: int, k: int, bits: int) -> int:
    return tasks(n, k) * lane_words(bits) * LANES


__all__ = ["LANES", "LANE_BYTES", "PACKINGS", "TASK_COLS", "TASK_TILES", "TILE_ROWS", "check_shape", "fragment_stream",
           "lane_words", "pack", "payloads", "plain_fragments", "store_words", "tasks", "unfragment", "unpack", "unpayload",
           "unplain"]
