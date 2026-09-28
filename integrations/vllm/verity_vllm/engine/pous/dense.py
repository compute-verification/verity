"""The GPU twin of `verity_pous.dense.DENSE`: one layer on the complete DAG, the overwrite chain over `Pi2`, no `P`.

Per segment `s` with `tag_s = salt || u64(s)`, block `i` of `B` blocks of `L` bytes (`dense.POINT_64K`: 512 of 65,536):

    h_0   = 0^L || tag_s || u256(i)                  (the chaining value is `L + 64` bytes; `dom` fills its last 64)
    key_i = r of the last (r, h) <- Pi2(B || h) over C_{i-1}, ..., C_0, pad        pad = 0x01 || 0 .. 0 || 0x80 (L bytes)
    C_i   = key_i ^ W_i

`Pi2` is Feistel-10 on `2L + 64` bytes (`codec.permute`).  Block `i` costs `i + 1` calls.  Rows are kept block-major (row
`i * G + g` for block `i` of segment `g`), so at chain step `j` the rows still absorbing (`i >= j - 1`) are a contiguous suffix of
the state, the first `G` of them taking the pad and the rest their parents `C_{i-j}`, a contiguous prefix of the block-major `C`:
each step is one Feistel launch on a view.  Decode runs every row at once (`B` steps); encode is sequential in `i`
(`B (B + 1) / 2` calls on the critical path), so it is only used on short segments here.
"""
from __future__ import annotations

import torch

from verity_vllm.engine.pous import codec


def _initial(tags: torch.Tensor, blocks: int, width: int) -> torch.Tensor:
    """`[blocks * G, 2L + 64]` block-major states `0^L || 0^L || tag_g || u256(i)`."""
    g = tags.shape[0]
    state = torch.zeros(blocks * g, 2 * width + 64, dtype=torch.uint8, device=tags.device)
    view = state.view(blocks, g, 2 * width + 64)
    view[:, :, 2 * width:2 * width + 32] = tags
    i = torch.arange(blocks, device=tags.device)
    view[:, :, 2 * width + 32] = (i & 0xFF).to(torch.uint8).view(blocks, 1)
    view[:, :, 2 * width + 33] = (i >> 8).to(torch.uint8).view(blocks, 1)
    return state


def _pad(rows: torch.Tensor, width: int) -> None:
    rows[:, :width] = 0
    rows[:, 0], rows[:, width - 1] = 0x01, 0x80


def keys(c_block_major: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
    """`key_i` of every block of every segment, from `C` block-major (`[B, G, L]`): `[B * G, L]`, block-major."""
    b, g, width = c_block_major.shape
    state = _initial(tags, b, width)
    parents = c_block_major.reshape(-1, width)
    pending, stream = [], torch.cuda.current_stream()
    for j in range(1, b + 1):
        suffix = state[(j - 1) * g:]
        _pad(suffix[:g], width)
        if j < b:
            suffix[g:, :width] = parents[:(b - j) * g]
        codec.permute(suffix)
        _throttle(pending, stream)
    return state[:, :width]


def decode(c: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
    """`W` of `G` segments from `C` (`[G, B, L]`): `W_i = C_i ^ key_i`, every block at once."""
    g, b, width = c.shape
    c_bm = c.transpose(0, 1).contiguous()
    w = keys(c_bm, tags).view(b, g, width)
    w ^= c_bm
    return w.transpose(0, 1).contiguous()


IN_FLIGHT = 2
"""Chain steps a decode keeps queued on the device.  A deeper queue fills the launch queue, and a launch blocked on it holds
the context against the in-process responder's copies (up to a kernel's duration late)."""


def _throttle(pending: list, stream) -> None:
    import torch

    ev = torch.cuda.Event()
    ev.record(stream)
    pending.append(ev)
    if len(pending) > IN_FLIGHT:
        pending.pop(0).synchronize()


def decode_blocks(c_flat: torch.Tensor, tags: torch.Tensor, segs: torch.Tensor, blocks: torch.Tensor,
                  per_segment: int) -> torch.Tensor:
    """`W` of the blocks `(segs[r], blocks[r])` alone, from every segment's `C` (`[S * B, L]`, segment-major) and tags
    (`[S, 32]`): `[R, L]` in the given order.  Block `i` needs only `C_0 .. C_i`; rows are ordered by block so the rows still
    absorbing at chain step `j` (`i >= j - 1`) are a suffix, and each step gathers their parents `C_{i-j}`."""
    width = c_flat.shape[1]
    order = torch.argsort(blocks, stable=True)
    seg, blk = segs[order], blocks[order]
    r = len(order)
    state = torch.zeros(r, 2 * width + 64, dtype=torch.uint8, device=c_flat.device)
    state[:, 2 * width:2 * width + 32] = tags[seg]
    state[:, 2 * width + 32] = (blk & 0xFF).to(torch.uint8)
    state[:, 2 * width + 33] = (blk >> 8).to(torch.uint8)
    starts = torch.searchsorted(blk, torch.arange(int(blk[-1]) + 2, device=blk.device)).tolist()
    base = seg * per_segment
    pending, stream = [], torch.cuda.current_stream()
    for j in range(1, int(blk[-1]) + 2):
        first = starts[j - 1]
        pad_end = starts[j] if j < len(starts) else r
        suffix = state[first:]
        _pad(suffix[:pad_end - first], width)
        if pad_end < r:
            suffix[pad_end - first:, :width] = c_flat[base[pad_end:] + blk[pad_end:] - j]
        codec.permute(suffix)
        _throttle(pending, stream)
    w = state[:, :width] ^ c_flat[base + blk]
    out = torch.empty_like(w)
    out[order] = w
    return out


def encode(w: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
    """`C` of `G` segments from `W` (`[G, B, L]`), block by block."""
    g, b, width = w.shape
    c_bm = torch.zeros(b, g, width, dtype=torch.uint8, device=w.device)
    w_bm = w.transpose(0, 1)
    for i in range(b):
        state = _initial(tags, i + 1, width)[i * g:].clone()
        for j in range(1, i + 2):
            if j <= i:
                state[:, :width] = c_bm[i - j]
            else:
                _pad(state, width)
            codec.permute(state)
        c_bm[i] = state[:, :width] ^ w_bm[i]
    return c_bm.transpose(0, 1).contiguous()
