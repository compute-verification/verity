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


CHUNKED = [False]
"""`decode_blocks` of at least `CHUNKED_MIN_ROWS` blocks keeps its state chunk-major (`_Chunked`, `feistel_chunked_kernel`)
instead of row-major; smaller decodes keep the row-major kernels, whose warp and lane-pair launches have the lower latency."""

CHUNKED_MIN_ROWS = [4096]
CHUNKED_THREADS = [128]
CHUNKED_SPECIALIZE = [True]
"""Chunk-major launches at the 64 KiB width run the kernel compiled for its half of 8,196 words."""
"""Threads per block of the chunk-major launches (fewer spread a launch over more SMs)."""

CHUNK = 136
SLOT = 144
"""A chunk of 136 bytes (one Keccak rate block) in a 16-byte-aligned slot of 144."""


class _Chunked:
    """`rows` states of `2L + 64` bytes stored chunk-major: byte `k` of half `h` of row `r` at `[h, k // 136, r, k % 136]`
    of int64 `[2, nch, rows, 18]` (144-byte slots, the last 8 bytes padding), so a launch reads every row's chunk `c` from
    one contiguous run."""

    def __init__(self, rows: int, width: int, device) -> None:
        self.rows, self.width, self.half = rows, width, width + 32
        self.nch = -(-self.half // CHUNK)
        self.words = torch.zeros(2, self.nch, rows, SLOT // 8, dtype=torch.int64, device=device)
        self.bytes = self.words.view(torch.uint8).view(2, self.nch, rows, SLOT)

    def _pieces(self, offset: int, length: int):
        k = 0
        while k < length:
            c, b = divmod(offset + k, CHUNK)
            if b == 0 and length - k >= CHUNK:
                n = (length - k) // CHUNK
                yield k, c, n, None
                k += n * CHUNK
            else:
                take = min(CHUNK - b, length - k)
                yield k, c, b, take
                k += take

    def put(self, half: int, offset: int, rows: slice, data: torch.Tensor) -> None:
        """Bytes `offset .. offset + len` of `half` of rows `rows` := `data` (`[n, len]`)."""
        for k, c, b, take in self._pieces(offset, data.shape[1]):
            if take is None:
                self.bytes[half, c:c + b, rows, :CHUNK] = data[:, k:k + b * CHUNK].reshape(-1, b, CHUNK).transpose(0, 1)
            else:
                self.bytes[half, c, rows, b:b + take] = data[:, k:k + take]

    def fill(self, half: int, offset: int, length: int, rows: slice, value: int) -> None:
        for _, c, b, take in self._pieces(offset, length):
            if take is None:
                self.bytes[half, c:c + b, rows, :CHUNK] = value
            else:
                self.bytes[half, c, rows, b:b + take] = value

    def get(self, half: int, offset: int, length: int) -> torch.Tensor:
        parts = []
        for _, c, b, take in self._pieces(offset, length):
            if take is None:
                parts.append(self.bytes[half, c:c + b, :, :CHUNK].transpose(0, 1).reshape(self.rows, b * CHUNK))
            else:
                parts.append(self.bytes[half, c, :, b:b + take])
        return torch.cat(parts, 1)

    def pad(self, rows: slice) -> None:
        self.fill(0, 0, self.width, rows, 0)
        self.bytes[0, 0, rows, 0] = 0x01
        c, b = divmod(self.width - 1, CHUNK)
        self.bytes[0, c, rows, b] = 0x80

    def permute(self, row0: int) -> None:
        from verity_vllm.engine.pous import native

        native.ext().feistel_chunked(self.words, row0, self.rows - row0, self.half // 8, codec.PI_DOMAIN, CHUNKED_THREADS[0],
                                     CHUNKED_SPECIALIZE[0])


def _decode_chunked(c_flat, seg, blk, ln, tags, base, starts, width: int) -> torch.Tensor:
    r = len(blk)
    st = _Chunked(r, width, c_flat.device)
    st.put(1, width - 32, slice(None), tags[seg])
    dom = torch.zeros(r, 32, dtype=torch.uint8, device=c_flat.device)
    dom[:, 0], dom[:, 1] = (blk & 0xFF).to(torch.uint8), (blk >> 8).to(torch.uint8)
    st.put(1, width, slice(None), dom)
    pending, stream = [], torch.cuda.current_stream()
    for j in range(1, int(ln[-1]) + 2):
        first, pad_end = starts[j - 1], starts[j]
        st.pad(slice(first, pad_end))
        if pad_end < r:
            from verity_vllm.engine.pous import native

            native.ext().chunked_gather(st.words, pad_end, c_flat, (base[pad_end:] + blk[pad_end:] - j).contiguous(), width // 8)
        st.permute(first)
        _throttle(pending, stream)
    return st.get(0, 0, width)


def decode_blocks(c_flat: torch.Tensor, tags: torch.Tensor, segs: torch.Tensor, blocks: torch.Tensor,
                  per_segment: int, window: int | None = None) -> torch.Tensor:
    """`W` of the blocks `(segs[r], blocks[r])` alone, from every segment's `C` (`[S * B, L]`, segment-major) and tags
    (`[S, 32]`): `[R, L]` in the given order.  Block `i`'s chain absorbs `C_{i-1}, .., C_{i-n}` with `n = i` on the complete
    DAG and `n = min(i, window)` on the band graph of in-degree `window`.  Rows are ordered by `n`, so the rows still absorbing
    at chain step `j` (`n >= j - 1`) are a suffix, and each step gathers their parents `C_{i-j}`."""
    width = c_flat.shape[1]
    length = blocks if window is None else torch.clamp(blocks, max=window)
    order = torch.argsort(length, stable=True)
    seg, blk, ln = segs[order], blocks[order], length[order]
    r = len(order)
    state = torch.zeros(r, 2 * width + 64, dtype=torch.uint8, device=c_flat.device)
    state[:, 2 * width:2 * width + 32] = tags[seg]
    state[:, 2 * width + 32] = (blk & 0xFF).to(torch.uint8)
    state[:, 2 * width + 33] = (blk >> 8).to(torch.uint8)
    starts = torch.searchsorted(ln, torch.arange(int(ln[-1]) + 2, device=ln.device)).tolist()
    base = seg * per_segment
    if CHUNKED[0] and r >= CHUNKED_MIN_ROWS[0]:
        w = _decode_chunked(c_flat, seg, blk, ln, tags, base, starts, width) ^ c_flat[base + blk]
        out = torch.empty_like(w)
        out[order] = w
        return out
    pending, stream = [], torch.cuda.current_stream()
    for j in range(1, int(ln[-1]) + 2):
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


def decode_segments(c: torch.Tensor, tags: torch.Tensor, window: int | None = None) -> torch.Tensor:
    """`W` of `G` whole segments from `C` (`[G, B, L]`) by `decode_blocks` (every block at once)."""
    g, b, width = c.shape
    blocks = torch.arange(g * b, device=c.device)
    return decode_blocks(c.reshape(g * b, width), tags, blocks // b, blocks % b, b, window).view(g, b, width)


def encode(w: torch.Tensor, tags: torch.Tensor, window: int | None = None) -> torch.Tensor:
    """`C` of `G` segments from `W` (`[G, B, L]`), block by block (`window` as in `decode_blocks`)."""
    g, b, width = w.shape
    c_bm = torch.zeros(b, g, width, dtype=torch.uint8, device=w.device)
    w_bm = w.transpose(0, 1)
    for i in range(b):
        n = i if window is None else min(i, window)
        state = _initial(tags, i + 1, width)[i * g:].clone()
        for j in range(1, n + 2):
            if j <= n:
                state[:, :width] = c_bm[i - j]
            else:
                _pad(state, width)
            codec.permute(state)
        c_bm[i] = state[:, :width] ^ w_bm[i]
    return c_bm.transpose(0, 1).contiguous()
