"""The GPU twin of `verity.core.protocols.pous.dense.DENSE`: one layer on the complete DAG, the overwrite chain over `Pi2`, no `P`.

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

from . import codec


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

CHUNKED_MIN_ROWS = [1024]
CHUNKED_THREADS = [128]
CHUNKED_SPECIALIZE = [True]
CHUNKED_SPLIT = [False]
CHUNKED_SPLIT_MAX_ROWS = [9_500]
"""Chunk-major launches at the 64 KiB width below `CHUNKED_SPLIT_MAX_ROWS` rows split every Keccak state across a lane pair
(`feistel_split_kernel`): 25.9 against 39.1 ms at 4,608 rows on the L40S, but 63.9 against 47.3 ms at 15,075, where the
unsplit kernel already fills the GPU and the pair's shuffles cost more than its parallelism buys."""

CHUNKED_FUSED_X = [False]
"""Unsplit chunk-major decodes read each step's parents (and the pad) inside the Feistel launch instead of gathering them."""

CHUNKED_WHOLE = [False]
"""Chunk-major decodes of at least `CHUNKED_SPLIT_MAX_ROWS` rows at the 64 KiB width run every row's whole chain in one
unsplit launch (`band_chain_kernel`), writing `W` straight to its place: no per-step launches and no key extraction.  Smaller
decodes keep the per-step split launches, which are faster there."""

CHAIN_PAIRS = [False]
"""The whole-chain decode runs on lane pairs in one persistent launch (`band_chain_pairs_kernel`): one Keccak state per lane,
the squeezing and absorbing sponges of a row on two lanes, a warp per scheduler taking (rows, call) tasks, the lanes
bit-interleaved, and `X` read from an interleaved copy of `C` (1 GB more device memory for the 0.5B weight set's 30
segments).  `band_chain_kernel`'s 15,360 threads fill 480 of the RTX PRO 6000's 752 schedulers; this fills them all, and
decodes at 2.80 against 1.60 GB/s there (untimed screens, art:4354a29314cf against art:468c10537551).  It runs from
`CHUNKED_MIN_ROWS` rows (two segments), so a gates-only run on two segments exercises it."""

DECODE_PATHS: list[tuple[str, int]] = []
"""`(kernel, rows)` of each `decode_blocks` call, in order; the harness takes and clears them to record which decode ran."""

CHUNKED_MAX_ROWS = [18_000]
"""Chunk-major launches over more rows run as equal sequential pieces of at most this many: past about one warp per
scheduler (18k rows on the L40S) a launch's per-row rate collapses (23,744 rows: 93.6 ms at once, 84.7 ms in two)."""
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

    def launches(self, row0: int, allow_split: bool = True) -> list[tuple[int, int, bool]]:
        """`(start, rows, split)` of the launches covering rows `row0 ..`: pieces of at most `CHUNKED_MAX_ROWS`."""
        rows = self.rows - row0
        pieces = -(-rows // CHUNKED_MAX_ROWS[0])
        step = -(-rows // pieces)
        split = allow_split and CHUNKED_SPLIT[0] and step < CHUNKED_SPLIT_MAX_ROWS[0]
        return [(start, min(step, self.rows - start), split) for start in range(row0, self.rows, step)]

    def permute(self, row0: int, xsrc: torch.Tensor | None = None, xidx: torch.Tensor | None = None) -> None:
        """Π₂ on rows `row0 ..`; with `xsrc`, round 1 reads each row's X from row `xidx[r]` of `xsrc` (the pad where < 0)."""
        from . import native

        for start, n, split in self.launches(row0, allow_split=xsrc is None):
            native.ext().feistel_chunked(self.words, start, n, self.half // 8, codec.PI_DOMAIN, CHUNKED_THREADS[0],
                                         CHUNKED_SPECIALIZE[0], split, xsrc, xidx)


def _decode_chunked(c_flat, seg, blk, ln, tags, base, starts, width: int) -> torch.Tensor:
    r = len(blk)
    st = _Chunked(r, width, c_flat.device)
    st.put(1, width - 32, slice(None), tags[seg])
    dom = torch.zeros(r, 32, dtype=torch.uint8, device=c_flat.device)
    dom[:, 0], dom[:, 1] = (blk & 0xFF).to(torch.uint8), (blk >> 8).to(torch.uint8)
    st.put(1, width, slice(None), dom)
    pending, stream = [], torch.cuda.current_stream()
    fused = CHUNKED_FUSED_X[0] and not any(split for _, _, split in st.launches(0)) and c_flat.is_contiguous()
    xidx = torch.full((r,), -1, dtype=torch.int64, device=c_flat.device) if fused else None
    for j in range(1, int(ln[-1]) + 2):
        first, pad_end = starts[j - 1], starts[j]
        if fused:
            xidx[first:pad_end] = -1
            if pad_end < r:
                xidx[pad_end:] = base[pad_end:] + blk[pad_end:] - j
            st.permute(first, c_flat, xidx)
        else:
            st.pad(slice(first, pad_end))
            if pad_end < r:
                from . import native

                native.ext().chunked_gather(st.words, pad_end, c_flat, (base[pad_end:] + blk[pad_end:] - j).contiguous(),
                                            width // 8)
            st.permute(first)
        _throttle(pending, stream)
    return st.get(0, 0, width)


def _decode_whole(c_flat, seg, blk, ln, tags, base, order, width: int) -> torch.Tensor:
    from . import native

    r = len(blk)
    st = _Chunked(r, width, c_flat.device)
    st.put(1, width - 32, slice(None), tags[seg])
    dom = torch.zeros(r, 32, dtype=torch.uint8, device=c_flat.device)
    dom[:, 0], dom[:, 1] = (blk & 0xFF).to(torch.uint8), (blk >> 8).to(torch.uint8)
    st.put(1, width, slice(None), dom)
    out = torch.empty(r, width, dtype=torch.uint8, device=c_flat.device)
    base, blk, ln, dest = base.contiguous(), blk.contiguous(), ln.contiguous(), order.contiguous()
    if CHAIN_PAIRS[0]:
        # The two puts above are the state's only nonzero bytes, all from chunk (width - 32) // CHUNK of half 1 on.
        native.ext().band_chain_pairs(st.words, 0, r, codec.PI_DOMAIN, int(ln.max()) + 1, c_flat, base, blk, ln, dest, out,
                                      (width - 32) // CHUNK)
        return out
    for start, n, _ in st.launches(0, allow_split=False):
        native.ext().band_chain(st.words, start, n, codec.PI_DOMAIN, CHUNKED_THREADS[0], c_flat, base, blk, ln, dest, out)
    return out


def chain_call(width: int, device):
    """One row's chain of length 0 on `band_chain_pairs_kernel`, its inputs staged: one `Π₂` call, its first absorb then
    rounds 1 .. 9 (W is written in round 9), each round's absorb one chunk behind the previous round's squeeze, so 483 +
    9 * 484 = 4,839 sequential Keccak-f where absorbing and squeezing in turn takes 10 * (483 + 482) = 9,650.  A chain's later
    calls cost the same (10 rounds of 484, the next first absorb overlapping round 10).  Returns the call."""
    from . import native

    st = _Chunked(1, width, device)
    c = torch.zeros(1, width, dtype=torch.uint8, device=device)
    zero = torch.zeros(1, dtype=torch.int64, device=device)
    out = torch.empty(1, width, dtype=torch.uint8, device=device)
    return lambda: native.ext().band_chain_pairs(st.words, 0, 1, codec.PI_DOMAIN, 1, c, zero, zero, zero, zero, out,
                                                 (width - 32) // CHUNK)


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
    base = seg * per_segment
    whole_from = CHUNKED_MIN_ROWS[0] if CHAIN_PAIRS[0] else max(CHUNKED_MIN_ROWS[0], CHUNKED_SPLIT_MAX_ROWS[0])
    if CHUNKED[0] and CHUNKED_WHOLE[0] and r >= whole_from and width == 65_536 and c_flat.is_contiguous():
        DECODE_PATHS.append(("band_chain_pairs" if CHAIN_PAIRS[0] else "band_chain", r))
        return _decode_whole(c_flat, seg, blk, ln, tags, base, order, width)
    starts = torch.searchsorted(ln, torch.arange(int(ln[-1]) + 2, device=ln.device)).tolist()
    if CHUNKED[0] and r >= CHUNKED_MIN_ROWS[0]:
        DECODE_PATHS.append(("feistel_chunked per step", r))
        w = _decode_chunked(c_flat, seg, blk, ln, tags, base, starts, width) ^ c_flat[base + blk]
        out = torch.empty_like(w)
        out[order] = w
        return out
    DECODE_PATHS.append(("codec.permute per step", r))
    state = torch.zeros(r, 2 * width + 64, dtype=torch.uint8, device=c_flat.device)
    state[:, 2 * width:2 * width + 32] = tags[seg]
    state[:, 2 * width + 32] = (blk & 0xFF).to(torch.uint8)
    state[:, 2 * width + 33] = (blk >> 8).to(torch.uint8)
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
