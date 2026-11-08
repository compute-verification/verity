"""The deterministic `moe_align_block_size`'s Triton kernels (`moe_align.Align` launches them). No atomic decides where a
pair goes: a pair's slot is its expert's padded offset ``base[e]``, plus the expert's pairs before its tile of ``T``
pairs, plus its rank in the tile: the number of the tile's earlier pairs with its expert, from a compare against the
tile's pairs ``JB`` at a time. A pair's ``bin`` is its local expert, or ``E`` for a pair no expert takes. One program
places one tile, and the program that places an expert's last pair writes that expert's pad slots, ``PW`` at a time.

- `align_small`: one launch, a program a tile. Each program histograms all the pairs for the padded prefix sum, ``SB``
  at a time, adding the blocks before its tile to its offsets (and the one its tile starts in, masked).
- `count_chunks`, `scan_chunks`, `place_chunks`: chunks of ``CHUNK`` pairs, over one workspace of ``2 C + 1`` rows of
  ``EP``: each chunk's histogram (rows ``[0, C)``); then, over slices of the experts, each (chunk, expert)'s pairs in
  earlier chunks (rows ``[C, 2 C)``) and each expert's total (row ``2 C``); then a program a tile, which histograms its
  chunk's pairs before the tile, ``SB`` at a time.

Each writes every slot vLLM's kernel writes: a pair's index at its slot, ``numel`` at every other slot of
``sorted_token_ids``, each block's expert at ``expert_ids`` up to ``num_tokens_post_pad / BM`` and ``-1`` from there to
``cdiv(len(sorted_token_ids), BM)``; the first program writes ``num_tokens_post_pad`` and the slots past it. The slots
of different stores are disjoint, so no store waits on another.
"""
import triton
import triton.language as tl


@triton.jit
def _bins(ids_ptr, map_ptr, offs, P, E, HAS_MAP: tl.constexpr):
    """Each pair's local expert in [0, E), or E: an id outside [0, E) (truncated to int32 first, as vLLM's kernel reads it)
    or one the expert map sends to -1."""
    live = offs < P
    e = tl.load(ids_ptr + offs, mask=live, other=-1).to(tl.int32)
    ok = live & (e >= 0) & (e < E)
    if HAS_MAP:
        e = tl.load(map_ptr + e, mask=ok, other=-1).to(tl.int32)
        ok = ok & (e >= 0) & (e < E)
    return tl.where(ok, e, E)


@triton.jit
def _base(cnt, E, BM, EP: tl.constexpr):
    """Each expert's first slot and padded count (bins from E up count nothing)."""
    cnt = tl.where(tl.arange(0, EP) < E, cnt, 0)
    padded = (cnt + BM - 1) // BM * BM
    return tl.cumsum(padded, 0) - padded, padded, cnt


@triton.jit
def _counts(ids_ptr, map_ptr, start, lo, P, E, HAS_MAP: tl.constexpr, N: tl.constexpr, SB: tl.constexpr,
            EP: tl.constexpr, ALL: tl.constexpr):
    """Each bin's pairs in [start, start + N) (zeros unless ``ALL``) and in [start, lo), a histogram of ``SB`` pairs at a
    time; a block wholly at or past ``P`` (or, unless ``ALL``, past ``lo``) is skipped."""
    every = tl.zeros([EP], tl.int32)
    before = tl.zeros([EP], tl.int32)
    for k in tl.static_range(0, N, SB):
        a = start + k
        if ALL:
            if a < P:
                ia = a + tl.arange(0, SB)
                ea = _bins(ids_ptr, map_ptr, ia, P, E, HAS_MAP)
                h = tl.histogram(ea, EP)
                every += h
                if a + SB <= lo:
                    before += h
                elif a < lo:
                    before += tl.histogram(tl.where(ia < lo, ea, E), EP)
        else:
            if a < lo:
                ia = a + tl.arange(0, SB)
                before += tl.histogram(tl.where(ia < lo, _bins(ids_ptr, map_ptr, ia, P, E, HAS_MAP), E), EP)
    return every, before


@triton.jit
def _tail(sorted_ptr, eids_ptr, ntpp_ptr, padded, P, L, NB, BM, TB: tl.constexpr):
    """``num_tokens_post_pad``; ``numel`` from the total to ``L``; -1 at the blocks from ``total / BM`` to ``NB``."""
    total = tl.sum(padded, 0)
    tl.store(ntpp_ptr, total)
    q = tl.arange(0, TB)
    for s in range(total, L, TB):
        tl.store(sorted_ptr + s + q, tl.zeros_like(q) + P, mask=s + q < L)
    for s in range(total // BM, NB, TB):
        tl.store(eids_ptr + s + q, tl.full([TB], -1, tl.int32), mask=s + q < NB)


@triton.jit
def _tile(ids_ptr, map_ptr, sorted_ptr, eids_ptr, nxt, stop, end, lo, P, L, E, BM, HAS_MAP: tl.constexpr,
          T: tl.constexpr, JB: tl.constexpr, PW: tl.constexpr):
    """Store the pairs [lo, lo + T), and the pads of each expert whose last pair is among them: ``nxt[e]`` is the slot
    of expert e's first pair at ``lo``, ``stop[e]`` one past its last pair's and ``end[e]`` one past its last pad's."""
    i = tl.arange(0, T)
    eb = _bins(ids_ptr, map_ptr, lo + i, P, E, HAS_MAP)
    rank = tl.zeros([T], tl.int32)
    for jb in tl.static_range(0, T, JB):
        j = jb + tl.arange(0, JB)
        ej = _bins(ids_ptr, map_ptr, lo + j, P, E, HAS_MAP)
        rank += tl.sum(((ej[None, :] == eb[:, None]) & (j[None, :] < i[:, None])).to(tl.int32), 1)
    dest = tl.gather(nxt, eb, 0) + rank
    ok = (eb < E) & (dest < L)
    tl.store(sorted_ptr + dest, lo + i, mask=ok)
    tl.store(eids_ptr + dest // BM, eb, mask=ok & (dest % BM == 0))
    fin = tl.gather(stop, eb, 0)
    last = ok & (dest == fin - 1)
    end = tl.minimum(tl.gather(end, eb, 0), L)
    w = tl.arange(0, PW)
    for p0 in range(0, BM - 1, PW):
        s = fin[:, None] + p0 + w[None, :]
        tl.store(sorted_ptr + s, tl.zeros([T, PW], tl.int32) + P, mask=last[:, None] & (s < end[:, None]))


@triton.jit(do_not_specialize=["P", "L", "NB", "E", "BM"])
def align_small(ids_ptr, map_ptr, sorted_ptr, eids_ptr, ntpp_ptr, P, L, NB, E, BM, HAS_MAP: tl.constexpr,
                PB: tl.constexpr, T: tl.constexpr, JB: tl.constexpr, SB: tl.constexpr, EP: tl.constexpr,
                TB: tl.constexpr, PW: tl.constexpr):
    t = tl.program_id(0)
    every, before = _counts(ids_ptr, map_ptr, 0, t * T, P, E, HAS_MAP, PB, SB, EP, True)
    base, padded, cnt = _base(every, E, BM, EP)
    if t == 0:
        _tail(sorted_ptr, eids_ptr, ntpp_ptr, padded, P, L, NB, BM, TB)
    nxt = base + before
    _tile(ids_ptr, map_ptr, sorted_ptr, eids_ptr, nxt, base + cnt, base + padded, t * T, P, L, E, BM, HAS_MAP, T, JB,
          PW)


@triton.jit(do_not_specialize=["P", "E"])
def count_chunks(ids_ptr, map_ptr, ws_ptr, P, E, HAS_MAP: tl.constexpr, CHUNK: tl.constexpr, EP: tl.constexpr):
    c = tl.program_id(0)
    eb = _bins(ids_ptr, map_ptr, c * CHUNK + tl.arange(0, CHUNK), P, E, HAS_MAP)
    tl.store(ws_ptr + c * EP + tl.arange(0, EP), tl.histogram(eb, EP))


@triton.jit(do_not_specialize=["C"])
def scan_chunks(ws_ptr, C, EP: tl.constexpr, ES: tl.constexpr, CB: tl.constexpr):
    """For the experts [g * ES, g * ES + ES): ``prefix[c, e]``, e's pairs in chunks before c, and ``total[e]``."""
    hist_ptr, prefix_ptr, total_ptr = ws_ptr, ws_ptr + C * EP, ws_ptr + 2 * C * EP
    e = tl.program_id(0) * ES + tl.arange(0, ES)
    rows = tl.arange(0, CB)
    run = tl.zeros([ES], tl.int32)
    for c0 in range(0, C, CB):
        r = c0 + rows
        at = r[:, None] * EP + e[None, :]
        h = tl.load(hist_ptr + at, mask=(r < C)[:, None], other=0)
        tl.store(prefix_ptr + at, run[None, :] + tl.cumsum(h, 0) - h, mask=(r < C)[:, None])
        run += tl.sum(h, 0)
    tl.store(total_ptr + e, run)


@triton.jit(do_not_specialize=["P", "L", "NB", "E", "BM", "C"])
def place_chunks(ids_ptr, map_ptr, ws_ptr, sorted_ptr, eids_ptr, ntpp_ptr, P, L, NB, E, BM, C,
                 HAS_MAP: tl.constexpr, CHUNK: tl.constexpr, T: tl.constexpr, JB: tl.constexpr, SB: tl.constexpr,
                 EP: tl.constexpr, TB: tl.constexpr, PW: tl.constexpr):
    c = tl.program_id(0)
    t = tl.program_id(1)
    ex = tl.arange(0, EP)
    base, padded, cnt = _base(tl.load(ws_ptr + 2 * C * EP + ex), E, BM, EP)
    if c + t == 0:
        _tail(sorted_ptr, eids_ptr, ntpp_ptr, padded, P, L, NB, BM, TB)
    _, before = _counts(ids_ptr, map_ptr, c * CHUNK, c * CHUNK + t * T, P, E, HAS_MAP, CHUNK, SB, EP, False)
    nxt = base + tl.load(ws_ptr + (C + c) * EP + ex) + before
    _tile(ids_ptr, map_ptr, sorted_ptr, eids_ptr, nxt, base + cnt, base + padded, c * CHUNK + t * T, P, L, E, BM,
          HAS_MAP, T, JB, PW)
