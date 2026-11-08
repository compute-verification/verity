"""The deterministic `moe_align_block_size`'s Triton kernels (`moe_align.Align` launches them). No atomic decides where a
pair goes: a pair's slot is its expert's padded offset ``base[e]``, plus the expert's pairs in earlier chunks, plus its
rank in its own chunk: the number of the chunk's earlier pairs with its expert, from a compare against the chunk's pairs
``JB`` at a time. A pair's ``bin`` is its local expert, or ``E`` for a pair no expert takes. Each program loads its pairs
once and ranks them all at once.

- `align_one`: one program over the whole call (``PB`` a power of two at least the pair count): the histogram, the
  padded prefix sum, the pad slots, and each pair's slot.
- `count_chunks`, `scan_chunks`, `place_chunks`: chunks of ``CHUNK`` pairs, one program each. Each chunk's histogram;
  then, over slices of the experts, each (chunk, expert)'s pairs in earlier chunks and each expert's total; then each
  chunk's pairs, the first chunk's program also writing the pad slots.

Each writes every slot vLLM's kernel writes: a pair's index at its slot, ``numel`` at every other slot of
``sorted_token_ids``, each block's expert at ``expert_ids`` up to ``num_tokens_post_pad / BM`` and ``-1`` from there to
``cdiv(len(sorted_token_ids), BM)``. The pad slots and the pairs' slots are disjoint, so no store waits on another.
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
def _pads(sorted_ptr, eids_ptr, ntpp_ptr, cnt, padded, base, P, L, NB, BM, TB: tl.constexpr):
    """``num_tokens_post_pad``; ``numel`` at each expert's pad slots and from the total to ``L``; -1 at the blocks from
    ``total / BM`` to ``NB``."""
    total = tl.sum(padded, 0)
    tl.store(ntpp_ptr, total)
    fill = tl.zeros_like(cnt) + P
    for t in range(0, BM - 1):
        tl.store(sorted_ptr + base + cnt + t, fill, mask=cnt + t < padded)
    q = tl.arange(0, TB)
    for s in range(total, L, TB):
        tl.store(sorted_ptr + s + q, tl.zeros_like(q) + P, mask=s + q < L)
    for s in range(total // BM, NB, TB):
        tl.store(eids_ptr + s + q, tl.full([TB], -1, tl.int32), mask=s + q < NB)


@triton.jit
def _rank(ids_ptr, map_ptr, eb, lo, P, E, HAS_MAP: tl.constexpr, PB: tl.constexpr, JB: tl.constexpr):
    """Each pair's rank among the pairs [lo, lo + PB) with its bin: those before it."""
    i = tl.arange(0, PB)
    rank = tl.zeros([PB], tl.int32)
    for jb in tl.static_range(0, PB, JB):
        j = jb + tl.arange(0, JB)
        ej = _bins(ids_ptr, map_ptr, lo + j, P, E, HAS_MAP)
        rank += tl.sum(((ej[None, :] == eb[:, None]) & (j[None, :] < i[:, None])).to(tl.int32), 1)
    return rank


@triton.jit
def _put(eb, dest, pair, sorted_ptr, eids_ptr, L, E, BM):
    ok = (eb < E) & (dest < L)
    tl.store(sorted_ptr + dest, pair, mask=ok)
    tl.store(eids_ptr + dest // BM, eb, mask=ok & (dest % BM == 0))


@triton.jit(do_not_specialize=["P", "L", "NB", "E", "BM"])
def align_one(ids_ptr, map_ptr, sorted_ptr, eids_ptr, ntpp_ptr, P, L, NB, E, BM,
              HAS_MAP: tl.constexpr, PB: tl.constexpr, JB: tl.constexpr, EP: tl.constexpr, TB: tl.constexpr):
    i = tl.arange(0, PB)
    eb = _bins(ids_ptr, map_ptr, i, P, E, HAS_MAP)
    base, padded, cnt = _base(tl.histogram(eb, EP), E, BM, EP)
    _pads(sorted_ptr, eids_ptr, ntpp_ptr, cnt, padded, base, P, L, NB, BM, TB)
    dest = tl.gather(base, eb, 0) + _rank(ids_ptr, map_ptr, eb, 0, P, E, HAS_MAP, PB, JB)
    _put(eb, dest, i, sorted_ptr, eids_ptr, L, E, BM)


@triton.jit(do_not_specialize=["P", "E"])
def count_chunks(ids_ptr, map_ptr, hist_ptr, P, E, HAS_MAP: tl.constexpr, CHUNK: tl.constexpr, EP: tl.constexpr):
    c = tl.program_id(0)
    eb = _bins(ids_ptr, map_ptr, c * CHUNK + tl.arange(0, CHUNK), P, E, HAS_MAP)
    tl.store(hist_ptr + c * EP + tl.arange(0, EP), tl.histogram(eb, EP))


@triton.jit(do_not_specialize=["C"])
def scan_chunks(hist_ptr, prefix_ptr, total_ptr, C, EP: tl.constexpr, ES: tl.constexpr, CB: tl.constexpr):
    """For the experts [g * ES, g * ES + ES): ``prefix[c, e]``, e's pairs in chunks before c, and ``total[e]``."""
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


@triton.jit(do_not_specialize=["P", "L", "NB", "E", "BM"])
def place_chunks(ids_ptr, map_ptr, prefix_ptr, total_ptr, sorted_ptr, eids_ptr, ntpp_ptr, P, L, NB, E, BM,
                 HAS_MAP: tl.constexpr, CHUNK: tl.constexpr, JB: tl.constexpr, EP: tl.constexpr, TB: tl.constexpr):
    c = tl.program_id(0)
    ex = tl.arange(0, EP)
    base, padded, cnt = _base(tl.load(total_ptr + ex), E, BM, EP)
    if c == 0:
        _pads(sorted_ptr, eids_ptr, ntpp_ptr, cnt, padded, base, P, L, NB, BM, TB)
    lo = c * CHUNK
    i = tl.arange(0, CHUNK)
    eb = _bins(ids_ptr, map_ptr, lo + i, P, E, HAS_MAP)
    nxt = base + tl.load(prefix_ptr + c * EP + ex)
    dest = tl.gather(nxt, eb, 0) + _rank(ids_ptr, map_ptr, eb, lo, P, E, HAS_MAP, CHUNK, JB)
    _put(eb, dest, lo + i, sorted_ptr, eids_ptr, L, E, BM)
