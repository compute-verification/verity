"""The deterministic `moe_align_block_size`'s Triton kernels (`moe_align.Align` launches them). No atomic decides where a
pair goes: each pair's slot is its expert's padded offset plus its rank among that expert's pairs by pair index, from a
histogram, an exclusive prefix sum and a sort of the distinct keys ``bin * n + i`` (``bin`` a pair's local expert, ``E``
for a pair no expert takes, ``i`` its index within the sorted range of ``n`` pairs).

- `align_one`: one program, every pair of the call at once (``PB`` a power of two at least the pair count).
- `count_chunks`, `scan_chunks`, `place_chunks`: chunks of ``CH`` pairs; each chunk's histogram, then one program's scan
  over the chunks (each expert's offset, the pad slots, ``num_tokens_post_pad``, and each chunk's destination shift per
  expert), then each chunk's sort and placement.

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
def _pads(sorted_ptr, eids_ptr, ntpp_ptr, cnt, padded, base, total, P, L, NB, BM, TB: tl.constexpr):
    """``num_tokens_post_pad``; ``numel`` at each expert's pad slots and from ``total`` to ``L``; -1 at the blocks from
    ``total / BM`` to ``NB``."""
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
def _place(srt, n: tl.constexpr, shift, first, sorted_ptr, eids_ptr, L, E, BM):
    """Store the pairs of a sorted key range: key j (``bin * n + i``) goes to slot j + shift, as pair ``first + i``."""
    es = srt // n
    dest = tl.arange(0, n) + shift
    ok = (es < E) & (dest < L)
    tl.store(sorted_ptr + dest, first + srt - es * n, mask=ok)
    tl.store(eids_ptr + dest // BM, es, mask=ok & (dest % BM == 0))


@triton.jit(do_not_specialize=["P", "L", "NB", "E", "BM"])
def align_one(ids_ptr, map_ptr, sorted_ptr, eids_ptr, ntpp_ptr, P, L, NB, E, BM,
              HAS_MAP: tl.constexpr, PB: tl.constexpr, EP: tl.constexpr, TB: tl.constexpr):
    i = tl.arange(0, PB)
    eb = _bins(ids_ptr, map_ptr, i, P, E, HAS_MAP)
    ex = tl.arange(0, EP)
    cnt = tl.where(ex < E, tl.histogram(eb, EP), 0)
    padded = (cnt + BM - 1) // BM * BM
    base = tl.cumsum(padded, 0) - padded
    _pads(sorted_ptr, eids_ptr, ntpp_ptr, cnt, padded, base, tl.sum(padded, 0), P, L, NB, BM, TB)
    srt = tl.sort(eb * PB + i)
    shift = tl.gather(base - (tl.cumsum(cnt, 0) - cnt), srt // PB, 0)
    _place(srt, PB, shift, 0, sorted_ptr, eids_ptr, L, E, BM)


@triton.jit(do_not_specialize=["P", "E"])
def count_chunks(ids_ptr, map_ptr, hist_ptr, P, E, HAS_MAP: tl.constexpr, CH: tl.constexpr, EP: tl.constexpr):
    c = tl.program_id(0)
    eb = _bins(ids_ptr, map_ptr, c * CH + tl.arange(0, CH), P, E, HAS_MAP)
    tl.store(hist_ptr + c * EP + tl.arange(0, EP), tl.histogram(eb, EP))


@triton.jit(do_not_specialize=["C", "P", "L", "NB", "E", "BM"])
def scan_chunks(hist_ptr, shift_ptr, sorted_ptr, eids_ptr, ntpp_ptr, C, P, L, NB, E, BM,
                EP: tl.constexpr, CB: tl.constexpr, TB: tl.constexpr):
    ex = tl.arange(0, EP)
    rows = tl.arange(0, CB)
    cnt = tl.zeros([EP], tl.int32)
    for c0 in range(0, C, CB):
        r = c0 + rows
        cnt += tl.sum(tl.load(hist_ptr + r[:, None] * EP + ex[None, :], mask=(r < C)[:, None], other=0), 0)
    cnt = tl.where(ex < E, cnt, 0)
    padded = (cnt + BM - 1) // BM * BM
    base = tl.cumsum(padded, 0) - padded
    _pads(sorted_ptr, eids_ptr, ntpp_ptr, cnt, padded, base, tl.sum(padded, 0), P, L, NB, BM, TB)
    before = base
    for c0 in range(0, C, CB):
        r = c0 + rows
        h = tl.load(hist_ptr + r[:, None] * EP + ex[None, :], mask=(r < C)[:, None], other=0)
        h = tl.where(ex[None, :] < E, h, 0)
        # chunk r's pairs of expert e start at base[e] plus e's pairs in earlier chunks; in r's sorted keys they start
        # after r's pairs of smaller experts
        shift = before[None, :] + tl.cumsum(h, 0) - h - (tl.cumsum(h, 1) - h)
        tl.store(shift_ptr + r[:, None] * EP + ex[None, :], shift, mask=(r < C)[:, None])
        before += tl.sum(h, 0)


@triton.jit(do_not_specialize=["P", "L", "E", "BM"])
def place_chunks(ids_ptr, map_ptr, shift_ptr, sorted_ptr, eids_ptr, P, L, E, BM,
                 HAS_MAP: tl.constexpr, CH: tl.constexpr, EP: tl.constexpr):
    c = tl.program_id(0)
    i = tl.arange(0, CH)
    srt = tl.sort(_bins(ids_ptr, map_ptr, c * CH + i, P, E, HAS_MAP) * CH + i)
    shift = tl.load(shift_ptr + c * EP + srt // CH)
    _place(srt, CH, shift, c * CH, sorted_ptr, eids_ptr, L, E, BM)
