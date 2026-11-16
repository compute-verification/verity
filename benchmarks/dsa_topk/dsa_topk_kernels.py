"""The Triton kernels of `dsa_topk`'s device pass (its docstring has their contract), in a module of their own so
that `dsa_topk` imports without Triton and Triton resolves the helpers through this module's globals.

A block's two counts ride in one int32 (``gt + (eq << 16)``), so one prefix sum and one reduction a block serve both;
``BLOCK`` must stay below 2**15, and canon_row reads it as 32-key words. A row's counts accumulate per lane and
reduce once.

A split row (``count_seg``, ``emit_seg``, then under REFINE ``fix_find``, ``fix_count`` and ``fix_emit``) whose
candidate fails is refined when the k-th key is in the candidate's window: the keys whose top 13 bits (sign, exponent,
4 mantissa bits: the fast kernels' fp16 coarse bin, in which their candidate overflow drops keys) are the candidate's
or a neighbour's, for fp16 rounds a key just below a bin's edge into the bin. ``emit_seg``'s programs histogram
their segments' window keys by their offset in the window (`FINE` bins of 2**`LOW_BITS`) into the row's table,
``fix_find``'s find the fine bin of the k-th key and count their keys above it and in it by its last `LOW_BITS`,
``fix_count``'s read the k-th key from those counts and count their segments against it, and ``fix_emit``'s write
them at their ranks: three reads of a segment where bisection makes 34 of the row. Every table is all zeros between
calls. A failed candidate whose window misses the k-th key is bisected by one program. ws holds, a row: each
segment's (gt, eq), the candidate, each segment's keys above the window, the row's path (`REFINED`, `BISECTED`), its
fine bin and k-th key, and each segment's (gt, eq) against that key."""
import triton
import triton.language as tl
from triton.language.extra import libdevice

WIN = tl.constexpr(19)
FINE = tl.constexpr(2048)
LOW = tl.constexpr(1024)
LOW_BITS = tl.constexpr(10)
#: the window's keys: three 13-bit prefixes of 2**19 keys
SPAN = tl.constexpr(3 << 19)
#: a row's words of hist: its window table, its k-th key's fine bin's counts, and its keys above that bin
HROW = tl.constexpr(2048 + 1024 + 1)
REFINED = tl.constexpr(1)
BISECTED = tl.constexpr(2)


@triton.jit
def _ordered(x):
    b = x.to(tl.int32, bitcast=True)
    return tl.where(b < 0, b ^ 0x7FFFFFFF, b)


@triton.jit
def _bounds(r, A, B, ncols, next_n, MODE: tl.constexpr):
    # MODE 0: keys [A[r], B[r]); 1: [0, clamp(A[r], 0, ncols)); 2: top_k_per_row_decode's 1-D lengths per next_n
    if MODE == 0:
        lo = tl.load(A + r)
        n = tl.load(B + r) - lo
    elif MODE == 1:
        lo = r * 0
        n = tl.minimum(tl.maximum(tl.load(A + r), 0), ncols)
    else:
        lo = r * 0
        n = tl.minimum(tl.maximum(tl.load(A + r // next_n) - next_n + r % next_n + 1, 0), ncols)
    return lo, n


@triton.jit
def _hint(row, orow, n, ls1, K: tl.constexpr, KP2: tl.constexpr):
    # the fast kernel's selection's smallest key (the candidate k-th key) and how many of its keys exceed it
    offs = tl.arange(0, KP2)
    sel = tl.load(orow + offs, mask=offs < K, other=-1)
    ok = (sel >= 0) & (sel < n)
    x = tl.load(row + tl.where(ok, sel, 0).to(tl.int64) * ls1, mask=ok, other=0.0)
    key = tl.where(ok, _ordered(x), 2147483647)
    tau = tl.min(key, axis=0)
    return tau, tl.sum((ok & (key > tau)).to(tl.int32), axis=0)


@triton.jit
def _count(row, start, end, tau, ls1, BLOCK: tl.constexpr):
    c_gt = tl.zeros([BLOCK], dtype=tl.int32)
    c_eq = tl.zeros([BLOCK], dtype=tl.int32)
    for b in range(start, end, BLOCK):
        idx = b + tl.arange(0, BLOCK)
        m = idx < end
        key = _ordered(tl.load(row + idx.to(tl.int64) * ls1, mask=m, other=0.0))
        c_gt += (m & (key > tau)).to(tl.int32)
        c_eq += (m & (key == tau)).to(tl.int32)
    return tl.sum(c_gt, axis=0), tl.sum(c_eq, axis=0)


@triton.jit
def _emit(row, orow, start, end, tau, need, gt_before, eq_before, ls1, BLOCK: tl.constexpr):
    # keys [start, end) of the row in ascending order: every key above tau, and the row's first `need` keys equal to
    # it; each at its rank among those kept (gt_before, eq_before: the counts before start)
    gt_before = gt_before + tau * 0
    eq_before = eq_before + tau * 0
    for b in range(start, end, BLOCK):
        idx = b + tl.arange(0, BLOCK)
        m = idx < end
        key = _ordered(tl.load(row + idx.to(tl.int64) * ls1, mask=m, other=0.0))
        gt = (m & (key > tau)).to(tl.int32)
        eq = (m & (key == tau)).to(tl.int32)
        both = gt + (eq << 16)
        p = tl.cumsum(both, axis=0) - both
        gt_ex = gt_before + (p & 0xFFFF)
        eq_ex = eq_before + (p >> 16)
        keep = (gt > 0) | ((eq > 0) & (eq_ex < need))
        tl.store(orow + gt_ex + tl.minimum(eq_ex, need), idx, mask=keep)
        tot = tl.sum(both, axis=0)
        gt_before += tot & 0xFFFF
        eq_before += tot >> 16


@triton.jit
def _bisect(row, orow, n, ls1, K: tl.constexpr, BLOCK: tl.constexpr):
    # the k-th largest key by bisection over all 2^32 keys, then the row emitted from it
    lo_t = (n * 0).to(tl.int64) - 2147483648
    hi_t = lo_t + 4294967295
    for _ in range(32):
        mid = (lo_t + hi_t + 1) >> 1
        cnt = tl.zeros([BLOCK], dtype=tl.int32)
        for b in range(0, n, BLOCK):
            idx = b + tl.arange(0, BLOCK)
            m = idx < n
            key = _ordered(tl.load(row + idx.to(tl.int64) * ls1, mask=m, other=0.0)).to(tl.int64)
            cnt += (m & (key >= mid)).to(tl.int32)
        ge = tl.sum(cnt, axis=0) >= K
        lo_t = tl.where(ge, mid, lo_t)
        hi_t = tl.where(ge, hi_t, mid - 1)
    tau = lo_t.to(tl.int32)
    n_gt, n_eq = _count(row, 0, n, tau, ls1, BLOCK)
    _emit(row, orow, 0, n, tau, K - n_gt, n * 0, n * 0, ls1, BLOCK)
    return n_eq > K - n_gt


@triton.jit
def _popc(x, HW: tl.constexpr):
    # HW: the device's popcount (Triton's interpreter has none)
    if HW:
        return libdevice.popc(x.to(tl.int32, bitcast=True))
    u = x.to(tl.uint32, bitcast=True)
    u = u - ((u >> 1) & 0x55555555)
    u = (u & 0x33333333) + ((u >> 2) & 0x33333333)
    u = (u + (u >> 4)) & 0x0F0F0F0F
    return ((u * 0x01010101) >> 24).to(tl.int32)


@triton.jit
def _below(counts, x):
    # the bins wholly below the one where a count from the top reaches the k-th key: x is the keys under it, so the
    # bin index of the k-th key
    return tl.sum((tl.cumsum(counts, axis=0) <= x).to(tl.int32), axis=0)


@triton.jit
def _base(tau):
    # the first key of the candidate's window: the 13-bit prefix below the candidate's (`WIN`)
    return ((tau >> WIN).to(tl.int64) - 1) << WIN


@triton.jit
def _row_ws(ws, r, SEG):
    return ws + r * (5 * SEG + 4)


@triton.jit
def canon_row(logits, ls0, ls1, out, os0, A, B, ncols, next_n, stats,
              K: tl.constexpr, KP2: tl.constexpr, BLOCK: tl.constexpr, MODE: tl.constexpr, HW: tl.constexpr):
    # one program a row, one read: the keys at or above the fast kernel's smallest selected key, in ascending order,
    # each at its rank (the words before it, then its word's bits below it: no scan over keys); then the row's
    # counts decide: no tie across the k-th key leaves the row as written, a tie has it written again from the
    # counts, and a candidate that isn't the k-th key has it bisected
    r = tl.program_id(0)
    lo, n = _bounds(r, A, B, ncols, next_n, MODE)
    if n > K:
        row = logits + r.to(tl.int64) * ls0 + lo.to(tl.int64) * ls1
        orow = out + r.to(tl.int64) * os0
        tau, _gt = _hint(row, orow, n, ls1, K, KP2)
        W: tl.constexpr = BLOCK // 32
        lane = tl.arange(0, 32).to(tl.uint32)[None, :]
        below = (tl.full([1, 32], 1, tl.uint32) << lane) - 1
        c_gt = tl.zeros([W, 32], dtype=tl.int32)
        c_eq = tl.zeros([W, 32], dtype=tl.int32)
        kept = n * 0
        for b in range(0, n, BLOCK):
            idx = b + tl.arange(0, W)[:, None] * 32 + tl.arange(0, 32)[None, :]
            m = idx < n
            key = _ordered(tl.load(row + idx.to(tl.int64) * ls1, mask=m, other=0.0))
            gt = m & (key > tau)
            eq = m & (key == tau)
            c_gt += gt.to(tl.int32)
            c_eq += eq.to(tl.int32)
            ge = (gt | eq).to(tl.uint32)
            words = tl.sum(ge << lane, axis=1)
            cnt = _popc(words, HW)
            pos = kept + (tl.cumsum(cnt, axis=0) - cnt)[:, None] + _popc(words[:, None] & below, HW)
            tl.store(orow + pos, idx, mask=(ge != 0) & (pos < K))
            kept += tl.sum(cnt, axis=0)
        n_gt = tl.sum(tl.sum(c_gt, axis=1), axis=0)
        n_eq = tl.sum(tl.sum(c_eq, axis=1), axis=0)
        tl.atomic_add(stats + 0, 1)
        tl.debug_barrier()
        if (n_gt < K) & (n_gt + n_eq >= K):
            if n_gt + n_eq > K:
                tl.atomic_add(stats + 1, 1)
                _emit(row, orow, 0, n, tau, K - n_gt, n * 0, n * 0, ls1, BLOCK)
        else:
            tl.atomic_add(stats + 2, 1)
            if _bisect(row, orow, n, ls1, K, BLOCK):
                tl.atomic_add(stats + 1, 1)


@triton.jit
def count_seg(logits, ls0, ls1, out, os0, A, B, ncols, next_n, ws, SEG,
              K: tl.constexpr, KP2: tl.constexpr, BLOCK: tl.constexpr, MODE: tl.constexpr):
    r = tl.program_id(0)
    s = tl.program_id(1)
    lo, n = _bounds(r, A, B, ncols, next_n, MODE)
    if n > K:
        row = logits + r.to(tl.int64) * ls0 + lo.to(tl.int64) * ls1
        tau, _gt = _hint(row, out + r.to(tl.int64) * os0, n, ls1, K, KP2)
        seg = tl.cdiv(tl.cdiv(n, SEG), BLOCK) * BLOCK
        start = s * seg
        c_gt, c_eq = _count(row, start, tl.minimum(start + seg, n), tau, ls1, BLOCK)
        w = _row_ws(ws, r, SEG)
        tl.store(w + 2 * s, c_gt)
        tl.store(w + 2 * s + 1, c_eq)
        if s == 0:
            tl.store(w + 2 * SEG, tau)


@triton.jit
def _counts(w, SEG, SEGP2: tl.constexpr):
    # each segment's counts against the row's candidate, and the row's
    segs = tl.arange(0, SEGP2)
    sm = segs < SEG
    gts = tl.load(w + 2 * segs, mask=sm, other=0)
    eqs = tl.load(w + 2 * segs + 1, mask=sm, other=0)
    n_gt = tl.sum(gts, axis=0)
    n_eq = tl.sum(eqs, axis=0)
    return gts, eqs, n_gt, n_eq


@triton.jit
def emit_seg(logits, ls0, ls1, out, os0, A, B, ncols, next_n, ws, hist, stats, SEG,
             K: tl.constexpr, BLOCK: tl.constexpr, MODE: tl.constexpr, SEGP2: tl.constexpr, REFINE: tl.constexpr):
    # each segment emits its keys at their ranks; the row's totals and the counts before the segment come from
    # count_seg's, so the candidate is checked before any write. A failed candidate is bisected by one program, or
    # under REFINE its segments fill the row's table and count their keys above the window for fix_find
    r = tl.program_id(0)
    s = tl.program_id(1)
    lo, n = _bounds(r, A, B, ncols, next_n, MODE)
    if n > K:
        row = logits + r.to(tl.int64) * ls0 + lo.to(tl.int64) * ls1
        orow = out + r.to(tl.int64) * os0
        w = _row_ws(ws, r, SEG)
        tau = tl.load(w + 2 * SEG)
        segs = tl.arange(0, SEGP2)
        gts, eqs, n_gt, n_eq = _counts(w, SEG, SEGP2)
        seg = tl.cdiv(tl.cdiv(n, SEG), BLOCK) * BLOCK
        start = s * seg
        end = tl.minimum(start + seg, n)
        if (n_gt < K) & (n_gt + n_eq >= K):
            _emit(row, orow, start, end, tau, K - n_gt,
                  tl.sum(tl.where(segs < s, gts, 0), axis=0), tl.sum(tl.where(segs < s, eqs, 0), axis=0),
                  ls1, BLOCK)
            if s == 0:
                tl.atomic_add(stats + 0, 1)
                if n_gt + n_eq > K:
                    tl.atomic_add(stats + 1, 1)
        elif REFINE:
            base = _base(tau)
            h = hist + r.to(tl.int64) * HROW
            c_up = tl.zeros([BLOCK], dtype=tl.int32)
            for b in range(start, end, BLOCK):
                idx = b + tl.arange(0, BLOCK)
                m = idx < end
                rel = _ordered(tl.load(row + idx.to(tl.int64) * ls1, mask=m, other=0.0)).to(tl.int64) - base
                c_up += (m & (rel >= SPAN)).to(tl.int32)
                tl.atomic_add(h + (rel >> LOW_BITS), tl.full([BLOCK], 1, tl.int32),
                              mask=m & (rel >= 0) & (rel < SPAN))
            tl.store(w + 2 * SEG + 1 + s, tl.sum(c_up, axis=0))
        elif s == 0:
            tl.atomic_add(stats + 0, 1)
            tl.atomic_add(stats + 2, 1)
            if _bisect(row, orow, n, ls1, K, BLOCK):
                tl.atomic_add(stats + 1, 1)


@triton.jit
def _zero_slice(h, s, SEG, N: tl.constexpr):
    # this segment's share of the N words at h back to zeros
    per = tl.cdiv(N, SEG)
    offs = s * per + tl.arange(0, N)
    tl.store(h + offs, tl.zeros([N], dtype=tl.int32), mask=offs < tl.minimum(s * per + per, N))


@triton.jit
def _failed(w, SEG, K: tl.constexpr, SEGP2: tl.constexpr):
    _gts, _eqs, n_gt, n_eq = _counts(w, SEG, SEGP2)
    return (n_gt >= K) | (n_gt + n_eq < K)


@triton.jit
def fix_find(logits, ls0, ls1, out, os0, A, B, ncols, next_n, ws, hist, stats, SEG,
             K: tl.constexpr, BLOCK: tl.constexpr, MODE: tl.constexpr, SEGP2: tl.constexpr):
    # a failed row whose window holds the k-th key: the fine bin that holds it, and each segment's keys above that
    # bin and in it by their last bits, into the row's counts; any other failed row bisected by one program
    r = tl.program_id(0)
    s = tl.program_id(1)
    lo, n = _bounds(r, A, B, ncols, next_n, MODE)
    if n > K:
        w = _row_ws(ws, r, SEG)
        if _failed(w, SEG, K, SEGP2):
            row = logits + r.to(tl.int64) * ls0 + lo.to(tl.int64) * ls1
            h = hist + r.to(tl.int64) * HROW
            segs = tl.arange(0, SEGP2)
            up = tl.sum(tl.load(w + 2 * SEG + 1 + segs, mask=segs < SEG, other=0), axis=0)
            fine = tl.load(h + tl.arange(0, FINE))
            t = tl.sum(fine, axis=0)
            if (up < K) & (up + t >= K):
                base = _base(tl.load(w + 2 * SEG))
                fb = _below(fine, up + t - K)
                seg = tl.cdiv(tl.cdiv(n, SEG), BLOCK) * BLOCK
                start = s * seg
                end = tl.minimum(start + seg, n)
                c_up = tl.zeros([BLOCK], dtype=tl.int32)
                for b in range(start, end, BLOCK):
                    idx = b + tl.arange(0, BLOCK)
                    m = idx < end
                    rel = _ordered(tl.load(row + idx.to(tl.int64) * ls1, mask=m, other=0.0)).to(tl.int64) - base
                    c_up += (m & ((rel >> LOW_BITS) > fb)).to(tl.int32)
                    tl.atomic_add(h + FINE + (rel & (LOW - 1)), tl.full([BLOCK], 1, tl.int32),
                                  mask=m & ((rel >> LOW_BITS) == fb))
                tl.atomic_add(h + FINE + LOW, tl.sum(c_up, axis=0))
                if s == 0:
                    tl.store(w + 3 * SEG + 1, tl.full([], REFINED, tl.int32))
                    tl.store(w + 3 * SEG + 2, fb)
            elif s == 0:
                tl.store(w + 3 * SEG + 1, tl.full([], BISECTED, tl.int32))
                tl.atomic_add(stats + 0, 1)
                tl.atomic_add(stats + 2, 1)
                if _bisect(row, out + r.to(tl.int64) * os0, n, ls1, K, BLOCK):
                    tl.atomic_add(stats + 1, 1)


@triton.jit
def fix_count(logits, ls0, ls1, out, os0, A, B, ncols, next_n, ws, hist, stats, SEG,
              K: tl.constexpr, BLOCK: tl.constexpr, MODE: tl.constexpr, SEGP2: tl.constexpr):
    # a refined row: the k-th key from the row's counts, and each segment's keys above and at it; every failed
    # row's window table back to zeros
    r = tl.program_id(0)
    s = tl.program_id(1)
    lo, n = _bounds(r, A, B, ncols, next_n, MODE)
    if n > K:
        w = _row_ws(ws, r, SEG)
        if _failed(w, SEG, K, SEGP2):
            h = hist + r.to(tl.int64) * HROW
            _zero_slice(h, s, SEG, FINE)
            if tl.load(w + 3 * SEG + 1) == REFINED:
                row = logits + r.to(tl.int64) * ls0 + lo.to(tl.int64) * ls1
                lows = tl.load(h + FINE + tl.arange(0, LOW))
                j = _below(lows, tl.load(h + FINE + LOW) + tl.sum(lows, axis=0) - K)
                kth = (_base(tl.load(w + 2 * SEG)) + (tl.load(w + 3 * SEG + 2).to(tl.int64) << LOW_BITS) + j)
                kth = kth.to(tl.int32)
                seg = tl.cdiv(tl.cdiv(n, SEG), BLOCK) * BLOCK
                start = s * seg
                c_gt, c_eq = _count(row, start, tl.minimum(start + seg, n), kth, ls1, BLOCK)
                tl.store(w + 3 * SEG + 4 + 2 * s, c_gt)
                tl.store(w + 3 * SEG + 5 + 2 * s, c_eq)
                if s == 0:
                    tl.store(w + 3 * SEG + 3, kth)


@triton.jit
def fix_emit(logits, ls0, ls1, out, os0, A, B, ncols, next_n, ws, hist, stats, SEG,
             K: tl.constexpr, BLOCK: tl.constexpr, MODE: tl.constexpr, SEGP2: tl.constexpr):
    # a refined row: each segment's keys at their ranks from the k-th key, as emit_seg's; the row's counts back to
    # zeros
    r = tl.program_id(0)
    s = tl.program_id(1)
    lo, n = _bounds(r, A, B, ncols, next_n, MODE)
    if n > K:
        w = _row_ws(ws, r, SEG)
        if _failed(w, SEG, K, SEGP2):
            if tl.load(w + 3 * SEG + 1) == REFINED:
                h = hist + r.to(tl.int64) * HROW
                _zero_slice(h + FINE, s, SEG, LOW)
                row = logits + r.to(tl.int64) * ls0 + lo.to(tl.int64) * ls1
                orow = out + r.to(tl.int64) * os0
                segs = tl.arange(0, SEGP2)
                sm = segs < SEG
                gts = tl.load(w + 3 * SEG + 4 + 2 * segs, mask=sm, other=0)
                eqs = tl.load(w + 3 * SEG + 5 + 2 * segs, mask=sm, other=0)
                n_gt = tl.sum(gts, axis=0)
                seg = tl.cdiv(tl.cdiv(n, SEG), BLOCK) * BLOCK
                start = s * seg
                _emit(row, orow, start, tl.minimum(start + seg, n), tl.load(w + 3 * SEG + 3), K - n_gt,
                      tl.sum(tl.where(segs < s, gts, 0), axis=0), tl.sum(tl.where(segs < s, eqs, 0), axis=0),
                      ls1, BLOCK)
                if s == 0:
                    tl.store(h + FINE + LOW, 0)
                    tl.atomic_add(stats + 0, 1)
                    tl.atomic_add(stats + 3, 1)
                    if n_gt + tl.sum(eqs, axis=0) > K:
                        tl.atomic_add(stats + 1, 1)
