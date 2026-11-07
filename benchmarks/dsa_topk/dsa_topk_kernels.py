"""The Triton kernels of `dsa_topk`'s device pass (its docstring has their contract), in a module of their own so
that `dsa_topk` imports without Triton and Triton resolves the helpers through this module's globals.

A block's two counts ride in one int32 (``gt + (eq << 16)``), so one prefix sum and one reduction a block serve both;
``BLOCK`` must stay below 2**15. A row's counts accumulate per lane and reduce once."""
import triton
import triton.language as tl


@triton.jit
def _ordered(x):
    b = x.to(tl.int32, bitcast=True)
    mag = b & 0x7FFFFFFF
    b = tl.where(mag > 0x7F800000, -8388608, b)
    b = tl.where(mag == 0, 0, b)
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
def _emit(row, orow, start, end, tau, skip, gt_before, eq_before, ls1, BLOCK: tl.constexpr):
    # keys [start, end) of the row in ascending order: every key above tau, and the tied keys after the first
    # `skip` of the row's; each at its rank among those kept (gt_before, eq_before: the counts before start)
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
        keep = (gt > 0) | ((eq > 0) & (eq_ex >= skip))
        tl.store(orow + gt_ex + tl.maximum(eq_ex - skip, 0), idx, mask=keep)
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
    _emit(row, orow, 0, n, tau, n_eq - (K - n_gt), n * 0, n * 0, ls1, BLOCK)
    return n_eq > K - n_gt


@triton.jit
def canon_one(logits, ls0, ls1, out, os0, A, B, ncols, next_n, stats,
              K: tl.constexpr, KP2: tl.constexpr, BLOCK: tl.constexpr, MODE: tl.constexpr):
    # one program a row: one pass from the row's end that keeps every key above the candidate and the m tied keys
    # with the largest indices (m from the fast kernel's selection), checked against the row's counts once done
    r = tl.program_id(0)
    lo, n = _bounds(r, A, B, ncols, next_n, MODE)
    if n > K:
        row = logits + r.to(tl.int64) * ls0 + lo.to(tl.int64) * ls1
        orow = out + r.to(tl.int64) * os0
        tau, gt_hint = _hint(row, orow, n, ls1, K, KP2)
        m_eq = K - gt_hint
        kept_after = n * 0
        eq_after = n * 0
        n_gt = n * 0
        nblocks = tl.cdiv(n, BLOCK)
        for bi in range(nblocks):
            idx = (nblocks - 1 - bi) * BLOCK + tl.arange(0, BLOCK)
            m = idx < n
            key = _ordered(tl.load(row + idx.to(tl.int64) * ls1, mask=m, other=0.0))
            gt = (m & (key > tau)).to(tl.int32)
            eq = (m & (key == tau)).to(tl.int32)
            both = gt + (eq << 16)
            p = tl.cumsum(both, axis=0)
            tot = tl.sum(both, axis=0)
            gt_tot = tot & 0xFFFF
            eq_tot = tot >> 16
            gt_later = gt_tot - (p & 0xFFFF)
            eq_later = eq_tot - (p >> 16)
            quota = tl.maximum(m_eq - eq_after, 0)
            keep = (gt > 0) | ((eq > 0) & (eq_later < quota))
            pos = K - 1 - (kept_after + gt_later + tl.minimum(eq_later, quota))
            tl.store(orow + pos, idx, mask=keep & (pos >= 0))
            kept_after += gt_tot + tl.minimum(eq_tot, quota)
            eq_after += eq_tot
            n_gt += gt_tot
        tl.atomic_add(stats + 0, 1)
        if (n_gt == gt_hint) & (m_eq >= 1) & (eq_after >= m_eq):
            if eq_after > m_eq:
                tl.atomic_add(stats + 1, 1)
        else:
            tl.atomic_add(stats + 2, 1)
            if _bisect(row, orow, n, ls1, K, BLOCK):
                tl.atomic_add(stats + 1, 1)


@triton.jit
def count_seg(logits, ls0, ls1, out, os0, A, B, ncols, next_n, ws, SEG,
              K: tl.constexpr, KP2: tl.constexpr, BLOCK: tl.constexpr, MODE: tl.constexpr):
    # ws per row: SEG (gt, eq) pairs, then the candidate
    r = tl.program_id(0)
    s = tl.program_id(1)
    lo, n = _bounds(r, A, B, ncols, next_n, MODE)
    if n > K:
        row = logits + r.to(tl.int64) * ls0 + lo.to(tl.int64) * ls1
        tau, _gt = _hint(row, out + r.to(tl.int64) * os0, n, ls1, K, KP2)
        seg = tl.cdiv(tl.cdiv(n, SEG), BLOCK) * BLOCK
        start = s * seg
        c_gt, c_eq = _count(row, start, tl.minimum(start + seg, n), tau, ls1, BLOCK)
        w = ws + r * (2 * SEG + 1)
        tl.store(w + 2 * s, c_gt)
        tl.store(w + 2 * s + 1, c_eq)
        if s == 0:
            tl.store(w + 2 * SEG, tau)


@triton.jit
def emit_seg(logits, ls0, ls1, out, os0, A, B, ncols, next_n, ws, stats, SEG,
             K: tl.constexpr, BLOCK: tl.constexpr, MODE: tl.constexpr, SEGP2: tl.constexpr):
    # each segment emits its keys at their ranks; the row's totals and the counts before the segment come from
    # count_seg's, so the candidate is checked before any write
    r = tl.program_id(0)
    s = tl.program_id(1)
    lo, n = _bounds(r, A, B, ncols, next_n, MODE)
    if n > K:
        row = logits + r.to(tl.int64) * ls0 + lo.to(tl.int64) * ls1
        orow = out + r.to(tl.int64) * os0
        w = ws + r * (2 * SEG + 1)
        tau = tl.load(w + 2 * SEG)
        segs = tl.arange(0, SEGP2)
        sm = segs < SEG
        gts = tl.load(w + 2 * segs, mask=sm, other=0)
        eqs = tl.load(w + 2 * segs + 1, mask=sm, other=0)
        n_gt = tl.sum(gts, axis=0)
        n_eq = tl.sum(eqs, axis=0)
        if (n_gt < K) & (n_gt + n_eq >= K):
            seg = tl.cdiv(tl.cdiv(n, SEG), BLOCK) * BLOCK
            start = s * seg
            _emit(row, orow, start, tl.minimum(start + seg, n), tau, n_eq - (K - n_gt),
                  tl.sum(tl.where(segs < s, gts, 0), axis=0), tl.sum(tl.where(segs < s, eqs, 0), axis=0),
                  ls1, BLOCK)
            if s == 0:
                tl.atomic_add(stats + 0, 1)
                if n_gt + n_eq > K:
                    tl.atomic_add(stats + 1, 1)
        elif s == 0:
            tl.atomic_add(stats + 0, 1)
            tl.atomic_add(stats + 2, 1)
            if _bisect(row, orow, n, ls1, K, BLOCK):
                tl.atomic_add(stats + 1, 1)
