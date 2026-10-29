"""Sparse commitments: what makes the uncommitted k32 steps of a dot product unskippable, and what each mechanism costs.

A design commits fewer accumulator words than one a k32 step and still wants every step credited. Under DistinctWrites a
committed word is credited one register write (32 units, H32); anything more needs the uncommitted words forced.

Checks (exact integer arithmetic, small shapes, seeded):
  served_sparse  the lead's layout (28 data lanes a step, 3 coin lanes against public weights per 64-row tile, step and
                 column, one row coin bit) at a committed word every L lanes, zero data: an adversary writes each
                 committed word with one IADD3 from a row prefix and a column prefix, bit for bit, and runs no GEMM;
                 with coins drawn after each committed word (chained rounds) it still does, since a word reads only
                 coins already drawn. The middle-segment skip needs the total, which reads coins not yet drawn.
  lane_floor     over every q noise lanes a step, L and noise alphabet (s8, s4, b1), the zero-data packing count's
                 gap; the fewest committed words a data MAC at which it stays within T1.
  exact_chain    ncp3 with its coins drawn block by block after each block's words: the skip still runs inside the
                 blocks whose cancelling coins were drawn before them, bit for bit.
  checkpoint     positions drawn after the sparse words are committed: an adversary that kept nothing recomputes each
                 drawn word from the committed word before it, bit for bit, in at most L/32 - 1 k32 steps at random
                 data and one write at zero data; only answered words gain credit.
  folds          every k32 word folded into one state a dot product, the state committed once:
                   linear  N <- N k + D: at zero data the state is a row term plus a column term, one write an output
                   xor     N <- N ^ D: one LOP3 folds two words, 1.5 writes a step against the honest 2
                   keyed   N <- N ^ (D & K_t), K_t a live u32 a step and call: the sparse and skip adversaries' states
                           differ; the cheapest word-grain adversary writes every word and folds it, 2 writes a step
  bitslice_xor   the keyed xor fold at zero data, 32 outputs of a row a register a bit: the carry chain of row prefix
                 plus column prefix and paired folds of the carries, bit for bit, in fewer writes a step than the
                 honest 2
  mulfold        N <- (N ^ D_t) k_t, k_t a live odd u32: each step a permutation; the word-grain adversary matches at
                 the honest 3 writes a step; sparse and skip adversaries miss; no row-plus-column form over Z or
                 GF(2); a bit-sliced multiply by k_t costs several LOP3s a step
  row_sharing    zero data: rows of a tile with equal row-coin prefixes share the fold state; the state-steps an
                 adversary saves, for a 1-bit and an 8-bit row coin, and the T1 that leaves without a grain premise
  lop3_fanin     over all 256 LOP3 functions and every input triple from the registers at hand (the state, two words,
                 the 16 bitwise functions of two keys): none computes two keyed steps, and none fuses a word's last
                 add into its fold.

Pricing (estimates on stated parts): h (honest hashing over stock, W1), the fold's own work, T1 and T2 at T 512 x 64,
T 512 x 512 and T 8,192 on Llama-3.2-1B's linears, for the lead, sparse commits under DistinctWrites alone, and the
folds with SHA-512 on the prover or on the certifier (R1), with per-period pools on the card's basis.

usage: sparse.py [OUT.json]
"""
import json
import math
import sys

import numpy as np

H32 = 32
U32 = (1 << 32) - 1
LANE_W = (1, 127, 1)

LLAMA = [("qkv", 2048, 3072), ("o", 2048, 2048), ("gate_up", 2048, 16384), ("down", 8192, 2048)]
LLAMA_LAYERS = 16


def lead_steps(k):
    """The lead's steps a dot product: step 1 holds 20 data lanes, each later step 28."""
    return 1 + math.ceil((k - 20) / 28)


# --- the lead's layout at sparse commitment -------------------------------------------------------------------------


def served_words(rng, a, b, tile, coins=None, rowbit=None):
    """Every k32 word of the lead-like layout: step t holds 28 data lanes, lanes 28-30 the public weights (1, 127, 1)
    against coins s_t[e, tile, j], lane 31 (1 + a_t[i]) against 1. a (m, 28 S), b (n, 28 S). Returns (words (S, m, n),
    coins, rowbit)."""
    m, n = a.shape[0], b.shape[0]
    s_steps = a.shape[1] // 28
    tiles = m // tile
    if coins is None:
        coins = 2 * rng.integers(-64, 64, (s_steps, 3, tiles, n)) + 1
    if rowbit is None:
        rowbit = rng.integers(0, 2, (s_steps, m))
    w = np.array(LANE_W, dtype=np.int64)
    acc = np.zeros((m, n), dtype=np.int64)
    out = []
    for t in range(s_steps):
        acc = acc + a[:, 28 * t:28 * t + 28] @ b[:, 28 * t:28 * t + 28].T
        lane = np.einsum("e,etn->tn", w, coins[t])
        acc = acc + np.repeat(lane, tile, 0) + (1 + rowbit[t])[:, None]
        out.append(acc.copy())
    return np.stack(out), coins, rowbit


def served_sparse(rng, m=32, n=24, tile=8, s_steps=16):
    """Zero data, committed word every r steps: the adversary's words from row and column prefixes, one IADD3 a word.
    Chained rounds: a committed word at step t reads coins of steps <= t only. The skip's backward pass from the
    total reads every later step's coins."""
    z = np.zeros((m, 28 * s_steps), dtype=np.int64), np.zeros((n, 28 * s_steps), dtype=np.int64)
    honest, coins, rowbit = served_words(rng, z[0], z[1], tile)
    w = np.array(LANE_W, dtype=np.int64)
    rowp = np.cumsum(1 + rowbit, 0)
    colp = np.cumsum(np.einsum("e,setn->stn", w, coins), 0)
    out = {}
    for r in (2, 4, 8):
        idx = np.arange(r - 1, s_steps, r)
        adv = rowp[idx][:, :, None] + np.repeat(colp[idx], tile, 1)
        chained = all(
            (np.cumsum(1 + rowbit[:i + 1], 0)[-1][:, None]
             + np.repeat(np.cumsum(np.einsum("e,setn->stn", w, coins[:i + 1]), 0)[-1], tile, 0) == honest[i]).all()
            for i in idx)
        sk = (len(idx) - 1) // 2
        redrawn = coins.copy()
        redrawn[idx[sk] + 1:] = 2 * rng.integers(-64, 64, redrawn[idx[sk] + 1:].shape) + 1
        rb2 = rowbit.copy()
        rb2[idx[sk] + 1:] = rng.integers(0, 2, rb2[idx[sk] + 1:].shape)
        other_total = (np.cumsum(1 + rb2, 0)[-1][:, None]
                       + np.repeat(np.cumsum(np.einsum("e,setn->stn", w, redrawn), 0)[-1], tile, 0))
        honest_writes = m * n * s_steps
        adv_writes = m * n * len(idx) + m * s_steps + (m // tile) * n * s_steps * 2
        out[f"L{32 * r}"] = {"words_match": bool((adv == honest[idx]).all()),
                             "chained_words_from_drawn_coins_match": bool(chained),
                             "chained_total_undetermined_share": float((other_total != honest[-1]).mean()),
                             "honest_mma_writes": honest_writes, "adv_writes": adv_writes,
                             "gap_share_of_honest": 1 - adv_writes / honest_writes}
    return {"m": m, "n": n, "tile": tile, "steps": s_steps, "sparse": out, "llama_gap": llama_gap()}


def llama_gap(rows=512, tile=64):
    """The same adversary's writes against the honest MMA writes over Llama-3.2-1B's linears, 64-row coin tiles: one
    write a committed word, a row prefix a row and step, two writes a tile, column and step for the column prefix."""
    out = {}
    for r in (2, 4, 8):
        honest = sum(rows * n * lead_steps(k) for _, k, n in LLAMA)
        adv = sum(rows * n * math.ceil(lead_steps(k) / r) + rows * lead_steps(k) + rows // tile * n * lead_steps(k) * 2
                  for _, k, n in LLAMA)
        out[f"L{32 * r}"] = 1 - adv / honest
    return out


def packing_gap(q, L, depth=32):
    """Zero-data packing count, units of stock (data MACs): a design with q noise lanes a k32 step and 32 - q data
    lanes pays alpha = 32 / (32 - q) a data MAC; the adversary packs a segment's qL/32 noise lanes into the narrowest
    MMA the alphabet allows."""
    alpha = 32 / (32 - q)
    return max(0.0, alpha * (1 - H32 * math.ceil(q * L / 32 / depth) / L))


def lane_floor(eps=0.01, lmax=32 * 4096):
    """The fewest committed words a data MAC at which the packing gap is at most eps, over q in 1..31 (a word with no
    noise lane reads no coin), L a multiple of 32 up to lmax, and the three alphabets. The k32 lead's layout commits
    1/28 a data MAC."""
    best = {}
    for name, depth in (("s8", 32), ("s4", 64), ("b1", 256)):
        rows = []
        for q in range(1, 32):
            for L in range(32, lmax + 1, 32):
                g = packing_gap(q, L, depth)
                if g <= eps:
                    rows.append((32 / (L * (32 - q)), q, L, g))
        wpm, q, L, g = min(rows)
        best[name] = {"words_per_data_mac": wpm, "one_over": 1 / wpm, "q": q, "L": L, "gap": g,
                      "h_over_lead": wpm * 28}
    return best


# --- exact designs: ncp3 with chained coin draws ---------------------------------------------------------------------


def prefix_words(x, y, L):
    k = x.shape[1]
    out, acc = [], np.zeros((x.shape[0], y.shape[0]), dtype=np.int64)
    for j in range(0, k, L):
        acc = acc + x[:, j:j + L] @ y[:, j:j + L].T
        out.append(acc.copy())
    return np.stack(out)


def ncp3_stack(a, b, e1, f1, r):
    """X = [A+E1 | A+E1R | -(A+E1(I+R))], Y = [B+F1; B-RF1; B+(I-R)F1]: X Y^T = A B^T."""
    e3 = e1 + e1 @ r
    f3 = f1 - r @ f1
    x = np.concatenate([a + e1, a + e1 @ r, -(a + e3)], 1)
    y = np.concatenate([b.T + f1, b.T - r @ f1, b.T + f3], 0)
    return x, y.T


def block_perm(rng, k):
    r = np.zeros((k, k), dtype=np.int64)
    for i, p in enumerate(rng.permutation(k // 4)):
        r[4 * i:4 * i + 4, 4 * p:4 * p + 4] = np.eye(4, dtype=np.int64)
    return r


def skip_adversary(x, y, total, L, skip):
    k = x.shape[1]
    s = k // L
    seg = lambda t: x[:, t * L:(t + 1) * L] @ y[:, t * L:(t + 1) * L].T
    words = [None] * s
    acc = np.zeros_like(total)
    for t in range(skip):
        acc = acc + seg(t)
        words[t] = acc.copy()
    words[s - 1] = total.copy()
    acc = total.copy()
    for t in range(s - 1, skip, -1):
        acc = acc - seg(t)
        words[t - 1] = acc.copy()
    return np.stack(words), (s - 1) * L * x.shape[0] * y.shape[0]


def exact_chain(rng, m=12, n=10, k=64, L=16):
    """ncp3, coins drawn block by block: E1 and F1 before block 1, R after block 1's words and before block 2. The
    skip of segment s needs every later segment's coins before word s is due. Feasible for every segment of blocks 2
    and 3 (all their coins were drawn at block 2's start); run it in block 3 at zero data, bit for bit."""
    a, b = np.zeros((m, k), dtype=np.int64), np.zeros((n, k), dtype=np.int64)
    e1, f1 = rng.integers(-31, 33, (m, k)), rng.integers(-31, 33, (k, n))
    x, y = ncp3_stack(a, b, e1, f1, block_perm(rng, k))
    honest = prefix_words(x, y, L)
    s = 3 * k // L
    per_block = k // L
    drawn = lambda seg: 0 if seg < per_block else 1
    due = lambda seg: 0 if seg < per_block else 1
    feasible = [all(drawn(t) <= due(sk) for t in range(sk + 1, s)) for sk in range(s - 1)]
    sk = s - 2
    adv, macs = skip_adversary(x, y, a @ b.T, L, sk)
    return {"k": k, "L": L, "segments": s, "skip_feasible_segments": int(sum(feasible)),
            "skip_infeasible_segments": int(len(feasible) - sum(feasible)), "skipped": sk,
            "words_match": bool((adv == honest).all()), "saved_share": 1 - macs / (s * L * m * n)}


def checkpoint(rng, m=16, n=12, tile=8, s_steps=16, r=4, draws=5):
    """Sparse words committed every r steps; d positions drawn afterwards. The adversary recomputes each drawn word
    from the committed word before it, bit for bit: at random data at most r - 1 k32 steps of that word's dot product,
    at zero data one IADD3 from the row and column prefixes. Only answered words gain credit."""
    a = rng.integers(-64, 64, (m, 28 * s_steps))
    b = rng.integers(-64, 64, (n, 28 * s_steps))
    honest, coins, rowbit = served_words(rng, a, b, tile)
    w = np.array(LANE_W, dtype=np.int64)
    positions = rng.choice([p for p in range(s_steps) if (p + 1) % r], draws, replace=False)
    ok = True
    for pos in positions:
        base = (pos + 1) // r * r - 1
        acc = honest[base].copy() if base >= 0 else np.zeros((m, n), dtype=np.int64)
        for t in range(base + 1, pos + 1):
            acc = acc + a[:, 28 * t:28 * t + 28] @ b[:, 28 * t:28 * t + 28].T
            acc = acc + np.repeat(np.einsum("e,etn->tn", w, coins[t]), tile, 0) + (1 + rowbit[t])[:, None]
        ok &= bool((acc == honest[pos]).all())
    z = np.zeros((m, 28 * s_steps), dtype=np.int64), np.zeros((n, 28 * s_steps), dtype=np.int64)
    zh, zc, zr = served_words(rng, z[0], z[1], tile)
    rowp = np.cumsum(1 + zr, 0)
    colp = np.cumsum(np.einsum("e,setn->stn", w, zc), 0)
    zero_ok = all(bool((rowp[p][:, None] + np.repeat(colp[p], tile, 0) == zh[p]).all()) for p in positions)
    committed = s_steps // r
    return {"r": r, "draws": draws, "random_data_words_match": ok, "zero_data_words_match": zero_ok,
            "random_data_k32_steps_a_draw_at_most": r - 1, "zero_data_writes_a_draw": 1,
            "credited_words": committed + draws, "uncommitted_unanswered_words": s_steps - committed - draws}


# --- folds ----------------------------------------------------------------------------------------------------------


def fold(words, kind, keys=None, k=0x9E3779B1):
    """Fold every k32 word (S, m, n) into one u32 state a dot product."""
    u = words.astype(np.int64) & U32
    st = np.zeros(u.shape[1:], dtype=np.int64)
    for t in range(u.shape[0]):
        if kind == "linear":
            st = (st * k + u[t]) & U32
        elif kind == "xor":
            st = st ^ u[t]
        else:
            st = st ^ (u[t] & keys[t])
    return st


def folds(rng, m=32, n=24, tile=8, s_steps=16, r=4):
    """Zero data unless stated. Honest folds every word; adversaries and their writes against the honest 2 a step
    (one MMA write, one fold write)."""
    z = np.zeros((m, 28 * s_steps), dtype=np.int64), np.zeros((n, 28 * s_steps), dtype=np.int64)
    honest, coins, rowbit = served_words(rng, z[0], z[1], tile)
    keys = rng.integers(0, 1 << 32, s_steps, dtype=np.int64)[:, None, None]
    w = np.array(LANE_W, dtype=np.int64)
    rowp = np.cumsum(1 + rowbit, 0)
    colp = np.cumsum(np.einsum("e,setn->stn", w, coins), 0)
    words = rowp[:, :, None] + np.repeat(colp, tile, 1)
    assert (words == honest).all()
    res = {}
    k = 0x9E3779B1
    co = np.array([pow(k, s_steps - 1 - t, 1 << 32) for t in range(s_steps)], dtype=object)
    rterm = np.array([[int(sum(int(c) * int(v) for c, v in zip(co, rowp[:, i]))) & U32] for i in range(m)],
                     dtype=np.int64)
    cterm = np.array([[int(sum(int(c) * int(v) for c, v in zip(co, colp[:, tt, j]))) & U32 for j in range(n)]
                      for tt in range(m // tile)], dtype=np.int64)
    lin_adv = (rterm + np.repeat(cterm, tile, 0)) & U32
    res["linear"] = {"state_match": bool((lin_adv == fold(honest, "linear", k=k)).all()),
                     "adv_writes_a_step": round((m * n + m * s_steps + (m // tile) * n * s_steps) / (m * n * s_steps), 4),
                     "honest_writes_a_step": 2}
    acc = np.zeros((m, n), dtype=np.int64)
    for t in range(0, s_steps, 2):
        acc = acc ^ (words[t] & U32) ^ (words[t + 1] & U32)
    res["xor"] = {"state_match": bool((acc == fold(honest, "xor")).all()), "adv_writes_a_step": 1.5,
                  "honest_writes_a_step": 2}
    keyed = fold(honest, "keyed", keys)
    idx = np.arange(r - 1, s_steps, r)
    sparse_only = np.zeros((m, n), dtype=np.int64)
    for t in range(s_steps):
        near = idx[min(np.searchsorted(idx, t), len(idx) - 1)]
        sparse_only ^= (honest[near] & U32) & keys[t]
    sk = s_steps // 2
    skip = honest.copy()
    skip[sk] = honest[sk - 1]
    keyed_rand = fold(served_words(rng, rng.integers(-64, 64, (m, 28 * s_steps)),
                                   rng.integers(-64, 64, (n, 28 * s_steps)), tile)[0], "keyed", keys)
    res["keyed"] = {"cheapest_adv_state_match": bool((fold(words, "keyed", keys) == keyed).all()),
                    "cheapest_adv_writes_a_step": 2, "honest_writes_a_step": 2,
                    "sparse_words_state_differs_share": float((sparse_only != keyed).mean()),
                    "skip_one_word_state_differs_share": float((fold(skip, "keyed", keys) != keyed).mean()),
                    "random_data_states_distinct_share": float(len(np.unique(keyed_rand)) / keyed_rand.size)}
    return res


def bitslice_xor(rng, rows=8, tile=4, s_steps=64, transpose_ops=320):
    """The keyed xor fold at zero data, 32 outputs of one row a register a bit (bit j of register b is bit b of output
    j). D_t = R_t + C_jt with R_t the row's prefix, the same in every lane, so the adversary runs the carry chain, one
    LOP3 a bit (maj with a broadcast register), up to the key's top set bit, folds each set bit's carry two steps a
    LOP3 (N_b ^ c ^ c'), and adds the row's and column's parities of A and B once at the end. Core LOP3s counted
    exactly; transposes, column prefixes and broadcasts amortized at Llama's shapes (64-row tiles)."""
    m, n = rows, 32
    z = np.zeros((m, 28 * s_steps), dtype=np.int64), np.zeros((n, 28 * s_steps), dtype=np.int64)
    honest, coins, rowbit = served_words(rng, z[0], z[1], tile)
    keys = rng.integers(0, 1 << 32, s_steps, dtype=np.int64)
    want = fold(honest, "keyed", keys[:, None, None])
    w = np.array(LANE_W, dtype=np.int64)
    rowp = np.cumsum(1 + rowbit, 0) & U32
    colp = np.cumsum(np.einsum("e,setn->stn", w, coins), 0) & U32
    lanes = 1 << np.arange(32, dtype=np.int64)
    ok, ops = True, 0
    for i in range(m):
        cb = [[int(((colp[t, i // tile] >> b) & 1) @ lanes) for b in range(32)] for t in range(s_steps)]
        ab = [[U32 if (int(rowp[t, i]) >> b) & 1 else 0 for b in range(32)] for t in range(s_steps)]
        st, pend, par = [0] * 32, [None] * 32, [0] * 32
        for t in range(s_steps):
            kt, carry = int(keys[t]), 0
            for b in range(kt.bit_length()):
                if b:
                    x, y = ab[t][b - 1], cb[t][b - 1]
                    carry = (x & y) | (x & carry) | (y & carry)
                    ops += 1
                if (kt >> b) & 1:
                    par[b] ^= ab[t][b] ^ cb[t][b]
                    if not b:
                        continue
                    if pend[b] is None:
                        pend[b] = carry
                    else:
                        st[b] ^= pend[b] ^ carry
                        pend[b] = None
                        ops += 1
        for b in range(32):
            st[b] ^= (pend[b] or 0) ^ par[b]
            ops += 1 + (pend[b] is not None)
        got = np.array([sum(((st[b] >> j) & 1) << b for b in range(32)) for j in range(32)], dtype=np.int64)
        ok &= bool((got == want[i]).all())
    core = ops / (m * 32 * s_steps)
    amort = {}
    for k_steps, n_cols in ((lead_steps(2048), 2048), (lead_steps(8192), 2048)):
        cols = (2 * 32 + transpose_ops + 16) / (64 * 32)
        bcast = 32 / n_cols
        out_t = transpose_ops / (32 * k_steps)
        adv = core + cols + bcast + out_t
        amort[f"S{k_steps}"] = {"column_prefix_transpose": cols, "row_broadcast": bcast, "state_transpose": out_t,
                                "writes_a_step": adv, "t1_percent_against_honest_2": 100 * (2 - adv) * H32 / 28}
    return {"state_match": ok, "core_lop3_a_step": core, "honest_writes_a_step": 2, "llama": amort}


def naf(k):
    """Non-adjacent form of k: (position, sign) of each nonzero digit, lowest first."""
    out, i = [], 0
    while k:
        if k & 1:
            d = 2 - (k & 3)
            out.append((i, d))
            k -= d
        k >>= 1
        i += 1
    return out


def bitslice_mul(rng, keys=64):
    """N k mod 2^32 for 32 states at once, a register a bit, by NAF shift-add: shifts are free; adding or subtracting
    N << s costs a sum and a carry LOP3 a bit from s up (no carry out of bit 31; a subtraction's complement and its
    + 1 fold into the LOP3s). Counted exactly and checked bit for bit against the word product."""
    lanes = 1 << np.arange(32, dtype=np.int64)
    ops_all, ok = [], True
    for _ in range(keys):
        k = int(rng.integers(0, 1 << 31)) * 2 + 1
        x = rng.integers(0, 1 << 32, 32, dtype=np.int64)
        n = [int(((x >> b) & 1) @ lanes) for b in range(32)]
        digits = naf(k)
        top = max(digits)
        acc = [0] * top[0] + n[:32 - top[0]]
        ops = 0
        for s, d in digits:
            if (s, d) == top:
                continue
            y = [0] * s + n[:32 - s]
            if d < 0:
                y = [v ^ U32 if b >= s else 0 for b, v in enumerate(y)]
            carry = U32 if d < 0 else 0
            for b in range(s, 32):
                a_, y_ = acc[b], y[b]
                acc[b] = a_ ^ y_ ^ carry
                ops += 1
                if b < 31:
                    carry = (a_ & y_) | (a_ & carry) | (y_ & carry)
                    ops += 1
        got = np.array([sum(((acc[b] >> j) & 1) << b for b in range(32)) for j in range(32)], dtype=np.int64)
        ok &= bool((got == (x * k) & U32).all())
        ops_all.append(ops / 32)
    return {"state_match": ok, "lop3_a_multiply_mean": float(np.mean(ops_all)),
            "lop3_a_multiply_min": float(np.min(ops_all))}


def separable_share(st, tile):
    """Share of 2 x 2 output minors inside a coin tile on which a row-plus-column form fails, over Z/2^32 and over
    GF(2)^32."""
    rows = np.array([i for i in range(st.shape[0] - 1) if (i + 1) % tile])
    a, b = st[rows, :-1], st[rows + 1, 1:]
    c, d = st[rows, 1:], st[rows + 1, :-1]
    return {"z": float((((a + b - c - d) & U32) != 0).mean()), "gf2": float(((a ^ b ^ c ^ d) != 0).mean())}


def mulfold(rng, m=32, n=24, tile=8, s_steps=16, r=4):
    """N <- (N ^ D_t) k_t, k_t a live odd u32 a step and call: each step permutes the state. Zero data: the word-grain
    adversary (IADD3 for D_t from the prefixes, LOP3, IMUL) matches at 3 writes a step, the honest's (MMA, LOP3, IMUL);
    the sparse and skip adversaries miss; the state is no row-plus-column form over Z or GF(2), unlike the linear fold;
    a bit-sliced multiply by k_t alone costs several LOP3s an output (bitslice_mul)."""
    z = np.zeros((m, 28 * s_steps), dtype=np.int64), np.zeros((n, 28 * s_steps), dtype=np.int64)
    honest, coins, rowbit = served_words(rng, z[0], z[1], tile)
    ks = rng.integers(0, 1 << 31, s_steps, dtype=np.int64) * 2 + 1

    def mf(words):
        st = np.zeros(words.shape[1:], dtype=np.int64)
        for t in range(words.shape[0]):
            st = ((st ^ (words[t] & U32)) * int(ks[t])) & U32
        return st

    want = mf(honest)
    w = np.array(LANE_W, dtype=np.int64)
    words = np.cumsum(1 + rowbit, 0)[:, :, None] + np.repeat(np.cumsum(np.einsum("e,setn->stn", w, coins), 0), tile, 1)
    idx = np.arange(r - 1, s_steps, r)
    near = np.stack([honest[idx[min(np.searchsorted(idx, t), len(idx) - 1)]] for t in range(s_steps)])
    skip = honest.copy()
    skip[s_steps // 2] = honest[s_steps // 2 - 1]
    x = rng.integers(0, 1 << 32, 4096, dtype=np.int64)
    perm = all(len(np.unique(((x ^ int(d)) * int(k)) & U32)) == len(np.unique(x))
               for d, k in zip(rng.integers(0, 1 << 32, 8), ks[:8]))
    keyed = fold(honest, "keyed", rng.integers(0, 1 << 32, s_steps, dtype=np.int64)[:, None, None])
    lin = fold(honest, "linear")
    return {"word_grain_adv_state_match": bool((mf(words) == want).all()), "word_grain_adv_writes_a_step": 3,
            "honest_writes_a_step": 3, "step_is_permutation": perm,
            "sparse_words_state_differs_share": float((mf(near) != want).mean()),
            "skip_one_word_state_differs_share": float((mf(skip) != want).mean()),
            "separable_share": {"mulfold": separable_share(want, tile), "xor_keyed": separable_share(keyed, tile),
                                "linear": separable_share(lin, tile)},
            "bitslice_mul": bitslice_mul(rng)}


def row_sharing(rng, tile=64, writes=3):
    """Zero data: rows of a coin tile whose row-coin prefixes agree have equal words, so equal fold states; an adversary
    computes each distinct prefix's state once. Expected state-steps saved a (tile, column): sum over t of tile minus
    the expected distinct prefixes of t coins of `bits` bits, checked against a seeded draw; as a share of the fold
    and as T1 (a fold of `writes` writes a step over 28 data MACs) without a grain premise."""
    out = {}
    for bits in (1, 8):
        for k in (2048, 8192):
            s_steps = lead_steps(k)
            exp = sum(tile + 2.0 ** (bits * t) * math.expm1(tile * math.log1p(-2.0 ** (-bits * t)))
                      for t in range(1, s_steps + 1) if bits * t <= 200)
            draw = rng.integers(0, 1 << bits, (tile, s_steps))
            seen = sum(tile - len({tuple(r[:t]) for r in draw}) for t in range(1, s_steps + 1))
            share = exp / (tile * s_steps)
            out[f"b{bits} K{k}"] = {"saved_state_steps_expected": exp, "saved_state_steps_drawn": seen,
                                    "share_of_fold": share, "t1_percent_without_grain": 100 * share * writes * H32 / 28}
    return out


def start_edge(rng, m=64, n=48, s_steps=8, writes=3):
    """Zero data, a call's first steps on the lead's layout with the 8-bit row coin, for three starts N_0:
      zero    N_1 = R_1 k_1 + C_1 k_1: one IADD3 of a row product and a column product, against 3 writes;
      kappa   a live word a call: N_1 = (kappa ^ x) k_1 reads x = R_1 + C_1 alone, so it is one load from a table over
              x's range, after one IADD3;
      rowcol  a_i ^ b_j, live words a row and a column: the honest's LOP3 xors three inputs, so it costs nothing, and
              rows of a tile column that share a row-coin prefix no longer share a state.
    Each route bit for bit; the states rows share; the writes each start leaves an adversary a call, at K 2,048 and
    8,192 (edge plus rows sharing, at the state's cost), its call-average per step, and its T1."""
    coins = 2 * rng.integers(-64, 64, (s_steps, 3, n)) + 1
    col = np.cumsum(np.einsum("e,sen->sn", np.array(LANE_W, dtype=np.int64), coins), 0)
    row = np.cumsum(rng.integers(0, 256, (s_steps, m)), 0)
    words = (row[:, :, None] + col[:, None, :]) & U32
    ks = rng.integers(0, 1 << 31, s_steps, dtype=np.int64) * 2 + 1
    kappa = int(rng.integers(0, 1 << 32))
    a, b = rng.integers(0, 1 << 32, m, dtype=np.int64), rng.integers(0, 1 << 32, n, dtype=np.int64)
    lo, hi = -16383, 255 + 16383
    table = ((kappa ^ (np.arange(lo, hi + 1) & U32)) * int(ks[0])) & U32
    starts = {"zero": np.zeros((m, n), dtype=np.int64), "kappa": np.full((m, n), kappa, dtype=np.int64),
              "rowcol": a[:, None] ^ b[None, :]}
    first = {s: ((n0 ^ words[0]) * int(ks[0])) & U32 for s, n0 in starts.items()}
    out = {"zero_route_matches": bool((first["zero"] == ((row[0] * int(ks[0]))[:, None] + col[0] * int(ks[0])) & U32).all()),
           "kappa_table_entries": hi - lo + 1, "kappa_x_in_table": bool(((row[0][:, None] + col[0]) >= lo).all()
                                                                       and ((row[0][:, None] + col[0]) <= hi).all()),
           "kappa_route_matches": bool((first["kappa"] == table[row[0][:, None] + col[0][None, :] - lo]).all()),
           "states_shared": {}}
    for s, st in starts.items():
        shared = []
        for t in range(s_steps):
            st = ((st ^ words[t]) * int(ks[t])) & U32
            shared.append(int(sum(m - len(np.unique(st[:, j])) for j in range(n))))
        out["states_shared"][s] = shared
    edge, state_cost = {"zero": 2, "kappa": 1, "rowcol": 0}, {"zero": 1, "kappa": 2, "rowcol": 3}
    share = {t: m + 2.0 ** (8 * t) * math.expm1(m * math.log1p(-2.0 ** (-8 * t))) for t in range(1, 26)}
    for s in starts:
        for k in (2048, 8192):
            steps = lead_steps(k)
            sharing = 0.0 if s == "rowcol" else (state_cost[s] * share[1] + writes * sum(share[t] for t in range(2, 26))) / m
            saved = edge[s] + sharing
            out[f"{s} K{k}"] = {"writes_saved_a_call": saved, "per_step_call_average": (writes * steps - saved) / steps,
                                "mu_f_floor": saved / (writes * steps), "t1_percent": 100 * saved * H32 / (28 * steps)}
    return out


def lop3_fanin(rng, cases=64):
    """For each input triple from the registers at hand, a LOP3 exists computing `target` iff the target's bit is
    constant on every minterm class of the triple's bits. Registers: N, D1, D2 and the 16 bitwise functions of K1, K2
    (a call computes those once). Targets: two keyed steps, and one step with its word's last add fused."""
    n_, d1, d2, k1, k2, d0, de = (rng.integers(0, 1 << 32, cases, dtype=np.int64) for _ in range(7))
    nk1, nk2 = k1 ^ U32, k2 ^ U32
    kf = [sum(((f >> (2 * x + y)) & 1) * ((k1 if x else nk1) & (k2 if y else nk2)) for x in (0, 1) for y in (0, 1))
          for f in range(16)]
    two = n_ ^ (d1 & k1) ^ (d2 & k2)
    fused = n_ ^ (((d0 + de) & U32) & k1)
    one = n_ ^ (d1 & k1)

    def found(regs, target):
        hits = 0
        bits = np.arange(32)
        rb = [((r[:, None] >> bits) & 1) for r in regs]
        tb = (target[:, None] >> bits) & 1
        for i in range(len(regs)):
            for j in range(len(regs)):
                for k in range(len(regs)):
                    mt = rb[i] * 4 + rb[j] * 2 + rb[k]
                    ok = True
                    for c in range(8):
                        sel = tb[mt == c]
                        if sel.size and sel.min() != sel.max():
                            ok = False
                            break
                    hits += ok
        return hits

    return {"two_keyed_steps": found([n_, d1, d2] + kf, two),
            "one_step_fused_add": found([n_, d0, de, k1, k2] + kf, fused),
            "one_keyed_step_sanity": found([n_, d1, k1], one)}


# --- pricing --------------------------------------------------------------------------------------------------------

SHA_WORDS = 3530.8 / 32
SHA_A = 3449.05 / 128
POINTS = {
    "T 512 x 64": {"rows": 64, "beside": 35.68, "t2": 10.90, "hash": 25.24, "gemm": 29.09, "export": 8.0,
                   "link": 3.64, "t1": 0.066, "hide": 0.0251},
    "T 512 x 512": {"rows": 512, "beside": 149.03, "t2": 37.34, "hash": 146.94, "gemm": 54.14, "export": None,
                    "link": 23.2, "t1": 0.044, "hide": 0.0031},
    "T 8,192": {"rows": 8192, "beside": 2322.66, "t2": 62.08, "hash": 2299.0, "gemm": 585.0, "export": None,
                "link": 358.0, "t1": 0.041, "hide": 0.0002},
}
CHUNK_HEAD = 0.0405
EXPORT_UNITS = (1263.0, 1435.0)
NOEXPORT_UNITS = (101.0, 113.0)
FOLD_ISSUE = (32.0, 64.0)
SM_UNITS_S = 4096 * 132 * 1.98e9


def shape_sums():
    out = sum(n for _, k, n in LLAMA)
    mac = sum(n * k for _, k, n in LLAMA)
    words = sum(n * lead_steps(k) for _, k, n in LLAMA)
    a_bytes = sum(k for _, k, n in LLAMA)
    return out, mac, words, a_bytes


def price(words_a_output_fn, fold_ops_a_word=0.0):
    """h (honest SHA-512 over stock, W1) for a design committing words_a_output_fn(k) words a dot product, with A's
    commitment as in the lead; o, the fold's own writes over stock."""
    sha = sum(n * words_a_output_fn(k) * SHA_WORDS * H32 for _, k, n in LLAMA)
    sha_a = sum(k * SHA_A * H32 for _, k, n in LLAMA)
    mac = sum(n * k for _, k, n in LLAMA)
    o = sum(n * lead_steps(k) * fold_ops_a_word * H32 for _, k, n in LLAMA) / mac
    return {"h": (sha + sha_a) / mac, "h_words": sha / mac, "h_a": sha_a / mac, "o": o}


def t2_fold(point, words_ratio, r1=False, ops=1):
    """T2 of a fold (estimate): the hash stream scaled by the bytes hashed; the GEMM stream with the export GEMMs
    replaced by the fold's (no-export GEMM plus `ops` integer ops a word, each at an issue slot or the INT pipe a
    lane); the lane-coin link unchanged; calibrated by the lead's beside / max(streams)."""
    p = POINTS[point]
    stock = p["beside"] / p["t2"]
    words = p["rows"] * sum(n * lead_steps(k) for _, k, n in LLAMA) * LLAMA_LAYERS
    export_ms = p["export"] if p["export"] is not None else words * sum(EXPORT_UNITS) / 2 / SM_UNITS_S * 1e3
    fold_ms = [words * (NOEXPORT_UNITS[i] + ops * FOLD_ISSUE[i]) / SM_UNITS_S * 1e3 for i in (0, 1)]
    if p["export"] is not None:
        fold_ms = [export_ms * (NOEXPORT_UNITS[i] + ops * FOLD_ISSUE[i]) / EXPORT_UNITS[i] for i in (0, 1)]
    gemm = [p["gemm"] - export_ms + f for f in fold_ms]
    calib = p["beside"] / max(p["hash"], p["gemm"])
    if r1:
        outs = p["rows"] * sum(n for _, k, n in LLAMA) * LLAMA_LAYERS
        lead_bytes = 4 * words
        r1_ms = R1_T2[point] * stock * (8 * outs) / lead_bytes if R1_T2.get(point) else 0.0
        hash_ms = r1_ms
    else:
        hash_ms = p["hash"] * words_ratio
    lo = max(hash_ms, gemm[0]) / stock
    hi = max(hash_ms, gemm[1], p["link"]) * calib / stock
    return {"stock_ms": stock, "export_ms": export_ms, "fold_gemm_ms": fold_ms, "gemm_stream_ms": gemm,
            "hash_or_net_ms": hash_ms, "t2_range": (lo, hi)}


R1_T2 = {"T 512 x 64": 102.0, "T 512 x 512": 588.0, "T 8,192": 688.0}

H200_MACS = 9.89e14
MFU_SERVE = 0.4
DATACENTER_H200 = 100_000
WINDOW_S = 600
WORLD_MACS_YEAR = 9.36e30 / 2
AND_PER_MAC = 2079 / 16
PERIOD_S = {"window": WINDOW_S, "week": 7 * 86400, "month": 30 * 86400, "year": 365 * 86400}


def per_period(share):
    """A share of stock pooled in AND gates over each period, on the card's basis (stock work held fixed)."""
    out = {}
    for scope, macs_s in (("one H200", H200_MACS * MFU_SERVE), ("datacenter", DATACENTER_H200 * H200_MACS * MFU_SERVE),
                          ("world", WORLD_MACS_YEAR / PERIOD_S["year"])):
        out[scope] = {p: share * macs_s * s * AND_PER_MAC for p, s in PERIOD_S.items()}
    return out


def pricing():
    lead = price(lead_steps)
    out, mac, words, a_bytes = shape_sums()
    designs = {
        "lead (every k32 word)": price(lead_steps),
        "sparse every 2nd word (DistinctWrites only)": price(lambda k: math.ceil(lead_steps(k) / 2)),
        "keyed xor fold, N_T and D_T committed": price(lambda k: 2, 1.0),
        "mul-xor fold, N_T and D_T committed": price(lambda k: 2, 2.0),
        "mul-xor fold, N_T only": price(lambda k: 1, 2.0),
    }
    for d in designs.values():
        d["cut"] = lead["h"] / d["h"] if d["h"] else float("inf")
    designs["mul-xor fold + R1 (certifier hashes)"] = {"h": 0.0, "o": price(lambda k: 2, 2.0)["o"], "cut": float("inf")}
    words_avg = words / out
    ratio2 = (8 * out + a_bytes / 1.0) / (4 * words + a_bytes)
    t1 = {}
    for name, p in POINTS.items():
        resid = p["t1"] - CHUNK_HEAD - p["hide"]
        t1[name] = {"lead": p["t1"], "lead_residual_coin_terms": resid, "lead_hiding": p["hide"],
                    "fold_estimate": resid + p["hide"] + CHUNK_HEAD * 2 / words_avg,
                    "sparse_r2_distinctwrites": 100 * (1 - 0.5) * (32 / 28)}
    t2 = {name: {"xor_fold": t2_fold(name, ratio2), "mulfold": t2_fold(name, ratio2, ops=2),
                 "mulfold_r1": t2_fold(name, 0.0, r1=True, ops=2)} for name in POINTS}
    best = designs["mul-xor fold, N_T and D_T committed"]
    return {"lead": lead, "designs": designs, "words_a_output_avg": words_avg, "hashed_bytes_ratio_fold": ratio2,
            "t1_percent": t1, "t2": t2,
            "per_period": {"lead_h": per_period(lead["h"]), "fold_h": per_period(best["h"]),
                           "fold_o": per_period(best["o"]),
                           "fold_t1_8192": per_period(t1["T 8,192"]["fold_estimate"] / 100),
                           "fold_t1_64": per_period(t1["T 512 x 64"]["fold_estimate"] / 100)}}


def main():
    rng = np.random.default_rng(20261029)
    res = {
        "served_sparse": served_sparse(rng),
        "lane_floor": lane_floor(),
        "exact_chain": exact_chain(rng),
        "checkpoint": checkpoint(rng),
        "folds": folds(rng),
        "bitslice_xor": bitslice_xor(rng),
        "mulfold": mulfold(rng),
        "row_sharing": row_sharing(rng),
        "lop3_fanin": lop3_fanin(rng),
        "pricing": pricing(),
        "start_edge": start_edge(rng),
    }
    s = json.dumps(res, indent=1, default=float)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            f.write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
