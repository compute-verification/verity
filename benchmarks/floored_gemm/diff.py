"""First differential check of the tensor-core receipts TB4, TB2, TB1 (fgemm.cu's header), on the CPU, with the kernels' keys.

For a fixed key and random honest states, apply fixed value differences δ to one or two accumulator words and measure how
often the receipt accumulator R is unchanged. A word w at position t (TB1: the pair's index) feeds one row ρ(w) of R
(slot(): the receipt IMMA's A row) through rows κ(w, j) of K_t, one for each byte j the variant reads, so

    ΔR[ρ, :] = Σ_{w: ρ(w) = ρ} Σ_j (b_j(T_w + δ_w) − b_j(T_w)) K_t[κ(w, j), :]     (bytes u8, keys s8, exact in Z^8)

and R is unchanged iff that is 0 in every row. Honest states are running sums of 32 s products of uniform codes on
[−127, 127] (Ã B̃ᵀ's words at checkpoint s). Two modes:

- one checkpoint: δ added to the words at one checkpoint s;
- persistent: δ added at checkpoint s0 and every later one (a skipped product stays skipped), R summed over all of them.

python3 diff.py [--trials N] [--seed S] [--json OUT]
"""

import argparse
import itertools
import json
import sys

import numpy as np

from ref import BASE, CHAINED, emit_key, tb_key, tb_operand, u8bytes

VARIANTS = {"tb4": 5, "tb2": 6, "tb1": 7, "tb2w": 10, "tb1w": 11}
CHAINS = {"tb4c": 8, "tb1c": 9, "tb2wc": 12, "tb1wc": 13}
BYTES = {5: (0, 1, 2, 3), 6: (0, 1), 7: (1,), 10: (0, 1), 11: (1,)}
MI, NI = 2, 4   # the W tile config's warp tile (32 x 32): 8 positions, TB1 4 pairs


def slot(kind, ni, r, c, j):
    """(A row, k) of byte j of the word at row r, column c of tile ni (TB1: of pair ni // 2), as tb_operand lays it out."""
    if kind == 5:
        return (r % 8) + 8 * (c % 2), 16 * (r // 8) + 4 * (c // 2) + j
    if kind == 6:
        return r, 2 * c + j
    if kind == 10:
        return r, 16 * (ni % 2) + 2 * c + j
    if kind == 11:
        return r, 16 * (ni // 2) + 4 * (c // 2) + 2 * (ni % 2) + (c % 2)
    return r, 4 * (c // 2) + 2 * (ni % 2) + (c % 2)


def word_key(base, mi, ni):
    """The key position whose receipt IMMA reads tile (mi, ni)'s words."""
    return mi if base == 11 else (mi * NI + ni) // 2 if base in (7, 10) else mi * NI + ni


def keyrow(kind, keys, mi, ni, r, c, j):
    return keys[word_key(kind, mi, ni)][slot(kind, ni, r, c, j)[1]]


def same_row_partner(kind, r, c):
    """Another word of the same tile whose bytes reach the same row of R."""
    return (r + 8) % 16 if kind == 5 else r, c if kind == 5 else (c + 3) % 8


def honest(rng, n, s):
    """n running sums at checkpoint s (32 s products of uniform codes on [−127, 127]), as int64."""
    tot = np.zeros(n, dtype=np.int64)
    for _ in range(s):
        a = rng.integers(-127, 128, size=(n, 32), dtype=np.int64)
        b = rng.integers(-127, 128, size=(n, 32), dtype=np.int64)
        tot += (a * b).sum(axis=1)
    return tot


def trajectory(rng, n, S):
    """n honest trajectories T_1..T_S, [n, S]."""
    a = rng.integers(-127, 128, size=(n, S, 32), dtype=np.int64)
    b = rng.integers(-127, 128, size=(n, S, 32), dtype=np.int64)
    return np.cumsum((a * b).sum(axis=2), axis=1)


def dbytes(T, d, js):
    """Δb_j = b_j(T + δ) − b_j(T) for the bytes js, [..., len(js)], as u8 bytes of the s32 words."""
    u, v = T & 0xFFFFFFFF, (T + d) & 0xFFFFFFFF
    return np.stack([((v >> (8 * j)) & 0xFF) - ((u >> (8 * j)) & 0xFF) for j in js], axis=-1)


def unchanged(kind, keys, words, T, deltas):
    """P[ΔR = 0] and P[ΔR = 0 | some read byte changed], over the states T [N, len(words)] (or [N, len(words), S] summed
    over S), for the words (mi, ni, r, c) and fixed differences deltas."""
    js = BYTES[kind]
    rows = {}
    any_change = None
    for i, (mi, ni, r, c) in enumerate(words):
        db = dbytes(T[:, i], deltas[i], js)                    # [N, (S,) nb]
        if db.ndim == 3:
            db = db.sum(axis=1)
        K = np.stack([keyrow(kind, keys, mi, ni, r, c, j) for j in js])   # [nb, 8]
        ar = slot(kind, ni, r, c, 0)[0]
        rows[ar] = rows.get(ar, 0) + db @ K
        ch = (db != 0).any(axis=1)
        any_change = ch if any_change is None else any_change | ch
    zero = np.ones(T.shape[0], dtype=bool)
    for v in rows.values():
        zero &= (v == 0).all(axis=1)
    p = float(zero.mean())
    cancel = float((zero & any_change).sum() / max(1, any_change.sum()))
    return p, cancel, float(any_change.mean())


def product_deltas():
    return {"-1*1": -1, "-3*5": -15, "-11*13": -143, "-60*70": -4200, "-97*113": -10961, "-121*121": -14641,
            "+121*127": 15367}


def fixed_deltas():
    d = {"+1": 1, "-1": -1, "+2": 2, "-2": -2}
    for k in (4, 7, 8, 12, 15, 16, 20, 23, 24, 28, 30, 31):
        d[f"+2^{k}"] = 1 << k
        d[f"-2^{k}"] = -(1 << k)
    return d | product_deltas()


def op_maps(base):
    """For each key position t, the warp-tile source (row, column, byte) of every A slot [16, k] of its receipt IMMA."""
    kdim = 32 if base in (5, 10, 11) else 16
    maps = {}
    for mi in range(MI):
        for ni in range(NI):
            t = word_key(base, mi, ni)
            m = maps.setdefault(t, np.full((3, 16, kdim), -1, dtype=np.int64))
            for r in range(16):
                for c in range(8):
                    for j in BYTES[base]:
                        ar, k = slot(base, ni, r, c, j)
                        m[:, ar, k] = (16 * mi + r, 8 * ni + c, j)
    assert all((m >= 0).all() for m in maps.values())
    return maps


R_MAP = op_maps(5)[0] % np.array([16, 8, 4])[:, None, None]   # TB4's layout on one 16 x 8 tile: R's own bytes


def sim(kind, T):
    """Every trial's R [N, 16, 8] over a warp tile's trajectory T [N, S, 32, 32], as fgemm.cu folds it (chained or not)."""
    base, chained = BASE[kind], kind in CHAINED
    maps = op_maps(base)
    keys = {t: tb_key(kind, t) for t in maps}
    KR = tb_key(kind, 63)
    R = np.zeros((T.shape[0], 16, 8), dtype=np.int64)
    for s in range(T.shape[1]):
        B = u8bytes(T[:, s])
        if chained:
            RB = u8bytes(R)
            R = np.einsum("nrk,kc->nrc", RB[:, R_MAP[0], R_MAP[1], R_MAP[2]], KR)
        for t, m in maps.items():
            R = R + np.einsum("nrk,kc->nrc", B[:, m[0], m[1], m[2]], keys[t])
        R = ((R + 2**31) & 0xFFFFFFFF) - 2**31
    return R


def tile_trajectory(rng, n, S):
    """n warp-tile trajectories [n, S, 32, 32]: each checkpoint adds 32 products of uniform codes on [−127, 127], drawn as
    the rounded normal of the same variance."""
    sd = (127 * 128 / 3) * np.sqrt(32)
    return np.cumsum(np.rint(rng.normal(0, sd, size=(n, S, 32, 32))).astype(np.int64), axis=1)


def check_slots():
    """slot() against ref.py's tb_operand, the layout the gate checks on the device: every read byte where slot() puts it."""
    rng = np.random.default_rng(0)
    blk = rng.integers(0, 256, size=(32, 32, 4))
    for kind in VARIANTS.values():
        for mi in range(MI):
            for ni in range(NI):
                if emit_key(kind, mi, ni) is None:
                    continue
                A = tb_operand(kind, blk, mi, ni)
                for nj in ((ni - 1, ni) if kind in (7, 10) else (0, 1, 2, 3) if kind == 11 else (ni,)):
                    for r in range(16):
                        for c in range(8):
                            for j in BYTES[kind]:
                                ar, k = slot(kind, nj, r, c, j)
                                assert A[ar, k] == blk[16 * mi + r, 8 * nj + c, j], (kind, mi, nj, r, c, j)


def check_sim():
    """sim() against ref.py's tb_receipts and tbc_receipts (what the gate checks the device against) on one warp tile."""
    from ref import TB, TBC, tb_receipts, tbc_receipts

    rng = np.random.default_rng(1)
    T = tile_trajectory(rng, 1, 6)
    Ts = [(T[0, s] & 0xFFFFFFFF).astype(np.uint32) for s in range(6)]
    SB = sum(u8bytes(Tw) for Tw in Ts)
    lane = np.arange(32)
    g, q = lane >> 2, lane & 3
    for kind in list(TB.values()) + list(TBC.values()):
        want = (tbc_receipts(kind, Ts, 32, 32, 32, 32, 1, 1) if kind in CHAINED else tb_receipts(kind, SB, 32, 32, 32, 32, 1, 1))[0]
        R = (sim(kind, T)[0] & 0xFFFFFFFF).astype(np.uint32)
        got = np.stack([R[g, 2 * q], R[g, 2 * q + 1], R[g + 8, 2 * q], R[g + 8, 2 * q + 1]], axis=1)
        assert (got == want[:, :4]).all(), kind


def main() -> int:
    check_slots()
    check_sim()
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=12)
    ap.add_argument("--json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    N = args.trials
    out = {"trials": N, "seed": args.seed, "single": {}, "persistent": {}, "pairs": {}, "pair_search": {}}
    deltas = fixed_deltas()
    s_points = (1, 16, 128)
    states = {s: honest(rng, 2 * N, s).reshape(N, 2) for s in s_points}
    w1 = [(0, 1, 5, 3)]

    # one word, one checkpoint
    for vname, kind in VARIANTS.items():
        keys = {t: tb_key(kind, t) for t in range(MI * NI)}
        out["single"][vname] = {}
        for s in s_points:
            row = {}
            for dn, d in deltas.items():
                p, _, _ = unchanged(kind, keys, w1, states[s][:, :1], [d])
                row[dn] = p
            # one skipped product with nonzero codes a and b drawn per state
            a = rng.integers(1, 128, size=N) * rng.choice([-1, 1], size=N)
            b = rng.integers(1, 128, size=N) * rng.choice([-1, 1], size=N)
            T = states[s][:, :1]
            dv = -(a * b)
            js = BYTES[kind]
            db = dbytes(T[:, 0], dv, js)
            K = np.stack([keyrow(kind, keys, 0, 1, 5, 3, j) for j in js])
            z = ((db @ K) == 0).all(axis=1)
            row["-a*b random"] = float(z.mean())
            out["single"][vname][f"s={s}"] = row

    # one word, persistent from checkpoint s0 to S = 128 (k = 4,096)
    S, NP = 128, max(2000, N // 20)
    traj = trajectory(rng, NP, S)
    for vname, kind in VARIANTS.items():
        keys = {t: tb_key(kind, t) for t in range(MI * NI)}
        out["persistent"][vname] = {}
        # s0 = 113 and 127 leave 16 and 2 checkpoints: a byte that gains d each checkpoint and wraps W times sums to
        # S d − 256 W, which is 0 when W = S d / 256 (δ = 2^12 over 16, δ = ±2^15 over 2)
        for s0 in (1, 64, 113, 120, 127, 128):
            row = {}
            for dn in ("+1", "-1", "+2^4", "-3*5", "-11*13", "-121*121", "+2^8", "+2^12", "+2^15", "-2^15", "+2^16",
                       "+2^24", "+2^31"):
                T = traj[:, None, s0 - 1 :]
                p, _, _ = unchanged(kind, keys, w1, T, [deltas[dn]])
                row[dn] = p
            out["persistent"][vname][f"s0={s0}"] = row

    # two words, one checkpoint, in the three geometries that can share R's row
    def geoms(kind):
        r2, c2 = same_row_partner(kind, 5, 3)
        return {"same tile, same R row": [(0, 1, 5, 3), (0, 1, r2, c2)],
                "same R row, other position": [(0, 1, 5, 3), (1, 2, 5, 3)],
                "same R row, the pair's other tile": [(0, 0, 5, 3), (0, 1, 5, 3)],
                "other R rows": [(0, 1, 5, 3), (0, 1, 6, 3)]}
    pair_d = {"(+1, -1)": (1, -1), "(+2^8, -2^8)": (256, -256), "(-a*b, +a*b), a*b = 143": (-143, 143),
              "(+2^16, -2^16)": (1 << 16, -(1 << 16)), "(+1, +2^8)": (1, 256)}
    T2 = states[16]
    for vname, kind in VARIANTS.items():
        keys = {t: tb_key(kind, t) for t in range(MI * NI)}
        out["pairs"][vname] = {}
        for gn, ws in geoms(kind).items():
            out["pairs"][vname][gn] = {}
            for dn, (d1, d2) in pair_d.items():
                p, c, ch = unchanged(kind, keys, ws, T2, [d1, d2])
                out["pairs"][vname][gn][dn] = {"unchanged": p, "cancel_given_change": c, "read_bytes_changed": ch}

    # one skipped k column at one checkpoint: δ[r, c] = −a_r b_c over a whole warp tile (32 x 32, the W config), with
    # nonzero codes; R unchanged needs all 16 receipt rows to cancel
    NT = min(4000, max(2000, N // 20))
    out["rank1_skip"] = {}
    Tt = honest(rng, NT * 32 * 32, 16).reshape(NT, 32, 32)
    av = rng.integers(1, 128, size=(NT, 32)) * rng.choice([-1, 1], size=(NT, 32))
    bv = rng.integers(1, 128, size=(NT, 32)) * rng.choice([-1, 1], size=(NT, 32))
    dT = -(av[:, :, None] * bv[:, None, :])
    for vname, kind in VARIANTS.items():
        keys = {t: tb_key(kind, t) for t in range(MI * NI)}
        js = BYTES[kind]
        dR = np.zeros((NT, 16, 8), dtype=np.int64)
        for mi in range(MI):
            for ni in range(NI):
                blk = (slice(None), slice(16 * mi, 16 * mi + 16), slice(8 * ni, 8 * ni + 8))
                db = dbytes(Tt[blk], dT[blk], js)                          # [NT, 16, 8, nb]
                for r in range(16):
                    for c in range(8):
                        K = np.stack([keyrow(kind, keys, mi, ni, r, c, j) for j in js])   # [nb, 8]
                        dR[:, slot(kind, ni, r, c, 0)[0]] += db[:, r, c] @ K
        out["rank1_skip"][vname] = {"unchanged": float((dR == 0).all(axis=(1, 2)).mean()),
                                    "row_unchanged": float((dR == 0).all(axis=2).mean()), "tiles": NT}

    # the small search: the best two-word cancellation, P[ΔR = 0 | a read byte changed], over every pair of word slots in
    # one receipt row (the W config's 8 positions x 8 columns) and differences ±2^k (k < 32) and ±a b
    cand = sorted({1 << k for k in range(32)} | {(1 << k) + 1 for k in range(1, 31)} | {15, 143, 4200, 10961, 14641, 15367})
    cand = cand + [-x for x in cand]
    NS = 4000
    Ts = honest(rng, 2 * NS, 16).reshape(NS, 2)
    for vname, kind in VARIANTS.items():
        keys = {t: tb_key(kind, t) for t in range(MI * NI)}
        # every word whose bytes reach R's row 0 (64 of them on the W config)
        slots = [(mi, ni, r, c) for mi in range(MI) for ni in range(NI) for r in range(16) for c in range(8)
                 if slot(kind, ni, r, c, 0)[0] == 0]
        # the key rows' rank decides whether any pair can cancel: check it over every pair of slots first
        js = BYTES[kind]
        min_rank, worst = 99, None
        for w1_, w2_ in itertools.combinations(slots, 2):
            M = np.stack([keyrow(kind, keys, *w1_, j) for j in js] + [keyrow(kind, keys, *w2_, j) for j in js])
            rk = int(np.linalg.matrix_rank(M.astype(float)))
            if rk < min_rank:
                min_rank, worst = rk, (w1_, w2_)
        best = (0.0, None)
        for d1 in cand[:: max(1, len(cand) // 40)]:
            for d2 in cand[:: max(1, len(cand) // 40)]:
                p, c, ch = unchanged(kind, keys, list(worst), Ts, [d1, d2])
                if ch > 0 and c > best[0]:
                    best = (c, (d1, d2))
        out["pair_search"][vname] = {"read_rows": 2 * len(js), "slots": len(slots), "min_rank_over_slot_pairs": min_rank,
                                     "worst_pair": [list(worst[0]), list(worst[1])], "best_cancel_given_change": best[0],
                                     "best_deltas": best[1], "deltas_tried": len(cand[:: max(1, len(cand) // 40)]) ** 2,
                                     "states": NS}
    # every kind, chained or not, on whole warp-tile trajectories (S = 16 checkpoints, k = 512): one word (or two in one R
    # row) off by δ from checkpoint s0 to the last, R compared with the honest R over the whole tile
    NC, SC = max(1000, min(4000, N // 50)), 16
    TT = tile_trajectory(rng, NC, SC)
    out["tile_sim"] = {"trials": NC, "checkpoints": SC}
    cases = [("+1", 1, 1), ("-3*5", -15, 1), ("+2^4", 16, 1), ("+2^12", 1 << 12, 1), ("+2^15", 1 << 15, 15),
             ("-2^15", -(1 << 15), 15), ("+2^16", 1 << 16, 15), ("+2^31", 1 << 31, 15), ("+1", 1, 16), ("-3*5", -15, 16),
             ("-11*13", -143, 16), ("+2^8", 256, 16)]
    for vname, kind in list(VARIANTS.items()) + list(CHAINS.items()):
        R0 = sim(kind, TT)
        res = {}
        for dn, d, s0 in cases:
            T1 = TT.copy()
            T1[:, s0 - 1 :, 5, 11] += d
            res[f"{dn} from s0={s0}"] = float((sim(kind, T1) == R0).all(axis=(1, 2)).mean())
        # two words reaching one R row, (+2^15, −2^15) over the last two checkpoints
        r2, c2 = same_row_partner(BASE[kind], 5, 3)
        T1 = TT.copy()
        T1[:, SC - 2 :, 5, 8 + 3] += 1 << 15
        T1[:, SC - 2 :, r2, 8 + c2] -= 1 << 15
        res["pair (+2^15, -2^15) from s0=15"] = float((sim(kind, T1) == R0).all(axis=(1, 2)).mean())
        out["tile_sim"][vname] = res
    txt = json.dumps(out, indent=1)
    if args.json:
        open(args.json, "w").write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
