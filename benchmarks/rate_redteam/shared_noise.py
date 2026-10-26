"""J10's probe: H1 + fx1 at K = 64 with the noise derived once a call row and a call column (shared noise) against
one seed a tile (`h1tile`, today), on a call of 2 x 2 tiles of 32 x 32.

Shared noise is `h1tile.noise` of one call seed on the call's whole layout: call row i takes blocks
``[i RB, (i + 1) RB)`` and call column j ``M RB + [j CB, (j + 1) CB)``, so tile (I, J) reads rows ``32 I ..`` and
columns ``32 J ..`` of it. Fold keys stay per tile (σ_R of the tile's position) in both schemes.

The cheater computes tile (I, J0) honestly and skips its row neighbour (I, J1), whose W rows equal J0's
(`B_J1 = B_J0`), the case where a neighbour helps most. Its guess for every word of the skipped tile is the
neighbour's word plus every correction that costs a row's or a column's work, not a cell's:

* `T_s`: the neighbour's `T_s` plus ``Y_s . (U_J1 - U_J0)``, Y_s the row's residue prefix sums (one row's work), so
  the residual is ``X . δ``, δ = W'_J1 - W'_J0 - ΔU the coin difference (and int8 wraps);
* `O'` from the guessed ``T_last``, and `D_d = U A_d` exactly (a row's digits and a column's U).

Under one seed a tile the same guess reads the skipped tile's own X and U for the corrections. It reports, per
scheme and data regime, the exact-match rate of each set's words, of a cell's two `T` words, of a lane's sets, of
the receipt's words and of the whole receipt, with the residual's point mass at 0. Attack A2 is the column neighbour
(I1, J) of (I0, J) with ``A_I1 = A_I0``; A3 measures whether two skipped row neighbours copied from one honest tile are
lucky together more often than apart (the shared coin's correlation).

``python benchmarks/rate_redteam/shared_noise.py [TRIALS] [SEED]`` prints JSON lines; ``--check`` runs only the
self-check (the per-tile scheme here equals `h1tile.reference`).
"""
from __future__ import annotations

import json
import random
import sys

from verity.core.protocols.pouw.circuit import h1tile as H
from verity.core.protocols.pouw.circuit import seeds as SD
from verity.core.protocols.pouw.circuit import segment as SG

R = Cc = 32
K = 64
MC = NC = 2
M, N = MC * R, NC * Cc
CALL_POSITION = 1 << 40


def _sx8(v: int) -> int:
    return SG._sx8(v)


def tile_position(I: int, J: int) -> int:
    return I * NC + J


def call_noise(seed_c: bytes):
    """The call's noise: per call row (dither bits, digit coins), per call column (coin bits, U)."""
    return H.noise(seed_c, H.Layout(M, N, K))


def tile_noise(salt, roots, I, J):
    """One seed a tile: tile (I, J)'s own rows and columns."""
    return H.noise(SD.seed(salt, tile_position(I, J), *roots), H.Layout(R, Cc, K))


def tile_keys(salt, roots, I, J):
    return H.fold_keys(SD.receipt_seed(salt, tile_position(I, J), *roots), H.Layout(R, Cc, K))


def lane_inputs(a, b, rows, cols, ln, lay):
    Rr, Cs = lay.rows(ln), lay.cols(ln)
    seg = [range(32 * s, 32 * s + 32) for s in range(lay.ks)]
    xr = [[[a[i][t] for t in ts] for i in Rr] for ts in seg]
    bb = [[[rows[i][0][t] for t in ts] for i in Rr] for ts in seg]
    wr = [[[b[j][t] for t in ts] for j in Cs] for ts in seg]
    cc = [[[cols[j][0][t] for t in ts] for j in Cs] for ts in seg]
    u = [cols[j][1] for j in Cs]
    z = [[rows[i][1][d] for i in Rr] for d in range(lay.m - 1)]
    return xr, bb, wr, cc, u, z


def tile_words(a, b, rows, cols):
    """Each lane's absorbed sets (T_0 .. T_(KS-1), O', D_1 .. D_(m-1)), 32 words each."""
    lay = H.Layout(len(a), len(b), len(a[0]))
    return [SG.call_words(*lane_inputs(a, b, rows, cols, ln, lay)) for ln in range(lay.lanes)]


def receipt(words, keys):
    """CP-W from each lane's sets and keys, as `h1tile.reference` folds them."""
    nsets = len(words[0])
    cp = [[0, 0] for _ in range(nsets)]
    for ln, sets in enumerate(words):
        st = list(keys[ln])
        for s, p in enumerate(sets):
            st = SG.absorb(H.FOLD, st, p)
            for h in range(2):
                for v in st[8 * h:8 * h + 8]:
                    cp[s][h] ^= v
    return [w & H.M32 for s in cp for w in s]


def lane_x(a, rows, ln, lay):
    """The lane's 4 rows of X = A + b, per segment (int8 encodings)."""
    return [[[(a[i][t] + (rows[i][0][t] & 1)) & 0xFF for t in range(32 * s, 32 * s + 32)] for i in lay.rows(ln)]
            for s in range(lay.ks)]


def guess_words(a1, rows1, cols1, a0, rows0, cols0, words0, lay):
    """The generous guess for every lane of a skipped tile (rows `rows1`, columns `cols1`) from an honest tile's
    words (`rows0`, `cols0`): T_s from the neighbour's plus ``Y1_s . U1 - Y0_s . U0`` (Y_s a row's residue prefix
    sums), O' from the guessed T_last, D exactly. The U term is given free: it costs K multiply-adds a cell over all
    the prefix sets, a cell's honest work, so the rates here bound a cheaper guess's from above."""
    out = []
    m = lay.m
    for ln in range(lay.lanes):
        X1, X0 = lane_x(a1, rows1, ln, lay), lane_x(a0, rows0, ln, lay)
        Cs = lay.cols(ln)
        u1 = [cols1[j][1] for j in Cs]
        u0 = [cols0[j][1] for j in Cs]
        z = [[rows1[i][1][d] for i in lay.rows(ln)] for d in range(m - 1)]
        sets = []
        Y1 = [[0] * 32 for _ in range(4)]
        Y0 = [[0] * 32 for _ in range(4)]
        for s in range(lay.ks):
            for r in range(4):
                for rho in range(32):
                    Y1[r][rho] += _sx8(X1[s][r][rho])
                    Y0[r][rho] += _sx8(X0[s][r][rho])
            T = []
            for q in range(32):
                r, c = SG.row_slot(q), SG.col_slot(q)
                du = sum(Y1[r][rho] * _sx8(u1[c][rho]) - Y0[r][rho] * _sx8(u0[c][rho]) for rho in range(32))
                T.append((words0[ln][s][q] + du) & H.M32)
            sets.append(T)
        X = X1
        A = SG.digit_words(X, z, m)
        neg = [[(-v) & 0xFF for v in row] for row in u1]
        sets.append([SG.seg_dot(sets[lay.ks - 1][q], neg[SG.col_slot(q)], A[0][SG.row_slot(q)]) for q in range(32)])
        sets += [[SG.seg_dot(0, u1[SG.col_slot(q)], A[d][SG.row_slot(q)]) for q in range(32)] for d in range(1, m)]
        out.append(sets)
    return out


def _bytes(rng: random.Random, n: int) -> bytes:
    return bytes(rng.getrandbits(8) for _ in range(n))


def operands(rng: random.Random, regime: str):
    """A (M x K) and B (N x K) int8 encodings: `zero` is the cheater's best case (X = b, W' = c + U, no wraps),
    `random` uniform int8."""
    if regime == "zero":
        return [[0] * K for _ in range(M)], [[0] * K for _ in range(N)]
    return ([[rng.getrandbits(8) for _ in range(K)] for _ in range(M)],
            [[rng.getrandbits(8) for _ in range(K)] for _ in range(N)])


def tiles_of(a, b, I, J):
    return a[R * I:R * I + R], b[Cc * J:Cc * J + Cc]


def scheme_noise(scheme, salt, roots, call, I, J):
    if scheme == "shared":
        rows, cols = call
        return rows[R * I:R * I + R], cols[Cc * J:Cc * J + Cc]
    return tile_noise(salt, roots, I, J)


class Tally:
    def __init__(self, nsets: int):
        self.n = 0
        self.set_words = [0] * nsets
        self.set_total = [0] * nsets
        self.cell_t = 0
        self.cell_total = 0
        self.lane_all = 0
        self.lane_total = 0
        self.rc_words = 0
        self.rc_total = 0
        self.rc_all = 0

    def add(self, guess, honest, rc_guess, rc_honest, ks):
        self.n += 1
        for gl, hl in zip(guess, honest):
            self.lane_all += gl == hl
            self.lane_total += 1
            for s, (g, h) in enumerate(zip(gl, hl)):
                self.set_words[s] += sum(x == y for x, y in zip(g, h))
                self.set_total[s] += len(g)
            for q in range(32):
                self.cell_t += all(gl[s][q] == hl[s][q] for s in range(ks))
                self.cell_total += 1
        self.rc_words += sum(x == y for x, y in zip(rc_guess, rc_honest))
        self.rc_total += len(rc_honest)
        self.rc_all += rc_guess == rc_honest

    def report(self, names):
        return {
            "tiles": self.n,
            "word_match": {nm: round(w / t, 5) for nm, w, t in zip(names, self.set_words, self.set_total)},
            "cell_T_match": round(self.cell_t / self.cell_total, 5),
            "lane_match": round(self.lane_all / self.lane_total, 5),
            "receipt_word_match": round(self.rc_words / self.rc_total, 5),
            "receipt_match": self.rc_all,
        }


def self_check(rng: random.Random) -> dict:
    """The per-tile scheme's words and receipt here equal `h1tile.reference` on every tile of a random call."""
    a, b = operands(rng, "random")
    salt, roots = _bytes(rng, 64), (_bytes(rng, 64), _bytes(rng, 64))
    ok = True
    for I in range(MC):
        for J in range(NC):
            at, bt = tiles_of(a, b, I, J)
            rows, cols = tile_noise(salt, roots, I, J)
            rc = receipt(tile_words(at, bt, rows, cols), tile_keys(salt, roots, I, J))
            ok &= rc == H.reference(at, bt, salt, tile_position(I, J), roots)[1]
    return {"check": "per-tile receipt == h1tile.reference", "ok": bool(ok)}


def attack(trials: int, rng: random.Random, regime: str, kind: str) -> dict:
    """A1 (`row`): skip (0, 1), copy from (0, 0) with B_1 = B_0. A2 (`col`): skip (1, 0), copy from (0, 0) with
    A_1 = A_0 (the guess then reads the skipped tile's X, which differs only in its dither)."""
    lay = H.Layout(R, Cc, K)
    names = [f"T{s}" for s in range(lay.ks)] + ["O'"] + [f"D{d}" for d in range(1, lay.m)]
    out = {}
    for scheme in ("shared", "per-tile"):
        tl = Tally(lay.sets)
        for _ in range(trials):
            a, b = operands(rng, regime)
            if kind == "row":
                b[Cc:2 * Cc] = [list(r) for r in b[:Cc]]
                (I0, J0), (I1, J1) = (0, 0), (0, 1)
            else:
                a[R:2 * R] = [list(r) for r in a[:R]]
                (I0, J0), (I1, J1) = (0, 0), (1, 0)
            salt, roots = _bytes(rng, 64), (_bytes(rng, 64), _bytes(rng, 64))
            call = call_noise(SD.seed(salt, CALL_POSITION, *roots)) if scheme == "shared" else None
            a0, b0 = tiles_of(a, b, I0, J0)
            a1, b1 = tiles_of(a, b, I1, J1)
            rows0, cols0 = scheme_noise(scheme, salt, roots, call, I0, J0)
            rows1, cols1 = scheme_noise(scheme, salt, roots, call, I1, J1)
            w0 = tile_words(a0, b0, rows0, cols0)
            w1 = tile_words(a1, b1, rows1, cols1)
            g1 = guess_words(a1, rows1, cols1, a0, rows0, cols0, w0, lay)
            keys1 = tile_keys(salt, roots, I1, J1)
            tl.add(g1, w1, receipt(g1, keys1), receipt(w1, keys1), lay.ks)
        out[scheme] = tl.report(names)
    return {"attack": "A1 row neighbour" if kind == "row" else "A2 column neighbour", "regime": regime, **out}


def joint(trials: int, rng: random.Random, regime: str) -> dict:
    """A3: skip (0, 1) and a third tile column's (0, 2) twin, both copied from (0, 0) with equal B: per cell and word
    T_0, P(both right) against P(right)^2 (one call of 1 x 3 tiles)."""
    global NC, N
    NC0, N0 = NC, N
    NC, N = 3, 3 * Cc
    try:
        lay = H.Layout(R, Cc, K)
        out = {}
        for scheme in ("shared", "per-tile"):
            n = h1 = h2 = both = 0
            for _ in range(trials):
                a, b = operands(rng, regime)
                b[Cc:2 * Cc] = [list(r) for r in b[:Cc]]
                b[2 * Cc:3 * Cc] = [list(r) for r in b[:Cc]]
                salt, roots = _bytes(rng, 64), (_bytes(rng, 64), _bytes(rng, 64))
                call = call_noise(SD.seed(salt, CALL_POSITION, *roots)) if scheme == "shared" else None
                a0, b0 = tiles_of(a, b, 0, 0)
                rows0, cols0 = scheme_noise(scheme, salt, roots, call, 0, 0)
                w0 = tile_words(a0, b0, rows0, cols0)
                hits = []
                for J in (1, 2):
                    rows1, cols1 = scheme_noise(scheme, salt, roots, call, 0, J)
                    _, b1 = tiles_of(a, b, 0, J)
                    w1 = tile_words(a0, b1, rows1, cols1)
                    g1 = guess_words(a0, rows1, cols1, a0, rows0, cols0, w0, lay)
                    hits.append([[g1[ln][0][q] == w1[ln][0][q] for q in range(32)] for ln in range(lay.lanes)])
                for ln in range(lay.lanes):
                    for q in range(32):
                        x, y = hits[0][ln][q], hits[1][ln][q]
                        n += 1
                        h1 += x
                        h2 += y
                        both += x and y
            p1, p2 = h1 / n, h2 / n
            out[scheme] = {"words": n, "p_right": round((p1 + p2) / 2, 5), "p_both": round(both / n, 5),
                           "p_right_sq": round(p1 * p2, 5)}
        return {"attack": "A3 two row neighbours, T0 jointly", "regime": regime, **out}
    finally:
        NC, N = NC0, N0


def main(argv: list[str]) -> int:
    check_only = "--check" in argv
    args = [x for x in argv if not x.startswith("--")]
    trials = int(args[0]) if args else 40
    rng = random.Random(int(args[1]) if len(args) > 1 else 10)
    sc = self_check(rng)
    print(json.dumps(sc))
    if not sc["ok"]:
        return 1
    if check_only:
        return 0
    print(json.dumps({"K": K, "tile": [R, Cc], "call_tiles": [MC, NC], "trials": trials,
                      "sets": H.Layout(R, Cc, K).sets}))
    for regime in ("zero", "random"):
        for kind in ("row", "col"):
            print(json.dumps(attack(trials, rng, regime, kind)), flush=True)
        print(json.dumps(joint(trials, rng, regime)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
