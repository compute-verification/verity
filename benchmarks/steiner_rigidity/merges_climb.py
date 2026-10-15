"""Hill-climb for d*L on links adding combinations of d words (CMP-8, A3; note 20261015T0436Z-draft-merges-and-reuse).

`python merges_climb.py P_FIELD RxC L D [--open] [--restarts N] [--steps N] [--seed S] [--out DIR]` climbs the D words
of a span V on the R x C D-grid (one level, so words are R*L x C*L over GF(P_FIELD)) toward a large total saving
sum_t (L - min_{w in V, w != 0} rank(D_t - w))_+, and writes the largest total against d*L to
`climb_<p>_<R>x<C>_L<L>_d<D>[_open].json` in DIR (default $OUT). With --open (d = 2) the objective also charges every direction where the word has full row rank, which steers the climb into generic rank
below R*L (half the restarts start from S K T for a Kronecker canonical pencil K with a left minimal index, already
there): for R = 2 that is the case merges_check.py part 14 leaves open. Each restart also records the best total
among states with generic rank < R*L, and among those that also have four or more winning directions and a D-column
with two helped cells (the open case).

`python merges_climb.py --fixed SRC [--Ls 2,3] [--restarts N] [--steps N] [--seed S] [--out DIR]` instead takes
every split miss in SRC/split_*.json (from merges_split_search.py) that no coupled pair closes, and anneals its words
for the fixed B at each L, writing the best totals against L * rank B to `fixed_climb.json`.
"""

import itertools
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def brank(M, p):
    """Ranks over GF(p) of a batch (n, r, c) of matrices."""
    M = M % p
    n, r, c = M.shape
    inv = np.array([0] + [pow(x, p - 2, p) for x in range(1, p)], dtype=np.int64)
    row = np.zeros(n, dtype=np.int64)
    ar = np.arange(r)
    for col in range(c):
        cand = (M[:, :, col] != 0) & (ar[None, :] >= row[:, None])
        ks = np.nonzero(cand.any(axis=1))[0]
        if not len(ks):
            continue
        piv, rr = np.argmax(cand[ks], axis=1), row[ks]
        a, b = M[ks, piv].copy(), M[ks, rr].copy()
        M[ks, rr], M[ks, piv] = a, b
        M[ks, rr] = M[ks, rr] * inv[M[ks, rr, col]][:, None] % p
        f = M[ks, :, col].copy()
        f[np.arange(len(ks)), rr] = 0
        M[ks] = (M[ks] - f[:, :, None] * M[ks, rr][:, None, :]) % p
        row[ks] += 1
    return row


def cells_D(R, C, L):
    cells = [(i, j) for i in range(R) for j in range(C)]
    Ds = np.zeros((len(cells), R * L, C * L), dtype=np.int64)
    for n, (i, j) in enumerate(cells):
        for k in range(L):
            Ds[n, i * L + k, j * L + k] = 1
    return cells, Ds


def elements(p, d):
    """All nonzero coefficient vectors, and the index of each one's projective direction."""
    vs = [v for v in itertools.product(range(p), repeat=d) if any(v)]
    dirs = sorted({tuple(x * pow(next(y for y in v if y), p - 2, p) % p for x in v) for v in vs})
    di = {v: n for n, v in enumerate(dirs)}
    return np.array(vs, dtype=np.int64), np.array(
        [di[tuple(x * pow(next(y for y in v if y), p - 2, p) % p for x in v)] for v in vs]), np.array(dirs)


class Scorer:
    def __init__(self, p, R, C, L, d):
        self.p, self.L, self.d = p, L, d
        self.cells, self.Ds = cells_D(R, C, L)
        self.coef, self.dir_of, self.dirs = elements(p, d)

    def __call__(self, Zs):
        p, Ds = self.p, self.Ds
        W = np.tensordot(self.coef, Zs, axes=(1, 0)) % p
        M = (Ds[None] - W[:, None]) % p
        rk = brank(M.reshape(-1, *Ds.shape[1:]), p).reshape(len(W), len(Ds))
        mn = rk.min(axis=0)
        total = int(np.maximum(0, self.L - mn).sum())
        win = {int(self.dir_of[np.argmin(rk[:, t])]) for t in range(len(Ds)) if mn[t] < self.L}
        cols = [j for (i, j), m in zip(self.cells, mn) if m < self.L]
        return total, mn, win, len(cols) > len(set(cols))

    def full_directions(self, Zs):
        """Number of directions of the span where the word has full row rank."""
        W = np.tensordot(self.dirs, Zs, axes=(1, 0)) % self.p
        return int((brank(W, self.p) == Zs.shape[1]).sum())

    def generic_rank(self, Zs):
        W = np.tensordot(self.dirs, Zs, axes=(1, 0)) % self.p
        return int(brank(W, self.p).max())


def kronecker(rng, rows, cols):
    """K_1, K_2 (rows x cols) of a canonical pencil lam_1 K_1 + lam_2 K_2 with one or two left minimal indices."""
    K1, K2 = np.zeros((rows, cols), dtype=np.int64), np.zeros((rows, cols), dtype=np.int64)
    r = c = 0
    for _ in range(rng.choice([1, 1, 2])):
        e = rng.randrange(0, max(1, (rows - r) // 2))
        if r + e + 1 > rows:
            break
        for i in range(e):
            K1[r + i, c + i] = 1
            K2[r + i + 1, c + i] = 1
        r, c = r + e + 1, c + e
    while r < rows and c < cols:
        f = rng.choice([0, 0, 1, 2])
        if f == 0 or r + f > rows or c + f + 1 > cols:
            a, b = rng.choice([(1, 0), (0, 1), (1, 1), (1, 2)])
            K1[r, c], K2[r, c] = a, b
            r, c = r + 1, c + 1
        else:
            for i in range(f):
                K1[r + i, c + i] = 1
                K2[r + i, c + i + 1] = 1
            r, c = r + f, c + f + 1
    return K1, K2


def seed(sc, rng, nr, nc):
    """Each word a sum of partial cells (random diagonal entries of one to three cells), maybe with rank-1 noise."""
    p, L = sc.p, sc.L
    Zs = np.zeros((sc.d, nr, nc), dtype=np.int64)
    for i in range(sc.d):
        for _ in range(rng.randint(1, 3)):
            r0, c0 = np.nonzero(sc.Ds[rng.randrange(len(sc.cells))])
            for k in range(L):
                if rng.random() < 0.7:
                    Zs[i, r0[k], c0[k]] = (Zs[i, r0[k], c0[k]] + rng.randrange(1, p)) % p
        if rng.random() < 0.3:
            u, v = [rng.randrange(p) for _ in range(nr)], [rng.randrange(p) for _ in range(nc)]
            Zs[i] = (Zs[i] + np.outer(u, v)) % p
    return Zs


def climb(sc, rng, steps, open_mode, R, C):
    """One restart of annealing. Moves change one entry of a word, add to one or all diagonal entries of a cell, add a
    rank-1 matrix, or redraw the word. With open_mode the objective charges each direction of the pencil where Z_lam
    has full row rank, which steers the climb into generic rank below R*L and keeps it there, and pays for up to
    four winning directions and for a D-column with two helped cells (else part 8 applies): the open case."""
    p, L, d = sc.p, sc.L, sc.d
    nr, nc = R * L, C * L
    if open_mode and rng.random() < 0.5:
        K = np.array(kronecker(rng, nr, nc))
        S = np.array([[rng.randrange(p) for _ in range(nr)] for _ in range(nr)])
        T = np.array([[rng.randrange(p) for _ in range(nc)] for _ in range(nc)])
        Zs = np.einsum("ab,ibc,cd->iad", S, K, T) % p
    else:
        Zs = seed(sc, rng, nr, nc)

    def value(Z):
        total, mn, win, shared = sc(Z)
        nfull = sc.full_directions(Z) if open_mode else 0
        bonus = 1500 * (min(len(win), 4) + shared) if open_mode else 0
        return 1000 * total - int(mn.sum()) - 2000 * nfull + bonus, total, (win, shared)

    obj, total, win = value(Zs)
    best, best_low, best_open, best_Z = total, -1, -1, Zs.copy()
    for step in range(steps):
        Z2, i, u = Zs.copy(), rng.randrange(d), rng.random()
        r0, c0 = np.nonzero(sc.Ds[rng.randrange(len(sc.cells))])
        if u < 0.5:
            Z2[i, rng.randrange(nr), rng.randrange(nc)] = rng.randrange(p)
        elif u < 0.65:
            k = rng.randrange(L)
            Z2[i, r0[k], c0[k]] = (Z2[i, r0[k], c0[k]] + rng.randrange(1, p)) % p
        elif u < 0.75:
            Z2[i, r0, c0] = (Z2[i, r0, c0] + rng.randrange(1, p)) % p
        elif u < 0.85:
            Z2[i] = (Z2[i] + np.outer([rng.randrange(p) for _ in range(nr)], [rng.randrange(p) for _ in range(nc)])) % p
        else:
            Z2[i] = seed(sc, rng, nr, nc)[0]
        o2, t2, w2 = value(Z2)
        temp = 600 * (1 - step / steps) + 1
        if o2 >= obj or rng.random() < np.exp((o2 - obj) / temp):
            Zs, obj, total, win = Z2, o2, t2, w2
            if total > best:
                best, best_Z = total, Zs.copy()
            is_open = len(win[0]) >= 4 and win[1]
            if d == 2 and total > min(best_low, best_open if is_open else total) and sc.generic_rank(Zs) < nr:
                best_low = max(best_low, total)
                if is_open:
                    best_open = max(best_open, total)
    return best, best_low, best_open, best_Z


def fixed_score(Zs, Ds, B, L, p):
    """sum_t (L - rank(D_t - sum_j B_tj Z_j))_+ for one fixed B (rows = links)."""
    return int(np.maximum(0, L - brank((Ds - np.einsum("tj,jrc->trc", B, Zs)) % p, p)).sum())


def basis_seed(Ds, B, p, rng):
    """Words with W_t = D_t on d random links whose rows of B are independent (a start at saving d*L)."""
    from check import rank
    k, d = B.shape
    for _ in range(200):
        S = rng.sample(range(k), d)
        if rank(B[S], p) < d:
            continue
        A = np.concatenate([B[S], np.eye(d, dtype=np.int64)], axis=1) % p
        for c in range(d):
            piv = next(r for r in range(c, d) if A[r, c])
            A[[c, piv]] = A[[piv, c]]
            A[c] = A[c] * pow(int(A[c, c]), p - 2, p) % p
            for r in range(d):
                if r != c and A[r, c]:
                    A[r] = (A[r] - A[r, c] * A[c]) % p
        return np.einsum("jt,trc->jrc", A[:, d:], Ds[S]) % p
    return None


def fixed_climb(links, betas, p, L, restarts, steps, rng):
    """Anneal the words for a fixed cell pattern and B (half the restarts from basis_seed): the best total."""
    nR, nC = 1 + max(r for r, _ in links), 1 + max(c for _, c in links)
    B, d = np.array(betas, dtype=np.int64) % p, len(betas[0])
    Ds = np.zeros((len(links), nR * L, nC * L), dtype=np.int64)
    for t, (r, c) in enumerate(links):
        for i in range(L):
            Ds[t, r * L + i, c * L + i] = 1
    top = 0
    for n in range(restarts):
        Zs = basis_seed(Ds, B, p, rng) if n % 2 == 0 else None
        if Zs is None:
            Zs = np.array([[[rng.randrange(p) for _ in range(nC * L)] for _ in range(nR * L)] for _ in range(d)])
        f = fixed_score(Zs, Ds, B, L, p)
        top = max(top, f)
        for s in range(steps):
            j, r, c = rng.randrange(d), rng.randrange(nR * L), rng.randrange(nC * L)
            old = Zs[j, r, c]
            Zs[j, r, c] = rng.randrange(p)
            g = fixed_score(Zs, Ds, B, L, p)
            if g >= f or rng.random() < math.exp((g - f) / max(0.05, 1 - s / steps)):
                f, top = g, max(top, g)
            else:
                Zs[j, r, c] = old
    return top


def fixed_main(src, out, restarts, steps, rng, Ls):
    """The split misses in src/split_*.json that no coupled pair closes: anneal their words at each L in Ls."""
    res = []
    for f in sorted(Path(src).glob("split_*.json")):
        j = json.loads(f.read_text())
        for o in j["over_rank"]:
            if o["coupled_pair"] <= o["rank_B"]:
                continue
            links = [tuple(x) for x in o["links"]]
            for L in Ls:
                top = fixed_climb(links, o["betas"], j["p"], L, restarts, steps, rng)
                res.append({"file": f.name, "links": links, "betas": o["betas"], "labels_reach": o["labels_reach"],
                            "L": L, "max_total": top, "bound": o["rank_B"] * L})
                print(f.name, len(links), "L", L, "best total", top, "rank B * L", o["rank_B"] * L, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fixed_climb.json").write_text(json.dumps(res, indent=1))
    print(f"{len(res)} climbs on split misses no coupled pair closes: "
          f"{sum(r['max_total'] > r['bound'] for r in res)} above rank B * L")


def main():
    if "--fixed" in sys.argv:
        opt = {a: sys.argv[sys.argv.index(a) + 1] for a in ("--fixed", "--restarts", "--steps", "--seed", "--out",
                                                             "--Ls") if a in sys.argv}
        return fixed_main(opt["--fixed"], Path(opt.get("--out", os.environ.get("OUT", "."))),
                          int(opt.get("--restarts", 12)), int(opt.get("--steps", 6000)),
                          random.Random(int(opt.get("--seed", 20261015))),
                          [int(x) for x in opt.get("--Ls", "2,3").split(",")])
    p, (R, C), L, d = int(sys.argv[1]), map(int, sys.argv[2].split("x")), int(sys.argv[3]), int(sys.argv[4])
    opt = {a: sys.argv[sys.argv.index(a) + 1] for a in ("--restarts", "--steps", "--seed", "--out") if a in sys.argv}
    open_mode = "--open" in sys.argv
    restarts, steps = int(opt.get("--restarts", 40)), int(opt.get("--steps", 1500))
    rng = random.Random(int(opt.get("--seed", 20261015)))
    out = Path(opt.get("--out", os.environ.get("OUT", ".")))
    t0, sc = time.time(), Scorer(p, R, C, L, d)
    hist, hist_low, hist_open, top, examples = {}, {}, {}, 0, []
    for n in range(restarts):
        best, best_low, best_open, Z = climb(sc, rng, steps, open_mode, R, C)
        hist[best] = hist.get(best, 0) + 1
        if best_low >= 0:
            hist_low[best_low] = hist_low.get(best_low, 0) + 1
        if best_open >= 0:
            hist_open[best_open] = hist_open.get(best_open, 0) + 1
        top = max(top, best)
        if best > d * L:
            if d == 2:
                from merges_check import Shape, pencil_profile
                _, bp = pencil_profile(Shape(R, C, L, L), Z[0], Z[1], p)
                assert sum(s for s, _ in bp.values()) == best, "climb total disagrees with merges_check ranks"
            examples.append({"total": best, "words": Z.tolist()})
        print(f"  restart {n}: best {best}, rank < RL {best_low}, open case {best_open}, top {top}, "
              f"{time.time() - t0:.0f} s", flush=True)
    res = {"p": p, "grid": [R, C], "L": L, "d": d, "open": open_mode, "restarts": restarts, "steps": steps,
           "histogram": {str(k): v for k, v in sorted(hist.items())},
           "low_rank_histogram": {str(k): v for k, v in sorted(hist_low.items())},
           "open_case_histogram": {str(k): v for k, v in sorted(hist_open.items())}, "max_total": top,
           "bound_dL": d * L, "counterexamples": examples[:5], "seconds": round(time.time() - t0, 1)}
    out.mkdir(parents=True, exist_ok=True)
    name = f"climb_{p}_{R}x{C}_L{L}_d{d}{'_open' if open_mode else ''}.json"
    (out / name).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "counterexamples"} | {"counterexamples": len(examples)}))


if __name__ == "__main__":
    main()
