"""Exact test of the LOP3 shapes of an R80X word entry g(R, C) = ((R + C) ^ c) + k mod 2^w, over every row and column:
out = N + A(C) + sign*L(P(R), Q(R), M(C)) (shape a) or N + A(C) + sign*L(P(R), M1(C), M2(C)) (shape d), with A, P, Q,
M any functions. Shape a exists iff some A leaves, at every bit i, at most two distinct functions of R among the bits i
of sign*(g(., C) - A(C)); shape d iff they are among 0, ~0, P and ~P. The search goes bit by bit over A mod 2^(i+1) per
column residue. g is symmetric in R and C, so the transposed shapes are the same question.

These are every program of fold_window's LOP3 then IADD3 that computes N + g for every N: the IADD3 must read N (a
LOP3 of N plus terms is N + g only when its value is N or ~N up to the top bit, which leaves g separable), and its other
operands are the LOP3 once (twice makes g even) and one row, column or constant term. They are also what IADD3 then
LOP3 leaves when the IADD3 reads N two or three times: at N = 0 its value is a LOP3 of single terms. A hi half's word
entry is the same question at its (c, k), on lo coins and an h lo half chosen so neither carry into the hi half occurs.
A width where no shape exists for either sign certifies every wider one (a T-function program's low bits are a program
at the lower width).

Run: `python3 -B word_entry_mux.py --all PAIRS.json OUT.json JOBS` (PAIRS: `r80x_batch.py --pairs-out`), or
`python3 -B word_entry_mux.py PAIRS.json W,W [COVERS,...]`."""
import sys
import time

sys.setrecursionlimit(10000)


def table(c, k, w, sign):
    m = (1 << w) - 1
    return [[(sign * ((((r + col) & m) ^ c) + k)) & m for col in range(1 << w)] for r in range(1 << w)]


def exists(c, k, w, sign, budget=2_000_000, shape="a"):
    """True/False, or None if the search budget ran out. The functions at bit i are bitmasks over R mod 2^(i+1).
    Shape a: L(P(R), Q(R), M(C)) + A(C), each bit's columns among two row functions. Shape d: L(P(R), M1(C), M2(C))
    + A(C), each bit's columns among 0, ~0, P and ~P."""
    G = table(c, k, w, sign)
    nodes = [0]

    def func(i, rho, a):
        n, mod = 1 << (i + 1), (1 << (i + 1)) - 1
        f = 0
        for r in range(n):
            if ((G[r][rho] - a) & mod) >> i & 1:
                f |= 1 << r
        return f

    def dfs(i, valid):
        nodes[0] += 1
        if nodes[0] > budget:
            raise TimeoutError
        if i == w:
            return True
        F = {}
        for rho2 in range(1 << (i + 1)):
            cand = {}
            for a in valid[rho2 & ((1 << i) - 1)]:
                for b in (0, 1):
                    a2 = a | b << i
                    cand.setdefault(func(i, rho2, a2), []).append(a2)
            F[rho2] = cand
        allf = set().union(*F.values())
        tried = set()
        if shape == "d":
            ones = (1 << (1 << (i + 1))) - 1
            for P in [None] + sorted(f for f in allf if f not in (0, ones)):
                pair = frozenset({0, ones} | ({P, P ^ ones} if P is not None else set()))
                if frozenset({P, P ^ ones} if P is not None else ()) in tried:
                    continue
                tried.add(frozenset({P, P ^ ones} if P is not None else ()))
                nv = {x: [a for f in pair if f in F[x] for a in F[x][f]] for x in F}
                if all(nv.values()) and dfs(i + 1, nv):
                    return True
            return False
        star = min(F, key=lambda x: len(F[x]))
        for U in F[star]:
            unc = [x for x in F if U not in F[x]]
            Vs = set(allf) if not unc else set.intersection(*(set(F[x]) for x in unc))
            for V in Vs:
                pair = frozenset((U, V))
                if pair in tried:
                    continue
                tried.add(pair)
                nv = {x: [a for f in pair if f in F[x] for a in F[x][f]] for x in F}
                if all(nv.values()) and dfs(i + 1, nv):
                    return True
        return False

    try:
        return dfs(0, {0: [0]})
    except TimeoutError:
        return None


def first_dead(p, widths=range(4, 9)):
    """The least width at which no shape and sign exists (none dies again higher: a T-function program's low bits
    are one at the lower width), with every width's results."""
    out = {"covers": p["covers"], "c": p["c"], "k": p["k"], "carry": bool(p.get("carry")), "widths": {}}
    for w in widths:
        t0 = time.time()
        r = {f"{sh}{'+' if s > 0 else '-'}": exists(p["c"] & ((1 << w) - 1), p["k"] & ((1 << w) - 1), w, s, shape=sh)
             for sh in ("a", "d") for s in (1, -1)}
        out["widths"][w] = dict(r, seconds=round(time.time() - t0, 2))
        if all(v is False for v in r.values()):
            out["dead_at"] = w
            break
    return out


if __name__ == "__main__" and sys.argv[1] == "--all":
    import json
    import multiprocessing as mp
    with open(sys.argv[2]) as fh:
        pairs = json.load(fh)
    with mp.Pool(int(sys.argv[4])) as pool:
        res = pool.map(first_dead, pairs)
    with open(sys.argv[3], "w") as fh:
        json.dump(res, fh, indent=1)
    from collections import Counter
    print("dead_at:", sorted(Counter(x.get("dead_at") for x in res).items(), key=str))
    print("undecided:", [x["covers"] for x in res if any(v is None for w in x["widths"].values() for v in w.values())])
elif __name__ == "__main__":
    import json
    with open(sys.argv[1]) as fh:
        pairs = json.load(fh)
    widths = [int(x) for x in sys.argv[2].split(",")]
    sel = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    for p in pairs:
        if sel and p["covers"][0] not in sel:
            continue
        out = []
        for w in widths:
            t0 = time.time()
            r = [exists(p["c"] & ((1 << w) - 1), p["k"] & ((1 << w) - 1), w, s, shape=sh)
                 for sh in ("a", "d") for s in (1, -1)]
            out.append(f"w{w}: a+{r[0]} a-{r[1]} d+{r[2]} d-{r[3]} ({time.time() - t0:.1f}s)")
        print(p["covers"][0], hex(p["c"] & 255), hex(p["k"] & 255), " ".join(out), flush=True)
