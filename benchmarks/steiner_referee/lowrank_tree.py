#!/usr/bin/env python3
"""One optimal Steiner tree for an instance recorded by lowrank_cell.py (RS-D2F, CMP-13).

Reruns Dreyfus-Wagner on the instance's G_1..G_S, keeping each subset's split table, and backtracks from (all
terminals, 0) to a tree of total length sigma_S. Prints its edges (P0 = 0, P1..PS the prefixes, X* Steiner points),
each Steiner point's degree, and its rank distance to every terminal, which is what shows a tree's shape (pair hubs,
centres).

Usage: lowrank_tree.py RECORDS.json INDEX   (INDEX counts the instance lines, from 0)
"""

import json
import sys

import numpy as np

from lowrank_cell import relax, setup


def main():
    path, index = sys.argv[1], int(sys.argv[2])
    rec = [json.loads(x) for x in open(path) if x.strip() and '"summary"' not in x][index]
    r, q = rec["r"], rec["q"]
    digits, pw, nb, rank = setup(r, q)
    codes = [int((np.array(g).reshape(-1) * pw).sum()) for g in rec["G"]]
    dist = lambda a, b: int(rank[int(((digits[b] - digits[a]) % q) @ pw)])
    dp, best = {}, {}
    for i, t in enumerate(codes):
        dp[1 << i] = rank[((digits - digits[t]) % q) @ pw]
    full = (1 << len(codes)) - 1
    for mask in sorted(range(1, full + 1), key=lambda m: bin(m).count("1")):
        if mask in dp:
            continue
        low, b, sub = mask & -mask, None, (mask - 1) & mask
        while sub:
            if sub & low:
                b = dp[sub] + dp[mask ^ sub] if b is None else np.minimum(b, dp[sub] + dp[mask ^ sub])
            sub = (sub - 1) & mask
        best[mask] = b.astype(np.int16)
        dp[mask] = relax(best[mask], nb)

    edges = []
    stack = [(full, 0)]
    while stack:
        mask, v = stack.pop()
        if bin(mask).count("1") == 1:
            t = codes[mask.bit_length() - 1]
            if v != t:
                edges.append((v, t, dist(v, t)))
            continue
        cur = int(dp[mask][v])
        if int(best[mask][v]) == cur:
            low, sub = mask & -mask, (mask - 1) & mask
            while not (sub & low and int(dp[sub][v]) + int(dp[mask ^ sub][v]) == cur):
                sub = (sub - 1) & mask
            stack += [(sub, v), (mask ^ sub, v)]
        else:
            u = next(int(u) for u in nb[v] if int(dp[mask][u]) == cur - 1)
            edges.append((u, v, 1))
            stack.append((mask, u))

    names = {0: "P0", **{c: f"P{i + 1}" for i, c in enumerate(codes)}}
    steiner = [x for e in edges for x in e[:2] if x not in names]
    for x in dict.fromkeys(steiner):
        names[x] = f"X{len(names) - len(codes) - 1}"
    deg = {}
    for a, b, _ in edges:
        deg[a], deg[b] = deg.get(a, 0) + 1, deg.get(b, 0) + 1
    print(json.dumps({"steiner": int(dp[full][0]), "chain": rec["chain"], "length": sum(w for *_, w in edges),
                      "edges": [[names[a], names[b], w] for a, b, w in edges]}))
    for x in dict.fromkeys(steiner):
        print(json.dumps({"point": names[x], "degree": deg[x], "rank_to_P": [dist(x, t) for t in [0] + codes]}))


if __name__ == "__main__":
    main()
