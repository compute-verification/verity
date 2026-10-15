"""Lemma 3 on every binary hierarchy of the full 3x3 grid (the n = 3 case of Conjecture A): 2,027,025 hierarchies,
each checked over GF(2) and GF(10007) as in check.py part 2 (typing never stuck, (L), (I), handle recursion, Claim 4).

Usage: exhaustive_3x3.py [WORKERS] [LIMIT]; LIMIT checks only the first LIMIT hierarchies of each worker's share.
"""

import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check import all_binary_trees, check_tree  # noqa: E402

CELLS = [(r, c) for r in range(3) for c in range(3)]


def work(args):
    wid, workers, limit = args
    done = 0
    for i, tree in enumerate(all_binary_trees(CELLS)):
        if i % workers != wid:
            continue
        for p in (2, 10007):
            check_tree(tree, p)
        done += 1
        if limit and done >= limit:
            break
    return done


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    t0 = time.time()
    with mp.Pool(workers) as pool:
        counts = pool.map(work, [(w, workers, limit) for w in range(workers)])
    print(f"[labeling] exhaustive 3x3: {sum(counts)} binary hierarchies on the full 3x3 grid, GF(2) and GF(10007): ok "
          f"({time.time() - t0:.0f} s, {workers} workers)")
