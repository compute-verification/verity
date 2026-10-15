"""General schedules for the schemes `crossing.py` found: can merges of merges or products after a merge beat the
tree-plus-branch charge, or Question 6.3's half charge?

Reads the "best scheme at each saving" lines of crossing.py logs, starts `sched.anneal` (weight objective, internal
merge nodes allowed) from each scheme's tree-plus-branch schedule, and prints the exact weight of the best schedule
found (`sched.weight`) against twice the saving. A weight below 2s would refute Question 6.3.

    python general.py LOG [min_saving] [steps] [restarts]
"""

import ast
import random
import re
import sys

from crossing import min_deletion, schedule
from flip import is_scheme
from sched import Scheme, anneal, weight


def schemes(path, min_saving):
    pat = re.compile(r"<(\d),(\d),(\d)> seed=(\d+) gamma=\S+ best scheme at each saving: (\{.*\})$")
    for line in open(path):
        m = pat.match(line.strip())
        if not m:
            continue
        p, q, r, seed = (int(x) for x in m.group(1, 2, 3, 4))
        for s, (charge, terms) in ast.literal_eval(m.group(5)).items():
            if s >= min_saving:
                yield (p, q, r), seed, s, charge, [tuple(t) for t in terms]


def main():
    path = sys.argv[1]
    min_saving = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    restarts = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    rng = random.Random(0)
    for (p, q, r), seed, s, charge, terms in schemes(path, min_saving):
        assert is_scheme(terms, p, q, r)
        sch = Scheme(terms, p, q, r)
        start = schedule(sch, min_deletion([t[2] for t in terms], 10 ** 6)[1])
        base = weight(sch, start)
        best = base
        for _ in range(restarts):
            _, G = anneal(sch, rng, steps=steps, max_merges=2 * s + 2, start=start)
            best = min(best, weight(sch, G), key=lambda x: x[1])
        merges, w, sum_rank = best
        saving = sum_rank - sch.m
        assert saving == s
        print(f"<{p},{q},{r}> seed={seed} s={s} tree-plus-branch: merges={base[0]} weight={base[1]}; "
              f"general: merges={merges} weight={w}; weight-2s={w - 2 * s}", flush=True)


if __name__ == "__main__":
    main()
