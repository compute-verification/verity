"""Every numeric leaf of two D6 build pickles, compared: how many differ, and the largest relative difference.

    python pkl_eq.py A.pkl B.pkl
"""
import math
import pickle
import sys
from pathlib import Path


def flat(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f"{p}/{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from flat(v, f"{p}/{i}")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield p, float(x)


A, B = (dict(flat(pickle.loads(Path(p).read_bytes()))) for p in sys.argv[1:3])
only = sorted(set(A) ^ set(B))
diff = [(p, A[p], B[p]) for p in sorted(set(A) & set(B))
        if not (A[p] == B[p] or (math.isnan(A[p]) and math.isnan(B[p])))]
worst = max(diff, key=lambda d: abs(d[1] - d[2]) / max(abs(d[1]), abs(d[2]), 1e-300), default=None)
print(f"{len(set(A) & set(B))} numeric fields in both, {len(diff)} differ, {len(only)} in one only")
if worst:
    print("largest relative difference:", worst)
for p in only[:10]:
    print("only in", "A" if p in A else "B", p)
