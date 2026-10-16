"""absorb_nnz.py: a chain of N duo absorbs (`PouwDuoAbsorb_v2`) lowered two ways, its ANDs and its unit's nonzeros (A₀'s and
B₀'s, what C-Flock's prove time and memory follow): through the Definition's traced gates, as a statement's lowering takes it
(`ir_lower.lower`; the builder's commits are no gates there, `boolean.trace`), and through its builder with the commits kept.

    PYTHONPATH=.:verity/ml/flock/python python3 benchmarks/e2e_slice/absorb_nnz.py 1 2 4
"""
from __future__ import annotations

import sys

from verity.core.protocols.pouw.circuit import segment as SG
from verity_flock import gf2
from verity_flock import ir_lower as IL


def chain(n: int, kept: bool) -> tuple[int, int]:
    C = gf2.Circuit()
    st = [b for j in range(32) for b in C.input(f"st{j}", 32)]
    ps = [[C.input(f"p{i}_{q}", 32) for q in range(32)] for i in range(n)]
    for i in range(n):
        if kept:
            st = SG._duo_absorb_bits(C, st, *ps[i])
        else:
            st = [w for word in IL.lower(C, SG.PouwDuoAbsorb, [[b] for b in st] + [[b] for w in ps[i] for b in w]) for w in word]
    for j in range(16):
        C.output(f"o{j}", st[32 * j:32 * j + 32])
    lay = IL.layout(C, [f"st{j}" for j in range(32)] + [f"p{i}_{q}" for i in range(n) for q in range(32)],
                    [f"o{j}" for j in range(16)])
    return lay.counts["ands"], lay.counts["nnz_a"] + lay.counts["nnz_b"]


if __name__ == "__main__":
    print("| absorbs | commits | ANDs | nnz | nnz an AND |\n|---|---|---|---|---|")
    for n in [int(x) for x in sys.argv[1:]] or [1, 2, 4]:
        for kept in (False, True):
            a, z = chain(n, kept)
            print(f"| {n} | {'kept' if kept else 'traced'} | {a:,} | {z:,} | {z / a:.0f} |", flush=True)
