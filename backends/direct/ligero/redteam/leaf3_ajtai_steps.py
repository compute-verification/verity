"""red-team-leaf-3 H1: the verifier never pins a statement's ``steps`` (columns per VU) to the relation, and the Ajtai
gadget is column-uniform (``acc_out = X acc_in + B s``): ``steps <= n`` is only a compile-time assert in
``AjtaiLeaf.gadget``.  With ``steps > n`` the negacyclic wrap ``X^n = -1`` cancels column ``i`` against column ``n + i``,
so a VU row of ``steps' = 48 + n`` columns that is NOT the committed row has the committed row's digest exactly.

This script shows it with the leaf's own accumulation (``leaf.ajtai.digest_rows``, the function ``chain_witness`` uses
for the carried state and the digest the circuit pins), numpy only:

    x* : a committed fp8 row, 48 columns x 32 E4M3 bytes          -> h48 = digest(x*)   (the tree leaf's digest)
    x' : 112 columns: t | 0^16 | (x* | t)   with t any bits where x* has 0 bits
         digest over 112 columns = sum_i X^(47-i) B ((x*|t)_i - t_i) = digest(x*)

and that a GEMM over the forged operands (x', W') gives a prover-chosen value instead of <x*, W*>.

    PYTHONPATH=<ajtai-leaf tree> python leaf3_ajtai_steps.py [--n 64] [--role 0]

Exit 0 iff the forged 112-column row has the committed 48-column row's digest (the finding reproduces)."""
from __future__ import annotations

import argparse

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    from backends.direct.ligero.leaf import ajtai as A

    n, k, wb, steps = args.n, 32, 8, 48                      # fp8-ada: 48 columns x 32 E4M3 words per committed row
    assert steps <= n
    B = A.derive_matrix(n)
    rng = np.random.default_rng(args.seed)
    ok = True
    for role in (0, 1):
        x_star = rng.integers(0, 256, size=(steps * k,), dtype=np.int64)
        t = rng.integers(0, 256, size=(steps * k,), dtype=np.int64) & ~x_star & 0xFF   # bits only where x* has 0 bits
        s2 = steps + n                                       # 112 columns > n = 64
        x_forged = np.zeros(s2 * k, dtype=np.int64)
        x_forged[:steps * k] = t                             # columns 0..47   (exponents 111..64 = -X^(47..0))
        x_forged[n * k:] = x_star | t                        # columns 64..111 (exponents 47..0); 48..63 zero
        h48 = A.digest_rows(x_star[None], wb, B, role=role)[1][0, -1]
        h112 = A.digest_rows(x_forged[None], wb, B, role=role)[1][0, -1]
        same = bool(np.array_equal(h48 % A.P if hasattr(A, "P") else h48, h112))
        print(f"role {role}: n = {n}; committed row 48 cols, forged row {s2} cols; t has {int(np.unpackbits(t.astype(np.uint8)).sum())} "
              f"set bits; digest(forged) == digest(committed): {same}")
        ok &= same
        if role == 0:
            # the GEMM the chain would accumulate over the forged operands (integers here, the point is the dependence on t)
            w_star = rng.integers(0, 256, size=(steps * k,), dtype=np.int64)
            u = rng.integers(0, 256, size=(steps * k,), dtype=np.int64) & ~w_star & 0xFF
            w_forged = np.zeros(s2 * k, dtype=np.int64)
            w_forged[:steps * k] = u
            w_forged[n * k:] = w_star | u
            hw48 = A.digest_rows(w_star[None], wb, B, role=1)[1][0, -1]
            hw112 = A.digest_rows(w_forged[None], wb, B, role=1)[1][0, -1]
            print(f"  W side: digest(forged W) == digest(committed W): {bool(np.array_equal(hw48, hw112))}; "
                  f"<x*, W*> = {int(x_star @ w_star)} vs <x', W'> = {int(x_forged @ w_forged)} (prover-chosen via t, u)")
            ok &= bool(np.array_equal(hw48, hw112))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
