"""ATTACK-LAB: Dither's honest kernel and its top attacks in exact CPU emulation, counted by instruction class and priced
at the H200's data-sheet rates (ACCEL BET-ATTACK-LAB, Linear CMP-125).

Every route here computes Dither's transcript, the prefix words T^(l)_ij = sum_{t < L (l + 1)} At_it Bt_tj mod 2^32, and
is checked word for word against the honest one, and its receipts (duo's or li's fold, `segment.py`'s) against
honest's. Each instruction is executed on exact integers (`isa.Kernel`) and counted in its class.

- `isa`: the counted instruction set.
- `dither`: the instance (At = A + R, Bt = B + S, S_tj = U_(j, t mod L)), the honest transcript and the folds.
- `routes`: honest routes (the tensor kernel, CUDA-core DP4A and FFMA, general-A Strassen) and the attacks that need
  A = B = 0 (Strassen on the narrow cross term, IMAD packing, POPC planes, bit-slicing, Four Russians).
- `folds`: the receipt folds on their pipe routes (attack (e)).
- `price`: H200 resources, standalone time (an LP over pipe routes) and R-ACCEL-1's LP with a GEMM filler.
- `lab`: runs everything and prints the table (``python -m attack_lab`` from ``benchmarks/pouw``).
"""
