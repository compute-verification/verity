"""ALGO-LAB: Dither's reference kernel and alternative exact algorithms for its words and receipts, executed on exact
integers, counted by instruction class and priced at the H200's data-sheet rates (ACCEL BET-ALGO-LAB, Linear CMP-125).

Every route computes Dither's transcript, the prefix words T^(l)_ij = sum_{t < L (l + 1)} At_it Bt_tj mod 2^32, and is
checked word for word against `dither.words`, and every receipt against `segment.py`'s folds and warp checkpoint.

- `isa`: the counted instruction set (exact lane semantics, one counter per class, MMAs refuse operands outside P1).
- `dither`: the instance (At = A + R, Bt = B + S, S_tj = U_(t mod L, j)), its words, and the receipts by `segment.py`.
- `price`: DS-90's prices (`machine_model/data/devices/sm_90/census/DS-90.csv`) a warp-word, time the max over resources.
- `reference`: (0), the H200 reference kernel (wgmma m64n256k32, then LI or duo, then the warp checkpoint).
- `strassen`, `packed`, `bitslice`, `russians`: (a) to (d), each producing the words another way.
- `folds`: (e), the receipt folds' alternative implementations and the per-word floors.
- `lab`: runs everything and prints the table (``python -m algo_lab`` from ``benchmarks/pouw``).
"""
