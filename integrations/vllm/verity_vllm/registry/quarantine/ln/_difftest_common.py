"""Case scheduling shared by the lane's `*_difftest.py` adapters (torch-free; no definitions).  `difftest check` requires at
least one case per edge kind (zeros / max / subnormal / alternating); the adapters cycle through the kinds so any `--n >= 5`
covers them, and cycle N over the shapes Pythia-160M uses (H = 768, I = 3072) plus small ones."""
from __future__ import annotations

KINDS = ("random", "zeros", "max", "subnormal", "alternating")


def kind_for(i: int) -> str:
    return KINDS[i % len(KINDS)]


def pick(seq, i: int):
    return seq[(i // len(KINDS)) % len(seq)]
