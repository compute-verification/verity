"""Case scheduling shared by the lane's `*_difftest.py` adapters (torch-free; no definitions).

`difftest check` requires at least one case per edge kind (zeros / max / subnormal / alternating); the adapters cycle through
the kinds so any `--n >= 5` covers them, and cycle the specialization's N over the shapes Gemma-2 actually uses plus small
ones."""
from __future__ import annotations

KINDS = ("random", "zeros", "max", "subnormal", "alternating")


def kind_for(i: int) -> str:
    return KINDS[i % len(KINDS)]


def pick(seq, i: int):
    return seq[(i // len(KINDS)) % len(seq)]


def one_input(i: int, N: int, dtype: str, kind: str | None = None) -> dict:
    k = kind_for(i) if kind is None else kind
    return {"statics": {"N": N}, "inputs": [{"shape": [N], "dtype": dtype, "kind": k}], "edge": k != "random"}
