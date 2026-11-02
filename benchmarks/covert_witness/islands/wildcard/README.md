# Island `wildcard`: updates the theorem prices at ρ = 1, trained as cheaply as possible

The program of the training-efficiency search's `wildcard` direction (any class). `train.py` is the one file it edits;
`launch.py` ships this island from the branch `cursor/te-wildcard-NNNN` and runs `train.py` with the named class's caps.

## The structural lever

ρ = max(1, φcβ / (k(k + β))) when D ≥ k/β, else 1, with D the updated multiplications per token upstream of the deepest
updated one. An update confined to one (layer, stage) position has no updated multiplication upstream of another, so
D = 0 and ρ = 1 at every (k, r), whatever E's size. Within a block the parallel positions are q/k/v (stage 1) and
gate/up (stage 4). Norm gains are 0-FLOP nodes in the witness, so gains downstream of the stage (`norms-after`), or the
gains alone (`norms`), keep φ = 0. `train.py` refuses a set whose witnessed φ is above 0 unless `--allow-deep`.

## Mechanisms

- `--method adamw|muon`: the set trained in full for the budget, E the trained tensors (`encode_full`).
- `--method kfac`: closed form, one damped natural-gradient step per matrix from statistics gathered at θ.
- `--set`: `mlp-in:B`, `qkv:B`, `o:B`, `down:B` (B < 0 from the top), `norms`, `norms-after`, `final`, or a regex.

Run: `python benchmarks/covert_witness/launch.py shallow s50 --note "onestage-mlpin ..." -- --set mlp-in:-1 --lr 1e-3`.
The note's first word is the mechanism the scorer files the record under.
