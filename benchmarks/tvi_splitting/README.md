# Splitting values across blocks: the toy attacks

The training lane's split-attack measurement for the covering bound (worker C3 of the training-vs-inference program).
It builds a small training graph at word level, rewrites it with three ways of splitting values across blocks, and solves
each one's cheapest `(s, k)`-covering exactly, the covering `Nci.Covering.Covers` states: blocks may overlap and
recompute, and every wire into a block that isn't in `θ` counts toward its `k`.

Nothing in `verity` reads this directory.

| Path | Holds |
| --- | --- |
| `toy.py` | the graph (two SGD steps of a one-layer model with an RMSNorm-like coupling, or a linear one) and the variants: `mono` (every value and dot product whole), `inner` (partial sums cross blocks), `shares` (additive weight shares), `+bits` (two bit-planes per word, carries between them); `check_equivalent` evaluates every variant against `mono` |
| `cover.py` | the cheapest covering: a direct MILP (`min_blocks`, small cases only) and column generation (`colgen`): a set-cover LP priced by a one-block MILP, with Farley's bound as the certified lower bound |
| `run.py` | the grid: `python run.py --out DIR [--workers N]`, writing `results.jsonl` and `summary.md` |

Results go to the evidence store, not here.
