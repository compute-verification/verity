# Direction `seed-mezo` (class `seed`)

Zeroth-order training whose update is a seed log: MeZO and its variants, and the loss they buy per training FLOP and per bit
of log. Launch it with `--island seed-mezo`; `islands/seed/train.py` is a symlink to this `train.py`, whose docstring has
the options. `BOARD.md` holds this direction's board lines.

What sets MeZO's gain, and what the runs measure: a step along z with coefficient c = η ĝ, ĝ = g_B · z, changes the loss by
−η ĝ (g · z) + η² ĝ² z^T H z / 2. With z isotropic over the named set S, z^T H z ≈ tr H_S, so the best η is
|g_S|² / (E[ĝ²] tr H_S) and the gain a direction is |g_S|⁴ / (2 E[ĝ²] tr H_S), where E[ĝ²] = |g_S|² + V / batch, V the
per-sequence noise. The seed's lr 1e-6 sits orders of magnitude below that best η, which is why it didn't learn.

`--lr-auto` takes tr H_S from per-group diagnostics before the first step, and |g_S|² and E[ĝ²] from the run's own steps:
the product of the two half-batch estimates, ĝ_A ĝ_B, is an unbiased |g_S|², and both are EMAs over ~1000 steps. The
diagnostics' |g_S|² (a few hundred sequences) is too noisy to set the lr: it came out about 2x the measured value on every
subset tried, and the one-matrix subset it picks as the best of 37 is biased up besides.

What the s50 runs found (BOARD.md has the records): V is ~0.1 a sequence on one matrix and ~0.85 on all blocks, so the
batch where signal equals noise is ~100-200 sequences. A one-matrix subset has the highest gain a direction at θ, but it
runs out: its measured |g_S|² falls to ~0 within ~3k steps (η·steps ~ 13 descends every gradient component with curvature
above ~0.07), so it buys −0.04% and then nothing. The gain a direction at θ is a myopic score.

Each MeZO step declares its two perturbed weight values as the log so far plus a one-entry log (`prepare.declare_step`), and
the diagnostics declare their curvature probes, so the harness credits every step and ρ is 1. Its rebuild of step t replays
t entries over S, within c = 4096 × the forward MACs a token, so steps × |S| ≤ c: on s, all blocks (1.93e7 weights) allow
10,236 steps, i.e. batch ≥ 44 at the full budget; on m (5.66e7), 7,343 steps, batch ≥ 29. Below that the run refuses.

    python benchmarks/covert_witness/launch.py seed s50 --island seed-mezo --note "MeZO ..." -- --names blocks --lr-auto 1.0 --batch 48
