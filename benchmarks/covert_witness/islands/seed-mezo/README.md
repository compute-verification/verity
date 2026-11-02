# Direction `seed-mezo` (class `seed`)

Zeroth-order training whose update is a seed log: MeZO and its variants, and the loss they buy per training FLOP and per bit
of log. `train.py` runs as `islands/seed/train.py` (a symlink; the launcher runs `islands/CLASS/train.py`). Its docstring has
the options. `board.md` holds this direction's board lines, since the board branch refuses pushes from this agent.

What sets MeZO's gain, and what the runs measure: a step along z with coefficient c = η ĝ, ĝ = g_B · z, changes the loss by
−η ĝ (g · z) + η² ĝ² z^T H z / 2. With z isotropic over the named set S, z^T H z ≈ tr H_S, so the best η is
|g_S|² / (E[ĝ²] tr H_S) and the gain a direction is |g_S|⁴ / (2 E[ĝ²] tr H_S), where E[ĝ²] = |g_S|² + noise / batch
(+ the forward's rounding noise). The seed's lr 1e-6 sits orders of magnitude below that best η, which is why it didn't
learn. The levers are the subset S (its noise scale and curvature), the batch against the number of directions (the bits),
the forward's precision, and ε.

    python benchmarks/covert_witness/launch.py seed s50 --note "MeZO ..." -- --names blocks --diag --lr-auto 1.0
