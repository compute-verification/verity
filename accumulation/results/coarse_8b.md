# 8B coarse upper bounds `U` (acyclic RU quotient) -- llama3-8b, `Q_inf = 8192`, `F_hat = 1.5`

Generated 2026-09-15 19:17 by `accumulation/bounds/coarse.py` sha1 `48ee653bf3`.  Calibration: `fwd` = checker total `Work` of the inference circuit at `Q_inf` = 7.96525e+13 (`F = 1.5 fwd`, `G = G_hat fwd`).  Every `U` below is a plan that passed `check_coarse_plan` (`Work <= G`, `Up <= F` per unit, acyclic RU quotient).  The `cyclic` columns are the superseded family (`joint` fwd+bwd chunk units / `kf`-fused chunks; `search=dict(acyclic=False)`): NOT legal partitions, shown for comparison only.

## Inference (`inference-dense`, one session)

| G_hat | U [B] | b_inf [B/token] | family | acyclic |
|---|---|---|---|---|
| 1 | 32768 | 4.0 | single | True |
| 4 | 32768 | 4.0 | single | True |
| 16 | 32768 | 4.0 | single | True |
| 100 | 32768 | 4.0 | single | True |

## Training (`local-sgd`, `K` steps of `Q_step` tokens, sequences of 8192)

### `Q_step = 8192`

| G_hat | K | U/Q acyclic [MB/token] | U/Q cyclic (superseded) | weights | activations | dW partials | units | plan (acyclic) | plan (cyclic) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 3.535 | 1.120 | 93% | 7% | 0% | 11 | fwd 32x1, bwd 10x1, top T=6 s_t=1, chain/fused-bwd | m=4 s=1 kf=2 chain/joint/fused-bwd |
| 1 | 3 | 3.513 | 0.924 | 93% | 7% | 0% | 16 | fwd 32x1, bwd 10x1, top T=6 s_t=1, chain/fused-bwd | m=2 s=1 kf=3 chain/joint/fused-bwd |
| 4 | 2 | 3.150 | 1.087 | 94% | 6% | 0% | 7 | fwd 20x1, bwd 20x1, top T=12 s_t=1, chain/fused-bwd | m=6 s=1 kf=2 chain/joint/fused-bwd |
| 4 | 3 | 3.128 | 0.793 | 94% | 6% | 0% | 10 | fwd 20x1, bwd 20x1, top T=12 s_t=1, chain/fused-bwd | m=4 s=1 kf=3 chain/joint/fused-bwd |
| 16 | 2 | 3.150 | 1.087 | 94% | 6% | 0% | 7 | fwd 20x1, bwd 20x1, top T=12 s_t=1, chain/fused-bwd | m=6 s=1 kf=2 chain/joint/fused-bwd |
| 16 | 3 | 3.128 | 0.793 | 94% | 6% | 0% | 10 | fwd 20x1, bwd 20x1, top T=12 s_t=1, chain/fused-bwd | m=4 s=1 kf=3 chain/joint/fused-bwd |
| 100 | 2 | 3.150 | 1.087 | 94% | 6% | 0% | 7 | fwd 20x1, bwd 20x1, top T=12 s_t=1, chain/fused-bwd | m=6 s=1 kf=2 chain/joint/fused-bwd |
| 100 | 3 | 3.128 | 0.793 | 94% | 6% | 0% | 10 | fwd 20x1, bwd 20x1, top T=12 s_t=1, chain/fused-bwd | m=4 s=1 kf=3 chain/joint/fused-bwd |

Marginal `b_train = (U(K=3) - U(K=2)) / Q_step` at `Q_step = 8192`:

| G_hat | b_train acyclic [MB/token] | b_train cyclic (superseded) | ratio | marginal by class (acyclic): weights / activations / partials [MB/token] |
|---|---|---|---|---|
| 1 | 3.471 | 0.532 | 6.52x | 3.217 / 0.254 / 0.000 |
| 4 | 3.086 | 0.205 | 15.07x | 2.897 / 0.188 / 0.000 |
| 16 | 3.086 | 0.205 | 15.07x | 2.897 / 0.188 / 0.000 |
| 100 | 3.086 | 0.205 | 15.07x | 2.897 / 0.188 / 0.000 |

### `Q_step = 131072`

| G_hat | K | U/Q acyclic [MB/token] | U/Q cyclic (superseded) | weights | activations | dW partials | units | plan (acyclic) | plan (cyclic) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 1.256 | 1.338 | 28% | 53% | 20% | 167 | fwd 2x16, bwd 1x8, top T=0 s_t=6, chain/fused-bwd | m=1 s=6 kf=1 chain/joint/fused-bwd |
| 1 | 3 | 1.255 | 1.337 | 28% | 53% | 20% | 250 | fwd 2x16, bwd 1x8, top T=0 s_t=6, chain/fused-bwd | m=1 s=6 kf=1 chain/joint/fused-bwd |
| 4 | 2 | 0.872 | 0.732 | 27% | 71% | 2% | 91 | fwd 3x16, bwd 1x16, top T=0 s_t=8, chain/fused-bwd | m=2 s=8 kf=1 chain/joint/fused-bwd |
| 4 | 3 | 0.870 | 0.731 | 27% | 72% | 2% | 136 | fwd 3x16, bwd 1x16, top T=0 s_t=8, chain/fused-bwd | m=2 s=8 kf=1 chain/joint/fused-bwd |
| 16 | 2 | 0.872 | 0.732 | 27% | 71% | 2% | 91 | fwd 3x16, bwd 1x16, top T=0 s_t=8, chain/fused-bwd | m=2 s=8 kf=1 chain/joint/fused-bwd |
| 16 | 3 | 0.870 | 0.731 | 27% | 72% | 2% | 136 | fwd 3x16, bwd 1x16, top T=0 s_t=8, chain/fused-bwd | m=2 s=8 kf=1 chain/joint/fused-bwd |
| 100 | 2 | 0.872 | 0.732 | 27% | 71% | 2% | 91 | fwd 3x16, bwd 1x16, top T=0 s_t=8, chain/fused-bwd | m=2 s=8 kf=1 chain/joint/fused-bwd |
| 100 | 3 | 0.870 | 0.731 | 27% | 72% | 2% | 136 | fwd 3x16, bwd 1x16, top T=0 s_t=8, chain/fused-bwd | m=2 s=8 kf=1 chain/joint/fused-bwd |

Marginal `b_train = (U(K=3) - U(K=2)) / Q_step` at `Q_step = 131072`:

| G_hat | b_train acyclic [MB/token] | b_train cyclic (superseded) | ratio | marginal by class (acyclic): weights / activations / partials [MB/token] |
|---|---|---|---|---|
| 1 | 1.252 | 1.334 | 0.94x | 0.344 / 0.664 / 0.245 |
| 4 | 0.868 | 0.728 | 1.19x | 0.229 / 0.623 / 0.016 |
| 16 | 0.868 | 0.728 | 1.19x | 0.229 / 0.623 / 0.016 |
| 100 | 0.868 | 0.728 | 1.19x | 0.229 / 0.623 / 0.016 |

## What carries the convex `U`

Default point (`F_hat 1.5, G_hat 4, Q_inf 8192, Q_step 131072`), per step, from the checked plan (`fwd 3 layers x 16 seq`, `bwd 1 layer x 16 seq`, head-only top unit `s_t = 8`, `chain` wgrad, updates fused into the unit that completes `dW`):

| carrier | per step | MB/token | why |
|---|---|---|---|
| block weights, forward units | 11 chunks x 3 layers = 14.0 GB (all 32 layers once) | 0.107 | every forward unit spans all 131072 tokens, so each layer's weights enter once for the forward |
| block weights, backward units | 32 chunks x 1 layer = 14.0 GB (all 32 layers again) | 0.107 | a layer's forward and backward cannot share a unit (2-cycle through the head) -> the weights enter a second time |
| LM head weights + `dW` accumulators | 2 x 1.05 GB + 2.1 GB partials | 0.024 | head split in 2 sequence units of 65536 tokens (`Up <= F`); the 32-bit head accumulator crosses once |
| checkpoint rows into backward units | 32 row sets (1.07 GB each = `Q_step x d x 2 B`) | 0.262 | `AccBlockBwd` recomputes layer `l` from `x_l`; `x_l` is produced by a forward unit, so every layer's backward imports its own checkpoint |
| gradient rows into backward units | 32 row sets | 0.262 | `dY` from the unit above (head or bwd of `l+1`); `m_b = 1` because `F = 1.5 fwd` caps a backward unit's `Up` at ~1 layer x 131072 tokens (recompute + dgrad ~ 3.6 fwd-equivalents per 32-layer step) |
| forward chunk boundaries + head input | 11 + 1 row sets | 0.098 | `x` rows crossing between forward chunks and into the head |
| **total** | **228.5 GB / 2 steps** | **0.872** | weights 27 %, activations 71 %, partials 2 % |

Superseded cyclic witness at the same point: 0.732 MB/token (`m=2, s=8, joint fwd+bwd, chain, fused-bwd`) -- it imported each layer's weights once (joint unit) but paid 8 sequence units of `dW` partial crossings; the convex answer is 1.19x higher, carried by the second weight pass and the per-layer checkpoint + gradient row sets (a convex backward unit cannot contain the forward it needs).  At `Q_step = 8192` the gap is 15x: the cyclic family fused all `K` steps of a chunk (`kf = K`, weights enter once for the whole run), which is a cycle through the rest of each step; convex units re-import the weights every step (fwd + bwd + top stack), so `b_train` there is ~1.9 x model bytes / `Q_step` = 3.1 MB/token.

