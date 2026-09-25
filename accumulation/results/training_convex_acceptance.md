# Convex training family: acceptance (2026-09-16)

`bounds/coarse.py` @ `f4baa2d7be`. Harnesses `/tmp/tiny2_sgd.py`, `/tmp/train8b*.py` (not kept).

## What changed

- **Structural memo of unit evaluations** in the convex search (`_Planner._unit_cache`, keyed by unit label +
  candidate fields `c, emb, upd, wgrad, kf` + `red_last`): per-sequence member lists are only materialised on a
  miss, so the fwd / bwd / top part searches share units across `T`, `wgrad`, `emb` and the other parts'
  `(m, s)`. Labels were made unambiguous where they were not (`wgrad[top a:b]`, `update[top a:b]`); the chosen
  candidate's memoised estimate is stored as `detail['planner_cost']` and `test_training_marginal_positive_and_consistent`
  asserts it equals the checked total.
- **Forward tail under the top unit** (`_CCand.af`): the top unit's forward may start at layer `af < L - T`
  while its backward still starts at `L - T`; the bwd units of `[af, L - T)` read its checkpoints / gradient
  rows (convex). Depths tried: `L - T - m` for `m` in the powers of two of the layer candidates, plus the whole
  stack; `search=dict(af=[...])` pins them. It buys 8 % at tiny2 (`U` 3136 -> 2880 at `G_hat >= 4`, 4264 -> 4008
  at `G_hat = 1`) and nothing at 8B (the head unit is sequence-split there, so a tail would re-import weights).
- **Anatomy per unit class** (`anatomy['by_unit_class']`): count, max `Work` / `Up` (fwd units and MACs),
  `Up/Work`, imports, and which cap binds (`F` / `G` / none).

## 8B default point (`local-sgd`, Q_step 131072, seq 8192, F_hat 1.5, G_hat 4), planned locally

| K | plan time | peak RSS | units | U / (K Q_step) | weights | activations | partials |
|---|---|---|---|---|---|---|---|
| 2 | 200-290 s | 1.2 GB | 76 | 810,144 B/token | 237,048 | 557,056 | 16,032 |
| 3 | 320-376 s | 1.25 GB | 114 | 810,144 B/token | 237,048 | 557,056 | 16,032 |

(range = before / after the `af` dimension; 70B K=2 planned in 21 min / 1.75 GB before `af`.)  Marginal
`(U(3) - U(2)) / Q_step` = 810,144 B/token: the per-step cost is exactly reproduced.

Chosen plan (both K): `fwd[m_f=8] x all 16 sequences` (4 units/step, **G-bound**, 3.78 fwd of 4),
`head x 8 sequences` (2/step, 1.30 fwd), `bwd[m_b=1] x all 16 sequences` (32/step, **F-bound**: `Up = Work =
1.375 fwd` -- the layer's `dW` reduction couples every token), `wgrad=chain`, `upd=fused-bwd`, `emb=fused`.

| class | n/step | Work max (fwd) | Up max (fwd) | Up/Work | imports B/token | binds |
|---|---|---|---|---|---|---|
| fwd | 4 | 3.784 | 0.236 | 0.062 (= 1/16, one sequence) | 139,096 | G |
| head | 2 | 1.298 | 1.298 | 1.0 | 40,260 | none |
| bwd | 32 | 1.375 | 1.375 | 1.0 (`dW` couples all sequences) | 630,788 | F |

What carries the cost: every bwd unit imports its layer's checkpoint rows + the gradient rows (2.15 GB) and its
weights again (0.44 GB) -- 32 x 2.58 GB = 83 GB/step of the 106 GB/step. This is the structural F cost: a
bwd unit of `m_b > 1` layers still needs one checkpoint per layer (the backward composite recomputes each layer
from its own checkpoint) and a sequence-split `chain` of `dW` partials costs `4 B x P_layer` per boundary, more
than the gradient rows it saves; a whole-model per-sequence top stack (legal under `F`) re-imports all weights
per sequence. The planner evaluated all of these (`n_candidates` ~ 12k) and they lose.

## tiny2 `local-sgd` K = 2, 3 (Q_inf 16, seq 4, F_hat 1.5): `U` vs the certificate `L`

Exact `I*` is out of reach for the registry training circuit (2888 gates at `tiny`, 5776+ at tiny2:
`redteam/coarse_validation.py` marks these cells "beyond exact scale"), so the acceptance ratio is `U / L`
with `L = lower_coarse`; the micro local-SGD chains that *are* exact give `U/I*` median 1.00-1.13 (d=1 chains,
K=1..3, n=72 cells; max 1.25), 0/972 soundness violations in the full validation run at this coarse.


| K | G_hat | U | L | U/L | units/step | plan | unit classes (Up, Work in fwd; binds) |
|---|---|---|---|---|---|---|---|
| 2 | 1 | 4008 | 1142 | 3.51 | 3 | T=0, s_t=4, chain, fused-bwd, emb=fused | `{'head': (0.682, 0.682, 'none'), 'bwd': (0.673, 0.673, 'none')}` |
| 2 | 4 | 2880 | 1004 | 2.87 | 2 | T=1, s_t=4, chain, fused-bwd, emb=fused | `{'top': (1.356, 1.356, 'F'), 'bwd': (0.673, 0.673, 'none')}` |
| 2 | 16 | 2880 | 1004 | 2.87 | 2 | T=1, s_t=4, chain, fused-bwd, emb=fused | `{'top': (1.356, 1.356, 'F'), 'bwd': (0.673, 0.673, 'none')}` |
| 3 | 1 | 6012 | 1742 | 3.45 | 3 | T=0, s_t=4, chain, fused-bwd, emb=fused | `{'head': (0.682, 0.682, 'none'), 'bwd': (0.673, 0.673, 'none')}` |
| 3 | 4 | 4320 | 1147 | 3.77 | 2 | T=1, s_t=4, chain, fused-bwd, emb=fused | `{'top': (1.356, 1.356, 'F'), 'bwd': (0.673, 0.673, 'none')}` |
| 3 | 16 | 4320 | 1147 | 3.77 | 2 | T=1, s_t=4, chain, fused-bwd, emb=fused | `{'top': (1.356, 1.356, 'F'), 'bwd': (0.673, 0.673, 'none')}` |

Per step the plan is `head+fwd[0:2]` + `bwd[1:2]` + `bwd[0:1]` (G_hat = 1: three whole-token units, G-bound)
or `top[fwd 0:2, bwd 1:2]+head` + `bwd[0:1]` (G_hat >= 4: the new `af` tail, F-bound at 1.356 fwd). `U/L` = 2.9-3.8; `L` sits below one
weight import per step (502 B/step vs W = 744 B), so most of the gap is the certificate's, not the plan's:
the plan imports the weights 1.4x per step plus the two activation boundaries that `F` forces.
