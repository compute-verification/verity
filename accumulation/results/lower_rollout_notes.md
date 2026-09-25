# lower_coarse -- dynamic-rollout certificate notes (frozen 2026-09-16 04:30)

Frozen hashes: `module_versions()['lower'] = ac6c3bb0e3` (`bounds/lower.py` 8c8626f5f6 + `bounds/lower_coarse.py`
df35839fdf). Validation at this hash: `tests/test_lower.py` + `tests/test_lower_coarse.py` 157 passed;
`redteam.coarse_validation --fuzz 200 --reuse-istar` 0/1772 `L > I*`; `redteam.block_term` (tiny2 local-sgd,
K = 2, 3) 0/51 violations, 15/51 block-term cells fired; tiny2 `forward-nonfixed` + tiny-moe `forward-nonfixed-moe`
rollout cells (60, `redteam.rollout_scale --reuse-brackets`) 0 violations, L = I* on all 14 exactly solved cells.

## What changed tonight (in order)

1. MoE-aware block cells: `BlockInfo.{E, topk, P_dense, P_exp, T_e}`; per-layer fwd MACs = (active params per
   token) x Q; the bilinear (tokens x weights) relaxation is split into a dense piece (`m_d <= Q w_d`) and an expert
   piece (`m_x <= T_e w_x`, `T_e = Q topk / E`: an expert element is amortised over the tokens routed to it only).
   Extractor: `Edge.ranges` of an axis-0 batched argument are widened over the batch members, so the read-root
   floor (`root_read`) covers the *stacked* expert weights (before: one expert's window).
2. `LowerBound` metadata: `gap`, `root_read`, `recurring`, `certified`, `detail` (see below).
3. Chain partition (C): the working tokens of a block are split `A = W & I_l`, `B = (W & I_{l-1}) minus I_l`,
   `P = hoppers`, with `n^P_l <= n^B_{l-1} + n^P_{l-1}`. Before, `n^W <= n^I + n^P + n^I_{l-1}` let the same 150K
   tokens be counted as `I_{l-1}` *and* as hoppers, so an RU got 2x the MACs for one row import (8B/1M: 0.0622 ->
   0.0768 MB/tok).
4. Token slabs (single-step circuits): the shape MILP is solved once per slab `max token count in [lo, hi]`
   (ratio sqrt 2, 12 slabs) with `m <= hi w`; removes the `m <= Q w` slack (partial weights serving all Q tokens)
   (8B/1M: 0.0768 -> 0.0847 MB/tok). Cost: ~13 MILPs per separation.
5. Price refinement: after the (step, class) prices converge, warm-started cutting planes over (step, class,
   dyadic depth band) groups, 120 s budget (`LOWER_COARSE_REFINE_S`); never worse than the coarse vector.
   Gain: +0.2 % on dense 8B/1M, 0 on mixtral/1M -- kept because it is cheap and helps when restarts from the free
   block-0 rows are the cheapest attack.

## Headline numbers at the frozen hash (local, default point F_hat 1.5, G_hat 4, Q_inf 8192)

| model | Q_roll | L MB/tok | recurring MB/tok | root read GB | wall s |
|---|---:|---:|---:|---:|---:|
| llama3-8b | 1M | 0.0847 | 0.0694 | 16.06 | ~120 |
| llama3-405b | 1M | 1.5013 | 0.7272 | 811.7 | 1057 |
| mixtral-8x7b | 1M | 0.1337 | 0.0446 | 93.4 | 134 |
| mixtral-8x7b | 131K | 0.7183 | 0.0037 | 93.4 | 330 |
| deepseek-v3 | 4M | 0.4286 | 0.1069 | 1349.4 | 1934 |

(Other P0 / P1 / P4 cells are being recomputed at this hash; the pod `--redo stale` pass is authoritative.)
Acceptance test: 8B/1M went 0.0559 -> 0.0847 MB/tok (certificate improvements 3-4 above); U/L with the current
planner U = 0.125 is 1.48.

## What limits the certificate (worst-RU shapes, 8B/1M)

The binding cuts at the certified prices are honest tiles: (i) layers 1-5 x 262K tokens started from the free
block-0 rows (1.9 GB imported: weights only; 318T MACs) and (ii) layers 22-27 x 262K tokens with the entry rows
imported at layer 22 (4.0 GB; 318T MACs). Both are legal RUs; the certificate is the LP (fractional set-cover)
dual over these shapes.

Known slack, in order of size:

1. Free block-0 rows: embedding gathers from the dynamic table are not charged per RU (token ids may repeat;
   only the union of read table rows is charged once). Restart tiles are therefore cheap for shallow layers; band
   prices absorb most of it (price of band 0 = 3.5e-6 vs 11.2e-6 B/MAC for deep bands on mixtral).
2. Fractional cover: signed prices are a relaxation of the integral partition (the analytic cost of the best
   tiling under G alone is ~0.105 MB/tok for 8B/1M vs L = 0.0847).
3. Slab ratio: within a slab a weight element may serve up to sqrt 2 x the true token count (<= 41 % on the
   weight part of a tile; ratio 2 gave 0.0823).
4. `F` is not used for single-step circuits (`z` vacuous). For a multi-sequence rollout the per-gate upstream
   cone is one sequence (8192 tokens x 33 layers < F), so F is genuinely non-binding at the default point.
5. MoE: `m_x <= T_e w_x` allows T_e tokens to share the same topk experts. Under the registry's static balanced
   routing this is exact (the `topk` token blocks `b = b0 mod E/topk` share one expert set), so it is not slack for
   these circuits; a data-dependent router would make it a relaxation.
6. Partial MoE outputs (a token's MoE sum computed with only some of its experts present) are not charged by (X);
   bounded indirectly by 5.

What would tighten it: charging the table rows per distinct token id (needs an assumption on the id
distribution, not sound as-is); exact-partition (integer) pricing over the binding tiles; slabs of ratio 2^(1/4)
(2x MILP cost); using the routing pattern to bound tokens per expert *set* for real routers.

## Runtime and knobs

Shape MILP: HiGHS, 1 thread (`LOWER_COARSE_THREADS`), time limit `LOWER_COARSE_MILP_S` = 180 s per MILP; at a
time-out the solver's dual bound is used (L is always the proven bound, never an incumbent), the cutting plane
stops after 3 unclosed separations and returns the last certified centre, and the final re-check sets
`certified = False` / `recheck = "time limit"` if it cannot be closed. Slabs: `LOWER_COARSE_MC_ENUM` (12) /
`LOWER_COARSE_MC_RATIO` (sqrt 2); refinement `LOWER_COARSE_REFINE_S` (120 s). deepseek-v3 is the slowest
(62 MoE blocks, E = 256): 32 min at 4M locally.

## Metadata (`LowerBound`)

- `gap`: (master LP upper bound on the certifiable total - certified total) / L; 0 when the cutting plane closed
  or when the root floor dominates.
- `root_read`: bytes of dynamic roots read by some op (union of read ranges), charged once; `recurring = total -
  root_read - token bytes`.
- `certified`: the returned price vector passed the independent separation re-check (every legal shape pays at
  least its charge, proven bound >= -tol).
- `detail`: K, L, Q, d, V, P per block, active params per block, verified counts, hop constants, lambda per group,
  mu, iterations, recheck, cheapest shape (tokens per cell), MILP time limit, MoE geometry (E, topk, T_e,
  P_dense, P_exp, P_act per block).
