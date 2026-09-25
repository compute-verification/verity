# Coarse planner validation against the exact solver (`X = inf`)

Generated 2026-09-15 19:32:44.  `U` = `upper_coarse` (every plan through `check_coarse_plan`, then mapped to a gate assignment and re-costed / re-checked by `accumulation.exact.solve`); `I*` = `exact_min_input` (time box 20 s) where the circuit is within exact scale.  Same circuits and `(F, G)` grid as `wedge_exact.json` (`fwd` = total work of the inference forward of the same geometry).

**Aggregate.** 336 cells; `U` available in 318 (the rest are infeasible at op-slice granularity: a single contraction chunk of one op exceeds `min(F, G)`); gate-level legal in 318/318; `U == gate cost` in all but 0 (where the coarse import accounting is conservative: `U > cost`); exact `I*` proved in 230 cells, `U == I*` in 176 of them, max `U/I*` = 2.200; soundness violations (`U < I*`): 0.

## micro inference

| circuit | F | G | U (B) | family | units | (m, s, kf) | gate cost | gate legal | I* | U/I* |
|---|---|---|---|---|---|---|---|---|---|---|
| micro_inference[q=1,d=2] | 1·fwd (6) | 1·fwd (6) | 4 | single | 1 | - | 4 | True | 4 | 1.000 |
| micro_inference[q=1,d=2] | 1·fwd (6) | 4·fwd (24) | 4 | single | 1 | - | 4 | True | 4 | 1.000 |
| micro_inference[q=1,d=2] | 1·fwd (6) | inf (-) | 4 | single | 1 | - | 4 | True | 4 | 1.000 |
| micro_inference[q=1,d=2] | 1.5·fwd (9) | 1·fwd (6) | 4 | single | 1 | - | 4 | True | 4 | 1.000 |
| micro_inference[q=1,d=2] | 1.5·fwd (9) | 4·fwd (24) | 4 | single | 1 | - | 4 | True | 4 | 1.000 |
| micro_inference[q=1,d=2] | 1.5·fwd (9) | inf (-) | 4 | single | 1 | - | 4 | True | 4 | 1.000 |
| micro_inference[q=1,d=2] | 3·fwd (18) | 1·fwd (6) | 4 | single | 1 | - | 4 | True | 4 | 1.000 |
| micro_inference[q=1,d=2] | 3·fwd (18) | 4·fwd (24) | 4 | single | 1 | - | 4 | True | 4 | 1.000 |
| micro_inference[q=1,d=2] | 3·fwd (18) | inf (-) | 4 | single | 1 | - | 4 | True | 4 | 1.000 |
| micro_inference[q=1,d=2] | inf (-) | 1·fwd (6) | 4 | single | 1 | - | 4 | True | 4 | 1.000 |
| micro_inference[q=1,d=2] | inf (-) | 4·fwd (24) | 4 | single | 1 | - | 4 | True | 4 | 1.000 |
| micro_inference[q=1,d=2] | inf (-) | inf (-) | 4 | single | 1 | - | 4 | True | 4 | 1.000 |
| micro_inference[q=2,d=2] | 1·fwd (12) | 1·fwd (12) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| micro_inference[q=2,d=2] | 1·fwd (12) | 4·fwd (48) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| micro_inference[q=2,d=2] | 1·fwd (12) | inf (-) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| micro_inference[q=2,d=2] | 1.5·fwd (18) | 1·fwd (12) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| micro_inference[q=2,d=2] | 1.5·fwd (18) | 4·fwd (48) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| micro_inference[q=2,d=2] | 1.5·fwd (18) | inf (-) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| micro_inference[q=2,d=2] | 3·fwd (36) | 1·fwd (12) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| micro_inference[q=2,d=2] | 3·fwd (36) | 4·fwd (48) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| micro_inference[q=2,d=2] | 3·fwd (36) | inf (-) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| micro_inference[q=2,d=2] | inf (-) | 1·fwd (12) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| micro_inference[q=2,d=2] | inf (-) | 4·fwd (48) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| micro_inference[q=2,d=2] | inf (-) | inf (-) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |

## micro local-sgd chain (d=1, q=1)

| circuit | F | G | U (B) | family | units | (m, s, kf) | gate cost | gate legal | I* | U/I* |
|---|---|---|---|---|---|---|---|---|---|---|
| local_sgd_chain[K=1,q=1,d=1] | 1·fwd (2) | 1·fwd (2) | 20 | bands | 4 | - | 20 | True | 20 | 1.000 |
| local_sgd_chain[K=1,q=1,d=1] | 1·fwd (2) | 4·fwd (8) | 20 | bands | 4 | - | 20 | True | 20 | 1.000 |
| local_sgd_chain[K=1,q=1,d=1] | 1·fwd (2) | inf (-) | 20 | bands | 4 | - | 20 | True | 20 | 1.000 |
| local_sgd_chain[K=1,q=1,d=1] | 1.5·fwd (3) | 1·fwd (2) | 20 | bands | 4 | - | 20 | True | 20 | 1.000 |
| local_sgd_chain[K=1,q=1,d=1] | 1.5·fwd (3) | 4·fwd (8) | 18 | bands | 3 | - | 18 | True | 18 | 1.000 |
| local_sgd_chain[K=1,q=1,d=1] | 1.5·fwd (3) | inf (-) | 18 | bands | 3 | - | 18 | True | 18 | 1.000 |
| local_sgd_chain[K=1,q=1,d=1] | 3·fwd (6) | 1·fwd (2) | 20 | bands | 4 | - | 20 | True | 20 | 1.000 |
| local_sgd_chain[K=1,q=1,d=1] | 3·fwd (6) | 4·fwd (8) | 14 | bands | 2 | - | 14 | True | 14 | 1.000 |
| local_sgd_chain[K=1,q=1,d=1] | 3·fwd (6) | inf (-) | 14 | bands | 2 | - | 14 | True | 14 | 1.000 |
| local_sgd_chain[K=1,q=1,d=1] | inf (-) | 1·fwd (2) | 20 | bands | 4 | - | 20 | True | 20 | 1.000 |
| local_sgd_chain[K=1,q=1,d=1] | inf (-) | 4·fwd (8) | 10 | single | 1 | - | 10 | True | 10 | 1.000 |
| local_sgd_chain[K=1,q=1,d=1] | inf (-) | inf (-) | 10 | single | 1 | - | 10 | True | 10 | 1.000 |
| local_sgd_chain[K=2,q=1,d=1] | 1·fwd (2) | 1·fwd (2) | 40 | bands | 8 | - | 40 | True | 40 | 1.000 |
| local_sgd_chain[K=2,q=1,d=1] | 1·fwd (2) | 4·fwd (8) | 40 | bands | 8 | - | 40 | True | 40 | 1.000 |
| local_sgd_chain[K=2,q=1,d=1] | 1·fwd (2) | inf (-) | 40 | bands | 8 | - | 40 | True | 40 | 1.000 |
| local_sgd_chain[K=2,q=1,d=1] | 1.5·fwd (3) | 1·fwd (2) | 40 | bands | 8 | - | 40 | True | 40 | 1.000 |
| local_sgd_chain[K=2,q=1,d=1] | 1.5·fwd (3) | 4·fwd (8) | 36 | bands | 6 | - | 36 | True | 34 | 1.059 |
| local_sgd_chain[K=2,q=1,d=1] | 1.5·fwd (3) | inf (-) | 36 | bands | 6 | - | 36 | True | 34 | 1.059 |
| local_sgd_chain[K=2,q=1,d=1] | 3·fwd (6) | 1·fwd (2) | 40 | bands | 8 | - | 40 | True | 40 | 1.000 |
| local_sgd_chain[K=2,q=1,d=1] | 3·fwd (6) | 4·fwd (8) | 28 | bands | 4 | - | 28 | True | 26 | 1.077 |
| local_sgd_chain[K=2,q=1,d=1] | 3·fwd (6) | inf (-) | 28 | bands | 4 | - | 28 | True | 26 | 1.077 |
| local_sgd_chain[K=2,q=1,d=1] | inf (-) | 1·fwd (2) | 40 | bands | 8 | - | 40 | True | 40 | 1.000 |
| local_sgd_chain[K=2,q=1,d=1] | inf (-) | 4·fwd (8) | 20 | bands | 2 | - | 20 | True | 20 | 1.000 |
| local_sgd_chain[K=2,q=1,d=1] | inf (-) | inf (-) | 14 | single | 1 | - | 14 | True | 14 | 1.000 |
| local_sgd_chain[K=3,q=1,d=1] | 1·fwd (2) | 1·fwd (2) | 60 | bands | 12 | - | 60 | True | 60 | 1.000 |
| local_sgd_chain[K=3,q=1,d=1] | 1·fwd (2) | 4·fwd (8) | 60 | bands | 12 | - | 60 | True | 60 | 1.000 |
| local_sgd_chain[K=3,q=1,d=1] | 1·fwd (2) | inf (-) | 60 | bands | 12 | - | 60 | True | 60 | 1.000 |
| local_sgd_chain[K=3,q=1,d=1] | 1.5·fwd (3) | 1·fwd (2) | 60 | bands | 12 | - | 60 | True | 60 | 1.000 |
| local_sgd_chain[K=3,q=1,d=1] | 1.5·fwd (3) | 4·fwd (8) | 54 | bands | 9 | - | 54 | True | 52 | 1.038 |
| local_sgd_chain[K=3,q=1,d=1] | 1.5·fwd (3) | inf (-) | 54 | bands | 9 | - | 54 | True | 52 | 1.038 |
| local_sgd_chain[K=3,q=1,d=1] | 3·fwd (6) | 1·fwd (2) | 60 | bands | 12 | - | 60 | True | 60 | 1.000 |
| local_sgd_chain[K=3,q=1,d=1] | 3·fwd (6) | 4·fwd (8) | 42 | bands | 6 | - | 42 | True | 40 | 1.050 |
| local_sgd_chain[K=3,q=1,d=1] | 3·fwd (6) | inf (-) | 42 | bands | 6 | - | 42 | True | 40 | 1.050 |
| local_sgd_chain[K=3,q=1,d=1] | inf (-) | 1·fwd (2) | 60 | bands | 12 | - | 60 | True | 60 | 1.000 |
| local_sgd_chain[K=3,q=1,d=1] | inf (-) | 4·fwd (8) | 30 | bands | 3 | - | 30 | True | 30 | 1.000 |
| local_sgd_chain[K=3,q=1,d=1] | inf (-) | inf (-) | 18 | single | 1 | - | 18 | True | 18 | 1.000 |

## micro local-sgd chain (d=1, q=2)

| circuit | F | G | U (B) | family | units | (m, s, kf) | gate cost | gate legal | I* | U/I* |
|---|---|---|---|---|---|---|---|---|---|---|
| local_sgd_chain[K=1,q=2,d=1] | 1·fwd (4) | 1·fwd (4) | 30 | bands | 4 | - | 30 | True | 26 | 1.154 |
| local_sgd_chain[K=1,q=2,d=1] | 1·fwd (4) | 4·fwd (16) | 30 | bands | 4 | - | 30 | True | 24 | 1.250 |
| local_sgd_chain[K=1,q=2,d=1] | 1·fwd (4) | inf (-) | 30 | bands | 4 | - | 30 | True | 24 | 1.250 |
| local_sgd_chain[K=1,q=2,d=1] | 1.5·fwd (6) | 1·fwd (4) | 30 | bands | 4 | - | 30 | True | 26 | 1.154 |
| local_sgd_chain[K=1,q=2,d=1] | 1.5·fwd (6) | 4·fwd (16) | 24 | bands | 2 | - | 24 | True | 24 | 1.000 |
| local_sgd_chain[K=1,q=2,d=1] | 1.5·fwd (6) | inf (-) | 24 | bands | 2 | - | 24 | True | 24 | 1.000 |
| local_sgd_chain[K=1,q=2,d=1] | 3·fwd (12) | 1·fwd (4) | 30 | bands | 4 | - | 30 | True | 26 | 1.154 |
| local_sgd_chain[K=1,q=2,d=1] | 3·fwd (12) | 4·fwd (16) | 14 | single | 1 | - | 14 | True | 14 | 1.000 |
| local_sgd_chain[K=1,q=2,d=1] | 3·fwd (12) | inf (-) | 14 | single | 1 | - | 14 | True | 14 | 1.000 |
| local_sgd_chain[K=1,q=2,d=1] | inf (-) | 1·fwd (4) | 30 | bands | 4 | - | 30 | True | 26 | 1.154 |
| local_sgd_chain[K=1,q=2,d=1] | inf (-) | 4·fwd (16) | 14 | single | 1 | - | 14 | True | 14 | 1.000 |
| local_sgd_chain[K=1,q=2,d=1] | inf (-) | inf (-) | 14 | single | 1 | - | 14 | True | 14 | 1.000 |
| local_sgd_chain[K=2,q=2,d=1] | 1·fwd (4) | 1·fwd (4) | 60 | bands | 8 | - | 60 | True | 52 | 1.154 |
| local_sgd_chain[K=2,q=2,d=1] | 1·fwd (4) | 4·fwd (16) | 60 | bands | 8 | - | 60 | True | 48 | 1.250 |
| local_sgd_chain[K=2,q=2,d=1] | 1·fwd (4) | inf (-) | 60 | bands | 8 | - | 60 | True | 48 | 1.250 |
| local_sgd_chain[K=2,q=2,d=1] | 1.5·fwd (6) | 1·fwd (4) | 60 | bands | 8 | - | 60 | True | 52 | 1.154 |
| local_sgd_chain[K=2,q=2,d=1] | 1.5·fwd (6) | 4·fwd (16) | 48 | bands | 4 | - | 48 | True | 46 | 1.043 |
| local_sgd_chain[K=2,q=2,d=1] | 1.5·fwd (6) | inf (-) | 48 | bands | 4 | - | 48 | True | 46 | 1.043 |
| local_sgd_chain[K=2,q=2,d=1] | 3·fwd (12) | 1·fwd (4) | 60 | bands | 8 | - | 60 | True | 52 | 1.154 |
| local_sgd_chain[K=2,q=2,d=1] | 3·fwd (12) | 4·fwd (16) | 28 | bands | 2 | - | 28 | True | 28 | 1.000 |
| local_sgd_chain[K=2,q=2,d=1] | 3·fwd (12) | inf (-) | 28 | bands | 2 | - | 28 | True | 28 | 1.000 |
| local_sgd_chain[K=2,q=2,d=1] | inf (-) | 1·fwd (4) | 60 | bands | 8 | - | 60 | True | 52 | 1.154 |
| local_sgd_chain[K=2,q=2,d=1] | inf (-) | 4·fwd (16) | 26 | bands | 2 | - | 26 | True | 26 | 1.000 |
| local_sgd_chain[K=2,q=2,d=1] | inf (-) | inf (-) | 22 | single | 1 | - | 22 | True | 22 | 1.000 |
| local_sgd_chain[K=3,q=2,d=1] | 1·fwd (4) | 1·fwd (4) | 90 | bands | 12 | - | 90 | True | timeout | - |
| local_sgd_chain[K=3,q=2,d=1] | 1·fwd (4) | 4·fwd (16) | 90 | bands | 12 | - | 90 | True | 72 | 1.250 |
| local_sgd_chain[K=3,q=2,d=1] | 1·fwd (4) | inf (-) | 90 | bands | 12 | - | 90 | True | 72 | 1.250 |
| local_sgd_chain[K=3,q=2,d=1] | 1.5·fwd (6) | 1·fwd (4) | 90 | bands | 12 | - | 90 | True | 78 | 1.154 |
| local_sgd_chain[K=3,q=2,d=1] | 1.5·fwd (6) | 4·fwd (16) | 72 | bands | 6 | - | 72 | True | 68 | 1.059 |
| local_sgd_chain[K=3,q=2,d=1] | 1.5·fwd (6) | inf (-) | 72 | bands | 6 | - | 72 | True | 68 | 1.059 |
| local_sgd_chain[K=3,q=2,d=1] | 3·fwd (12) | 1·fwd (4) | 90 | bands | 12 | - | 90 | True | timeout | - |
| local_sgd_chain[K=3,q=2,d=1] | 3·fwd (12) | 4·fwd (16) | 42 | bands | 3 | - | 42 | True | 42 | 1.000 |
| local_sgd_chain[K=3,q=2,d=1] | 3·fwd (12) | inf (-) | 42 | bands | 3 | - | 42 | True | 42 | 1.000 |
| local_sgd_chain[K=3,q=2,d=1] | inf (-) | 1·fwd (4) | 90 | bands | 12 | - | 90 | True | 78 | 1.154 |
| local_sgd_chain[K=3,q=2,d=1] | inf (-) | 4·fwd (16) | 40 | bands | 3 | - | 40 | True | 40 | 1.000 |
| local_sgd_chain[K=3,q=2,d=1] | inf (-) | inf (-) | 30 | single | 1 | - | 30 | True | 30 | 1.000 |

## micro local-sgd chain (d=2, q=1)

| circuit | F | G | U (B) | family | units | (m, s, kf) | gate cost | gate legal | I* | U/I* |
|---|---|---|---|---|---|---|---|---|---|---|
| local_sgd_chain[K=1,q=1,d=2] | 1·fwd (6) | 1·fwd (6) | infeasible at op-slice granularity: op 2 (AccMatmulTT_v1{M=2 | - | - | - | - | - | timeout | - |
| local_sgd_chain[K=1,q=1,d=2] | 1·fwd (6) | 4·fwd (24) | infeasible at op-slice granularity: op 2 (AccMatmulTT_v1{M=2 | - | - | - | - | - | 36 | - |
| local_sgd_chain[K=1,q=1,d=2] | 1·fwd (6) | inf (-) | infeasible at op-slice granularity: op 2 (AccMatmulTT_v1{M=2 | - | - | - | - | - | 36 | - |
| local_sgd_chain[K=1,q=1,d=2] | 1.5·fwd (9) | 1·fwd (6) | infeasible at op-slice granularity: op 2 (AccMatmulTT_v1{M=2 | - | - | - | - | - | timeout | - |
| local_sgd_chain[K=1,q=1,d=2] | 1.5·fwd (9) | 4·fwd (24) | 44 | bands | 3 | - | 44 | True | 20 | 2.200 |
| local_sgd_chain[K=1,q=1,d=2] | 1.5·fwd (9) | inf (-) | 44 | bands | 3 | - | 44 | True | 20 | 2.200 |
| local_sgd_chain[K=1,q=1,d=2] | 3·fwd (18) | 1·fwd (6) | infeasible at op-slice granularity: op 2 (AccMatmulTT_v1{M=2 | - | - | - | - | - | timeout | - |
| local_sgd_chain[K=1,q=1,d=2] | 3·fwd (18) | 4·fwd (24) | 36 | bands | 2 | - | 36 | True | 20 | 1.800 |
| local_sgd_chain[K=1,q=1,d=2] | 3·fwd (18) | inf (-) | 36 | bands | 2 | - | 36 | True | 20 | 1.800 |
| local_sgd_chain[K=1,q=1,d=2] | inf (-) | 1·fwd (6) | infeasible at op-slice granularity: op 2 (AccMatmulTT_v1{M=2 | - | - | - | - | - | 48 | - |
| local_sgd_chain[K=1,q=1,d=2] | inf (-) | 4·fwd (24) | 20 | single | 1 | - | 20 | True | 20 | 1.000 |
| local_sgd_chain[K=1,q=1,d=2] | inf (-) | inf (-) | 20 | single | 1 | - | 20 | True | 20 | 1.000 |
| local_sgd_chain[K=2,q=1,d=2] | 1·fwd (6) | 1·fwd (6) | infeasible at op-slice granularity: op 2 (AccMatmulTT_v1{M=2 | - | - | - | - | - | timeout | - |
| local_sgd_chain[K=2,q=1,d=2] | 1·fwd (6) | 4·fwd (24) | infeasible at op-slice granularity: op 2 (AccMatmulTT_v1{M=2 | - | - | - | - | - | timeout | - |
| local_sgd_chain[K=2,q=1,d=2] | 1·fwd (6) | inf (-) | infeasible at op-slice granularity: op 2 (AccMatmulTT_v1{M=2 | - | - | - | - | - | timeout | - |
| local_sgd_chain[K=2,q=1,d=2] | 1.5·fwd (9) | 1·fwd (6) | infeasible at op-slice granularity: op 2 (AccMatmulTT_v1{M=2 | - | - | - | - | - | timeout | - |
| local_sgd_chain[K=2,q=1,d=2] | 1.5·fwd (9) | 4·fwd (24) | 88 | bands | 6 | - | 88 | True | 40 | 2.200 |
| local_sgd_chain[K=2,q=1,d=2] | 1.5·fwd (9) | inf (-) | 88 | bands | 6 | - | 88 | True | 40 | 2.200 |
| local_sgd_chain[K=2,q=1,d=2] | 3·fwd (18) | 1·fwd (6) | infeasible at op-slice granularity: op 2 (AccMatmulTT_v1{M=2 | - | - | - | - | - | timeout | - |
| local_sgd_chain[K=2,q=1,d=2] | 3·fwd (18) | 4·fwd (24) | 64 | bands | 3 | - | 64 | True | 40 | 1.600 |
| local_sgd_chain[K=2,q=1,d=2] | 3·fwd (18) | inf (-) | 64 | bands | 3 | - | 64 | True | 40 | 1.600 |
| local_sgd_chain[K=2,q=1,d=2] | inf (-) | 1·fwd (6) | infeasible at op-slice granularity: op 2 (AccMatmulTT_v1{M=2 | - | - | - | - | - | timeout | - |
| local_sgd_chain[K=2,q=1,d=2] | inf (-) | 4·fwd (24) | 40 | bands | 2 | - | 40 | True | 40 | 1.000 |
| local_sgd_chain[K=2,q=1,d=2] | inf (-) | inf (-) | 28 | single | 1 | - | 28 | True | 28 | 1.000 |
| local_sgd_chain[K=3,q=1,d=2] | 1·fwd (6) | 1·fwd (6) | infeasible at op-slice granularity: op 2 (AccMatmulTT_v1{M=2 | - | - | - | - | - | timeout | - |
| local_sgd_chain[K=3,q=1,d=2] | 1·fwd (6) | 4·fwd (24) | infeasible at op-slice granularity: op 2 (AccMatmulTT_v1{M=2 | - | - | - | - | - | timeout | - |
| local_sgd_chain[K=3,q=1,d=2] | 1·fwd (6) | inf (-) | infeasible at op-slice granularity: op 2 (AccMatmulTT_v1{M=2 | - | - | - | - | - | timeout | - |
| local_sgd_chain[K=3,q=1,d=2] | 1.5·fwd (9) | 1·fwd (6) | infeasible at op-slice granularity: op 2 (AccMatmulTT_v1{M=2 | - | - | - | - | - | timeout | - |
| local_sgd_chain[K=3,q=1,d=2] | 1.5·fwd (9) | 4·fwd (24) | 132 | bands | 9 | - | 132 | True | 60 | 2.200 |
| local_sgd_chain[K=3,q=1,d=2] | 1.5·fwd (9) | inf (-) | 132 | bands | 9 | - | 132 | True | 60 | 2.200 |
| local_sgd_chain[K=3,q=1,d=2] | 3·fwd (18) | 1·fwd (6) | infeasible at op-slice granularity: op 2 (AccMatmulTT_v1{M=2 | - | - | - | - | - | timeout | - |
| local_sgd_chain[K=3,q=1,d=2] | 3·fwd (18) | 4·fwd (24) | 100 | bands | 5 | - | 100 | True | 60 | 1.667 |
| local_sgd_chain[K=3,q=1,d=2] | 3·fwd (18) | inf (-) | 100 | bands | 5 | - | 100 | True | 56 | 1.786 |
| local_sgd_chain[K=3,q=1,d=2] | inf (-) | 1·fwd (6) | infeasible at op-slice granularity: op 2 (AccMatmulTT_v1{M=2 | - | - | - | - | - | timeout | - |
| local_sgd_chain[K=3,q=1,d=2] | inf (-) | 4·fwd (24) | 60 | bands | 3 | - | 60 | True | 60 | 1.000 |
| local_sgd_chain[K=3,q=1,d=2] | inf (-) | inf (-) | 36 | single | 1 | - | 36 | True | 36 | 1.000 |

## micro local-sgd chain (d=2, q=2)

| circuit | F | G | U (B) | family | units | (m, s, kf) | gate cost | gate legal | I* | U/I* |
|---|---|---|---|---|---|---|---|---|---|---|
| local_sgd_chain[K=1,q=2,d=2] | 1·fwd (12) | 1·fwd (12) | 68 | bands | 4 | - | 68 | True | timeout | - |
| local_sgd_chain[K=1,q=2,d=2] | 1·fwd (12) | 4·fwd (48) | 68 | bands | 4 | - | 68 | True | 42 | 1.619 |
| local_sgd_chain[K=1,q=2,d=2] | 1·fwd (12) | inf (-) | 68 | bands | 4 | - | 68 | True | 42 | 1.619 |
| local_sgd_chain[K=1,q=2,d=2] | 1.5·fwd (18) | 1·fwd (12) | 68 | bands | 4 | - | 68 | True | timeout | - |
| local_sgd_chain[K=1,q=2,d=2] | 1.5·fwd (18) | 4·fwd (48) | 60 | bands | 3 | - | 60 | True | 28 | 2.143 |
| local_sgd_chain[K=1,q=2,d=2] | 1.5·fwd (18) | inf (-) | 60 | bands | 3 | - | 60 | True | 28 | 2.143 |
| local_sgd_chain[K=1,q=2,d=2] | 3·fwd (36) | 1·fwd (12) | 68 | bands | 4 | - | 68 | True | timeout | - |
| local_sgd_chain[K=1,q=2,d=2] | 3·fwd (36) | 4·fwd (48) | 28 | single | 1 | - | 28 | True | 28 | 1.000 |
| local_sgd_chain[K=1,q=2,d=2] | 3·fwd (36) | inf (-) | 28 | single | 1 | - | 28 | True | 28 | 1.000 |
| local_sgd_chain[K=1,q=2,d=2] | inf (-) | 1·fwd (12) | 68 | bands | 4 | - | 68 | True | timeout | - |
| local_sgd_chain[K=1,q=2,d=2] | inf (-) | 4·fwd (48) | 28 | single | 1 | - | 28 | True | 28 | 1.000 |
| local_sgd_chain[K=1,q=2,d=2] | inf (-) | inf (-) | 28 | single | 1 | - | 28 | True | 28 | 1.000 |
| local_sgd_chain[K=2,q=2,d=2] | 1·fwd (12) | 1·fwd (12) | 136 | bands | 8 | - | 136 | True | timeout | - |
| local_sgd_chain[K=2,q=2,d=2] | 1·fwd (12) | 4·fwd (48) | 136 | bands | 8 | - | 136 | True | timeout | - |
| local_sgd_chain[K=2,q=2,d=2] | 1·fwd (12) | inf (-) | 136 | bands | 8 | - | 136 | True | timeout | - |
| local_sgd_chain[K=2,q=2,d=2] | 1.5·fwd (18) | 1·fwd (12) | 136 | bands | 8 | - | 136 | True | timeout | - |
| local_sgd_chain[K=2,q=2,d=2] | 1.5·fwd (18) | 4·fwd (48) | 120 | bands | 6 | - | 120 | True | 56 | 2.143 |
| local_sgd_chain[K=2,q=2,d=2] | 1.5·fwd (18) | inf (-) | 120 | bands | 6 | - | 120 | True | 56 | 2.143 |
| local_sgd_chain[K=2,q=2,d=2] | 3·fwd (36) | 1·fwd (12) | 136 | bands | 8 | - | 136 | True | timeout | - |
| local_sgd_chain[K=2,q=2,d=2] | 3·fwd (36) | 4·fwd (48) | 56 | bands | 2 | - | 56 | True | 56 | 1.000 |
| local_sgd_chain[K=2,q=2,d=2] | 3·fwd (36) | inf (-) | 56 | bands | 2 | - | 56 | True | 44 | 1.273 |
| local_sgd_chain[K=2,q=2,d=2] | inf (-) | 1·fwd (12) | 136 | bands | 8 | - | 136 | True | timeout | - |
| local_sgd_chain[K=2,q=2,d=2] | inf (-) | 4·fwd (48) | 56 | bands | 2 | - | 56 | True | 56 | 1.000 |
| local_sgd_chain[K=2,q=2,d=2] | inf (-) | inf (-) | 44 | single | 1 | - | 44 | True | 44 | 1.000 |

## micro fan-out (shared produced weight)

| circuit | F | G | U (B) | family | units | (m, s, kf) | gate cost | gate legal | I* | U/I* |
|---|---|---|---|---|---|---|---|---|---|---|
| fanout_shared_weight[n=4,q=1,d=2] | 1·fwd (6) | 1·fwd (6) | 72 | fine | 6 | - | 72 | True | timeout | - |
| fanout_shared_weight[n=4,q=1,d=2] | 1·fwd (6) | 4·fwd (24) | 72 | fine | 6 | - | 72 | True | 44 | 1.636 |
| fanout_shared_weight[n=4,q=1,d=2] | 1·fwd (6) | inf (-) | 72 | fine | 6 | - | 72 | True | 44 | 1.636 |
| fanout_shared_weight[n=4,q=1,d=2] | 1.5·fwd (9) | 1·fwd (6) | 72 | fine | 6 | - | 72 | True | timeout | - |
| fanout_shared_weight[n=4,q=1,d=2] | 1.5·fwd (9) | 4·fwd (24) | 44 | bands | 2 | - | 44 | True | 44 | 1.000 |
| fanout_shared_weight[n=4,q=1,d=2] | 1.5·fwd (9) | inf (-) | 44 | bands | 2 | - | 44 | True | 36 | 1.222 |
| fanout_shared_weight[n=4,q=1,d=2] | 3·fwd (18) | 1·fwd (6) | 72 | fine | 6 | - | 72 | True | timeout | - |
| fanout_shared_weight[n=4,q=1,d=2] | 3·fwd (18) | 4·fwd (24) | 44 | bands | 2 | - | 44 | True | 44 | 1.000 |
| fanout_shared_weight[n=4,q=1,d=2] | 3·fwd (18) | inf (-) | 36 | single | 1 | - | 36 | True | 36 | 1.000 |
| fanout_shared_weight[n=4,q=1,d=2] | inf (-) | 1·fwd (6) | 72 | fine | 6 | - | 72 | True | timeout | - |
| fanout_shared_weight[n=4,q=1,d=2] | inf (-) | 4·fwd (24) | 44 | bands | 2 | - | 44 | True | 44 | 1.000 |
| fanout_shared_weight[n=4,q=1,d=2] | inf (-) | inf (-) | 36 | single | 1 | - | 36 | True | 36 | 1.000 |

## micro deep chain (fixed weights)

| circuit | F | G | U (B) | family | units | (m, s, kf) | gate cost | gate legal | I* | U/I* |
|---|---|---|---|---|---|---|---|---|---|---|
| deep_chain[depth=4,q=1,d=2,W=fixed] | 1·fwd (6) | 1·fwd (6) | 16 | bands | 4 | - | 16 | True | 16 | 1.000 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 1·fwd (6) | 4·fwd (24) | 16 | bands | 4 | - | 16 | True | 16 | 1.000 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 1·fwd (6) | inf (-) | 16 | bands | 4 | - | 16 | True | 16 | 1.000 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 1.5·fwd (9) | 1·fwd (6) | 16 | bands | 4 | - | 16 | True | 16 | 1.000 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 1.5·fwd (9) | 4·fwd (24) | 16 | bands | 4 | - | 16 | True | 8 | 2.000 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 1.5·fwd (9) | inf (-) | 16 | bands | 4 | - | 16 | True | 8 | 2.000 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 3·fwd (18) | 1·fwd (6) | 16 | bands | 4 | - | 16 | True | 16 | 1.000 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 3·fwd (18) | 4·fwd (24) | 8 | bands | 2 | - | 8 | True | 8 | 1.000 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 3·fwd (18) | inf (-) | 8 | bands | 2 | - | 8 | True | 8 | 1.000 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | inf (-) | 1·fwd (6) | 16 | bands | 4 | - | 16 | True | 16 | 1.000 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | inf (-) | 4·fwd (24) | 4 | single | 1 | - | 4 | True | 4 | 1.000 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | inf (-) | inf (-) | 4 | single | 1 | - | 4 | True | 4 | 1.000 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 1·fwd (6) | 1·fwd (6) | 24 | bands | 6 | - | 24 | True | 24 | 1.000 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 1·fwd (6) | 4·fwd (24) | 24 | bands | 6 | - | 24 | True | 24 | 1.000 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 1·fwd (6) | inf (-) | 24 | bands | 6 | - | 24 | True | 24 | 1.000 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 1.5·fwd (9) | 1·fwd (6) | 24 | bands | 6 | - | 24 | True | 24 | 1.000 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 1.5·fwd (9) | 4·fwd (24) | 24 | bands | 6 | - | 24 | True | 12 | 2.000 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 1.5·fwd (9) | inf (-) | 24 | bands | 6 | - | 24 | True | 12 | 2.000 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 3·fwd (18) | 1·fwd (6) | 24 | bands | 6 | - | 24 | True | 24 | 1.000 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 3·fwd (18) | 4·fwd (24) | 8 | bands | 2 | - | 8 | True | 8 | 1.000 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 3·fwd (18) | inf (-) | 8 | bands | 2 | - | 8 | True | 8 | 1.000 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | inf (-) | 1·fwd (6) | 24 | bands | 6 | - | 24 | True | 24 | 1.000 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | inf (-) | 4·fwd (24) | 8 | bands | 2 | - | 8 | True | 8 | 1.000 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | inf (-) | inf (-) | 4 | single | 1 | - | 4 | True | 4 | 1.000 |

## registry inference-dense @ tiny

| circuit | F | G | U (B) | family | units | (m, s, kf) | gate cost | gate legal | I* | U/I* |
|---|---|---|---|---|---|---|---|---|---|---|
| inference-dense | 1·fwd (592) | 1·fwd (592) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | 1·fwd (592) | 4·fwd (2368) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | 1·fwd (592) | inf (-) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | 1.5·fwd (888) | 1·fwd (592) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | 1.5·fwd (888) | 4·fwd (2368) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | 1.5·fwd (888) | inf (-) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | 3·fwd (1776) | 1·fwd (592) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | 3·fwd (1776) | 4·fwd (2368) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | 3·fwd (1776) | inf (-) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | inf (-) | 1·fwd (592) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | inf (-) | 4·fwd (2368) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | inf (-) | inf (-) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | 1·fwd (936) | 1·fwd (936) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | 1·fwd (936) | 4·fwd (3744) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | 1·fwd (936) | inf (-) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | 1.5·fwd (1404) | 1·fwd (936) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | 1.5·fwd (1404) | 4·fwd (3744) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | 1.5·fwd (1404) | inf (-) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | 3·fwd (2808) | 1·fwd (936) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | 3·fwd (2808) | 4·fwd (3744) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | 3·fwd (2808) | inf (-) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | inf (-) | 1·fwd (936) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | inf (-) | 4·fwd (3744) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | inf (-) | inf (-) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | 1·fwd (1312) | 1·fwd (1312) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | 1·fwd (1312) | 4·fwd (5248) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | 1·fwd (1312) | inf (-) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | 1.5·fwd (1968) | 1·fwd (1312) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | 1.5·fwd (1968) | 4·fwd (5248) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | 1.5·fwd (1968) | inf (-) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | 3·fwd (3936) | 1·fwd (1312) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | 3·fwd (3936) | 4·fwd (5248) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | 3·fwd (3936) | inf (-) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | inf (-) | 1·fwd (1312) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | inf (-) | 4·fwd (5248) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | inf (-) | inf (-) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |

## registry local-sgd @ tiny (q=2)

| circuit | F | G | U (B) | family | units | (m, s, kf) | gate cost | gate legal | I* | U/I* |
|---|---|---|---|---|---|---|---|---|---|---|
| local-sgd | 1·fwd (592) | 1·fwd (592) | 3196 | fine | 82 | - | 3196 | True | beyond exact scale (2888 gates) | - |
| local-sgd | 1·fwd (592) | 4·fwd (2368) | 3196 | fine | 82 | - | 3196 | True | beyond exact scale (2888 gates) | - |
| local-sgd | 1·fwd (592) | inf (-) | 3196 | fine | 82 | - | 3196 | True | beyond exact scale (2888 gates) | - |
| local-sgd | 1.5·fwd (888) | 1·fwd (592) | 3196 | fine | 82 | - | 3196 | True | beyond exact scale (2888 gates) | - |
| local-sgd | 1.5·fwd (888) | 4·fwd (2368) | 1288 | convex-block | 6 | (1, 1, 1) | 1288 | True | beyond exact scale (2888 gates) | - |
| local-sgd | 1.5·fwd (888) | inf (-) | 1288 | convex-block | 6 | (1, 1, 1) | 1288 | True | beyond exact scale (2888 gates) | - |
| local-sgd | 3·fwd (1776) | 1·fwd (592) | 3196 | fine | 82 | - | 3196 | True | beyond exact scale (2888 gates) | - |
| local-sgd | 3·fwd (1776) | 4·fwd (2368) | 996 | convex-block | 3 | (1, 1, 1) | 996 | True | beyond exact scale (2888 gates) | - |
| local-sgd | 3·fwd (1776) | inf (-) | 412 | convex-block | 2 | (1, 1, 1) | 412 | True | beyond exact scale (2888 gates) | - |
| local-sgd | inf (-) | 1·fwd (592) | 3196 | fine | 82 | - | 3196 | True | beyond exact scale (2888 gates) | - |
| local-sgd | inf (-) | 4·fwd (2368) | 996 | convex-block | 3 | (1, 1, 1) | 996 | True | beyond exact scale (2888 gates) | - |
| local-sgd | inf (-) | inf (-) | 412 | convex-block | 2 | (1, 1, 1) | 412 | True | beyond exact scale (2888 gates) | - |
| local-sgd | 1·fwd (592) | 1·fwd (592) | 6392 | fine | 164 | - | 6392 | True | beyond exact scale (5776 gates) | - |
| local-sgd | 1·fwd (592) | 4·fwd (2368) | 6392 | fine | 164 | - | 6392 | True | beyond exact scale (5776 gates) | - |
| local-sgd | 1·fwd (592) | inf (-) | 6392 | fine | 164 | - | 6392 | True | beyond exact scale (5776 gates) | - |
| local-sgd | 1.5·fwd (888) | 1·fwd (592) | 6392 | fine | 164 | - | 6392 | True | beyond exact scale (5776 gates) | - |
| local-sgd | 1.5·fwd (888) | 4·fwd (2368) | 2512 | convex-block | 11 | (1, 1, 1) | 2512 | True | beyond exact scale (5776 gates) | - |
| local-sgd | 1.5·fwd (888) | inf (-) | 2512 | convex-block | 11 | (1, 1, 1) | 2512 | True | beyond exact scale (5776 gates) | - |
| local-sgd | 3·fwd (1776) | 1·fwd (592) | 6392 | fine | 164 | - | 6392 | True | beyond exact scale (5776 gates) | - |
| local-sgd | 3·fwd (1776) | 4·fwd (2368) | 1928 | convex-block | 5 | (1, 1, 1) | 1928 | True | beyond exact scale (5776 gates) | - |
| local-sgd | 3·fwd (1776) | inf (-) | 760 | convex-block | 3 | (1, 1, 1) | 760 | True | beyond exact scale (5776 gates) | - |
| local-sgd | inf (-) | 1·fwd (592) | 6392 | fine | 164 | - | 6392 | True | beyond exact scale (5776 gates) | - |
| local-sgd | inf (-) | 4·fwd (2368) | 1928 | convex-block | 5 | (1, 1, 1) | 1928 | True | beyond exact scale (5776 gates) | - |
| local-sgd | inf (-) | inf (-) | 412 | convex-block | 1 | (1, 1, 2) | 412 | True | beyond exact scale (5776 gates) | - |
| local-sgd | 1·fwd (592) | 1·fwd (592) | 9588 | fine | 246 | - | 9588 | True | beyond exact scale (8664 gates) | - |
| local-sgd | 1·fwd (592) | 4·fwd (2368) | 9588 | fine | 246 | - | 9588 | True | beyond exact scale (8664 gates) | - |
| local-sgd | 1·fwd (592) | inf (-) | 9588 | fine | 246 | - | 9588 | True | beyond exact scale (8664 gates) | - |
| local-sgd | 1.5·fwd (888) | 1·fwd (592) | 9588 | fine | 246 | - | 9588 | True | beyond exact scale (8664 gates) | - |
| local-sgd | 1.5·fwd (888) | 4·fwd (2368) | 3736 | convex-block | 16 | (1, 1, 1) | 3736 | True | beyond exact scale (8664 gates) | - |
| local-sgd | 1.5·fwd (888) | inf (-) | 3736 | convex-block | 16 | (1, 1, 1) | 3736 | True | beyond exact scale (8664 gates) | - |
| local-sgd | 3·fwd (1776) | 1·fwd (592) | 9588 | fine | 246 | - | 9588 | True | beyond exact scale (8664 gates) | - |
| local-sgd | 3·fwd (1776) | 4·fwd (2368) | 2860 | convex-block | 7 | (1, 1, 1) | 2860 | True | beyond exact scale (8664 gates) | - |
| local-sgd | 3·fwd (1776) | inf (-) | 1108 | convex-block | 4 | (1, 1, 1) | 1108 | True | beyond exact scale (8664 gates) | - |
| local-sgd | inf (-) | 1·fwd (592) | 9588 | fine | 246 | - | 9588 | True | beyond exact scale (8664 gates) | - |
| local-sgd | inf (-) | 4·fwd (2368) | 2860 | convex-block | 7 | (1, 1, 1) | 2860 | True | beyond exact scale (8664 gates) | - |
| local-sgd | inf (-) | inf (-) | 428 | convex-block | 1 | (1, 1, 3) | 428 | True | beyond exact scale (8664 gates) | - |

## registry inference-dense @ tiny2

| circuit | F | G | U (B) | family | units | (m, s, kf) | gate cost | gate legal | I* | U/I* |
|---|---|---|---|---|---|---|---|---|---|---|
| inference-dense | 1·fwd (1290) | 1·fwd (1290) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | 1·fwd (1290) | 4·fwd (5160) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | 1·fwd (1290) | inf (-) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | 1.5·fwd (1935) | 1·fwd (1290) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | 1.5·fwd (1935) | 4·fwd (5160) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | 1.5·fwd (1935) | inf (-) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | 3·fwd (3870) | 1·fwd (1290) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | 3·fwd (3870) | 4·fwd (5160) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | 3·fwd (3870) | inf (-) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | inf (-) | 1·fwd (1290) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | inf (-) | 4·fwd (5160) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | inf (-) | inf (-) | 8 | single | 1 | - | 8 | True | 8 | 1.000 |
| inference-dense | 1·fwd (2079) | 1·fwd (2079) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | 1·fwd (2079) | 4·fwd (8316) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | 1·fwd (2079) | inf (-) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | 1.5·fwd (3119) | 1·fwd (2079) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | 1.5·fwd (3119) | 4·fwd (8316) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | 1.5·fwd (3119) | inf (-) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | 3·fwd (6237) | 1·fwd (2079) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | 3·fwd (6237) | 4·fwd (8316) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | 3·fwd (6237) | inf (-) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | inf (-) | 1·fwd (2079) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | inf (-) | 4·fwd (8316) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | inf (-) | inf (-) | 12 | single | 1 | - | 12 | True | 12 | 1.000 |
| inference-dense | 1·fwd (2964) | 1·fwd (2964) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | 1·fwd (2964) | 4·fwd (11856) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | 1·fwd (2964) | inf (-) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | 1.5·fwd (4446) | 1·fwd (2964) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | 1.5·fwd (4446) | 4·fwd (11856) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | 1.5·fwd (4446) | inf (-) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | 3·fwd (8892) | 1·fwd (2964) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | 3·fwd (8892) | 4·fwd (11856) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | 3·fwd (8892) | inf (-) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | inf (-) | 1·fwd (2964) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | inf (-) | 4·fwd (11856) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |
| inference-dense | inf (-) | inf (-) | 16 | single | 1 | - | 16 | True | 16 | 1.000 |

## registry local-sgd @ tiny2 (q=2)

| circuit | F | G | U (B) | family | units | (m, s, kf) | gate cost | gate legal | I* | U/I* |
|---|---|---|---|---|---|---|---|---|---|---|
| local-sgd | 1·fwd (1290) | 1·fwd (1290) | 2700 | convex-block | 8 | (1, 1, 1) | 2700 | True | beyond exact scale (6272 gates) | - |
| local-sgd | 1·fwd (1290) | 4·fwd (5160) | 1476 | convex-block | 5 | (1, 1, 1) | 1476 | True | beyond exact scale (6272 gates) | - |
| local-sgd | 1·fwd (1290) | inf (-) | 1476 | convex-block | 5 | (1, 1, 1) | 1476 | True | beyond exact scale (6272 gates) | - |
| local-sgd | 1.5·fwd (1935) | 1·fwd (1290) | 2700 | convex-block | 8 | (1, 1, 1) | 2700 | True | beyond exact scale (6272 gates) | - |
| local-sgd | 1.5·fwd (1935) | 4·fwd (5160) | 1476 | convex-block | 5 | (1, 1, 1) | 1476 | True | beyond exact scale (6272 gates) | - |
| local-sgd | 1.5·fwd (1935) | inf (-) | 1476 | convex-block | 5 | (1, 1, 1) | 1476 | True | beyond exact scale (6272 gates) | - |
| local-sgd | 3·fwd (3870) | 1·fwd (1290) | 2700 | convex-block | 8 | (1, 1, 1) | 2700 | True | beyond exact scale (6272 gates) | - |
| local-sgd | 3·fwd (3870) | 4·fwd (5160) | 2004 | convex-block | 3 | (1, 1, 1) | 2004 | True | beyond exact scale (6272 gates) | - |
| local-sgd | 3·fwd (3870) | inf (-) | 780 | convex-block | 2 | (1, 1, 1) | 780 | True | beyond exact scale (6272 gates) | - |
| local-sgd | inf (-) | 1·fwd (1290) | 2700 | convex-block | 8 | (1, 1, 1) | 2700 | True | beyond exact scale (6272 gates) | - |
| local-sgd | inf (-) | 4·fwd (5160) | 2004 | convex-block | 3 | (1, 1, 1) | 2004 | True | beyond exact scale (6272 gates) | - |
| local-sgd | inf (-) | inf (-) | 780 | convex-block | 2 | (1, 1, 1) | 780 | True | beyond exact scale (6272 gates) | - |
| local-sgd | 1·fwd (1290) | 1·fwd (1290) | 5336 | convex-block | 15 | (1, 1, 1) | 5336 | True | beyond exact scale (12544 gates) | - |
| local-sgd | 1·fwd (1290) | 4·fwd (5160) | 2888 | convex-block | 9 | (1, 1, 1) | 2888 | True | beyond exact scale (12544 gates) | - |
| local-sgd | 1·fwd (1290) | inf (-) | 2888 | convex-block | 9 | (1, 1, 1) | 2888 | True | beyond exact scale (12544 gates) | - |
| local-sgd | 1.5·fwd (1935) | 1·fwd (1290) | 5336 | convex-block | 15 | (1, 1, 1) | 5336 | True | beyond exact scale (12544 gates) | - |
| local-sgd | 1.5·fwd (1935) | 4·fwd (5160) | 2888 | convex-block | 9 | (1, 1, 1) | 2888 | True | beyond exact scale (12544 gates) | - |
| local-sgd | 1.5·fwd (1935) | inf (-) | 2888 | convex-block | 9 | (1, 1, 1) | 2888 | True | beyond exact scale (12544 gates) | - |
| local-sgd | 3·fwd (3870) | 1·fwd (1290) | 5336 | convex-block | 15 | (1, 1, 1) | 5336 | True | beyond exact scale (12544 gates) | - |
| local-sgd | 3·fwd (3870) | 4·fwd (5160) | 3944 | convex-block | 5 | (1, 1, 1) | 3944 | True | beyond exact scale (12544 gates) | - |
| local-sgd | 3·fwd (3870) | inf (-) | 1496 | convex-block | 3 | (1, 1, 1) | 1496 | True | beyond exact scale (12544 gates) | - |
| local-sgd | inf (-) | 1·fwd (1290) | 5336 | convex-block | 15 | (1, 1, 1) | 5336 | True | beyond exact scale (12544 gates) | - |
| local-sgd | inf (-) | 4·fwd (5160) | 3944 | convex-block | 5 | (1, 1, 1) | 3944 | True | beyond exact scale (12544 gates) | - |
| local-sgd | inf (-) | inf (-) | 780 | convex-block | 1 | (1, 1, 2) | 780 | True | beyond exact scale (12544 gates) | - |
| local-sgd | 1·fwd (1290) | 1·fwd (1290) | 7972 | convex-block | 22 | (1, 1, 1) | 7972 | True | beyond exact scale (18816 gates) | - |
| local-sgd | 1·fwd (1290) | 4·fwd (5160) | 4300 | convex-block | 13 | (1, 1, 1) | 4300 | True | beyond exact scale (18816 gates) | - |
| local-sgd | 1·fwd (1290) | inf (-) | 4300 | convex-block | 13 | (1, 1, 1) | 4300 | True | beyond exact scale (18816 gates) | - |
| local-sgd | 1.5·fwd (1935) | 1·fwd (1290) | 7972 | convex-block | 22 | (1, 1, 1) | 7972 | True | beyond exact scale (18816 gates) | - |
| local-sgd | 1.5·fwd (1935) | 4·fwd (5160) | 4300 | convex-block | 13 | (1, 1, 1) | 4300 | True | beyond exact scale (18816 gates) | - |
| local-sgd | 1.5·fwd (1935) | inf (-) | 4300 | convex-block | 13 | (1, 1, 1) | 4300 | True | beyond exact scale (18816 gates) | - |
| local-sgd | 3·fwd (3870) | 1·fwd (1290) | 7972 | convex-block | 22 | (1, 1, 1) | 7972 | True | beyond exact scale (18816 gates) | - |
| local-sgd | 3·fwd (3870) | 4·fwd (5160) | 5884 | convex-block | 7 | (1, 1, 1) | 5884 | True | beyond exact scale (18816 gates) | - |
| local-sgd | 3·fwd (3870) | inf (-) | 2212 | convex-block | 4 | (1, 1, 1) | 2212 | True | beyond exact scale (18816 gates) | - |
| local-sgd | inf (-) | 1·fwd (1290) | 7972 | convex-block | 22 | (1, 1, 1) | 7972 | True | beyond exact scale (18816 gates) | - |
| local-sgd | inf (-) | 4·fwd (5160) | 5884 | convex-block | 7 | (1, 1, 1) | 5884 | True | beyond exact scale (18816 gates) | - |
| local-sgd | inf (-) | inf (-) | 796 | convex-block | 1 | (1, 1, 3) | 796 | True | beyond exact scale (18816 gates) | - |
