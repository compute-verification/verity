# Exact tiny wedge: `I*(C; F, G)` under the corrected policy (`X = inf`)

Generated 2026-09-15 18:54:49 by `accumulation.redteam.wedge`.  Policy: X = inf; legal iff work(R) <= G and work(Up_R(g)) <= F; charged roles: token, accumulated, carried, seed.  Work measure: primitive .work summed over gates == OpGraph op work (Mac16 = Round16 = Add16 = Scale16 = Sub16 = 1, Zero32 = 0).  `fwd` = work of the inference forward of the same layer geometry; `F in {1, 1.5, 3} fwd` and `inf`, `G in {1, 4} fwd` and `inf`.  Exact solver time box 240.0 s per cell; circuits above 64 gates get the MILP only where the single-RU partition is legal (then it is optimal: every partition pays every distinct charged root leaf at least once); otherwise a local-search upper value and the trivial charged-root lower bound.  Unproved cells are shown as `[lower bound, best found]`.

## Circuits

| circuit | gates | work | work/fwd | tokens | useful MACs | charged roots (B) | by role |
|---|---|---|---|---|---|---|---|
| micro_inference[q=1,d=2] | 8 | 6 | 1.00 | 1 | 4 | 4 | {'token': 4} |
| micro_inference[q=2,d=2] | 16 | 12 | 1.00 | 2 | 8 | 8 | {'token': 8} |
| local_sgd_chain[K=1,q=1,d=1] | 9 | 7 | 3.50 | 1 | 2 | 10 | {'accumulated': 2, 'seed': 4, 'token': 4} |
| local_sgd_chain[K=2,q=1,d=1] | 18 | 14 | 7.00 | 2 | 4 | 14 | {'accumulated': 2, 'seed': 4, 'token': 8} |
| local_sgd_chain[K=3,q=1,d=1] | 27 | 21 | 10.50 | 3 | 6 | 18 | {'accumulated': 2, 'seed': 4, 'token': 12} |
| local_sgd_chain[K=1,q=2,d=1] | 14 | 11 | 2.75 | 2 | 4 | 14 | {'accumulated': 2, 'seed': 4, 'token': 8} |
| local_sgd_chain[K=2,q=2,d=1] | 28 | 22 | 5.50 | 4 | 8 | 22 | {'accumulated': 2, 'seed': 4, 'token': 16} |
| local_sgd_chain[K=3,q=2,d=1] | 42 | 33 | 8.25 | 6 | 12 | 30 | {'accumulated': 2, 'seed': 4, 'token': 24} |
| local_sgd_chain[K=1,q=1,d=2] | 30 | 24 | 4.00 | 1 | 8 | 20 | {'accumulated': 8, 'seed': 4, 'token': 8} |
| local_sgd_chain[K=2,q=1,d=2] | 60 | 48 | 8.00 | 2 | 16 | 28 | {'accumulated': 8, 'seed': 4, 'token': 16} |
| local_sgd_chain[K=3,q=1,d=2] | 90 | 72 | 12.00 | 3 | 24 | 36 | {'accumulated': 8, 'seed': 4, 'token': 24} |
| local_sgd_chain[K=1,q=2,d=2] | 44 | 36 | 3.00 | 2 | 16 | 28 | {'accumulated': 8, 'seed': 4, 'token': 16} |
| local_sgd_chain[K=2,q=2,d=2] | 88 | 72 | 6.00 | 4 | 32 | 44 | {'accumulated': 8, 'seed': 4, 'token': 32} |
| fanout_shared_weight[n=4,q=1,d=2] | 40 | 32 | 5.33 | 4 | 16 | 36 | {'accumulated': 8, 'token': 24, 'seed': 4} |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 32 | 24 | 4.00 | 1 | 16 | 4 | {'token': 4} |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 48 | 36 | 6.00 | 1 | 24 | 4 | {'token': 4} |
| inference-dense | 656 | 592 | 1.00 | 2 | 320 | 8 | {'token': 8} |
| local-sgd | 2888 | 2534 | 4.28 | 2 | 976 | 396 | {'token': 16, 'accumulated': 376, 'seed': 4} |

## Table: (circuit, F, G) -> I*

### micro inference

| circuit | F | G | I* (B) | I*/token | useful MACs | kappa = MACs/I* | RUs | solver | s |
|---|---|---|---|---|---|---|---|---|---|
| micro_inference[q=1,d=2] | 1·fwd (6) | 1·fwd (6) | 4 | 4.00 | 4 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=1,d=2] | 1·fwd (6) | 4·fwd (24) | 4 | 4.00 | 4 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=1,d=2] | 1·fwd (6) | inf (-) | 4 | 4.00 | 4 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=1,d=2] | 1.5·fwd (9) | 1·fwd (6) | 4 | 4.00 | 4 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=1,d=2] | 1.5·fwd (9) | 4·fwd (24) | 4 | 4.00 | 4 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=1,d=2] | 1.5·fwd (9) | inf (-) | 4 | 4.00 | 4 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=1,d=2] | 3·fwd (18) | 1·fwd (6) | 4 | 4.00 | 4 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=1,d=2] | 3·fwd (18) | 4·fwd (24) | 4 | 4.00 | 4 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=1,d=2] | 3·fwd (18) | inf (-) | 4 | 4.00 | 4 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=1,d=2] | inf (-) | 1·fwd (6) | 4 | 4.00 | 4 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=1,d=2] | inf (-) | 4·fwd (24) | 4 | 4.00 | 4 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=1,d=2] | inf (-) | inf (-) | 4 | 4.00 | 4 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=2,d=2] | 1·fwd (12) | 1·fwd (12) | 8 | 4.00 | 8 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=2,d=2] | 1·fwd (12) | 4·fwd (48) | 8 | 4.00 | 8 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=2,d=2] | 1·fwd (12) | inf (-) | 8 | 4.00 | 8 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=2,d=2] | 1.5·fwd (18) | 1·fwd (12) | 8 | 4.00 | 8 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=2,d=2] | 1.5·fwd (18) | 4·fwd (48) | 8 | 4.00 | 8 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=2,d=2] | 1.5·fwd (18) | inf (-) | 8 | 4.00 | 8 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=2,d=2] | 3·fwd (36) | 1·fwd (12) | 8 | 4.00 | 8 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=2,d=2] | 3·fwd (36) | 4·fwd (48) | 8 | 4.00 | 8 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=2,d=2] | 3·fwd (36) | inf (-) | 8 | 4.00 | 8 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=2,d=2] | inf (-) | 1·fwd (12) | 8 | 4.00 | 8 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=2,d=2] | inf (-) | 4·fwd (48) | 8 | 4.00 | 8 | 1.00 | 1 | one-ru | 0.0 |
| micro_inference[q=2,d=2] | inf (-) | inf (-) | 8 | 4.00 | 8 | 1.00 | 1 | one-ru | 0.0 |

### micro local-sgd chain (d=1, q=1)

| circuit | F | G | I* (B) | I*/token | useful MACs | kappa = MACs/I* | RUs | solver | s |
|---|---|---|---|---|---|---|---|---|---|
| local_sgd_chain[K=1,q=1,d=1] | 1·fwd (2) | 1·fwd (2) | 20 | 20.00 | 2 | 0.10 | 4 | brute | 0.0 |
| local_sgd_chain[K=1,q=1,d=1] | 1·fwd (2) | 4·fwd (8) | 20 | 20.00 | 2 | 0.10 | 4 | brute | 0.0 |
| local_sgd_chain[K=1,q=1,d=1] | 1·fwd (2) | inf (-) | 20 | 20.00 | 2 | 0.10 | 4 | brute | 0.0 |
| local_sgd_chain[K=1,q=1,d=1] | 1.5·fwd (3) | 1·fwd (2) | 20 | 20.00 | 2 | 0.10 | 4 | brute | 0.0 |
| local_sgd_chain[K=1,q=1,d=1] | 1.5·fwd (3) | 4·fwd (8) | 18 | 18.00 | 2 | 0.11 | 3 | brute | 0.0 |
| local_sgd_chain[K=1,q=1,d=1] | 1.5·fwd (3) | inf (-) | 18 | 18.00 | 2 | 0.11 | 3 | brute | 0.0 |
| local_sgd_chain[K=1,q=1,d=1] | 3·fwd (6) | 1·fwd (2) | 20 | 20.00 | 2 | 0.10 | 4 | brute | 0.0 |
| local_sgd_chain[K=1,q=1,d=1] | 3·fwd (6) | 4·fwd (8) | 14 | 14.00 | 2 | 0.14 | 2 | brute | 0.0 |
| local_sgd_chain[K=1,q=1,d=1] | 3·fwd (6) | inf (-) | 14 | 14.00 | 2 | 0.14 | 2 | brute | 0.0 |
| local_sgd_chain[K=1,q=1,d=1] | inf (-) | 1·fwd (2) | 20 | 20.00 | 2 | 0.10 | 4 | brute | 0.0 |
| local_sgd_chain[K=1,q=1,d=1] | inf (-) | 4·fwd (8) | 10 | 10.00 | 2 | 0.20 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=1,q=1,d=1] | inf (-) | inf (-) | 10 | 10.00 | 2 | 0.20 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=2,q=1,d=1] | 1·fwd (2) | 1·fwd (2) | 40 | 20.00 | 4 | 0.10 | 8 | brute | 0.0 |
| local_sgd_chain[K=2,q=1,d=1] | 1·fwd (2) | 4·fwd (8) | 40 | 20.00 | 4 | 0.10 | 8 | brute | 0.0 |
| local_sgd_chain[K=2,q=1,d=1] | 1·fwd (2) | inf (-) | 40 | 20.00 | 4 | 0.10 | 8 | brute | 0.0 |
| local_sgd_chain[K=2,q=1,d=1] | 1.5·fwd (3) | 1·fwd (2) | 40 | 20.00 | 4 | 0.10 | 8 | brute | 0.0 |
| local_sgd_chain[K=2,q=1,d=1] | 1.5·fwd (3) | 4·fwd (8) | 34 | 17.00 | 4 | 0.12 | 5 | brute | 0.0 |
| local_sgd_chain[K=2,q=1,d=1] | 1.5·fwd (3) | inf (-) | 34 | 17.00 | 4 | 0.12 | 5 | brute | 0.0 |
| local_sgd_chain[K=2,q=1,d=1] | 3·fwd (6) | 1·fwd (2) | 40 | 20.00 | 4 | 0.10 | 8 | brute | 0.0 |
| local_sgd_chain[K=2,q=1,d=1] | 3·fwd (6) | 4·fwd (8) | 26 | 13.00 | 4 | 0.15 | 3 | brute | 0.0 |
| local_sgd_chain[K=2,q=1,d=1] | 3·fwd (6) | inf (-) | 26 | 13.00 | 4 | 0.15 | 3 | brute | 0.0 |
| local_sgd_chain[K=2,q=1,d=1] | inf (-) | 1·fwd (2) | 40 | 20.00 | 4 | 0.10 | 8 | brute | 0.0 |
| local_sgd_chain[K=2,q=1,d=1] | inf (-) | 4·fwd (8) | 20 | 10.00 | 4 | 0.20 | 2 | brute | 0.0 |
| local_sgd_chain[K=2,q=1,d=1] | inf (-) | inf (-) | 14 | 7.00 | 4 | 0.29 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=3,q=1,d=1] | 1·fwd (2) | 1·fwd (2) | 60 | 20.00 | 6 | 0.10 | 12 | brute | 0.1 |
| local_sgd_chain[K=3,q=1,d=1] | 1·fwd (2) | 4·fwd (8) | 60 | 20.00 | 6 | 0.10 | 12 | brute | 0.0 |
| local_sgd_chain[K=3,q=1,d=1] | 1·fwd (2) | inf (-) | 60 | 20.00 | 6 | 0.10 | 12 | brute | 0.0 |
| local_sgd_chain[K=3,q=1,d=1] | 1.5·fwd (3) | 1·fwd (2) | 60 | 20.00 | 6 | 0.10 | 12 | brute | 0.0 |
| local_sgd_chain[K=3,q=1,d=1] | 1.5·fwd (3) | 4·fwd (8) | 52 | 17.33 | 6 | 0.12 | 8 | brute | 0.2 |
| local_sgd_chain[K=3,q=1,d=1] | 1.5·fwd (3) | inf (-) | 52 | 17.33 | 6 | 0.12 | 8 | brute | 0.1 |
| local_sgd_chain[K=3,q=1,d=1] | 3·fwd (6) | 1·fwd (2) | 60 | 20.00 | 6 | 0.10 | 12 | brute | 0.0 |
| local_sgd_chain[K=3,q=1,d=1] | 3·fwd (6) | 4·fwd (8) | 40 | 13.33 | 6 | 0.15 | 4 | brute | 0.1 |
| local_sgd_chain[K=3,q=1,d=1] | 3·fwd (6) | inf (-) | 40 | 13.33 | 6 | 0.15 | 4 | brute | 0.1 |
| local_sgd_chain[K=3,q=1,d=1] | inf (-) | 1·fwd (2) | 60 | 20.00 | 6 | 0.10 | 12 | brute | 0.1 |
| local_sgd_chain[K=3,q=1,d=1] | inf (-) | 4·fwd (8) | 30 | 10.00 | 6 | 0.20 | 3 | brute | 0.0 |
| local_sgd_chain[K=3,q=1,d=1] | inf (-) | inf (-) | 18 | 6.00 | 6 | 0.33 | 1 | one-ru | 0.0 |

### micro local-sgd chain (d=1, q=2)

| circuit | F | G | I* (B) | I*/token | useful MACs | kappa = MACs/I* | RUs | solver | s |
|---|---|---|---|---|---|---|---|---|---|
| local_sgd_chain[K=1,q=2,d=1] | 1·fwd (4) | 1·fwd (4) | 26 | 13.00 | 4 | 0.15 | 3 | brute | 0.0 |
| local_sgd_chain[K=1,q=2,d=1] | 1·fwd (4) | 4·fwd (16) | 24 | 12.00 | 4 | 0.17 | 2 | brute | 0.0 |
| local_sgd_chain[K=1,q=2,d=1] | 1·fwd (4) | inf (-) | 24 | 12.00 | 4 | 0.17 | 2 | brute | 0.0 |
| local_sgd_chain[K=1,q=2,d=1] | 1.5·fwd (6) | 1·fwd (4) | 26 | 13.00 | 4 | 0.15 | 3 | brute | 0.0 |
| local_sgd_chain[K=1,q=2,d=1] | 1.5·fwd (6) | 4·fwd (16) | 24 | 12.00 | 4 | 0.17 | 2 | brute | 0.0 |
| local_sgd_chain[K=1,q=2,d=1] | 1.5·fwd (6) | inf (-) | 24 | 12.00 | 4 | 0.17 | 2 | brute | 0.0 |
| local_sgd_chain[K=1,q=2,d=1] | 3·fwd (12) | 1·fwd (4) | 26 | 13.00 | 4 | 0.15 | 3 | brute | 0.0 |
| local_sgd_chain[K=1,q=2,d=1] | 3·fwd (12) | 4·fwd (16) | 14 | 7.00 | 4 | 0.29 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=1,q=2,d=1] | 3·fwd (12) | inf (-) | 14 | 7.00 | 4 | 0.29 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=1,q=2,d=1] | inf (-) | 1·fwd (4) | 26 | 13.00 | 4 | 0.15 | 3 | brute | 0.0 |
| local_sgd_chain[K=1,q=2,d=1] | inf (-) | 4·fwd (16) | 14 | 7.00 | 4 | 0.29 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=1,q=2,d=1] | inf (-) | inf (-) | 14 | 7.00 | 4 | 0.29 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=2,q=2,d=1] | 1·fwd (4) | 1·fwd (4) | 52 | 13.00 | 8 | 0.15 | 6 | brute | 2.5 |
| local_sgd_chain[K=2,q=2,d=1] | 1·fwd (4) | 4·fwd (16) | 48 | 12.00 | 8 | 0.17 | 4 | brute | 1.0 |
| local_sgd_chain[K=2,q=2,d=1] | 1·fwd (4) | inf (-) | 48 | 12.00 | 8 | 0.17 | 4 | brute | 0.9 |
| local_sgd_chain[K=2,q=2,d=1] | 1.5·fwd (6) | 1·fwd (4) | 52 | 13.00 | 8 | 0.15 | 6 | brute | 2.7 |
| local_sgd_chain[K=2,q=2,d=1] | 1.5·fwd (6) | 4·fwd (16) | 46 | 11.50 | 8 | 0.17 | 4 | brute | 1.2 |
| local_sgd_chain[K=2,q=2,d=1] | 1.5·fwd (6) | inf (-) | 46 | 11.50 | 8 | 0.17 | 4 | brute | 1.2 |
| local_sgd_chain[K=2,q=2,d=1] | 3·fwd (12) | 1·fwd (4) | 52 | 13.00 | 8 | 0.15 | 6 | brute | 2.7 |
| local_sgd_chain[K=2,q=2,d=1] | 3·fwd (12) | 4·fwd (16) | 28 | 7.00 | 8 | 0.29 | 2 | brute | 0.0 |
| local_sgd_chain[K=2,q=2,d=1] | 3·fwd (12) | inf (-) | 28 | 7.00 | 8 | 0.29 | 2 | brute | 0.0 |
| local_sgd_chain[K=2,q=2,d=1] | inf (-) | 1·fwd (4) | 52 | 13.00 | 8 | 0.15 | 6 | brute | 2.5 |
| local_sgd_chain[K=2,q=2,d=1] | inf (-) | 4·fwd (16) | 26 | 6.50 | 8 | 0.31 | 2 | brute | 0.0 |
| local_sgd_chain[K=2,q=2,d=1] | inf (-) | inf (-) | 22 | 5.50 | 8 | 0.36 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=3,q=2,d=1] | 1·fwd (4) | 1·fwd (4) | 78 | 13.00 | 12 | 0.15 | 9 | milp | 38.5 |
| local_sgd_chain[K=3,q=2,d=1] | 1·fwd (4) | 4·fwd (16) | 72 | 12.00 | 12 | 0.17 | 6 | milp | 33.1 |
| local_sgd_chain[K=3,q=2,d=1] | 1·fwd (4) | inf (-) | 72 | 12.00 | 12 | 0.17 | 6 | milp | 31.9 |
| local_sgd_chain[K=3,q=2,d=1] | 1.5·fwd (6) | 1·fwd (4) | 78 | 13.00 | 12 | 0.15 | 9 | milp | 37.2 |
| local_sgd_chain[K=3,q=2,d=1] | 1.5·fwd (6) | 4·fwd (16) | 68 | 11.33 | 12 | 0.18 | 7 | milp | 34.1 |
| local_sgd_chain[K=3,q=2,d=1] | 1.5·fwd (6) | inf (-) | 68 | 11.33 | 12 | 0.18 | 6 | milp | 33.5 |
| local_sgd_chain[K=3,q=2,d=1] | 3·fwd (12) | 1·fwd (4) | 78 | 13.00 | 12 | 0.15 | 9 | milp | 34.7 |
| local_sgd_chain[K=3,q=2,d=1] | 3·fwd (12) | 4·fwd (16) | 42 | 7.00 | 12 | 0.29 | 3 | brute | 0.0 |
| local_sgd_chain[K=3,q=2,d=1] | 3·fwd (12) | inf (-) | 42 | 7.00 | 12 | 0.29 | 3 | brute | 0.0 |
| local_sgd_chain[K=3,q=2,d=1] | inf (-) | 1·fwd (4) | 78 | 13.00 | 12 | 0.15 | 9 | milp | 33.1 |
| local_sgd_chain[K=3,q=2,d=1] | inf (-) | 4·fwd (16) | 40 | 6.67 | 12 | 0.30 | 3 | brute | 0.0 |
| local_sgd_chain[K=3,q=2,d=1] | inf (-) | inf (-) | 30 | 5.00 | 12 | 0.40 | 1 | one-ru | 0.0 |

### micro local-sgd chain (d=2, q=1)

| circuit | F | G | I* (B) | I*/token | useful MACs | kappa = MACs/I* | RUs | solver | s |
|---|---|---|---|---|---|---|---|---|---|
| local_sgd_chain[K=1,q=1,d=2] | 1·fwd (6) | 1·fwd (6) | 48 | 48.00 | 8 | 0.17 | 4 | brute | 6.9 |
| local_sgd_chain[K=1,q=1,d=2] | 1·fwd (6) | 4·fwd (24) | 36 | 36.00 | 8 | 0.22 | 2 | brute | 0.2 |
| local_sgd_chain[K=1,q=1,d=2] | 1·fwd (6) | inf (-) | 36 | 36.00 | 8 | 0.22 | 2 | brute | 0.2 |
| local_sgd_chain[K=1,q=1,d=2] | 1.5·fwd (9) | 1·fwd (6) | 48 | 48.00 | 8 | 0.17 | 4 | brute | 7.5 |
| local_sgd_chain[K=1,q=1,d=2] | 1.5·fwd (9) | 4·fwd (24) | 20 | 20.00 | 8 | 0.40 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=1,q=1,d=2] | 1.5·fwd (9) | inf (-) | 20 | 20.00 | 8 | 0.40 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=1,q=1,d=2] | 3·fwd (18) | 1·fwd (6) | 48 | 48.00 | 8 | 0.17 | 4 | brute | 6.9 |
| local_sgd_chain[K=1,q=1,d=2] | 3·fwd (18) | 4·fwd (24) | 20 | 20.00 | 8 | 0.40 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=1,q=1,d=2] | 3·fwd (18) | inf (-) | 20 | 20.00 | 8 | 0.40 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=1,q=1,d=2] | inf (-) | 1·fwd (6) | 48 | 48.00 | 8 | 0.17 | 4 | brute | 7.0 |
| local_sgd_chain[K=1,q=1,d=2] | inf (-) | 4·fwd (24) | 20 | 20.00 | 8 | 0.40 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=1,q=1,d=2] | inf (-) | inf (-) | 20 | 20.00 | 8 | 0.40 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=2,q=1,d=2] | 1·fwd (6) | 1·fwd (6) | [84, 98] (not proved) | 49.00 | 16 | 0.16 | 9 | milp | 296.2 |
| local_sgd_chain[K=2,q=1,d=2] | 1·fwd (6) | 4·fwd (24) | 68 | 34.00 | 16 | 0.24 | 3 | milp | 35.1 |
| local_sgd_chain[K=2,q=1,d=2] | 1·fwd (6) | inf (-) | 68 | 34.00 | 16 | 0.24 | 3 | milp | 75.5 |
| local_sgd_chain[K=2,q=1,d=2] | 1.5·fwd (9) | 1·fwd (6) | [84, 98] (not proved) | 49.00 | 16 | 0.16 | 9 | milp | 284.3 |
| local_sgd_chain[K=2,q=1,d=2] | 1.5·fwd (9) | 4·fwd (24) | 40 | 20.00 | 16 | 0.40 | 2 | brute | 0.0 |
| local_sgd_chain[K=2,q=1,d=2] | 1.5·fwd (9) | inf (-) | 40 | 20.00 | 16 | 0.40 | 2 | brute | 0.0 |
| local_sgd_chain[K=2,q=1,d=2] | 3·fwd (18) | 1·fwd (6) | [84, 98] (not proved) | 49.00 | 16 | 0.16 | 9 | milp+ls | 273.5 |
| local_sgd_chain[K=2,q=1,d=2] | 3·fwd (18) | 4·fwd (24) | 40 | 20.00 | 16 | 0.40 | 2 | brute | 0.0 |
| local_sgd_chain[K=2,q=1,d=2] | 3·fwd (18) | inf (-) | 40 | 20.00 | 16 | 0.40 | 2 | brute | 0.1 |
| local_sgd_chain[K=2,q=1,d=2] | inf (-) | 1·fwd (6) | [84, 98] (not proved) | 49.00 | 16 | 0.16 | 9 | milp | 269.8 |
| local_sgd_chain[K=2,q=1,d=2] | inf (-) | 4·fwd (24) | 40 | 20.00 | 16 | 0.40 | 2 | brute | 0.0 |
| local_sgd_chain[K=2,q=1,d=2] | inf (-) | inf (-) | 28 | 14.00 | 16 | 0.57 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=3,q=1,d=2] | 1·fwd (6) | 1·fwd (6) | [36, 148] (not proved) | 49.33 | 24 | 0.16 | 14 | ls | 23.6 |
| local_sgd_chain[K=3,q=1,d=2] | 1·fwd (6) | 4·fwd (24) | [36, 108] (not proved) | 36.00 | 24 | 0.22 | 6 | ls | 24.7 |
| local_sgd_chain[K=3,q=1,d=2] | 1·fwd (6) | inf (-) | [36, 108] (not proved) | 36.00 | 24 | 0.22 | 6 | ls | 23.6 |
| local_sgd_chain[K=3,q=1,d=2] | 1.5·fwd (9) | 1·fwd (6) | [36, 148] (not proved) | 49.33 | 24 | 0.16 | 14 | ls | 23.5 |
| local_sgd_chain[K=3,q=1,d=2] | 1.5·fwd (9) | 4·fwd (24) | [36, 60] (not proved) | 20.00 | 24 | 0.40 | 3 | ls | 24.1 |
| local_sgd_chain[K=3,q=1,d=2] | 1.5·fwd (9) | inf (-) | [36, 60] (not proved) | 20.00 | 24 | 0.40 | 3 | ls | 23.9 |
| local_sgd_chain[K=3,q=1,d=2] | 3·fwd (18) | 1·fwd (6) | [36, 148] (not proved) | 49.33 | 24 | 0.16 | 14 | ls | 23.5 |
| local_sgd_chain[K=3,q=1,d=2] | 3·fwd (18) | 4·fwd (24) | [36, 60] (not proved) | 20.00 | 24 | 0.40 | 3 | ls | 24.3 |
| local_sgd_chain[K=3,q=1,d=2] | 3·fwd (18) | inf (-) | [36, 56] (not proved) | 18.67 | 24 | 0.43 | 2 | ls | 22.8 |
| local_sgd_chain[K=3,q=1,d=2] | inf (-) | 1·fwd (6) | [36, 148] (not proved) | 49.33 | 24 | 0.16 | 14 | ls | 18.6 |
| local_sgd_chain[K=3,q=1,d=2] | inf (-) | 4·fwd (24) | [36, 60] (not proved) | 20.00 | 24 | 0.40 | 3 | ls | 19.3 |
| local_sgd_chain[K=3,q=1,d=2] | inf (-) | inf (-) | 36 | 12.00 | 24 | 0.67 | 1 | one-ru | 0.0 |

### micro local-sgd chain (d=2, q=2)

| circuit | F | G | I* (B) | I*/token | useful MACs | kappa = MACs/I* | RUs | solver | s |
|---|---|---|---|---|---|---|---|---|---|
| local_sgd_chain[K=1,q=2,d=2] | 1·fwd (12) | 1·fwd (12) | 60 | 30.00 | 16 | 0.27 | 3 | milp | 238.1 |
| local_sgd_chain[K=1,q=2,d=2] | 1·fwd (12) | 4·fwd (48) | 42 | 21.00 | 16 | 0.38 | 2 | brute | 0.1 |
| local_sgd_chain[K=1,q=2,d=2] | 1·fwd (12) | inf (-) | 42 | 21.00 | 16 | 0.38 | 2 | brute | 0.1 |
| local_sgd_chain[K=1,q=2,d=2] | 1.5·fwd (18) | 1·fwd (12) | 60 | 30.00 | 16 | 0.27 | 3 | milp | 92.1 |
| local_sgd_chain[K=1,q=2,d=2] | 1.5·fwd (18) | 4·fwd (48) | 28 | 14.00 | 16 | 0.57 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=1,q=2,d=2] | 1.5·fwd (18) | inf (-) | 28 | 14.00 | 16 | 0.57 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=1,q=2,d=2] | 3·fwd (36) | 1·fwd (12) | 60 | 30.00 | 16 | 0.27 | 3 | milp | 104.7 |
| local_sgd_chain[K=1,q=2,d=2] | 3·fwd (36) | 4·fwd (48) | 28 | 14.00 | 16 | 0.57 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=1,q=2,d=2] | 3·fwd (36) | inf (-) | 28 | 14.00 | 16 | 0.57 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=1,q=2,d=2] | inf (-) | 1·fwd (12) | 60 | 30.00 | 16 | 0.27 | 3 | milp | 107.0 |
| local_sgd_chain[K=1,q=2,d=2] | inf (-) | 4·fwd (48) | 28 | 14.00 | 16 | 0.57 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=1,q=2,d=2] | inf (-) | inf (-) | 28 | 14.00 | 16 | 0.57 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=2,q=2,d=2] | 1·fwd (12) | 1·fwd (12) | [44, 136] (not proved) | 34.00 | 32 | 0.24 | 8 | ls | 19.7 |
| local_sgd_chain[K=2,q=2,d=2] | 1·fwd (12) | 4·fwd (48) | [44, 88] (not proved) | 22.00 | 32 | 0.36 | 3 | ls | 18.9 |
| local_sgd_chain[K=2,q=2,d=2] | 1·fwd (12) | inf (-) | [44, 88] (not proved) | 22.00 | 32 | 0.36 | 3 | ls | 18.7 |
| local_sgd_chain[K=2,q=2,d=2] | 1.5·fwd (18) | 1·fwd (12) | [44, 136] (not proved) | 34.00 | 32 | 0.24 | 8 | ls | 20.4 |
| local_sgd_chain[K=2,q=2,d=2] | 1.5·fwd (18) | 4·fwd (48) | [44, 56] (not proved) | 14.00 | 32 | 0.57 | 2 | ls | 19.1 |
| local_sgd_chain[K=2,q=2,d=2] | 1.5·fwd (18) | inf (-) | [44, 56] (not proved) | 14.00 | 32 | 0.57 | 2 | ls | 18.3 |
| local_sgd_chain[K=2,q=2,d=2] | 3·fwd (36) | 1·fwd (12) | [44, 136] (not proved) | 34.00 | 32 | 0.24 | 8 | ls | 20.7 |
| local_sgd_chain[K=2,q=2,d=2] | 3·fwd (36) | 4·fwd (48) | [44, 56] (not proved) | 14.00 | 32 | 0.57 | 2 | ls | 19.1 |
| local_sgd_chain[K=2,q=2,d=2] | 3·fwd (36) | inf (-) | 44 | 11.00 | 32 | 0.73 | 1 | one-ru | 0.0 |
| local_sgd_chain[K=2,q=2,d=2] | inf (-) | 1·fwd (12) | [44, 136] (not proved) | 34.00 | 32 | 0.24 | 8 | ls | 17.0 |
| local_sgd_chain[K=2,q=2,d=2] | inf (-) | 4·fwd (48) | [44, 56] (not proved) | 14.00 | 32 | 0.57 | 2 | ls | 16.7 |
| local_sgd_chain[K=2,q=2,d=2] | inf (-) | inf (-) | 44 | 11.00 | 32 | 0.73 | 1 | one-ru | 0.0 |

### micro fan-out (shared produced weight)

| circuit | F | G | I* (B) | I*/token | useful MACs | kappa = MACs/I* | RUs | solver | s |
|---|---|---|---|---|---|---|---|---|---|
| fanout_shared_weight[n=4,q=1,d=2] | 1·fwd (6) | 1·fwd (6) | [56, 72] (not proved) | 18.00 | 16 | 0.22 | 6 | milp | 259.0 |
| fanout_shared_weight[n=4,q=1,d=2] | 1·fwd (6) | 4·fwd (24) | 44 | 11.00 | 16 | 0.36 | 2 | brute | 0.0 |
| fanout_shared_weight[n=4,q=1,d=2] | 1·fwd (6) | inf (-) | 44 | 11.00 | 16 | 0.36 | 2 | brute | 0.0 |
| fanout_shared_weight[n=4,q=1,d=2] | 1.5·fwd (9) | 1·fwd (6) | [56, 72] (not proved) | 18.00 | 16 | 0.22 | 6 | milp | 254.1 |
| fanout_shared_weight[n=4,q=1,d=2] | 1.5·fwd (9) | 4·fwd (24) | 44 | 11.00 | 16 | 0.36 | 2 | brute | 0.0 |
| fanout_shared_weight[n=4,q=1,d=2] | 1.5·fwd (9) | inf (-) | 36 | 9.00 | 16 | 0.44 | 1 | one-ru | 0.0 |
| fanout_shared_weight[n=4,q=1,d=2] | 3·fwd (18) | 1·fwd (6) | [56, 72] (not proved) | 18.00 | 16 | 0.22 | 6 | milp | 254.7 |
| fanout_shared_weight[n=4,q=1,d=2] | 3·fwd (18) | 4·fwd (24) | 44 | 11.00 | 16 | 0.36 | 2 | brute | 0.0 |
| fanout_shared_weight[n=4,q=1,d=2] | 3·fwd (18) | inf (-) | 36 | 9.00 | 16 | 0.44 | 1 | one-ru | 0.0 |
| fanout_shared_weight[n=4,q=1,d=2] | inf (-) | 1·fwd (6) | [56, 72] (not proved) | 18.00 | 16 | 0.22 | 6 | milp | 250.3 |
| fanout_shared_weight[n=4,q=1,d=2] | inf (-) | 4·fwd (24) | 44 | 11.00 | 16 | 0.36 | 2 | brute | 0.0 |
| fanout_shared_weight[n=4,q=1,d=2] | inf (-) | inf (-) | 36 | 9.00 | 16 | 0.44 | 1 | one-ru | 0.0 |

### micro deep chain (fixed weights)

| circuit | F | G | I* (B) | I*/token | useful MACs | kappa = MACs/I* | RUs | solver | s |
|---|---|---|---|---|---|---|---|---|---|
| deep_chain[depth=4,q=1,d=2,W=fixed] | 1·fwd (6) | 1·fwd (6) | 16 | 16.00 | 16 | 1.00 | 4 | brute | 0.0 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 1·fwd (6) | 4·fwd (24) | 16 | 16.00 | 16 | 1.00 | 4 | brute | 0.0 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 1·fwd (6) | inf (-) | 16 | 16.00 | 16 | 1.00 | 4 | brute | 0.0 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 1.5·fwd (9) | 1·fwd (6) | 16 | 16.00 | 16 | 1.00 | 4 | brute | 0.0 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 1.5·fwd (9) | 4·fwd (24) | 8 | 8.00 | 16 | 2.00 | 2 | brute | 0.0 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 1.5·fwd (9) | inf (-) | 8 | 8.00 | 16 | 2.00 | 2 | brute | 0.0 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 3·fwd (18) | 1·fwd (6) | 16 | 16.00 | 16 | 1.00 | 4 | brute | 0.0 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 3·fwd (18) | 4·fwd (24) | 8 | 8.00 | 16 | 2.00 | 2 | brute | 0.0 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 3·fwd (18) | inf (-) | 8 | 8.00 | 16 | 2.00 | 2 | brute | 0.0 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | inf (-) | 1·fwd (6) | 16 | 16.00 | 16 | 1.00 | 4 | brute | 0.0 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | inf (-) | 4·fwd (24) | 4 | 4.00 | 16 | 4.00 | 1 | one-ru | 0.0 |
| deep_chain[depth=4,q=1,d=2,W=fixed] | inf (-) | inf (-) | 4 | 4.00 | 16 | 4.00 | 1 | one-ru | 0.0 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 1·fwd (6) | 1·fwd (6) | 24 | 24.00 | 24 | 1.00 | 6 | brute | 0.0 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 1·fwd (6) | 4·fwd (24) | 24 | 24.00 | 24 | 1.00 | 6 | brute | 0.2 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 1·fwd (6) | inf (-) | 24 | 24.00 | 24 | 1.00 | 6 | brute | 0.2 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 1.5·fwd (9) | 1·fwd (6) | 24 | 24.00 | 24 | 1.00 | 6 | brute | 0.0 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 1.5·fwd (9) | 4·fwd (24) | 12 | 12.00 | 24 | 2.00 | 3 | brute | 0.0 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 1.5·fwd (9) | inf (-) | 12 | 12.00 | 24 | 2.00 | 3 | brute | 0.0 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 3·fwd (18) | 1·fwd (6) | 24 | 24.00 | 24 | 1.00 | 6 | brute | 0.0 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 3·fwd (18) | 4·fwd (24) | 8 | 8.00 | 24 | 3.00 | 2 | brute | 0.0 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 3·fwd (18) | inf (-) | 8 | 8.00 | 24 | 3.00 | 2 | brute | 0.0 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | inf (-) | 1·fwd (6) | 24 | 24.00 | 24 | 1.00 | 6 | brute | 0.0 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | inf (-) | 4·fwd (24) | 8 | 8.00 | 24 | 3.00 | 2 | brute | 0.0 |
| deep_chain[depth=6,q=1,d=2,W=fixed] | inf (-) | inf (-) | 4 | 4.00 | 24 | 6.00 | 1 | one-ru | 0.0 |

### registry inference-dense @ tiny

| circuit | F | G | I* (B) | I*/token | useful MACs | kappa = MACs/I* | RUs | solver | s |
|---|---|---|---|---|---|---|---|---|---|
| inference-dense | 1·fwd (592) | 1·fwd (592) | 8 | 4.00 | 320 | 40.00 | 1 | one-ru | 0.0 |
| inference-dense | 1·fwd (592) | 4·fwd (2368) | 8 | 4.00 | 320 | 40.00 | 1 | one-ru | 0.0 |
| inference-dense | 1·fwd (592) | inf (-) | 8 | 4.00 | 320 | 40.00 | 1 | one-ru | 0.0 |
| inference-dense | 1.5·fwd (888) | 1·fwd (592) | 8 | 4.00 | 320 | 40.00 | 1 | one-ru | 0.0 |
| inference-dense | 1.5·fwd (888) | 4·fwd (2368) | 8 | 4.00 | 320 | 40.00 | 1 | one-ru | 0.0 |
| inference-dense | 1.5·fwd (888) | inf (-) | 8 | 4.00 | 320 | 40.00 | 1 | one-ru | 0.0 |
| inference-dense | 3·fwd (1776) | 1·fwd (592) | 8 | 4.00 | 320 | 40.00 | 1 | one-ru | 0.0 |
| inference-dense | 3·fwd (1776) | 4·fwd (2368) | 8 | 4.00 | 320 | 40.00 | 1 | one-ru | 0.0 |
| inference-dense | 3·fwd (1776) | inf (-) | 8 | 4.00 | 320 | 40.00 | 1 | one-ru | 0.0 |
| inference-dense | inf (-) | 1·fwd (592) | 8 | 4.00 | 320 | 40.00 | 1 | one-ru | 0.0 |
| inference-dense | inf (-) | 4·fwd (2368) | 8 | 4.00 | 320 | 40.00 | 1 | one-ru | 0.0 |
| inference-dense | inf (-) | inf (-) | 8 | 4.00 | 320 | 40.00 | 1 | one-ru | 0.0 |
| inference-dense | 1·fwd (936) | 1·fwd (936) | 12 | 4.00 | 504 | 42.00 | 1 | one-ru | 0.0 |
| inference-dense | 1·fwd (936) | 4·fwd (3744) | 12 | 4.00 | 504 | 42.00 | 1 | one-ru | 0.0 |
| inference-dense | 1·fwd (936) | inf (-) | 12 | 4.00 | 504 | 42.00 | 1 | one-ru | 0.0 |
| inference-dense | 1.5·fwd (1404) | 1·fwd (936) | 12 | 4.00 | 504 | 42.00 | 1 | one-ru | 0.0 |
| inference-dense | 1.5·fwd (1404) | 4·fwd (3744) | 12 | 4.00 | 504 | 42.00 | 1 | one-ru | 0.0 |
| inference-dense | 1.5·fwd (1404) | inf (-) | 12 | 4.00 | 504 | 42.00 | 1 | one-ru | 0.0 |
| inference-dense | 3·fwd (2808) | 1·fwd (936) | 12 | 4.00 | 504 | 42.00 | 1 | one-ru | 0.0 |
| inference-dense | 3·fwd (2808) | 4·fwd (3744) | 12 | 4.00 | 504 | 42.00 | 1 | one-ru | 0.0 |
| inference-dense | 3·fwd (2808) | inf (-) | 12 | 4.00 | 504 | 42.00 | 1 | one-ru | 0.0 |
| inference-dense | inf (-) | 1·fwd (936) | 12 | 4.00 | 504 | 42.00 | 1 | one-ru | 0.0 |
| inference-dense | inf (-) | 4·fwd (3744) | 12 | 4.00 | 504 | 42.00 | 1 | one-ru | 0.0 |
| inference-dense | inf (-) | inf (-) | 12 | 4.00 | 504 | 42.00 | 1 | one-ru | 0.0 |
| inference-dense | 1·fwd (1312) | 1·fwd (1312) | 16 | 4.00 | 704 | 44.00 | 1 | one-ru | 0.0 |
| inference-dense | 1·fwd (1312) | 4·fwd (5248) | 16 | 4.00 | 704 | 44.00 | 1 | one-ru | 0.0 |
| inference-dense | 1·fwd (1312) | inf (-) | 16 | 4.00 | 704 | 44.00 | 1 | one-ru | 0.0 |
| inference-dense | 1.5·fwd (1968) | 1·fwd (1312) | 16 | 4.00 | 704 | 44.00 | 1 | one-ru | 0.0 |
| inference-dense | 1.5·fwd (1968) | 4·fwd (5248) | 16 | 4.00 | 704 | 44.00 | 1 | one-ru | 0.0 |
| inference-dense | 1.5·fwd (1968) | inf (-) | 16 | 4.00 | 704 | 44.00 | 1 | one-ru | 0.0 |
| inference-dense | 3·fwd (3936) | 1·fwd (1312) | 16 | 4.00 | 704 | 44.00 | 1 | one-ru | 0.0 |
| inference-dense | 3·fwd (3936) | 4·fwd (5248) | 16 | 4.00 | 704 | 44.00 | 1 | one-ru | 0.0 |
| inference-dense | 3·fwd (3936) | inf (-) | 16 | 4.00 | 704 | 44.00 | 1 | one-ru | 0.0 |
| inference-dense | inf (-) | 1·fwd (1312) | 16 | 4.00 | 704 | 44.00 | 1 | one-ru | 0.0 |
| inference-dense | inf (-) | 4·fwd (5248) | 16 | 4.00 | 704 | 44.00 | 1 | one-ru | 0.0 |
| inference-dense | inf (-) | inf (-) | 16 | 4.00 | 704 | 44.00 | 1 | one-ru | 0.0 |

### registry local-sgd @ tiny (q=2)

| circuit | F | G | I* (B) | I*/token | useful MACs | kappa = MACs/I* | RUs | solver | s |
|---|---|---|---|---|---|---|---|---|---|
| local-sgd | 1·fwd (592) | 1·fwd (592) | beyond exact scale (2888 gates) | - | 976 | - | - |  | 0.0 |
| local-sgd | 1·fwd (592) | 4·fwd (2368) | beyond exact scale (2888 gates) | - | 976 | - | - |  | 0.0 |
| local-sgd | 1·fwd (592) | inf (-) | beyond exact scale (2888 gates) | - | 976 | - | - |  | 0.0 |
| local-sgd | 1.5·fwd (888) | 1·fwd (592) | beyond exact scale (2888 gates) | - | 976 | - | - |  | 0.0 |
| local-sgd | 1.5·fwd (888) | 4·fwd (2368) | beyond exact scale (2888 gates) | - | 976 | - | - |  | 0.0 |
| local-sgd | 1.5·fwd (888) | inf (-) | beyond exact scale (2888 gates) | - | 976 | - | - |  | 0.0 |
| local-sgd | 3·fwd (1776) | 1·fwd (592) | beyond exact scale (2888 gates) | - | 976 | - | - |  | 0.0 |
| local-sgd | 3·fwd (1776) | 4·fwd (2368) | beyond exact scale (2888 gates) | - | 976 | - | - |  | 0.0 |
| local-sgd | 3·fwd (1776) | inf (-) | 396 | 198.00 | 976 | 2.46 | 1 | one-ru | 0.0 |
| local-sgd | inf (-) | 1·fwd (592) | beyond exact scale (2888 gates) | - | 976 | - | - |  | 0.0 |
| local-sgd | inf (-) | 4·fwd (2368) | beyond exact scale (2888 gates) | - | 976 | - | - |  | 0.0 |
| local-sgd | inf (-) | inf (-) | 396 | 198.00 | 976 | 2.46 | 1 | one-ru | 0.0 |
| local-sgd | 1·fwd (592) | 1·fwd (592) | beyond exact scale (5776 gates) | - | 1952 | - | - |  | 0.0 |
| local-sgd | 1·fwd (592) | 4·fwd (2368) | beyond exact scale (5776 gates) | - | 1952 | - | - |  | 0.0 |
| local-sgd | 1·fwd (592) | inf (-) | beyond exact scale (5776 gates) | - | 1952 | - | - |  | 0.0 |
| local-sgd | 1.5·fwd (888) | 1·fwd (592) | beyond exact scale (5776 gates) | - | 1952 | - | - |  | 0.0 |
| local-sgd | 1.5·fwd (888) | 4·fwd (2368) | beyond exact scale (5776 gates) | - | 1952 | - | - |  | 0.0 |
| local-sgd | 1.5·fwd (888) | inf (-) | beyond exact scale (5776 gates) | - | 1952 | - | - |  | 0.0 |
| local-sgd | 3·fwd (1776) | 1·fwd (592) | beyond exact scale (5776 gates) | - | 1952 | - | - |  | 0.0 |
| local-sgd | 3·fwd (1776) | 4·fwd (2368) | beyond exact scale (5776 gates) | - | 1952 | - | - |  | 0.0 |
| local-sgd | 3·fwd (1776) | inf (-) | beyond exact scale (5776 gates) | - | 1952 | - | - |  | 0.0 |
| local-sgd | inf (-) | 1·fwd (592) | beyond exact scale (5776 gates) | - | 1952 | - | - |  | 0.0 |
| local-sgd | inf (-) | 4·fwd (2368) | beyond exact scale (5776 gates) | - | 1952 | - | - |  | 0.0 |
| local-sgd | inf (-) | inf (-) | 412 | 103.00 | 1952 | 4.74 | 1 | one-ru | 0.0 |
| local-sgd | 1·fwd (592) | 1·fwd (592) | beyond exact scale (8664 gates) | - | 2928 | - | - |  | 0.0 |
| local-sgd | 1·fwd (592) | 4·fwd (2368) | beyond exact scale (8664 gates) | - | 2928 | - | - |  | 0.0 |
| local-sgd | 1·fwd (592) | inf (-) | beyond exact scale (8664 gates) | - | 2928 | - | - |  | 0.0 |
| local-sgd | 1.5·fwd (888) | 1·fwd (592) | beyond exact scale (8664 gates) | - | 2928 | - | - |  | 0.0 |
| local-sgd | 1.5·fwd (888) | 4·fwd (2368) | beyond exact scale (8664 gates) | - | 2928 | - | - |  | 0.0 |
| local-sgd | 1.5·fwd (888) | inf (-) | beyond exact scale (8664 gates) | - | 2928 | - | - |  | 0.0 |
| local-sgd | 3·fwd (1776) | 1·fwd (592) | beyond exact scale (8664 gates) | - | 2928 | - | - |  | 0.0 |
| local-sgd | 3·fwd (1776) | 4·fwd (2368) | beyond exact scale (8664 gates) | - | 2928 | - | - |  | 0.0 |
| local-sgd | 3·fwd (1776) | inf (-) | beyond exact scale (8664 gates) | - | 2928 | - | - |  | 0.0 |
| local-sgd | inf (-) | 1·fwd (592) | beyond exact scale (8664 gates) | - | 2928 | - | - |  | 0.0 |
| local-sgd | inf (-) | 4·fwd (2368) | beyond exact scale (8664 gates) | - | 2928 | - | - |  | 0.0 |
| local-sgd | inf (-) | inf (-) | 428 | 71.33 | 2928 | 6.84 | 1 | one-ru | 0.0 |

### registry inference-dense @ tiny2

| circuit | F | G | I* (B) | I*/token | useful MACs | kappa = MACs/I* | RUs | solver | s |
|---|---|---|---|---|---|---|---|---|---|
| inference-dense | 1·fwd (1290) | 1·fwd (1290) | 8 | 4.00 | 704 | 88.00 | 1 | one-ru | 0.0 |
| inference-dense | 1·fwd (1290) | 4·fwd (5160) | 8 | 4.00 | 704 | 88.00 | 1 | one-ru | 0.0 |
| inference-dense | 1·fwd (1290) | inf (-) | 8 | 4.00 | 704 | 88.00 | 1 | one-ru | 0.0 |
| inference-dense | 1.5·fwd (1935) | 1·fwd (1290) | 8 | 4.00 | 704 | 88.00 | 1 | one-ru | 0.0 |
| inference-dense | 1.5·fwd (1935) | 4·fwd (5160) | 8 | 4.00 | 704 | 88.00 | 1 | one-ru | 0.0 |
| inference-dense | 1.5·fwd (1935) | inf (-) | 8 | 4.00 | 704 | 88.00 | 1 | one-ru | 0.0 |
| inference-dense | 3·fwd (3870) | 1·fwd (1290) | 8 | 4.00 | 704 | 88.00 | 1 | one-ru | 0.0 |
| inference-dense | 3·fwd (3870) | 4·fwd (5160) | 8 | 4.00 | 704 | 88.00 | 1 | one-ru | 0.0 |
| inference-dense | 3·fwd (3870) | inf (-) | 8 | 4.00 | 704 | 88.00 | 1 | one-ru | 0.0 |
| inference-dense | inf (-) | 1·fwd (1290) | 8 | 4.00 | 704 | 88.00 | 1 | one-ru | 0.0 |
| inference-dense | inf (-) | 4·fwd (5160) | 8 | 4.00 | 704 | 88.00 | 1 | one-ru | 0.0 |
| inference-dense | inf (-) | inf (-) | 8 | 4.00 | 704 | 88.00 | 1 | one-ru | 0.0 |
| inference-dense | 1·fwd (2079) | 1·fwd (2079) | 12 | 4.00 | 1104 | 92.00 | 1 | one-ru | 0.0 |
| inference-dense | 1·fwd (2079) | 4·fwd (8316) | 12 | 4.00 | 1104 | 92.00 | 1 | one-ru | 0.0 |
| inference-dense | 1·fwd (2079) | inf (-) | 12 | 4.00 | 1104 | 92.00 | 1 | one-ru | 0.0 |
| inference-dense | 1.5·fwd (3119) | 1·fwd (2079) | 12 | 4.00 | 1104 | 92.00 | 1 | one-ru | 0.0 |
| inference-dense | 1.5·fwd (3119) | 4·fwd (8316) | 12 | 4.00 | 1104 | 92.00 | 1 | one-ru | 0.0 |
| inference-dense | 1.5·fwd (3119) | inf (-) | 12 | 4.00 | 1104 | 92.00 | 1 | one-ru | 0.0 |
| inference-dense | 3·fwd (6237) | 1·fwd (2079) | 12 | 4.00 | 1104 | 92.00 | 1 | one-ru | 0.0 |
| inference-dense | 3·fwd (6237) | 4·fwd (8316) | 12 | 4.00 | 1104 | 92.00 | 1 | one-ru | 0.0 |
| inference-dense | 3·fwd (6237) | inf (-) | 12 | 4.00 | 1104 | 92.00 | 1 | one-ru | 0.0 |
| inference-dense | inf (-) | 1·fwd (2079) | 12 | 4.00 | 1104 | 92.00 | 1 | one-ru | 0.0 |
| inference-dense | inf (-) | 4·fwd (8316) | 12 | 4.00 | 1104 | 92.00 | 1 | one-ru | 0.0 |
| inference-dense | inf (-) | inf (-) | 12 | 4.00 | 1104 | 92.00 | 1 | one-ru | 0.0 |
| inference-dense | 1·fwd (2964) | 1·fwd (2964) | 16 | 4.00 | 1536 | 96.00 | 1 | one-ru | 0.0 |
| inference-dense | 1·fwd (2964) | 4·fwd (11856) | 16 | 4.00 | 1536 | 96.00 | 1 | one-ru | 0.0 |
| inference-dense | 1·fwd (2964) | inf (-) | 16 | 4.00 | 1536 | 96.00 | 1 | one-ru | 0.0 |
| inference-dense | 1.5·fwd (4446) | 1·fwd (2964) | 16 | 4.00 | 1536 | 96.00 | 1 | one-ru | 0.0 |
| inference-dense | 1.5·fwd (4446) | 4·fwd (11856) | 16 | 4.00 | 1536 | 96.00 | 1 | one-ru | 0.0 |
| inference-dense | 1.5·fwd (4446) | inf (-) | 16 | 4.00 | 1536 | 96.00 | 1 | one-ru | 0.0 |
| inference-dense | 3·fwd (8892) | 1·fwd (2964) | 16 | 4.00 | 1536 | 96.00 | 1 | one-ru | 0.0 |
| inference-dense | 3·fwd (8892) | 4·fwd (11856) | 16 | 4.00 | 1536 | 96.00 | 1 | one-ru | 0.0 |
| inference-dense | 3·fwd (8892) | inf (-) | 16 | 4.00 | 1536 | 96.00 | 1 | one-ru | 0.0 |
| inference-dense | inf (-) | 1·fwd (2964) | 16 | 4.00 | 1536 | 96.00 | 1 | one-ru | 0.0 |
| inference-dense | inf (-) | 4·fwd (11856) | 16 | 4.00 | 1536 | 96.00 | 1 | one-ru | 0.0 |
| inference-dense | inf (-) | inf (-) | 16 | 4.00 | 1536 | 96.00 | 1 | one-ru | 0.0 |

### registry local-sgd @ tiny2 (q=2)

| circuit | F | G | I* (B) | I*/token | useful MACs | kappa = MACs/I* | RUs | solver | s |
|---|---|---|---|---|---|---|---|---|---|
| local-sgd | 1·fwd (1290) | 1·fwd (1290) | beyond exact scale (6272 gates) | - | 2144 | - | - |  | 0.0 |
| local-sgd | 1·fwd (1290) | 4·fwd (5160) | beyond exact scale (6272 gates) | - | 2144 | - | - |  | 0.0 |
| local-sgd | 1·fwd (1290) | inf (-) | beyond exact scale (6272 gates) | - | 2144 | - | - |  | 0.0 |
| local-sgd | 1.5·fwd (1935) | 1·fwd (1290) | beyond exact scale (6272 gates) | - | 2144 | - | - |  | 0.0 |
| local-sgd | 1.5·fwd (1935) | 4·fwd (5160) | beyond exact scale (6272 gates) | - | 2144 | - | - |  | 0.0 |
| local-sgd | 1.5·fwd (1935) | inf (-) | beyond exact scale (6272 gates) | - | 2144 | - | - |  | 0.0 |
| local-sgd | 3·fwd (3870) | 1·fwd (1290) | beyond exact scale (6272 gates) | - | 2144 | - | - |  | 0.0 |
| local-sgd | 3·fwd (3870) | 4·fwd (5160) | beyond exact scale (6272 gates) | - | 2144 | - | - |  | 0.0 |
| local-sgd | 3·fwd (3870) | inf (-) | 764 | 382.00 | 2144 | 2.81 | 1 | one-ru | 0.0 |
| local-sgd | inf (-) | 1·fwd (1290) | beyond exact scale (6272 gates) | - | 2144 | - | - |  | 0.0 |
| local-sgd | inf (-) | 4·fwd (5160) | beyond exact scale (6272 gates) | - | 2144 | - | - |  | 0.0 |
| local-sgd | inf (-) | inf (-) | 764 | 382.00 | 2144 | 2.81 | 1 | one-ru | 0.0 |
| local-sgd | 1·fwd (1290) | 1·fwd (1290) | beyond exact scale (12544 gates) | - | 4288 | - | - |  | 0.0 |
| local-sgd | 1·fwd (1290) | 4·fwd (5160) | beyond exact scale (12544 gates) | - | 4288 | - | - |  | 0.0 |
| local-sgd | 1·fwd (1290) | inf (-) | beyond exact scale (12544 gates) | - | 4288 | - | - |  | 0.0 |
| local-sgd | 1.5·fwd (1935) | 1·fwd (1290) | beyond exact scale (12544 gates) | - | 4288 | - | - |  | 0.0 |
| local-sgd | 1.5·fwd (1935) | 4·fwd (5160) | beyond exact scale (12544 gates) | - | 4288 | - | - |  | 0.0 |
| local-sgd | 1.5·fwd (1935) | inf (-) | beyond exact scale (12544 gates) | - | 4288 | - | - |  | 0.0 |
| local-sgd | 3·fwd (3870) | 1·fwd (1290) | beyond exact scale (12544 gates) | - | 4288 | - | - |  | 0.0 |
| local-sgd | 3·fwd (3870) | 4·fwd (5160) | beyond exact scale (12544 gates) | - | 4288 | - | - |  | 0.0 |
| local-sgd | 3·fwd (3870) | inf (-) | beyond exact scale (12544 gates) | - | 4288 | - | - |  | 0.0 |
| local-sgd | inf (-) | 1·fwd (1290) | beyond exact scale (12544 gates) | - | 4288 | - | - |  | 0.0 |
| local-sgd | inf (-) | 4·fwd (5160) | beyond exact scale (12544 gates) | - | 4288 | - | - |  | 0.0 |
| local-sgd | inf (-) | inf (-) | 780 | 195.00 | 4288 | 5.50 | 1 | one-ru | 0.0 |
| local-sgd | 1·fwd (1290) | 1·fwd (1290) | beyond exact scale (18816 gates) | - | 6432 | - | - |  | 0.0 |
| local-sgd | 1·fwd (1290) | 4·fwd (5160) | beyond exact scale (18816 gates) | - | 6432 | - | - |  | 0.0 |
| local-sgd | 1·fwd (1290) | inf (-) | beyond exact scale (18816 gates) | - | 6432 | - | - |  | 0.0 |
| local-sgd | 1.5·fwd (1935) | 1·fwd (1290) | beyond exact scale (18816 gates) | - | 6432 | - | - |  | 0.0 |
| local-sgd | 1.5·fwd (1935) | 4·fwd (5160) | beyond exact scale (18816 gates) | - | 6432 | - | - |  | 0.0 |
| local-sgd | 1.5·fwd (1935) | inf (-) | beyond exact scale (18816 gates) | - | 6432 | - | - |  | 0.0 |
| local-sgd | 3·fwd (3870) | 1·fwd (1290) | beyond exact scale (18816 gates) | - | 6432 | - | - |  | 0.0 |
| local-sgd | 3·fwd (3870) | 4·fwd (5160) | beyond exact scale (18816 gates) | - | 6432 | - | - |  | 0.0 |
| local-sgd | 3·fwd (3870) | inf (-) | beyond exact scale (18816 gates) | - | 6432 | - | - |  | 0.0 |
| local-sgd | inf (-) | 1·fwd (1290) | beyond exact scale (18816 gates) | - | 6432 | - | - |  | 0.0 |
| local-sgd | inf (-) | 4·fwd (5160) | beyond exact scale (18816 gates) | - | 6432 | - | - |  | 0.0 |
| local-sgd | inf (-) | inf (-) | 796 | 132.67 | 6432 | 8.08 | 1 | one-ru | 0.0 |

## Per-step marginal `I*(K) - I*(K-1)` on the local-SGD chains

| chain | F | G | I*(K=1) | I*(K=2) | I*(K=3) | marginal K=2 | marginal K=3 |
|---|---|---|---|---|---|---|---|
| micro local-sgd chain (d=1, q=1) | 1·fwd | 1·fwd | 20 | 40 | 60 | 20 | 20 |
| micro local-sgd chain (d=1, q=1) | 1·fwd | 4·fwd | 20 | 40 | 60 | 20 | 20 |
| micro local-sgd chain (d=1, q=1) | 1·fwd | inf | 20 | 40 | 60 | 20 | 20 |
| micro local-sgd chain (d=1, q=1) | 1.5·fwd | 1·fwd | 20 | 40 | 60 | 20 | 20 |
| micro local-sgd chain (d=1, q=1) | 1.5·fwd | 4·fwd | 18 | 34 | 52 | 16 | 18 |
| micro local-sgd chain (d=1, q=1) | 1.5·fwd | inf | 18 | 34 | 52 | 16 | 18 |
| micro local-sgd chain (d=1, q=1) | 3·fwd | 1·fwd | 20 | 40 | 60 | 20 | 20 |
| micro local-sgd chain (d=1, q=1) | 3·fwd | 4·fwd | 14 | 26 | 40 | 12 | 14 |
| micro local-sgd chain (d=1, q=1) | 3·fwd | inf | 14 | 26 | 40 | 12 | 14 |
| micro local-sgd chain (d=1, q=1) | inf | 1·fwd | 20 | 40 | 60 | 20 | 20 |
| micro local-sgd chain (d=1, q=1) | inf | 4·fwd | 10 | 20 | 30 | 10 | 10 |
| micro local-sgd chain (d=1, q=1) | inf | inf | 10 | 14 | 18 | 4 | 4 |
| micro local-sgd chain (d=1, q=2) | 1·fwd | 1·fwd | 26 | 52 | 78 | 26 | 26 |
| micro local-sgd chain (d=1, q=2) | 1·fwd | 4·fwd | 24 | 48 | 72 | 24 | 24 |
| micro local-sgd chain (d=1, q=2) | 1·fwd | inf | 24 | 48 | 72 | 24 | 24 |
| micro local-sgd chain (d=1, q=2) | 1.5·fwd | 1·fwd | 26 | 52 | 78 | 26 | 26 |
| micro local-sgd chain (d=1, q=2) | 1.5·fwd | 4·fwd | 24 | 46 | 68 | 22 | 22 |
| micro local-sgd chain (d=1, q=2) | 1.5·fwd | inf | 24 | 46 | 68 | 22 | 22 |
| micro local-sgd chain (d=1, q=2) | 3·fwd | 1·fwd | 26 | 52 | 78 | 26 | 26 |
| micro local-sgd chain (d=1, q=2) | 3·fwd | 4·fwd | 14 | 28 | 42 | 14 | 14 |
| micro local-sgd chain (d=1, q=2) | 3·fwd | inf | 14 | 28 | 42 | 14 | 14 |
| micro local-sgd chain (d=1, q=2) | inf | 1·fwd | 26 | 52 | 78 | 26 | 26 |
| micro local-sgd chain (d=1, q=2) | inf | 4·fwd | 14 | 26 | 40 | 12 | 14 |
| micro local-sgd chain (d=1, q=2) | inf | inf | 14 | 22 | 30 | 8 | 8 |
| micro local-sgd chain (d=2, q=1) | 1·fwd | 1·fwd | 48 | [84, 98] (not proved) | [36, 148] (not proved) | 50 | 50 |
| micro local-sgd chain (d=2, q=1) | 1·fwd | 4·fwd | 36 | 68 | [36, 108] (not proved) | 32 | 40 |
| micro local-sgd chain (d=2, q=1) | 1·fwd | inf | 36 | 68 | [36, 108] (not proved) | 32 | 40 |
| micro local-sgd chain (d=2, q=1) | 1.5·fwd | 1·fwd | 48 | [84, 98] (not proved) | [36, 148] (not proved) | 50 | 50 |
| micro local-sgd chain (d=2, q=1) | 1.5·fwd | 4·fwd | 20 | 40 | [36, 60] (not proved) | 20 | 20 |
| micro local-sgd chain (d=2, q=1) | 1.5·fwd | inf | 20 | 40 | [36, 60] (not proved) | 20 | 20 |
| micro local-sgd chain (d=2, q=1) | 3·fwd | 1·fwd | 48 | [84, 98] (not proved) | [36, 148] (not proved) | 50 | 50 |
| micro local-sgd chain (d=2, q=1) | 3·fwd | 4·fwd | 20 | 40 | [36, 60] (not proved) | 20 | 20 |
| micro local-sgd chain (d=2, q=1) | 3·fwd | inf | 20 | 40 | [36, 56] (not proved) | 20 | 16 |
| micro local-sgd chain (d=2, q=1) | inf | 1·fwd | 48 | [84, 98] (not proved) | [36, 148] (not proved) | 50 | 50 |
| micro local-sgd chain (d=2, q=1) | inf | 4·fwd | 20 | 40 | [36, 60] (not proved) | 20 | 20 |
| micro local-sgd chain (d=2, q=1) | inf | inf | 20 | 28 | 36 | 8 | 8 |
| micro local-sgd chain (d=2, q=2) | 1·fwd | 1·fwd | 60 | [44, 136] (not proved) | - | 76 | - |
| micro local-sgd chain (d=2, q=2) | 1·fwd | 4·fwd | 42 | [44, 88] (not proved) | - | 46 | - |
| micro local-sgd chain (d=2, q=2) | 1·fwd | inf | 42 | [44, 88] (not proved) | - | 46 | - |
| micro local-sgd chain (d=2, q=2) | 1.5·fwd | 1·fwd | 60 | [44, 136] (not proved) | - | 76 | - |
| micro local-sgd chain (d=2, q=2) | 1.5·fwd | 4·fwd | 28 | [44, 56] (not proved) | - | 28 | - |
| micro local-sgd chain (d=2, q=2) | 1.5·fwd | inf | 28 | [44, 56] (not proved) | - | 28 | - |
| micro local-sgd chain (d=2, q=2) | 3·fwd | 1·fwd | 60 | [44, 136] (not proved) | - | 76 | - |
| micro local-sgd chain (d=2, q=2) | 3·fwd | 4·fwd | 28 | [44, 56] (not proved) | - | 28 | - |
| micro local-sgd chain (d=2, q=2) | 3·fwd | inf | 28 | 44 | - | 16 | - |
| micro local-sgd chain (d=2, q=2) | inf | 1·fwd | 60 | [44, 136] (not proved) | - | 76 | - |
| micro local-sgd chain (d=2, q=2) | inf | 4·fwd | 28 | [44, 56] (not proved) | - | 28 | - |
| micro local-sgd chain (d=2, q=2) | inf | inf | 28 | 44 | - | 16 | - |
| registry local-sgd @ tiny (q=2) | 1·fwd | 1·fwd | beyond exact scale (2888 gates) | beyond exact scale (5776 gates) | beyond exact scale (8664 gates) | - | - |
| registry local-sgd @ tiny (q=2) | 1·fwd | 4·fwd | beyond exact scale (2888 gates) | beyond exact scale (5776 gates) | beyond exact scale (8664 gates) | - | - |
| registry local-sgd @ tiny (q=2) | 1·fwd | inf | beyond exact scale (2888 gates) | beyond exact scale (5776 gates) | beyond exact scale (8664 gates) | - | - |
| registry local-sgd @ tiny (q=2) | 1.5·fwd | 1·fwd | beyond exact scale (2888 gates) | beyond exact scale (5776 gates) | beyond exact scale (8664 gates) | - | - |
| registry local-sgd @ tiny (q=2) | 1.5·fwd | 4·fwd | beyond exact scale (2888 gates) | beyond exact scale (5776 gates) | beyond exact scale (8664 gates) | - | - |
| registry local-sgd @ tiny (q=2) | 1.5·fwd | inf | beyond exact scale (2888 gates) | beyond exact scale (5776 gates) | beyond exact scale (8664 gates) | - | - |
| registry local-sgd @ tiny (q=2) | 3·fwd | 1·fwd | beyond exact scale (2888 gates) | beyond exact scale (5776 gates) | beyond exact scale (8664 gates) | - | - |
| registry local-sgd @ tiny (q=2) | 3·fwd | 4·fwd | beyond exact scale (2888 gates) | beyond exact scale (5776 gates) | beyond exact scale (8664 gates) | - | - |
| registry local-sgd @ tiny (q=2) | 3·fwd | inf | 396 | beyond exact scale (5776 gates) | beyond exact scale (8664 gates) | - | - |
| registry local-sgd @ tiny (q=2) | inf | 1·fwd | beyond exact scale (2888 gates) | beyond exact scale (5776 gates) | beyond exact scale (8664 gates) | - | - |
| registry local-sgd @ tiny (q=2) | inf | 4·fwd | beyond exact scale (2888 gates) | beyond exact scale (5776 gates) | beyond exact scale (8664 gates) | - | - |
| registry local-sgd @ tiny (q=2) | inf | inf | 396 | 412 | 428 | 16 | 16 |
| registry local-sgd @ tiny2 (q=2) | 1·fwd | 1·fwd | beyond exact scale (6272 gates) | beyond exact scale (12544 gates) | beyond exact scale (18816 gates) | - | - |
| registry local-sgd @ tiny2 (q=2) | 1·fwd | 4·fwd | beyond exact scale (6272 gates) | beyond exact scale (12544 gates) | beyond exact scale (18816 gates) | - | - |
| registry local-sgd @ tiny2 (q=2) | 1·fwd | inf | beyond exact scale (6272 gates) | beyond exact scale (12544 gates) | beyond exact scale (18816 gates) | - | - |
| registry local-sgd @ tiny2 (q=2) | 1.5·fwd | 1·fwd | beyond exact scale (6272 gates) | beyond exact scale (12544 gates) | beyond exact scale (18816 gates) | - | - |
| registry local-sgd @ tiny2 (q=2) | 1.5·fwd | 4·fwd | beyond exact scale (6272 gates) | beyond exact scale (12544 gates) | beyond exact scale (18816 gates) | - | - |
| registry local-sgd @ tiny2 (q=2) | 1.5·fwd | inf | beyond exact scale (6272 gates) | beyond exact scale (12544 gates) | beyond exact scale (18816 gates) | - | - |
| registry local-sgd @ tiny2 (q=2) | 3·fwd | 1·fwd | beyond exact scale (6272 gates) | beyond exact scale (12544 gates) | beyond exact scale (18816 gates) | - | - |
| registry local-sgd @ tiny2 (q=2) | 3·fwd | 4·fwd | beyond exact scale (6272 gates) | beyond exact scale (12544 gates) | beyond exact scale (18816 gates) | - | - |
| registry local-sgd @ tiny2 (q=2) | 3·fwd | inf | 764 | beyond exact scale (12544 gates) | beyond exact scale (18816 gates) | - | - |
| registry local-sgd @ tiny2 (q=2) | inf | 1·fwd | beyond exact scale (6272 gates) | beyond exact scale (12544 gates) | beyond exact scale (18816 gates) | - | - |
| registry local-sgd @ tiny2 (q=2) | inf | 4·fwd | beyond exact scale (6272 gates) | beyond exact scale (12544 gates) | beyond exact scale (18816 gates) | - | - |
| registry local-sgd @ tiny2 (q=2) | inf | inf | 764 | 780 | 796 | 16 | 16 |

## Optimal partitions (micro circuits)

**local_sgd_chain[K=1,q=1,d=1]** at F=1·fwd (2), G=1·fwd (2): I* = 20, 4 RUs
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B

**local_sgd_chain[K=1,q=1,d=1]** at F=1·fwd (2), G=4·fwd (8): I* = 20, 4 RUs
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B

**local_sgd_chain[K=1,q=1,d=1]** at F=1·fwd (2), G=inf (-): I* = 20, 4 RUs
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B

**local_sgd_chain[K=1,q=1,d=1]** at F=1.5·fwd (3), G=1·fwd (2): I* = 20, 4 RUs
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B

**local_sgd_chain[K=1,q=1,d=1]** at F=1.5·fwd (3), G=4·fwd (8): I* = 18, 3 RUs
- RU0: work 3, input 6 B; ops [op0:matmul x3, op1:add x1]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 3, input 8 B; ops [op2:matmul x3, op3:sgdupdate x1]; imports lr[seed]=4B, op1:add.out=2B, x0[token]=2B
- RU2: work 1, input 4 B; ops [op3:sgdupdate x1]; imports W[accumulated]=2B, op3:sgdupdate.out=2B

**local_sgd_chain[K=1,q=1,d=1]** at F=1.5·fwd (3), G=inf (-): I* = 18, 3 RUs
- RU0: work 3, input 6 B; ops [op0:matmul x3, op1:add x1]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 3, input 8 B; ops [op2:matmul x3, op3:sgdupdate x1]; imports lr[seed]=4B, op1:add.out=2B, x0[token]=2B
- RU2: work 1, input 4 B; ops [op3:sgdupdate x1]; imports W[accumulated]=2B, op3:sgdupdate.out=2B

**local_sgd_chain[K=1,q=1,d=1]** at F=3·fwd (6), G=1·fwd (2): I* = 20, 4 RUs
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B

**local_sgd_chain[K=1,q=1,d=1]** at F=3·fwd (6), G=4·fwd (8): I* = 14, 2 RUs
- RU0: work 6, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x3, op3:sgdupdate x1]; imports W[accumulated]=2B, lr[seed]=4B, t0[token]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op3:sgdupdate x1]; imports W[accumulated]=2B, op3:sgdupdate.out=2B

**local_sgd_chain[K=1,q=1,d=1]** at F=3·fwd (6), G=inf (-): I* = 14, 2 RUs
- RU0: work 6, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x3, op3:sgdupdate x1]; imports W[accumulated]=2B, lr[seed]=4B, t0[token]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op3:sgdupdate x1]; imports W[accumulated]=2B, op3:sgdupdate.out=2B

**local_sgd_chain[K=1,q=1,d=1]** at F=inf (-), G=1·fwd (2): I* = 20, 4 RUs
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B

**local_sgd_chain[K=2,q=1,d=1]** at F=1·fwd (2), G=1·fwd (2): I* = 40, 8 RUs
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B
- RU4: work 2, input 4 B; ops [op4:matmul x3]; imports op3:sgdupdate.out=2B, x1[token]=2B
- RU5: work 1, input 4 B; ops [op5:add x1]; imports op4:matmul.out=2B, t1[token]=2B
- RU6: work 2, input 4 B; ops [op6:matmul x3]; imports op5:add.out=2B, x1[token]=2B
- RU7: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B

**local_sgd_chain[K=2,q=1,d=1]** at F=1·fwd (2), G=4·fwd (8): I* = 40, 8 RUs
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B
- RU4: work 2, input 4 B; ops [op4:matmul x3]; imports op3:sgdupdate.out=2B, x1[token]=2B
- RU5: work 1, input 4 B; ops [op5:add x1]; imports op4:matmul.out=2B, t1[token]=2B
- RU6: work 2, input 4 B; ops [op6:matmul x3]; imports op5:add.out=2B, x1[token]=2B
- RU7: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B

**local_sgd_chain[K=2,q=1,d=1]** at F=1·fwd (2), G=inf (-): I* = 40, 8 RUs
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B
- RU4: work 2, input 4 B; ops [op4:matmul x3]; imports op3:sgdupdate.out=2B, x1[token]=2B
- RU5: work 1, input 4 B; ops [op5:add x1]; imports op4:matmul.out=2B, t1[token]=2B
- RU6: work 2, input 4 B; ops [op6:matmul x3]; imports op5:add.out=2B, x1[token]=2B
- RU7: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B

**local_sgd_chain[K=2,q=1,d=1]** at F=1.5·fwd (3), G=1·fwd (2): I* = 40, 8 RUs
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B
- RU4: work 2, input 4 B; ops [op4:matmul x3]; imports op3:sgdupdate.out=2B, x1[token]=2B
- RU5: work 1, input 4 B; ops [op5:add x1]; imports op4:matmul.out=2B, t1[token]=2B
- RU6: work 2, input 4 B; ops [op6:matmul x3]; imports op5:add.out=2B, x1[token]=2B
- RU7: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B

**local_sgd_chain[K=2,q=1,d=1]** at F=1.5·fwd (3), G=4·fwd (8): I* = 34, 5 RUs
- RU0: work 3, input 6 B; ops [op0:matmul x3, op1:add x1]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 3, input 8 B; ops [op2:matmul x3, op3:sgdupdate x1]; imports lr[seed]=4B, op1:add.out=2B, x0[token]=2B
- RU2: work 3, input 6 B; ops [op3:sgdupdate x1, op4:matmul x3]; imports W[accumulated]=2B, op3:sgdupdate.out=2B, x1[token]=2B
- RU3: work 3, input 6 B; ops [op5:add x1, op6:matmul x3]; imports op4:matmul.out=2B, t1[token]=2B, x1[token]=2B
- RU4: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B

**local_sgd_chain[K=2,q=1,d=1]** at F=1.5·fwd (3), G=inf (-): I* = 34, 5 RUs
- RU0: work 3, input 6 B; ops [op0:matmul x3, op1:add x1]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 3, input 8 B; ops [op2:matmul x3, op3:sgdupdate x1]; imports lr[seed]=4B, op1:add.out=2B, x0[token]=2B
- RU2: work 3, input 6 B; ops [op3:sgdupdate x1, op4:matmul x3]; imports W[accumulated]=2B, op3:sgdupdate.out=2B, x1[token]=2B
- RU3: work 3, input 6 B; ops [op5:add x1, op6:matmul x3]; imports op4:matmul.out=2B, t1[token]=2B, x1[token]=2B
- RU4: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B

**local_sgd_chain[K=2,q=1,d=1]** at F=3·fwd (6), G=1·fwd (2): I* = 40, 8 RUs
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B
- RU4: work 2, input 4 B; ops [op4:matmul x3]; imports op3:sgdupdate.out=2B, x1[token]=2B
- RU5: work 1, input 4 B; ops [op5:add x1]; imports op4:matmul.out=2B, t1[token]=2B
- RU6: work 2, input 4 B; ops [op6:matmul x3]; imports op5:add.out=2B, x1[token]=2B
- RU7: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B

**local_sgd_chain[K=2,q=1,d=1]** at F=3·fwd (6), G=4·fwd (8): I* = 26, 3 RUs
- RU0: work 6, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x3, op3:sgdupdate x1]; imports W[accumulated]=2B, lr[seed]=4B, t0[token]=2B, x0[token]=2B
- RU1: work 6, input 8 B; ops [op3:sgdupdate x1, op4:matmul x3, op5:add x1, op6:matmul x3]; imports W[accumulated]=2B, op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU2: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B

**local_sgd_chain[K=2,q=1,d=1]** at F=3·fwd (6), G=inf (-): I* = 26, 3 RUs
- RU0: work 6, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x3, op3:sgdupdate x1]; imports W[accumulated]=2B, lr[seed]=4B, t0[token]=2B, x0[token]=2B
- RU1: work 6, input 8 B; ops [op3:sgdupdate x1, op4:matmul x3, op5:add x1, op6:matmul x3]; imports W[accumulated]=2B, op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU2: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B

**local_sgd_chain[K=2,q=1,d=1]** at F=inf (-), G=1·fwd (2): I* = 40, 8 RUs
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B
- RU4: work 2, input 4 B; ops [op4:matmul x3]; imports op3:sgdupdate.out=2B, x1[token]=2B
- RU5: work 1, input 4 B; ops [op5:add x1]; imports op4:matmul.out=2B, t1[token]=2B
- RU6: work 2, input 4 B; ops [op6:matmul x3]; imports op5:add.out=2B, x1[token]=2B
- RU7: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B

**local_sgd_chain[K=2,q=1,d=1]** at F=inf (-), G=4·fwd (8): I* = 20, 2 RUs
- RU0: work 7, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x3, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, t0[token]=2B, x0[token]=2B
- RU1: work 7, input 10 B; ops [op4:matmul x3, op5:add x1, op6:matmul x3, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B

**local_sgd_chain[K=3,q=1,d=1]** at F=1·fwd (2), G=1·fwd (2): I* = 60, 12 RUs
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B
- RU4: work 2, input 4 B; ops [op4:matmul x3]; imports op3:sgdupdate.out=2B, x1[token]=2B
- RU5: work 1, input 4 B; ops [op5:add x1]; imports op4:matmul.out=2B, t1[token]=2B
- RU6: work 2, input 4 B; ops [op6:matmul x3]; imports op5:add.out=2B, x1[token]=2B
- RU7: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B
- RU8: work 2, input 4 B; ops [op8:matmul x3]; imports op7:sgdupdate.out=2B, x2[token]=2B
- RU9: work 1, input 4 B; ops [op9:add x1]; imports op8:matmul.out=2B, t2[token]=2B
- RU10: work 2, input 4 B; ops [op10:matmul x3]; imports op9:add.out=2B, x2[token]=2B
- RU11: work 2, input 8 B; ops [op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=2B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=1,d=1]** at F=1·fwd (2), G=4·fwd (8): I* = 60, 12 RUs
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B
- RU4: work 2, input 4 B; ops [op4:matmul x3]; imports op3:sgdupdate.out=2B, x1[token]=2B
- RU5: work 1, input 4 B; ops [op5:add x1]; imports op4:matmul.out=2B, t1[token]=2B
- RU6: work 2, input 4 B; ops [op6:matmul x3]; imports op5:add.out=2B, x1[token]=2B
- RU7: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B
- RU8: work 2, input 4 B; ops [op8:matmul x3]; imports op7:sgdupdate.out=2B, x2[token]=2B
- RU9: work 1, input 4 B; ops [op9:add x1]; imports op8:matmul.out=2B, t2[token]=2B
- RU10: work 2, input 4 B; ops [op10:matmul x3]; imports op9:add.out=2B, x2[token]=2B
- RU11: work 2, input 8 B; ops [op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=2B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=1,d=1]** at F=1·fwd (2), G=inf (-): I* = 60, 12 RUs
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B
- RU4: work 2, input 4 B; ops [op4:matmul x3]; imports op3:sgdupdate.out=2B, x1[token]=2B
- RU5: work 1, input 4 B; ops [op5:add x1]; imports op4:matmul.out=2B, t1[token]=2B
- RU6: work 2, input 4 B; ops [op6:matmul x3]; imports op5:add.out=2B, x1[token]=2B
- RU7: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B
- RU8: work 2, input 4 B; ops [op8:matmul x3]; imports op7:sgdupdate.out=2B, x2[token]=2B
- RU9: work 1, input 4 B; ops [op9:add x1]; imports op8:matmul.out=2B, t2[token]=2B
- RU10: work 2, input 4 B; ops [op10:matmul x3]; imports op9:add.out=2B, x2[token]=2B
- RU11: work 2, input 8 B; ops [op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=2B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=1,d=1]** at F=1.5·fwd (3), G=1·fwd (2): I* = 60, 12 RUs
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B
- RU4: work 2, input 4 B; ops [op4:matmul x3]; imports op3:sgdupdate.out=2B, x1[token]=2B
- RU5: work 1, input 4 B; ops [op5:add x1]; imports op4:matmul.out=2B, t1[token]=2B
- RU6: work 2, input 4 B; ops [op6:matmul x3]; imports op5:add.out=2B, x1[token]=2B
- RU7: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B
- RU8: work 2, input 4 B; ops [op8:matmul x3]; imports op7:sgdupdate.out=2B, x2[token]=2B
- RU9: work 1, input 4 B; ops [op9:add x1]; imports op8:matmul.out=2B, t2[token]=2B
- RU10: work 2, input 4 B; ops [op10:matmul x3]; imports op9:add.out=2B, x2[token]=2B
- RU11: work 2, input 8 B; ops [op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=2B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=1,d=1]** at F=1.5·fwd (3), G=4·fwd (8): I* = 52, 8 RUs
- RU0: work 3, input 6 B; ops [op0:matmul x3, op1:add x1]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 3, input 8 B; ops [op2:matmul x3, op3:sgdupdate x1]; imports lr[seed]=4B, op1:add.out=2B, x0[token]=2B
- RU2: work 3, input 6 B; ops [op3:sgdupdate x1, op4:matmul x3]; imports W[accumulated]=2B, op3:sgdupdate.out=2B, x1[token]=2B
- RU3: work 3, input 6 B; ops [op5:add x1, op6:matmul x3]; imports op4:matmul.out=2B, t1[token]=2B, x1[token]=2B
- RU4: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B
- RU5: work 3, input 6 B; ops [op8:matmul x3, op9:add x1]; imports op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU6: work 3, input 8 B; ops [op10:matmul x3, op11:sgdupdate x1]; imports lr[seed]=4B, op9:add.out=2B, x2[token]=2B
- RU7: work 1, input 4 B; ops [op11:sgdupdate x1]; imports op11:sgdupdate.out=2B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=1,d=1]** at F=1.5·fwd (3), G=inf (-): I* = 52, 8 RUs
- RU0: work 3, input 6 B; ops [op0:matmul x3, op1:add x1]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 3, input 8 B; ops [op2:matmul x3, op3:sgdupdate x1]; imports lr[seed]=4B, op1:add.out=2B, x0[token]=2B
- RU2: work 3, input 6 B; ops [op3:sgdupdate x1, op4:matmul x3]; imports W[accumulated]=2B, op3:sgdupdate.out=2B, x1[token]=2B
- RU3: work 3, input 6 B; ops [op5:add x1, op6:matmul x3]; imports op4:matmul.out=2B, t1[token]=2B, x1[token]=2B
- RU4: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B
- RU5: work 3, input 6 B; ops [op8:matmul x3, op9:add x1]; imports op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU6: work 3, input 8 B; ops [op10:matmul x3, op11:sgdupdate x1]; imports lr[seed]=4B, op9:add.out=2B, x2[token]=2B
- RU7: work 1, input 4 B; ops [op11:sgdupdate x1]; imports op11:sgdupdate.out=2B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=1,d=1]** at F=3·fwd (6), G=1·fwd (2): I* = 60, 12 RUs
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B
- RU4: work 2, input 4 B; ops [op4:matmul x3]; imports op3:sgdupdate.out=2B, x1[token]=2B
- RU5: work 1, input 4 B; ops [op5:add x1]; imports op4:matmul.out=2B, t1[token]=2B
- RU6: work 2, input 4 B; ops [op6:matmul x3]; imports op5:add.out=2B, x1[token]=2B
- RU7: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B
- RU8: work 2, input 4 B; ops [op8:matmul x3]; imports op7:sgdupdate.out=2B, x2[token]=2B
- RU9: work 1, input 4 B; ops [op9:add x1]; imports op8:matmul.out=2B, t2[token]=2B
- RU10: work 2, input 4 B; ops [op10:matmul x3]; imports op9:add.out=2B, x2[token]=2B
- RU11: work 2, input 8 B; ops [op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=2B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=1,d=1]** at F=3·fwd (6), G=4·fwd (8): I* = 40, 4 RUs
- RU0: work 6, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x3, op3:sgdupdate x1]; imports W[accumulated]=2B, lr[seed]=4B, t0[token]=2B, x0[token]=2B
- RU1: work 6, input 8 B; ops [op3:sgdupdate x1, op4:matmul x3, op5:add x1, op6:matmul x3]; imports W[accumulated]=2B, op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU2: work 6, input 12 B; ops [op7:sgdupdate x2, op8:matmul x3, op9:add x1, op10:matmul x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B, t2[token]=2B, x2[token]=2B
- RU3: work 3, input 10 B; ops [op10:matmul x1, op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=4B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=1,d=1]** at F=3·fwd (6), G=inf (-): I* = 40, 4 RUs
- RU0: work 6, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x3, op3:sgdupdate x1]; imports W[accumulated]=2B, lr[seed]=4B, t0[token]=2B, x0[token]=2B
- RU1: work 6, input 8 B; ops [op3:sgdupdate x1, op4:matmul x3, op5:add x1, op6:matmul x3]; imports W[accumulated]=2B, op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU2: work 6, input 12 B; ops [op7:sgdupdate x2, op8:matmul x3, op9:add x1, op10:matmul x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B, t2[token]=2B, x2[token]=2B
- RU3: work 3, input 10 B; ops [op10:matmul x1, op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=4B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=1,d=1]** at F=inf (-), G=1·fwd (2): I* = 60, 12 RUs
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B
- RU4: work 2, input 4 B; ops [op4:matmul x3]; imports op3:sgdupdate.out=2B, x1[token]=2B
- RU5: work 1, input 4 B; ops [op5:add x1]; imports op4:matmul.out=2B, t1[token]=2B
- RU6: work 2, input 4 B; ops [op6:matmul x3]; imports op5:add.out=2B, x1[token]=2B
- RU7: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B
- RU8: work 2, input 4 B; ops [op8:matmul x3]; imports op7:sgdupdate.out=2B, x2[token]=2B
- RU9: work 1, input 4 B; ops [op9:add x1]; imports op8:matmul.out=2B, t2[token]=2B
- RU10: work 2, input 4 B; ops [op10:matmul x3]; imports op9:add.out=2B, x2[token]=2B
- RU11: work 2, input 8 B; ops [op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=2B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=1,d=1]** at F=inf (-), G=4·fwd (8): I* = 30, 3 RUs
- RU0: work 7, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x3, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, t0[token]=2B, x0[token]=2B
- RU1: work 7, input 10 B; ops [op4:matmul x3, op5:add x1, op6:matmul x3, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU2: work 7, input 10 B; ops [op8:matmul x3, op9:add x1, op10:matmul x3, op11:sgdupdate x2]; imports lr[seed]=4B, op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B

**local_sgd_chain[K=1,q=2,d=1]** at F=1·fwd (4), G=1·fwd (4): I* = 26, 3 RUs
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 4, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x1]; imports W[accumulated]=2B, op2:matmul.out=4B, t0[token]=2B, x0[token]=2B
- RU2: work 3, input 10 B; ops [op2:matmul x1, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=4B

**local_sgd_chain[K=1,q=2,d=1]** at F=1·fwd (4), G=4·fwd (16): I* = 24, 2 RUs
- RU0: work 7, input 10 B; ops [op0:matmul x6, op1:add x2, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=4B, x0[token]=4B
- RU1: work 4, input 14 B; ops [op2:matmul x2, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=2B, op2:matmul.out=4B, x0[token]=2B

**local_sgd_chain[K=1,q=2,d=1]** at F=1·fwd (4), G=inf (-): I* = 24, 2 RUs
- RU0: work 7, input 10 B; ops [op0:matmul x6, op1:add x2, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=4B, x0[token]=4B
- RU1: work 4, input 14 B; ops [op2:matmul x2, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=2B, op2:matmul.out=4B, x0[token]=2B

**local_sgd_chain[K=1,q=2,d=1]** at F=1.5·fwd (6), G=1·fwd (4): I* = 26, 3 RUs
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 4, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x1]; imports W[accumulated]=2B, op2:matmul.out=4B, t0[token]=2B, x0[token]=2B
- RU2: work 3, input 10 B; ops [op2:matmul x1, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=4B

**local_sgd_chain[K=1,q=2,d=1]** at F=1.5·fwd (6), G=4·fwd (16): I* = 24, 2 RUs
- RU0: work 7, input 10 B; ops [op0:matmul x6, op1:add x2, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=4B, x0[token]=4B
- RU1: work 4, input 14 B; ops [op2:matmul x2, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=2B, op2:matmul.out=4B, x0[token]=2B

**local_sgd_chain[K=1,q=2,d=1]** at F=1.5·fwd (6), G=inf (-): I* = 24, 2 RUs
- RU0: work 7, input 10 B; ops [op0:matmul x6, op1:add x2, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=4B, x0[token]=4B
- RU1: work 4, input 14 B; ops [op2:matmul x2, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=2B, op2:matmul.out=4B, x0[token]=2B

**local_sgd_chain[K=1,q=2,d=1]** at F=3·fwd (12), G=1·fwd (4): I* = 26, 3 RUs
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 4, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x1]; imports W[accumulated]=2B, op2:matmul.out=4B, t0[token]=2B, x0[token]=2B
- RU2: work 3, input 10 B; ops [op2:matmul x1, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=4B

**local_sgd_chain[K=1,q=2,d=1]** at F=inf (-), G=1·fwd (4): I* = 26, 3 RUs
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 4, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x1]; imports W[accumulated]=2B, op2:matmul.out=4B, t0[token]=2B, x0[token]=2B
- RU2: work 3, input 10 B; ops [op2:matmul x1, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=4B

**local_sgd_chain[K=2,q=2,d=1]** at F=1·fwd (4), G=1·fwd (4): I* = 52, 6 RUs
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 4, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x1]; imports W[accumulated]=2B, op2:matmul.out=4B, t0[token]=2B, x0[token]=2B
- RU2: work 3, input 10 B; ops [op2:matmul x1, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=4B
- RU3: work 4, input 6 B; ops [op4:matmul x3, op5:add x1, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU4: work 4, input 10 B; ops [op4:matmul x3, op5:add x1, op6:matmul x1]; imports op3:sgdupdate.out=2B, op6:matmul.out=4B, t1[token]=2B, x1[token]=2B
- RU5: work 3, input 10 B; ops [op6:matmul x1, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=4B

**local_sgd_chain[K=2,q=2,d=1]** at F=1·fwd (4), G=4·fwd (16): I* = 48, 4 RUs
- RU0: work 7, input 10 B; ops [op0:matmul x6, op1:add x2, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=4B, x0[token]=4B
- RU1: work 4, input 14 B; ops [op2:matmul x2, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=2B, op2:matmul.out=4B, x0[token]=2B
- RU2: work 7, input 10 B; ops [op4:matmul x6, op5:add x2, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=4B, x1[token]=4B
- RU3: work 4, input 14 B; ops [op6:matmul x2, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op5:add.out=2B, op6:matmul.out=4B, x1[token]=2B

**local_sgd_chain[K=2,q=2,d=1]** at F=1·fwd (4), G=inf (-): I* = 48, 4 RUs
- RU0: work 7, input 10 B; ops [op0:matmul x6, op1:add x2, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=4B, x0[token]=4B
- RU1: work 4, input 14 B; ops [op2:matmul x2, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=2B, op2:matmul.out=4B, x0[token]=2B
- RU2: work 7, input 10 B; ops [op4:matmul x6, op5:add x2, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=4B, x1[token]=4B
- RU3: work 4, input 14 B; ops [op6:matmul x2, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op5:add.out=2B, op6:matmul.out=4B, x1[token]=2B

**local_sgd_chain[K=2,q=2,d=1]** at F=1.5·fwd (6), G=1·fwd (4): I* = 52, 6 RUs
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 4, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x1]; imports W[accumulated]=2B, op2:matmul.out=4B, t0[token]=2B, x0[token]=2B
- RU2: work 3, input 10 B; ops [op2:matmul x1, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=4B
- RU3: work 4, input 6 B; ops [op4:matmul x3, op5:add x1, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU4: work 4, input 10 B; ops [op4:matmul x3, op5:add x1, op6:matmul x1]; imports op3:sgdupdate.out=2B, op6:matmul.out=4B, t1[token]=2B, x1[token]=2B
- RU5: work 3, input 10 B; ops [op6:matmul x1, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=4B

**local_sgd_chain[K=2,q=2,d=1]** at F=1.5·fwd (6), G=4·fwd (16): I* = 46, 4 RUs
- RU0: work 6, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x4]; imports W[accumulated]=2B, op1:add.out=2B, t0[token]=2B, x0[token]=4B
- RU1: work 3, input 6 B; ops [op0:matmul x3, op1:add x1]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU2: work 9, input 16 B; ops [op3:sgdupdate x2, op4:matmul x6, op5:add x2, op6:matmul x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B, t1[token]=4B, x1[token]=4B
- RU3: work 4, input 14 B; ops [op6:matmul x2, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op5:add.out=2B, op6:matmul.out=4B, x1[token]=2B

**local_sgd_chain[K=2,q=2,d=1]** at F=1.5·fwd (6), G=inf (-): I* = 46, 4 RUs
- RU0: work 6, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x4]; imports W[accumulated]=2B, op1:add.out=2B, t0[token]=2B, x0[token]=4B
- RU1: work 3, input 6 B; ops [op0:matmul x3, op1:add x1]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU2: work 9, input 16 B; ops [op3:sgdupdate x2, op4:matmul x6, op5:add x2, op6:matmul x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B, t1[token]=4B, x1[token]=4B
- RU3: work 4, input 14 B; ops [op6:matmul x2, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op5:add.out=2B, op6:matmul.out=4B, x1[token]=2B

**local_sgd_chain[K=2,q=2,d=1]** at F=3·fwd (12), G=1·fwd (4): I* = 52, 6 RUs
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 4, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x1]; imports W[accumulated]=2B, op2:matmul.out=4B, t0[token]=2B, x0[token]=2B
- RU2: work 3, input 10 B; ops [op2:matmul x1, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=4B
- RU3: work 4, input 6 B; ops [op4:matmul x3, op5:add x1, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU4: work 4, input 10 B; ops [op4:matmul x3, op5:add x1, op6:matmul x1]; imports op3:sgdupdate.out=2B, op6:matmul.out=4B, t1[token]=2B, x1[token]=2B
- RU5: work 3, input 10 B; ops [op6:matmul x1, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=4B

**local_sgd_chain[K=2,q=2,d=1]** at F=3·fwd (12), G=4·fwd (16): I* = 28, 2 RUs
- RU0: work 11, input 14 B; ops [op0:matmul x6, op1:add x2, op2:matmul x4, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, t0[token]=4B, x0[token]=4B
- RU1: work 11, input 14 B; ops [op4:matmul x6, op5:add x2, op6:matmul x4, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, t1[token]=4B, x1[token]=4B

**local_sgd_chain[K=2,q=2,d=1]** at F=3·fwd (12), G=inf (-): I* = 28, 2 RUs
- RU0: work 11, input 14 B; ops [op0:matmul x6, op1:add x2, op2:matmul x4, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, t0[token]=4B, x0[token]=4B
- RU1: work 11, input 14 B; ops [op4:matmul x6, op5:add x2, op6:matmul x4, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, t1[token]=4B, x1[token]=4B

**local_sgd_chain[K=2,q=2,d=1]** at F=inf (-), G=1·fwd (4): I* = 52, 6 RUs
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 4, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x1]; imports W[accumulated]=2B, op2:matmul.out=4B, t0[token]=2B, x0[token]=2B
- RU2: work 3, input 10 B; ops [op2:matmul x1, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=4B
- RU3: work 4, input 6 B; ops [op4:matmul x3, op5:add x1, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU4: work 4, input 10 B; ops [op4:matmul x3, op5:add x1, op6:matmul x1]; imports op3:sgdupdate.out=2B, op6:matmul.out=4B, t1[token]=2B, x1[token]=2B
- RU5: work 3, input 10 B; ops [op6:matmul x1, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=4B

**local_sgd_chain[K=2,q=2,d=1]** at F=inf (-), G=4·fwd (16): I* = 26, 2 RUs
- RU0: work 9, input 10 B; ops [op0:matmul x6, op1:add x2, op2:matmul x4]; imports W[accumulated]=2B, t0[token]=4B, x0[token]=4B
- RU1: work 13, input 16 B; ops [op3:sgdupdate x2, op4:matmul x6, op5:add x2, op6:matmul x4, op7:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B, t1[token]=4B, x1[token]=4B

**local_sgd_chain[K=3,q=2,d=1]** at F=1·fwd (4), G=1·fwd (4): I* = 78, 9 RUs
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 3, input 6 B; ops [op0:matmul x3, op1:add x1]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU2: work 4, input 14 B; ops [op2:matmul x2, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=2B, op2:matmul.out=4B, x0[token]=2B
- RU3: work 4, input 6 B; ops [op4:matmul x3, op5:add x1, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU4: work 3, input 6 B; ops [op4:matmul x3, op5:add x1]; imports op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU5: work 4, input 14 B; ops [op6:matmul x2, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op5:add.out=2B, op6:matmul.out=4B, x1[token]=2B
- RU6: work 4, input 6 B; ops [op8:matmul x3, op9:add x1, op10:matmul x2]; imports op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU7: work 3, input 6 B; ops [op8:matmul x3, op9:add x1]; imports op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU8: work 4, input 14 B; ops [op10:matmul x2, op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=4B, op7:sgdupdate.out=2B, op9:add.out=2B, x2[token]=2B

**local_sgd_chain[K=3,q=2,d=1]** at F=1·fwd (4), G=4·fwd (16): I* = 72, 6 RUs
- RU0: work 7, input 10 B; ops [op0:matmul x6, op1:add x2, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=4B, x0[token]=4B
- RU1: work 4, input 14 B; ops [op2:matmul x2, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=2B, op2:matmul.out=4B, x0[token]=2B
- RU2: work 7, input 10 B; ops [op4:matmul x6, op5:add x2, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=4B, x1[token]=4B
- RU3: work 4, input 14 B; ops [op6:matmul x2, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op5:add.out=2B, op6:matmul.out=4B, x1[token]=2B
- RU4: work 7, input 10 B; ops [op8:matmul x6, op9:add x2, op10:matmul x2]; imports op7:sgdupdate.out=2B, t2[token]=4B, x2[token]=4B
- RU5: work 4, input 14 B; ops [op10:matmul x2, op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=4B, op7:sgdupdate.out=2B, op9:add.out=2B, x2[token]=2B

**local_sgd_chain[K=3,q=2,d=1]** at F=1·fwd (4), G=inf (-): I* = 72, 6 RUs
- RU0: work 7, input 10 B; ops [op0:matmul x6, op1:add x2, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=4B, x0[token]=4B
- RU1: work 4, input 14 B; ops [op2:matmul x2, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=2B, op2:matmul.out=4B, x0[token]=2B
- RU2: work 7, input 10 B; ops [op4:matmul x6, op5:add x2, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=4B, x1[token]=4B
- RU3: work 4, input 14 B; ops [op6:matmul x2, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op5:add.out=2B, op6:matmul.out=4B, x1[token]=2B
- RU4: work 7, input 10 B; ops [op8:matmul x6, op9:add x2, op10:matmul x2]; imports op7:sgdupdate.out=2B, t2[token]=4B, x2[token]=4B
- RU5: work 4, input 14 B; ops [op10:matmul x2, op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=4B, op7:sgdupdate.out=2B, op9:add.out=2B, x2[token]=2B

**local_sgd_chain[K=3,q=2,d=1]** at F=1.5·fwd (6), G=1·fwd (4): I* = 78, 9 RUs
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 3, input 6 B; ops [op0:matmul x3, op1:add x1]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU2: work 2, input 8 B; ops [op2:matmul x2]; imports op1:add.out=2B, op2:matmul.out=4B, x0[token]=2B
- RU3: work 4, input 10 B; ops [op3:sgdupdate x2, op4:matmul x3]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B, x1[token]=2B
- RU4: work 4, input 6 B; ops [op4:matmul x3, op5:add x1, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU5: work 4, input 14 B; ops [op5:add x1, op6:matmul x2, op7:sgdupdate x1]; imports lr[seed]=4B, op4:matmul.out=2B, op6:matmul.out=4B, t1[token]=2B, x1[token]=2B
- RU6: work 4, input 8 B; ops [op7:sgdupdate x1, op8:matmul x3, op9:add x1]; imports op3:sgdupdate.out=2B, op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU7: work 4, input 6 B; ops [op8:matmul x3, op9:add x1, op10:matmul x2]; imports op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU8: work 4, input 14 B; ops [op10:matmul x2, op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=4B, op7:sgdupdate.out=2B, op9:add.out=2B, x2[token]=2B

**local_sgd_chain[K=3,q=2,d=1]** at F=1.5·fwd (6), G=4·fwd (16): I* = 68, 7 RUs
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 5, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, op2:matmul.out=4B, t0[token]=2B, x0[token]=2B
- RU2: work 5, input 12 B; ops [op3:sgdupdate x2, op4:matmul x3, op5:add x1]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B, t1[token]=2B, x1[token]=2B
- RU3: work 6, input 10 B; ops [op4:matmul x3, op5:add x1, op6:matmul x4]; imports op3:sgdupdate.out=2B, op5:add.out=2B, t1[token]=2B, x1[token]=4B
- RU4: work 6, input 12 B; ops [op7:sgdupdate x2, op8:matmul x3, op9:add x1, op10:matmul x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B, t2[token]=2B, x2[token]=2B
- RU5: work 5, input 10 B; ops [op8:matmul x3, op9:add x1, op10:matmul x2]; imports op10:matmul.out=4B, op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU6: work 2, input 8 B; ops [op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=2B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=2,d=1]** at F=1.5·fwd (6), G=inf (-): I* = 68, 6 RUs
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 5, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, op2:matmul.out=4B, t0[token]=2B, x0[token]=2B
- RU2: work 5, input 12 B; ops [op3:sgdupdate x2, op4:matmul x3, op5:add x1]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B, t1[token]=2B, x1[token]=2B
- RU3: work 6, input 10 B; ops [op4:matmul x3, op5:add x1, op6:matmul x4]; imports op3:sgdupdate.out=2B, op5:add.out=2B, t1[token]=2B, x1[token]=4B
- RU4: work 7, input 14 B; ops [op7:sgdupdate x2, op8:matmul x6, op9:add x1]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B, t2[token]=2B, x2[token]=4B
- RU5: work 6, input 16 B; ops [op9:add x1, op10:matmul x4, op11:sgdupdate x2]; imports lr[seed]=4B, op7:sgdupdate.out=2B, op8:matmul.out=2B, op9:add.out=2B, t2[token]=2B, x2[token]=4B

**local_sgd_chain[K=3,q=2,d=1]** at F=3·fwd (12), G=1·fwd (4): I* = 78, 9 RUs
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 4, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x1]; imports W[accumulated]=2B, op2:matmul.out=4B, t0[token]=2B, x0[token]=2B
- RU2: work 3, input 10 B; ops [op2:matmul x1, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=4B
- RU3: work 4, input 6 B; ops [op4:matmul x3, op5:add x1, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU4: work 3, input 6 B; ops [op4:matmul x3, op5:add x1]; imports op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU5: work 3, input 12 B; ops [op6:matmul x2, op7:sgdupdate x1]; imports lr[seed]=4B, op5:add.out=2B, op6:matmul.out=4B, x1[token]=2B
- RU6: work 4, input 8 B; ops [op7:sgdupdate x1, op8:matmul x3, op9:add x1]; imports op3:sgdupdate.out=2B, op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU7: work 4, input 6 B; ops [op8:matmul x3, op9:add x1, op10:matmul x2]; imports op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU8: work 4, input 14 B; ops [op10:matmul x2, op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=4B, op7:sgdupdate.out=2B, op9:add.out=2B, x2[token]=2B

**local_sgd_chain[K=3,q=2,d=1]** at F=3·fwd (12), G=4·fwd (16): I* = 42, 3 RUs
- RU0: work 11, input 14 B; ops [op0:matmul x6, op1:add x2, op2:matmul x4, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, t0[token]=4B, x0[token]=4B
- RU1: work 11, input 14 B; ops [op4:matmul x6, op5:add x2, op6:matmul x4, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, t1[token]=4B, x1[token]=4B
- RU2: work 11, input 14 B; ops [op8:matmul x6, op9:add x2, op10:matmul x4, op11:sgdupdate x2]; imports lr[seed]=4B, op7:sgdupdate.out=2B, t2[token]=4B, x2[token]=4B

**local_sgd_chain[K=3,q=2,d=1]** at F=3·fwd (12), G=inf (-): I* = 42, 3 RUs
- RU0: work 11, input 14 B; ops [op0:matmul x6, op1:add x2, op2:matmul x4, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, t0[token]=4B, x0[token]=4B
- RU1: work 11, input 14 B; ops [op4:matmul x6, op5:add x2, op6:matmul x4, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, t1[token]=4B, x1[token]=4B
- RU2: work 11, input 14 B; ops [op8:matmul x6, op9:add x2, op10:matmul x4, op11:sgdupdate x2]; imports lr[seed]=4B, op7:sgdupdate.out=2B, t2[token]=4B, x2[token]=4B

**local_sgd_chain[K=3,q=2,d=1]** at F=inf (-), G=1·fwd (4): I* = 78, 9 RUs
- RU0: work 4, input 6 B; ops [op0:matmul x6]; imports W[accumulated]=2B, x0[token]=4B
- RU1: work 4, input 10 B; ops [op1:add x1, op2:matmul x4]; imports op0:matmul.out=2B, op1:add.out=2B, t0[token]=2B, x0[token]=4B
- RU2: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU3: work 4, input 10 B; ops [op3:sgdupdate x2, op4:matmul x3]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B, x1[token]=2B
- RU4: work 4, input 6 B; ops [op4:matmul x3, op5:add x1, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU5: work 4, input 14 B; ops [op5:add x1, op6:matmul x2, op7:sgdupdate x1]; imports lr[seed]=4B, op4:matmul.out=2B, op6:matmul.out=4B, t1[token]=2B, x1[token]=2B
- RU6: work 4, input 8 B; ops [op7:sgdupdate x1, op8:matmul x3, op9:add x1]; imports op3:sgdupdate.out=2B, op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU7: work 4, input 6 B; ops [op8:matmul x3, op9:add x1, op10:matmul x2]; imports op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU8: work 4, input 14 B; ops [op10:matmul x2, op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=4B, op7:sgdupdate.out=2B, op9:add.out=2B, x2[token]=2B

**local_sgd_chain[K=3,q=2,d=1]** at F=inf (-), G=4·fwd (16): I* = 40, 3 RUs
- RU0: work 11, input 14 B; ops [op0:matmul x6, op1:add x2, op2:matmul x4, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, t0[token]=4B, x0[token]=4B
- RU1: work 9, input 10 B; ops [op4:matmul x6, op5:add x2, op6:matmul x4]; imports op3:sgdupdate.out=2B, t1[token]=4B, x1[token]=4B
- RU2: work 13, input 16 B; ops [op7:sgdupdate x2, op8:matmul x6, op9:add x2, op10:matmul x4, op11:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B, t2[token]=4B, x2[token]=4B

**local_sgd_chain[K=1,q=1,d=2]** at F=1·fwd (6), G=1·fwd (6): I* = 48, 4 RUs
- 2 RUs: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU2: work 6, input 12 B; ops [op2:matmul x6, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=4B, x0[token]=2B
- RU3: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=6B, lr[seed]=4B, op2:matmul.out=6B

**local_sgd_chain[K=1,q=1,d=2]** at F=1·fwd (6), G=4·fwd (24): I* = 36, 2 RUs
- RU0: work 16, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x12]; imports W[accumulated]=8B, t0[token]=4B, x0[token]=4B
- RU1: work 8, input 20 B; ops [op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=8B

**local_sgd_chain[K=1,q=1,d=2]** at F=1·fwd (6), G=inf (-): I* = 36, 2 RUs
- RU0: work 16, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x12]; imports W[accumulated]=8B, t0[token]=4B, x0[token]=4B
- RU1: work 8, input 20 B; ops [op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=8B

**local_sgd_chain[K=1,q=1,d=2]** at F=1.5·fwd (9), G=1·fwd (6): I* = 48, 4 RUs
- 2 RUs: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU2: work 6, input 12 B; ops [op2:matmul x6, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=4B, x0[token]=2B
- RU3: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=6B, lr[seed]=4B, op2:matmul.out=6B

**local_sgd_chain[K=1,q=1,d=2]** at F=3·fwd (18), G=1·fwd (6): I* = 48, 4 RUs
- 2 RUs: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU2: work 6, input 12 B; ops [op2:matmul x6, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=4B, x0[token]=2B
- RU3: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=6B, lr[seed]=4B, op2:matmul.out=6B

**local_sgd_chain[K=1,q=1,d=2]** at F=inf (-), G=1·fwd (6): I* = 48, 4 RUs
- 2 RUs: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU2: work 6, input 12 B; ops [op2:matmul x6, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=4B, x0[token]=2B
- RU3: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=6B, lr[seed]=4B, op2:matmul.out=6B

**local_sgd_chain[K=2,q=1,d=2]** at F=1·fwd (6), G=1·fwd (6): I* = [84, 98] (not proved), 9 RUs
- 2 RUs: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 6, input 14 B; ops [op2:matmul x3, op3:sgdupdate x4]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=2B, op2:matmul.out=4B, x0[token]=2B
- RU4: work 6, input 14 B; ops [op3:sgdupdate x2, op6:matmul x3, op7:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B, op3:sgdupdate.out=2B, op5:add.out=2B, x1[token]=2B
- RU5: work 4, input 8 B; ops [op3:sgdupdate x1, op4:matmul x2, op6:matmul x3]; imports W[accumulated]=2B, op3:sgdupdate.out=2B, op5:add.out=2B, x1[token]=2B
- RU6: work 6, input 12 B; ops [op3:sgdupdate x1, op4:matmul x2, op5:add x1, op6:matmul x3]; imports W[accumulated]=2B, op3:sgdupdate.out=2B, op4:matmul.out=4B, t1[token]=2B, x1[token]=2B
- RU7: work 6, input 10 B; ops [op4:matmul x4, op5:add x1, op6:matmul x3]; imports op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU8: work 6, input 16 B; ops [op7:sgdupdate x6]; imports lr[seed]=4B, op3:sgdupdate.out=6B, op6:matmul.out=6B

**local_sgd_chain[K=2,q=1,d=2]** at F=1·fwd (6), G=4·fwd (24): I* = 68, 3 RUs
- RU0: work 16, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x12]; imports W[accumulated]=8B, t0[token]=4B, x0[token]=4B
- RU1: work 12, input 24 B; ops [op3:sgdupdate x8, op4:matmul x6]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=8B, x1[token]=4B
- RU2: work 20, input 28 B; ops [op4:matmul x2, op5:add x2, op6:matmul x12, op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, op4:matmul.out=8B, t1[token]=4B, x1[token]=4B

**local_sgd_chain[K=2,q=1,d=2]** at F=1·fwd (6), G=inf (-): I* = 68, 3 RUs
- RU0: work 16, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x12]; imports W[accumulated]=8B, t0[token]=4B, x0[token]=4B
- RU1: work 12, input 24 B; ops [op3:sgdupdate x8, op4:matmul x6]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=8B, x1[token]=4B
- RU2: work 20, input 28 B; ops [op4:matmul x2, op5:add x2, op6:matmul x12, op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, op4:matmul.out=8B, t1[token]=4B, x1[token]=4B

**local_sgd_chain[K=2,q=1,d=2]** at F=1.5·fwd (9), G=1·fwd (6): I* = [84, 98] (not proved), 9 RUs
- 2 RUs: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU2: work 4, input 6 B; ops [op2:matmul x6]; imports op1:add.out=4B, x0[token]=2B
- RU3: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=6B, lr[seed]=4B, op2:matmul.out=6B
- RU4: work 3, input 10 B; ops [op3:sgdupdate x1, op7:sgdupdate x2]; imports lr[seed]=4B, op2:matmul.out=2B, op3:sgdupdate.out=2B, op6:matmul.out=2B
- RU5: work 6, input 12 B; ops [op3:sgdupdate x1, op4:matmul x4, op6:matmul x3]; imports W[accumulated]=2B, op3:sgdupdate.out=4B, op5:add.out=2B, x1[token]=4B
- RU6: work 6, input 10 B; ops [op4:matmul x4, op5:add x1, op6:matmul x3]; imports op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU7: work 5, input 8 B; ops [op5:add x1, op6:matmul x6]; imports op4:matmul.out=2B, t1[token]=2B, x1[token]=4B
- RU8: work 6, input 16 B; ops [op7:sgdupdate x6]; imports lr[seed]=4B, op3:sgdupdate.out=6B, op6:matmul.out=6B

**local_sgd_chain[K=2,q=1,d=2]** at F=1.5·fwd (9), G=4·fwd (24): I* = 40, 2 RUs
- RU0: work 24, input 20 B; ops [op0:matmul x8, op1:add x2, op2:matmul x12, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, t0[token]=4B, x0[token]=4B
- RU1: work 24, input 20 B; ops [op4:matmul x8, op5:add x2, op6:matmul x12, op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, t1[token]=4B, x1[token]=4B

**local_sgd_chain[K=2,q=1,d=2]** at F=1.5·fwd (9), G=inf (-): I* = 40, 2 RUs
- RU0: work 24, input 20 B; ops [op0:matmul x8, op1:add x2, op2:matmul x12, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, t0[token]=4B, x0[token]=4B
- RU1: work 24, input 20 B; ops [op4:matmul x8, op5:add x2, op6:matmul x12, op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, t1[token]=4B, x1[token]=4B

**local_sgd_chain[K=2,q=1,d=2]** at F=3·fwd (18), G=1·fwd (6): I* = [84, 98] (not proved), 9 RUs
- 2 RUs: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU2: work 4, input 6 B; ops [op2:matmul x6]; imports op1:add.out=4B, x0[token]=2B
- RU3: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=4B, lr[seed]=4B, op2:matmul.out=8B
- RU4: work 5, input 12 B; ops [op3:sgdupdate x2, op4:matmul x4]; imports W[accumulated]=4B, op3:sgdupdate.out=4B, x1[token]=4B
- RU5: work 6, input 10 B; ops [op4:matmul x4, op5:add x1, op6:matmul x3]; imports op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU6: work 5, input 8 B; ops [op5:add x1, op6:matmul x6]; imports op4:matmul.out=2B, t1[token]=2B, x1[token]=4B
- RU7: work 4, input 10 B; ops [op6:matmul x3, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op5:add.out=2B, x1[token]=2B
- RU8: work 6, input 16 B; ops [op7:sgdupdate x6]; imports lr[seed]=4B, op3:sgdupdate.out=6B, op6:matmul.out=6B

**local_sgd_chain[K=2,q=1,d=2]** at F=3·fwd (18), G=4·fwd (24): I* = 40, 2 RUs
- RU0: work 24, input 20 B; ops [op0:matmul x8, op1:add x2, op2:matmul x12, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, t0[token]=4B, x0[token]=4B
- RU1: work 24, input 20 B; ops [op4:matmul x8, op5:add x2, op6:matmul x12, op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, t1[token]=4B, x1[token]=4B

**local_sgd_chain[K=2,q=1,d=2]** at F=3·fwd (18), G=inf (-): I* = 40, 2 RUs
- RU0: work 24, input 20 B; ops [op0:matmul x8, op1:add x2, op2:matmul x12, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, t0[token]=4B, x0[token]=4B
- RU1: work 24, input 20 B; ops [op4:matmul x8, op5:add x2, op6:matmul x12, op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, t1[token]=4B, x1[token]=4B

**local_sgd_chain[K=2,q=1,d=2]** at F=inf (-), G=1·fwd (6): I* = [84, 98] (not proved), 9 RUs
- 2 RUs: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU2: work 4, input 6 B; ops [op2:matmul x6]; imports op1:add.out=4B, x0[token]=2B
- RU3: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=6B, lr[seed]=4B, op2:matmul.out=6B
- RU4: work 6, input 14 B; ops [op3:sgdupdate x2, op6:matmul x3, op7:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B, op3:sgdupdate.out=2B, op5:add.out=2B, x1[token]=2B
- RU5: work 6, input 12 B; ops [op4:matmul x8]; imports op3:sgdupdate.out=8B, x1[token]=4B
- RU6: work 5, input 8 B; ops [op5:add x1, op6:matmul x6]; imports op4:matmul.out=2B, op5:add.out=2B, t1[token]=2B, x1[token]=2B
- RU7: work 3, input 6 B; ops [op5:add x1, op6:matmul x3]; imports op4:matmul.out=2B, t1[token]=2B, x1[token]=2B
- RU8: work 6, input 16 B; ops [op7:sgdupdate x6]; imports lr[seed]=4B, op3:sgdupdate.out=6B, op6:matmul.out=6B

**local_sgd_chain[K=2,q=1,d=2]** at F=inf (-), G=4·fwd (24): I* = 40, 2 RUs
- RU0: work 24, input 20 B; ops [op0:matmul x8, op1:add x2, op2:matmul x12, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, t0[token]=4B, x0[token]=4B
- RU1: work 24, input 20 B; ops [op4:matmul x8, op5:add x2, op6:matmul x12, op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, t1[token]=4B, x1[token]=4B

**local_sgd_chain[K=3,q=1,d=2]** at F=1·fwd (6), G=1·fwd (6): I* = [36, 148] (not proved), 14 RUs
- RU18: work 6, input 12 B; ops [op0:matmul x8]; imports W[accumulated]=8B, x0[token]=4B
- 2 RUs: work 5, input 8 B; ops [op1:add x1, op2:matmul x6]; imports op0:matmul.out=2B, t0[token]=2B, x0[token]=4B
- RU40: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=4B, lr[seed]=4B, op2:matmul.out=8B
- RU48: work 6, input 10 B; ops [op4:matmul x4, op5:add x1, op6:matmul x3]; imports op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU52: work 6, input 14 B; ops [op3:sgdupdate x2, op4:matmul x4, op5:add x1]; imports W[accumulated]=4B, op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU61: work 6, input 8 B; ops [op6:matmul x9]; imports op5:add.out=4B, x1[token]=4B
- RU70: work 6, input 16 B; ops [op7:sgdupdate x6]; imports lr[seed]=4B, op3:sgdupdate.out=4B, op6:matmul.out=8B
- RU78: work 6, input 10 B; ops [op8:matmul x4, op9:add x1, op10:matmul x3]; imports op7:sgdupdate.out=4B, t2[token]=2B, x2[token]=4B
- RU82: work 6, input 14 B; ops [op7:sgdupdate x2, op8:matmul x4, op9:add x1]; imports op3:sgdupdate.out=4B, op7:sgdupdate.out=4B, t2[token]=2B, x2[token]=4B
- RU91: work 6, input 8 B; ops [op10:matmul x9]; imports op9:add.out=4B, x2[token]=4B
- RU100: work 6, input 16 B; ops [op11:sgdupdate x6]; imports lr[seed]=4B, op10:matmul.out=8B, op7:sgdupdate.out=4B
- 2 RUs: work 1, input 4 B; ops [op11:sgdupdate x1]; imports op11:sgdupdate.out=2B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=1,d=2]** at F=1·fwd (6), G=4·fwd (24): I* = [36, 108] (not proved), 6 RUs
- RU0: work 16, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x12]; imports W[accumulated]=8B, t0[token]=4B, x0[token]=4B
- RU3: work 8, input 20 B; ops [op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=8B
- RU4: work 16, input 16 B; ops [op4:matmul x8, op5:add x2, op6:matmul x12]; imports op3:sgdupdate.out=8B, t1[token]=4B, x1[token]=4B
- RU7: work 8, input 20 B; ops [op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, op6:matmul.out=8B
- RU8: work 16, input 16 B; ops [op8:matmul x8, op9:add x2, op10:matmul x12]; imports op7:sgdupdate.out=8B, t2[token]=4B, x2[token]=4B
- RU11: work 8, input 20 B; ops [op11:sgdupdate x8]; imports lr[seed]=4B, op10:matmul.out=8B, op7:sgdupdate.out=8B

**local_sgd_chain[K=3,q=1,d=2]** at F=1·fwd (6), G=inf (-): I* = [36, 108] (not proved), 6 RUs
- RU18: work 16, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x12]; imports W[accumulated]=8B, t0[token]=4B, x0[token]=4B
- RU40: work 8, input 20 B; ops [op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=8B
- RU48: work 16, input 16 B; ops [op4:matmul x8, op5:add x2, op6:matmul x12]; imports op3:sgdupdate.out=8B, t1[token]=4B, x1[token]=4B
- RU70: work 8, input 20 B; ops [op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, op6:matmul.out=8B
- RU78: work 16, input 16 B; ops [op8:matmul x8, op9:add x2, op10:matmul x12]; imports op7:sgdupdate.out=8B, t2[token]=4B, x2[token]=4B
- RU100: work 8, input 20 B; ops [op11:sgdupdate x8]; imports lr[seed]=4B, op10:matmul.out=8B, op7:sgdupdate.out=8B

**local_sgd_chain[K=3,q=1,d=2]** at F=1.5·fwd (9), G=1·fwd (6): I* = [36, 148] (not proved), 14 RUs
- RU18: work 6, input 12 B; ops [op0:matmul x8]; imports W[accumulated]=8B, x0[token]=4B
- 2 RUs: work 5, input 8 B; ops [op1:add x1, op2:matmul x6]; imports op0:matmul.out=2B, t0[token]=2B, x0[token]=4B
- RU40: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=4B, lr[seed]=4B, op2:matmul.out=8B
- RU48: work 6, input 10 B; ops [op4:matmul x4, op5:add x1, op6:matmul x3]; imports op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU52: work 6, input 14 B; ops [op3:sgdupdate x2, op4:matmul x4, op5:add x1]; imports W[accumulated]=4B, op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU61: work 6, input 8 B; ops [op6:matmul x9]; imports op5:add.out=4B, x1[token]=4B
- RU70: work 6, input 16 B; ops [op7:sgdupdate x6]; imports lr[seed]=4B, op3:sgdupdate.out=4B, op6:matmul.out=8B
- RU78: work 6, input 10 B; ops [op8:matmul x4, op9:add x1, op10:matmul x3]; imports op7:sgdupdate.out=4B, t2[token]=2B, x2[token]=4B
- RU82: work 6, input 14 B; ops [op7:sgdupdate x2, op8:matmul x4, op9:add x1]; imports op3:sgdupdate.out=4B, op7:sgdupdate.out=4B, t2[token]=2B, x2[token]=4B
- RU91: work 6, input 8 B; ops [op10:matmul x9]; imports op9:add.out=4B, x2[token]=4B
- RU100: work 6, input 16 B; ops [op11:sgdupdate x6]; imports lr[seed]=4B, op10:matmul.out=8B, op7:sgdupdate.out=4B
- 2 RUs: work 1, input 4 B; ops [op11:sgdupdate x1]; imports op11:sgdupdate.out=2B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=1,d=2]** at F=1.5·fwd (9), G=4·fwd (24): I* = [36, 60] (not proved), 3 RUs
- RU18: work 24, input 20 B; ops [op0:matmul x8, op1:add x2, op2:matmul x12, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, t0[token]=4B, x0[token]=4B
- RU48: work 24, input 20 B; ops [op4:matmul x8, op5:add x2, op6:matmul x12, op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, t1[token]=4B, x1[token]=4B
- RU78: work 24, input 20 B; ops [op8:matmul x8, op9:add x2, op10:matmul x12, op11:sgdupdate x8]; imports lr[seed]=4B, op7:sgdupdate.out=8B, t2[token]=4B, x2[token]=4B

**local_sgd_chain[K=3,q=1,d=2]** at F=1.5·fwd (9), G=inf (-): I* = [36, 60] (not proved), 3 RUs
- RU18: work 24, input 20 B; ops [op0:matmul x8, op1:add x2, op2:matmul x12, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, t0[token]=4B, x0[token]=4B
- RU48: work 24, input 20 B; ops [op4:matmul x8, op5:add x2, op6:matmul x12, op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, t1[token]=4B, x1[token]=4B
- RU78: work 24, input 20 B; ops [op8:matmul x8, op9:add x2, op10:matmul x12, op11:sgdupdate x8]; imports lr[seed]=4B, op7:sgdupdate.out=8B, t2[token]=4B, x2[token]=4B

**local_sgd_chain[K=3,q=1,d=2]** at F=3·fwd (18), G=1·fwd (6): I* = [36, 148] (not proved), 14 RUs
- RU18: work 6, input 12 B; ops [op0:matmul x8]; imports W[accumulated]=8B, x0[token]=4B
- 2 RUs: work 5, input 8 B; ops [op1:add x1, op2:matmul x6]; imports op0:matmul.out=2B, t0[token]=2B, x0[token]=4B
- RU40: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=4B, lr[seed]=4B, op2:matmul.out=8B
- RU48: work 6, input 10 B; ops [op4:matmul x4, op5:add x1, op6:matmul x3]; imports op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU52: work 6, input 14 B; ops [op3:sgdupdate x2, op4:matmul x4, op5:add x1]; imports W[accumulated]=4B, op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU61: work 6, input 8 B; ops [op6:matmul x9]; imports op5:add.out=4B, x1[token]=4B
- RU70: work 6, input 16 B; ops [op7:sgdupdate x6]; imports lr[seed]=4B, op3:sgdupdate.out=4B, op6:matmul.out=8B
- RU78: work 6, input 10 B; ops [op8:matmul x4, op9:add x1, op10:matmul x3]; imports op7:sgdupdate.out=4B, t2[token]=2B, x2[token]=4B
- RU82: work 6, input 14 B; ops [op7:sgdupdate x2, op8:matmul x4, op9:add x1]; imports op3:sgdupdate.out=4B, op7:sgdupdate.out=4B, t2[token]=2B, x2[token]=4B
- RU91: work 6, input 8 B; ops [op10:matmul x9]; imports op9:add.out=4B, x2[token]=4B
- RU100: work 6, input 16 B; ops [op11:sgdupdate x6]; imports lr[seed]=4B, op10:matmul.out=8B, op7:sgdupdate.out=4B
- 2 RUs: work 1, input 4 B; ops [op11:sgdupdate x1]; imports op11:sgdupdate.out=2B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=1,d=2]** at F=3·fwd (18), G=4·fwd (24): I* = [36, 60] (not proved), 3 RUs
- RU18: work 24, input 20 B; ops [op0:matmul x8, op1:add x2, op2:matmul x12, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, t0[token]=4B, x0[token]=4B
- RU48: work 24, input 20 B; ops [op4:matmul x8, op5:add x2, op6:matmul x12, op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, t1[token]=4B, x1[token]=4B
- RU78: work 24, input 20 B; ops [op8:matmul x8, op9:add x2, op10:matmul x12, op11:sgdupdate x8]; imports lr[seed]=4B, op7:sgdupdate.out=8B, t2[token]=4B, x2[token]=4B

**local_sgd_chain[K=3,q=1,d=2]** at F=3·fwd (18), G=inf (-): I* = [36, 56] (not proved), 2 RUs
- RU18: work 40, input 28 B; ops [op0:matmul x8, op1:add x2, op2:matmul x12, op3:sgdupdate x8, op4:matmul x8, op5:add x2, op6:matmul x12]; imports W[accumulated]=8B, lr[seed]=4B, t0[token]=4B, t1[token]=4B, x0[token]=4B, x1[token]=4B
- RU70: work 32, input 28 B; ops [op7:sgdupdate x8, op8:matmul x8, op9:add x2, op10:matmul x12, op11:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, op6:matmul.out=8B, t2[token]=4B, x2[token]=4B

**local_sgd_chain[K=3,q=1,d=2]** at F=inf (-), G=1·fwd (6): I* = [36, 148] (not proved), 14 RUs
- RU18: work 6, input 12 B; ops [op0:matmul x8]; imports W[accumulated]=8B, x0[token]=4B
- 2 RUs: work 5, input 8 B; ops [op1:add x1, op2:matmul x6]; imports op0:matmul.out=2B, t0[token]=2B, x0[token]=4B
- RU40: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=4B, lr[seed]=4B, op2:matmul.out=8B
- RU48: work 6, input 10 B; ops [op4:matmul x4, op5:add x1, op6:matmul x3]; imports op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU52: work 6, input 14 B; ops [op3:sgdupdate x2, op4:matmul x4, op5:add x1]; imports W[accumulated]=4B, op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU61: work 6, input 8 B; ops [op6:matmul x9]; imports op5:add.out=4B, x1[token]=4B
- RU70: work 6, input 16 B; ops [op7:sgdupdate x6]; imports lr[seed]=4B, op3:sgdupdate.out=4B, op6:matmul.out=8B
- RU78: work 6, input 10 B; ops [op8:matmul x4, op9:add x1, op10:matmul x3]; imports op7:sgdupdate.out=4B, t2[token]=2B, x2[token]=4B
- RU82: work 6, input 14 B; ops [op7:sgdupdate x2, op8:matmul x4, op9:add x1]; imports op3:sgdupdate.out=4B, op7:sgdupdate.out=4B, t2[token]=2B, x2[token]=4B
- RU91: work 6, input 8 B; ops [op10:matmul x9]; imports op9:add.out=4B, x2[token]=4B
- RU100: work 6, input 16 B; ops [op11:sgdupdate x6]; imports lr[seed]=4B, op10:matmul.out=8B, op7:sgdupdate.out=4B
- 2 RUs: work 1, input 4 B; ops [op11:sgdupdate x1]; imports op11:sgdupdate.out=2B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=1,d=2]** at F=inf (-), G=4·fwd (24): I* = [36, 60] (not proved), 3 RUs
- RU18: work 24, input 20 B; ops [op0:matmul x8, op1:add x2, op2:matmul x12, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, t0[token]=4B, x0[token]=4B
- RU48: work 24, input 20 B; ops [op4:matmul x8, op5:add x2, op6:matmul x12, op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, t1[token]=4B, x1[token]=4B
- RU78: work 24, input 20 B; ops [op8:matmul x8, op9:add x2, op10:matmul x12, op11:sgdupdate x8]; imports lr[seed]=4B, op7:sgdupdate.out=8B, t2[token]=4B, x2[token]=4B

**local_sgd_chain[K=1,q=2,d=2]** at F=1·fwd (12), G=1·fwd (12): I* = 60, 3 RUs
- 2 RUs: work 12, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x6]; imports W[accumulated]=4B, t0[token]=4B, x0[token]=8B
- RU2: work 12, input 28 B; ops [op2:matmul x4, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=16B

**local_sgd_chain[K=1,q=2,d=2]** at F=1·fwd (12), G=4·fwd (48): I* = 42, 2 RUs
- RU0: work 34, input 36 B; ops [op0:matmul x12, op1:add x4, op2:matmul x16, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op0:matmul.out=8B, t0[token]=8B, x0[token]=8B
- RU1: work 2, input 6 B; ops [op0:matmul x4]; imports W[accumulated]=4B, x0[token]=2B

**local_sgd_chain[K=1,q=2,d=2]** at F=1·fwd (12), G=inf (-): I* = 42, 2 RUs
- RU0: work 34, input 36 B; ops [op0:matmul x12, op1:add x4, op2:matmul x16, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op0:matmul.out=8B, t0[token]=8B, x0[token]=8B
- RU1: work 2, input 6 B; ops [op0:matmul x4]; imports W[accumulated]=4B, x0[token]=2B

**local_sgd_chain[K=1,q=2,d=2]** at F=1.5·fwd (18), G=1·fwd (12): I* = 60, 3 RUs
- 2 RUs: work 12, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x6]; imports W[accumulated]=4B, t0[token]=4B, x0[token]=8B
- RU2: work 12, input 28 B; ops [op2:matmul x4, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=16B

**local_sgd_chain[K=1,q=2,d=2]** at F=3·fwd (36), G=1·fwd (12): I* = 60, 3 RUs
- 2 RUs: work 12, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x6]; imports W[accumulated]=4B, t0[token]=4B, x0[token]=8B
- RU2: work 12, input 28 B; ops [op2:matmul x4, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=16B

**local_sgd_chain[K=1,q=2,d=2]** at F=inf (-), G=1·fwd (12): I* = 60, 3 RUs
- 2 RUs: work 12, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x6]; imports W[accumulated]=4B, t0[token]=4B, x0[token]=8B
- RU2: work 12, input 28 B; ops [op2:matmul x4, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=16B

**local_sgd_chain[K=2,q=2,d=2]** at F=1·fwd (12), G=1·fwd (12): I* = [44, 136] (not proved), 8 RUs
- RU22: work 12, input 16 B; ops [op0:matmul x16]; imports W[accumulated]=8B, x0[token]=8B
- RU42: work 11, input 20 B; ops [op1:add x2, op2:matmul x12]; imports op0:matmul.out=4B, op1:add.out=4B, t0[token]=4B, x0[token]=8B
- RU50: work 5, input 12 B; ops [op1:add x2, op2:matmul x4]; imports op0:matmul.out=4B, t0[token]=4B, x0[token]=4B
- RU58: work 8, input 20 B; ops [op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=8B
- RU66: work 12, input 16 B; ops [op4:matmul x16]; imports op3:sgdupdate.out=8B, x1[token]=8B
- RU86: work 11, input 20 B; ops [op5:add x2, op6:matmul x12]; imports op4:matmul.out=4B, op5:add.out=4B, t1[token]=4B, x1[token]=8B
- RU94: work 5, input 12 B; ops [op5:add x2, op6:matmul x4]; imports op4:matmul.out=4B, t1[token]=4B, x1[token]=4B
- RU102: work 8, input 20 B; ops [op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, op6:matmul.out=8B

**local_sgd_chain[K=2,q=2,d=2]** at F=1·fwd (12), G=4·fwd (48): I* = [44, 88] (not proved), 3 RUs
- RU0: work 28, input 24 B; ops [op0:matmul x16, op1:add x4, op2:matmul x16]; imports W[accumulated]=8B, t0[token]=8B, x0[token]=8B
- RU3: work 24, input 36 B; ops [op3:sgdupdate x8, op4:matmul x16, op5:add x4]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=8B, t1[token]=8B, x1[token]=8B
- RU6: work 20, input 28 B; ops [op6:matmul x16, op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, op5:add.out=8B, x1[token]=8B

**local_sgd_chain[K=2,q=2,d=2]** at F=1·fwd (12), G=inf (-): I* = [44, 88] (not proved), 3 RUs
- RU0: work 28, input 24 B; ops [op0:matmul x16, op1:add x4, op2:matmul x16]; imports W[accumulated]=8B, t0[token]=8B, x0[token]=8B
- RU3: work 24, input 36 B; ops [op3:sgdupdate x8, op4:matmul x16, op5:add x4]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=8B, t1[token]=8B, x1[token]=8B
- RU6: work 20, input 28 B; ops [op6:matmul x16, op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, op5:add.out=8B, x1[token]=8B

**local_sgd_chain[K=2,q=2,d=2]** at F=1.5·fwd (18), G=1·fwd (12): I* = [44, 136] (not proved), 8 RUs
- RU22: work 12, input 16 B; ops [op0:matmul x16]; imports W[accumulated]=8B, x0[token]=8B
- RU42: work 11, input 20 B; ops [op1:add x2, op2:matmul x12]; imports op0:matmul.out=4B, op1:add.out=4B, t0[token]=4B, x0[token]=8B
- RU50: work 5, input 12 B; ops [op1:add x2, op2:matmul x4]; imports op0:matmul.out=4B, t0[token]=4B, x0[token]=4B
- RU58: work 8, input 20 B; ops [op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=8B
- RU66: work 12, input 16 B; ops [op4:matmul x16]; imports op3:sgdupdate.out=8B, x1[token]=8B
- RU86: work 11, input 20 B; ops [op5:add x2, op6:matmul x12]; imports op4:matmul.out=4B, op5:add.out=4B, t1[token]=4B, x1[token]=8B
- RU94: work 5, input 12 B; ops [op5:add x2, op6:matmul x4]; imports op4:matmul.out=4B, t1[token]=4B, x1[token]=4B
- RU102: work 8, input 20 B; ops [op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, op6:matmul.out=8B

**local_sgd_chain[K=2,q=2,d=2]** at F=1.5·fwd (18), G=4·fwd (48): I* = [44, 56] (not proved), 2 RUs
- RU0: work 36, input 28 B; ops [op0:matmul x16, op1:add x4, op2:matmul x16, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, t0[token]=8B, x0[token]=8B
- RU4: work 36, input 28 B; ops [op4:matmul x16, op5:add x4, op6:matmul x16, op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, t1[token]=8B, x1[token]=8B

**local_sgd_chain[K=2,q=2,d=2]** at F=1.5·fwd (18), G=inf (-): I* = [44, 56] (not proved), 2 RUs
- RU0: work 36, input 28 B; ops [op0:matmul x16, op1:add x4, op2:matmul x16, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, t0[token]=8B, x0[token]=8B
- RU4: work 36, input 28 B; ops [op4:matmul x16, op5:add x4, op6:matmul x16, op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, t1[token]=8B, x1[token]=8B

**local_sgd_chain[K=2,q=2,d=2]** at F=3·fwd (36), G=1·fwd (12): I* = [44, 136] (not proved), 8 RUs
- RU22: work 12, input 16 B; ops [op0:matmul x16]; imports W[accumulated]=8B, x0[token]=8B
- RU42: work 11, input 20 B; ops [op1:add x2, op2:matmul x12]; imports op0:matmul.out=4B, op1:add.out=4B, t0[token]=4B, x0[token]=8B
- RU50: work 5, input 12 B; ops [op1:add x2, op2:matmul x4]; imports op0:matmul.out=4B, t0[token]=4B, x0[token]=4B
- RU58: work 8, input 20 B; ops [op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=8B
- RU66: work 12, input 16 B; ops [op4:matmul x16]; imports op3:sgdupdate.out=8B, x1[token]=8B
- RU86: work 11, input 20 B; ops [op5:add x2, op6:matmul x12]; imports op4:matmul.out=4B, op5:add.out=4B, t1[token]=4B, x1[token]=8B
- RU94: work 5, input 12 B; ops [op5:add x2, op6:matmul x4]; imports op4:matmul.out=4B, t1[token]=4B, x1[token]=4B
- RU102: work 8, input 20 B; ops [op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, op6:matmul.out=8B

**local_sgd_chain[K=2,q=2,d=2]** at F=3·fwd (36), G=4·fwd (48): I* = [44, 56] (not proved), 2 RUs
- RU0: work 36, input 28 B; ops [op0:matmul x16, op1:add x4, op2:matmul x16, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, t0[token]=8B, x0[token]=8B
- RU4: work 36, input 28 B; ops [op4:matmul x16, op5:add x4, op6:matmul x16, op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, t1[token]=8B, x1[token]=8B

**local_sgd_chain[K=2,q=2,d=2]** at F=inf (-), G=1·fwd (12): I* = [44, 136] (not proved), 8 RUs
- RU22: work 12, input 16 B; ops [op0:matmul x16]; imports W[accumulated]=8B, x0[token]=8B
- RU42: work 11, input 20 B; ops [op1:add x2, op2:matmul x12]; imports op0:matmul.out=4B, op1:add.out=4B, t0[token]=4B, x0[token]=8B
- RU50: work 5, input 12 B; ops [op1:add x2, op2:matmul x4]; imports op0:matmul.out=4B, t0[token]=4B, x0[token]=4B
- RU58: work 8, input 20 B; ops [op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=8B
- RU66: work 12, input 16 B; ops [op4:matmul x16]; imports op3:sgdupdate.out=8B, x1[token]=8B
- RU86: work 11, input 20 B; ops [op5:add x2, op6:matmul x12]; imports op4:matmul.out=4B, op5:add.out=4B, t1[token]=4B, x1[token]=8B
- RU94: work 5, input 12 B; ops [op5:add x2, op6:matmul x4]; imports op4:matmul.out=4B, t1[token]=4B, x1[token]=4B
- RU102: work 8, input 20 B; ops [op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, op6:matmul.out=8B

**local_sgd_chain[K=2,q=2,d=2]** at F=inf (-), G=4·fwd (48): I* = [44, 56] (not proved), 2 RUs
- RU0: work 36, input 28 B; ops [op0:matmul x16, op1:add x4, op2:matmul x16, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, t0[token]=8B, x0[token]=8B
- RU4: work 36, input 28 B; ops [op4:matmul x16, op5:add x4, op6:matmul x16, op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, t1[token]=8B, x1[token]=8B

**fanout_shared_weight[n=4,q=1,d=2]** at F=1·fwd (6), G=1·fwd (6): I* = [56, 72] (not proved), 6 RUs
- 2 RUs: work 4, input 12 B; ops [op0:sgdupdate x4]; imports Gt[token]=4B, W[accumulated]=4B, lr[seed]=4B
- RU2: work 6, input 12 B; ops [op1:matmul x8]; imports op0:sgdupdate.out=8B, x0[token]=4B
- RU3: work 6, input 12 B; ops [op2:matmul x8]; imports op0:sgdupdate.out=8B, x1[token]=4B
- RU4: work 6, input 12 B; ops [op3:matmul x8]; imports op0:sgdupdate.out=8B, x2[token]=4B
- RU5: work 6, input 12 B; ops [op4:matmul x8]; imports op0:sgdupdate.out=8B, x3[token]=4B

**fanout_shared_weight[n=4,q=1,d=2]** at F=1·fwd (6), G=4·fwd (24): I* = 44, 2 RUs
- RU0: work 8, input 20 B; ops [op0:sgdupdate x8]; imports Gt[token]=8B, W[accumulated]=8B, lr[seed]=4B
- RU1: work 24, input 24 B; ops [op1:matmul x8, op2:matmul x8, op3:matmul x8, op4:matmul x8]; imports op0:sgdupdate.out=8B, x0[token]=4B, x1[token]=4B, x2[token]=4B, x3[token]=4B

**fanout_shared_weight[n=4,q=1,d=2]** at F=1·fwd (6), G=inf (-): I* = 44, 2 RUs
- RU0: work 8, input 20 B; ops [op0:sgdupdate x8]; imports Gt[token]=8B, W[accumulated]=8B, lr[seed]=4B
- RU1: work 24, input 24 B; ops [op1:matmul x8, op2:matmul x8, op3:matmul x8, op4:matmul x8]; imports op0:sgdupdate.out=8B, x0[token]=4B, x1[token]=4B, x2[token]=4B, x3[token]=4B

**fanout_shared_weight[n=4,q=1,d=2]** at F=1.5·fwd (9), G=1·fwd (6): I* = [56, 72] (not proved), 6 RUs
- 2 RUs: work 4, input 12 B; ops [op0:sgdupdate x4]; imports Gt[token]=4B, W[accumulated]=4B, lr[seed]=4B
- RU2: work 6, input 12 B; ops [op1:matmul x8]; imports op0:sgdupdate.out=8B, x0[token]=4B
- RU3: work 6, input 12 B; ops [op2:matmul x8]; imports op0:sgdupdate.out=8B, x1[token]=4B
- RU4: work 6, input 12 B; ops [op3:matmul x8]; imports op0:sgdupdate.out=8B, x2[token]=4B
- RU5: work 6, input 12 B; ops [op4:matmul x8]; imports op0:sgdupdate.out=8B, x3[token]=4B

**fanout_shared_weight[n=4,q=1,d=2]** at F=1.5·fwd (9), G=4·fwd (24): I* = 44, 2 RUs
- RU0: work 20, input 28 B; ops [op0:sgdupdate x8, op1:matmul x8, op2:matmul x8]; imports Gt[token]=8B, W[accumulated]=8B, lr[seed]=4B, x0[token]=4B, x1[token]=4B
- RU1: work 12, input 16 B; ops [op3:matmul x8, op4:matmul x8]; imports op0:sgdupdate.out=8B, x2[token]=4B, x3[token]=4B

**fanout_shared_weight[n=4,q=1,d=2]** at F=3·fwd (18), G=1·fwd (6): I* = [56, 72] (not proved), 6 RUs
- 2 RUs: work 4, input 12 B; ops [op0:sgdupdate x4]; imports Gt[token]=4B, W[accumulated]=4B, lr[seed]=4B
- RU2: work 6, input 12 B; ops [op1:matmul x8]; imports op0:sgdupdate.out=8B, x0[token]=4B
- RU3: work 6, input 12 B; ops [op2:matmul x8]; imports op0:sgdupdate.out=8B, x1[token]=4B
- RU4: work 6, input 12 B; ops [op3:matmul x8]; imports op0:sgdupdate.out=8B, x2[token]=4B
- RU5: work 6, input 12 B; ops [op4:matmul x8]; imports op0:sgdupdate.out=8B, x3[token]=4B

**fanout_shared_weight[n=4,q=1,d=2]** at F=3·fwd (18), G=4·fwd (24): I* = 44, 2 RUs
- RU0: work 20, input 28 B; ops [op0:sgdupdate x8, op1:matmul x8, op2:matmul x8]; imports Gt[token]=8B, W[accumulated]=8B, lr[seed]=4B, x0[token]=4B, x1[token]=4B
- RU1: work 12, input 16 B; ops [op3:matmul x8, op4:matmul x8]; imports op0:sgdupdate.out=8B, x2[token]=4B, x3[token]=4B

**fanout_shared_weight[n=4,q=1,d=2]** at F=inf (-), G=1·fwd (6): I* = [56, 72] (not proved), 6 RUs
- 2 RUs: work 4, input 12 B; ops [op0:sgdupdate x4]; imports Gt[token]=4B, W[accumulated]=4B, lr[seed]=4B
- RU2: work 6, input 12 B; ops [op1:matmul x8]; imports op0:sgdupdate.out=8B, x0[token]=4B
- RU3: work 6, input 12 B; ops [op2:matmul x8]; imports op0:sgdupdate.out=8B, x1[token]=4B
- RU4: work 6, input 12 B; ops [op3:matmul x8]; imports op0:sgdupdate.out=8B, x2[token]=4B
- RU5: work 6, input 12 B; ops [op4:matmul x8]; imports op0:sgdupdate.out=8B, x3[token]=4B

**fanout_shared_weight[n=4,q=1,d=2]** at F=inf (-), G=4·fwd (24): I* = 44, 2 RUs
- RU0: work 20, input 28 B; ops [op0:sgdupdate x8, op1:matmul x8, op2:matmul x8]; imports Gt[token]=8B, W[accumulated]=8B, lr[seed]=4B, x0[token]=4B, x1[token]=4B
- RU1: work 12, input 16 B; ops [op3:matmul x8, op4:matmul x8]; imports op0:sgdupdate.out=8B, x2[token]=4B, x3[token]=4B

**deep_chain[depth=4,q=1,d=2,W=fixed]** at F=1·fwd (6), G=1·fwd (6): I* = 16, 4 RUs
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B

**deep_chain[depth=4,q=1,d=2,W=fixed]** at F=1·fwd (6), G=4·fwd (24): I* = 16, 4 RUs
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B

**deep_chain[depth=4,q=1,d=2,W=fixed]** at F=1·fwd (6), G=inf (-): I* = 16, 4 RUs
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B

**deep_chain[depth=4,q=1,d=2,W=fixed]** at F=1.5·fwd (9), G=1·fwd (6): I* = 16, 4 RUs
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B

**deep_chain[depth=4,q=1,d=2,W=fixed]** at F=1.5·fwd (9), G=4·fwd (24): I* = 8, 2 RUs
- RU0: work 12, input 4 B; ops [op0:matmul x8, op1:matmul x8]; imports x[token]=4B
- RU1: work 12, input 4 B; ops [op2:matmul x8, op3:matmul x8]; imports op1:matmul.out=4B

**deep_chain[depth=4,q=1,d=2,W=fixed]** at F=1.5·fwd (9), G=inf (-): I* = 8, 2 RUs
- RU0: work 12, input 4 B; ops [op0:matmul x8, op1:matmul x8]; imports x[token]=4B
- RU1: work 12, input 4 B; ops [op2:matmul x8, op3:matmul x8]; imports op1:matmul.out=4B

**deep_chain[depth=4,q=1,d=2,W=fixed]** at F=3·fwd (18), G=1·fwd (6): I* = 16, 4 RUs
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B

**deep_chain[depth=4,q=1,d=2,W=fixed]** at F=3·fwd (18), G=4·fwd (24): I* = 8, 2 RUs
- RU0: work 18, input 4 B; ops [op0:matmul x8, op1:matmul x8, op2:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B

**deep_chain[depth=4,q=1,d=2,W=fixed]** at F=3·fwd (18), G=inf (-): I* = 8, 2 RUs
- RU0: work 18, input 4 B; ops [op0:matmul x8, op1:matmul x8, op2:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B

**deep_chain[depth=4,q=1,d=2,W=fixed]** at F=inf (-), G=1·fwd (6): I* = 16, 4 RUs
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B

**deep_chain[depth=6,q=1,d=2,W=fixed]** at F=1·fwd (6), G=1·fwd (6): I* = 24, 6 RUs
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B
- RU4: work 6, input 4 B; ops [op4:matmul x8]; imports op3:matmul.out=4B
- RU5: work 6, input 4 B; ops [op5:matmul x8]; imports op4:matmul.out=4B

**deep_chain[depth=6,q=1,d=2,W=fixed]** at F=1·fwd (6), G=4·fwd (24): I* = 24, 6 RUs
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B
- RU4: work 6, input 4 B; ops [op4:matmul x8]; imports op3:matmul.out=4B
- RU5: work 6, input 4 B; ops [op5:matmul x8]; imports op4:matmul.out=4B

**deep_chain[depth=6,q=1,d=2,W=fixed]** at F=1·fwd (6), G=inf (-): I* = 24, 6 RUs
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B
- RU4: work 6, input 4 B; ops [op4:matmul x8]; imports op3:matmul.out=4B
- RU5: work 6, input 4 B; ops [op5:matmul x8]; imports op4:matmul.out=4B

**deep_chain[depth=6,q=1,d=2,W=fixed]** at F=1.5·fwd (9), G=1·fwd (6): I* = 24, 6 RUs
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B
- RU4: work 6, input 4 B; ops [op4:matmul x8]; imports op3:matmul.out=4B
- RU5: work 6, input 4 B; ops [op5:matmul x8]; imports op4:matmul.out=4B

**deep_chain[depth=6,q=1,d=2,W=fixed]** at F=1.5·fwd (9), G=4·fwd (24): I* = 12, 3 RUs
- RU0: work 12, input 4 B; ops [op0:matmul x8, op1:matmul x8]; imports x[token]=4B
- RU1: work 12, input 4 B; ops [op2:matmul x8, op3:matmul x8]; imports op1:matmul.out=4B
- RU2: work 12, input 4 B; ops [op4:matmul x8, op5:matmul x8]; imports op3:matmul.out=4B

**deep_chain[depth=6,q=1,d=2,W=fixed]** at F=1.5·fwd (9), G=inf (-): I* = 12, 3 RUs
- RU0: work 12, input 4 B; ops [op0:matmul x8, op1:matmul x8]; imports x[token]=4B
- RU1: work 12, input 4 B; ops [op2:matmul x8, op3:matmul x8]; imports op1:matmul.out=4B
- RU2: work 12, input 4 B; ops [op4:matmul x8, op5:matmul x8]; imports op3:matmul.out=4B

**deep_chain[depth=6,q=1,d=2,W=fixed]** at F=3·fwd (18), G=1·fwd (6): I* = 24, 6 RUs
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B
- RU4: work 6, input 4 B; ops [op4:matmul x8]; imports op3:matmul.out=4B
- RU5: work 6, input 4 B; ops [op5:matmul x8]; imports op4:matmul.out=4B

**deep_chain[depth=6,q=1,d=2,W=fixed]** at F=3·fwd (18), G=4·fwd (24): I* = 8, 2 RUs
- RU0: work 18, input 4 B; ops [op0:matmul x8, op1:matmul x8, op2:matmul x8]; imports x[token]=4B
- RU1: work 18, input 4 B; ops [op3:matmul x8, op4:matmul x8, op5:matmul x8]; imports op2:matmul.out=4B

**deep_chain[depth=6,q=1,d=2,W=fixed]** at F=3·fwd (18), G=inf (-): I* = 8, 2 RUs
- RU0: work 18, input 4 B; ops [op0:matmul x8, op1:matmul x8, op2:matmul x8]; imports x[token]=4B
- RU1: work 18, input 4 B; ops [op3:matmul x8, op4:matmul x8, op5:matmul x8]; imports op2:matmul.out=4B

**deep_chain[depth=6,q=1,d=2,W=fixed]** at F=inf (-), G=1·fwd (6): I* = 24, 6 RUs
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B
- RU4: work 6, input 4 B; ops [op4:matmul x8]; imports op3:matmul.out=4B
- RU5: work 6, input 4 B; ops [op5:matmul x8]; imports op4:matmul.out=4B

**deep_chain[depth=6,q=1,d=2,W=fixed]** at F=inf (-), G=4·fwd (24): I* = 8, 2 RUs
- RU0: work 24, input 4 B; ops [op0:matmul x8, op1:matmul x8, op2:matmul x8, op3:matmul x8]; imports x[token]=4B
- RU1: work 12, input 4 B; ops [op4:matmul x8, op5:matmul x8]; imports op3:matmul.out=4B

