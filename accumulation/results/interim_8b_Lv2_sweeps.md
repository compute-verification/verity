# Interim: Llama-3 8B, L v2 (closure-budget, uniform 1/share splits) vs U, across G and X

Orchestrator's own runs (2026-09-15), `pretrain-dense` vs `inference-dense`, Q = seq = 4096, F = 1e12, per token.
M_inf = 8.58e9 MAC/token (all matmul MACs), M_train = 2.63e10 MAC/token (credited, recompute excluded).
rho_cert = (M_tr/L_tr)/(M_inf/U_inf); rho_ach = (M_tr/U_tr)/(M_inf/U_inf). These will be superseded by
`results/REPORT.md` once the tightened certificate lands; kept so the before/after is on record.

## G sweep at X = 5 MB

| G | U_inf MB/tok | kappa_inf | L_tr MB/tok | U_tr MB/tok | U/L | kappa_L | kappa_U | rho_cert | penalty | rho_ach |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 2.44 | 3517 | 19.9 | 217 | 10.9 | 1319 | 121 | 0.375 | 2.7x | 0.034 |
| 1e10 | 2.50 | 3433 | 20.1 | 219 | 10.9 | 1309 | 120 | 0.381 | 2.6x | 0.035 |
| 1e9 | 4.60 | 1866 | 43.5 | 248 | 5.7 | 605 | 106 | 0.324 | 3.1x | 0.057 |
| 1e8 | 11.7 | 731 | 106 | 652 | 6.2 | 249 | 40 | 0.340 | 2.9x | 0.055 |
| 3e7 | 21.6 | 397 | 164 | 1360 | 8.3 | 161 | 19 | 0.405 | 2.5x | 0.049 |

## X sweep (G = none / G = 1e9)

| G | X | U_inf | kappa_inf | L_tr | U_tr | U/L | kappa_L | kappa_U | rho_cert | rho_ach |
|---|---|---|---|---|---|---|---|---|---|---|
| none | 0.5 MB | 10.96 | 782 | 120.6 | 742 | 6.2 | 218 | 35 | 0.279 | 0.045 |
| none | 1 MB | 6.41 | 1338 | 69.6 | 530 | 7.6 | 377 | 50 | 0.282 | 0.037 |
| none | 2 MB | 4.36 | 1966 | 41.9 | 384 | 9.2 | 627 | 68 | 0.319 | 0.035 |
| none | 5 MB | 2.44 | 3517 | 20.0 | 217 | 10.8 | 1310 | 121 | 0.373 | 0.034 |
| none | 10 MB | 2.31 | 3712 | 10.6 | 160 | 15.1 | 2482 | 164 | 0.669 | 0.044 |
| none | 20 MB | 2.31 | 3713 | 5.7 | 113 | 19.6 | 4575 | 233 | 1.232 | 0.063 |
| none | 50 MB | 1.04 | 8244 | 3.9 (= L_source) | 65 | 16.5 | 6700 | 407 | 0.813 | 0.049 |
| 1e9 | 0.5 MB | 11.29 | 760 | 120.9 | 742 | 6.1 | 217 | 35 | 0.286 | 0.047 |
| 1e9 | 1 MB | 6.60 | 1301 | 70.9 | 530 | 7.5 | 370 | 50 | 0.285 | 0.038 |
| 1e9 | 2 MB | 5.01 | 1714 | 51.5 | 385 | 7.5 | 510 | 68 | 0.298 | 0.040 |
| 1e9 | 5 MB | 4.60 | 1866 | 43.5 | 248 | 5.7 | 605 | 106 | 0.324 | 0.057 |
| 1e9 | 10 MB | 4.60 | 1866 | 36.8 | 238 | 6.5 | 715 | 111 | 0.383 | 0.059 |
| 1e9 | 20 MB | 4.60 | 1866 | 28.6 | 237 | 8.3 | 917 | 111 | 0.491 | 0.059 |
| 1e9 | 50 MB | 4.60 | 1866 | 18.9 | 237 | 12.5 | 1389 | 111 | 0.744 | 0.059 |

## Reading

1. The best *known* attack is 17-30x less byte-efficient than honest inference at every (G, X) tried (rho_ach
   0.034-0.063). The *certified* penalty is only 2.5-3.6x (rho_cert 0.28-0.40) for X <= 5 MB and vanishes for
   X >= 20 MB. The whole difference is certificate slack: U/L = 6-20.
2. G does not move rho_cert: the training L and the honest inference U both scale like G^(-1/3) (Loomis-Whitney
   in the G-bound regime), so their ratio is flat. G controls RU granularity, not the relative penalty.
3. L v2 scales like 1/X (collapsing to L_source at X = 50 MB) whereas the Loomis-Whitney X-form predicts
   1/sqrt(X). The steep decay is the credit recursion dying (forced-import base case n_B >= N), not adversary
   capability: regenerating one element of a dense activation inside an RU from the previous layer costs
   >= w K (a+b)/(ab) >= 16 K^2 / X = 54 B (X = 5 MB) and still 5.4 B at X = 50 MB, i.e. always more than importing it
   at 2 B. So kappa ~ w should persist for dense activations at every X <= 50 MB.
4. Worst-RU diagnostic at G = 1e8 shows the binding classes (MLP fwd/dgrad/wgrad) priced 0.004-0.005 B/MAC with
   alpha ~ 0.012 B per activation element: the activation credit reaching the matmuls is ~0.1 B (of w = 2 B)
   after uniform 1/share divisions along the residual stream (share 7) and again at x_norm (q/k/v + wgrads).
   Tightening task sent to the lower-bound agent: claiming-consumer splits, fused Q/K/V LW charge, generation-
   cost floor. Expected: alpha ~ 1, beta ~ 0.67 -> L ~ 50 MB/token at X = 5 MB (U/L ~ 4, rho_cert ~ 0.15).
