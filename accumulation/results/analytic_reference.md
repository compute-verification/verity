# Analytic estimates (plausibility reference and pod fill, not a result)

Closed-form estimates of the training runtime input under the F/G policy -- a plausibility reference for the coarse
planner (`upper_coarse`) and certificate (`lower_coarse`), and the fill for cells that cannot run on this machine
(`status = "pod"`); never a bound.  Imports only are counted (an export is the consumer's import).

Units.  `fwd = Work(inference-dense, Q_inf)` in the checker's units (all ops); `F = F_hat fwd`, `G = G_hat fwd`.
`w_tl` = all-op work per token-layer of the training circuit, `w_c` = credited MACs per token-layer (8B: 1.165e9 /
9.38e8, from `calibration.json`); `M_train/token = w_c L`.

Area cap.  `Up_R(g)` is the ancestor-set work.  Any training unit that reduces `dW` over its tokens has a gate whose
ancestors include every token's forward chain below and backward chain above inside the unit, so `Up ~ (1/2-2/3)
Work(unit)` and `F` is an *area* cap: `Work(unit) <~ c_F F` with `c_F ~ 1.5` (the planner finds the exact factor).
The effective cap in token-layers is `A = min(c_F F, G) / w_tl`; `G` is irrelevant whenever `G_hat > c_F F_hat`
(branching allowance beyond that is free for security).

Steady-state block model.  A unit covering `m` layers x `u` tokens of forward + backward (`m u <= A`) imports the
activation rows `4 d u` (`x` in below, `dY` in above) and the weights: `c_W = 2` B/param when `u >= Q_step`
(whole steps inside the unit -- weights enter once, updates are internal) and `c_W = 6` when `u < Q_step` (weights
in + partial `dW` out to the reducer).  Per credited MAC `cost(m, u) = 4 d / (m w_c) + c_W P_l / (u w_c)`; at
`u = A / m` the optimum is `m* = sqrt(4 d A / (c_W P_l))`, `cost* = 2 sqrt(4 d c_W P_l / A) / w_c`, and
`b_train = cost* M_train/token`, always `>= 4 d (L / m)` per token (the row floor).  8B default (`F_hat 1.5, G_hat
4, Q_inf 8192`): `A ~ 1.5e5`, `m* ~ 2.4`, `u* ~ 6e4` -> ~0.44 MB/token; at `Q_step = 131072 > u*` `c_W = 6`
-> ~0.75 MB/token (a larger batch is *worse* for the adversary).  The `2 P / Q_step` weight floor is the `m = L` case
and applies only when `L Q_step <= A` (`Q_step <~ 4.8K` at 8B) -- never for real batches under these policies.

Transient.  While `K Q_step m w_tl <= c_F F` every step of the session fits in one chunk unit of `m` layers and only
rows cross: `4 d (L / m)` per token (0.26 MB/token at `m = 2` for 8B, the measured K = 2, 4 value).  The knee is
where the transient (with `m_t(K) = A / (K Q_step)` shrinking as `K` grows) crosses the steady state:
`K_knee = A b_ss / (4 d L Q_step)`; the K sweep must be fit *after* the knee.  `I(K) / (K Q_step) ~ min(transient,
steady) + P / (K Q_step)` (initial weights enter once per session).

Single-step block (labelled alternative, kept for comparison): one step, `t <= Q_step` tokens, imports `4 d t + 6 m
P_l`, MACs `3 t m A_l <= G`, `c* = 2 sqrt(8 d P_l / (A_l G))` -- see `block_estimate`.

Inference: one RU per session, `b_inf = 4 B/token` (token ids as V32; 2 B/token with 16-bit ids) whenever `F_hat >= 1`
and `G_hat >= 1` (the session's Work is 0.95-0.99 fwd).

Precision note: the IR has no precision knob.  Every charged width (activations 2 B, weights 2 B, gradients 4 B) scales
together with the numeric format, so `b_train` scales linearly with it; token ingress (4 B/token ids) does not, so
`rho` and the penalty scale linearly with the training precision and are unchanged by inference precision.



Calibration used for 288 of 756 rows (the rest use the closed-form MAC model of `characterize.analytic_macs`).

## Default policy (Q_inf = 8192, F_hat = 1.5, G_hat = 4, Q_step = 16 Q_inf)

| model | fwd (work) | A (token-layers) | cap | m* | u* (tokens) | c_W | cost* (B/credited MAC) | row floor 4dL/m (MB/tok) | b_train steady (MB/tok) | b transient K=2 (MB/tok) | K_knee pred | b single-step block (MB/tok) | b_inf B/tok | kappa_inf | kappa_train est | penalty est | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama3-1b | 1.48e+13 | 7.68e+04 | F-limited (area c_F F; G irrelevant) | 1.32 | 5.81e+04 | 6 | 3.57e-05 | 0.0992 | 0.2 | 0 | 0.892 | 0.248 | 4 (2 with 16-bit ids) | 4.43e+08 | 2.8e+04 | 1.58e+04 | calibration |
| llama3-8b | 7.97e+13 | 1.53e+05 | F-limited (area c_F F; G irrelevant) | 1.15 | 1.33e+05 | 2 | 1.87e-05 | 0.456 | 0.561 | 0 | 1.25 | 0.834 | 4 (2 with 16-bit ids) | 2.41e+09 | 5.35e+04 | 4.51e+04 | calibration |
| llama3-70b | 6.6e+14 | 3.86e+05 | F-limited (area c_F F; G irrelevant) | 2.69 | 1.43e+05 | 2 | 7.83e-06 | 0.973 | 1.93 | 1.78 | 2.17 | 4.54 | 4 (2 with 16-bit ids) | 2.01e+10 | 1.28e+05 | 1.57e+05 | calibration |
| llama3-405b | 3.61e+15 | 5.54e+05 | F-limited (area c_F F; G irrelevant) | 2.4 | 2.31e+05 | 2 | 4.65e-06 | 3.44 | 6.92 | 3.91 | 3.54 | 24.4 | 4 (2 with 16-bit ids) | 1.09e+11 | 2.15e+05 | 5.08e+05 | analytic MACs |
| mixtral-8x7b | 1.23e+14 | 1.31e+05 | F-limited (area c_F F; G irrelevant) | 1 | 1.31e+05 | 6 | 4.87e-05 | 0.524 | 2.65 | 0 | 5.05 | 3.81 | 4 (2 with 16-bit ids) | 3.72e+09 | 2.05e+04 | 1.81e+05 | analytic MACs |
| qwen3-235b-a22b | 2.82e+14 | 3.75e+05 | F-limited (area c_F F; G irrelevant) | 1.12 | 3.35e+05 | 2 | 2.16e-05 | 1.37 | 2.77 | 1.08 | 5.15 | 22.3 | 4 (2 with 16-bit ids) | 8.55e+09 | 4.62e+04 | 1.85e+05 | analytic MACs |
| deepseek-v3 | 4.67e+14 | 2.46e+05 | F-limited (area c_F F; G irrelevant) | 1 | 2.46e+05 | 2 | 3.56e-05 | 1.75 | 7.49 | 0 | 8.04 | 56.9 | 4 (2 with 16-bit ids) | 1.42e+10 | 2.81e+04 | 5.05e+05 | analytic MACs |

## All policy points: steady-state b_train estimate (MB/token)

Rows: model x Q_inf x F_hat; columns: G_hat x Q_step/Q_inf.  Cells with G_hat > 1.5 F_hat are identical along G (F-limited).

| model | Q_inf | F_hat | G=1 Qs=1x | G=1 Qs=16x | G=1 Qs=512x | G=4 Qs=1x | G=4 Qs=16x | G=4 Qs=512x | G=16 Qs=1x | G=16 Qs=16x | G=16 Qs=512x | G=100 Qs=1x | G=100 Qs=16x | G=100 Qs=512x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama3-1b | 4096 | 1 | 0.241 | 0.463 | 0.463 | 0.197 | 0.353 | 0.353 | 0.197 | 0.353 | 0.353 | 0.197 | 0.353 | 0.353 |
| llama3-1b | 4096 | 1.5 | 0.241 | 0.463 | 0.463 | 0.161 | 0.279 | 0.279 | 0.161 | 0.279 | 0.279 | 0.161 | 0.279 | 0.279 |
| llama3-1b | 4096 | 2 | 0.241 | 0.463 | 0.463 | 0.139 | 0.241 | 0.241 | 0.139 | 0.241 | 0.241 | 0.139 | 0.241 | 0.241 |
| llama3-1b | 8192 | 1 | 0.173 | 0.302 | 0.302 | 0.141 | 0.245 | 0.245 | 0.141 | 0.245 | 0.245 | 0.141 | 0.245 | 0.245 |
| llama3-1b | 8192 | 1.5 | 0.173 | 0.302 | 0.302 | 0.115 | 0.2 | 0.2 | 0.115 | 0.2 | 0.2 | 0.115 | 0.2 | 0.2 |
| llama3-1b | 8192 | 2 | 0.173 | 0.302 | 0.302 | 0.0998 | 0.173 | 0.173 | 0.0998 | 0.173 | 0.173 | 0.0998 | 0.173 | 0.173 |
| llama3-1b | 32768 | 1 | 0.0896 | 0.155 | 0.155 | 0.0732 | 0.127 | 0.127 | 0.0732 | 0.127 | 0.127 | 0.0732 | 0.127 | 0.127 |
| llama3-1b | 32768 | 1.5 | 0.0896 | 0.155 | 0.155 | 0.0597 | 0.103 | 0.103 | 0.0597 | 0.103 | 0.103 | 0.0597 | 0.103 | 0.103 |
| llama3-1b | 32768 | 2 | 0.0896 | 0.155 | 0.155 | 0.0517 | 0.0896 | 0.0896 | 0.0517 | 0.0896 | 0.0896 | 0.0517 | 0.0896 | 0.0896 |
| llama3-8b | 4096 | 1 | 0.919 | 1.73 | 1.73 | 0.75 | 1.33 | 1.33 | 0.75 | 1.33 | 1.33 | 0.75 | 1.33 | 1.33 |
| llama3-8b | 4096 | 1.5 | 0.919 | 1.73 | 1.73 | 0.613 | 0.653 | 1.06 | 0.613 | 0.653 | 1.06 | 0.613 | 0.653 | 1.06 |
| llama3-8b | 4096 | 2 | 0.919 | 1.73 | 1.73 | 0.531 | 0.544 | 0.919 | 0.531 | 0.544 | 0.919 | 0.531 | 0.544 | 0.919 |
| llama3-8b | 8192 | 1 | 0.656 | 1.14 | 1.14 | 0.536 | 0.928 | 0.928 | 0.536 | 0.928 | 0.928 | 0.536 | 0.928 | 0.928 |
| llama3-8b | 8192 | 1.5 | 0.656 | 1.14 | 1.14 | 0.438 | 0.561 | 0.758 | 0.438 | 0.561 | 0.758 | 0.438 | 0.561 | 0.758 |
| llama3-8b | 8192 | 2 | 0.656 | 1.14 | 1.14 | 0.379 | 0.445 | 0.656 | 0.379 | 0.445 | 0.656 | 0.379 | 0.445 | 0.656 |
| llama3-8b | 32768 | 1 | 0.338 | 0.586 | 0.586 | 0.276 | 0.479 | 0.479 | 0.276 | 0.479 | 0.479 | 0.276 | 0.479 | 0.479 |
| llama3-8b | 32768 | 1.5 | 0.338 | 0.586 | 0.586 | 0.226 | 0.391 | 0.391 | 0.226 | 0.391 | 0.391 | 0.226 | 0.391 | 0.391 |
| llama3-8b | 32768 | 2 | 0.338 | 0.586 | 0.586 | 0.195 | 0.338 | 0.338 | 0.195 | 0.338 | 0.338 | 0.195 | 0.338 | 0.338 |
| llama3-70b | 4096 | 1 | 4.06 | 4.06 | 7.35 | 3.32 | 3.32 | 5.77 | 3.32 | 3.32 | 5.77 | 3.32 | 3.32 | 5.77 |
| llama3-70b | 4096 | 1.5 | 4.06 | 4.06 | 7.35 | 2.71 | 2.71 | 4.69 | 2.71 | 2.71 | 4.69 | 2.71 | 2.71 | 4.69 |
| llama3-70b | 4096 | 2 | 4.06 | 4.06 | 7.35 | 2.35 | 2.35 | 4.06 | 2.35 | 2.35 | 4.06 | 2.35 | 2.35 | 4.06 |
| llama3-70b | 8192 | 1 | 2.89 | 3.05 | 5.01 | 2.36 | 2.38 | 4.09 | 2.36 | 2.38 | 4.09 | 2.36 | 2.38 | 4.09 |
| llama3-70b | 8192 | 1.5 | 2.89 | 3.05 | 5.01 | 1.93 | 1.93 | 3.34 | 1.93 | 1.93 | 3.34 | 1.93 | 1.93 | 3.34 |
| llama3-70b | 8192 | 2 | 2.89 | 3.05 | 5.01 | 1.67 | 1.67 | 2.89 | 1.67 | 1.67 | 2.89 | 1.67 | 1.67 | 2.89 |
| llama3-70b | 32768 | 1 | 1.56 | 2.61 | 2.7 | 1.27 | 1.84 | 2.21 | 1.27 | 1.84 | 2.21 | 1.27 | 1.84 | 2.21 |
| llama3-70b | 32768 | 1.5 | 1.56 | 2.61 | 2.7 | 1.04 | 1.3 | 1.8 | 1.04 | 1.3 | 1.8 | 1.04 | 1.3 | 1.8 |
| llama3-70b | 32768 | 2 | 1.56 | 2.61 | 2.7 | 0.901 | 1.04 | 1.56 | 0.901 | 1.04 | 1.56 | 0.901 | 1.04 | 1.56 |
| llama3-405b | 4096 | 1 | 14.6 | 14.6 | 27.7 | 11.9 | 11.9 | 21.2 | 11.9 | 11.9 | 21.2 | 11.9 | 11.9 | 21.2 |
| llama3-405b | 4096 | 1.5 | 14.6 | 14.6 | 27.7 | 9.75 | 9.75 | 16.9 | 9.75 | 9.75 | 16.9 | 9.75 | 9.75 | 16.9 |
| llama3-405b | 4096 | 2 | 14.6 | 14.6 | 27.7 | 8.44 | 8.44 | 14.6 | 8.44 | 8.44 | 14.6 | 8.44 | 8.44 | 14.6 |
| llama3-405b | 8192 | 1 | 10.4 | 10.4 | 18 | 8.48 | 8.48 | 14.7 | 8.48 | 8.48 | 14.7 | 8.48 | 8.48 | 14.7 |
| llama3-405b | 8192 | 1.5 | 10.4 | 10.4 | 18 | 6.92 | 6.92 | 12 | 6.92 | 6.92 | 12 | 6.92 | 6.92 | 12 |
| llama3-405b | 8192 | 2 | 10.4 | 10.4 | 18 | 5.99 | 5.99 | 10.4 | 5.99 | 5.99 | 10.4 | 5.99 | 5.99 | 10.4 |
| llama3-405b | 32768 | 1 | 5.28 | 6.12 | 9.15 | 4.31 | 4.57 | 7.47 | 4.31 | 4.57 | 7.47 | 4.31 | 4.57 | 7.47 |
| llama3-405b | 32768 | 1.5 | 5.28 | 6.12 | 9.15 | 3.52 | 3.57 | 6.1 | 3.52 | 3.57 | 6.1 | 3.52 | 3.57 | 6.1 |
| llama3-405b | 32768 | 2 | 5.28 | 6.12 | 9.15 | 3.05 | 3.05 | 5.28 | 3.05 | 3.05 | 5.28 | 3.05 | 3.05 | 5.28 |
| mixtral-8x7b | 4096 | 1 | 3.69 | 10 | 10 | 2.63 | 6.85 | 6.85 | 2.63 | 6.85 | 6.85 | 2.63 | 6.85 | 6.85 |
| mixtral-8x7b | 4096 | 1.5 | 3.69 | 10 | 10 | 1.93 | 1.93 | 4.74 | 1.93 | 1.93 | 4.74 | 1.93 | 1.93 | 4.74 |
| mixtral-8x7b | 4096 | 2 | 3.69 | 10 | 10 | 1.58 | 1.58 | 3.69 | 1.58 | 1.58 | 3.69 | 1.58 | 1.58 | 3.69 |
| mixtral-8x7b | 8192 | 1 | 2.12 | 5.31 | 5.31 | 1.59 | 3.72 | 3.72 | 1.59 | 3.72 | 3.72 | 1.59 | 3.72 | 3.72 |
| mixtral-8x7b | 8192 | 1.5 | 2.12 | 5.31 | 5.31 | 1.23 | 2.65 | 2.65 | 1.23 | 2.65 | 2.65 | 1.23 | 2.65 | 2.65 |
| mixtral-8x7b | 8192 | 2 | 2.12 | 5.31 | 5.31 | 1.06 | 1.06 | 2.12 | 1.06 | 1.06 | 2.12 | 1.06 | 1.06 | 2.12 |
| mixtral-8x7b | 32768 | 1 | 0.929 | 1.76 | 1.76 | 0.759 | 1.35 | 1.35 | 0.759 | 1.35 | 1.35 | 0.759 | 1.35 | 1.35 |
| mixtral-8x7b | 32768 | 1.5 | 0.929 | 1.76 | 1.76 | 0.62 | 1.07 | 1.07 | 0.62 | 1.07 | 1.07 | 0.62 | 1.07 | 1.07 |
| mixtral-8x7b | 32768 | 2 | 0.929 | 1.76 | 1.76 | 0.537 | 0.586 | 0.929 | 0.537 | 0.586 | 0.929 | 0.537 | 0.586 | 0.929 |
| qwen3-235b-a22b | 4096 | 1 | 7.04 | 7.04 | 18 | 5.21 | 5.21 | 12.5 | 5.21 | 5.21 | 12.5 | 5.21 | 5.21 | 12.5 |
| qwen3-235b-a22b | 4096 | 1.5 | 7.04 | 7.04 | 18 | 3.99 | 3.99 | 8.88 | 3.99 | 3.99 | 8.88 | 3.99 | 3.99 | 8.88 |
| qwen3-235b-a22b | 4096 | 2 | 7.04 | 7.04 | 18 | 3.37 | 3.37 | 7.04 | 3.37 | 3.37 | 7.04 | 3.37 | 3.37 | 7.04 |
| qwen3-235b-a22b | 8192 | 1 | 4.34 | 4.34 | 9.95 | 3.41 | 3.41 | 7.15 | 3.41 | 3.41 | 7.15 | 3.41 | 3.41 | 7.15 |
| qwen3-235b-a22b | 8192 | 1.5 | 4.34 | 4.34 | 9.95 | 2.77 | 2.77 | 5.28 | 2.77 | 2.77 | 5.28 | 2.77 | 2.77 | 5.28 |
| qwen3-235b-a22b | 8192 | 2 | 4.34 | 4.34 | 9.95 | 2.4 | 2.4 | 4.34 | 2.4 | 2.4 | 4.34 | 2.4 | 2.4 | 4.34 |
| qwen3-235b-a22b | 32768 | 1 | 2.12 | 2.16 | 3.73 | 1.73 | 1.73 | 3 | 1.73 | 1.73 | 3 | 1.73 | 1.73 | 3 |
| qwen3-235b-a22b | 32768 | 1.5 | 2.12 | 2.16 | 3.73 | 1.41 | 1.41 | 2.45 | 1.41 | 1.41 | 2.45 | 1.41 | 1.41 | 2.45 |
| qwen3-235b-a22b | 32768 | 2 | 2.12 | 2.16 | 3.73 | 1.22 | 1.22 | 2.12 | 1.22 | 1.22 | 2.12 | 1.22 | 1.22 | 2.12 |
| deepseek-v3 | 4096 | 1 | 27.2 | 78.1 | 78.1 | 18.7 | 18.7 | 52.7 | 18.7 | 18.7 | 52.7 | 18.7 | 18.7 | 52.7 |
| deepseek-v3 | 4096 | 1.5 | 27.2 | 78.1 | 78.1 | 13.1 | 13.1 | 35.7 | 13.1 | 13.1 | 35.7 | 13.1 | 13.1 | 35.7 |
| deepseek-v3 | 4096 | 2 | 27.2 | 78.1 | 78.1 | 10.2 | 10.2 | 27.2 | 10.2 | 10.2 | 27.2 | 10.2 | 10.2 | 27.2 |
| deepseek-v3 | 8192 | 1 | 14.7 | 40.5 | 40.5 | 10.4 | 10.4 | 27.6 | 10.4 | 10.4 | 27.6 | 10.4 | 10.4 | 27.6 |
| deepseek-v3 | 8192 | 1.5 | 14.7 | 40.5 | 40.5 | 7.49 | 7.49 | 19 | 7.49 | 7.49 | 19 | 7.49 | 7.49 | 19 |
| deepseek-v3 | 8192 | 2 | 14.7 | 40.5 | 40.5 | 6.06 | 6.06 | 14.7 | 6.06 | 6.06 | 14.7 | 6.06 | 6.06 | 14.7 |
| deepseek-v3 | 32768 | 1 | 5.12 | 11.9 | 11.9 | 3.99 | 3.99 | 8.49 | 3.99 | 3.99 | 8.49 | 3.99 | 3.99 | 8.49 |
| deepseek-v3 | 32768 | 1.5 | 5.12 | 11.9 | 11.9 | 3.24 | 3.24 | 6.24 | 3.24 | 3.24 | 6.24 | 3.24 | 3.24 | 6.24 |
| deepseek-v3 | 32768 | 2 | 5.12 | 11.9 | 11.9 | 2.8 | 2.8 | 5.12 | 2.8 | 2.8 | 5.12 | 2.8 | 2.8 | 5.12 |
