# Validation: coarse bounds vs. exact `I*` under the corrected policy (`X = inf`)

Generated 2026-09-16 08:30:08 by `accumulation.redteam.coarse_validation` (0.2 min).  Policy: X = inf; legal iff work(R) <= G and work(Up_R(g)) <= F; charged roles token/accumulated/carried/seed.  `U = upper_coarse` (`bounds/coarse.py`), expanded to gates with `plan_gate_assignment` and re-judged by `exact.solve.is_legal` / `partition_cost` ("lit").  `L = lower_coarse` (`bounds/lower.py`).  Exact time box 30.0 s per cell; unproved cells show the proven interval `[lower bound, best found]`.

## Aggregate

- **0/1772 soundness violations** (`L <= I*`, `U` legal, `U >= literal`, `U >= I*`) over 1772 cells (1729 with proved `I*`, 42 with an interval); by kind: {'L_gt_Istar': 0, 'U_illegal': 0, 'U_lt_literal': 0, 'U_lt_Istar': 0}.
- `L/I*` on proved cells: median 0.75, 5th percentile 0.30, min 0.00 (n=1729).
- `U/I*` on proved cells: median 1.00, 95th percentile 2.00, max 5.00 (n=1041); `U` unavailable in 707 cells; `U == literal` in 1057 cells, `U > literal` (conservative accounting) in 8.

## Validation panel (DELIVERABLES §4)

- `bounds/lower.py` sha1 `8c8626f5f6`, `bounds/lower_coarse.py` sha1 `df35839fdf`; other modules sha256[:10] {'bounds/coarse.py': 'db0e43d181', 'bounds/upper.py': '5c1dbc6803', 'graph/opgraph.py': 'e8052d0b2a', 'exact/solve.py': '56e945d7f6'}.
- **N = 1729 cells solved exactly** (+42 cells with a proven interval only, 1772 total); legality model: **acyclic RU quotient** (SPEC §0).
- vs the cyclic-quotient model (archived `validation_cyclic_model.json`): 1678 exact cells compared, **66 changed** (`I*_acyclic / I*_cyclic` median 1.00, max 1.33; median over changed cells 1.12); changed by family: {'micro local-sgd chain (d=1, q=1)': 24, 'micro local-sgd chain (d=1, q=2)': 16, 'micro local-sgd chain (d=2, q=1)': 3, 'micro local-sgd chain (d=2, q=2)': 2, 't': 2, 'i': 1, 'add+colsum+mul|shared,view,chain': 1, 'add+softmax': 2, 'softmax': 7, 'matmul+rmsnorm|view,chain': 3, 'add+matmul+rowscale+softmax|shared,chain': 3, 'gain+softmax|shared': 2}.
- **`U < I*`: 0** (incl. illegal / under-accounted plans); **`L > I*`: 0**; registry transformer bracket cells: 51, `L > I_hi`: 0, block term silent at 28 binding cells; registry rollout cells (dense + MoE): 95, `L > I_hi`: 0, `U` violations: 0, inference one-RU-tokens-only: 5/5.
- hashes checked: `lower_coarse.py` `df35839fdf`, `coarse.py` `f4baa2d7be` (block-term rows at `lower_coarse.py` `df35839fdf`) (rollout rows at `lower_coarse.py` `df35839fdf`, `coarse.py` `f4baa2d7be`).
- `U/I*` (legal, acyclic plans only): median 1.00, max 5.00 (n=1041; `U = I*` in 665 cells; `U` unavailable in 688; planner returned a **cyclic** (illegal under SPEC §0) plan in 0 cells, 0 of them below the acyclic `I*` -- expected, not counted).
- `L/I*`: median 0.75, 10th percentile 0.40, minimum 0.00 (n=1729; `L = I*` in 757 cells; `L` unavailable in 0).

### By circuit family

| split | exact cells | U/I* median | U/I* max | L/I* median | L/I* p10 | L/I* min | U<I* | L>I* |
|---|---|---|---|---|---|---|---|---|
| a | 18 | 1.00 | 1.00 | 0.50 | 0.33 | 0.33 | 0 | 0 |
| add | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| add+colsum+embed+rowscale|shared,view | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| add+colsum+matmul|shared,view | 4 | 1.57 | 1.75 | 0.90 | 0.67 | 0.62 | 0 | 0 |
| add+colsum+mul|shared,view,chain | 12 | 1.25 | 2.25 | 0.67 | 0.14 | 0.14 | 0 | 0 |
| add+colsum+rowscale|shared | 4 | 2.17 | 2.17 | 0.75 | 0.75 | 0.75 | 0 | 0 |
| add+embed+gain+matmul|shared,chain | 4 | 1.29 | 1.89 | 0.91 | 0.70 | 0.64 | 0 | 0 |
| add+embed+gain+matmul|shared,view,chain | 4 | 2.19 | 2.75 | 0.75 | 0.51 | 0.40 | 0 | 0 |
| add+embed+matmul_tt|shared,chain | 4 | 2.03 | 3.08 | 1.00 | 0.82 | 0.75 | 0 | 0 |
| add+embed+matmul|view,chain | 4 | 1.36 | 1.46 | 1.00 | 0.91 | 0.87 | 0 | 0 |
| add+embed+mul|shared,chain | 4 | 1.10 | 1.47 | 1.00 | 0.87 | 0.81 | 0 | 0 |
| add+embed+rowscale|shared,view | 4 | 1.29 | 1.38 | 0.97 | 0.92 | 0.90 | 0 | 0 |
| add+embed|shared | 8 | 1.67 | 2.00 | 1.00 | 0.65 | 0.62 | 0 | 0 |
| add+embed|shared,chain | 4 | 1.15 | 1.22 | 0.68 | 0.60 | 0.60 | 0 | 0 |
| add+gain+matmul+mul+swiglu|shared,view,chain | 4 | 1.89 | 2.38 | 0.88 | 0.75 | 0.70 | 0 | 0 |
| add+gain+matmul|shared,view,chain | 8 | 1.00 | 1.93 | 0.76 | 0.62 | 0.62 | 0 | 0 |
| add+gain+mul|shared | 4 | 2.00 | 2.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| add+gain|shared | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| add+gain|shared,view,chain | 4 | 2.25 | 2.50 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| add+matmul+mul+rowscale|shared,chain | 4 | 1.30 | 1.30 | 0.40 | 0.40 | 0.40 | 0 | 0 |
| add+matmul+mul|shared | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| add+matmul+rmsnorm|shared,view,chain | 8 | 1.00 | 1.00 | 0.71 | 0.59 | 0.50 | 0 | 0 |
| add+matmul+rowscale+softmax|shared,chain | 4 | - | - | 0.09 | 0.07 | 0.07 | 0 | 0 |
| add+matmul_tt+mul+rowscale|shared,view,chain | 4 | 1.50 | 1.50 | 0.62 | 0.50 | 0.50 | 0 | 0 |
| add+matmul_tt+swiglu|shared,chain | 4 | 1.00 | 1.00 | 0.69 | 0.57 | 0.57 | 0 | 0 |
| add+matmul_tt|shared,view | 8 | 1.00 | 1.00 | 1.00 | 0.54 | 0.33 | 0 | 0 |
| add+matmul|shared | 8 | 1.00 | 1.00 | 0.83 | 0.57 | 0.57 | 0 | 0 |
| add+matmul|shared,chain | 4 | 1.50 | 3.00 | 0.50 | 0.50 | 0.50 | 0 | 0 |
| add+matmul|shared,view | 4 | 1.00 | 1.00 | 0.67 | 0.67 | 0.67 | 0 | 0 |
| add+mul+rowscale|shared | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| add+mul|shared | 8 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| add+softmax | 4 | - | - | 0.60 | 0.50 | 0.50 | 0 | 0 |
| add+swiglu|chain | 4 | 2.00 | 2.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| add|shared | 36 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| add|shared,chain | 4 | 2.00 | 2.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| add|view | 8 | 1.00 | 2.00 | 1.00 | 0.67 | 0.67 | 0 | 0 |
| b | 9 | 1.00 | 1.00 | 0.29 | 0.17 | 0.17 | 0 | 0 |
| c | 18 | 1.00 | 1.00 | 1.00 | 0.50 | 0.50 | 0 | 0 |
| chain | 54 | 1.00 | 1.50 | 0.67 | 0.50 | 0.38 | 0 | 0 |
| colsum | 48 | 2.50 | 2.50 | 0.67 | 0.45 | 0.33 | 0 | 0 |
| colsum+embed+gain+matmul|shared | 4 | 1.30 | 1.40 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| colsum+embed+gain|shared,view,chain | 4 | 1.02 | 1.04 | 0.89 | 0.72 | 0.69 | 0 | 0 |
| colsum+gain | 4 | - | - | 1.00 | 1.00 | 1.00 | 0 | 0 |
| colsum+gain|shared | 4 | 1.00 | 1.00 | 0.60 | 0.53 | 0.50 | 0 | 0 |
| colsum+gain|view | 4 | 1.22 | 1.44 | 0.38 | 0.31 | 0.31 | 0 | 0 |
| colsum+matmul+mul|shared | 4 | 1.00 | 1.22 | 0.95 | 0.84 | 0.82 | 0 | 0 |
| colsum+matmul+rowscale+swiglu|shared | 4 | 1.05 | 1.33 | 0.90 | 0.84 | 0.82 | 0 | 0 |
| colsum+matmul|shared,view | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| colsum+matmul|shared,view,chain | 4 | 1.50 | 1.50 | 0.62 | 0.62 | 0.62 | 0 | 0 |
| colsum+mul|shared | 4 | 1.00 | 1.00 | 0.58 | 0.50 | 0.50 | 0 | 0 |
| colsum+mul|shared,chain | 4 | 2.17 | 3.33 | 0.75 | 0.50 | 0.50 | 0 | 0 |
| colsum+rowscale | 4 | - | - | 1.00 | 1.00 | 1.00 | 0 | 0 |
| colsum+softmax|shared | 4 | 1.00 | 1.00 | 0.18 | 0.17 | 0.17 | 0 | 0 |
| colsum|view | 4 | - | - | 0.00 | 0.00 | 0.00 | 0 | 0 |
| d | 18 | 1.43 | 2.00 | 0.77 | 0.50 | 0.50 | 0 | 0 |
| dense rollout micro | 20 | 1.00 | 1.71 | 0.62 | 0.44 | 0.36 | 0 | 0 |
| e | 18 | 1.06 | 3.00 | 0.81 | 0.33 | 0.33 | 0 | 0 |
| embed | 36 | 1.00 | 2.29 | 0.79 | 0.50 | 0.33 | 0 | 0 |
| embed+gain+matmul+matmul_tt|shared,view,chain | 4 | 1.07 | 1.18 | 0.87 | 0.80 | 0.76 | 0 | 0 |
| embed+gain+matmul|shared,view,chain | 4 | 1.83 | 2.50 | 0.83 | 0.67 | 0.67 | 0 | 0 |
| embed+matmul | 4 | 1.80 | 2.60 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| embed+matmul+matmul_tt|shared,view,chain | 4 | - | - | 0.75 | 0.65 | 0.60 | 0 | 0 |
| embed+matmul+mul+rowscale|shared,view,chain | 8 | 1.39 | 1.82 | 0.76 | 0.47 | 0.47 | 0 | 0 |
| embed+matmul+mul|shared,view | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| embed+matmul+mul|shared,view,chain | 4 | - | - | 0.83 | 0.78 | 0.78 | 0 | 0 |
| embed+matmul+rowscale|shared | 4 | 1.00 | 1.00 | 0.52 | 0.41 | 0.41 | 0 | 0 |
| embed+matmul_tt+mul|shared,chain | 4 | 1.00 | 1.00 | 0.88 | 0.75 | 0.75 | 0 | 0 |
| embed+matmul|shared | 12 | 1.00 | 1.37 | 1.00 | 0.67 | 0.67 | 0 | 0 |
| embed+matmul|shared,chain | 4 | 1.50 | 2.14 | 0.34 | 0.25 | 0.25 | 0 | 0 |
| embed+matmul|shared,view | 8 | - | - | 0.56 | 0.20 | 0.20 | 0 | 0 |
| embed+matmul|view | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| embed+mul | 4 | - | - | 0.88 | 0.85 | 0.85 | 0 | 0 |
| embed+mul+rowscale|shared | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| embed+mul|shared | 4 | 2.00 | 2.00 | 0.83 | 0.67 | 0.67 | 0 | 0 |
| embed+swiglu|shared | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| f | 18 | 1.38 | 2.33 | 0.52 | 0.38 | 0.38 | 0 | 0 |
| fixacc | 63 | 1.00 | 2.33 | 0.60 | 0.38 | 0.29 | 0 | 0 |
| forward-nonfixed @ tiny | 24 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| forward-nonfixed micro | 36 | 1.00 | 1.50 | 1.00 | 0.68 | 0.67 | 0 | 0 |
| g | 36 | 1.00 | 2.00 | 0.67 | 0.50 | 0.43 | 0 | 0 |
| gain | 20 | 1.00 | 3.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| gain+matmul+mul|shared,chain | 4 | 1.60 | 2.20 | 0.44 | 0.38 | 0.38 | 0 | 0 |
| gain+matmul+mul|shared,view,chain | 4 | 5.00 | 5.00 | 0.24 | 0.20 | 0.20 | 0 | 0 |
| gain+matmul+rmsnorm|shared,view | 4 | - | - | 0.45 | 0.34 | 0.31 | 0 | 0 |
| gain+matmul+swiglu|shared | 4 | 1.07 | 1.14 | 0.97 | 0.93 | 0.93 | 0 | 0 |
| gain+matmul_tt|shared | 4 | - | - | 0.53 | 0.40 | 0.40 | 0 | 0 |
| gain+matmul|chain | 4 | 1.23 | 3.20 | 0.73 | 0.36 | 0.31 | 0 | 0 |
| gain+matmul|shared | 4 | 1.53 | 2.17 | 0.83 | 0.52 | 0.46 | 0 | 0 |
| gain+matmul|shared,view | 8 | 1.12 | 1.29 | 1.00 | 0.78 | 0.78 | 0 | 0 |
| gain+mul+swiglu|shared,view,chain | 3 | 1.00 | 1.00 | 0.50 | 0.50 | 0.50 | 0 | 0 |
| gain+mul|view,chain | 4 | 1.17 | 1.33 | 0.58 | 0.50 | 0.50 | 0 | 0 |
| gain+softmax|shared | 4 | - | - | 0.25 | 0.22 | 0.22 | 0 | 0 |
| h | 36 | 1.00 | 2.00 | 0.71 | 0.50 | 0.33 | 0 | 0 |
| i | 9 | 1.33 | 1.60 | 0.22 | 0.17 | 0.17 | 0 | 0 |
| k | 9 | 1.00 | 1.50 | 0.67 | 0.50 | 0.50 | 0 | 0 |
| matmul | 24 | 1.12 | 1.80 | 0.72 | 0.52 | 0.50 | 0 | 0 |
| matmul+matmul_tt+mul|shared,chain | 4 | 1.24 | 1.40 | 0.75 | 0.71 | 0.71 | 0 | 0 |
| matmul+matmul_tt+mul|shared,view,chain | 4 | 1.06 | 1.19 | 0.83 | 0.78 | 0.78 | 0 | 0 |
| matmul+matmul_tt|shared,view,chain | 8 | 1.00 | 1.00 | 0.52 | 0.44 | 0.31 | 0 | 0 |
| matmul+mul | 4 | 1.00 | 1.00 | 0.83 | 0.67 | 0.67 | 0 | 0 |
| matmul+mul+swiglu|shared,chain | 4 | 1.25 | 1.25 | 0.75 | 0.75 | 0.75 | 0 | 0 |
| matmul+mul|shared,view,chain | 12 | 1.25 | 3.33 | 0.62 | 0.28 | 0.25 | 0 | 0 |
| matmul+rmsnorm|view,chain | 4 | 1.00 | 1.00 | 0.46 | 0.46 | 0.46 | 0 | 0 |
| matmul+rowscale|shared | 4 | 1.15 | 1.35 | 0.65 | 0.65 | 0.65 | 0 | 0 |
| matmul+rowscale|view | 4 | 1.00 | 1.00 | 0.72 | 0.62 | 0.62 | 0 | 0 |
| matmul+swiglu|view | 4 | - | - | 0.94 | 0.89 | 0.89 | 0 | 0 |
| matmul_tt | 4 | 1.00 | 1.00 | 0.85 | 0.59 | 0.55 | 0 | 0 |
| matmul_tt+mul+rmsnorm|shared,view | 4 | 1.00 | 1.33 | 0.80 | 0.60 | 0.60 | 0 | 0 |
| matmul_tt+mul|shared,chain | 4 | 1.00 | 1.00 | 0.75 | 0.50 | 0.50 | 0 | 0 |
| matmul_tt+rowscale|shared | 8 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| matmul_tt|shared | 4 | - | - | 0.42 | 0.33 | 0.33 | 0 | 0 |
| matmul_tt|view | 8 | 1.00 | 1.00 | 0.62 | 0.50 | 0.50 | 0 | 0 |
| matmul|shared | 26 | 1.50 | 3.00 | 0.50 | 0.35 | 0.33 | 0 | 0 |
| matmul|shared,view | 36 | 1.20 | 5.00 | 0.64 | 0.25 | 0.20 | 0 | 0 |
| matmul|view | 12 | 1.00 | 1.33 | 0.61 | 0.43 | 0.40 | 0 | 0 |
| micro deep chain (fixed weights) | 24 | 1.00 | 2.00 | 0.29 | 0.17 | 0.17 | 0 | 0 |
| micro fan-out (shared produced weight) | 8 | 1.00 | 1.64 | 0.82 | 0.82 | 0.82 | 0 | 0 |
| micro inference | 24 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| micro local-sgd chain (d=1, q=1) | 36 | 1.00 | 1.08 | 0.47 | 0.30 | 0.30 | 0 | 0 |
| micro local-sgd chain (d=1, q=2) | 36 | 1.11 | 1.25 | 0.54 | 0.40 | 0.38 | 0 | 0 |
| micro local-sgd chain (d=2, q=1) | 21 | 1.60 | 2.20 | 0.70 | 0.42 | 0.41 | 0 | 0 |
| micro local-sgd chain (d=2, q=2) | 14 | 1.13 | 2.14 | 1.00 | 0.47 | 0.47 | 0 | 0 |
| mm | 114 | 1.00 | 2.00 | 0.67 | 0.50 | 0.42 | 0 | 0 |
| mmadd | 54 | 1.00 | 1.67 | 0.69 | 0.50 | 0.50 | 0 | 0 |
| moe rollout micro | 20 | 1.11 | 1.50 | 0.70 | 0.39 | 0.39 | 0 | 0 |
| mul | 16 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| mul+rmsnorm|shared,view | 4 | - | - | 0.60 | 0.60 | 0.60 | 0 | 0 |
| mul+rowscale | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| mul+rowscale|shared,view,chain | 4 | 1.79 | 2.71 | 0.57 | 0.44 | 0.44 | 0 | 0 |
| mul|shared | 16 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| mul|shared,view | 4 | - | - | 1.00 | 1.00 | 1.00 | 0 | 0 |
| mul|view | 8 | 1.00 | 1.00 | 0.50 | 0.00 | 0.00 | 0 | 0 |
| r | 15 | 1.31 | 1.80 | 0.71 | 0.45 | 0.45 | 0 | 0 |
| registry inference-dense @ tiny | 36 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| registry inference-dense @ tiny2 | 36 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| rmsnorm | 8 | 1.00 | 1.00 | 0.62 | 0.38 | 0.38 | 0 | 0 |
| rowscale | 24 | 1.00 | 1.00 | 1.00 | 0.88 | 0.50 | 0 | 0 |
| rowscale|view | 8 | 1.00 | 1.80 | 1.00 | 0.60 | 0.56 | 0 | 0 |
| softmax | 12 | 1.00 | 1.00 | 0.25 | 0.18 | 0.15 | 0 | 0 |
| swiglu|shared | 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0 | 0 |
| swiglu|shared,view | 4 | - | - | 0.00 | 0.00 | 0.00 | 0 | 0 |
| t | 108 | 1.00 | 2.67 | 0.62 | 0.40 | 0.17 | 0 | 0 |

### By binding policy constraint (which cap forbids the single-RU partition)

| split | exact cells | U/I* median | U/I* max | L/I* median | L/I* p10 | L/I* min | U<I* | L>I* |
|---|---|---|---|---|---|---|---|---|
| F | 144 | 1.06 | 2.20 | 0.58 | 0.33 | 0.00 | 0 | 0 |
| F+G | 253 | 1.06 | 2.20 | 0.50 | 0.25 | 0.00 | 0 | 0 |
| G | 791 | 1.00 | 3.33 | 0.67 | 0.41 | 0.00 | 0 | 0 |
| none | 541 | 1.00 | 5.00 | 1.00 | 1.00 | 0.00 | 0 | 0 |

### Registry dynamic-rollout cells (`forward-nonfixed` @ tiny2, `forward-nonfixed-moe` @ tiny-moe; ROLLOUT_PLAN)

- from `redteam/rollout_scale.py` at `lower_coarse.py` sha1 `df35839fdf`, `coarse.py` sha1 `f4baa2d7be`; 95 cells, 26 exact (single RU legal), 69 bracketed; inference references one RU with imports = tokens: 5/5.
- **`L > I_hi`: 0**; `U` violations: 0; inference-not-tokens violations: 0; marginal-fit bracket checks failed: 0 (see `validation_rollout.md`).
- `forward-nonfixed` at the default point (F_hat=1.5, G_hat=4; n=13): U/ref median 1.00 max 1.10; L/ref median 0.87 min 0.33; U/L max 3.00.
- `forward-nonfixed-moe` at the default point (F_hat=1.5, G_hat=4; n=5): U/ref median 1.00 max 1.06; L/ref median 1.00 min 0.41; U/L max 2.42.

| circuit | Q_roll | gates | regime | binds | I* / [I_lo, I_hi] | L | U | L/ref | U/ref | U/L | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=4] | 4 | 3472 | F_hat=0.5 (session cone does not fit), G_hat=4 | F | [1048, 1080] | 1048 | 1112 | 0.97 | 1.03 | 1.06 | proved (L <= I_lo) |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=4] | 4 | 3472 | G-only: G_hat=1, F=inf | none | 1048 | 1048 | 1048 | 1.00 | 1.00 | 1.00 | exact |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=4] | 4 | 3472 | G_hat=1 (one session per RU; F_hat=1.5) | none | 1048 | 1048 | 1048 | 1.00 | 1.00 | 1.00 | exact |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=4] | 4 | 3472 | G_hat=2 (F_hat=1.5) | none | 1048 | 1048 | 1048 | 1.00 | 1.00 | 1.00 | exact |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=4] | 4 | 3472 | default: F_hat=1.5, G_hat=4 | none | 1048 | 1048 | 1048 | 1.00 | 1.00 | 1.00 | exact |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=8] | 8 | 6944 | F_hat=0.5 (session cone does not fit), G_hat=4 | F | [1064, 1192] | 1064 | 5032 | 0.89 | 4.22 | 4.73 | proved (L <= I_lo) |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=8] | 8 | 6944 | G-only: G_hat=1, F=inf | G | [1064, 1192] | 1064 | 1192 | 0.89 | 1.00 | 1.12 | proved (L <= I_lo) |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=8] | 8 | 6944 | G_hat=1 (one session per RU; F_hat=1.5) | G | [1064, 1192] | 1064 | 1192 | 0.89 | 1.00 | 1.12 | proved (L <= I_lo) |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=8] | 8 | 6944 | G_hat=2 (F_hat=1.5) | none | 1064 | 1064 | 1128 | 1.00 | 1.06 | 1.06 | exact |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=8] | 8 | 6944 | default: F_hat=1.5, G_hat=4 | none | 1064 | 1064 | 1128 | 1.00 | 1.06 | 1.06 | exact |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=16] | 16 | 13888 | F_hat=0.5 (session cone does not fit), G_hat=4 | F | [1096, 1352] | 1096 | 9032 | 0.81 | 6.68 | 8.24 | proved (L <= I_lo) |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=16] | 16 | 13888 | G-only: G_hat=1, F=inf | G | [1096, 2376] | 1145 | 2944 | 0.48 | 1.24 | 2.57 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=16] | 16 | 13888 | G_hat=1 (one session per RU; F_hat=1.5) | G | [1096, 2376] | 1145 | 2944 | 0.48 | 1.24 | 2.57 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=16] | 16 | 13888 | G_hat=2 (F_hat=1.5) | G | [1096, 1352] | 1096 | 1480 | 0.81 | 1.09 | 1.35 | proved (L <= I_lo) |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=16] | 16 | 13888 | default: F_hat=1.5, G_hat=4 | none | 1096 | 1096 | 1096 | 1.00 | 1.00 | 1.00 | exact |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=32] | 32 | 27776 | F_hat=0.5 (session cone does not fit), G_hat=4 | F+G | [1160, 1928] | 1160 | 17032 | 0.60 | 8.83 | 14.68 | proved (L <= I_lo) |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=32] | 32 | 27776 | G-only: G_hat=1, F=inf | G | [1160, 5712] | 1508 | 5712 | 0.26 | 1.00 | 3.79 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=32] | 32 | 27776 | G_hat=1 (one session per RU; F_hat=1.5) | G | [1160, 5712] | 1508 | 5712 | 0.26 | 1.00 | 3.79 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=32] | 32 | 27776 | G_hat=2 (F_hat=1.5) | G | [1160, 2984] | 1247 | 2984 | 0.42 | 1.00 | 2.39 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=32] | 32 | 27776 | default: F_hat=1.5, G_hat=4 | G | [1160, 1928] | 1160 | 1952 | 0.60 | 1.01 | 1.68 | proved (L <= I_lo) |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=64] | 64 | 55552 | F_hat=0.5 (session cone does not fit), G_hat=4 | F+G | [1288, 6408] | 1393 | 34120 | 0.22 | 5.32 | 24.49 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=64] | 64 | 55552 | G-only: G_hat=1, F=inf | G | [1288, 11360] | 2716 | 11360 | 0.24 | 1.00 | 4.18 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=64] | 64 | 55552 | G_hat=1 (one session per RU; F_hat=1.5) | G | [1288, 11360] | 2716 | 11360 | 0.24 | 1.00 | 4.18 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=64] | 64 | 55552 | G_hat=2 (F_hat=1.5) | G | [1288, 5584] | 1926 | 5584 | 0.34 | 1.00 | 2.90 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed-moe[tiny-moe,Q_inf=4,Q_roll=64] | 64 | 55552 | default: F_hat=1.5, G_hat=4 | G | [1288, 3368] | 1393 | 3368 | 0.41 | 1.00 | 2.42 | consistent (I_lo < L <= I_hi), not proved |
| inference-moe[tiny-moe,Q_inf=4,Q_roll=4] | 4 | 3472 | default: F_hat=1.5, G_hat=4 | none | 16 | 16 | 16 | 1.00 | 1.00 | 1.00 | exact |
| inference-moe[tiny-moe,Q_inf=8,Q_roll=8] | 8 | 7968 | default: F_hat=1.5, G_hat=4 | none | 32 | 32 | 32 | 1.00 | 1.00 | 1.00 | exact |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=1] | 1 | 662 | F_hat=0.5 (session cone does not fit), G_hat=4 | F | [748, 756] | 748 | 764 | 0.99 | 1.01 | 1.02 | proved (L <= I_lo) |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=1] | 1 | 662 | G-only: G_hat=1, F=inf | none | 748 | 748 | 748 | 1.00 | 1.00 | 1.00 | exact |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=1] | 1 | 662 | G_hat=1 (one session per RU; F_hat=1.5) | none | 748 | 748 | 748 | 1.00 | 1.00 | 1.00 | exact |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=1] | 1 | 662 | G_hat=2 (F_hat=1.5) | none | 748 | 748 | 748 | 1.00 | 1.00 | 1.00 | exact |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=1] | 1 | 662 | default: F_hat=1.5, G_hat=4 | none | 748 | 748 | 748 | 1.00 | 1.00 | 1.00 | exact |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=2] | 2 | 1324 | F_hat=0.5 (session cone does not fit), G_hat=4 | F | [752, 768] | 752 | 784 | 0.98 | 1.02 | 1.04 | proved (L <= I_lo) |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=2] | 2 | 1324 | G-only: G_hat=1, F=inf | G | [752, 784] | 762 | 784 | 0.97 | 1.00 | 1.03 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=2] | 2 | 1324 | G_hat=1 (one session per RU; F_hat=1.5) | G | [752, 784] | 762 | 784 | 0.97 | 1.00 | 1.03 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=2] | 2 | 1324 | G_hat=2 (F_hat=1.5) | none | 752 | 752 | 752 | 1.00 | 1.00 | 1.00 | exact |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=2] | 2 | 1324 | default: F_hat=1.5, G_hat=4 | none | 752 | 752 | 752 | 1.00 | 1.00 | 1.00 | exact |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=4] | 4 | 2648 | F_hat=0.5 (session cone does not fit), G_hat=4 | F | [760, 792] | 760 | 824 | 0.96 | 1.04 | 1.08 | proved (L <= I_lo) |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=4] | 4 | 2648 | G-only: G_hat=1, F=inf | G | [760, 952] | 792 | 1464 | 0.83 | 1.54 | 1.85 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=4] | 4 | 2648 | G_hat=1 (one session per RU; F_hat=1.5) | G | [760, 952] | 792 | 1464 | 0.83 | 1.54 | 1.85 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=4] | 4 | 2648 | G_hat=2 (F_hat=1.5) | G | [760, 824] | 765 | 824 | 0.93 | 1.00 | 1.08 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=4] | 4 | 2648 | default: F_hat=1.5, G_hat=4 | none | 760 | 760 | 760 | 1.00 | 1.00 | 1.00 | exact |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=8] | 8 | 5296 | F_hat=0.5 (session cone does not fit), G_hat=4 | F+G | [776, 904] | 782 | 904 | 0.87 | 1.00 | 1.16 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=8] | 8 | 5296 | G-only: G_hat=1, F=inf | G | [776, 2120] | 914 | 2792 | 0.43 | 1.32 | 3.05 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=8] | 8 | 5296 | G_hat=1 (one session per RU; F_hat=1.5) | G | [776, 2120] | 914 | 2792 | 0.43 | 1.32 | 3.05 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=8] | 8 | 5296 | G_hat=2 (F_hat=1.5) | G | [776, 1160] | 840 | 1576 | 0.72 | 1.36 | 1.88 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=8] | 8 | 5296 | default: F_hat=1.5, G_hat=4 | G | [776, 904] | 782 | 904 | 0.87 | 1.00 | 1.16 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=16] | 16 | 10592 | F_hat=0.5 (session cone does not fit), G_hat=4 | F+G | [808, 1576] | 936 | 1736 | 0.59 | 1.10 | 1.85 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=16] | 16 | 10592 | G-only: G_hat=1, F=inf | G | [808, 5264] | 1297 | 5520 | 0.25 | 1.05 | 4.26 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=16] | 16 | 10592 | G_hat=1 (one session per RU; F_hat=1.5) | G | [808, 5264] | 1297 | 5520 | 0.25 | 1.05 | 4.26 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=16] | 16 | 10592 | G_hat=2 (F_hat=1.5) | G | [808, 3016] | 1085 | 3016 | 0.36 | 1.00 | 2.78 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=16] | 16 | 10592 | default: F_hat=1.5, G_hat=4 | G | [808, 1576] | 936 | 1736 | 0.59 | 1.10 | 1.85 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=32] | 32 | 21184 | F_hat=0.5 (session cone does not fit), G_hat=4 | F+G | [872, 3400] | 1361 | 3400 | 0.40 | 1.00 | 2.50 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=32] | 32 | 21184 | G-only: G_hat=1, F=inf | G | [872, 10832] | 2139 | 10976 | 0.20 | 1.01 | 5.13 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=32] | 32 | 21184 | G_hat=1 (one session per RU; F_hat=1.5) | G | [872, 10832] | 2139 | 10976 | 0.20 | 1.01 | 5.13 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=32] | 32 | 21184 | G_hat=2 (F_hat=1.5) | G | [872, 5968] | 1683 | 5968 | 0.28 | 1.00 | 3.55 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=32] | 32 | 21184 | default: F_hat=1.5, G_hat=4 | G | [872, 3400] | 1361 | 3400 | 0.40 | 1.00 | 2.50 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=64] | 64 | 42368 | F_hat=0.5 (session cone does not fit), G_hat=4 | F+G | [1000, 6800] | 2267 | 6800 | 0.33 | 1.00 | 3.00 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=64] | 64 | 42368 | G-only: G_hat=1, F=inf | G | [1000, 22008] | 3823 | 23288 | 0.17 | 1.06 | 6.09 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=64] | 64 | 42368 | G_hat=1 (one session per RU; F_hat=1.5) | G | [1000, 22008] | 3823 | 23288 | 0.17 | 1.06 | 6.09 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=64] | 64 | 42368 | G_hat=2 (F_hat=1.5) | G | [1000, 11872] | 2911 | 11872 | 0.25 | 1.00 | 4.08 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=1,Q_roll=64] | 64 | 42368 | default: F_hat=1.5, G_hat=4 | G | [1000, 6800] | 2267 | 6800 | 0.33 | 1.00 | 3.00 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=2] | 2 | 1420 | F_hat=0.5 (session cone does not fit), G_hat=4 | F | [752, 768] | 752 | 784 | 0.98 | 1.02 | 1.04 | proved (L <= I_lo) |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=2] | 2 | 1420 | G-only: G_hat=1, F=inf | none | 752 | 752 | 752 | 1.00 | 1.00 | 1.00 | exact |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=2] | 2 | 1420 | G_hat=1 (one session per RU; F_hat=1.5) | none | 752 | 752 | 752 | 1.00 | 1.00 | 1.00 | exact |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=2] | 2 | 1420 | G_hat=2 (F_hat=1.5) | none | 752 | 752 | 752 | 1.00 | 1.00 | 1.00 | exact |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=2] | 2 | 1420 | default: F_hat=1.5, G_hat=4 | none | 752 | 752 | 752 | 1.00 | 1.00 | 1.00 | exact |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=4] | 4 | 2840 | F_hat=0.5 (session cone does not fit), G_hat=4 | F | [760, 792] | 760 | 824 | 0.96 | 1.04 | 1.08 | proved (L <= I_lo) |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=4] | 4 | 2840 | G-only: G_hat=1, F=inf | G | [760, 824] | 764 | 824 | 0.93 | 1.00 | 1.08 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=4] | 4 | 2840 | G_hat=1 (one session per RU; F_hat=1.5) | G | [760, 824] | 764 | 824 | 0.93 | 1.00 | 1.08 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=4] | 4 | 2840 | G_hat=2 (F_hat=1.5) | none | 760 | 760 | 760 | 1.00 | 1.00 | 1.00 | exact |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=4] | 4 | 2840 | default: F_hat=1.5, G_hat=4 | none | 760 | 760 | 760 | 1.00 | 1.00 | 1.00 | exact |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=8] | 8 | 5680 | F_hat=0.5 (session cone does not fit), G_hat=4 | F | [776, 840] | 776 | 904 | 0.92 | 1.08 | 1.16 | proved (L <= I_lo) |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=8] | 8 | 5680 | G-only: G_hat=1, F=inf | G | [776, 1160] | 840 | 1576 | 0.72 | 1.36 | 1.88 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=8] | 8 | 5680 | G_hat=1 (one session per RU; F_hat=1.5) | G | [776, 1160] | 840 | 1576 | 0.72 | 1.36 | 1.88 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=8] | 8 | 5680 | G_hat=2 (F_hat=1.5) | G | [776, 904] | 776 | 904 | 0.86 | 1.00 | 1.16 | proved (L <= I_lo) |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=8] | 8 | 5680 | default: F_hat=1.5, G_hat=4 | none | 776 | 776 | 776 | 1.00 | 1.00 | 1.00 | exact |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=16] | 16 | 11360 | F_hat=0.5 (session cone does not fit), G_hat=4 | F+G | [808, 1064] | 808 | 1064 | 0.76 | 1.00 | 1.32 | proved (L <= I_lo) |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=16] | 16 | 11360 | G-only: G_hat=1, F=inf | G | [808, 3016] | 1065 | 3016 | 0.35 | 1.00 | 2.83 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=16] | 16 | 11360 | G_hat=1 (one session per RU; F_hat=1.5) | G | [808, 3016] | 1065 | 3016 | 0.35 | 1.00 | 2.83 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=16] | 16 | 11360 | G_hat=2 (F_hat=1.5) | G | [808, 1576] | 936 | 1736 | 0.59 | 1.10 | 1.85 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=16] | 16 | 11360 | default: F_hat=1.5, G_hat=4 | G | [808, 1064] | 808 | 1064 | 0.76 | 1.00 | 1.32 | proved (L <= I_lo) |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=32] | 32 | 22720 | F_hat=0.5 (session cone does not fit), G_hat=4 | F+G | [872, 2056] | 1128 | 2056 | 0.55 | 1.00 | 1.82 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=32] | 32 | 22720 | G-only: G_hat=1, F=inf | G | [872, 5968] | 1642 | 5968 | 0.28 | 1.00 | 3.63 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=32] | 32 | 22720 | G_hat=1 (one session per RU; F_hat=1.5) | G | [872, 5968] | 1642 | 5968 | 0.28 | 1.00 | 3.63 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=32] | 32 | 22720 | G_hat=2 (F_hat=1.5) | G | [872, 3400] | 1333 | 3400 | 0.39 | 1.00 | 2.55 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=32] | 32 | 22720 | default: F_hat=1.5, G_hat=4 | G | [872, 2056] | 1128 | 2056 | 0.55 | 1.00 | 1.82 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=64] | 64 | 45440 | F_hat=0.5 (session cone does not fit), G_hat=4 | F+G | [1000, 4040] | 1800 | 4040 | 0.45 | 1.00 | 2.24 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=64] | 64 | 45440 | G-only: G_hat=1, F=inf | G | [1000, 11872] | 2828 | 11872 | 0.24 | 1.00 | 4.20 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=64] | 64 | 45440 | G_hat=1 (one session per RU; F_hat=1.5) | G | [1000, 11872] | 2828 | 11872 | 0.24 | 1.00 | 4.20 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=64] | 64 | 45440 | G_hat=2 (F_hat=1.5) | G | [1000, 6800] | 2210 | 6800 | 0.33 | 1.00 | 3.08 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,Q_inf=2,Q_roll=64] | 64 | 45440 | default: F_hat=1.5, G_hat=4 | G | [1000, 4040] | 1800 | 4040 | 0.45 | 1.00 | 2.24 | consistent (I_lo < L <= I_hi), not proved |
| inference-dense[tiny2,Q_inf=1,Q_roll=1] | 1 | 662 | default: F_hat=1.5, G_hat=4 | none | 4 | 4 | 4 | 1.00 | 1.00 | 1.00 | exact |
| inference-dense[tiny2,Q_inf=2,Q_roll=2] | 2 | 1420 | default: F_hat=1.5, G_hat=4 | none | 8 | 8 | 8 | 1.00 | 1.00 | 1.00 | exact |
| inference-dense[tiny2,Q_inf=4,Q_roll=4] | 4 | 3224 | default: F_hat=1.5, G_hat=4 | none | 16 | 16 | 16 | 1.00 | 1.00 | 1.00 | exact |

Steady-state marginal (bytes per rollout token; fits over the binding `Q_roll` points): `r_lo <= r* <= r_hi`, checks `r_L <= r_hi`, `r_lo <= r_U`.

| circuit | Q_inf | regime | Q (steady) | r_lo | r_hi | r_L | r_U | r_L <= r_hi | r_lo <= r_U |
|---|---|---|---|---|---|---|---|---|---|
| forward-nonfixed-moe[tiny-moe] | 4 | F_hat=0.5 (session cone does not fit), G_hat=4 | [4, 8, 16, 32, 64] | 4.00 | 88.49 | 5.73 | 534.69 | True | True |
| forward-nonfixed-moe[tiny-moe] | 4 | G-only: G_hat=1, F=inf | [8, 16, 32, 64] | 4.00 | 183.79 | 30.39 | 179.47 | True | True |
| forward-nonfixed-moe[tiny-moe] | 4 | G_hat=1 (one session per RU; F_hat=1.5) | [8, 16, 32, 64] | 4.00 | 183.79 | 30.39 | 179.47 | True | True |
| forward-nonfixed-moe[tiny-moe] | 4 | G_hat=2 (F_hat=1.5) | [16, 32, 64] | 4.00 | 87.18 | 17.85 | 84.89 | True | True |
| forward-nonfixed-moe[tiny-moe] | 4 | default: F_hat=1.5, G_hat=4 | [32, 64] | 4.00 | 45.00 | 7.28 | 44.25 | True | True |
| forward-nonfixed[tiny2] | 1 | F_hat=0.5 (session cone does not fit), G_hat=4 | [1, 2, 4, 8, 16, 32, 64] | 4.00 | 98.15 | 24.45 | 97.77 | True | True |
| forward-nonfixed[tiny2] | 1 | G-only: G_hat=1, F=inf | [2, 4, 8, 16, 32, 64] | 4.00 | 348.45 | 50.32 | 361.92 | True | True |
| forward-nonfixed[tiny2] | 1 | G_hat=1 (one session per RU; F_hat=1.5) | [2, 4, 8, 16, 32, 64] | 4.00 | 348.45 | 50.32 | 361.92 | True | True |
| forward-nonfixed[tiny2] | 1 | G_hat=2 (F_hat=1.5) | [4, 8, 16, 32, 64] | 4.00 | 186.99 | 36.40 | 184.05 | True | True |
| forward-nonfixed[tiny2] | 1 | default: F_hat=1.5, G_hat=4 | [8, 16, 32, 64] | 4.00 | 106.55 | 26.90 | 105.33 | True | True |
| forward-nonfixed[tiny2] | 2 | F_hat=0.5 (session cone does not fit), G_hat=4 | [2, 4, 8, 16, 32, 64] | 4.00 | 54.03 | 17.23 | 53.43 | True | True |
| forward-nonfixed[tiny2] | 2 | G-only: G_hat=1, F=inf | [4, 8, 16, 32, 64] | 4.00 | 186.99 | 34.99 | 184.05 | True | True |
| forward-nonfixed[tiny2] | 2 | G_hat=1 (one session per RU; F_hat=1.5) | [4, 8, 16, 32, 64] | 4.00 | 186.99 | 34.99 | 184.05 | True | True |
| forward-nonfixed[tiny2] | 2 | G_hat=2 (F_hat=1.5) | [8, 16, 32, 64] | 4.00 | 106.55 | 25.89 | 105.33 | True | True |
| forward-nonfixed[tiny2] | 2 | default: F_hat=1.5, G_hat=4 | [16, 32, 64] | 4.00 | 62.00 | 20.71 | 62.00 | True | True |

### Registry transformer cells (`local-sgd` @ tiny/tiny2, `forward-nonfixed` @ tiny2; bracketed)

- from `redteam/block_term.py` at `lower_coarse.py` sha1 `df35839fdf`; 51 cells, 9 exact (single RU legal), 42 bracketed `[I_lo, I_hi]` (660-8700 gates: beyond the exact solver).
- **`L > I_hi` (L above a legal partition's cost): 0**; `U` violations: 0; block term fired in 15/51 cells; **did not fire at 28 binding cells** (see `validation_block_term.md`).

| circuit | gates | regime | binds | I* / [I_lo, I_hi] | L | block? | U | L/ref | U/ref | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| forward-nonfixed[tiny2,q=1] | 662 | F < layer (F=layer/2), G=inf | F | [748, 796] | 748 | no | 1100 | 0.94 | 1.38 | proved (L <= I_lo) |
| forward-nonfixed[tiny2,q=1] | 662 | F=layer/2, G=layer | F+G | [748, 796] | 755 | yes | 1100 | 0.95 | 1.38 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,q=1] | 662 | G-only, G=layer (>=2 RUs along layers or tokens), F=inf | G | [748, 764] | 755 | yes | 764 | 0.99 | 1.00 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,q=1] | 662 | G-only, G=layer/2 (>=4 RUs), F=inf | G | [748, 796] | 759 | yes | 1100 | 0.95 | 1.38 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,q=1] | 662 | anchor: one RU legal | none | 748 | 748 | no | 748 | 1.00 | 1.00 | proved (L <= I_lo) |
| forward-nonfixed[tiny2,q=2] | 1420 | F < layer (F=layer/2), G=inf | F | [752, 800] | 752 | no | 1488 | 0.94 | 1.86 | proved (L <= I_lo) |
| forward-nonfixed[tiny2,q=2] | 1420 | F=layer/2, G=layer | F+G | [752, 800] | 758 | yes | 1488 | 0.95 | 1.86 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,q=2] | 1420 | G-only, G=layer (>=2 RUs along layers or tokens), F=inf | G | [752, 784] | 758 | yes | 784 | 0.97 | 1.00 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,q=2] | 1420 | G-only, G=layer/2 (>=4 RUs), F=inf | G | [752, 848] | 768 | yes | 1488 | 0.91 | 1.75 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,q=2] | 1420 | anchor: one RU legal | none | 752 | 752 | no | 752 | 1.00 | 1.00 | proved (L <= I_lo) |
| forward-nonfixed[tiny2,q=4] | 3224 | F < layer (F=layer/2), G=inf | F | [760, 792] | 760 | no | 2360 | 0.96 | 2.98 | proved (L <= I_lo) |
| forward-nonfixed[tiny2,q=4] | 3224 | F=layer/2, G=layer | F+G | [760, 856] | 760 | yes | 2360 | 0.89 | 2.76 | proved (L <= I_lo) |
| forward-nonfixed[tiny2,q=4] | 3224 | G-only, G=layer (>=2 RUs along layers or tokens), F=inf | G | [760, 824] | 760 | yes | 824 | 0.92 | 1.00 | proved (L <= I_lo) |
| forward-nonfixed[tiny2,q=4] | 3224 | G-only, G=layer/2 (>=4 RUs), F=inf | G | [760, 952] | 792 | yes | 2360 | 0.83 | 2.48 | consistent (I_lo < L <= I_hi), not proved |
| forward-nonfixed[tiny2,q=4] | 3224 | anchor: one RU legal | none | 760 | 760 | no | 760 | 1.00 | 1.00 | proved (L <= I_lo) |
| local-sgd[tiny,K=1,Q_step=1,seq=1] | 1693 | a+G: F=step/2, G=step/2 | F+G | [388, 720] | 388 | no | 1096 | 0.54 | 1.52 | proved (L <= I_lo) |
| local-sgd[tiny,K=1,Q_step=1,seq=1] | 1693 | a: F < step (F=step/2), G=inf | F | [388, 432] | 388 | no | 648 | 0.90 | 1.50 | proved (L <= I_lo) |
| local-sgd[tiny,K=1,Q_step=1,seq=1] | 1693 | anchor: one RU legal | none | 388 | 388 | no | 388 | 1.00 | 1.00 | proved (L <= I_lo) |
| local-sgd[tiny,K=1,Q_step=1,seq=1] | 1693 | c: G-only, G=step/2 (>=2 RUs), F=inf | G | [388, 720] | 388 | no | 1096 | 0.54 | 1.52 | proved (L <= I_lo) |
| local-sgd[tiny,K=2,Q_step=1,seq=1] | 3386 | a+G: F=step/2, G=step | F+G | [396, 728] | 396 | no | 1248 | 0.54 | 1.71 | proved (L <= I_lo) |
| local-sgd[tiny,K=2,Q_step=1,seq=1] | 3386 | a: F < step (F=step/2), G=inf | F | [396, 728] | 396 | no | 1248 | 0.54 | 1.71 | proved (L <= I_lo) |
| local-sgd[tiny,K=2,Q_step=1,seq=1] | 3386 | anchor: one RU legal | none | 396 | 396 | no | 396 | 1.00 | 1.00 | proved (L <= I_lo) |
| local-sgd[tiny,K=2,Q_step=1,seq=1] | 3386 | b2: F=1.5*step (just below the two-step Up: only the embeds must leave), G=inf | F | [396, 412] | 396 | no | 728 | 0.96 | 1.77 | proved (L <= I_lo) |
| local-sgd[tiny,K=2,Q_step=1,seq=1] | 3386 | b: one step fits, two do not (F=step: Up(one step) ~ step/2 < F < Up(two steps) ~ 1.5*step), G=inf | F | [396, 720] | 396 | no | 728 | 0.55 | 1.01 | proved (L <= I_lo) |
| local-sgd[tiny,K=2,Q_step=1,seq=1] | 3386 | c: G-only, G=1.5*step, F=inf | G | [396, 714] | 396 | no | 728 | 0.55 | 1.02 | proved (L <= I_lo) |
| local-sgd[tiny,K=2,Q_step=1,seq=1] | 3386 | c: G-only, G=step (>=K RUs), F=inf | G | [396, 728] | 396 | no | 728 | 0.54 | 1.00 | proved (L <= I_lo) |
| local-sgd[tiny,K=2,Q_step=2,seq=2] | 5776 | a+G: F=step/2, G=step | F+G | [412, 1296] | 412 | no | 1312 | 0.32 | 1.01 | proved (L <= I_lo) |
| local-sgd[tiny,K=2,Q_step=2,seq=2] | 5776 | a: F < step (F=step/2), G=inf | F | [412, 1296] | 412 | no | 1312 | 0.32 | 1.01 | proved (L <= I_lo) |
| local-sgd[tiny,K=2,Q_step=2,seq=2] | 5776 | anchor: one RU legal | none | 412 | 412 | no | 412 | 1.00 | 1.00 | proved (L <= I_lo) |
| local-sgd[tiny,K=2,Q_step=2,seq=2] | 5776 | b2: F=1.5*step (just below the two-step Up: only the embeds must leave), G=inf | F | [412, 744] | 412 | no | 760 | 0.55 | 1.02 | proved (L <= I_lo) |
| local-sgd[tiny,K=2,Q_step=2,seq=2] | 5776 | b: one step fits, two do not (F=step: Up(one step) ~ step/2 < F < Up(two steps) ~ 1.5*step), G=inf | F | [412, 744] | 412 | no | 760 | 0.55 | 1.02 | proved (L <= I_lo) |
| local-sgd[tiny,K=2,Q_step=2,seq=2] | 5776 | c: G-only, G=1.5*step, F=inf | G | [412, 744] | 412 | no | 760 | 0.55 | 1.02 | proved (L <= I_lo) |
| local-sgd[tiny,K=2,Q_step=2,seq=2] | 5776 | c: G-only, G=step (>=K RUs), F=inf | G | [412, 760] | 412 | no | 760 | 0.54 | 1.00 | proved (L <= I_lo) |
| local-sgd[tiny,K=3,Q_step=1,seq=1] | 5079 | a+G: F=step/2, G=step | F+G | [404, 1060] | 404 | no | 1840 | 0.38 | 1.74 | proved (L <= I_lo) |
| local-sgd[tiny,K=3,Q_step=1,seq=1] | 5079 | a: F < step (F=step/2), G=inf | F | [404, 1060] | 404 | no | 1840 | 0.38 | 1.74 | proved (L <= I_lo) |
| local-sgd[tiny,K=3,Q_step=1,seq=1] | 5079 | anchor: one RU legal | none | 404 | 404 | no | 404 | 1.00 | 1.00 | proved (L <= I_lo) |
| local-sgd[tiny,K=3,Q_step=1,seq=1] | 5079 | b2: F=1.5*step (just below the two-step Up: only the embeds must leave), G=inf | F | [404, 744] | 404 | no | 1060 | 0.54 | 1.42 | proved (L <= I_lo) |
| local-sgd[tiny,K=3,Q_step=1,seq=1] | 5079 | b: one step fits, two do not (F=step: Up(one step) ~ step/2 < F < Up(two steps) ~ 1.5*step), G=inf | F | [404, 1052] | 404 | no | 1060 | 0.38 | 1.01 | proved (L <= I_lo) |
| local-sgd[tiny,K=3,Q_step=1,seq=1] | 5079 | c: G-only, G=1.5*step, F=inf | G | [404, 978] | 404 | no | 1060 | 0.41 | 1.08 | proved (L <= I_lo) |
| local-sgd[tiny,K=3,Q_step=1,seq=1] | 5079 | c: G-only, G=step (>=K RUs), F=inf | G | [404, 1060] | 404 | no | 1060 | 0.38 | 1.00 | proved (L <= I_lo) |
| local-sgd[tiny2,K=1,Q_step=1,seq=1] | 3636 | a+G: F=step/2, G=step/2 | F+G | [756, 1404] | 756 | yes | 1404 | 0.54 | 1.00 | proved (L <= I_lo) |
| local-sgd[tiny2,K=1,Q_step=1,seq=1] | 3636 | a: F < step (F=step/2), G=inf | F | [756, 876] | 756 | yes | 1080 | 0.86 | 1.23 | proved (L <= I_lo) |
| local-sgd[tiny2,K=1,Q_step=1,seq=1] | 3636 | anchor: one RU legal | none | 756 | 756 | yes | 756 | 1.00 | 1.00 | proved (L <= I_lo) |
| local-sgd[tiny2,K=1,Q_step=1,seq=1] | 3636 | c: G-only, G=step/2 (>=2 RUs), F=inf | G | [756, 1404] | 756 | yes | 1404 | 0.54 | 1.00 | proved (L <= I_lo) |
| local-sgd[tiny2,K=2,Q_step=1,seq=1] | 7272 | a+G: F=step/2, G=step | F+G | [764, 1940] | 770 | yes | 2112 | 0.40 | 1.09 | consistent (I_lo < L <= I_hi), not proved |
| local-sgd[tiny2,K=2,Q_step=1,seq=1] | 7272 | a: F < step (F=step/2), G=inf | F | [764, 1940] | 770 | yes | 2112 | 0.40 | 1.09 | consistent (I_lo < L <= I_hi), not proved |
| local-sgd[tiny2,K=2,Q_step=1,seq=1] | 7272 | anchor: one RU legal | none | 764 | 764 | no | 764 | 1.00 | 1.00 | proved (L <= I_lo) |
| local-sgd[tiny2,K=2,Q_step=1,seq=1] | 7272 | b2: F=1.5*step (just below the two-step Up: only the embeds must leave), G=inf | F | [764, 884] | 764 | no | 1464 | 0.86 | 1.66 | proved (L <= I_lo) |
| local-sgd[tiny2,K=2,Q_step=1,seq=1] | 7272 | b: one step fits, two do not (F=step: Up(one step) ~ step/2 < F < Up(two steps) ~ 1.5*step), G=inf | F | [764, 1456] | 764 | no | 1464 | 0.52 | 1.01 | proved (L <= I_lo) |
| local-sgd[tiny2,K=2,Q_step=1,seq=1] | 7272 | c: G-only, G=1.5*step, F=inf | G | [764, 1456] | 764 | no | 1464 | 0.52 | 1.01 | proved (L <= I_lo) |
| local-sgd[tiny2,K=2,Q_step=1,seq=1] | 7272 | c: G-only, G=step (>=K RUs), F=inf | G | [764, 1464] | 764 | no | 1464 | 0.52 | 1.00 | proved (L <= I_lo) |

### Worst 5 `L/I*` cells: the attack the certificate misses

- **fuzz7-41** (swiglu|shared,view; 3 gates) F=W/2 (3) G=inf (-), binding: none: I*=6, L=0, U=-, L/I*=0.00, U/I*=-
  - L breakdown: total=0, source=0, token_seed=0, state_floor=0, cap=0, target_macs=0, F=3, G=6
    - note: fixed weights, no weight versions beyond step 0: nothing is credited (inference circuit)
    - worst_ru: {'model': {'K': 1, 'L': 0, 'Q': 0, 'd': 0, 'V': 0, 'P': [0], 'c_hop_f': [0], 'c_hop_b': [0], 'verified': [(False, False)], 'block_notes': [['block 0: no step-0 fwd ops', 'no cells']], 'dyn_roots': False, 'moe': None}, 'by_class': {'tokens': 0, 'roots': 0, 'block_prices': 0.0, 'weight_floor': 0.0, 'r
  - exact optimal partition:
    - RU0: work 6, input 6 B; ops [op0:swiglu x3]; crossing in: p0[token]=6B

- **fuzz7-41** (swiglu|shared,view; 3 gates) F=W/2 (3) G=W/2 (3), binding: G: I*=6, L=0, U=-, L/I*=0.00, U/I*=-
  - L breakdown: total=0, source=0, token_seed=0, state_floor=0, cap=0, target_macs=0, F=3, G=3
    - note: fixed weights, no weight versions beyond step 0: nothing is credited (inference circuit)
    - worst_ru: {'model': {'K': 1, 'L': 0, 'Q': 0, 'd': 0, 'V': 0, 'P': [0], 'c_hop_f': [0], 'c_hop_b': [0], 'verified': [(False, False)], 'block_notes': [['block 0: no step-0 fwd ops', 'no cells']], 'dyn_roots': False, 'moe': None}, 'by_class': {'tokens': 0, 'roots': 0, 'block_prices': 0.0, 'weight_floor': 0.0, 'r
  - exact optimal partition:
    - 3 RUs: work 2, input 2 B; ops [op0:swiglu x1]; crossing in: p0[token]=2B

- **fuzz7-41** (swiglu|shared,view; 3 gates) F=inf (-) G=W/4 (2), binding: G: I*=6, L=0, U=-, L/I*=0.00, U/I*=-
  - L breakdown: total=0, source=0, token_seed=0, state_floor=0, cap=0, target_macs=0, F=6, G=2
    - note: fixed weights, no weight versions beyond step 0: nothing is credited (inference circuit)
    - worst_ru: {'model': {'K': 1, 'L': 0, 'Q': 0, 'd': 0, 'V': 0, 'P': [0], 'c_hop_f': [0], 'c_hop_b': [0], 'verified': [(False, False)], 'block_notes': [['block 0: no step-0 fwd ops', 'no cells']], 'dyn_roots': False, 'moe': None}, 'by_class': {'tokens': 0, 'roots': 0, 'block_prices': 0.0, 'weight_floor': 0.0, 'r
  - exact optimal partition:
    - 3 RUs: work 2, input 2 B; ops [op0:swiglu x1]; crossing in: p0[token]=2B

- **fuzz7-41** (swiglu|shared,view; 3 gates) F=inf (-) G=W/2 (3), binding: G: I*=6, L=0, U=-, L/I*=0.00, U/I*=-
  - L breakdown: total=0, source=0, token_seed=0, state_floor=0, cap=0, target_macs=0, F=6, G=3
    - note: fixed weights, no weight versions beyond step 0: nothing is credited (inference circuit)
    - worst_ru: {'model': {'K': 1, 'L': 0, 'Q': 0, 'd': 0, 'V': 0, 'P': [0], 'c_hop_f': [0], 'c_hop_b': [0], 'verified': [(False, False)], 'block_notes': [['block 0: no step-0 fwd ops', 'no cells']], 'dyn_roots': False, 'moe': None}, 'by_class': {'tokens': 0, 'roots': 0, 'block_prices': 0.0, 'weight_floor': 0.0, 'r
  - exact optimal partition:
    - 3 RUs: work 2, input 2 B; ops [op0:swiglu x1]; crossing in: p0[token]=2B

- **fuzz7-111** (colsum|view; 12 gates) F=W/4 (3) G=inf (-), binding: F: I*=16, L=0, U=-, L/I*=0.00, U/I*=-
  - L breakdown: total=0, source=0, token_seed=0, state_floor=0, cap=0, target_macs=0, F=3, G=10
    - note: fixed weights, no weight versions beyond step 0: nothing is credited (inference circuit)
    - worst_ru: {'model': {'K': 1, 'L': 0, 'Q': 0, 'd': 0, 'V': 0, 'P': [0], 'c_hop_f': [0], 'c_hop_b': [0], 'verified': [(False, False)], 'block_notes': [['block 0: no step-0 fwd ops', 'no cells']], 'dyn_roots': False, 'moe': None}, 'by_class': {'tokens': 0, 'roots': 0, 'block_prices': 0.0, 'weight_floor': 0.0, 'r
  - exact optimal partition:
    - RU0: work 4, input 4 B; ops [op0:colsum x6]; crossing in: p0[carried]=4B
    - RU1: work 6, input 12 B; ops [op0:colsum x6]; crossing in: op0:colsum.out=8B, p0[carried]=4B

## Per-group aggregates

| group | cells | violations | median L/I* | p05 L/I* | median U/I* | p95 U/I* | U n/a |
|---|---|---|---|---|---|---|---|
| multi-step micro suite | 192 | 0 | 0.58 | 0.25 | 1.00 | 2.00 | 18 |
| registry inference (tiny/tiny2) | 72 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 0 |
| forward-nonfixed micro (accumulated weights) | 36 | 0 | 1.00 | 0.67 | 1.00 | 1.45 | 0 |
| rollout micro analogues (dense / MoE, accumulated weights) | 45 | 0 | 0.67 | 0.39 | 1.00 | 1.45 | 5 |
| registry forward-nonfixed @ tiny | 24 | 0 | 1.00 | 1.00 | 1.00 | 1.00 | 0 |
| classic micro suite | 288 | 0 | 0.67 | 0.42 | 1.00 | 1.50 | 157 |
| torture cases | 315 | 0 | 0.68 | 0.31 | 1.00 | 2.00 | 178 |
| fuzz | 800 | 0 | 1.00 | 0.25 | 1.00 | 2.50 | 349 |

## Table

### multi-step micro suite

| circuit | gates | F | G | I* | L | U (lit) | L/I* | U/I* | flags |
|---|---|---|---|---|---|---|---|---|---|
| micro_inference[q=1,d=2] | 8 | 1·fwd (6) | 1·fwd (6) | 4 | 4 | 4 | 1.00 | 1.00 |  |
| micro_inference[q=1,d=2] | 8 | 1·fwd (6) | 4·fwd (24) | 4 | 4 | 4 | 1.00 | 1.00 |  |
| micro_inference[q=1,d=2] | 8 | 1·fwd (6) | inf (-) | 4 | 4 | 4 | 1.00 | 1.00 |  |
| micro_inference[q=1,d=2] | 8 | 1.5·fwd (9) | 1·fwd (6) | 4 | 4 | 4 | 1.00 | 1.00 |  |
| micro_inference[q=1,d=2] | 8 | 1.5·fwd (9) | 4·fwd (24) | 4 | 4 | 4 | 1.00 | 1.00 |  |
| micro_inference[q=1,d=2] | 8 | 1.5·fwd (9) | inf (-) | 4 | 4 | 4 | 1.00 | 1.00 |  |
| micro_inference[q=1,d=2] | 8 | 3·fwd (18) | 1·fwd (6) | 4 | 4 | 4 | 1.00 | 1.00 |  |
| micro_inference[q=1,d=2] | 8 | 3·fwd (18) | 4·fwd (24) | 4 | 4 | 4 | 1.00 | 1.00 |  |
| micro_inference[q=1,d=2] | 8 | 3·fwd (18) | inf (-) | 4 | 4 | 4 | 1.00 | 1.00 |  |
| micro_inference[q=1,d=2] | 8 | inf (-) | 1·fwd (6) | 4 | 4 | 4 | 1.00 | 1.00 |  |
| micro_inference[q=1,d=2] | 8 | inf (-) | 4·fwd (24) | 4 | 4 | 4 | 1.00 | 1.00 |  |
| micro_inference[q=1,d=2] | 8 | inf (-) | inf (-) | 4 | 4 | 4 | 1.00 | 1.00 |  |
| micro_inference[q=2,d=2] | 16 | 1·fwd (12) | 1·fwd (12) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| micro_inference[q=2,d=2] | 16 | 1·fwd (12) | 4·fwd (48) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| micro_inference[q=2,d=2] | 16 | 1·fwd (12) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| micro_inference[q=2,d=2] | 16 | 1.5·fwd (18) | 1·fwd (12) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| micro_inference[q=2,d=2] | 16 | 1.5·fwd (18) | 4·fwd (48) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| micro_inference[q=2,d=2] | 16 | 1.5·fwd (18) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| micro_inference[q=2,d=2] | 16 | 3·fwd (36) | 1·fwd (12) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| micro_inference[q=2,d=2] | 16 | 3·fwd (36) | 4·fwd (48) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| micro_inference[q=2,d=2] | 16 | 3·fwd (36) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| micro_inference[q=2,d=2] | 16 | inf (-) | 1·fwd (12) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| micro_inference[q=2,d=2] | 16 | inf (-) | 4·fwd (48) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| micro_inference[q=2,d=2] | 16 | inf (-) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| local_sgd_chain[K=1,q=1,d=1] | 9 | 1·fwd (2) | 1·fwd (2) | 20 | 10 | 20 | 0.50 | 1.00 |  |
| local_sgd_chain[K=1,q=1,d=1] | 9 | 1·fwd (2) | 4·fwd (8) | 20 | 10 | 20 | 0.50 | 1.00 |  |
| local_sgd_chain[K=1,q=1,d=1] | 9 | 1·fwd (2) | inf (-) | 20 | 10 | 20 | 0.50 | 1.00 |  |
| local_sgd_chain[K=1,q=1,d=1] | 9 | 1.5·fwd (3) | 1·fwd (2) | 20 | 10 | 20 | 0.50 | 1.00 |  |
| local_sgd_chain[K=1,q=1,d=1] | 9 | 1.5·fwd (3) | 4·fwd (8) | 18 | 10 | 18 | 0.56 | 1.00 |  |
| local_sgd_chain[K=1,q=1,d=1] | 9 | 1.5·fwd (3) | inf (-) | 18 | 10 | 18 | 0.56 | 1.00 |  |
| local_sgd_chain[K=1,q=1,d=1] | 9 | 3·fwd (6) | 1·fwd (2) | 20 | 10 | 20 | 0.50 | 1.00 |  |
| local_sgd_chain[K=1,q=1,d=1] | 9 | 3·fwd (6) | 4·fwd (8) | 14 | 10 | 14 | 0.71 | 1.00 |  |
| local_sgd_chain[K=1,q=1,d=1] | 9 | 3·fwd (6) | inf (-) | 14 | 10 | 14 | 0.71 | 1.00 |  |
| local_sgd_chain[K=1,q=1,d=1] | 9 | inf (-) | 1·fwd (2) | 20 | 10 | 20 | 0.50 | 1.00 |  |
| local_sgd_chain[K=1,q=1,d=1] | 9 | inf (-) | 4·fwd (8) | 10 | 10 | 10 | 1.00 | 1.00 |  |
| local_sgd_chain[K=1,q=1,d=1] | 9 | inf (-) | inf (-) | 10 | 10 | 10 | 1.00 | 1.00 |  |
| local_sgd_chain[K=2,q=1,d=1] | 18 | 1·fwd (2) | 1·fwd (2) | 40 | 14 | 40 | 0.35 | 1.00 |  |
| local_sgd_chain[K=2,q=1,d=1] | 18 | 1·fwd (2) | 4·fwd (8) | 40 | 14 | 40 | 0.35 | 1.00 |  |
| local_sgd_chain[K=2,q=1,d=1] | 18 | 1·fwd (2) | inf (-) | 40 | 14 | 40 | 0.35 | 1.00 |  |
| local_sgd_chain[K=2,q=1,d=1] | 18 | 1.5·fwd (3) | 1·fwd (2) | 40 | 14 | 40 | 0.35 | 1.00 |  |
| local_sgd_chain[K=2,q=1,d=1] | 18 | 1.5·fwd (3) | 4·fwd (8) | 34 | 14 | 36 | 0.41 | 1.06 |  |
| local_sgd_chain[K=2,q=1,d=1] | 18 | 1.5·fwd (3) | inf (-) | 34 | 14 | 36 | 0.41 | 1.06 |  |
| local_sgd_chain[K=2,q=1,d=1] | 18 | 3·fwd (6) | 1·fwd (2) | 40 | 14 | 40 | 0.35 | 1.00 |  |
| local_sgd_chain[K=2,q=1,d=1] | 18 | 3·fwd (6) | 4·fwd (8) | 26 | 14 | 28 | 0.54 | 1.08 |  |
| local_sgd_chain[K=2,q=1,d=1] | 18 | 3·fwd (6) | inf (-) | 26 | 14 | 28 | 0.54 | 1.08 |  |
| local_sgd_chain[K=2,q=1,d=1] | 18 | inf (-) | 1·fwd (2) | 40 | 14 | 40 | 0.35 | 1.00 |  |
| local_sgd_chain[K=2,q=1,d=1] | 18 | inf (-) | 4·fwd (8) | 20 | 14 | 20 | 0.70 | 1.00 |  |
| local_sgd_chain[K=2,q=1,d=1] | 18 | inf (-) | inf (-) | 14 | 14 | 14 | 1.00 | 1.00 |  |
| local_sgd_chain[K=3,q=1,d=1] | 27 | 1·fwd (2) | 1·fwd (2) | 60 | 18 | 60 | 0.30 | 1.00 |  |
| local_sgd_chain[K=3,q=1,d=1] | 27 | 1·fwd (2) | 4·fwd (8) | 60 | 18 | 60 | 0.30 | 1.00 |  |
| local_sgd_chain[K=3,q=1,d=1] | 27 | 1·fwd (2) | inf (-) | 60 | 18 | 60 | 0.30 | 1.00 |  |
| local_sgd_chain[K=3,q=1,d=1] | 27 | 1.5·fwd (3) | 1·fwd (2) | 60 | 18 | 60 | 0.30 | 1.00 |  |
| local_sgd_chain[K=3,q=1,d=1] | 27 | 1.5·fwd (3) | 4·fwd (8) | 52 | 18 | 54 | 0.35 | 1.04 |  |
| local_sgd_chain[K=3,q=1,d=1] | 27 | 1.5·fwd (3) | inf (-) | 52 | 18 | 54 | 0.35 | 1.04 |  |
| local_sgd_chain[K=3,q=1,d=1] | 27 | 3·fwd (6) | 1·fwd (2) | 60 | 18 | 60 | 0.30 | 1.00 |  |
| local_sgd_chain[K=3,q=1,d=1] | 27 | 3·fwd (6) | 4·fwd (8) | 40 | 18 | 42 | 0.45 | 1.05 |  |
| local_sgd_chain[K=3,q=1,d=1] | 27 | 3·fwd (6) | inf (-) | 40 | 18 | 42 | 0.45 | 1.05 |  |
| local_sgd_chain[K=3,q=1,d=1] | 27 | inf (-) | 1·fwd (2) | 60 | 18 | 60 | 0.30 | 1.00 |  |
| local_sgd_chain[K=3,q=1,d=1] | 27 | inf (-) | 4·fwd (8) | 30 | 18 | 30 | 0.60 | 1.00 |  |
| local_sgd_chain[K=3,q=1,d=1] | 27 | inf (-) | inf (-) | 18 | 18 | 18 | 1.00 | 1.00 |  |
| local_sgd_chain[K=1,q=2,d=1] | 14 | 1·fwd (4) | 1·fwd (4) | 26 | 14 | 30 | 0.54 | 1.15 |  |
| local_sgd_chain[K=1,q=2,d=1] | 14 | 1·fwd (4) | 4·fwd (16) | 24 | 14 | 30 | 0.58 | 1.25 |  |
| local_sgd_chain[K=1,q=2,d=1] | 14 | 1·fwd (4) | inf (-) | 24 | 14 | 30 | 0.58 | 1.25 |  |
| local_sgd_chain[K=1,q=2,d=1] | 14 | 1.5·fwd (6) | 1·fwd (4) | 26 | 14 | 30 | 0.54 | 1.15 |  |
| local_sgd_chain[K=1,q=2,d=1] | 14 | 1.5·fwd (6) | 4·fwd (16) | 24 | 14 | 24 | 0.58 | 1.00 |  |
| local_sgd_chain[K=1,q=2,d=1] | 14 | 1.5·fwd (6) | inf (-) | 24 | 14 | 24 | 0.58 | 1.00 |  |
| local_sgd_chain[K=1,q=2,d=1] | 14 | 3·fwd (12) | 1·fwd (4) | 26 | 14 | 30 | 0.54 | 1.15 |  |
| local_sgd_chain[K=1,q=2,d=1] | 14 | 3·fwd (12) | 4·fwd (16) | 14 | 14 | 14 | 1.00 | 1.00 |  |
| local_sgd_chain[K=1,q=2,d=1] | 14 | 3·fwd (12) | inf (-) | 14 | 14 | 14 | 1.00 | 1.00 |  |
| local_sgd_chain[K=1,q=2,d=1] | 14 | inf (-) | 1·fwd (4) | 26 | 14 | 30 | 0.54 | 1.15 |  |
| local_sgd_chain[K=1,q=2,d=1] | 14 | inf (-) | 4·fwd (16) | 14 | 14 | 14 | 1.00 | 1.00 |  |
| local_sgd_chain[K=1,q=2,d=1] | 14 | inf (-) | inf (-) | 14 | 14 | 14 | 1.00 | 1.00 |  |
| local_sgd_chain[K=2,q=2,d=1] | 28 | 1·fwd (4) | 1·fwd (4) | 52 | 22 | 60 | 0.42 | 1.15 |  |
| local_sgd_chain[K=2,q=2,d=1] | 28 | 1·fwd (4) | 4·fwd (16) | 48 | 22 | 60 | 0.46 | 1.25 |  |
| local_sgd_chain[K=2,q=2,d=1] | 28 | 1·fwd (4) | inf (-) | 48 | 22 | 60 | 0.46 | 1.25 |  |
| local_sgd_chain[K=2,q=2,d=1] | 28 | 1.5·fwd (6) | 1·fwd (4) | 52 | 22 | 60 | 0.42 | 1.15 |  |
| local_sgd_chain[K=2,q=2,d=1] | 28 | 1.5·fwd (6) | 4·fwd (16) | 46 | 22 | 48 | 0.48 | 1.04 |  |
| local_sgd_chain[K=2,q=2,d=1] | 28 | 1.5·fwd (6) | inf (-) | 46 | 22 | 48 | 0.48 | 1.04 |  |
| local_sgd_chain[K=2,q=2,d=1] | 28 | 3·fwd (12) | 1·fwd (4) | 52 | 22 | 60 | 0.42 | 1.15 |  |
| local_sgd_chain[K=2,q=2,d=1] | 28 | 3·fwd (12) | 4·fwd (16) | 28 | 22 | 28 | 0.79 | 1.00 |  |
| local_sgd_chain[K=2,q=2,d=1] | 28 | 3·fwd (12) | inf (-) | 28 | 22 | 28 | 0.79 | 1.00 |  |
| local_sgd_chain[K=2,q=2,d=1] | 28 | inf (-) | 1·fwd (4) | 52 | 22 | 60 | 0.42 | 1.15 |  |
| local_sgd_chain[K=2,q=2,d=1] | 28 | inf (-) | 4·fwd (16) | 26 | 22 | 26 | 0.85 | 1.00 |  |
| local_sgd_chain[K=2,q=2,d=1] | 28 | inf (-) | inf (-) | 22 | 22 | 22 | 1.00 | 1.00 |  |
| local_sgd_chain[K=3,q=2,d=1] | 42 | 1·fwd (4) | 1·fwd (4) | 78 | 30 | 90 | 0.38 | 1.15 |  |
| local_sgd_chain[K=3,q=2,d=1] | 42 | 1·fwd (4) | 4·fwd (16) | 72 | 30 | 90 | 0.42 | 1.25 |  |
| local_sgd_chain[K=3,q=2,d=1] | 42 | 1·fwd (4) | inf (-) | 72 | 30 | 90 | 0.42 | 1.25 |  |
| local_sgd_chain[K=3,q=2,d=1] | 42 | 1.5·fwd (6) | 1·fwd (4) | 78 | 30 | 90 | 0.38 | 1.15 |  |
| local_sgd_chain[K=3,q=2,d=1] | 42 | 1.5·fwd (6) | 4·fwd (16) | 68 | 30 | 72 | 0.44 | 1.06 |  |
| local_sgd_chain[K=3,q=2,d=1] | 42 | 1.5·fwd (6) | inf (-) | 68 | 30 | 72 | 0.44 | 1.06 |  |
| local_sgd_chain[K=3,q=2,d=1] | 42 | 3·fwd (12) | 1·fwd (4) | 78 | 30 | 90 | 0.38 | 1.15 |  |
| local_sgd_chain[K=3,q=2,d=1] | 42 | 3·fwd (12) | 4·fwd (16) | 42 | 30 | 42 | 0.71 | 1.00 |  |
| local_sgd_chain[K=3,q=2,d=1] | 42 | 3·fwd (12) | inf (-) | 42 | 30 | 42 | 0.71 | 1.00 |  |
| local_sgd_chain[K=3,q=2,d=1] | 42 | inf (-) | 1·fwd (4) | 78 | 30 | 90 | 0.38 | 1.15 |  |
| local_sgd_chain[K=3,q=2,d=1] | 42 | inf (-) | 4·fwd (16) | 40 | 30 | 40 | 0.75 | 1.00 |  |
| local_sgd_chain[K=3,q=2,d=1] | 42 | inf (-) | inf (-) | 30 | 30 | 30 | 1.00 | 1.00 |  |
| local_sgd_chain[K=1,q=1,d=2] | 30 | 1·fwd (6) | 1·fwd (6) | 48 | 20 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.42 | - |  |
| local_sgd_chain[K=1,q=1,d=2] | 30 | 1·fwd (6) | 4·fwd (24) | 36 | 20 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.56 | - |  |
| local_sgd_chain[K=1,q=1,d=2] | 30 | 1·fwd (6) | inf (-) | 36 | 20 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.56 | - |  |
| local_sgd_chain[K=1,q=1,d=2] | 30 | 1.5·fwd (9) | 1·fwd (6) | 48 | 20 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.42 | - |  |
| local_sgd_chain[K=1,q=1,d=2] | 30 | 1.5·fwd (9) | 4·fwd (24) | 20 | 20 | 44 | 1.00 | 2.20 |  |
| local_sgd_chain[K=1,q=1,d=2] | 30 | 1.5·fwd (9) | inf (-) | 20 | 20 | 44 | 1.00 | 2.20 |  |
| local_sgd_chain[K=1,q=1,d=2] | 30 | 3·fwd (18) | 1·fwd (6) | 48 | 20 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.42 | - |  |
| local_sgd_chain[K=1,q=1,d=2] | 30 | 3·fwd (18) | 4·fwd (24) | 20 | 20 | 36 | 1.00 | 1.80 |  |
| local_sgd_chain[K=1,q=1,d=2] | 30 | 3·fwd (18) | inf (-) | 20 | 20 | 36 | 1.00 | 1.80 |  |
| local_sgd_chain[K=1,q=1,d=2] | 30 | inf (-) | 1·fwd (6) | 48 | 20 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.42 | - |  |
| local_sgd_chain[K=1,q=1,d=2] | 30 | inf (-) | 4·fwd (24) | 20 | 20 | 20 | 1.00 | 1.00 |  |
| local_sgd_chain[K=1,q=1,d=2] | 30 | inf (-) | inf (-) | 20 | 20 | 20 | 1.00 | 1.00 |  |
| local_sgd_chain[K=2,q=1,d=2] | 60 | 1·fwd (6) | 1·fwd (6) | [84, 98] | 28 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.29 | - |  |
| local_sgd_chain[K=2,q=1,d=2] | 60 | 1·fwd (6) | 4·fwd (24) | 68 | 28 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.41 | - |  |
| local_sgd_chain[K=2,q=1,d=2] | 60 | 1·fwd (6) | inf (-) | 68 | 28 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.41 | - |  |
| local_sgd_chain[K=2,q=1,d=2] | 60 | 1.5·fwd (9) | 1·fwd (6) | [84, 98] | 28 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.29 | - |  |
| local_sgd_chain[K=2,q=1,d=2] | 60 | 1.5·fwd (9) | 4·fwd (24) | 40 | 28 | 88 | 0.70 | 2.20 |  |
| local_sgd_chain[K=2,q=1,d=2] | 60 | 1.5·fwd (9) | inf (-) | 40 | 28 | 88 | 0.70 | 2.20 |  |
| local_sgd_chain[K=2,q=1,d=2] | 60 | 3·fwd (18) | 1·fwd (6) | [84, 98] | 28 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.29 | - |  |
| local_sgd_chain[K=2,q=1,d=2] | 60 | 3·fwd (18) | 4·fwd (24) | 40 | 28 | 64 | 0.70 | 1.60 |  |
| local_sgd_chain[K=2,q=1,d=2] | 60 | 3·fwd (18) | inf (-) | 40 | 28 | 64 | 0.70 | 1.60 |  |
| local_sgd_chain[K=2,q=1,d=2] | 60 | inf (-) | 1·fwd (6) | [84, 98] | 28 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.29 | - |  |
| local_sgd_chain[K=2,q=1,d=2] | 60 | inf (-) | 4·fwd (24) | 40 | 28 | 40 | 0.70 | 1.00 |  |
| local_sgd_chain[K=2,q=1,d=2] | 60 | inf (-) | inf (-) | 28 | 28 | 28 | 1.00 | 1.00 |  |
| local_sgd_chain[K=3,q=1,d=2] | 90 | 1·fwd (6) | 1·fwd (6) | [36, 148] | 36 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.24 | - |  |
| local_sgd_chain[K=3,q=1,d=2] | 90 | 1·fwd (6) | 4·fwd (24) | [36, 108] | 36 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.33 | - |  |
| local_sgd_chain[K=3,q=1,d=2] | 90 | 1·fwd (6) | inf (-) | [36, 108] | 36 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.33 | - |  |
| local_sgd_chain[K=3,q=1,d=2] | 90 | 1.5·fwd (9) | 1·fwd (6) | [36, 148] | 36 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.24 | - |  |
| local_sgd_chain[K=3,q=1,d=2] | 90 | 1.5·fwd (9) | 4·fwd (24) | [36, 60] | 36 | 132 | 0.60 | 2.20 |  |
| local_sgd_chain[K=3,q=1,d=2] | 90 | 1.5·fwd (9) | inf (-) | [36, 60] | 36 | 132 | 0.60 | 2.20 |  |
| local_sgd_chain[K=3,q=1,d=2] | 90 | 3·fwd (18) | 1·fwd (6) | [36, 148] | 36 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.24 | - |  |
| local_sgd_chain[K=3,q=1,d=2] | 90 | 3·fwd (18) | 4·fwd (24) | [36, 60] | 36 | 100 | 0.60 | 1.67 |  |
| local_sgd_chain[K=3,q=1,d=2] | 90 | 3·fwd (18) | inf (-) | [36, 56] | 36 | 100 | 0.64 | 1.79 |  |
| local_sgd_chain[K=3,q=1,d=2] | 90 | inf (-) | 1·fwd (6) | [36, 148] | 36 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.24 | - |  |
| local_sgd_chain[K=3,q=1,d=2] | 90 | inf (-) | 4·fwd (24) | [36, 60] | 36 | 60 | 0.60 | 1.00 |  |
| local_sgd_chain[K=3,q=1,d=2] | 90 | inf (-) | inf (-) | 36 | 36 | 36 | 1.00 | 1.00 |  |
| local_sgd_chain[K=1,q=2,d=2] | 44 | 1·fwd (12) | 1·fwd (12) | 60 | 28 | 68 | 0.47 | 1.13 |  |
| local_sgd_chain[K=1,q=2,d=2] | 44 | 1·fwd (12) | 4·fwd (48) | 42 | 28 | 68 | 0.67 | 1.62 |  |
| local_sgd_chain[K=1,q=2,d=2] | 44 | 1·fwd (12) | inf (-) | 42 | 28 | 68 | 0.67 | 1.62 |  |
| local_sgd_chain[K=1,q=2,d=2] | 44 | 1.5·fwd (18) | 1·fwd (12) | 60 | 28 | 68 | 0.47 | 1.13 |  |
| local_sgd_chain[K=1,q=2,d=2] | 44 | 1.5·fwd (18) | 4·fwd (48) | 28 | 28 | 60 | 1.00 | 2.14 |  |
| local_sgd_chain[K=1,q=2,d=2] | 44 | 1.5·fwd (18) | inf (-) | 28 | 28 | 60 | 1.00 | 2.14 |  |
| local_sgd_chain[K=1,q=2,d=2] | 44 | 3·fwd (36) | 1·fwd (12) | 60 | 28 | 68 | 0.47 | 1.13 |  |
| local_sgd_chain[K=1,q=2,d=2] | 44 | 3·fwd (36) | 4·fwd (48) | 28 | 28 | 28 | 1.00 | 1.00 |  |
| local_sgd_chain[K=1,q=2,d=2] | 44 | 3·fwd (36) | inf (-) | 28 | 28 | 28 | 1.00 | 1.00 |  |
| local_sgd_chain[K=1,q=2,d=2] | 44 | inf (-) | 1·fwd (12) | 60 | 28 | 68 | 0.47 | 1.13 |  |
| local_sgd_chain[K=1,q=2,d=2] | 44 | inf (-) | 4·fwd (48) | 28 | 28 | 28 | 1.00 | 1.00 |  |
| local_sgd_chain[K=1,q=2,d=2] | 44 | inf (-) | inf (-) | 28 | 28 | 28 | 1.00 | 1.00 |  |
| local_sgd_chain[K=2,q=2,d=2] | 88 | 1·fwd (12) | 1·fwd (12) | [44, 136] | 44 | 136 | 0.32 | 1.00 |  |
| local_sgd_chain[K=2,q=2,d=2] | 88 | 1·fwd (12) | 4·fwd (48) | [44, 88] | 44 | 136 | 0.50 | 1.55 |  |
| local_sgd_chain[K=2,q=2,d=2] | 88 | 1·fwd (12) | inf (-) | [44, 88] | 44 | 136 | 0.50 | 1.55 |  |
| local_sgd_chain[K=2,q=2,d=2] | 88 | 1.5·fwd (18) | 1·fwd (12) | [44, 136] | 44 | 136 | 0.32 | 1.00 |  |
| local_sgd_chain[K=2,q=2,d=2] | 88 | 1.5·fwd (18) | 4·fwd (48) | [44, 56] | 44 | 120 | 0.79 | 2.14 |  |
| local_sgd_chain[K=2,q=2,d=2] | 88 | 1.5·fwd (18) | inf (-) | [44, 56] | 44 | 120 | 0.79 | 2.14 |  |
| local_sgd_chain[K=2,q=2,d=2] | 88 | 3·fwd (36) | 1·fwd (12) | [44, 136] | 44 | 136 | 0.32 | 1.00 |  |
| local_sgd_chain[K=2,q=2,d=2] | 88 | 3·fwd (36) | 4·fwd (48) | [44, 56] | 44 | 56 | 0.79 | 1.00 |  |
| local_sgd_chain[K=2,q=2,d=2] | 88 | 3·fwd (36) | inf (-) | 44 | 44 | 56 | 1.00 | 1.27 |  |
| local_sgd_chain[K=2,q=2,d=2] | 88 | inf (-) | 1·fwd (12) | [44, 136] | 44 | 136 | 0.32 | 1.00 |  |
| local_sgd_chain[K=2,q=2,d=2] | 88 | inf (-) | 4·fwd (48) | [44, 56] | 44 | 56 | 0.79 | 1.00 |  |
| local_sgd_chain[K=2,q=2,d=2] | 88 | inf (-) | inf (-) | 44 | 44 | 44 | 1.00 | 1.00 |  |
| fanout_shared_weight[n=4,q=1,d=2] | 40 | 1·fwd (6) | 1·fwd (6) | [56, 72] | 36 | 72 | 0.50 | 1.00 |  |
| fanout_shared_weight[n=4,q=1,d=2] | 40 | 1·fwd (6) | 4·fwd (24) | 44 | 36 | 72 | 0.82 | 1.64 |  |
| fanout_shared_weight[n=4,q=1,d=2] | 40 | 1·fwd (6) | inf (-) | 44 | 36 | 72 | 0.82 | 1.64 |  |
| fanout_shared_weight[n=4,q=1,d=2] | 40 | 1.5·fwd (9) | 1·fwd (6) | [56, 72] | 36 | 72 | 0.50 | 1.00 |  |
| fanout_shared_weight[n=4,q=1,d=2] | 40 | 1.5·fwd (9) | 4·fwd (24) | 44 | 36 | 44 | 0.82 | 1.00 |  |
| fanout_shared_weight[n=4,q=1,d=2] | 40 | 1.5·fwd (9) | inf (-) | 36 | 36 | 44 | 1.00 | 1.22 |  |
| fanout_shared_weight[n=4,q=1,d=2] | 40 | 3·fwd (18) | 1·fwd (6) | [56, 72] | 36 | 72 | 0.50 | 1.00 |  |
| fanout_shared_weight[n=4,q=1,d=2] | 40 | 3·fwd (18) | 4·fwd (24) | 44 | 36 | 44 | 0.82 | 1.00 |  |
| fanout_shared_weight[n=4,q=1,d=2] | 40 | 3·fwd (18) | inf (-) | 36 | 36 | 36 | 1.00 | 1.00 |  |
| fanout_shared_weight[n=4,q=1,d=2] | 40 | inf (-) | 1·fwd (6) | [56, 72] | 36 | 72 | 0.50 | 1.00 |  |
| fanout_shared_weight[n=4,q=1,d=2] | 40 | inf (-) | 4·fwd (24) | 44 | 36 | 44 | 0.82 | 1.00 |  |
| fanout_shared_weight[n=4,q=1,d=2] | 40 | inf (-) | inf (-) | 36 | 36 | 36 | 1.00 | 1.00 |  |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 32 | 1·fwd (6) | 1·fwd (6) | 16 | 4 | 16 | 0.25 | 1.00 |  |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 32 | 1·fwd (6) | 4·fwd (24) | 16 | 4 | 16 | 0.25 | 1.00 |  |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 32 | 1·fwd (6) | inf (-) | 16 | 4 | 16 | 0.25 | 1.00 |  |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 32 | 1.5·fwd (9) | 1·fwd (6) | 16 | 4 | 16 | 0.25 | 1.00 |  |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 32 | 1.5·fwd (9) | 4·fwd (24) | 8 | 4 | 16 | 0.50 | 2.00 |  |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 32 | 1.5·fwd (9) | inf (-) | 8 | 4 | 16 | 0.50 | 2.00 |  |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 32 | 3·fwd (18) | 1·fwd (6) | 16 | 4 | 16 | 0.25 | 1.00 |  |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 32 | 3·fwd (18) | 4·fwd (24) | 8 | 4 | 8 | 0.50 | 1.00 |  |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 32 | 3·fwd (18) | inf (-) | 8 | 4 | 8 | 0.50 | 1.00 |  |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 32 | inf (-) | 1·fwd (6) | 16 | 4 | 16 | 0.25 | 1.00 |  |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 32 | inf (-) | 4·fwd (24) | 4 | 4 | 4 | 1.00 | 1.00 |  |
| deep_chain[depth=4,q=1,d=2,W=fixed] | 32 | inf (-) | inf (-) | 4 | 4 | 4 | 1.00 | 1.00 |  |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 48 | 1·fwd (6) | 1·fwd (6) | 24 | 4 | 24 | 0.17 | 1.00 |  |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 48 | 1·fwd (6) | 4·fwd (24) | 24 | 4 | 24 | 0.17 | 1.00 |  |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 48 | 1·fwd (6) | inf (-) | 24 | 4 | 24 | 0.17 | 1.00 |  |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 48 | 1.5·fwd (9) | 1·fwd (6) | 24 | 4 | 24 | 0.17 | 1.00 |  |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 48 | 1.5·fwd (9) | 4·fwd (24) | 12 | 4 | 24 | 0.33 | 2.00 |  |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 48 | 1.5·fwd (9) | inf (-) | 12 | 4 | 24 | 0.33 | 2.00 |  |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 48 | 3·fwd (18) | 1·fwd (6) | 24 | 4 | 24 | 0.17 | 1.00 |  |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 48 | 3·fwd (18) | 4·fwd (24) | 8 | 4 | 8 | 0.50 | 1.00 |  |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 48 | 3·fwd (18) | inf (-) | 8 | 4 | 8 | 0.50 | 1.00 |  |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 48 | inf (-) | 1·fwd (6) | 24 | 4 | 24 | 0.17 | 1.00 |  |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 48 | inf (-) | 4·fwd (24) | 8 | 4 | 8 | 0.50 | 1.00 |  |
| deep_chain[depth=6,q=1,d=2,W=fixed] | 48 | inf (-) | inf (-) | 4 | 4 | 4 | 1.00 | 1.00 |  |

### registry inference (tiny/tiny2)

| circuit | gates | F | G | I* | L | U (lit) | L/I* | U/I* | flags |
|---|---|---|---|---|---|---|---|---|---|
| inference-dense | 656 | 1·fwd (592) | 1·fwd (592) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 656 | 1·fwd (592) | 4·fwd (2368) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 656 | 1·fwd (592) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 656 | 1.5·fwd (888) | 1·fwd (592) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 656 | 1.5·fwd (888) | 4·fwd (2368) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 656 | 1.5·fwd (888) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 656 | 3·fwd (1776) | 1·fwd (592) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 656 | 3·fwd (1776) | 4·fwd (2368) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 656 | 3·fwd (1776) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 656 | inf (-) | 1·fwd (592) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 656 | inf (-) | 4·fwd (2368) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 656 | inf (-) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 1032 | 1·fwd (936) | 1·fwd (936) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 1032 | 1·fwd (936) | 4·fwd (3744) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 1032 | 1·fwd (936) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 1032 | 1.5·fwd (1404) | 1·fwd (936) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 1032 | 1.5·fwd (1404) | 4·fwd (3744) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 1032 | 1.5·fwd (1404) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 1032 | 3·fwd (2808) | 1·fwd (936) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 1032 | 3·fwd (2808) | 4·fwd (3744) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 1032 | 3·fwd (2808) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 1032 | inf (-) | 1·fwd (936) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 1032 | inf (-) | 4·fwd (3744) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 1032 | inf (-) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 1440 | 1·fwd (1312) | 1·fwd (1312) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 1440 | 1·fwd (1312) | 4·fwd (5248) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 1440 | 1·fwd (1312) | inf (-) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 1440 | 1.5·fwd (1968) | 1·fwd (1312) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 1440 | 1.5·fwd (1968) | 4·fwd (5248) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 1440 | 1.5·fwd (1968) | inf (-) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 1440 | 3·fwd (3936) | 1·fwd (1312) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 1440 | 3·fwd (3936) | 4·fwd (5248) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 1440 | 3·fwd (3936) | inf (-) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 1440 | inf (-) | 1·fwd (1312) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 1440 | inf (-) | 4·fwd (5248) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 1440 | inf (-) | inf (-) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 1420 | 1·fwd (1290) | 1·fwd (1290) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 1420 | 1·fwd (1290) | 4·fwd (5160) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 1420 | 1·fwd (1290) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 1420 | 1.5·fwd (1935) | 1·fwd (1290) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 1420 | 1.5·fwd (1935) | 4·fwd (5160) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 1420 | 1.5·fwd (1935) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 1420 | 3·fwd (3870) | 1·fwd (1290) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 1420 | 3·fwd (3870) | 4·fwd (5160) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 1420 | 3·fwd (3870) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 1420 | inf (-) | 1·fwd (1290) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 1420 | inf (-) | 4·fwd (5160) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 1420 | inf (-) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| inference-dense | 2274 | 1·fwd (2079) | 1·fwd (2079) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 2274 | 1·fwd (2079) | 4·fwd (8316) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 2274 | 1·fwd (2079) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 2274 | 1.5·fwd (3119) | 1·fwd (2079) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 2274 | 1.5·fwd (3119) | 4·fwd (8316) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 2274 | 1.5·fwd (3119) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 2274 | 3·fwd (6237) | 1·fwd (2079) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 2274 | 3·fwd (6237) | 4·fwd (8316) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 2274 | 3·fwd (6237) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 2274 | inf (-) | 1·fwd (2079) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 2274 | inf (-) | 4·fwd (8316) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 2274 | inf (-) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| inference-dense | 3224 | 1·fwd (2964) | 1·fwd (2964) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 3224 | 1·fwd (2964) | 4·fwd (11856) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 3224 | 1·fwd (2964) | inf (-) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 3224 | 1.5·fwd (4446) | 1·fwd (2964) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 3224 | 1.5·fwd (4446) | 4·fwd (11856) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 3224 | 1.5·fwd (4446) | inf (-) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 3224 | 3·fwd (8892) | 1·fwd (2964) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 3224 | 3·fwd (8892) | 4·fwd (11856) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 3224 | 3·fwd (8892) | inf (-) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 3224 | inf (-) | 1·fwd (2964) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 3224 | inf (-) | 4·fwd (11856) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| inference-dense | 3224 | inf (-) | inf (-) | 16 | 16 | 16 | 1.00 | 1.00 |  |

### forward-nonfixed micro (accumulated weights)

| circuit | gates | F | G | I* | L | U (lit) | L/I* | U/I* | flags |
|---|---|---|---|---|---|---|---|---|---|
| deep_chain[depth=2,q=2,d=2,W=accumulated] | 32 | 1·fwd (12) | 1·fwd (12) | 32 | 24 | 32 | 0.75 | 1.00 |  |
| deep_chain[depth=2,q=2,d=2,W=accumulated] | 32 | 1·fwd (12) | 4·fwd (48) | 24 | 24 | 32 | 1.00 | 1.33 |  |
| deep_chain[depth=2,q=2,d=2,W=accumulated] | 32 | 1·fwd (12) | inf (-) | 24 | 24 | 32 | 1.00 | 1.33 |  |
| deep_chain[depth=2,q=2,d=2,W=accumulated] | 32 | 1.5·fwd (18) | 1·fwd (12) | 32 | 24 | 32 | 0.75 | 1.00 |  |
| deep_chain[depth=2,q=2,d=2,W=accumulated] | 32 | 1.5·fwd (18) | 4·fwd (48) | 24 | 24 | 32 | 1.00 | 1.33 |  |
| deep_chain[depth=2,q=2,d=2,W=accumulated] | 32 | 1.5·fwd (18) | inf (-) | 24 | 24 | 32 | 1.00 | 1.33 |  |
| deep_chain[depth=2,q=2,d=2,W=accumulated] | 32 | 3·fwd (36) | 1·fwd (12) | 32 | 24 | 32 | 0.75 | 1.00 |  |
| deep_chain[depth=2,q=2,d=2,W=accumulated] | 32 | 3·fwd (36) | 4·fwd (48) | 24 | 24 | 24 | 1.00 | 1.00 |  |
| deep_chain[depth=2,q=2,d=2,W=accumulated] | 32 | 3·fwd (36) | inf (-) | 24 | 24 | 24 | 1.00 | 1.00 |  |
| deep_chain[depth=2,q=2,d=2,W=accumulated] | 32 | inf (-) | 1·fwd (12) | 32 | 24 | 32 | 0.75 | 1.00 |  |
| deep_chain[depth=2,q=2,d=2,W=accumulated] | 32 | inf (-) | 4·fwd (48) | 24 | 24 | 24 | 1.00 | 1.00 |  |
| deep_chain[depth=2,q=2,d=2,W=accumulated] | 32 | inf (-) | inf (-) | 24 | 24 | 24 | 1.00 | 1.00 |  |
| deep_chain[depth=3,q=2,d=2,W=accumulated] | 48 | 1·fwd (12) | 1·fwd (12) | 48 | 32 | 48 | 0.67 | 1.00 |  |
| deep_chain[depth=3,q=2,d=2,W=accumulated] | 48 | 1·fwd (12) | 4·fwd (48) | 40 | 32 | 48 | 0.80 | 1.20 |  |
| deep_chain[depth=3,q=2,d=2,W=accumulated] | 48 | 1·fwd (12) | inf (-) | 40 | 32 | 48 | 0.80 | 1.20 |  |
| deep_chain[depth=3,q=2,d=2,W=accumulated] | 48 | 1.5·fwd (18) | 1·fwd (12) | 48 | 32 | 48 | 0.67 | 1.00 |  |
| deep_chain[depth=3,q=2,d=2,W=accumulated] | 48 | 1.5·fwd (18) | 4·fwd (48) | 32 | 32 | 48 | 1.00 | 1.50 |  |
| deep_chain[depth=3,q=2,d=2,W=accumulated] | 48 | 1.5·fwd (18) | inf (-) | 32 | 32 | 48 | 1.00 | 1.50 |  |
| deep_chain[depth=3,q=2,d=2,W=accumulated] | 48 | 3·fwd (36) | 1·fwd (12) | 48 | 32 | 48 | 0.67 | 1.00 |  |
| deep_chain[depth=3,q=2,d=2,W=accumulated] | 48 | 3·fwd (36) | 4·fwd (48) | 32 | 32 | 32 | 1.00 | 1.00 |  |
| deep_chain[depth=3,q=2,d=2,W=accumulated] | 48 | 3·fwd (36) | inf (-) | 32 | 32 | 32 | 1.00 | 1.00 |  |
| deep_chain[depth=3,q=2,d=2,W=accumulated] | 48 | inf (-) | 1·fwd (12) | 48 | 32 | 48 | 0.67 | 1.00 |  |
| deep_chain[depth=3,q=2,d=2,W=accumulated] | 48 | inf (-) | 4·fwd (48) | 32 | 32 | 32 | 1.00 | 1.00 |  |
| deep_chain[depth=3,q=2,d=2,W=accumulated] | 48 | inf (-) | inf (-) | 32 | 32 | 32 | 1.00 | 1.00 |  |
| deep_chain[depth=2,q=3,d=2,W=accumulated] | 48 | 1·fwd (18) | 1·fwd (18) | 40 | 28 | 40 | 0.70 | 1.00 |  |
| deep_chain[depth=2,q=3,d=2,W=accumulated] | 48 | 1·fwd (18) | 4·fwd (72) | 28 | 28 | 40 | 1.00 | 1.43 |  |
| deep_chain[depth=2,q=3,d=2,W=accumulated] | 48 | 1·fwd (18) | inf (-) | 28 | 28 | 40 | 1.00 | 1.43 |  |
| deep_chain[depth=2,q=3,d=2,W=accumulated] | 48 | 1.5·fwd (27) | 1·fwd (18) | 40 | 28 | 40 | 0.70 | 1.00 |  |
| deep_chain[depth=2,q=3,d=2,W=accumulated] | 48 | 1.5·fwd (27) | 4·fwd (72) | 28 | 28 | 40 | 1.00 | 1.43 |  |
| deep_chain[depth=2,q=3,d=2,W=accumulated] | 48 | 1.5·fwd (27) | inf (-) | 28 | 28 | 40 | 1.00 | 1.43 |  |
| deep_chain[depth=2,q=3,d=2,W=accumulated] | 48 | 3·fwd (54) | 1·fwd (18) | 40 | 28 | 40 | 0.70 | 1.00 |  |
| deep_chain[depth=2,q=3,d=2,W=accumulated] | 48 | 3·fwd (54) | 4·fwd (72) | 28 | 28 | 28 | 1.00 | 1.00 |  |
| deep_chain[depth=2,q=3,d=2,W=accumulated] | 48 | 3·fwd (54) | inf (-) | 28 | 28 | 28 | 1.00 | 1.00 |  |
| deep_chain[depth=2,q=3,d=2,W=accumulated] | 48 | inf (-) | 1·fwd (18) | 40 | 28 | 40 | 0.70 | 1.00 |  |
| deep_chain[depth=2,q=3,d=2,W=accumulated] | 48 | inf (-) | 4·fwd (72) | 28 | 28 | 28 | 1.00 | 1.00 |  |
| deep_chain[depth=2,q=3,d=2,W=accumulated] | 48 | inf (-) | inf (-) | 28 | 28 | 28 | 1.00 | 1.00 |  |

### rollout micro analogues (dense / MoE, accumulated weights)

| circuit | gates | F | G | I* | L | U (lit) | L/I* | U/I* | flags |
|---|---|---|---|---|---|---|---|---|---|
| dense_rollout[q=4,d=1,L=1] | 12 | 1.5·fwd (3) | 4·fwd (8) | 10 | 10 | 10 | 1.00 | 1.00 |  |
| dense_rollout[q=4,d=1,L=1] | 12 | 1.5·fwd (3) | 2·fwd (4) | 12 | 10 | 12 | 0.83 | 1.00 |  |
| dense_rollout[q=4,d=1,L=1] | 12 | 1.5·fwd (3) | 1·fwd (2) | 16 | 10 | 16 | 0.62 | 1.00 |  |
| dense_rollout[q=4,d=1,L=1] | 12 | inf (-) | 1·fwd (2) | 16 | 10 | 16 | 0.62 | 1.00 |  |
| dense_rollout[q=4,d=1,L=1] | 12 | 0.5·fwd (1) | 4·fwd (8) | 26 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.38 | - |  |
| dense_rollout[q=8,d=1,L=1] | 24 | 1.5·fwd (3) | 4·fwd (8) | 20 | 18 | 20 | 0.90 | 1.00 |  |
| dense_rollout[q=8,d=1,L=1] | 24 | 1.5·fwd (3) | 2·fwd (4) | 24 | 18 | 24 | 0.75 | 1.00 |  |
| dense_rollout[q=8,d=1,L=1] | 24 | 1.5·fwd (3) | 1·fwd (2) | 32 | 18 | 32 | 0.56 | 1.00 |  |
| dense_rollout[q=8,d=1,L=1] | 24 | inf (-) | 1·fwd (2) | 32 | 18 | 32 | 0.56 | 1.00 |  |
| dense_rollout[q=8,d=1,L=1] | 24 | 0.5·fwd (1) | 4·fwd (8) | 50 | 18 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.36 | - |  |
| dense_rollout[q=4,d=2,L=1] | 32 | 1.5·fwd (9) | 4·fwd (24) | 24 | 24 | 24 | 1.00 | 1.00 |  |
| dense_rollout[q=4,d=2,L=1] | 32 | 1.5·fwd (9) | 2·fwd (12) | 32 | 24 | 32 | 0.75 | 1.00 |  |
| dense_rollout[q=4,d=2,L=1] | 32 | 1.5·fwd (9) | 1·fwd (6) | 48 | 24 | 48 | 0.50 | 1.00 |  |
| dense_rollout[q=4,d=2,L=1] | 32 | inf (-) | 1·fwd (6) | 48 | 24 | 48 | 0.50 | 1.00 |  |
| dense_rollout[q=4,d=2,L=1] | 32 | 0.5·fwd (3) | 4·fwd (24) | 24 | 24 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 1.00 | - |  |
| dense_rollout[q=6,d=1,L=2] | 36 | 1.5·fwd (6) | 4·fwd (16) | 20 | 16 | 20 | 0.80 | 1.00 |  |
| dense_rollout[q=6,d=1,L=2] | 36 | 1.5·fwd (6) | 2·fwd (8) | 24 | 16 | 24 | 0.67 | 1.00 |  |
| dense_rollout[q=6,d=1,L=2] | 36 | 1.5·fwd (6) | 1·fwd (4) | 36 | 16 | 36 | 0.44 | 1.00 |  |
| dense_rollout[q=6,d=1,L=2] | 36 | inf (-) | 1·fwd (4) | 36 | 16 | 36 | 0.44 | 1.00 |  |
| dense_rollout[q=6,d=1,L=2] | 36 | 0.5·fwd (2) | 4·fwd (16) | 28 | 16 | 48 | 0.57 | 1.71 |  |
| moe_rollout[q=4,d=1,E=2,k=1,S=1] | 28 | 1.5·fwd (6) | 4·fwd (16) | 16 | 14 | 18 | 0.88 | 1.12 |  |
| moe_rollout[q=4,d=1,E=2,k=1,S=1] | 28 | 1.5·fwd (6) | 2·fwd (8) | 24 | 14 | 24 | 0.58 | 1.00 |  |
| moe_rollout[q=4,d=1,E=2,k=1,S=1] | 28 | 1.5·fwd (6) | 1·fwd (4) | 36 | 14 | 40 | 0.39 | 1.11 |  |
| moe_rollout[q=4,d=1,E=2,k=1,S=1] | 28 | inf (-) | 1·fwd (4) | 36 | 14 | 40 | 0.39 | 1.11 |  |
| moe_rollout[q=4,d=1,E=2,k=1,S=1] | 28 | 0.5·fwd (2) | 4·fwd (16) | 30 | 14 | 40 | 0.47 | 1.33 |  |
| moe_rollout[q=8,d=1,E=2,k=1,S=1] | 56 | 1.5·fwd (6) | 4·fwd (16) | 30 | 22 | 34 | 0.73 | 1.13 |  |
| moe_rollout[q=8,d=1,E=2,k=1,S=1] | 56 | 1.5·fwd (6) | 2·fwd (8) | [38, 48] | 22 | 48 | 0.46 | 1.00 |  |
| moe_rollout[q=8,d=1,E=2,k=1,S=1] | 56 | 1.5·fwd (6) | 1·fwd (4) | [66, 72] | 22 | 80 | 0.31 | 1.11 |  |
| moe_rollout[q=8,d=1,E=2,k=1,S=1] | 56 | inf (-) | 1·fwd (4) | [66, 72] | 22 | 80 | 0.31 | 1.11 |  |
| moe_rollout[q=8,d=1,E=2,k=1,S=1] | 56 | 0.5·fwd (2) | 4·fwd (16) | 56 | 22 | 80 | 0.39 | 1.43 |  |
| moe_rollout[q=4,d=1,E=4,k=2,S=1] | 44 | 1.5·fwd (9) | 4·fwd (24) | 24 | 18 | 24 | 0.75 | 1.00 |  |
| moe_rollout[q=4,d=1,E=4,k=2,S=1] | 44 | 1.5·fwd (9) | 2·fwd (12) | 32 | 18 | 32 | 0.56 | 1.00 |  |
| moe_rollout[q=4,d=1,E=4,k=2,S=1] | 44 | 1.5·fwd (9) | 1·fwd (6) | [42, 46] | 18 | 48 | 0.39 | 1.04 |  |
| moe_rollout[q=4,d=1,E=4,k=2,S=1] | 44 | inf (-) | 1·fwd (6) | [42, 46] | 18 | 48 | 0.39 | 1.04 |  |
| moe_rollout[q=4,d=1,E=4,k=2,S=1] | 44 | 0.5·fwd (3) | 4·fwd (24) | 42 | 18 | 58 | 0.43 | 1.38 |  |
| moe_rollout[q=6,d=1,E=3,k=1,S=0] | 18 | 1.5·fwd (3) | 4·fwd (8) | 18 | 18 | 22 | 1.00 | 1.22 |  |
| moe_rollout[q=6,d=1,E=3,k=1,S=0] | 18 | 1.5·fwd (3) | 2·fwd (4) | 18 | 18 | 24 | 1.00 | 1.33 |  |
| moe_rollout[q=6,d=1,E=3,k=1,S=0] | 18 | 1.5·fwd (3) | 1·fwd (2) | 24 | 18 | 24 | 0.75 | 1.00 |  |
| moe_rollout[q=6,d=1,E=3,k=1,S=0] | 18 | inf (-) | 1·fwd (2) | 24 | 18 | 24 | 0.75 | 1.00 |  |
| moe_rollout[q=6,d=1,E=3,k=1,S=0] | 18 | 0.5·fwd (1) | 4·fwd (8) | 42 | 18 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.43 | - |  |
| moe_rollout[q=4,d=2,E=2,k=1,S=0] | 32 | 1.5·fwd (9) | 4·fwd (24) | 32 | 32 | 32 | 1.00 | 1.00 |  |
| moe_rollout[q=4,d=2,E=2,k=1,S=0] | 32 | 1.5·fwd (9) | 2·fwd (12) | 32 | 32 | 48 | 1.00 | 1.50 |  |
| moe_rollout[q=4,d=2,E=2,k=1,S=0] | 32 | 1.5·fwd (9) | 1·fwd (6) | 48 | 32 | 48 | 0.67 | 1.00 |  |
| moe_rollout[q=4,d=2,E=2,k=1,S=0] | 32 | inf (-) | 1·fwd (6) | 48 | 32 | 48 | 0.67 | 1.00 |  |
| moe_rollout[q=4,d=2,E=2,k=1,S=0] | 32 | 0.5·fwd (3) | 4·fwd (24) | 32 | 32 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 1.00 | - |  |

### registry forward-nonfixed @ tiny

| circuit | gates | F | G | I* | L | U (lit) | L/I* | U/I* | flags |
|---|---|---|---|---|---|---|---|---|---|
| forward-nonfixed[tiny,q=2] | 656 | 1·fwd (592) | 1·fwd (592) | 384 | 384 | 384 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=2] | 656 | 1·fwd (592) | 4·fwd (2368) | 384 | 384 | 384 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=2] | 656 | 1·fwd (592) | inf (-) | 384 | 384 | 384 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=2] | 656 | 1.5·fwd (888) | 1·fwd (592) | 384 | 384 | 384 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=2] | 656 | 1.5·fwd (888) | 4·fwd (2368) | 384 | 384 | 384 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=2] | 656 | 1.5·fwd (888) | inf (-) | 384 | 384 | 384 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=2] | 656 | 3·fwd (1776) | 1·fwd (592) | 384 | 384 | 384 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=2] | 656 | 3·fwd (1776) | 4·fwd (2368) | 384 | 384 | 384 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=2] | 656 | 3·fwd (1776) | inf (-) | 384 | 384 | 384 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=2] | 656 | inf (-) | 1·fwd (592) | 384 | 384 | 384 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=2] | 656 | inf (-) | 4·fwd (2368) | 384 | 384 | 384 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=2] | 656 | inf (-) | inf (-) | 384 | 384 | 384 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=4] | 1440 | 1·fwd (1312) | 1·fwd (1312) | 392 | 392 | 392 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=4] | 1440 | 1·fwd (1312) | 4·fwd (5248) | 392 | 392 | 392 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=4] | 1440 | 1·fwd (1312) | inf (-) | 392 | 392 | 392 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=4] | 1440 | 1.5·fwd (1968) | 1·fwd (1312) | 392 | 392 | 392 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=4] | 1440 | 1.5·fwd (1968) | 4·fwd (5248) | 392 | 392 | 392 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=4] | 1440 | 1.5·fwd (1968) | inf (-) | 392 | 392 | 392 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=4] | 1440 | 3·fwd (3936) | 1·fwd (1312) | 392 | 392 | 392 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=4] | 1440 | 3·fwd (3936) | 4·fwd (5248) | 392 | 392 | 392 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=4] | 1440 | 3·fwd (3936) | inf (-) | 392 | 392 | 392 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=4] | 1440 | inf (-) | 1·fwd (1312) | 392 | 392 | 392 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=4] | 1440 | inf (-) | 4·fwd (5248) | 392 | 392 | 392 | 1.00 | 1.00 |  |
| forward-nonfixed[tiny,q=4] | 1440 | inf (-) | inf (-) | 392 | 392 | 392 | 1.00 | 1.00 |  |

### classic micro suite

| circuit | gates | F | G | I* | L | U (lit) | L/I* | U/I* | flags |
|---|---|---|---|---|---|---|---|---|---|
| mm_1x1x1 | 3 | W/4 (1) | W/4 (1) | 8 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.50 | - |  |
| mm_1x1x1 | 3 | W/4 (1) | W/2 (1) | 8 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.50 | - |  |
| mm_1x1x1 | 3 | W/4 (1) | inf (-) | 8 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.50 | - |  |
| mm_1x1x1 | 3 | W/2 (1) | W/4 (1) | 8 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.50 | - |  |
| mm_1x1x1 | 3 | W/2 (1) | W/2 (1) | 8 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.50 | - |  |
| mm_1x1x1 | 3 | W/2 (1) | inf (-) | 8 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.50 | - |  |
| mm_1x1x1 | 3 | inf (-) | W/4 (1) | 8 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.50 | - |  |
| mm_1x1x1 | 3 | inf (-) | W/2 (1) | 8 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.50 | - |  |
| mm_1x1x1 | 3 | inf (-) | inf (-) | 4 | 4 | 4 | 1.00 | 1.00 |  |
| mm_1x1x2 | 4 | W/4 (1) | W/4 (1) | 16 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.50 | - |  |
| mm_1x1x2 | 4 | W/4 (1) | W/2 (2) | 16 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.50 | - |  |
| mm_1x1x2 | 4 | W/4 (1) | inf (-) | 16 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.50 | - |  |
| mm_1x1x2 | 4 | W/2 (2) | W/4 (1) | 16 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.50 | - |  |
| mm_1x1x2 | 4 | W/2 (2) | W/2 (2) | 12 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.67 | - |  |
| mm_1x1x2 | 4 | W/2 (2) | inf (-) | 12 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.67 | - |  |
| mm_1x1x2 | 4 | inf (-) | W/4 (1) | 16 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.50 | - |  |
| mm_1x1x2 | 4 | inf (-) | W/2 (2) | 12 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.67 | - |  |
| mm_1x1x2 | 4 | inf (-) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| mm_1x1x3 | 5 | W/4 (1) | W/4 (1) | 24 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=3,CH=1}) | 0.50 | - |  |
| mm_1x1x3 | 5 | W/4 (1) | W/2 (2) | 24 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=3,CH=1}) | 0.50 | - |  |
| mm_1x1x3 | 5 | W/4 (1) | inf (-) | 24 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=3,CH=1}) | 0.50 | - |  |
| mm_1x1x3 | 5 | W/2 (2) | W/4 (1) | 24 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=3,CH=1}) | 0.50 | - |  |
| mm_1x1x3 | 5 | W/2 (2) | W/2 (2) | 16 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=3,CH=1}) | 0.75 | - |  |
| mm_1x1x3 | 5 | W/2 (2) | inf (-) | 16 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=3,CH=1}) | 0.75 | - |  |
| mm_1x1x3 | 5 | inf (-) | W/4 (1) | 24 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=3,CH=1}) | 0.50 | - |  |
| mm_1x1x3 | 5 | inf (-) | W/2 (2) | 16 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=3,CH=1}) | 0.75 | - |  |
| mm_1x1x3 | 5 | inf (-) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| mm_1x2x2 | 8 | W/4 (2) | W/4 (2) | 22 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.55 | - |  |
| mm_1x2x2 | 8 | W/4 (2) | W/2 (3) | 22 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.55 | - |  |
| mm_1x2x2 | 8 | W/4 (2) | inf (-) | 20 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.60 | - |  |
| mm_1x2x2 | 8 | W/2 (3) | W/4 (2) | 22 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.55 | - |  |
| mm_1x2x2 | 8 | W/2 (3) | W/2 (3) | 16 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.75 | - |  |
| mm_1x2x2 | 8 | W/2 (3) | inf (-) | 12 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 1.00 | - |  |
| mm_1x2x2 | 8 | inf (-) | W/4 (2) | 22 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.55 | - |  |
| mm_1x2x2 | 8 | inf (-) | W/2 (3) | 16 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.75 | - |  |
| mm_1x2x2 | 8 | inf (-) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| mm_2x1x2 | 8 | W/4 (2) | W/4 (2) | 22 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=1,K=2,CH=1}) | 0.55 | - |  |
| mm_2x1x2 | 8 | W/4 (2) | W/2 (3) | 22 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=1,K=2,CH=1}) | 0.55 | - |  |
| mm_2x1x2 | 8 | W/4 (2) | inf (-) | 20 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=1,K=2,CH=1}) | 0.60 | - |  |
| mm_2x1x2 | 8 | W/2 (3) | W/4 (2) | 22 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=1,K=2,CH=1}) | 0.55 | - |  |
| mm_2x1x2 | 8 | W/2 (3) | W/2 (3) | 16 | 12 | 16 | 0.75 | 1.00 |  |
| mm_2x1x2 | 8 | W/2 (3) | inf (-) | 12 | 12 | 16 | 1.00 | 1.33 |  |
| mm_2x1x2 | 8 | inf (-) | W/4 (2) | 22 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=1,K=2,CH=1}) | 0.55 | - |  |
| mm_2x1x2 | 8 | inf (-) | W/2 (3) | 16 | 12 | 16 | 0.75 | 1.00 |  |
| mm_2x1x2 | 8 | inf (-) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| mm_2x1x2_carried | 8 | W/4 (2) | W/4 (2) | 22 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=1,K=2,CH=1}) | 0.55 | - |  |
| mm_2x1x2_carried | 8 | W/4 (2) | W/2 (3) | 22 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=1,K=2,CH=1}) | 0.55 | - |  |
| mm_2x1x2_carried | 8 | W/4 (2) | inf (-) | 20 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=1,K=2,CH=1}) | 0.60 | - |  |
| mm_2x1x2_carried | 8 | W/2 (3) | W/4 (2) | 22 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=1,K=2,CH=1}) | 0.55 | - |  |
| mm_2x1x2_carried | 8 | W/2 (3) | W/2 (3) | 16 | 12 | 16 | 0.75 | 1.00 |  |
| mm_2x1x2_carried | 8 | W/2 (3) | inf (-) | 12 | 12 | 16 | 1.00 | 1.33 |  |
| mm_2x1x2_carried | 8 | inf (-) | W/4 (2) | 22 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=1,K=2,CH=1}) | 0.55 | - |  |
| mm_2x1x2_carried | 8 | inf (-) | W/2 (3) | 16 | 12 | 16 | 0.75 | 1.00 |  |
| mm_2x1x2_carried | 8 | inf (-) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| mm_3x1x1 | 9 | W/4 (2) | W/4 (2) | 12 | 8 | 12 | 0.67 | 1.00 |  |
| mm_3x1x1 | 9 | W/4 (2) | W/2 (3) | 12 | 8 | 12 | 0.67 | 1.00 |  |
| mm_3x1x1 | 9 | W/4 (2) | inf (-) | 8 | 8 | 12 | 1.00 | 1.50 |  |
| mm_3x1x1 | 9 | W/2 (3) | W/4 (2) | 12 | 8 | 12 | 0.67 | 1.00 |  |
| mm_3x1x1 | 9 | W/2 (3) | W/2 (3) | 12 | 8 | 12 | 0.67 | 1.00 |  |
| mm_3x1x1 | 9 | W/2 (3) | inf (-) | 8 | 8 | 12 | 1.00 | 1.50 |  |
| mm_3x1x1 | 9 | inf (-) | W/4 (2) | 12 | 8 | 12 | 0.67 | 1.00 |  |
| mm_3x1x1 | 9 | inf (-) | W/2 (3) | 12 | 8 | 12 | 0.67 | 1.00 |  |
| mm_3x1x1 | 9 | inf (-) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| mm_1x3x1 | 9 | W/4 (2) | W/4 (2) | 12 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=3,K=1,CH=1}) | 0.67 | - |  |
| mm_1x3x1 | 9 | W/4 (2) | W/2 (3) | 12 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=3,K=1,CH=1}) | 0.67 | - |  |
| mm_1x3x1 | 9 | W/4 (2) | inf (-) | 8 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=3,K=1,CH=1}) | 1.00 | - |  |
| mm_1x3x1 | 9 | W/2 (3) | W/4 (2) | 12 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=3,K=1,CH=1}) | 0.67 | - |  |
| mm_1x3x1 | 9 | W/2 (3) | W/2 (3) | 12 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=3,K=1,CH=1}) | 0.67 | - |  |
| mm_1x3x1 | 9 | W/2 (3) | inf (-) | 8 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=3,K=1,CH=1}) | 1.00 | - |  |
| mm_1x3x1 | 9 | inf (-) | W/4 (2) | 12 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=3,K=1,CH=1}) | 0.67 | - |  |
| mm_1x3x1 | 9 | inf (-) | W/2 (3) | 12 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=3,K=1,CH=1}) | 0.67 | - |  |
| mm_1x3x1 | 9 | inf (-) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| mm_2x2x1 | 12 | W/4 (2) | W/4 (2) | 16 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=1,CH=1}) | 0.50 | - |  |
| mm_2x2x1 | 12 | W/4 (2) | W/2 (4) | 12 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=1,CH=1}) | 0.67 | - |  |
| mm_2x2x1 | 12 | W/4 (2) | inf (-) | 8 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=1,CH=1}) | 1.00 | - |  |
| mm_2x2x1 | 12 | W/2 (4) | W/4 (2) | 16 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=1,CH=1}) | 0.50 | - |  |
| mm_2x2x1 | 12 | W/2 (4) | W/2 (4) | 12 | 8 | 12 | 0.67 | 1.00 |  |
| mm_2x2x1 | 12 | W/2 (4) | inf (-) | 8 | 8 | 12 | 1.00 | 1.50 |  |
| mm_2x2x1 | 12 | inf (-) | W/4 (2) | 16 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=1,CH=1}) | 0.50 | - |  |
| mm_2x2x1 | 12 | inf (-) | W/2 (4) | 12 | 8 | 12 | 0.67 | 1.00 |  |
| mm_2x2x1 | 12 | inf (-) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| mm_2x2x2 | 16 | W/4 (3) | W/4 (3) | 32 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 0.50 | - |  |
| mm_2x2x2 | 16 | W/4 (3) | W/2 (6) | 24 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 0.67 | - |  |
| mm_2x2x2 | 16 | W/4 (3) | inf (-) | 16 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 1.00 | - |  |
| mm_2x2x2 | 16 | W/2 (6) | W/4 (3) | 32 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 0.50 | - |  |
| mm_2x2x2 | 16 | W/2 (6) | W/2 (6) | 24 | 16 | 24 | 0.67 | 1.00 |  |
| mm_2x2x2 | 16 | W/2 (6) | inf (-) | 16 | 16 | 24 | 1.00 | 1.50 |  |
| mm_2x2x2 | 16 | inf (-) | W/4 (3) | 32 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 0.50 | - |  |
| mm_2x2x2 | 16 | inf (-) | W/2 (6) | 24 | 16 | 24 | 0.67 | 1.00 |  |
| mm_2x2x2 | 16 | inf (-) | inf (-) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| mm_2x2x3 | 20 | W/4 (4) | W/4 (4) | 48 | 24 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=3,CH=1}) | 0.50 | - |  |
| mm_2x2x3 | 20 | W/4 (4) | W/2 (8) | 36 | 24 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=3,CH=1}) | 0.67 | - |  |
| mm_2x2x3 | 20 | W/4 (4) | inf (-) | 24 | 24 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=3,CH=1}) | 1.00 | - |  |
| mm_2x2x3 | 20 | W/2 (8) | W/4 (4) | 48 | 24 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=3,CH=1}) | 0.50 | - |  |
| mm_2x2x3 | 20 | W/2 (8) | W/2 (8) | 36 | 24 | 36 | 0.67 | 1.00 |  |
| mm_2x2x3 | 20 | W/2 (8) | inf (-) | 24 | 24 | 36 | 1.00 | 1.50 |  |
| mm_2x2x3 | 20 | inf (-) | W/4 (4) | 48 | 24 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=3,CH=1}) | 0.50 | - |  |
| mm_2x2x3 | 20 | inf (-) | W/2 (8) | 36 | 24 | 36 | 0.67 | 1.00 |  |
| mm_2x2x3 | 20 | inf (-) | inf (-) | 24 | 24 | 24 | 1.00 | 1.00 |  |
| mm_3x2x2 | 24 | W/4 (5) | W/4 (5) | 48 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=2,K=2,CH=1}) | 0.42 | - |  |
| mm_3x2x2 | 24 | W/4 (5) | W/2 (9) | 32 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=2,K=2,CH=1}) | 0.62 | - |  |
| mm_3x2x2 | 24 | W/4 (5) | inf (-) | 20 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=2,K=2,CH=1}) | 1.00 | - |  |
| mm_3x2x2 | 24 | W/2 (9) | W/4 (5) | 48 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=2,K=2,CH=1}) | 0.42 | - |  |
| mm_3x2x2 | 24 | W/2 (9) | W/2 (9) | 32 | 20 | 36 | 0.62 | 1.12 |  |
| mm_3x2x2 | 24 | W/2 (9) | inf (-) | 20 | 20 | 36 | 1.00 | 1.80 |  |
| mm_3x2x2 | 24 | inf (-) | W/4 (5) | 48 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=2,K=2,CH=1}) | 0.42 | - |  |
| mm_3x2x2 | 24 | inf (-) | W/2 (9) | 32 | 20 | 36 | 0.62 | 1.12 |  |
| mm_3x2x2 | 24 | inf (-) | inf (-) | 20 | 20 | 20 | 1.00 | 1.00 |  |
| mm_3x3x2 | 36 | W/4 (7) | W/4 (7) | [50, 56] | 24 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=3,K=2,CH=1}) | 0.43 | - |  |
| mm_3x3x2 | 36 | W/4 (7) | W/2 (14) | 44 | 24 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=3,K=2,CH=1}) | 0.55 | - |  |
| mm_3x3x2 | 36 | W/4 (7) | inf (-) | 24 | 24 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=3,K=2,CH=1}) | 1.00 | - |  |
| mm_3x3x2 | 36 | W/2 (14) | W/4 (7) | [50, 56] | 24 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=3,K=2,CH=1}) | 0.43 | - |  |
| mm_3x3x2 | 36 | W/2 (14) | W/2 (14) | 44 | 24 | 48 | 0.55 | 1.09 |  |
| mm_3x3x2 | 36 | W/2 (14) | inf (-) | 24 | 24 | 48 | 1.00 | 2.00 |  |
| mm_3x3x2 | 36 | inf (-) | W/4 (7) | [50, 56] | 24 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=3,K=2,CH=1}) | 0.43 | - |  |
| mm_3x3x2 | 36 | inf (-) | W/2 (14) | 44 | 24 | 48 | 0.55 | 1.09 |  |
| mm_3x3x2 | 36 | inf (-) | inf (-) | 24 | 24 | 24 | 1.00 | 1.00 |  |
| chain_1x1x1x1 | 6 | W/4 (1) | W/4 (1) | 16 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.38 | - |  |
| chain_1x1x1x1 | 6 | W/4 (1) | W/2 (2) | 16 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.38 | - |  |
| chain_1x1x1x1 | 6 | W/4 (1) | inf (-) | 16 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.38 | - |  |
| chain_1x1x1x1 | 6 | W/2 (2) | W/4 (1) | 16 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.38 | - |  |
| chain_1x1x1x1 | 6 | W/2 (2) | W/2 (2) | 8 | 6 | 8 | 0.75 | 1.00 |  |
| chain_1x1x1x1 | 6 | W/2 (2) | inf (-) | 8 | 6 | 8 | 0.75 | 1.00 |  |
| chain_1x1x1x1 | 6 | inf (-) | W/4 (1) | 16 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.38 | - |  |
| chain_1x1x1x1 | 6 | inf (-) | W/2 (2) | 8 | 6 | 8 | 0.75 | 1.00 |  |
| chain_1x1x1x1 | 6 | inf (-) | inf (-) | 6 | 6 | 6 | 1.00 | 1.00 |  |
| chain_1x1x2x1 | 7 | W/4 (2) | W/4 (2) | 16 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.62 | - |  |
| chain_1x1x2x1 | 7 | W/4 (2) | W/2 (3) | 16 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.62 | - |  |
| chain_1x1x2x1 | 7 | W/4 (2) | inf (-) | 16 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.62 | - |  |
| chain_1x1x2x1 | 7 | W/2 (3) | W/4 (2) | 16 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.62 | - |  |
| chain_1x1x2x1 | 7 | W/2 (3) | W/2 (3) | 12 | 10 | 12 | 0.83 | 1.00 |  |
| chain_1x1x2x1 | 7 | W/2 (3) | inf (-) | 12 | 10 | 12 | 0.83 | 1.00 |  |
| chain_1x1x2x1 | 7 | inf (-) | W/4 (2) | 16 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.62 | - |  |
| chain_1x1x2x1 | 7 | inf (-) | W/2 (3) | 12 | 10 | 12 | 0.83 | 1.00 |  |
| chain_1x1x2x1 | 7 | inf (-) | inf (-) | 10 | 10 | 10 | 1.00 | 1.00 |  |
| chain_1x2x1x1 | 10 | W/4 (2) | W/4 (2) | 20 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=1,CH=1}) | 0.50 | - |  |
| chain_1x2x1x1 | 10 | W/4 (2) | W/2 (4) | 18 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=1,CH=1}) | 0.56 | - |  |
| chain_1x2x1x1 | 10 | W/4 (2) | inf (-) | 18 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=1,CH=1}) | 0.56 | - |  |
| chain_1x2x1x1 | 10 | W/2 (4) | W/4 (2) | 20 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=1,CH=1}) | 0.50 | - |  |
| chain_1x2x1x1 | 10 | W/2 (4) | W/2 (4) | 14 | 10 | 14 | 0.71 | 1.00 |  |
| chain_1x2x1x1 | 10 | W/2 (4) | inf (-) | 14 | 10 | 14 | 0.71 | 1.00 |  |
| chain_1x2x1x1 | 10 | inf (-) | W/4 (2) | 20 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=1,CH=1}) | 0.50 | - |  |
| chain_1x2x1x1 | 10 | inf (-) | W/2 (4) | 14 | 10 | 14 | 0.71 | 1.00 |  |
| chain_1x2x1x1 | 10 | inf (-) | inf (-) | 10 | 10 | 10 | 1.00 | 1.00 |  |
| chain_2x1x1x1 | 12 | W/4 (2) | W/4 (2) | 16 | 8 | 16 | 0.50 | 1.00 |  |
| chain_2x1x1x1 | 12 | W/4 (2) | W/2 (4) | 12 | 8 | 16 | 0.67 | 1.33 |  |
| chain_2x1x1x1 | 12 | W/4 (2) | inf (-) | 12 | 8 | 16 | 0.67 | 1.33 |  |
| chain_2x1x1x1 | 12 | W/2 (4) | W/4 (2) | 16 | 8 | 16 | 0.50 | 1.00 |  |
| chain_2x1x1x1 | 12 | W/2 (4) | W/2 (4) | 12 | 8 | 12 | 0.67 | 1.00 |  |
| chain_2x1x1x1 | 12 | W/2 (4) | inf (-) | 8 | 8 | 12 | 1.00 | 1.50 |  |
| chain_2x1x1x1 | 12 | inf (-) | W/4 (2) | 16 | 8 | 16 | 0.50 | 1.00 |  |
| chain_2x1x1x1 | 12 | inf (-) | W/2 (4) | 12 | 8 | 12 | 0.67 | 1.00 |  |
| chain_2x1x1x1 | 12 | inf (-) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| chain_1x2x2x2 | 16 | W/4 (3) | W/4 (3) | 32 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.62 | - |  |
| chain_1x2x2x2 | 16 | W/4 (3) | W/2 (6) | 24 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.83 | - |  |
| chain_1x2x2x2 | 16 | W/4 (3) | inf (-) | 24 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.83 | - |  |
| chain_1x2x2x2 | 16 | W/2 (6) | W/4 (3) | 32 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.62 | - |  |
| chain_1x2x2x2 | 16 | W/2 (6) | W/2 (6) | 24 | 20 | 24 | 0.83 | 1.00 |  |
| chain_1x2x2x2 | 16 | W/2 (6) | inf (-) | 24 | 20 | 24 | 0.83 | 1.00 |  |
| chain_1x2x2x2 | 16 | inf (-) | W/4 (3) | 32 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.62 | - |  |
| chain_1x2x2x2 | 16 | inf (-) | W/2 (6) | 24 | 20 | 24 | 0.83 | 1.00 |  |
| chain_1x2x2x2 | 16 | inf (-) | inf (-) | 20 | 20 | 20 | 1.00 | 1.00 |  |
| chain_2x2x2x2 | 32 | W/4 (6) | W/4 (6) | 48 | 24 | 48 | 0.50 | 1.00 |  |
| chain_2x2x2x2 | 32 | W/4 (6) | W/2 (12) | 32 | 24 | 48 | 0.75 | 1.50 |  |
| chain_2x2x2x2 | 32 | W/4 (6) | inf (-) | 32 | 24 | 48 | 0.75 | 1.50 |  |
| chain_2x2x2x2 | 32 | W/2 (12) | W/4 (6) | 48 | 24 | 48 | 0.50 | 1.00 |  |
| chain_2x2x2x2 | 32 | W/2 (12) | W/2 (12) | 32 | 24 | 32 | 0.75 | 1.00 |  |
| chain_2x2x2x2 | 32 | W/2 (12) | inf (-) | 24 | 24 | 32 | 1.00 | 1.33 |  |
| chain_2x2x2x2 | 32 | inf (-) | W/4 (6) | 48 | 24 | 48 | 0.50 | 1.00 |  |
| chain_2x2x2x2 | 32 | inf (-) | W/2 (12) | 32 | 24 | 32 | 0.75 | 1.00 |  |
| chain_2x2x2x2 | 32 | inf (-) | inf (-) | 24 | 24 | 24 | 1.00 | 1.00 |  |
| mmadd_1x1x1 | 4 | W/4 (1) | W/4 (1) | 12 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.50 | - |  |
| mmadd_1x1x1 | 4 | W/4 (1) | W/2 (2) | 12 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.50 | - |  |
| mmadd_1x1x1 | 4 | W/4 (1) | inf (-) | 12 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.50 | - |  |
| mmadd_1x1x1 | 4 | W/2 (2) | W/4 (1) | 12 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.50 | - |  |
| mmadd_1x1x1 | 4 | W/2 (2) | W/2 (2) | 8 | 6 | 8 | 0.75 | 1.00 |  |
| mmadd_1x1x1 | 4 | W/2 (2) | inf (-) | 8 | 6 | 8 | 0.75 | 1.00 |  |
| mmadd_1x1x1 | 4 | inf (-) | W/4 (1) | 12 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.50 | - |  |
| mmadd_1x1x1 | 4 | inf (-) | W/2 (2) | 8 | 6 | 8 | 0.75 | 1.00 |  |
| mmadd_1x1x1 | 4 | inf (-) | inf (-) | 6 | 6 | 6 | 1.00 | 1.00 |  |
| mmadd_1x1x2 | 5 | W/4 (1) | W/4 (1) | 20 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.50 | - |  |
| mmadd_1x1x2 | 5 | W/4 (1) | W/2 (2) | 20 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.50 | - |  |
| mmadd_1x1x2 | 5 | W/4 (1) | inf (-) | 20 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.50 | - |  |
| mmadd_1x1x2 | 5 | W/2 (2) | W/4 (1) | 20 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.50 | - |  |
| mmadd_1x1x2 | 5 | W/2 (2) | W/2 (2) | 14 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.71 | - |  |
| mmadd_1x1x2 | 5 | W/2 (2) | inf (-) | 14 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.71 | - |  |
| mmadd_1x1x2 | 5 | inf (-) | W/4 (1) | 20 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.50 | - |  |
| mmadd_1x1x2 | 5 | inf (-) | W/2 (2) | 14 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.71 | - |  |
| mmadd_1x1x2 | 5 | inf (-) | inf (-) | 10 | 10 | 10 | 1.00 | 1.00 |  |
| mmadd_2x1x1 | 8 | W/4 (2) | W/4 (2) | 16 | 10 | 16 | 0.62 | 1.00 |  |
| mmadd_2x1x1 | 8 | W/4 (2) | W/2 (3) | 16 | 10 | 16 | 0.62 | 1.00 |  |
| mmadd_2x1x1 | 8 | W/4 (2) | inf (-) | 14 | 10 | 16 | 0.71 | 1.14 |  |
| mmadd_2x1x1 | 8 | W/2 (3) | W/4 (2) | 16 | 10 | 16 | 0.62 | 1.00 |  |
| mmadd_2x1x1 | 8 | W/2 (3) | W/2 (3) | 12 | 10 | 16 | 0.83 | 1.33 |  |
| mmadd_2x1x1 | 8 | W/2 (3) | inf (-) | 10 | 10 | 16 | 1.00 | 1.60 |  |
| mmadd_2x1x1 | 8 | inf (-) | W/4 (2) | 16 | 10 | 16 | 0.62 | 1.00 |  |
| mmadd_2x1x1 | 8 | inf (-) | W/2 (3) | 12 | 10 | 16 | 0.83 | 1.33 |  |
| mmadd_2x1x1 | 8 | inf (-) | inf (-) | 10 | 10 | 10 | 1.00 | 1.00 |  |
| mmadd_1x2x2 | 10 | W/4 (2) | W/4 (2) | 28 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.57 | - |  |
| mmadd_1x2x2 | 10 | W/4 (2) | W/2 (4) | 24 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.67 | - |  |
| mmadd_1x2x2 | 10 | W/4 (2) | inf (-) | 24 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.67 | - |  |
| mmadd_1x2x2 | 10 | W/2 (4) | W/4 (2) | 28 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.57 | - |  |
| mmadd_1x2x2 | 10 | W/2 (4) | W/2 (4) | 20 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.80 | - |  |
| mmadd_1x2x2 | 10 | W/2 (4) | inf (-) | 16 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 1.00 | - |  |
| mmadd_1x2x2 | 10 | inf (-) | W/4 (2) | 28 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.57 | - |  |
| mmadd_1x2x2 | 10 | inf (-) | W/2 (4) | 20 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.80 | - |  |
| mmadd_1x2x2 | 10 | inf (-) | inf (-) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| mmadd_1x2x2_carried | 10 | W/4 (2) | W/4 (2) | 28 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.57 | - |  |
| mmadd_1x2x2_carried | 10 | W/4 (2) | W/2 (4) | 24 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.67 | - |  |
| mmadd_1x2x2_carried | 10 | W/4 (2) | inf (-) | 24 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.67 | - |  |
| mmadd_1x2x2_carried | 10 | W/2 (4) | W/4 (2) | 28 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.57 | - |  |
| mmadd_1x2x2_carried | 10 | W/2 (4) | W/2 (4) | 20 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.80 | - |  |
| mmadd_1x2x2_carried | 10 | W/2 (4) | inf (-) | 16 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 1.00 | - |  |
| mmadd_1x2x2_carried | 10 | inf (-) | W/4 (2) | 28 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.57 | - |  |
| mmadd_1x2x2_carried | 10 | inf (-) | W/2 (4) | 20 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.80 | - |  |
| mmadd_1x2x2_carried | 10 | inf (-) | inf (-) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| mmadd_2x2x2 | 20 | W/4 (4) | W/4 (4) | 40 | 24 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 0.60 | - |  |
| mmadd_2x2x2 | 20 | W/4 (4) | W/2 (8) | 32 | 24 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 0.75 | - |  |
| mmadd_2x2x2 | 20 | W/4 (4) | inf (-) | 24 | 24 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 1.00 | - |  |
| mmadd_2x2x2 | 20 | W/2 (8) | W/4 (4) | 40 | 24 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 0.60 | - |  |
| mmadd_2x2x2 | 20 | W/2 (8) | W/2 (8) | 32 | 24 | 40 | 0.75 | 1.25 |  |
| mmadd_2x2x2 | 20 | W/2 (8) | inf (-) | 24 | 24 | 40 | 1.00 | 1.67 |  |
| mmadd_2x2x2 | 20 | inf (-) | W/4 (4) | 40 | 24 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 0.60 | - |  |
| mmadd_2x2x2 | 20 | inf (-) | W/2 (8) | 32 | 24 | 40 | 0.75 | 1.25 |  |
| mmadd_2x2x2 | 20 | inf (-) | inf (-) | 24 | 24 | 24 | 1.00 | 1.00 |  |
| fixacc_1x1x1 | 6 | W/4 (1) | W/4 (1) | 14 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.29 | - |  |
| fixacc_1x1x1 | 6 | W/4 (1) | W/2 (2) | 14 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.29 | - |  |
| fixacc_1x1x1 | 6 | W/4 (1) | inf (-) | 14 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.29 | - |  |
| fixacc_1x1x1 | 6 | W/2 (2) | W/4 (1) | 14 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.29 | - |  |
| fixacc_1x1x1 | 6 | W/2 (2) | W/2 (2) | 6 | 4 | 6 | 0.67 | 1.00 |  |
| fixacc_1x1x1 | 6 | W/2 (2) | inf (-) | 6 | 4 | 6 | 0.67 | 1.00 |  |
| fixacc_1x1x1 | 6 | inf (-) | W/4 (1) | 14 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.29 | - |  |
| fixacc_1x1x1 | 6 | inf (-) | W/2 (2) | 6 | 4 | 6 | 0.67 | 1.00 |  |
| fixacc_1x1x1 | 6 | inf (-) | inf (-) | 4 | 4 | 4 | 1.00 | 1.00 |  |
| fixacc_1x1x2 | 7 | W/4 (2) | W/4 (2) | 12 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.50 | - |  |
| fixacc_1x1x2 | 7 | W/4 (2) | W/2 (3) | 12 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.50 | - |  |
| fixacc_1x1x2 | 7 | W/4 (2) | inf (-) | 12 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.50 | - |  |
| fixacc_1x1x2 | 7 | W/2 (3) | W/4 (2) | 12 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.50 | - |  |
| fixacc_1x1x2 | 7 | W/2 (3) | W/2 (3) | 8 | 6 | 8 | 0.75 | 1.00 |  |
| fixacc_1x1x2 | 7 | W/2 (3) | inf (-) | 8 | 6 | 8 | 0.75 | 1.00 |  |
| fixacc_1x1x2 | 7 | inf (-) | W/4 (2) | 12 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.50 | - |  |
| fixacc_1x1x2 | 7 | inf (-) | W/2 (3) | 8 | 6 | 8 | 0.75 | 1.00 |  |
| fixacc_1x1x2 | 7 | inf (-) | inf (-) | 6 | 6 | 6 | 1.00 | 1.00 |  |
| fixacc_2x1x1 | 12 | W/4 (2) | W/4 (2) | 12 | 6 | 12 | 0.50 | 1.00 |  |
| fixacc_2x1x1 | 12 | W/4 (2) | W/2 (4) | 10 | 6 | 12 | 0.60 | 1.20 |  |
| fixacc_2x1x1 | 12 | W/4 (2) | inf (-) | 10 | 6 | 12 | 0.60 | 1.20 |  |
| fixacc_2x1x1 | 12 | W/2 (4) | W/4 (2) | 12 | 6 | 12 | 0.50 | 1.00 |  |
| fixacc_2x1x1 | 12 | W/2 (4) | W/2 (4) | 8 | 6 | 10 | 0.75 | 1.25 |  |
| fixacc_2x1x1 | 12 | W/2 (4) | inf (-) | 6 | 6 | 10 | 1.00 | 1.67 |  |
| fixacc_2x1x1 | 12 | inf (-) | W/4 (2) | 12 | 6 | 12 | 0.50 | 1.00 |  |
| fixacc_2x1x1 | 12 | inf (-) | W/2 (4) | 8 | 6 | 10 | 0.75 | 1.25 |  |
| fixacc_2x1x1 | 12 | inf (-) | inf (-) | 6 | 6 | 6 | 1.00 | 1.00 |  |
| fixacc_1x2x1 | 14 | W/4 (3) | W/4 (3) | 20 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=1,CH=1}) | 0.50 | - |  |
| fixacc_1x2x1 | 14 | W/4 (3) | W/2 (5) | 18 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=1,CH=1}) | 0.56 | - |  |
| fixacc_1x2x1 | 14 | W/4 (3) | inf (-) | 14 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=1,CH=1}) | 0.71 | - |  |
| fixacc_1x2x1 | 14 | W/2 (5) | W/4 (3) | 20 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=1,CH=1}) | 0.50 | - |  |
| fixacc_1x2x1 | 14 | W/2 (5) | W/2 (5) | 16 | 10 | n/a: ValueError: op 1 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.62 | - |  |
| fixacc_1x2x1 | 14 | W/2 (5) | inf (-) | 14 | 10 | n/a: ValueError: op 1 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.71 | - |  |
| fixacc_1x2x1 | 14 | inf (-) | W/4 (3) | 20 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=1,CH=1}) | 0.50 | - |  |
| fixacc_1x2x1 | 14 | inf (-) | W/2 (5) | 16 | 10 | n/a: ValueError: op 1 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.62 | - |  |
| fixacc_1x2x1 | 14 | inf (-) | inf (-) | 10 | 10 | 10 | 1.00 | 1.00 |  |
| fixacc_1x2x2 | 16 | W/4 (3) | W/4 (3) | 24 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.50 | - |  |
| fixacc_1x2x2 | 16 | W/4 (3) | W/2 (6) | 16 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.75 | - |  |
| fixacc_1x2x2 | 16 | W/4 (3) | inf (-) | 16 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.75 | - |  |
| fixacc_1x2x2 | 16 | W/2 (6) | W/4 (3) | 24 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.50 | - |  |
| fixacc_1x2x2 | 16 | W/2 (6) | W/2 (6) | 16 | 12 | 16 | 0.75 | 1.00 |  |
| fixacc_1x2x2 | 16 | W/2 (6) | inf (-) | 16 | 12 | 16 | 0.75 | 1.00 |  |
| fixacc_1x2x2 | 16 | inf (-) | W/4 (3) | 24 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.50 | - |  |
| fixacc_1x2x2 | 16 | inf (-) | W/2 (6) | 16 | 12 | 16 | 0.75 | 1.00 |  |
| fixacc_1x2x2 | 16 | inf (-) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| fixacc_2x2x1 | 28 | W/4 (5) | W/4 (5) | 32 | 12 | n/a: ValueError: op 1 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 0.38 | - |  |
| fixacc_2x2x1 | 28 | W/4 (5) | W/2 (10) | 24 | 12 | n/a: ValueError: op 1 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 0.50 | - |  |
| fixacc_2x2x1 | 28 | W/4 (5) | inf (-) | 20 | 12 | n/a: ValueError: op 1 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 0.60 | - |  |
| fixacc_2x2x1 | 28 | W/2 (10) | W/4 (5) | 32 | 12 | n/a: ValueError: op 1 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 0.38 | - |  |
| fixacc_2x2x1 | 28 | W/2 (10) | W/2 (10) | 20 | 12 | 28 | 0.60 | 1.40 |  |
| fixacc_2x2x1 | 28 | W/2 (10) | inf (-) | 12 | 12 | 28 | 1.00 | 2.33 |  |
| fixacc_2x2x1 | 28 | inf (-) | W/4 (5) | 32 | 12 | n/a: ValueError: op 1 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 0.38 | - |  |
| fixacc_2x2x1 | 28 | inf (-) | W/2 (10) | 20 | 12 | 28 | 0.60 | 1.40 |  |
| fixacc_2x2x1 | 28 | inf (-) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| fixacc_2x2x2 | 32 | W/4 (6) | W/4 (6) | 32 | 16 | 32 | 0.50 | 1.00 |  |
| fixacc_2x2x2 | 32 | W/4 (6) | W/2 (12) | 24 | 16 | 32 | 0.67 | 1.33 |  |
| fixacc_2x2x2 | 32 | W/4 (6) | inf (-) | 24 | 16 | 32 | 0.67 | 1.33 |  |
| fixacc_2x2x2 | 32 | W/2 (12) | W/4 (6) | 32 | 16 | 32 | 0.50 | 1.00 |  |
| fixacc_2x2x2 | 32 | W/2 (12) | W/2 (12) | 24 | 16 | 24 | 0.67 | 1.00 |  |
| fixacc_2x2x2 | 32 | W/2 (12) | inf (-) | 16 | 16 | 24 | 1.00 | 1.50 |  |
| fixacc_2x2x2 | 32 | inf (-) | W/4 (6) | 32 | 16 | 32 | 0.50 | 1.00 |  |
| fixacc_2x2x2 | 32 | inf (-) | W/2 (12) | 24 | 16 | 24 | 0.67 | 1.00 |  |
| fixacc_2x2x2 | 32 | inf (-) | inf (-) | 16 | 16 | 16 | 1.00 | 1.00 |  |

### torture cases

| circuit | gates | F | G | I* | L | U (lit) | L/I* | U/I* | flags |
|---|---|---|---|---|---|---|---|---|---|
| a1_independent_rows | 24 | W/4 (5) | W/4 (5) | 24 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=2,K=2,CH=1}) | 0.50 | - |  |
| a1_independent_rows | 24 | W/4 (5) | W/2 (9) | 12 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=2,K=2,CH=1}) | 1.00 | - |  |
| a1_independent_rows | 24 | W/4 (5) | inf (-) | 12 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=2,K=2,CH=1}) | 1.00 | - |  |
| a1_independent_rows | 24 | W/2 (9) | W/4 (5) | 24 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=2,K=2,CH=1}) | 0.50 | - |  |
| a1_independent_rows | 24 | W/2 (9) | W/2 (9) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| a1_independent_rows | 24 | W/2 (9) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| a1_independent_rows | 24 | inf (-) | W/4 (5) | 24 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=2,K=2,CH=1}) | 0.50 | - |  |
| a1_independent_rows | 24 | inf (-) | W/2 (9) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| a1_independent_rows | 24 | inf (-) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| a2_fixed_chain_F_split | 12 | W/4 (3) | W/4 (3) | 12 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.33 | - |  |
| a2_fixed_chain_F_split | 12 | W/4 (3) | W/2 (5) | 12 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.33 | - |  |
| a2_fixed_chain_F_split | 12 | W/4 (3) | inf (-) | 8 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.50 | - |  |
| a2_fixed_chain_F_split | 12 | W/2 (5) | W/4 (3) | 12 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.33 | - |  |
| a2_fixed_chain_F_split | 12 | W/2 (5) | W/2 (5) | 10 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.40 | - |  |
| a2_fixed_chain_F_split | 12 | W/2 (5) | inf (-) | 8 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.50 | - |  |
| a2_fixed_chain_F_split | 12 | inf (-) | W/4 (3) | 12 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.33 | - |  |
| a2_fixed_chain_F_split | 12 | inf (-) | W/2 (5) | 10 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.40 | - |  |
| a2_fixed_chain_F_split | 12 | inf (-) | inf (-) | 4 | 4 | 4 | 1.00 | 1.00 |  |
| b1_compact_precursor_fanout | 28 | W/4 (6) | W/4 (6) | 24 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=4,K=2,CH=1}) | 0.17 | - |  |
| b1_compact_precursor_fanout | 28 | W/4 (6) | W/2 (11) | 14 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=4,K=2,CH=1}) | 0.29 | - |  |
| b1_compact_precursor_fanout | 28 | W/4 (6) | inf (-) | 12 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=4,K=2,CH=1}) | 0.33 | - |  |
| b1_compact_precursor_fanout | 28 | W/2 (11) | W/4 (6) | 24 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=4,K=2,CH=1}) | 0.17 | - |  |
| b1_compact_precursor_fanout | 28 | W/2 (11) | W/2 (11) | 14 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=4,K=2,CH=1}) | 0.29 | - |  |
| b1_compact_precursor_fanout | 28 | W/2 (11) | inf (-) | 12 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=4,K=2,CH=1}) | 0.33 | - |  |
| b1_compact_precursor_fanout | 28 | inf (-) | W/4 (6) | 24 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=4,K=2,CH=1}) | 0.17 | - |  |
| b1_compact_precursor_fanout | 28 | inf (-) | W/2 (11) | 14 | 4 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=4,K=2,CH=1}) | 0.29 | - |  |
| b1_compact_precursor_fanout | 28 | inf (-) | inf (-) | 4 | 4 | 4 | 1.00 | 1.00 |  |
| c1_independent_precursors | 6 | W/4 (2) | W/4 (2) | 24 | 24 | n/a: ValueError: op 0 (AccMulBatch_v1{Q=2,K=3}): one ro | 1.00 | - |  |
| c1_independent_precursors | 6 | W/4 (2) | W/2 (3) | 24 | 24 | n/a: ValueError: op 0 (AccMulBatch_v1{Q=2,K=3}): one ro | 1.00 | - |  |
| c1_independent_precursors | 6 | W/4 (2) | inf (-) | 24 | 24 | n/a: ValueError: op 0 (AccMulBatch_v1{Q=2,K=3}): one ro | 1.00 | - |  |
| c1_independent_precursors | 6 | W/2 (3) | W/4 (2) | 24 | 24 | n/a: ValueError: op 0 (AccMulBatch_v1{Q=2,K=3}): one ro | 1.00 | - |  |
| c1_independent_precursors | 6 | W/2 (3) | W/2 (3) | 24 | 24 | 24 | 1.00 | 1.00 |  |
| c1_independent_precursors | 6 | W/2 (3) | inf (-) | 24 | 24 | 24 | 1.00 | 1.00 |  |
| c1_independent_precursors | 6 | inf (-) | W/4 (2) | 24 | 24 | n/a: ValueError: op 0 (AccMulBatch_v1{Q=2,K=3}): one ro | 1.00 | - |  |
| c1_independent_precursors | 6 | inf (-) | W/2 (3) | 24 | 24 | 24 | 1.00 | 1.00 |  |
| c1_independent_precursors | 6 | inf (-) | inf (-) | 24 | 24 | 24 | 1.00 | 1.00 |  |
| c2_accumulated_rows_fixed_weights | 24 | W/4 (5) | W/4 (5) | 24 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=2,K=2,CH=1}) | 0.50 | - |  |
| c2_accumulated_rows_fixed_weights | 24 | W/4 (5) | W/2 (9) | 12 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=2,K=2,CH=1}) | 1.00 | - |  |
| c2_accumulated_rows_fixed_weights | 24 | W/4 (5) | inf (-) | 12 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=2,K=2,CH=1}) | 1.00 | - |  |
| c2_accumulated_rows_fixed_weights | 24 | W/2 (9) | W/4 (5) | 24 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=2,K=2,CH=1}) | 0.50 | - |  |
| c2_accumulated_rows_fixed_weights | 24 | W/2 (9) | W/2 (9) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| c2_accumulated_rows_fixed_weights | 24 | W/2 (9) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| c2_accumulated_rows_fixed_weights | 24 | inf (-) | W/4 (5) | 24 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=2,K=2,CH=1}) | 0.50 | - |  |
| c2_accumulated_rows_fixed_weights | 24 | inf (-) | W/2 (9) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| c2_accumulated_rows_fixed_weights | 24 | inf (-) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| d_two_layer_M1 | 18 | W/4 (4) | W/4 (4) | 28 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=3,CH=1}) | 0.71 | - |  |
| d_two_layer_M1 | 18 | W/4 (4) | W/2 (7) | 24 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=3,CH=1}) | 0.83 | - |  |
| d_two_layer_M1 | 18 | W/4 (4) | inf (-) | 24 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=3,CH=1}) | 0.83 | - |  |
| d_two_layer_M1 | 18 | W/2 (7) | W/4 (4) | 28 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=3,CH=1}) | 0.71 | - |  |
| d_two_layer_M1 | 18 | W/2 (7) | W/2 (7) | 24 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=3,CH=1}) | 0.83 | - |  |
| d_two_layer_M1 | 18 | W/2 (7) | inf (-) | 22 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=3,CH=1}) | 0.91 | - |  |
| d_two_layer_M1 | 18 | inf (-) | W/4 (4) | 28 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=3,CH=1}) | 0.71 | - |  |
| d_two_layer_M1 | 18 | inf (-) | W/2 (7) | 24 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=3,CH=1}) | 0.83 | - |  |
| d_two_layer_M1 | 18 | inf (-) | inf (-) | 20 | 20 | 20 | 1.00 | 1.00 |  |
| d_two_layer_M2 | 36 | W/4 (7) | W/4 (7) | 40 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=3,CH=1}) | 0.50 | - |  |
| d_two_layer_M2 | 36 | W/4 (7) | W/2 (14) | 28 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=3,CH=1}) | 0.71 | - |  |
| d_two_layer_M2 | 36 | W/4 (7) | inf (-) | 24 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=3,CH=1}) | 0.83 | - |  |
| d_two_layer_M2 | 36 | W/2 (14) | W/4 (7) | 40 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=3,CH=1}) | 0.50 | - |  |
| d_two_layer_M2 | 36 | W/2 (14) | W/2 (14) | 28 | 20 | 40 | 0.71 | 1.43 |  |
| d_two_layer_M2 | 36 | W/2 (14) | inf (-) | 20 | 20 | 40 | 1.00 | 2.00 |  |
| d_two_layer_M2 | 36 | inf (-) | W/4 (7) | 40 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=3,CH=1}) | 0.50 | - |  |
| d_two_layer_M2 | 36 | inf (-) | W/2 (14) | 28 | 20 | 40 | 0.71 | 1.43 |  |
| d_two_layer_M2 | 36 | inf (-) | inf (-) | 20 | 20 | 20 | 1.00 | 1.00 |  |
| e1_shared_weights_fixed_acts | 24 | W/4 (5) | W/4 (5) | 24 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.33 | - |  |
| e1_shared_weights_fixed_acts | 24 | W/4 (5) | W/2 (9) | 8 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 1.00 | - |  |
| e1_shared_weights_fixed_acts | 24 | W/4 (5) | inf (-) | 8 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 1.00 | - |  |
| e1_shared_weights_fixed_acts | 24 | W/2 (9) | W/4 (5) | 24 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.33 | - |  |
| e1_shared_weights_fixed_acts | 24 | W/2 (9) | W/2 (9) | 8 | 8 | 24 | 1.00 | 3.00 |  |
| e1_shared_weights_fixed_acts | 24 | W/2 (9) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| e1_shared_weights_fixed_acts | 24 | inf (-) | W/4 (5) | 24 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.33 | - |  |
| e1_shared_weights_fixed_acts | 24 | inf (-) | W/2 (9) | 8 | 8 | 24 | 1.00 | 3.00 |  |
| e1_shared_weights_fixed_acts | 24 | inf (-) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| e2_shared_weights_token_acts | 24 | W/4 (5) | W/4 (5) | 48 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.42 | - |  |
| e2_shared_weights_token_acts | 24 | W/4 (5) | W/2 (9) | 32 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.62 | - |  |
| e2_shared_weights_token_acts | 24 | W/4 (5) | inf (-) | 20 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 1.00 | - |  |
| e2_shared_weights_token_acts | 24 | W/2 (9) | W/4 (5) | 48 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.42 | - |  |
| e2_shared_weights_token_acts | 24 | W/2 (9) | W/2 (9) | 32 | 20 | 36 | 0.62 | 1.12 |  |
| e2_shared_weights_token_acts | 24 | W/2 (9) | inf (-) | 20 | 20 | 20 | 1.00 | 1.00 |  |
| e2_shared_weights_token_acts | 24 | inf (-) | W/4 (5) | 48 | 20 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.42 | - |  |
| e2_shared_weights_token_acts | 24 | inf (-) | W/2 (9) | 32 | 20 | 36 | 0.62 | 1.12 |  |
| e2_shared_weights_token_acts | 24 | inf (-) | inf (-) | 20 | 20 | 20 | 1.00 | 1.00 |  |
| f1_lora_fixed_x | 24 | W/4 (5) | W/4 (5) | 16 | 6 | 22 | 0.38 | 1.38 |  |
| f1_lora_fixed_x | 24 | W/4 (5) | W/2 (9) | 12 | 6 | 22 | 0.50 | 1.83 |  |
| f1_lora_fixed_x | 24 | W/4 (5) | inf (-) | 10 | 6 | 22 | 0.60 | 2.20 |  |
| f1_lora_fixed_x | 24 | W/2 (9) | W/4 (5) | 16 | 6 | 22 | 0.38 | 1.38 |  |
| f1_lora_fixed_x | 24 | W/2 (9) | W/2 (9) | 12 | 6 | 14 | 0.50 | 1.17 |  |
| f1_lora_fixed_x | 24 | W/2 (9) | inf (-) | 6 | 6 | 14 | 1.00 | 2.33 |  |
| f1_lora_fixed_x | 24 | inf (-) | W/4 (5) | 16 | 6 | 22 | 0.38 | 1.38 |  |
| f1_lora_fixed_x | 24 | inf (-) | W/2 (9) | 12 | 6 | 14 | 0.50 | 1.17 |  |
| f1_lora_fixed_x | 24 | inf (-) | inf (-) | 6 | 6 | 6 | 1.00 | 1.00 |  |
| f2_lora_token_x | 24 | W/4 (5) | W/4 (5) | 32 | 14 | 38 | 0.44 | 1.19 |  |
| f2_lora_token_x | 24 | W/4 (5) | W/2 (9) | 26 | 14 | 38 | 0.54 | 1.46 |  |
| f2_lora_token_x | 24 | W/4 (5) | inf (-) | 22 | 14 | 38 | 0.64 | 1.73 |  |
| f2_lora_token_x | 24 | W/2 (9) | W/4 (5) | 32 | 14 | 38 | 0.44 | 1.19 |  |
| f2_lora_token_x | 24 | W/2 (9) | W/2 (9) | 20 | 14 | 30 | 0.70 | 1.50 |  |
| f2_lora_token_x | 24 | W/2 (9) | inf (-) | 14 | 14 | 22 | 1.00 | 1.57 |  |
| f2_lora_token_x | 24 | inf (-) | W/4 (5) | 32 | 14 | 38 | 0.44 | 1.19 |  |
| f2_lora_token_x | 24 | inf (-) | W/2 (9) | 20 | 14 | 30 | 0.70 | 1.50 |  |
| f2_lora_token_x | 24 | inf (-) | inf (-) | 14 | 14 | 14 | 1.00 | 1.00 |  |
| g1_residual_add | 10 | W/4 (2) | W/4 (2) | 28 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.43 | - |  |
| g1_residual_add | 10 | W/4 (2) | W/2 (4) | 24 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.50 | - |  |
| g1_residual_add | 10 | W/4 (2) | inf (-) | 24 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.50 | - |  |
| g1_residual_add | 10 | W/2 (4) | W/4 (2) | 28 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.43 | - |  |
| g1_residual_add | 10 | W/2 (4) | W/2 (4) | 16 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.75 | - |  |
| g1_residual_add | 10 | W/2 (4) | inf (-) | 12 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 1.00 | - |  |
| g1_residual_add | 10 | inf (-) | W/4 (2) | 28 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.43 | - |  |
| g1_residual_add | 10 | inf (-) | W/2 (4) | 16 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.75 | - |  |
| g1_residual_add | 10 | inf (-) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| g2_embedding_gather | 6 | W/4 (2) | W/4 (2) | 36 | 20 | 36 | 0.56 | 1.00 |  |
| g2_embedding_gather | 6 | W/4 (2) | W/2 (3) | 32 | 20 | 36 | 0.62 | 1.12 |  |
| g2_embedding_gather | 6 | W/4 (2) | inf (-) | 20 | 20 | 36 | 1.00 | 1.80 |  |
| g2_embedding_gather | 6 | W/2 (3) | W/4 (2) | 36 | 20 | 36 | 0.56 | 1.00 |  |
| g2_embedding_gather | 6 | W/2 (3) | W/2 (3) | 32 | 20 | 36 | 0.62 | 1.12 |  |
| g2_embedding_gather | 6 | W/2 (3) | inf (-) | 20 | 20 | 36 | 1.00 | 1.80 |  |
| g2_embedding_gather | 6 | inf (-) | W/4 (2) | 36 | 20 | 36 | 0.56 | 1.00 |  |
| g2_embedding_gather | 6 | inf (-) | W/2 (3) | 32 | 20 | 36 | 0.62 | 1.12 |  |
| g2_embedding_gather | 6 | inf (-) | inf (-) | 20 | 20 | 20 | 1.00 | 1.00 |  |
| g3_embedding_then_matmul | 30 | W/4 (6) | W/4 (6) | 56 | 28 | 56 | 0.50 | 1.00 |  |
| g3_embedding_then_matmul | 30 | W/4 (6) | W/2 (12) | 44 | 28 | 56 | 0.64 | 1.27 |  |
| g3_embedding_then_matmul | 30 | W/4 (6) | inf (-) | 28 | 28 | 56 | 1.00 | 2.00 |  |
| g3_embedding_then_matmul | 30 | W/2 (12) | W/4 (6) | 56 | 28 | 56 | 0.50 | 1.00 |  |
| g3_embedding_then_matmul | 30 | W/2 (12) | W/2 (12) | 44 | 28 | 48 | 0.64 | 1.09 |  |
| g3_embedding_then_matmul | 30 | W/2 (12) | inf (-) | 28 | 28 | 48 | 1.00 | 1.71 |  |
| g3_embedding_then_matmul | 30 | inf (-) | W/4 (6) | 56 | 28 | 56 | 0.50 | 1.00 |  |
| g3_embedding_then_matmul | 30 | inf (-) | W/2 (12) | 44 | 28 | 48 | 0.64 | 1.09 |  |
| g3_embedding_then_matmul | 30 | inf (-) | inf (-) | 28 | 28 | 28 | 1.00 | 1.00 |  |
| g4_branching_intermediate | 16 | W/4 (3) | W/4 (3) | 24 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.67 | - |  |
| g4_branching_intermediate | 16 | W/4 (3) | W/2 (6) | 20 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.80 | - |  |
| g4_branching_intermediate | 16 | W/4 (3) | inf (-) | 20 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.80 | - |  |
| g4_branching_intermediate | 16 | W/2 (6) | W/4 (3) | 24 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.67 | - |  |
| g4_branching_intermediate | 16 | W/2 (6) | W/2 (6) | 20 | 16 | 20 | 0.80 | 1.00 |  |
| g4_branching_intermediate | 16 | W/2 (6) | inf (-) | 18 | 16 | 20 | 0.89 | 1.11 |  |
| g4_branching_intermediate | 16 | inf (-) | W/4 (3) | 24 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.67 | - |  |
| g4_branching_intermediate | 16 | inf (-) | W/2 (6) | 20 | 16 | 20 | 0.80 | 1.00 |  |
| g4_branching_intermediate | 16 | inf (-) | inf (-) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| h1_weight_gradient_tt | 16 | W/4 (3) | W/4 (3) | 32 | 16 | n/a: ValueError: op 0 (AccMatmulTT_v1{M=2,N=2,K=2,CH=1} | 0.50 | - |  |
| h1_weight_gradient_tt | 16 | W/4 (3) | W/2 (6) | 24 | 16 | n/a: ValueError: op 0 (AccMatmulTT_v1{M=2,N=2,K=2,CH=1} | 0.67 | - |  |
| h1_weight_gradient_tt | 16 | W/4 (3) | inf (-) | 16 | 16 | n/a: ValueError: op 0 (AccMatmulTT_v1{M=2,N=2,K=2,CH=1} | 1.00 | - |  |
| h1_weight_gradient_tt | 16 | W/2 (6) | W/4 (3) | 32 | 16 | n/a: ValueError: op 0 (AccMatmulTT_v1{M=2,N=2,K=2,CH=1} | 0.50 | - |  |
| h1_weight_gradient_tt | 16 | W/2 (6) | W/2 (6) | 24 | 16 | n/a: ValueError: op 0 (AccMatmulTT_v1{M=2,N=2,K=2,CH=1} | 0.67 | - |  |
| h1_weight_gradient_tt | 16 | W/2 (6) | inf (-) | 16 | 16 | n/a: ValueError: op 0 (AccMatmulTT_v1{M=2,N=2,K=2,CH=1} | 1.00 | - |  |
| h1_weight_gradient_tt | 16 | inf (-) | W/4 (3) | 32 | 16 | n/a: ValueError: op 0 (AccMatmulTT_v1{M=2,N=2,K=2,CH=1} | 0.50 | - |  |
| h1_weight_gradient_tt | 16 | inf (-) | W/2 (6) | 24 | 16 | n/a: ValueError: op 0 (AccMatmulTT_v1{M=2,N=2,K=2,CH=1} | 0.67 | - |  |
| h1_weight_gradient_tt | 16 | inf (-) | inf (-) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| h2_colsum | 16 | W/4 (4) | W/4 (4) | 20 | 12 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=3,K=2}): one | 0.60 | - |  |
| h2_colsum | 16 | W/4 (4) | W/2 (7) | 20 | 12 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=3,K=2}): one | 0.60 | - |  |
| h2_colsum | 16 | W/4 (4) | inf (-) | 20 | 12 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=3,K=2}): one | 0.60 | - |  |
| h2_colsum | 16 | W/2 (7) | W/4 (4) | 20 | 12 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=3,K=2}): one | 0.60 | - |  |
| h2_colsum | 16 | W/2 (7) | W/2 (7) | 12 | 12 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=3,K=2}): one | 1.00 | - |  |
| h2_colsum | 16 | W/2 (7) | inf (-) | 12 | 12 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=3,K=2}): one | 1.00 | - |  |
| h2_colsum | 16 | inf (-) | W/4 (4) | 20 | 12 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=3,K=2}): one | 0.60 | - |  |
| h2_colsum | 16 | inf (-) | W/2 (7) | 12 | 12 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=3,K=2}): one | 1.00 | - |  |
| h2_colsum | 16 | inf (-) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| h3_colsum_into_gain | 20 | W/4 (5) | W/4 (5) | 28 | 20 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=3,K=2}): one | 0.71 | - |  |
| h3_colsum_into_gain | 20 | W/4 (5) | W/2 (9) | 28 | 20 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=3,K=2}): one | 0.71 | - |  |
| h3_colsum_into_gain | 20 | W/4 (5) | inf (-) | 28 | 20 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=3,K=2}): one | 0.71 | - |  |
| h3_colsum_into_gain | 20 | W/2 (9) | W/4 (5) | 28 | 20 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=3,K=2}): one | 0.71 | - |  |
| h3_colsum_into_gain | 20 | W/2 (9) | W/2 (9) | 20 | 20 | 40 | 1.00 | 2.00 |  |
| h3_colsum_into_gain | 20 | W/2 (9) | inf (-) | 20 | 20 | 40 | 1.00 | 2.00 |  |
| h3_colsum_into_gain | 20 | inf (-) | W/4 (5) | 28 | 20 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=3,K=2}): one | 0.71 | - |  |
| h3_colsum_into_gain | 20 | inf (-) | W/2 (9) | 20 | 20 | 40 | 1.00 | 2.00 |  |
| h3_colsum_into_gain | 20 | inf (-) | inf (-) | 20 | 20 | 20 | 1.00 | 1.00 |  |
| h4_gram_matrix | 16 | W/4 (3) | W/4 (3) | 24 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 0.33 | - |  |
| h4_gram_matrix | 16 | W/4 (3) | W/2 (6) | 16 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 0.50 | - |  |
| h4_gram_matrix | 16 | W/4 (3) | inf (-) | 8 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 1.00 | - |  |
| h4_gram_matrix | 16 | W/2 (6) | W/4 (3) | 24 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 0.33 | - |  |
| h4_gram_matrix | 16 | W/2 (6) | W/2 (6) | 16 | 8 | 16 | 0.50 | 1.00 |  |
| h4_gram_matrix | 16 | W/2 (6) | inf (-) | 8 | 8 | 16 | 1.00 | 2.00 |  |
| h4_gram_matrix | 16 | inf (-) | W/4 (3) | 24 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 0.33 | - |  |
| h4_gram_matrix | 16 | inf (-) | W/2 (6) | 16 | 8 | 16 | 0.50 | 1.00 |  |
| h4_gram_matrix | 16 | inf (-) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| t1_same_row_read_twice | 8 | W/4 (2) | W/4 (2) | 22 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.55 | - |  |
| t1_same_row_read_twice | 8 | W/4 (2) | W/2 (3) | 22 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.55 | - |  |
| t1_same_row_read_twice | 8 | W/4 (2) | inf (-) | 20 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.60 | - |  |
| t1_same_row_read_twice | 8 | W/2 (3) | W/4 (2) | 22 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.55 | - |  |
| t1_same_row_read_twice | 8 | W/2 (3) | W/2 (3) | 16 | 12 | 16 | 0.75 | 1.00 |  |
| t1_same_row_read_twice | 8 | W/2 (3) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| t1_same_row_read_twice | 8 | inf (-) | W/4 (2) | 22 | 12 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=2,CH=1}) | 0.55 | - |  |
| t1_same_row_read_twice | 8 | inf (-) | W/2 (3) | 16 | 12 | 16 | 0.75 | 1.00 |  |
| t1_same_row_read_twice | 8 | inf (-) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| t2_repeated_row_view_B | 8 | W/4 (2) | W/4 (2) | 20 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.40 | - |  |
| t2_repeated_row_view_B | 8 | W/4 (2) | W/2 (3) | 20 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.40 | - |  |
| t2_repeated_row_view_B | 8 | W/4 (2) | inf (-) | 16 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.50 | - |  |
| t2_repeated_row_view_B | 8 | W/2 (3) | W/4 (2) | 20 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.40 | - |  |
| t2_repeated_row_view_B | 8 | W/2 (3) | W/2 (3) | 16 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.50 | - |  |
| t2_repeated_row_view_B | 8 | W/2 (3) | inf (-) | 8 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 1.00 | - |  |
| t2_repeated_row_view_B | 8 | inf (-) | W/4 (2) | 20 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.40 | - |  |
| t2_repeated_row_view_B | 8 | inf (-) | W/2 (3) | 16 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.50 | - |  |
| t2_repeated_row_view_B | 8 | inf (-) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| t3_repeated_row_view_A | 36 | W/4 (6) | W/4 (6) | 20 | 12 | 32 | 0.60 | 1.60 |  |
| t3_repeated_row_view_A | 36 | W/4 (6) | W/2 (12) | 14 | 12 | 32 | 0.86 | 2.29 |  |
| t3_repeated_row_view_A | 36 | W/4 (6) | inf (-) | 12 | 12 | 32 | 1.00 | 2.67 |  |
| t3_repeated_row_view_A | 36 | W/2 (12) | W/4 (6) | 20 | 12 | 32 | 0.60 | 1.60 |  |
| t3_repeated_row_view_A | 36 | W/2 (12) | W/2 (12) | 14 | 12 | 20 | 0.86 | 1.43 |  |
| t3_repeated_row_view_A | 36 | W/2 (12) | inf (-) | 12 | 12 | 20 | 1.00 | 1.67 |  |
| t3_repeated_row_view_A | 36 | inf (-) | W/4 (6) | 20 | 12 | 32 | 0.60 | 1.60 |  |
| t3_repeated_row_view_A | 36 | inf (-) | W/2 (12) | 14 | 12 | 20 | 0.86 | 1.43 |  |
| t3_repeated_row_view_A | 36 | inf (-) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| t4_joint_epilogue_F | 22 | W/4 (5) | W/4 (5) | 52 | 30 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=3,CH=1}) | 0.58 | - |  |
| t4_joint_epilogue_F | 22 | W/4 (5) | W/2 (9) | 40 | 30 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=3,CH=1}) | 0.75 | - |  |
| t4_joint_epilogue_F | 22 | W/4 (5) | inf (-) | 38 | 30 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=3,CH=1}) | 0.79 | - |  |
| t4_joint_epilogue_F | 22 | W/2 (9) | W/4 (5) | 52 | 30 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=3,CH=1}) | 0.58 | - |  |
| t4_joint_epilogue_F | 22 | W/2 (9) | W/2 (9) | 36 | 30 | 44 | 0.83 | 1.22 |  |
| t4_joint_epilogue_F | 22 | W/2 (9) | inf (-) | 30 | 30 | 38 | 1.00 | 1.27 |  |
| t4_joint_epilogue_F | 22 | inf (-) | W/4 (5) | 52 | 30 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=3,CH=1}) | 0.58 | - |  |
| t4_joint_epilogue_F | 22 | inf (-) | W/2 (9) | 36 | 30 | 44 | 0.83 | 1.22 |  |
| t4_joint_epilogue_F | 22 | inf (-) | inf (-) | 30 | 30 | 30 | 1.00 | 1.00 |  |
| t5_rowfull_epilogue_F | 18 | W/4 (4) | W/4 (4) | 34 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.47 | - |  |
| t5_rowfull_epilogue_F | 18 | W/4 (4) | W/2 (8) | 28 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.57 | - |  |
| t5_rowfull_epilogue_F | 18 | W/4 (4) | inf (-) | 28 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.57 | - |  |
| t5_rowfull_epilogue_F | 18 | W/2 (8) | W/4 (4) | 34 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.47 | - |  |
| t5_rowfull_epilogue_F | 18 | W/2 (8) | W/2 (8) | 22 | 16 | n/a: ValueError: op 1 (AccRmsNormBatch_v1{Q=1,K=2}): on | 0.73 | - |  |
| t5_rowfull_epilogue_F | 18 | W/2 (8) | inf (-) | 20 | 16 | n/a: ValueError: op 1 (AccRmsNormBatch_v1{Q=1,K=2}): on | 0.80 | - |  |
| t5_rowfull_epilogue_F | 18 | inf (-) | W/4 (4) | 34 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.47 | - |  |
| t5_rowfull_epilogue_F | 18 | inf (-) | W/2 (8) | 22 | 16 | n/a: ValueError: op 1 (AccRmsNormBatch_v1{Q=1,K=2}): on | 0.73 | - |  |
| t5_rowfull_epilogue_F | 18 | inf (-) | inf (-) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| t6_softmax_epilogue_F | 22 | W/4 (6) | W/4 (6) | 36 | 12 | n/a: ValueError: op 1 (AccSoftmaxBatch_v1{Q=1,K=2}): on | 0.33 | - |  |
| t6_softmax_epilogue_F | 22 | W/4 (6) | W/2 (11) | 32 | 12 | n/a: ValueError: op 1 (AccSoftmaxBatch_v1{Q=1,K=2}): on | 0.38 | - |  |
| t6_softmax_epilogue_F | 22 | W/4 (6) | inf (-) | 32 | 12 | n/a: ValueError: op 1 (AccSoftmaxBatch_v1{Q=1,K=2}): on | 0.38 | - |  |
| t6_softmax_epilogue_F | 22 | W/2 (11) | W/4 (6) | 36 | 12 | n/a: ValueError: op 1 (AccSoftmaxBatch_v1{Q=1,K=2}): on | 0.33 | - |  |
| t6_softmax_epilogue_F | 22 | W/2 (11) | W/2 (11) | 24 | 12 | n/a: ValueError: op 1 (AccSoftmaxBatch_v1{Q=1,K=2}): on | 0.50 | - |  |
| t6_softmax_epilogue_F | 22 | W/2 (11) | inf (-) | 24 | 12 | n/a: ValueError: op 1 (AccSoftmaxBatch_v1{Q=1,K=2}): on | 0.50 | - |  |
| t6_softmax_epilogue_F | 22 | inf (-) | W/4 (6) | 36 | 12 | n/a: ValueError: op 1 (AccSoftmaxBatch_v1{Q=1,K=2}): on | 0.33 | - |  |
| t6_softmax_epilogue_F | 22 | inf (-) | W/2 (11) | 24 | 12 | n/a: ValueError: op 1 (AccSoftmaxBatch_v1{Q=1,K=2}): on | 0.50 | - |  |
| t6_softmax_epilogue_F | 22 | inf (-) | inf (-) | 12 | 12 | 12 | 1.00 | 1.00 |  |
| t7_single_leaf_overlap_share | 12 | W/4 (2) | W/4 (2) | 16 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=1,CH=1}) | 0.62 | - |  |
| t7_single_leaf_overlap_share | 12 | W/4 (2) | W/2 (4) | 12 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=1,CH=1}) | 0.83 | - |  |
| t7_single_leaf_overlap_share | 12 | W/4 (2) | inf (-) | 10 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=1,CH=1}) | 1.00 | - |  |
| t7_single_leaf_overlap_share | 12 | W/2 (4) | W/4 (2) | 16 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=1,CH=1}) | 0.62 | - |  |
| t7_single_leaf_overlap_share | 12 | W/2 (4) | W/2 (4) | 12 | 10 | 12 | 0.83 | 1.00 |  |
| t7_single_leaf_overlap_share | 12 | W/2 (4) | inf (-) | 10 | 10 | 10 | 1.00 | 1.00 |  |
| t7_single_leaf_overlap_share | 12 | inf (-) | W/4 (2) | 16 | 10 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=1,CH=1}) | 0.62 | - |  |
| t7_single_leaf_overlap_share | 12 | inf (-) | W/2 (4) | 12 | 10 | 12 | 0.83 | 1.00 |  |
| t7_single_leaf_overlap_share | 12 | inf (-) | inf (-) | 10 | 10 | 10 | 1.00 | 1.00 |  |
| t9_scalar_read_by_two_matmuls | 6 | W/4 (1) | W/4 (1) | 12 | 2 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.17 | - |  |
| t9_scalar_read_by_two_matmuls | 6 | W/4 (1) | W/2 (2) | 10 | 2 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.20 | - |  |
| t9_scalar_read_by_two_matmuls | 6 | W/4 (1) | inf (-) | 10 | 2 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.20 | - |  |
| t9_scalar_read_by_two_matmuls | 6 | W/2 (2) | W/4 (1) | 12 | 2 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.17 | - |  |
| t9_scalar_read_by_two_matmuls | 6 | W/2 (2) | W/2 (2) | 4 | 2 | 4 | 0.50 | 1.00 |  |
| t9_scalar_read_by_two_matmuls | 6 | W/2 (2) | inf (-) | 2 | 2 | 2 | 1.00 | 1.00 |  |
| t9_scalar_read_by_two_matmuls | 6 | inf (-) | W/4 (1) | 12 | 2 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=1,K=1,CH=1}) | 0.17 | - |  |
| t9_scalar_read_by_two_matmuls | 6 | inf (-) | W/2 (2) | 4 | 2 | 4 | 0.50 | 1.00 |  |
| t9_scalar_read_by_two_matmuls | 6 | inf (-) | inf (-) | 2 | 2 | 2 | 1.00 | 1.00 |  |
| t10_last_leaf_overlap | 12 | W/4 (2) | W/4 (2) | 8 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=3,K=1,CH=1}) | 0.75 | - |  |
| t10_last_leaf_overlap | 12 | W/4 (2) | W/2 (4) | 6 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=3,K=1,CH=1}) | 1.00 | - |  |
| t10_last_leaf_overlap | 12 | W/4 (2) | inf (-) | 6 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=3,K=1,CH=1}) | 1.00 | - |  |
| t10_last_leaf_overlap | 12 | W/2 (4) | W/4 (2) | 8 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=3,K=1,CH=1}) | 0.75 | - |  |
| t10_last_leaf_overlap | 12 | W/2 (4) | W/2 (4) | 6 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=3,K=1,CH=1}) | 1.00 | - |  |
| t10_last_leaf_overlap | 12 | W/2 (4) | inf (-) | 6 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=3,K=1,CH=1}) | 1.00 | - |  |
| t10_last_leaf_overlap | 12 | inf (-) | W/4 (2) | 8 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=3,K=1,CH=1}) | 0.75 | - |  |
| t10_last_leaf_overlap | 12 | inf (-) | W/2 (4) | 6 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=3,K=1,CH=1}) | 1.00 | - |  |
| t10_last_leaf_overlap | 12 | inf (-) | inf (-) | 6 | 6 | 6 | 1.00 | 1.00 |  |
| t11_reprow_B_deep | 23 | W/4 (5) | W/4 (5) | 24 | 10 | n/a: ValueError: op 2 (AccMatmulT_v1{M=1,N=4,K=1,CH=1}) | 0.42 | - |  |
| t11_reprow_B_deep | 23 | W/4 (5) | W/2 (9) | 20 | 10 | n/a: ValueError: op 2 (AccMatmulT_v1{M=1,N=4,K=1,CH=1}) | 0.50 | - |  |
| t11_reprow_B_deep | 23 | W/4 (5) | inf (-) | 20 | 10 | n/a: ValueError: op 2 (AccMatmulT_v1{M=1,N=4,K=1,CH=1}) | 0.50 | - |  |
| t11_reprow_B_deep | 23 | W/2 (9) | W/4 (5) | 24 | 10 | n/a: ValueError: op 2 (AccMatmulT_v1{M=1,N=4,K=1,CH=1}) | 0.42 | - |  |
| t11_reprow_B_deep | 23 | W/2 (9) | W/2 (9) | 18 | 10 | 20 | 0.56 | 1.11 |  |
| t11_reprow_B_deep | 23 | W/2 (9) | inf (-) | 18 | 10 | 20 | 0.56 | 1.11 |  |
| t11_reprow_B_deep | 23 | inf (-) | W/4 (5) | 24 | 10 | n/a: ValueError: op 2 (AccMatmulT_v1{M=1,N=4,K=1,CH=1}) | 0.42 | - |  |
| t11_reprow_B_deep | 23 | inf (-) | W/2 (9) | 18 | 10 | 20 | 0.56 | 1.11 |  |
| t11_reprow_B_deep | 23 | inf (-) | inf (-) | 10 | 10 | 10 | 1.00 | 1.00 |  |
| t8_add_same_tensor_twice | 4 | W/4 (1) | W/4 (1) | 8 | 8 | n/a: ValueError: op 0 (AccAddBatch_v1{Q=2,K=2}): one ro | 1.00 | - |  |
| t8_add_same_tensor_twice | 4 | W/4 (1) | W/2 (2) | 8 | 8 | n/a: ValueError: op 0 (AccAddBatch_v1{Q=2,K=2}): one ro | 1.00 | - |  |
| t8_add_same_tensor_twice | 4 | W/4 (1) | inf (-) | 8 | 8 | n/a: ValueError: op 0 (AccAddBatch_v1{Q=2,K=2}): one ro | 1.00 | - |  |
| t8_add_same_tensor_twice | 4 | W/2 (2) | W/4 (1) | 8 | 8 | n/a: ValueError: op 0 (AccAddBatch_v1{Q=2,K=2}): one ro | 1.00 | - |  |
| t8_add_same_tensor_twice | 4 | W/2 (2) | W/2 (2) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| t8_add_same_tensor_twice | 4 | W/2 (2) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| t8_add_same_tensor_twice | 4 | inf (-) | W/4 (1) | 8 | 8 | n/a: ValueError: op 0 (AccAddBatch_v1{Q=2,K=2}): one ro | 1.00 | - |  |
| t8_add_same_tensor_twice | 4 | inf (-) | W/2 (2) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| t8_add_same_tensor_twice | 4 | inf (-) | inf (-) | 8 | 8 | 8 | 1.00 | 1.00 |  |
| t12_attn_decode_kv_undercount | 30 | W/4 (7) | W/4 (7) | 44 | 20 | n/a: ValueError: op 1 (AccSoftmaxBatch_v1{Q=1,K=2}): on | 0.45 | - |  |
| t12_attn_decode_kv_undercount | 30 | W/4 (7) | W/2 (14) | 40 | 20 | n/a: ValueError: op 1 (AccSoftmaxBatch_v1{Q=1,K=2}): on | 0.50 | - |  |
| t12_attn_decode_kv_undercount | 30 | W/4 (7) | inf (-) | 40 | 20 | n/a: ValueError: op 1 (AccSoftmaxBatch_v1{Q=1,K=2}): on | 0.50 | - |  |
| t12_attn_decode_kv_undercount | 30 | W/2 (14) | W/4 (7) | 44 | 20 | n/a: ValueError: op 1 (AccSoftmaxBatch_v1{Q=1,K=2}): on | 0.45 | - |  |
| t12_attn_decode_kv_undercount | 30 | W/2 (14) | W/2 (14) | 32 | 20 | n/a: ValueError: op 1 (AccSoftmaxBatch_v1{Q=1,K=2}): on | 0.62 | - |  |
| t12_attn_decode_kv_undercount | 30 | W/2 (14) | inf (-) | 28 | 20 | n/a: ValueError: op 1 (AccSoftmaxBatch_v1{Q=1,K=2}): on | 0.71 | - |  |
| t12_attn_decode_kv_undercount | 30 | inf (-) | W/4 (7) | 44 | 20 | n/a: ValueError: op 1 (AccSoftmaxBatch_v1{Q=1,K=2}): on | 0.45 | - |  |
| t12_attn_decode_kv_undercount | 30 | inf (-) | W/2 (14) | 32 | 20 | n/a: ValueError: op 1 (AccSoftmaxBatch_v1{Q=1,K=2}): on | 0.62 | - |  |
| t12_attn_decode_kv_undercount | 30 | inf (-) | inf (-) | 20 | 20 | 20 | 1.00 | 1.00 |  |
| k1_matmul_G_forced_split | 16 | W/4 (3) | W/4 (3) | 32 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 0.50 | - |  |
| k1_matmul_G_forced_split | 16 | W/4 (3) | W/2 (6) | 24 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 0.67 | - |  |
| k1_matmul_G_forced_split | 16 | W/4 (3) | inf (-) | 16 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 1.00 | - |  |
| k1_matmul_G_forced_split | 16 | W/2 (6) | W/4 (3) | 32 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 0.50 | - |  |
| k1_matmul_G_forced_split | 16 | W/2 (6) | W/2 (6) | 24 | 16 | 24 | 0.67 | 1.00 |  |
| k1_matmul_G_forced_split | 16 | W/2 (6) | inf (-) | 16 | 16 | 24 | 1.00 | 1.50 |  |
| k1_matmul_G_forced_split | 16 | inf (-) | W/4 (3) | 32 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=2,CH=1}) | 0.50 | - |  |
| k1_matmul_G_forced_split | 16 | inf (-) | W/2 (6) | 24 | 16 | 24 | 0.67 | 1.00 |  |
| k1_matmul_G_forced_split | 16 | inf (-) | inf (-) | 16 | 16 | 16 | 1.00 | 1.00 |  |
| i1_inference_mlp | 28 | W/4 (6) | W/4 (6) | 24 | 4 | 28 | 0.17 | 1.17 |  |
| i1_inference_mlp | 28 | W/4 (6) | W/2 (12) | 18 | 4 | 24 | 0.22 | 1.33 |  |
| i1_inference_mlp | 28 | W/4 (6) | inf (-) | 18 | 4 | 24 | 0.22 | 1.33 |  |
| i1_inference_mlp | 28 | W/2 (12) | W/4 (6) | 24 | 4 | 28 | 0.17 | 1.17 |  |
| i1_inference_mlp | 28 | W/2 (12) | W/2 (12) | 12 | 4 | 16 | 0.33 | 1.33 |  |
| i1_inference_mlp | 28 | W/2 (12) | inf (-) | 10 | 4 | 16 | 0.40 | 1.60 |  |
| i1_inference_mlp | 28 | inf (-) | W/4 (6) | 24 | 4 | 28 | 0.17 | 1.17 |  |
| i1_inference_mlp | 28 | inf (-) | W/2 (12) | 12 | 4 | 16 | 0.33 | 1.33 |  |
| i1_inference_mlp | 28 | inf (-) | inf (-) | 4 | 4 | 4 | 1.00 | 1.00 |  |
| r1_train_linear_sgd | 28 | W/4 (6) | W/4 (6) | 44 | 20 | n/a: ValueError: op 1 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.45 | - |  |
| r1_train_linear_sgd | 28 | W/4 (6) | W/2 (11) | 28 | 20 | n/a: ValueError: op 1 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.71 | - |  |
| r1_train_linear_sgd | 28 | W/4 (6) | inf (-) | 20 | 20 | n/a: ValueError: op 1 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 1.00 | - |  |
| r1_train_linear_sgd | 28 | W/2 (11) | W/4 (6) | 44 | 20 | n/a: ValueError: op 1 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.45 | - |  |
| r1_train_linear_sgd | 28 | W/2 (11) | W/2 (11) | 28 | 20 | 40 | 0.71 | 1.43 |  |
| r1_train_linear_sgd | 28 | W/2 (11) | inf (-) | 20 | 20 | 36 | 1.00 | 1.80 |  |
| r1_train_linear_sgd | 28 | inf (-) | W/4 (6) | 44 | 20 | n/a: ValueError: op 1 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.45 | - |  |
| r1_train_linear_sgd | 28 | inf (-) | W/2 (11) | 28 | 20 | 40 | 0.71 | 1.43 |  |
| r1_train_linear_sgd | 28 | inf (-) | inf (-) | 20 | 20 | 20 | 1.00 | 1.00 |  |
| r2_train_two_layer | 36 | W/4 (7) | W/4 (7) | [48, 56] | 28 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.50 | - |  |
| r2_train_two_layer | 36 | W/4 (7) | W/2 (14) | 40 | 28 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.70 | - |  |
| r2_train_two_layer | 36 | W/4 (7) | inf (-) | 32 | 28 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.88 | - |  |
| r2_train_two_layer | 36 | W/2 (14) | W/4 (7) | [48, 56] | 28 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.50 | - |  |
| r2_train_two_layer | 36 | W/2 (14) | W/2 (14) | 40 | 28 | 48 | 0.70 | 1.20 |  |
| r2_train_two_layer | 36 | W/2 (14) | inf (-) | 28 | 28 | 44 | 1.00 | 1.57 |  |
| r2_train_two_layer | 36 | inf (-) | W/4 (7) | [48, 56] | 28 | n/a: ValueError: op 2 (AccMatmulTT_v1{M=2,N=2,K=1,CH=1} | 0.50 | - |  |
| r2_train_two_layer | 36 | inf (-) | W/2 (14) | 40 | 28 | 48 | 0.70 | 1.20 |  |
| r2_train_two_layer | 36 | inf (-) | inf (-) | 28 | 28 | 28 | 1.00 | 1.00 |  |

### fuzz (worst U/I* and any L/I* < 0.5)

| circuit | family | gates | F | G | I* | L | U (lit) | L/I* | U/I* |
|---|---|---|---|---|---|---|---|---|---|
| fuzz7-29 | gain+matmul+mul|shared,view,chain | 20 | W/2 | inf | 4 | 4 | 20 | 1.00 | 5.00 |
| fuzz7-162 | matmul|shared,view | 30 | W/4 | inf | 12 | 12 | 60 | 1.00 | 5.00 |
| fuzz7-5 | colsum+mul|shared,chain | 22 | W/2 | W/2 | 12 | 12 | 40 | 1.00 | 3.33 |
| fuzz7-180 | matmul+mul|shared,view,chain | 24 | inf | W/2 | 6 | 4 | 20 | 0.67 | 3.33 |
| fuzz7-85 | gain+matmul|chain | 21 | W/2 | inf | 10 | 10 | 32 | 1.00 | 3.20 |
| fuzz7-22 | add+embed+matmul_tt|shared,chain | 22 | W/4 | inf | 48 | 48 | 148 | 1.00 | 3.08 |
| fuzz7-24 | gain | 12 | W/2 | W/2 | 8 | 8 | 24 | 1.00 | 3.00 |
| fuzz7-24 | gain | 12 | W/2 | inf | 8 | 8 | 24 | 1.00 | 3.00 |
| fuzz7-129 | matmul|shared | 45 | W/2 | inf | 18 | 18 | 54 | 1.00 | 3.00 |
| fuzz7-134 | add+matmul|shared,chain | 30 | W/2 | W/2 | 12 | 12 | 36 | 1.00 | 3.00 |
| fuzz7-69 | add+embed+gain+matmul|shared,view,chain | 26 | W/4 | W/2 | 16 | 12 | 44 | 0.75 | 2.75 |
| fuzz7-51 | mul+rowscale|shared,view,chain | 24 | W/2 | inf | 14 | 14 | 38 | 1.00 | 2.71 |
| fuzz7-155 | embed+matmul | 18 | W/2 | W/4 | 30 | 30 | 78 | 1.00 | 2.60 |
| fuzz7-16 | colsum | 20 | inf | W/2 | 16 | 16 | 40 | 1.00 | 2.50 |
| fuzz7-44 | colsum | 30 | W/2 | inf | 24 | 24 | 60 | 1.00 | 2.50 |
| fuzz7-13 | matmul|view | 18 | inf | W/4 | 56 | 24 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=3,K=4,CH=1}) | 0.43 | - |
| fuzz7-13 | matmul|view | 18 | W/2 | W/4 | 56 | 24 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=3,K=4,CH=1}) | 0.43 | - |
| fuzz7-18 | add+matmul+mul+rowscale|shared,chain | 19 | W/2 | W/4 | 20 | 8 | 26 | 0.40 | 1.30 |
| fuzz7-18 | add+matmul+mul+rowscale|shared,chain | 19 | W/4 | W/4 | 20 | 8 | 26 | 0.40 | 1.30 |
| fuzz7-18 | add+matmul+mul+rowscale|shared,chain | 19 | inf | W/4 | 20 | 8 | 26 | 0.40 | 1.30 |
| fuzz7-23 | add+colsum+mul|shared,view,chain | 10 | inf | W/2 | 8 | 2 | 18 | 0.25 | 2.25 |
| fuzz7-23 | add+colsum+mul|shared,view,chain | 10 | inf | W/4 | 14 | 2 | n/a: ValueError: op 1 (AccColSumBatch_v1{Q=3,K=1}): one | 0.14 | - |
| fuzz7-23 | add+colsum+mul|shared,view,chain | 10 | W/4 | W/4 | 14 | 2 | n/a: ValueError: op 1 (AccColSumBatch_v1{Q=3,K=1}): one | 0.14 | - |
| fuzz7-23 | add+colsum+mul|shared,view,chain | 10 | W/4 | inf | 14 | 2 | n/a: ValueError: op 1 (AccColSumBatch_v1{Q=3,K=1}): one | 0.14 | - |
| fuzz7-25 | embed+matmul|shared,chain | 30 | inf | W/2 | 14 | 6 | 30 | 0.43 | 2.14 |
| fuzz7-25 | embed+matmul|shared,chain | 30 | W/4 | W/4 | 24 | 6 | 36 | 0.25 | 1.50 |
| fuzz7-25 | embed+matmul|shared,chain | 30 | W/2 | W/4 | 24 | 6 | 36 | 0.25 | 1.50 |
| fuzz7-29 | gain+matmul+mul|shared,view,chain | 20 | W/4 | W/2 | 14 | 4 | n/a: ValueError: op 1 (AccMatmulT_v1{M=1,N=2,K=3,CH=1}) | 0.29 | - |
| fuzz7-29 | gain+matmul+mul|shared,view,chain | 20 | inf | W/4 | 20 | 4 | n/a: ValueError: op 1 (AccMatmulT_v1{M=1,N=2,K=3,CH=1}) | 0.20 | - |
| fuzz7-29 | gain+matmul+mul|shared,view,chain | 20 | W/4 | W/4 | 20 | 4 | n/a: ValueError: op 1 (AccMatmulT_v1{M=1,N=2,K=3,CH=1}) | 0.20 | - |
| fuzz7-33 | colsum | 8 | W/4 | inf | 12 | 4 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=1,K=2}): one | 0.33 | - |
| fuzz7-33 | colsum | 8 | W/4 | W/4 | 12 | 4 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=1,K=2}): one | 0.33 | - |
| fuzz7-34 | embed | 3 | W/2 | W/4 | 12 | 4 | 12 | 0.33 | 1.00 |
| fuzz7-37 | embed+matmul|shared,view | 27 | inf | W/4 | 30 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=4,K=3,CH=1}) | 0.20 | - |
| fuzz7-37 | embed+matmul|shared,view | 27 | W/4 | W/2 | 18 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=4,K=3,CH=1}) | 0.33 | - |
| fuzz7-37 | embed+matmul|shared,view | 27 | W/2 | W/4 | 30 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=4,K=3,CH=1}) | 0.20 | - |
| fuzz7-41 | swiglu|shared,view | 3 | W/2 | inf | 6 | 0 | n/a: ValueError: op 0 (AccSwiGluBatch_v1{Q=1,FF=3}): on | 0.00 | - |
| fuzz7-41 | swiglu|shared,view | 3 | W/2 | W/2 | 6 | 0 | n/a: ValueError: op 0 (AccSwiGluBatch_v1{Q=1,FF=3}): on | 0.00 | - |
| fuzz7-41 | swiglu|shared,view | 3 | inf | W/4 | 6 | 0 | n/a: ValueError: op 0 (AccSwiGluBatch_v1{Q=1,FF=3}): on | 0.00 | - |
| fuzz7-41 | swiglu|shared,view | 3 | inf | W/2 | 6 | 0 | n/a: ValueError: op 0 (AccSwiGluBatch_v1{Q=1,FF=3}): on | 0.00 | - |
| fuzz7-49 | gain+matmul|shared | 30 | W/4 | W/4 | 52 | 24 | 52 | 0.46 | 1.00 |
| fuzz7-51 | mul+rowscale|shared,view,chain | 24 | W/2 | W/4 | 32 | 14 | 54 | 0.44 | 1.69 |
| fuzz7-51 | mul+rowscale|shared,view,chain | 24 | W/4 | W/4 | 32 | 14 | 54 | 0.44 | 1.69 |
| fuzz7-56 | matmul|shared | 27 | W/4 | W/4 | 18 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=3,K=1,CH=1}) | 0.33 | - |
| fuzz7-60 | matmul|shared | 45 | W/2 | W/4 | [40, 58] | 18 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=3,K=3,CH=1}) | 0.31 | - |
| fuzz7-60 | matmul|shared | 45 | inf | W/4 | [40, 58] | 18 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=3,K=3,CH=1}) | 0.31 | - |
| fuzz7-61 | embed+matmul+mul+rowscale|shared,view,chain | 18 | W/2 | W/4 | 34 | 16 | 46 | 0.47 | 1.35 |
| fuzz7-61 | embed+matmul+mul+rowscale|shared,view,chain | 18 | W/4 | W/4 | 34 | 16 | 46 | 0.47 | 1.35 |
| fuzz7-68 | matmul|view | 12 | W/2 | W/4 | 40 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=1,K=4,CH=1}) | 0.40 | - |
| fuzz7-68 | matmul|view | 12 | W/4 | W/2 | 34 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=1,K=4,CH=1}) | 0.47 | - |
| fuzz7-69 | add+embed+gain+matmul|shared,view,chain | 26 | W/2 | W/4 | 30 | 12 | 44 | 0.40 | 1.47 |
| fuzz7-73 | matmul_tt|shared | 27 | W/2 | W/4 | 18 | 6 | n/a: ValueError: op 0 (AccMatmulTT_v1{M=3,N=3,K=1,CH=1} | 0.33 | - |
| fuzz7-73 | matmul_tt|shared | 27 | W/4 | W/4 | 18 | 6 | n/a: ValueError: op 0 (AccMatmulTT_v1{M=3,N=3,K=1,CH=1} | 0.33 | - |
| fuzz7-75 | matmul|shared,view | 24 | W/2 | W/4 | 32 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=3,K=2,CH=1}) | 0.25 | - |
| fuzz7-75 | matmul|shared,view | 24 | inf | W/4 | 32 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=3,K=2,CH=1}) | 0.25 | - |
| fuzz7-79 | gain+matmul_tt|shared | 24 | W/4 | W/4 | 20 | 8 | n/a: ValueError: op 0 (AccMatmulTT_v1{M=2,N=2,K=2,CH=1} | 0.40 | - |
| fuzz7-79 | gain+matmul_tt|shared | 24 | inf | W/4 | 20 | 8 | n/a: ValueError: op 0 (AccMatmulTT_v1{M=2,N=2,K=2,CH=1} | 0.40 | - |
| fuzz7-85 | gain+matmul|chain | 21 | inf | W/2 | 22 | 10 | 32 | 0.45 | 1.45 |
| fuzz7-85 | gain+matmul|chain | 21 | W/4 | W/4 | 32 | 10 | 32 | 0.31 | 1.00 |
| fuzz7-100 | softmax | 26 | W/4 | W/2 | 38 | 8 | n/a: ValueError: op 0 (AccSoftmaxBatch_v1{Q=1,K=4}): on | 0.21 | - |
| fuzz7-100 | softmax | 26 | W/4 | inf | 36 | 8 | n/a: ValueError: op 0 (AccSoftmaxBatch_v1{Q=1,K=4}): on | 0.22 | - |
| fuzz7-100 | softmax | 26 | W/2 | W/2 | 28 | 8 | n/a: ValueError: op 0 (AccSoftmaxBatch_v1{Q=1,K=4}): on | 0.29 | - |
| fuzz7-100 | softmax | 26 | W/4 | W/4 | 46 | 8 | n/a: ValueError: op 0 (AccSoftmaxBatch_v1{Q=1,K=4}): on | 0.17 | - |
| fuzz7-106 | colsum+gain|view | 13 | inf | W/2 | 18 | 8 | 26 | 0.44 | 1.44 |
| fuzz7-106 | colsum+gain|view | 13 | inf | W/4 | 26 | 8 | n/a: ValueError: op 1 (AccColSumBatch_v1{Q=4,K=1}): one | 0.31 | - |
| fuzz7-106 | colsum+gain|view | 13 | W/4 | W/4 | 26 | 8 | n/a: ValueError: op 1 (AccColSumBatch_v1{Q=4,K=1}): one | 0.31 | - |
| fuzz7-111 | colsum|view | 12 | W/4 | inf | 16 | 0 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=2,K=2}): one | 0.00 | - |
| fuzz7-111 | colsum|view | 12 | W/4 | W/2 | 16 | 0 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=2,K=2}): one | 0.00 | - |
| fuzz7-111 | colsum|view | 12 | W/2 | W/4 | 16 | 0 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=2,K=2}): one | 0.00 | - |
| fuzz7-111 | colsum|view | 12 | W/4 | W/4 | 16 | 0 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=2,K=2}): one | 0.00 | - |
| fuzz7-112 | matmul|shared | 24 | inf | W/4 | 44 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=4,CH=1}) | 0.36 | - |
| fuzz7-112 | matmul|shared | 24 | W/4 | W/4 | 44 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=4,CH=1}) | 0.36 | - |
| fuzz7-121 | matmul|shared | 24 | inf | W/4 | 44 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=4,CH=1}) | 0.36 | - |
| fuzz7-123 | embed | 6 | W/2 | W/4 | 72 | 32 | n/a: ValueError: op 0 (AccEmbedBatch_v1{Q=2,V=4,D=3}):  | 0.44 | - |
| fuzz7-128 | colsum+softmax|shared | 22 | W/2 | W/4 | 24 | 4 | n/a: ValueError: op 0 (AccSoftmaxBatch_v1{Q=2,K=1}): on | 0.17 | - |
| fuzz7-128 | colsum+softmax|shared | 22 | W/4 | W/2 | 20 | 4 | n/a: ValueError: op 0 (AccSoftmaxBatch_v1{Q=2,K=1}): on | 0.20 | - |
| fuzz7-128 | colsum+softmax|shared | 22 | W/4 | W/4 | 24 | 4 | n/a: ValueError: op 0 (AccSoftmaxBatch_v1{Q=2,K=1}): on | 0.17 | - |
| fuzz7-135 | gain+matmul+mul|shared,chain | 18 | inf | W/4 | 16 | 6 | 22 | 0.38 | 1.38 |
| fuzz7-135 | gain+matmul+mul|shared,chain | 18 | W/4 | W/4 | 16 | 6 | 22 | 0.38 | 1.38 |
| fuzz7-141 | embed+matmul+rowscale|shared | 11 | inf | W/4 | 34 | 14 | n/a: ValueError: op 1 (AccRowScaleBatch_v1{Q=1,K=4}): o | 0.41 | - |
| fuzz7-141 | embed+matmul+rowscale|shared | 11 | W/2 | W/4 | 34 | 14 | n/a: ValueError: op 1 (AccRowScaleBatch_v1{Q=1,K=4}): o | 0.41 | - |
| fuzz7-151 | matmul+rmsnorm|view,chain | 30 | W/2 | W/4 | 52 | 24 | n/a: ValueError: op 0 (AccRmsNormBatch_v1{Q=1,K=4}): on | 0.46 | - |
| fuzz7-151 | matmul+rmsnorm|view,chain | 30 | W/4 | W/4 | 52 | 24 | n/a: ValueError: op 0 (AccRmsNormBatch_v1{Q=1,K=4}): on | 0.46 | - |
| fuzz7-151 | matmul+rmsnorm|view,chain | 30 | inf | W/4 | 52 | 24 | n/a: ValueError: op 0 (AccRmsNormBatch_v1{Q=1,K=4}): on | 0.46 | - |
| fuzz7-152 | mul|view | 2 | W/2 | W/4 | 4 | 0 | n/a: ValueError: op 0 (AccMulBatch_v1{Q=1,K=2}): one ro | 0.00 | - |
| fuzz7-152 | mul|view | 2 | inf | W/2 | 4 | 0 | n/a: ValueError: op 0 (AccMulBatch_v1{Q=1,K=2}): one ro | 0.00 | - |
| fuzz7-152 | mul|view | 2 | inf | W/4 | 4 | 0 | n/a: ValueError: op 0 (AccMulBatch_v1{Q=1,K=2}): one ro | 0.00 | - |
| fuzz7-152 | mul|view | 2 | W/4 | W/4 | 4 | 0 | n/a: ValueError: op 0 (AccMulBatch_v1{Q=1,K=2}): one ro | 0.00 | - |
| fuzz7-156 | gain+matmul+rmsnorm|shared,view | 16 | W/4 | W/2 | 20 | 8 | n/a: ValueError: op 0 (AccRmsNormBatch_v1{Q=1,K=2}): on | 0.40 | - |
| fuzz7-156 | gain+matmul+rmsnorm|shared,view | 16 | W/4 | W/4 | 26 | 8 | n/a: ValueError: op 0 (AccRmsNormBatch_v1{Q=1,K=2}): on | 0.31 | - |
| fuzz7-157 | softmax | 28 | W/4 | inf | 32 | 8 | n/a: ValueError: op 0 (AccSoftmaxBatch_v1{Q=2,K=2}): on | 0.25 | - |
| fuzz7-157 | softmax | 28 | inf | W/4 | 36 | 8 | n/a: ValueError: op 0 (AccSoftmaxBatch_v1{Q=2,K=2}): on | 0.22 | - |
| fuzz7-157 | softmax | 28 | W/4 | W/2 | 32 | 8 | n/a: ValueError: op 0 (AccSoftmaxBatch_v1{Q=2,K=2}): on | 0.25 | - |
| fuzz7-159 | rmsnorm | 28 | W/4 | W/4 | 32 | 12 | n/a: ValueError: op 0 (AccRmsNormBatch_v1{Q=2,K=3}): on | 0.38 | - |
| fuzz7-159 | rmsnorm | 28 | W/2 | W/4 | 32 | 12 | n/a: ValueError: op 0 (AccRmsNormBatch_v1{Q=2,K=3}): on | 0.38 | - |
| fuzz7-162 | matmul|shared,view | 30 | W/4 | W/4 | 46 | 12 | 60 | 0.26 | 1.30 |
| fuzz7-164 | add+matmul+rowscale+softmax|shared,chain | 30 | W/4 | inf | 22 | 2 | n/a: ValueError: op 2 (AccSoftmaxBatch_v1{Q=1,K=3}): on | 0.09 | - |
| fuzz7-164 | add+matmul+rowscale+softmax|shared,chain | 30 | W/4 | W/2 | 22 | 2 | n/a: ValueError: op 2 (AccSoftmaxBatch_v1{Q=1,K=3}): on | 0.09 | - |
| fuzz7-164 | add+matmul+rowscale+softmax|shared,chain | 30 | W/2 | W/2 | 14 | 2 | n/a: ValueError: op 2 (AccSoftmaxBatch_v1{Q=1,K=3}): on | 0.14 | - |
| fuzz7-164 | add+matmul+rowscale+softmax|shared,chain | 30 | W/4 | W/4 | 30 | 2 | n/a: ValueError: op 2 (AccSoftmaxBatch_v1{Q=1,K=3}): on | 0.07 | - |
| fuzz7-166 | add+matmul_tt|shared,view | 15 | inf | W/4 | 6 | 2 | n/a: ValueError: op 0 (AccMatmulTT_v1{M=3,N=1,K=1,CH=1} | 0.33 | - |
| fuzz7-169 | matmul+matmul_tt|shared,view,chain | 20 | W/2 | W/4 | 26 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=2,K=2,CH=1}) | 0.31 | - |
| fuzz7-171 | matmul|shared,view | 24 | W/4 | W/4 | 32 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=4,K=4,CH=1}) | 0.25 | - |
| fuzz7-171 | matmul|shared,view | 24 | inf | W/4 | 32 | 8 | n/a: ValueError: op 0 (AccMatmulT_v1{M=1,N=4,K=4,CH=1}) | 0.25 | - |
| fuzz7-175 | matmul|shared | 24 | W/2 | W/4 | 44 | 16 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=2,K=4,CH=1}) | 0.36 | - |
| fuzz7-180 | matmul+mul|shared,view,chain | 24 | inf | W/4 | 16 | 4 | 24 | 0.25 | 1.50 |
| fuzz7-180 | matmul+mul|shared,view,chain | 24 | W/4 | W/4 | 16 | 4 | 24 | 0.25 | 1.50 |
| fuzz7-184 | matmul|shared | 27 | inf | W/4 | 18 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=3,K=1,CH=1}) | 0.33 | - |
| fuzz7-184 | matmul|shared | 27 | W/2 | W/4 | 18 | 6 | n/a: ValueError: op 0 (AccMatmulT_v1{M=3,N=3,K=1,CH=1}) | 0.33 | - |
| fuzz7-185 | softmax | 20 | W/2 | W/2 | 24 | 6 | n/a: ValueError: op 0 (AccSoftmaxBatch_v1{Q=1,K=3}): on | 0.25 | - |
| fuzz7-185 | softmax | 20 | inf | W/2 | 24 | 6 | n/a: ValueError: op 0 (AccSoftmaxBatch_v1{Q=1,K=3}): on | 0.25 | - |
| fuzz7-185 | softmax | 20 | W/2 | inf | 22 | 6 | n/a: ValueError: op 0 (AccSoftmaxBatch_v1{Q=1,K=3}): on | 0.27 | - |
| fuzz7-185 | softmax | 20 | W/4 | W/4 | 40 | 6 | n/a: ValueError: op 0 (AccSoftmaxBatch_v1{Q=1,K=3}): on | 0.15 | - |
| fuzz7-190 | colsum | 8 | W/4 | W/2 | 12 | 4 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=1,K=2}): one | 0.33 | - |
| fuzz7-190 | colsum | 8 | inf | W/4 | 12 | 4 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=1,K=2}): one | 0.33 | - |
| fuzz7-190 | colsum | 8 | W/4 | W/4 | 12 | 4 | n/a: ValueError: op 0 (AccColSumBatch_v1{Q=1,K=2}): one | 0.33 | - |
| fuzz7-195 | gain+softmax|shared | 9 | W/4 | inf | 18 | 4 | n/a: ValueError: op 1 (AccSoftmaxBatch_v1{Q=1,K=1}): on | 0.22 | - |
| fuzz7-195 | gain+softmax|shared | 9 | W/4 | W/2 | 18 | 4 | n/a: ValueError: op 1 (AccSoftmaxBatch_v1{Q=1,K=1}): on | 0.22 | - |
| fuzz7-195 | gain+softmax|shared | 9 | W/2 | inf | 12 | 4 | n/a: ValueError: op 1 (AccSoftmaxBatch_v1{Q=1,K=1}): on | 0.33 | - |
| fuzz7-195 | gain+softmax|shared | 9 | inf | W/2 | 14 | 4 | n/a: ValueError: op 1 (AccSoftmaxBatch_v1{Q=1,K=1}): on | 0.29 | - |
| fuzz7-198 | embed | 12 | inf | W/4 | 64 | 28 | n/a: ValueError: op 0 (AccEmbedBatch_v1{Q=3,V=2,D=4}):  | 0.44 | - |
| fuzz7-199 | matmul|shared,view | 6 | W/4 | W/2 | 10 | 2 | n/a: ValueError: op 0 (AccMatmulT_v1{M=2,N=1,K=1,CH=1}) | 0.20 | - |

`upper_coarse` failures on fuzz circuits (count, message):

- 11 x `ValueError:  op 0 (AccColSumBatch_v1{Q=2,K=2})  one contraction chunk exceeds min(`
- 10 x `ValueError:  op 0 (AccMatmulT_v1{M=1,N=2,K=4,CH=1})  one row has work 10 > min(F, `
- 10 x `ValueError:  op 0 (AccMatmulT_v1{M=2,N=2,K=4,CH=1})  one row has work 10 > min(F, `
- 9 x `ValueError:  op 0 (AccMatmulTT_v1{M=2,N=2,K=2,CH=1})  one contraction chunk exceed`
- 8 x `ValueError:  op 0 (AccMulBatch_v1{Q=1,K=2})  one row has work 2 > min(F, G)=1`
- 8 x `ValueError:  op 0 (AccMatmulT_v1{M=1,N=3,K=4,CH=1})  one row has work 15 > min(F, `
- 8 x `ValueError:  op 1 (AccMatmulT_v1{M=1,N=2,K=3,CH=1})  one row has work 8 > min(F, G`
- 8 x `ValueError:  op 0 (AccColSumBatch_v1{Q=4,K=3})  one contraction chunk exceeds min(`
- 8 x `ValueError:  op 0 (AccMatmulT_v1{M=3,N=3,K=1,CH=1})  one row has work 6 > min(F, G`
- 8 x `ValueError:  op 1 (AccColSumBatch_v1{Q=2,K=3})  one contraction chunk exceeds min(`
- 7 x `ValueError:  op 0 (AccColSumBatch_v1{Q=3,K=3})  one contraction chunk exceeds min(`
- 7 x `ValueError:  op 0 (AccMatmulT_v1{M=1,N=4,K=2,CH=1})  one row has work 12 > min(F, `

## Attacks the certificate misses (`L/I* < 0.5`): the optimal partition

**local_sgd_chain[K=2,q=1,d=1]** (micro local-sgd chain (d=1, q=1)) F=1·fwd G=1·fwd: L=14, I*=40
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B
- RU4: work 2, input 4 B; ops [op4:matmul x3]; imports op3:sgdupdate.out=2B, x1[token]=2B
- RU5: work 1, input 4 B; ops [op5:add x1]; imports op4:matmul.out=2B, t1[token]=2B
- RU6: work 2, input 4 B; ops [op6:matmul x3]; imports op5:add.out=2B, x1[token]=2B
- RU7: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B

**local_sgd_chain[K=2,q=1,d=1]** (micro local-sgd chain (d=1, q=1)) F=1·fwd G=4·fwd: L=14, I*=40
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B
- RU4: work 2, input 4 B; ops [op4:matmul x3]; imports op3:sgdupdate.out=2B, x1[token]=2B
- RU5: work 1, input 4 B; ops [op5:add x1]; imports op4:matmul.out=2B, t1[token]=2B
- RU6: work 2, input 4 B; ops [op6:matmul x3]; imports op5:add.out=2B, x1[token]=2B
- RU7: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B

**local_sgd_chain[K=2,q=1,d=1]** (micro local-sgd chain (d=1, q=1)) F=1·fwd G=inf: L=14, I*=40
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B
- RU4: work 2, input 4 B; ops [op4:matmul x3]; imports op3:sgdupdate.out=2B, x1[token]=2B
- RU5: work 1, input 4 B; ops [op5:add x1]; imports op4:matmul.out=2B, t1[token]=2B
- RU6: work 2, input 4 B; ops [op6:matmul x3]; imports op5:add.out=2B, x1[token]=2B
- RU7: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B

**local_sgd_chain[K=2,q=1,d=1]** (micro local-sgd chain (d=1, q=1)) F=1.5·fwd G=1·fwd: L=14, I*=40
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B
- RU4: work 2, input 4 B; ops [op4:matmul x3]; imports op3:sgdupdate.out=2B, x1[token]=2B
- RU5: work 1, input 4 B; ops [op5:add x1]; imports op4:matmul.out=2B, t1[token]=2B
- RU6: work 2, input 4 B; ops [op6:matmul x3]; imports op5:add.out=2B, x1[token]=2B
- RU7: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B

**local_sgd_chain[K=2,q=1,d=1]** (micro local-sgd chain (d=1, q=1)) F=1.5·fwd G=4·fwd: L=14, I*=34
- RU0: work 3, input 6 B; ops [op0:matmul x3, op1:add x1]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 3, input 8 B; ops [op2:matmul x3, op3:sgdupdate x1]; imports lr[seed]=4B, op1:add.out=2B, x0[token]=2B
- RU2: work 3, input 6 B; ops [op3:sgdupdate x1, op4:matmul x3]; imports W[accumulated]=2B, op3:sgdupdate.out=2B, x1[token]=2B
- RU3: work 3, input 6 B; ops [op5:add x1, op6:matmul x3]; imports op4:matmul.out=2B, t1[token]=2B, x1[token]=2B
- RU4: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B

**local_sgd_chain[K=2,q=1,d=1]** (micro local-sgd chain (d=1, q=1)) F=1.5·fwd G=inf: L=14, I*=34
- RU0: work 3, input 6 B; ops [op0:matmul x3, op1:add x1]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 3, input 8 B; ops [op2:matmul x3, op3:sgdupdate x1]; imports lr[seed]=4B, op1:add.out=2B, x0[token]=2B
- RU2: work 3, input 6 B; ops [op3:sgdupdate x1, op4:matmul x3]; imports W[accumulated]=2B, op3:sgdupdate.out=2B, x1[token]=2B
- RU3: work 3, input 6 B; ops [op5:add x1, op6:matmul x3]; imports op4:matmul.out=2B, t1[token]=2B, x1[token]=2B
- RU4: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B

**local_sgd_chain[K=2,q=1,d=1]** (micro local-sgd chain (d=1, q=1)) F=3·fwd G=1·fwd: L=14, I*=40
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B
- RU4: work 2, input 4 B; ops [op4:matmul x3]; imports op3:sgdupdate.out=2B, x1[token]=2B
- RU5: work 1, input 4 B; ops [op5:add x1]; imports op4:matmul.out=2B, t1[token]=2B
- RU6: work 2, input 4 B; ops [op6:matmul x3]; imports op5:add.out=2B, x1[token]=2B
- RU7: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B

**local_sgd_chain[K=2,q=1,d=1]** (micro local-sgd chain (d=1, q=1)) F=inf G=1·fwd: L=14, I*=40
- RU0: work 2, input 4 B; ops [op0:matmul x3]; imports W[accumulated]=2B, x0[token]=2B
- RU1: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 2, input 8 B; ops [op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B
- RU4: work 2, input 4 B; ops [op4:matmul x3]; imports op3:sgdupdate.out=2B, x1[token]=2B
- RU5: work 1, input 4 B; ops [op5:add x1]; imports op4:matmul.out=2B, t1[token]=2B
- RU6: work 2, input 4 B; ops [op6:matmul x3]; imports op5:add.out=2B, x1[token]=2B
- RU7: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B

**local_sgd_chain[K=3,q=1,d=1]** (micro local-sgd chain (d=1, q=1)) F=1·fwd G=1·fwd: L=18, I*=60
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

**local_sgd_chain[K=3,q=1,d=1]** (micro local-sgd chain (d=1, q=1)) F=1·fwd G=4·fwd: L=18, I*=60
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

**local_sgd_chain[K=3,q=1,d=1]** (micro local-sgd chain (d=1, q=1)) F=1·fwd G=inf: L=18, I*=60
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

**local_sgd_chain[K=3,q=1,d=1]** (micro local-sgd chain (d=1, q=1)) F=1.5·fwd G=1·fwd: L=18, I*=60
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

**local_sgd_chain[K=3,q=1,d=1]** (micro local-sgd chain (d=1, q=1)) F=1.5·fwd G=4·fwd: L=18, I*=52
- RU0: work 3, input 6 B; ops [op0:matmul x3, op1:add x1]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 3, input 8 B; ops [op2:matmul x3, op3:sgdupdate x1]; imports lr[seed]=4B, op1:add.out=2B, x0[token]=2B
- RU2: work 3, input 6 B; ops [op3:sgdupdate x1, op4:matmul x3]; imports W[accumulated]=2B, op3:sgdupdate.out=2B, x1[token]=2B
- RU3: work 3, input 6 B; ops [op5:add x1, op6:matmul x3]; imports op4:matmul.out=2B, t1[token]=2B, x1[token]=2B
- RU4: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B
- RU5: work 3, input 6 B; ops [op8:matmul x3, op9:add x1]; imports op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU6: work 3, input 8 B; ops [op10:matmul x3, op11:sgdupdate x1]; imports lr[seed]=4B, op9:add.out=2B, x2[token]=2B
- RU7: work 1, input 4 B; ops [op11:sgdupdate x1]; imports op11:sgdupdate.out=2B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=1,d=1]** (micro local-sgd chain (d=1, q=1)) F=1.5·fwd G=inf: L=18, I*=52
- RU0: work 3, input 6 B; ops [op0:matmul x3, op1:add x1]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 3, input 8 B; ops [op2:matmul x3, op3:sgdupdate x1]; imports lr[seed]=4B, op1:add.out=2B, x0[token]=2B
- RU2: work 3, input 6 B; ops [op3:sgdupdate x1, op4:matmul x3]; imports W[accumulated]=2B, op3:sgdupdate.out=2B, x1[token]=2B
- RU3: work 3, input 6 B; ops [op5:add x1, op6:matmul x3]; imports op4:matmul.out=2B, t1[token]=2B, x1[token]=2B
- RU4: work 2, input 8 B; ops [op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B
- RU5: work 3, input 6 B; ops [op8:matmul x3, op9:add x1]; imports op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU6: work 3, input 8 B; ops [op10:matmul x3, op11:sgdupdate x1]; imports lr[seed]=4B, op9:add.out=2B, x2[token]=2B
- RU7: work 1, input 4 B; ops [op11:sgdupdate x1]; imports op11:sgdupdate.out=2B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=1,d=1]** (micro local-sgd chain (d=1, q=1)) F=3·fwd G=1·fwd: L=18, I*=60
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

**local_sgd_chain[K=3,q=1,d=1]** (micro local-sgd chain (d=1, q=1)) F=3·fwd G=4·fwd: L=18, I*=40
- RU0: work 6, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x3, op3:sgdupdate x1]; imports W[accumulated]=2B, lr[seed]=4B, t0[token]=2B, x0[token]=2B
- RU1: work 6, input 8 B; ops [op3:sgdupdate x1, op4:matmul x3, op5:add x1, op6:matmul x3]; imports W[accumulated]=2B, op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU2: work 6, input 12 B; ops [op7:sgdupdate x2, op8:matmul x3, op9:add x1, op10:matmul x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B, t2[token]=2B, x2[token]=2B
- RU3: work 3, input 10 B; ops [op10:matmul x1, op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=4B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=1,d=1]** (micro local-sgd chain (d=1, q=1)) F=3·fwd G=inf: L=18, I*=40
- RU0: work 6, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x3, op3:sgdupdate x1]; imports W[accumulated]=2B, lr[seed]=4B, t0[token]=2B, x0[token]=2B
- RU1: work 6, input 8 B; ops [op3:sgdupdate x1, op4:matmul x3, op5:add x1, op6:matmul x3]; imports W[accumulated]=2B, op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU2: work 6, input 12 B; ops [op7:sgdupdate x2, op8:matmul x3, op9:add x1, op10:matmul x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B, t2[token]=2B, x2[token]=2B
- RU3: work 3, input 10 B; ops [op10:matmul x1, op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=4B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=1,d=1]** (micro local-sgd chain (d=1, q=1)) F=inf G=1·fwd: L=18, I*=60
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

**local_sgd_chain[K=2,q=2,d=1]** (micro local-sgd chain (d=1, q=2)) F=1·fwd G=1·fwd: L=22, I*=52
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 4, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x1]; imports W[accumulated]=2B, op2:matmul.out=4B, t0[token]=2B, x0[token]=2B
- RU2: work 3, input 10 B; ops [op2:matmul x1, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=4B
- RU3: work 4, input 6 B; ops [op4:matmul x3, op5:add x1, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU4: work 4, input 10 B; ops [op4:matmul x3, op5:add x1, op6:matmul x1]; imports op3:sgdupdate.out=2B, op6:matmul.out=4B, t1[token]=2B, x1[token]=2B
- RU5: work 3, input 10 B; ops [op6:matmul x1, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=4B

**local_sgd_chain[K=2,q=2,d=1]** (micro local-sgd chain (d=1, q=2)) F=1·fwd G=4·fwd: L=22, I*=48
- RU0: work 7, input 10 B; ops [op0:matmul x6, op1:add x2, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=4B, x0[token]=4B
- RU1: work 4, input 14 B; ops [op2:matmul x2, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=2B, op2:matmul.out=4B, x0[token]=2B
- RU2: work 7, input 10 B; ops [op4:matmul x6, op5:add x2, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=4B, x1[token]=4B
- RU3: work 4, input 14 B; ops [op6:matmul x2, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op5:add.out=2B, op6:matmul.out=4B, x1[token]=2B

**local_sgd_chain[K=2,q=2,d=1]** (micro local-sgd chain (d=1, q=2)) F=1·fwd G=inf: L=22, I*=48
- RU0: work 7, input 10 B; ops [op0:matmul x6, op1:add x2, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=4B, x0[token]=4B
- RU1: work 4, input 14 B; ops [op2:matmul x2, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=2B, op2:matmul.out=4B, x0[token]=2B
- RU2: work 7, input 10 B; ops [op4:matmul x6, op5:add x2, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=4B, x1[token]=4B
- RU3: work 4, input 14 B; ops [op6:matmul x2, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op5:add.out=2B, op6:matmul.out=4B, x1[token]=2B

**local_sgd_chain[K=2,q=2,d=1]** (micro local-sgd chain (d=1, q=2)) F=1.5·fwd G=1·fwd: L=22, I*=52
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 4, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x1]; imports W[accumulated]=2B, op2:matmul.out=4B, t0[token]=2B, x0[token]=2B
- RU2: work 3, input 10 B; ops [op2:matmul x1, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=4B
- RU3: work 4, input 6 B; ops [op4:matmul x3, op5:add x1, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU4: work 4, input 10 B; ops [op4:matmul x3, op5:add x1, op6:matmul x1]; imports op3:sgdupdate.out=2B, op6:matmul.out=4B, t1[token]=2B, x1[token]=2B
- RU5: work 3, input 10 B; ops [op6:matmul x1, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=4B

**local_sgd_chain[K=2,q=2,d=1]** (micro local-sgd chain (d=1, q=2)) F=1.5·fwd G=4·fwd: L=22, I*=46
- RU0: work 6, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x4]; imports W[accumulated]=2B, op1:add.out=2B, t0[token]=2B, x0[token]=4B
- RU1: work 3, input 6 B; ops [op0:matmul x3, op1:add x1]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU2: work 9, input 16 B; ops [op3:sgdupdate x2, op4:matmul x6, op5:add x2, op6:matmul x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B, t1[token]=4B, x1[token]=4B
- RU3: work 4, input 14 B; ops [op6:matmul x2, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op5:add.out=2B, op6:matmul.out=4B, x1[token]=2B

**local_sgd_chain[K=2,q=2,d=1]** (micro local-sgd chain (d=1, q=2)) F=1.5·fwd G=inf: L=22, I*=46
- RU0: work 6, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x4]; imports W[accumulated]=2B, op1:add.out=2B, t0[token]=2B, x0[token]=4B
- RU1: work 3, input 6 B; ops [op0:matmul x3, op1:add x1]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU2: work 9, input 16 B; ops [op3:sgdupdate x2, op4:matmul x6, op5:add x2, op6:matmul x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B, t1[token]=4B, x1[token]=4B
- RU3: work 4, input 14 B; ops [op6:matmul x2, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op5:add.out=2B, op6:matmul.out=4B, x1[token]=2B

**local_sgd_chain[K=2,q=2,d=1]** (micro local-sgd chain (d=1, q=2)) F=3·fwd G=1·fwd: L=22, I*=52
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 4, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x1]; imports W[accumulated]=2B, op2:matmul.out=4B, t0[token]=2B, x0[token]=2B
- RU2: work 3, input 10 B; ops [op2:matmul x1, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=4B
- RU3: work 4, input 6 B; ops [op4:matmul x3, op5:add x1, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU4: work 4, input 10 B; ops [op4:matmul x3, op5:add x1, op6:matmul x1]; imports op3:sgdupdate.out=2B, op6:matmul.out=4B, t1[token]=2B, x1[token]=2B
- RU5: work 3, input 10 B; ops [op6:matmul x1, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=4B

**local_sgd_chain[K=2,q=2,d=1]** (micro local-sgd chain (d=1, q=2)) F=inf G=1·fwd: L=22, I*=52
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 4, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x1]; imports W[accumulated]=2B, op2:matmul.out=4B, t0[token]=2B, x0[token]=2B
- RU2: work 3, input 10 B; ops [op2:matmul x1, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=4B
- RU3: work 4, input 6 B; ops [op4:matmul x3, op5:add x1, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU4: work 4, input 10 B; ops [op4:matmul x3, op5:add x1, op6:matmul x1]; imports op3:sgdupdate.out=2B, op6:matmul.out=4B, t1[token]=2B, x1[token]=2B
- RU5: work 3, input 10 B; ops [op6:matmul x1, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=4B

**local_sgd_chain[K=3,q=2,d=1]** (micro local-sgd chain (d=1, q=2)) F=1·fwd G=1·fwd: L=30, I*=78
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 3, input 6 B; ops [op0:matmul x3, op1:add x1]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU2: work 4, input 14 B; ops [op2:matmul x2, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=2B, op2:matmul.out=4B, x0[token]=2B
- RU3: work 4, input 6 B; ops [op4:matmul x3, op5:add x1, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU4: work 3, input 6 B; ops [op4:matmul x3, op5:add x1]; imports op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU5: work 4, input 14 B; ops [op6:matmul x2, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op5:add.out=2B, op6:matmul.out=4B, x1[token]=2B
- RU6: work 4, input 6 B; ops [op8:matmul x3, op9:add x1, op10:matmul x2]; imports op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU7: work 3, input 6 B; ops [op8:matmul x3, op9:add x1]; imports op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU8: work 4, input 14 B; ops [op10:matmul x2, op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=4B, op7:sgdupdate.out=2B, op9:add.out=2B, x2[token]=2B

**local_sgd_chain[K=3,q=2,d=1]** (micro local-sgd chain (d=1, q=2)) F=1·fwd G=4·fwd: L=30, I*=72
- RU0: work 7, input 10 B; ops [op0:matmul x6, op1:add x2, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=4B, x0[token]=4B
- RU1: work 4, input 14 B; ops [op2:matmul x2, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=2B, op2:matmul.out=4B, x0[token]=2B
- RU2: work 7, input 10 B; ops [op4:matmul x6, op5:add x2, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=4B, x1[token]=4B
- RU3: work 4, input 14 B; ops [op6:matmul x2, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op5:add.out=2B, op6:matmul.out=4B, x1[token]=2B
- RU4: work 7, input 10 B; ops [op8:matmul x6, op9:add x2, op10:matmul x2]; imports op7:sgdupdate.out=2B, t2[token]=4B, x2[token]=4B
- RU5: work 4, input 14 B; ops [op10:matmul x2, op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=4B, op7:sgdupdate.out=2B, op9:add.out=2B, x2[token]=2B

**local_sgd_chain[K=3,q=2,d=1]** (micro local-sgd chain (d=1, q=2)) F=1·fwd G=inf: L=30, I*=72
- RU0: work 7, input 10 B; ops [op0:matmul x6, op1:add x2, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=4B, x0[token]=4B
- RU1: work 4, input 14 B; ops [op2:matmul x2, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=2B, op2:matmul.out=4B, x0[token]=2B
- RU2: work 7, input 10 B; ops [op4:matmul x6, op5:add x2, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=4B, x1[token]=4B
- RU3: work 4, input 14 B; ops [op6:matmul x2, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op5:add.out=2B, op6:matmul.out=4B, x1[token]=2B
- RU4: work 7, input 10 B; ops [op8:matmul x6, op9:add x2, op10:matmul x2]; imports op7:sgdupdate.out=2B, t2[token]=4B, x2[token]=4B
- RU5: work 4, input 14 B; ops [op10:matmul x2, op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=4B, op7:sgdupdate.out=2B, op9:add.out=2B, x2[token]=2B

**local_sgd_chain[K=3,q=2,d=1]** (micro local-sgd chain (d=1, q=2)) F=1.5·fwd G=1·fwd: L=30, I*=78
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 3, input 6 B; ops [op0:matmul x3, op1:add x1]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU2: work 2, input 8 B; ops [op2:matmul x2]; imports op1:add.out=2B, op2:matmul.out=4B, x0[token]=2B
- RU3: work 4, input 10 B; ops [op3:sgdupdate x2, op4:matmul x3]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B, x1[token]=2B
- RU4: work 4, input 6 B; ops [op4:matmul x3, op5:add x1, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU5: work 4, input 14 B; ops [op5:add x1, op6:matmul x2, op7:sgdupdate x1]; imports lr[seed]=4B, op4:matmul.out=2B, op6:matmul.out=4B, t1[token]=2B, x1[token]=2B
- RU6: work 4, input 8 B; ops [op7:sgdupdate x1, op8:matmul x3, op9:add x1]; imports op3:sgdupdate.out=2B, op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU7: work 4, input 6 B; ops [op8:matmul x3, op9:add x1, op10:matmul x2]; imports op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU8: work 4, input 14 B; ops [op10:matmul x2, op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=4B, op7:sgdupdate.out=2B, op9:add.out=2B, x2[token]=2B

**local_sgd_chain[K=3,q=2,d=1]** (micro local-sgd chain (d=1, q=2)) F=1.5·fwd G=4·fwd: L=30, I*=68
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 5, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, op2:matmul.out=4B, t0[token]=2B, x0[token]=2B
- RU2: work 5, input 12 B; ops [op3:sgdupdate x2, op4:matmul x3, op5:add x1]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B, t1[token]=2B, x1[token]=2B
- RU3: work 6, input 10 B; ops [op4:matmul x3, op5:add x1, op6:matmul x4]; imports op3:sgdupdate.out=2B, op5:add.out=2B, t1[token]=2B, x1[token]=4B
- RU4: work 6, input 12 B; ops [op7:sgdupdate x2, op8:matmul x3, op9:add x1, op10:matmul x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B, t2[token]=2B, x2[token]=2B
- RU5: work 5, input 10 B; ops [op8:matmul x3, op9:add x1, op10:matmul x2]; imports op10:matmul.out=4B, op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU6: work 2, input 8 B; ops [op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=2B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=2,d=1]** (micro local-sgd chain (d=1, q=2)) F=1.5·fwd G=inf: L=30, I*=68
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 5, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, op2:matmul.out=4B, t0[token]=2B, x0[token]=2B
- RU2: work 5, input 12 B; ops [op3:sgdupdate x2, op4:matmul x3, op5:add x1]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B, t1[token]=2B, x1[token]=2B
- RU3: work 6, input 10 B; ops [op4:matmul x3, op5:add x1, op6:matmul x4]; imports op3:sgdupdate.out=2B, op5:add.out=2B, t1[token]=2B, x1[token]=4B
- RU4: work 7, input 14 B; ops [op7:sgdupdate x2, op8:matmul x6, op9:add x1]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op6:matmul.out=2B, t2[token]=2B, x2[token]=4B
- RU5: work 6, input 16 B; ops [op9:add x1, op10:matmul x4, op11:sgdupdate x2]; imports lr[seed]=4B, op7:sgdupdate.out=2B, op8:matmul.out=2B, op9:add.out=2B, t2[token]=2B, x2[token]=4B

**local_sgd_chain[K=3,q=2,d=1]** (micro local-sgd chain (d=1, q=2)) F=3·fwd G=1·fwd: L=30, I*=78
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:add x1, op2:matmul x2]; imports W[accumulated]=2B, t0[token]=2B, x0[token]=2B
- RU1: work 4, input 10 B; ops [op0:matmul x3, op1:add x1, op2:matmul x1]; imports W[accumulated]=2B, op2:matmul.out=4B, t0[token]=2B, x0[token]=2B
- RU2: work 3, input 10 B; ops [op2:matmul x1, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=4B
- RU3: work 4, input 6 B; ops [op4:matmul x3, op5:add x1, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU4: work 3, input 6 B; ops [op4:matmul x3, op5:add x1]; imports op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU5: work 3, input 12 B; ops [op6:matmul x2, op7:sgdupdate x1]; imports lr[seed]=4B, op5:add.out=2B, op6:matmul.out=4B, x1[token]=2B
- RU6: work 4, input 8 B; ops [op7:sgdupdate x1, op8:matmul x3, op9:add x1]; imports op3:sgdupdate.out=2B, op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU7: work 4, input 6 B; ops [op8:matmul x3, op9:add x1, op10:matmul x2]; imports op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU8: work 4, input 14 B; ops [op10:matmul x2, op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=4B, op7:sgdupdate.out=2B, op9:add.out=2B, x2[token]=2B

**local_sgd_chain[K=3,q=2,d=1]** (micro local-sgd chain (d=1, q=2)) F=inf G=1·fwd: L=30, I*=78
- RU0: work 4, input 6 B; ops [op0:matmul x6]; imports W[accumulated]=2B, x0[token]=4B
- RU1: work 4, input 10 B; ops [op1:add x1, op2:matmul x4]; imports op0:matmul.out=2B, op1:add.out=2B, t0[token]=2B, x0[token]=4B
- RU2: work 1, input 4 B; ops [op1:add x1]; imports op0:matmul.out=2B, t0[token]=2B
- RU3: work 4, input 10 B; ops [op3:sgdupdate x2, op4:matmul x3]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B, x1[token]=2B
- RU4: work 4, input 6 B; ops [op4:matmul x3, op5:add x1, op6:matmul x2]; imports op3:sgdupdate.out=2B, t1[token]=2B, x1[token]=2B
- RU5: work 4, input 14 B; ops [op5:add x1, op6:matmul x2, op7:sgdupdate x1]; imports lr[seed]=4B, op4:matmul.out=2B, op6:matmul.out=4B, t1[token]=2B, x1[token]=2B
- RU6: work 4, input 8 B; ops [op7:sgdupdate x1, op8:matmul x3, op9:add x1]; imports op3:sgdupdate.out=2B, op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU7: work 4, input 6 B; ops [op8:matmul x3, op9:add x1, op10:matmul x2]; imports op7:sgdupdate.out=2B, t2[token]=2B, x2[token]=2B
- RU8: work 4, input 14 B; ops [op10:matmul x2, op11:sgdupdate x2]; imports lr[seed]=4B, op10:matmul.out=4B, op7:sgdupdate.out=2B, op9:add.out=2B, x2[token]=2B

**local_sgd_chain[K=1,q=1,d=2]** (micro local-sgd chain (d=2, q=1)) F=1·fwd G=1·fwd: L=20, I*=48
- RU0: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU1: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU2: work 6, input 12 B; ops [op2:matmul x6, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=4B, x0[token]=2B
- RU3: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=6B, lr[seed]=4B, op2:matmul.out=6B

**local_sgd_chain[K=1,q=1,d=2]** (micro local-sgd chain (d=2, q=1)) F=1.5·fwd G=1·fwd: L=20, I*=48
- RU0: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU1: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU2: work 6, input 12 B; ops [op2:matmul x6, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=4B, x0[token]=2B
- RU3: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=6B, lr[seed]=4B, op2:matmul.out=6B

**local_sgd_chain[K=1,q=1,d=2]** (micro local-sgd chain (d=2, q=1)) F=3·fwd G=1·fwd: L=20, I*=48
- RU0: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU1: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU2: work 6, input 12 B; ops [op2:matmul x6, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=4B, x0[token]=2B
- RU3: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=6B, lr[seed]=4B, op2:matmul.out=6B

**local_sgd_chain[K=1,q=1,d=2]** (micro local-sgd chain (d=2, q=1)) F=inf G=1·fwd: L=20, I*=48
- RU0: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU1: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU2: work 6, input 12 B; ops [op2:matmul x6, op3:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=4B, x0[token]=2B
- RU3: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=6B, lr[seed]=4B, op2:matmul.out=6B

**local_sgd_chain[K=2,q=1,d=2]** (micro local-sgd chain (d=2, q=1)) F=1·fwd G=1·fwd: L=28, I*=[84, 98]
- RU0: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU1: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU2: work 2, input 4 B; ops [op2:matmul x3]; imports op1:add.out=2B, x0[token]=2B
- RU3: work 6, input 14 B; ops [op2:matmul x3, op3:sgdupdate x4]; imports W[accumulated]=2B, lr[seed]=4B, op1:add.out=2B, op2:matmul.out=4B, x0[token]=2B
- RU4: work 6, input 14 B; ops [op3:sgdupdate x2, op6:matmul x3, op7:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B, op3:sgdupdate.out=2B, op5:add.out=2B, x1[token]=2B
- RU5: work 4, input 8 B; ops [op3:sgdupdate x1, op4:matmul x2, op6:matmul x3]; imports W[accumulated]=2B, op3:sgdupdate.out=2B, op5:add.out=2B, x1[token]=2B
- RU6: work 6, input 12 B; ops [op3:sgdupdate x1, op4:matmul x2, op5:add x1, op6:matmul x3]; imports W[accumulated]=2B, op3:sgdupdate.out=2B, op4:matmul.out=4B, t1[token]=2B, x1[token]=2B
- RU7: work 6, input 10 B; ops [op4:matmul x4, op5:add x1, op6:matmul x3]; imports op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU8: work 6, input 16 B; ops [op7:sgdupdate x6]; imports lr[seed]=4B, op3:sgdupdate.out=6B, op6:matmul.out=6B

**local_sgd_chain[K=2,q=1,d=2]** (micro local-sgd chain (d=2, q=1)) F=1·fwd G=4·fwd: L=28, I*=68
- RU0: work 16, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x12]; imports W[accumulated]=8B, t0[token]=4B, x0[token]=4B
- RU1: work 12, input 24 B; ops [op3:sgdupdate x8, op4:matmul x6]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=8B, x1[token]=4B
- RU2: work 20, input 28 B; ops [op4:matmul x2, op5:add x2, op6:matmul x12, op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, op4:matmul.out=8B, t1[token]=4B, x1[token]=4B

**local_sgd_chain[K=2,q=1,d=2]** (micro local-sgd chain (d=2, q=1)) F=1·fwd G=inf: L=28, I*=68
- RU0: work 16, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x12]; imports W[accumulated]=8B, t0[token]=4B, x0[token]=4B
- RU1: work 12, input 24 B; ops [op3:sgdupdate x8, op4:matmul x6]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=8B, x1[token]=4B
- RU2: work 20, input 28 B; ops [op4:matmul x2, op5:add x2, op6:matmul x12, op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, op4:matmul.out=8B, t1[token]=4B, x1[token]=4B

**local_sgd_chain[K=2,q=1,d=2]** (micro local-sgd chain (d=2, q=1)) F=1.5·fwd G=1·fwd: L=28, I*=[84, 98]
- RU0: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU1: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU2: work 4, input 6 B; ops [op2:matmul x6]; imports op1:add.out=4B, x0[token]=2B
- RU3: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=6B, lr[seed]=4B, op2:matmul.out=6B
- RU4: work 3, input 10 B; ops [op3:sgdupdate x1, op7:sgdupdate x2]; imports lr[seed]=4B, op2:matmul.out=2B, op3:sgdupdate.out=2B, op6:matmul.out=2B
- RU5: work 6, input 12 B; ops [op3:sgdupdate x1, op4:matmul x4, op6:matmul x3]; imports W[accumulated]=2B, op3:sgdupdate.out=4B, op5:add.out=2B, x1[token]=4B
- RU6: work 6, input 10 B; ops [op4:matmul x4, op5:add x1, op6:matmul x3]; imports op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU7: work 5, input 8 B; ops [op5:add x1, op6:matmul x6]; imports op4:matmul.out=2B, t1[token]=2B, x1[token]=4B
- RU8: work 6, input 16 B; ops [op7:sgdupdate x6]; imports lr[seed]=4B, op3:sgdupdate.out=6B, op6:matmul.out=6B

**local_sgd_chain[K=2,q=1,d=2]** (micro local-sgd chain (d=2, q=1)) F=3·fwd G=1·fwd: L=28, I*=[84, 98]
- RU0: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU1: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU2: work 4, input 6 B; ops [op2:matmul x6]; imports op1:add.out=4B, x0[token]=2B
- RU3: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=4B, lr[seed]=4B, op2:matmul.out=8B
- RU4: work 5, input 12 B; ops [op3:sgdupdate x2, op4:matmul x4]; imports W[accumulated]=4B, op3:sgdupdate.out=4B, x1[token]=4B
- RU5: work 6, input 10 B; ops [op4:matmul x4, op5:add x1, op6:matmul x3]; imports op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU6: work 5, input 8 B; ops [op5:add x1, op6:matmul x6]; imports op4:matmul.out=2B, t1[token]=2B, x1[token]=4B
- RU7: work 4, input 10 B; ops [op6:matmul x3, op7:sgdupdate x2]; imports lr[seed]=4B, op3:sgdupdate.out=2B, op5:add.out=2B, x1[token]=2B
- RU8: work 6, input 16 B; ops [op7:sgdupdate x6]; imports lr[seed]=4B, op3:sgdupdate.out=6B, op6:matmul.out=6B

**local_sgd_chain[K=2,q=1,d=2]** (micro local-sgd chain (d=2, q=1)) F=inf G=1·fwd: L=28, I*=[84, 98]
- RU0: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU1: work 6, input 10 B; ops [op0:matmul x4, op1:add x1, op2:matmul x3]; imports W[accumulated]=4B, t0[token]=2B, x0[token]=4B
- RU2: work 4, input 6 B; ops [op2:matmul x6]; imports op1:add.out=4B, x0[token]=2B
- RU3: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=6B, lr[seed]=4B, op2:matmul.out=6B
- RU4: work 6, input 14 B; ops [op3:sgdupdate x2, op6:matmul x3, op7:sgdupdate x2]; imports W[accumulated]=2B, lr[seed]=4B, op2:matmul.out=2B, op3:sgdupdate.out=2B, op5:add.out=2B, x1[token]=2B
- RU5: work 6, input 12 B; ops [op4:matmul x8]; imports op3:sgdupdate.out=8B, x1[token]=4B
- RU6: work 5, input 8 B; ops [op5:add x1, op6:matmul x6]; imports op4:matmul.out=2B, op5:add.out=2B, t1[token]=2B, x1[token]=2B
- RU7: work 3, input 6 B; ops [op5:add x1, op6:matmul x3]; imports op4:matmul.out=2B, t1[token]=2B, x1[token]=2B
- RU8: work 6, input 16 B; ops [op7:sgdupdate x6]; imports lr[seed]=4B, op3:sgdupdate.out=6B, op6:matmul.out=6B

**local_sgd_chain[K=3,q=1,d=2]** (micro local-sgd chain (d=2, q=1)) F=1·fwd G=1·fwd: L=36, I*=[36, 148]
- RU18: work 6, input 12 B; ops [op0:matmul x8]; imports W[accumulated]=8B, x0[token]=4B
- RU28: work 5, input 8 B; ops [op1:add x1, op2:matmul x6]; imports op0:matmul.out=2B, t0[token]=2B, x0[token]=4B
- RU34: work 5, input 8 B; ops [op1:add x1, op2:matmul x6]; imports op0:matmul.out=2B, t0[token]=2B, x0[token]=4B
- RU40: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=4B, lr[seed]=4B, op2:matmul.out=8B
- RU48: work 6, input 10 B; ops [op4:matmul x4, op5:add x1, op6:matmul x3]; imports op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU52: work 6, input 14 B; ops [op3:sgdupdate x2, op4:matmul x4, op5:add x1]; imports W[accumulated]=4B, op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU61: work 6, input 8 B; ops [op6:matmul x9]; imports op5:add.out=4B, x1[token]=4B
- RU70: work 6, input 16 B; ops [op7:sgdupdate x6]; imports lr[seed]=4B, op3:sgdupdate.out=4B, op6:matmul.out=8B
- RU78: work 6, input 10 B; ops [op8:matmul x4, op9:add x1, op10:matmul x3]; imports op7:sgdupdate.out=4B, t2[token]=2B, x2[token]=4B
- RU82: work 6, input 14 B; ops [op7:sgdupdate x2, op8:matmul x4, op9:add x1]; imports op3:sgdupdate.out=4B, op7:sgdupdate.out=4B, t2[token]=2B, x2[token]=4B
- RU91: work 6, input 8 B; ops [op10:matmul x9]; imports op9:add.out=4B, x2[token]=4B
- RU100: work 6, input 16 B; ops [op11:sgdupdate x6]; imports lr[seed]=4B, op10:matmul.out=8B, op7:sgdupdate.out=4B
- RU105: work 1, input 4 B; ops [op11:sgdupdate x1]; imports op11:sgdupdate.out=2B, op7:sgdupdate.out=2B
- RU106: work 1, input 4 B; ops [op11:sgdupdate x1]; imports op11:sgdupdate.out=2B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=1,d=2]** (micro local-sgd chain (d=2, q=1)) F=1·fwd G=4·fwd: L=36, I*=[36, 108]
- RU0: work 16, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x12]; imports W[accumulated]=8B, t0[token]=4B, x0[token]=4B
- RU3: work 8, input 20 B; ops [op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=8B
- RU4: work 16, input 16 B; ops [op4:matmul x8, op5:add x2, op6:matmul x12]; imports op3:sgdupdate.out=8B, t1[token]=4B, x1[token]=4B
- RU7: work 8, input 20 B; ops [op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, op6:matmul.out=8B
- RU8: work 16, input 16 B; ops [op8:matmul x8, op9:add x2, op10:matmul x12]; imports op7:sgdupdate.out=8B, t2[token]=4B, x2[token]=4B
- RU11: work 8, input 20 B; ops [op11:sgdupdate x8]; imports lr[seed]=4B, op10:matmul.out=8B, op7:sgdupdate.out=8B

**local_sgd_chain[K=3,q=1,d=2]** (micro local-sgd chain (d=2, q=1)) F=1·fwd G=inf: L=36, I*=[36, 108]
- RU18: work 16, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x12]; imports W[accumulated]=8B, t0[token]=4B, x0[token]=4B
- RU40: work 8, input 20 B; ops [op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=8B
- RU48: work 16, input 16 B; ops [op4:matmul x8, op5:add x2, op6:matmul x12]; imports op3:sgdupdate.out=8B, t1[token]=4B, x1[token]=4B
- RU70: work 8, input 20 B; ops [op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, op6:matmul.out=8B
- RU78: work 16, input 16 B; ops [op8:matmul x8, op9:add x2, op10:matmul x12]; imports op7:sgdupdate.out=8B, t2[token]=4B, x2[token]=4B
- RU100: work 8, input 20 B; ops [op11:sgdupdate x8]; imports lr[seed]=4B, op10:matmul.out=8B, op7:sgdupdate.out=8B

**local_sgd_chain[K=3,q=1,d=2]** (micro local-sgd chain (d=2, q=1)) F=1.5·fwd G=1·fwd: L=36, I*=[36, 148]
- RU18: work 6, input 12 B; ops [op0:matmul x8]; imports W[accumulated]=8B, x0[token]=4B
- RU28: work 5, input 8 B; ops [op1:add x1, op2:matmul x6]; imports op0:matmul.out=2B, t0[token]=2B, x0[token]=4B
- RU34: work 5, input 8 B; ops [op1:add x1, op2:matmul x6]; imports op0:matmul.out=2B, t0[token]=2B, x0[token]=4B
- RU40: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=4B, lr[seed]=4B, op2:matmul.out=8B
- RU48: work 6, input 10 B; ops [op4:matmul x4, op5:add x1, op6:matmul x3]; imports op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU52: work 6, input 14 B; ops [op3:sgdupdate x2, op4:matmul x4, op5:add x1]; imports W[accumulated]=4B, op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU61: work 6, input 8 B; ops [op6:matmul x9]; imports op5:add.out=4B, x1[token]=4B
- RU70: work 6, input 16 B; ops [op7:sgdupdate x6]; imports lr[seed]=4B, op3:sgdupdate.out=4B, op6:matmul.out=8B
- RU78: work 6, input 10 B; ops [op8:matmul x4, op9:add x1, op10:matmul x3]; imports op7:sgdupdate.out=4B, t2[token]=2B, x2[token]=4B
- RU82: work 6, input 14 B; ops [op7:sgdupdate x2, op8:matmul x4, op9:add x1]; imports op3:sgdupdate.out=4B, op7:sgdupdate.out=4B, t2[token]=2B, x2[token]=4B
- RU91: work 6, input 8 B; ops [op10:matmul x9]; imports op9:add.out=4B, x2[token]=4B
- RU100: work 6, input 16 B; ops [op11:sgdupdate x6]; imports lr[seed]=4B, op10:matmul.out=8B, op7:sgdupdate.out=4B
- RU105: work 1, input 4 B; ops [op11:sgdupdate x1]; imports op11:sgdupdate.out=2B, op7:sgdupdate.out=2B
- RU106: work 1, input 4 B; ops [op11:sgdupdate x1]; imports op11:sgdupdate.out=2B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=1,d=2]** (micro local-sgd chain (d=2, q=1)) F=3·fwd G=1·fwd: L=36, I*=[36, 148]
- RU18: work 6, input 12 B; ops [op0:matmul x8]; imports W[accumulated]=8B, x0[token]=4B
- RU28: work 5, input 8 B; ops [op1:add x1, op2:matmul x6]; imports op0:matmul.out=2B, t0[token]=2B, x0[token]=4B
- RU34: work 5, input 8 B; ops [op1:add x1, op2:matmul x6]; imports op0:matmul.out=2B, t0[token]=2B, x0[token]=4B
- RU40: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=4B, lr[seed]=4B, op2:matmul.out=8B
- RU48: work 6, input 10 B; ops [op4:matmul x4, op5:add x1, op6:matmul x3]; imports op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU52: work 6, input 14 B; ops [op3:sgdupdate x2, op4:matmul x4, op5:add x1]; imports W[accumulated]=4B, op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU61: work 6, input 8 B; ops [op6:matmul x9]; imports op5:add.out=4B, x1[token]=4B
- RU70: work 6, input 16 B; ops [op7:sgdupdate x6]; imports lr[seed]=4B, op3:sgdupdate.out=4B, op6:matmul.out=8B
- RU78: work 6, input 10 B; ops [op8:matmul x4, op9:add x1, op10:matmul x3]; imports op7:sgdupdate.out=4B, t2[token]=2B, x2[token]=4B
- RU82: work 6, input 14 B; ops [op7:sgdupdate x2, op8:matmul x4, op9:add x1]; imports op3:sgdupdate.out=4B, op7:sgdupdate.out=4B, t2[token]=2B, x2[token]=4B
- RU91: work 6, input 8 B; ops [op10:matmul x9]; imports op9:add.out=4B, x2[token]=4B
- RU100: work 6, input 16 B; ops [op11:sgdupdate x6]; imports lr[seed]=4B, op10:matmul.out=8B, op7:sgdupdate.out=4B
- RU105: work 1, input 4 B; ops [op11:sgdupdate x1]; imports op11:sgdupdate.out=2B, op7:sgdupdate.out=2B
- RU106: work 1, input 4 B; ops [op11:sgdupdate x1]; imports op11:sgdupdate.out=2B, op7:sgdupdate.out=2B

**local_sgd_chain[K=3,q=1,d=2]** (micro local-sgd chain (d=2, q=1)) F=inf G=1·fwd: L=36, I*=[36, 148]
- RU18: work 6, input 12 B; ops [op0:matmul x8]; imports W[accumulated]=8B, x0[token]=4B
- RU28: work 5, input 8 B; ops [op1:add x1, op2:matmul x6]; imports op0:matmul.out=2B, t0[token]=2B, x0[token]=4B
- RU34: work 5, input 8 B; ops [op1:add x1, op2:matmul x6]; imports op0:matmul.out=2B, t0[token]=2B, x0[token]=4B
- RU40: work 6, input 16 B; ops [op3:sgdupdate x6]; imports W[accumulated]=4B, lr[seed]=4B, op2:matmul.out=8B
- RU48: work 6, input 10 B; ops [op4:matmul x4, op5:add x1, op6:matmul x3]; imports op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU52: work 6, input 14 B; ops [op3:sgdupdate x2, op4:matmul x4, op5:add x1]; imports W[accumulated]=4B, op3:sgdupdate.out=4B, t1[token]=2B, x1[token]=4B
- RU61: work 6, input 8 B; ops [op6:matmul x9]; imports op5:add.out=4B, x1[token]=4B
- RU70: work 6, input 16 B; ops [op7:sgdupdate x6]; imports lr[seed]=4B, op3:sgdupdate.out=4B, op6:matmul.out=8B
- RU78: work 6, input 10 B; ops [op8:matmul x4, op9:add x1, op10:matmul x3]; imports op7:sgdupdate.out=4B, t2[token]=2B, x2[token]=4B
- RU82: work 6, input 14 B; ops [op7:sgdupdate x2, op8:matmul x4, op9:add x1]; imports op3:sgdupdate.out=4B, op7:sgdupdate.out=4B, t2[token]=2B, x2[token]=4B
- RU91: work 6, input 8 B; ops [op10:matmul x9]; imports op9:add.out=4B, x2[token]=4B
- RU100: work 6, input 16 B; ops [op11:sgdupdate x6]; imports lr[seed]=4B, op10:matmul.out=8B, op7:sgdupdate.out=4B
- RU105: work 1, input 4 B; ops [op11:sgdupdate x1]; imports op11:sgdupdate.out=2B, op7:sgdupdate.out=2B
- RU106: work 1, input 4 B; ops [op11:sgdupdate x1]; imports op11:sgdupdate.out=2B, op7:sgdupdate.out=2B

**local_sgd_chain[K=1,q=2,d=2]** (micro local-sgd chain (d=2, q=2)) F=1·fwd G=1·fwd: L=28, I*=60
- RU0: work 12, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x6]; imports W[accumulated]=4B, t0[token]=4B, x0[token]=8B
- RU1: work 12, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x6]; imports W[accumulated]=4B, t0[token]=4B, x0[token]=8B
- RU2: work 12, input 28 B; ops [op2:matmul x4, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=16B

**local_sgd_chain[K=1,q=2,d=2]** (micro local-sgd chain (d=2, q=2)) F=1.5·fwd G=1·fwd: L=28, I*=60
- RU0: work 12, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x6]; imports W[accumulated]=4B, t0[token]=4B, x0[token]=8B
- RU1: work 12, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x6]; imports W[accumulated]=4B, t0[token]=4B, x0[token]=8B
- RU2: work 12, input 28 B; ops [op2:matmul x4, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=16B

**local_sgd_chain[K=1,q=2,d=2]** (micro local-sgd chain (d=2, q=2)) F=3·fwd G=1·fwd: L=28, I*=60
- RU0: work 12, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x6]; imports W[accumulated]=4B, t0[token]=4B, x0[token]=8B
- RU1: work 12, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x6]; imports W[accumulated]=4B, t0[token]=4B, x0[token]=8B
- RU2: work 12, input 28 B; ops [op2:matmul x4, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=16B

**local_sgd_chain[K=1,q=2,d=2]** (micro local-sgd chain (d=2, q=2)) F=inf G=1·fwd: L=28, I*=60
- RU0: work 12, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x6]; imports W[accumulated]=4B, t0[token]=4B, x0[token]=8B
- RU1: work 12, input 16 B; ops [op0:matmul x8, op1:add x2, op2:matmul x6]; imports W[accumulated]=4B, t0[token]=4B, x0[token]=8B
- RU2: work 12, input 28 B; ops [op2:matmul x4, op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=16B

**local_sgd_chain[K=2,q=2,d=2]** (micro local-sgd chain (d=2, q=2)) F=1·fwd G=1·fwd: L=44, I*=[44, 136]
- RU22: work 12, input 16 B; ops [op0:matmul x16]; imports W[accumulated]=8B, x0[token]=8B
- RU42: work 11, input 20 B; ops [op1:add x2, op2:matmul x12]; imports op0:matmul.out=4B, op1:add.out=4B, t0[token]=4B, x0[token]=8B
- RU50: work 5, input 12 B; ops [op1:add x2, op2:matmul x4]; imports op0:matmul.out=4B, t0[token]=4B, x0[token]=4B
- RU58: work 8, input 20 B; ops [op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=8B
- RU66: work 12, input 16 B; ops [op4:matmul x16]; imports op3:sgdupdate.out=8B, x1[token]=8B
- RU86: work 11, input 20 B; ops [op5:add x2, op6:matmul x12]; imports op4:matmul.out=4B, op5:add.out=4B, t1[token]=4B, x1[token]=8B
- RU94: work 5, input 12 B; ops [op5:add x2, op6:matmul x4]; imports op4:matmul.out=4B, t1[token]=4B, x1[token]=4B
- RU102: work 8, input 20 B; ops [op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, op6:matmul.out=8B

**local_sgd_chain[K=2,q=2,d=2]** (micro local-sgd chain (d=2, q=2)) F=1.5·fwd G=1·fwd: L=44, I*=[44, 136]
- RU22: work 12, input 16 B; ops [op0:matmul x16]; imports W[accumulated]=8B, x0[token]=8B
- RU42: work 11, input 20 B; ops [op1:add x2, op2:matmul x12]; imports op0:matmul.out=4B, op1:add.out=4B, t0[token]=4B, x0[token]=8B
- RU50: work 5, input 12 B; ops [op1:add x2, op2:matmul x4]; imports op0:matmul.out=4B, t0[token]=4B, x0[token]=4B
- RU58: work 8, input 20 B; ops [op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=8B
- RU66: work 12, input 16 B; ops [op4:matmul x16]; imports op3:sgdupdate.out=8B, x1[token]=8B
- RU86: work 11, input 20 B; ops [op5:add x2, op6:matmul x12]; imports op4:matmul.out=4B, op5:add.out=4B, t1[token]=4B, x1[token]=8B
- RU94: work 5, input 12 B; ops [op5:add x2, op6:matmul x4]; imports op4:matmul.out=4B, t1[token]=4B, x1[token]=4B
- RU102: work 8, input 20 B; ops [op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, op6:matmul.out=8B

**local_sgd_chain[K=2,q=2,d=2]** (micro local-sgd chain (d=2, q=2)) F=3·fwd G=1·fwd: L=44, I*=[44, 136]
- RU22: work 12, input 16 B; ops [op0:matmul x16]; imports W[accumulated]=8B, x0[token]=8B
- RU42: work 11, input 20 B; ops [op1:add x2, op2:matmul x12]; imports op0:matmul.out=4B, op1:add.out=4B, t0[token]=4B, x0[token]=8B
- RU50: work 5, input 12 B; ops [op1:add x2, op2:matmul x4]; imports op0:matmul.out=4B, t0[token]=4B, x0[token]=4B
- RU58: work 8, input 20 B; ops [op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=8B
- RU66: work 12, input 16 B; ops [op4:matmul x16]; imports op3:sgdupdate.out=8B, x1[token]=8B
- RU86: work 11, input 20 B; ops [op5:add x2, op6:matmul x12]; imports op4:matmul.out=4B, op5:add.out=4B, t1[token]=4B, x1[token]=8B
- RU94: work 5, input 12 B; ops [op5:add x2, op6:matmul x4]; imports op4:matmul.out=4B, t1[token]=4B, x1[token]=4B
- RU102: work 8, input 20 B; ops [op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, op6:matmul.out=8B

**local_sgd_chain[K=2,q=2,d=2]** (micro local-sgd chain (d=2, q=2)) F=inf G=1·fwd: L=44, I*=[44, 136]
- RU22: work 12, input 16 B; ops [op0:matmul x16]; imports W[accumulated]=8B, x0[token]=8B
- RU42: work 11, input 20 B; ops [op1:add x2, op2:matmul x12]; imports op0:matmul.out=4B, op1:add.out=4B, t0[token]=4B, x0[token]=8B
- RU50: work 5, input 12 B; ops [op1:add x2, op2:matmul x4]; imports op0:matmul.out=4B, t0[token]=4B, x0[token]=4B
- RU58: work 8, input 20 B; ops [op3:sgdupdate x8]; imports W[accumulated]=8B, lr[seed]=4B, op2:matmul.out=8B
- RU66: work 12, input 16 B; ops [op4:matmul x16]; imports op3:sgdupdate.out=8B, x1[token]=8B
- RU86: work 11, input 20 B; ops [op5:add x2, op6:matmul x12]; imports op4:matmul.out=4B, op5:add.out=4B, t1[token]=4B, x1[token]=8B
- RU94: work 5, input 12 B; ops [op5:add x2, op6:matmul x4]; imports op4:matmul.out=4B, t1[token]=4B, x1[token]=4B
- RU102: work 8, input 20 B; ops [op7:sgdupdate x8]; imports lr[seed]=4B, op3:sgdupdate.out=8B, op6:matmul.out=8B

**deep_chain[depth=4,q=1,d=2,W=fixed]** (micro deep chain (fixed weights)) F=1·fwd G=1·fwd: L=4, I*=16
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B

**deep_chain[depth=4,q=1,d=2,W=fixed]** (micro deep chain (fixed weights)) F=1·fwd G=4·fwd: L=4, I*=16
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B

**deep_chain[depth=4,q=1,d=2,W=fixed]** (micro deep chain (fixed weights)) F=1·fwd G=inf: L=4, I*=16
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B

**deep_chain[depth=4,q=1,d=2,W=fixed]** (micro deep chain (fixed weights)) F=1.5·fwd G=1·fwd: L=4, I*=16
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B

**deep_chain[depth=4,q=1,d=2,W=fixed]** (micro deep chain (fixed weights)) F=3·fwd G=1·fwd: L=4, I*=16
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B

**deep_chain[depth=4,q=1,d=2,W=fixed]** (micro deep chain (fixed weights)) F=inf G=1·fwd: L=4, I*=16
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B

**deep_chain[depth=6,q=1,d=2,W=fixed]** (micro deep chain (fixed weights)) F=1·fwd G=1·fwd: L=4, I*=24
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B
- RU4: work 6, input 4 B; ops [op4:matmul x8]; imports op3:matmul.out=4B
- RU5: work 6, input 4 B; ops [op5:matmul x8]; imports op4:matmul.out=4B

**deep_chain[depth=6,q=1,d=2,W=fixed]** (micro deep chain (fixed weights)) F=1·fwd G=4·fwd: L=4, I*=24
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B
- RU4: work 6, input 4 B; ops [op4:matmul x8]; imports op3:matmul.out=4B
- RU5: work 6, input 4 B; ops [op5:matmul x8]; imports op4:matmul.out=4B

**deep_chain[depth=6,q=1,d=2,W=fixed]** (micro deep chain (fixed weights)) F=1·fwd G=inf: L=4, I*=24
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B
- RU4: work 6, input 4 B; ops [op4:matmul x8]; imports op3:matmul.out=4B
- RU5: work 6, input 4 B; ops [op5:matmul x8]; imports op4:matmul.out=4B

**deep_chain[depth=6,q=1,d=2,W=fixed]** (micro deep chain (fixed weights)) F=1.5·fwd G=1·fwd: L=4, I*=24
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B
- RU4: work 6, input 4 B; ops [op4:matmul x8]; imports op3:matmul.out=4B
- RU5: work 6, input 4 B; ops [op5:matmul x8]; imports op4:matmul.out=4B

**deep_chain[depth=6,q=1,d=2,W=fixed]** (micro deep chain (fixed weights)) F=1.5·fwd G=4·fwd: L=4, I*=12
- RU0: work 12, input 4 B; ops [op0:matmul x8, op1:matmul x8]; imports x[token]=4B
- RU1: work 12, input 4 B; ops [op2:matmul x8, op3:matmul x8]; imports op1:matmul.out=4B
- RU2: work 12, input 4 B; ops [op4:matmul x8, op5:matmul x8]; imports op3:matmul.out=4B

**deep_chain[depth=6,q=1,d=2,W=fixed]** (micro deep chain (fixed weights)) F=1.5·fwd G=inf: L=4, I*=12
- RU0: work 12, input 4 B; ops [op0:matmul x8, op1:matmul x8]; imports x[token]=4B
- RU1: work 12, input 4 B; ops [op2:matmul x8, op3:matmul x8]; imports op1:matmul.out=4B
- RU2: work 12, input 4 B; ops [op4:matmul x8, op5:matmul x8]; imports op3:matmul.out=4B

**deep_chain[depth=6,q=1,d=2,W=fixed]** (micro deep chain (fixed weights)) F=3·fwd G=1·fwd: L=4, I*=24
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B
- RU4: work 6, input 4 B; ops [op4:matmul x8]; imports op3:matmul.out=4B
- RU5: work 6, input 4 B; ops [op5:matmul x8]; imports op4:matmul.out=4B

**deep_chain[depth=6,q=1,d=2,W=fixed]** (micro deep chain (fixed weights)) F=inf G=1·fwd: L=4, I*=24
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports op0:matmul.out=4B
- RU2: work 6, input 4 B; ops [op2:matmul x8]; imports op1:matmul.out=4B
- RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:matmul.out=4B
- RU4: work 6, input 4 B; ops [op4:matmul x8]; imports op3:matmul.out=4B
- RU5: work 6, input 4 B; ops [op5:matmul x8]; imports op4:matmul.out=4B

**dense_rollout[q=4,d=1,L=1]** (dense rollout micro) F=0.5·fwd G=4·fwd: L=10, I*=26
- RU0: work 4, input 10 B; ops [op0:matmul x2, op1:matmul x2, op2:matmul x2, op3:matmul x2]; imports W0[accumulated]=2B, x0[token]=2B, x1[token]=2B, x2[token]=2B, x3[token]=2B
- RU1: work 4, input 16 B; ops [op0:matmul x1, op1:matmul x1, op2:matmul x1, op3:matmul x1]; imports op0:matmul.out=4B, op1:matmul.out=4B, op2:matmul.out=4B, op3:matmul.out=4B

**dense_rollout[q=8,d=1,L=1]** (dense rollout micro) F=0.5·fwd G=4·fwd: L=18, I*=50
- RU0: work 8, input 18 B; ops [op0:matmul x2, op1:matmul x2, op2:matmul x2, op3:matmul x2, op4:matmul x2, op5:matmul x2, op6:matmul x2, op7:matmul x2]; imports W0[accumulated]=2B, x0[token]=2B, x1[token]=2B, x2[token]=2B, x3[token]=2B, x4[token]=2B, x5[token]=2B, x6[token]=2B, x7[token]=2B
- RU1: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
- RU2: work 1, input 4 B; ops [op1:matmul x1]; imports op1:matmul.out=4B
- RU3: work 2, input 8 B; ops [op2:matmul x1, op6:matmul x1]; imports op2:matmul.out=4B, op6:matmul.out=4B
- RU4: work 1, input 4 B; ops [op3:matmul x1]; imports op3:matmul.out=4B
- RU5: work 2, input 8 B; ops [op4:matmul x1, op7:matmul x1]; imports op4:matmul.out=4B, op7:matmul.out=4B
- RU6: work 1, input 4 B; ops [op5:matmul x1]; imports op5:matmul.out=4B

**dense_rollout[q=6,d=1,L=2]** (dense rollout micro) F=1.5·fwd G=1·fwd: L=16, I*=36
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:matmul x3]; imports W0[accumulated]=2B, W1[accumulated]=2B, x0[token]=2B
- RU1: work 4, input 6 B; ops [op2:matmul x3, op3:matmul x3]; imports W0[accumulated]=2B, W1[accumulated]=2B, x1[token]=2B
- RU2: work 4, input 6 B; ops [op4:matmul x3, op5:matmul x3]; imports W0[accumulated]=2B, W1[accumulated]=2B, x2[token]=2B
- RU3: work 4, input 6 B; ops [op6:matmul x3, op7:matmul x3]; imports W0[accumulated]=2B, W1[accumulated]=2B, x3[token]=2B
- RU4: work 4, input 6 B; ops [op8:matmul x3, op9:matmul x3]; imports W0[accumulated]=2B, W1[accumulated]=2B, x4[token]=2B
- RU5: work 4, input 6 B; ops [op10:matmul x3, op11:matmul x3]; imports W0[accumulated]=2B, W1[accumulated]=2B, x5[token]=2B

**dense_rollout[q=6,d=1,L=2]** (dense rollout micro) F=inf G=1·fwd: L=16, I*=36
- RU0: work 4, input 6 B; ops [op0:matmul x3, op1:matmul x3]; imports W0[accumulated]=2B, W1[accumulated]=2B, x0[token]=2B
- RU1: work 4, input 6 B; ops [op2:matmul x3, op3:matmul x3]; imports W0[accumulated]=2B, W1[accumulated]=2B, x1[token]=2B
- RU2: work 4, input 6 B; ops [op4:matmul x3, op5:matmul x3]; imports W0[accumulated]=2B, W1[accumulated]=2B, x2[token]=2B
- RU3: work 4, input 6 B; ops [op6:matmul x3, op7:matmul x3]; imports W0[accumulated]=2B, W1[accumulated]=2B, x3[token]=2B
- RU4: work 4, input 6 B; ops [op8:matmul x3, op9:matmul x3]; imports W0[accumulated]=2B, W1[accumulated]=2B, x4[token]=2B
- RU5: work 4, input 6 B; ops [op10:matmul x3, op11:matmul x3]; imports W0[accumulated]=2B, W1[accumulated]=2B, x5[token]=2B

**moe_rollout[q=4,d=1,E=2,k=1,S=1]** (moe rollout micro) F=1.5·fwd G=1·fwd: L=14, I*=36
- RU0: work 3, input 6 B; ops [op0:matmul x3, op2:add x1]; imports S0[accumulated]=2B, op1:matmul.out=2B, x0[token]=2B
- RU1: work 4, input 6 B; ops [op1:matmul x3, op7:matmul x3]; imports E0[accumulated]=2B, x0[token]=2B, x2[token]=2B
- RU2: work 3, input 6 B; ops [op3:matmul x3, op5:add x1]; imports S0[accumulated]=2B, op4:matmul.out=2B, x1[token]=2B
- RU3: work 4, input 6 B; ops [op4:matmul x3, op10:matmul x3]; imports E1[accumulated]=2B, x1[token]=2B, x3[token]=2B
- RU4: work 3, input 6 B; ops [op6:matmul x3, op8:add x1]; imports S0[accumulated]=2B, op7:matmul.out=2B, x2[token]=2B
- RU5: work 3, input 6 B; ops [op9:matmul x3, op11:add x1]; imports S0[accumulated]=2B, op10:matmul.out=2B, x3[token]=2B

**moe_rollout[q=4,d=1,E=2,k=1,S=1]** (moe rollout micro) F=inf G=1·fwd: L=14, I*=36
- RU0: work 3, input 6 B; ops [op0:matmul x3, op2:add x1]; imports S0[accumulated]=2B, op1:matmul.out=2B, x0[token]=2B
- RU1: work 4, input 6 B; ops [op1:matmul x3, op7:matmul x3]; imports E0[accumulated]=2B, x0[token]=2B, x2[token]=2B
- RU2: work 3, input 6 B; ops [op3:matmul x3, op5:add x1]; imports S0[accumulated]=2B, op4:matmul.out=2B, x1[token]=2B
- RU3: work 4, input 6 B; ops [op4:matmul x3, op10:matmul x3]; imports E1[accumulated]=2B, x1[token]=2B, x3[token]=2B
- RU4: work 3, input 6 B; ops [op6:matmul x3, op8:add x1]; imports S0[accumulated]=2B, op7:matmul.out=2B, x2[token]=2B
- RU5: work 3, input 6 B; ops [op9:matmul x3, op11:add x1]; imports S0[accumulated]=2B, op10:matmul.out=2B, x3[token]=2B

**moe_rollout[q=4,d=1,E=2,k=1,S=1]** (moe rollout micro) F=0.5·fwd G=4·fwd: L=14, I*=30
- RU0: work 16, input 14 B; ops [op0:matmul x3, op1:matmul x3, op3:matmul x3, op4:matmul x3, op6:matmul x3, op7:matmul x3, op9:matmul x3, op10:matmul x3]; imports E0[accumulated]=2B, E1[accumulated]=2B, S0[accumulated]=2B, x0[token]=2B, x1[token]=2B, x2[token]=2B, x3[token]=2B
- RU1: work 4, input 16 B; ops [op2:add x1, op5:add x1, op8:add x1, op11:add x1]; imports op0:matmul.out=2B, op10:matmul.out=2B, op1:matmul.out=2B, op3:matmul.out=2B, op4:matmul.out=2B, op6:matmul.out=2B, op7:matmul.out=2B, op9:matmul.out=2B

**moe_rollout[q=8,d=1,E=2,k=1,S=1]** (moe rollout micro) F=1.5·fwd G=2·fwd: L=22, I*=[38, 48]
- RU0: work 7, input 8 B; ops [op0:matmul x3, op1:matmul x3, op2:add x1, op18:matmul x3]; imports E0[accumulated]=2B, S0[accumulated]=2B, x0[token]=2B, x6[token]=2B
- RU1: work 5, input 6 B; ops [op3:matmul x3, op4:matmul x3, op5:add x1]; imports E1[accumulated]=2B, S0[accumulated]=2B, x1[token]=2B
- RU2: work 5, input 6 B; ops [op6:matmul x3, op7:matmul x3, op8:add x1]; imports E0[accumulated]=2B, S0[accumulated]=2B, x2[token]=2B
- RU3: work 5, input 6 B; ops [op9:matmul x3, op10:matmul x3, op11:add x1]; imports E1[accumulated]=2B, S0[accumulated]=2B, x3[token]=2B
- RU4: work 8, input 10 B; ops [op12:matmul x3, op13:matmul x3, op14:add x1, op19:matmul x3, op20:add x1]; imports E0[accumulated]=2B, S0[accumulated]=2B, op18:matmul.out=2B, x4[token]=2B, x6[token]=2B
- RU5: work 5, input 6 B; ops [op15:matmul x3, op16:matmul x3, op17:add x1]; imports E1[accumulated]=2B, S0[accumulated]=2B, x5[token]=2B
- RU6: work 5, input 6 B; ops [op21:matmul x3, op22:matmul x3, op23:add x1]; imports E1[accumulated]=2B, S0[accumulated]=2B, x7[token]=2B

**moe_rollout[q=8,d=1,E=2,k=1,S=1]** (moe rollout micro) F=1.5·fwd G=1·fwd: L=22, I*=[66, 72]
- RU12: work 3, input 6 B; ops [op0:matmul x3, op2:add x1]; imports S0[accumulated]=2B, op1:matmul.out=2B, x0[token]=2B
- RU15: work 4, input 6 B; ops [op1:matmul x3, op7:matmul x3]; imports E0[accumulated]=2B, x0[token]=2B, x2[token]=2B
- RU19: work 3, input 6 B; ops [op3:matmul x3, op5:add x1]; imports S0[accumulated]=2B, op4:matmul.out=2B, x1[token]=2B
- RU22: work 4, input 6 B; ops [op4:matmul x3, op10:matmul x3]; imports E1[accumulated]=2B, x1[token]=2B, x3[token]=2B
- RU26: work 3, input 6 B; ops [op6:matmul x3, op8:add x1]; imports S0[accumulated]=2B, op7:matmul.out=2B, x2[token]=2B
- RU33: work 3, input 6 B; ops [op9:matmul x3, op11:add x1]; imports S0[accumulated]=2B, op10:matmul.out=2B, x3[token]=2B
- RU40: work 3, input 6 B; ops [op12:matmul x3, op14:add x1]; imports S0[accumulated]=2B, op13:matmul.out=2B, x4[token]=2B
- RU43: work 4, input 6 B; ops [op13:matmul x3, op19:matmul x3]; imports E0[accumulated]=2B, x4[token]=2B, x6[token]=2B
- RU47: work 3, input 6 B; ops [op15:matmul x3, op17:add x1]; imports S0[accumulated]=2B, op16:matmul.out=2B, x5[token]=2B
- RU50: work 4, input 6 B; ops [op16:matmul x3, op22:matmul x3]; imports E1[accumulated]=2B, x5[token]=2B, x7[token]=2B
- RU54: work 3, input 6 B; ops [op18:matmul x3, op20:add x1]; imports S0[accumulated]=2B, op19:matmul.out=2B, x6[token]=2B
- RU61: work 3, input 6 B; ops [op21:matmul x3, op23:add x1]; imports S0[accumulated]=2B, op22:matmul.out=2B, x7[token]=2B

**moe_rollout[q=8,d=1,E=2,k=1,S=1]** (moe rollout micro) F=inf G=1·fwd: L=22, I*=[66, 72]
- RU12: work 3, input 6 B; ops [op0:matmul x3, op2:add x1]; imports S0[accumulated]=2B, op1:matmul.out=2B, x0[token]=2B
- RU15: work 4, input 6 B; ops [op1:matmul x3, op7:matmul x3]; imports E0[accumulated]=2B, x0[token]=2B, x2[token]=2B
- RU19: work 3, input 6 B; ops [op3:matmul x3, op5:add x1]; imports S0[accumulated]=2B, op4:matmul.out=2B, x1[token]=2B
- RU22: work 4, input 6 B; ops [op4:matmul x3, op10:matmul x3]; imports E1[accumulated]=2B, x1[token]=2B, x3[token]=2B
- RU26: work 3, input 6 B; ops [op6:matmul x3, op8:add x1]; imports S0[accumulated]=2B, op7:matmul.out=2B, x2[token]=2B
- RU33: work 3, input 6 B; ops [op9:matmul x3, op11:add x1]; imports S0[accumulated]=2B, op10:matmul.out=2B, x3[token]=2B
- RU40: work 3, input 6 B; ops [op12:matmul x3, op14:add x1]; imports S0[accumulated]=2B, op13:matmul.out=2B, x4[token]=2B
- RU43: work 4, input 6 B; ops [op13:matmul x3, op19:matmul x3]; imports E0[accumulated]=2B, x4[token]=2B, x6[token]=2B
- RU47: work 3, input 6 B; ops [op15:matmul x3, op17:add x1]; imports S0[accumulated]=2B, op16:matmul.out=2B, x5[token]=2B
- RU50: work 4, input 6 B; ops [op16:matmul x3, op22:matmul x3]; imports E1[accumulated]=2B, x5[token]=2B, x7[token]=2B
- RU54: work 3, input 6 B; ops [op18:matmul x3, op20:add x1]; imports S0[accumulated]=2B, op19:matmul.out=2B, x6[token]=2B
- RU61: work 3, input 6 B; ops [op21:matmul x3, op23:add x1]; imports S0[accumulated]=2B, op22:matmul.out=2B, x7[token]=2B

**moe_rollout[q=8,d=1,E=2,k=1,S=1]** (moe rollout micro) F=0.5·fwd G=4·fwd: L=22, I*=56
- RU0: work 16, input 12 B; ops [op0:matmul x3, op1:matmul x3, op6:matmul x3, op7:matmul x3, op12:matmul x3, op13:matmul x3, op18:matmul x3, op19:matmul x3]; imports E0[accumulated]=2B, S0[accumulated]=2B, x0[token]=2B, x2[token]=2B, x4[token]=2B, x6[token]=2B
- RU1: work 1, input 4 B; ops [op2:add x1]; imports op0:matmul.out=2B, op1:matmul.out=2B
- RU2: work 16, input 12 B; ops [op3:matmul x3, op4:matmul x3, op9:matmul x3, op10:matmul x3, op15:matmul x3, op16:matmul x3, op21:matmul x3, op22:matmul x3]; imports E1[accumulated]=2B, S0[accumulated]=2B, x1[token]=2B, x3[token]=2B, x5[token]=2B, x7[token]=2B
- RU3: work 1, input 4 B; ops [op5:add x1]; imports op3:matmul.out=2B, op4:matmul.out=2B
- RU4: work 1, input 4 B; ops [op8:add x1]; imports op6:matmul.out=2B, op7:matmul.out=2B
- RU5: work 1, input 4 B; ops [op11:add x1]; imports op10:matmul.out=2B, op9:matmul.out=2B
- RU6: work 1, input 4 B; ops [op14:add x1]; imports op12:matmul.out=2B, op13:matmul.out=2B
- RU7: work 1, input 4 B; ops [op17:add x1]; imports op15:matmul.out=2B, op16:matmul.out=2B
- RU8: work 1, input 4 B; ops [op20:add x1]; imports op18:matmul.out=2B, op19:matmul.out=2B
- RU9: work 1, input 4 B; ops [op23:add x1]; imports op21:matmul.out=2B, op22:matmul.out=2B

**moe_rollout[q=4,d=1,E=4,k=2,S=1]** (moe rollout micro) F=1.5·fwd G=1·fwd: L=18, I*=[42, 46]
- RU0: work 6, input 8 B; ops [op0:matmul x3, op1:matmul x3, op3:matmul x3]; imports E0[accumulated]=2B, E1[accumulated]=2B, S0[accumulated]=2B, x0[token]=2B
- RU1: work 2, input 6 B; ops [op2:add x1, op4:add x1]; imports op0:matmul.out=2B, op1:matmul.out=2B, op3:matmul.out=2B
- RU2: work 6, input 8 B; ops [op5:matmul x3, op10:matmul x3, op15:matmul x3]; imports S0[accumulated]=2B, x1[token]=2B, x2[token]=2B, x3[token]=2B
- RU3: work 6, input 8 B; ops [op6:matmul x3, op7:add x1, op8:matmul x3, op9:add x1]; imports E1[accumulated]=2B, E2[accumulated]=2B, op5:matmul.out=2B, x1[token]=2B
- RU4: work 6, input 8 B; ops [op11:matmul x3, op12:add x1, op13:matmul x3, op14:add x1]; imports E2[accumulated]=2B, E3[accumulated]=2B, op10:matmul.out=2B, x2[token]=2B
- RU5: work 6, input 8 B; ops [op16:matmul x3, op17:add x1, op18:matmul x3, op19:add x1]; imports E0[accumulated]=2B, E3[accumulated]=2B, op15:matmul.out=2B, x3[token]=2B

**moe_rollout[q=4,d=1,E=4,k=2,S=1]** (moe rollout micro) F=inf G=1·fwd: L=18, I*=[42, 46]
- RU0: work 6, input 8 B; ops [op0:matmul x3, op1:matmul x3, op3:matmul x3]; imports E0[accumulated]=2B, E1[accumulated]=2B, S0[accumulated]=2B, x0[token]=2B
- RU1: work 2, input 6 B; ops [op2:add x1, op4:add x1]; imports op0:matmul.out=2B, op1:matmul.out=2B, op3:matmul.out=2B
- RU2: work 6, input 8 B; ops [op5:matmul x3, op10:matmul x3, op15:matmul x3]; imports S0[accumulated]=2B, x1[token]=2B, x2[token]=2B, x3[token]=2B
- RU3: work 6, input 8 B; ops [op6:matmul x3, op7:add x1, op8:matmul x3, op9:add x1]; imports E1[accumulated]=2B, E2[accumulated]=2B, op5:matmul.out=2B, x1[token]=2B
- RU4: work 6, input 8 B; ops [op11:matmul x3, op12:add x1, op13:matmul x3, op14:add x1]; imports E2[accumulated]=2B, E3[accumulated]=2B, op10:matmul.out=2B, x2[token]=2B
- RU5: work 6, input 8 B; ops [op16:matmul x3, op17:add x1, op18:matmul x3, op19:add x1]; imports E0[accumulated]=2B, E3[accumulated]=2B, op15:matmul.out=2B, x3[token]=2B

**moe_rollout[q=4,d=1,E=4,k=2,S=1]** (moe rollout micro) F=0.5·fwd G=4·fwd: L=18, I*=42
- RU0: work 24, input 18 B; ops [op0:matmul x3, op1:matmul x3, op3:matmul x3, op5:matmul x3, op6:matmul x3, op8:matmul x3, op10:matmul x3, op11:matmul x3, op13:matmul x3, op15:matmul x3, op16:matmul x3, op18:matmul x3]; imports E0[accumulated]=2B, E1[accumulated]=2B, E2[accumulated]=2B, E3[accumulated]=2B, S0[accumulated]=2B, x0[token]=2B, x1[token]=2B, x2[token]=2B, x3[token]=2B
- RU1: work 2, input 6 B; ops [op2:add x1, op4:add x1]; imports op0:matmul.out=2B, op1:matmul.out=2B, op3:matmul.out=2B
- RU2: work 2, input 6 B; ops [op7:add x1, op9:add x1]; imports op5:matmul.out=2B, op6:matmul.out=2B, op8:matmul.out=2B
- RU3: work 2, input 6 B; ops [op12:add x1, op14:add x1]; imports op10:matmul.out=2B, op11:matmul.out=2B, op13:matmul.out=2B
- RU4: work 2, input 6 B; ops [op17:add x1, op19:add x1]; imports op15:matmul.out=2B, op16:matmul.out=2B, op18:matmul.out=2B

**moe_rollout[q=6,d=1,E=3,k=1,S=0]** (moe rollout micro) F=0.5·fwd G=4·fwd: L=18, I*=42
- RU0: work 6, input 18 B; ops [op0:matmul x2, op1:matmul x2, op2:matmul x2, op3:matmul x2, op4:matmul x2, op5:matmul x2]; imports E0[accumulated]=2B, E1[accumulated]=2B, E2[accumulated]=2B, x0[token]=2B, x1[token]=2B, x2[token]=2B, x3[token]=2B, x4[token]=2B, x5[token]=2B
- RU1: work 6, input 24 B; ops [op0:matmul x1, op1:matmul x1, op2:matmul x1, op3:matmul x1, op4:matmul x1, op5:matmul x1]; imports op0:matmul.out=4B, op1:matmul.out=4B, op2:matmul.out=4B, op3:matmul.out=4B, op4:matmul.out=4B, op5:matmul.out=4B

**mm_3x2x2** (mm) F=W/4 G=W/4: L=20, I*=48
- RU0: work 5, input 12 B; ops [op0:matmul x7]; imports A[token]=4B, B[accumulated]=8B
- RU1: work 5, input 14 B; ops [op0:matmul x7]; imports A[token]=4B, B[accumulated]=6B, op0:matmul.out=4B
- RU2: work 5, input 14 B; ops [op0:matmul x6]; imports A[token]=6B, B[accumulated]=4B, op0:matmul.out=4B
- RU3: work 3, input 8 B; ops [op0:matmul x4]; imports A[token]=4B, B[accumulated]=4B

**mm_3x2x2** (mm) F=W/2 G=W/4: L=20, I*=48
- RU0: work 5, input 12 B; ops [op0:matmul x7]; imports A[token]=4B, B[accumulated]=8B
- RU1: work 5, input 14 B; ops [op0:matmul x7]; imports A[token]=4B, B[accumulated]=6B, op0:matmul.out=4B
- RU2: work 5, input 14 B; ops [op0:matmul x6]; imports A[token]=6B, B[accumulated]=4B, op0:matmul.out=4B
- RU3: work 3, input 8 B; ops [op0:matmul x4]; imports A[token]=4B, B[accumulated]=4B

**mm_3x2x2** (mm) F=inf G=W/4: L=20, I*=48
- RU0: work 5, input 12 B; ops [op0:matmul x7]; imports A[token]=4B, B[accumulated]=8B
- RU1: work 5, input 14 B; ops [op0:matmul x7]; imports A[token]=4B, B[accumulated]=6B, op0:matmul.out=4B
- RU2: work 5, input 14 B; ops [op0:matmul x6]; imports A[token]=6B, B[accumulated]=4B, op0:matmul.out=4B
- RU3: work 3, input 8 B; ops [op0:matmul x4]; imports A[token]=4B, B[accumulated]=4B

**mm_3x3x2** (mm) F=W/4 G=W/4: L=24, I*=[50, 56]
- RU13: work 6, input 12 B; ops [op0:matmul x8]; imports A[token]=4B, B[accumulated]=8B
- RU21: work 6, input 12 B; ops [op0:matmul x8]; imports A[token]=8B, B[accumulated]=4B
- RU25: work 6, input 12 B; ops [op0:matmul x8]; imports A[token]=4B, B[accumulated]=8B
- RU37: work 6, input 12 B; ops [op0:matmul x8]; imports A[token]=4B, B[accumulated]=8B
- RU45: work 3, input 8 B; ops [op0:matmul x4]; imports A[token]=4B, B[accumulated]=4B

**mm_3x3x2** (mm) F=W/2 G=W/4: L=24, I*=[50, 56]
- RU13: work 6, input 12 B; ops [op0:matmul x8]; imports A[token]=4B, B[accumulated]=8B
- RU21: work 6, input 12 B; ops [op0:matmul x8]; imports A[token]=8B, B[accumulated]=4B
- RU25: work 6, input 12 B; ops [op0:matmul x8]; imports A[token]=4B, B[accumulated]=8B
- RU37: work 6, input 12 B; ops [op0:matmul x8]; imports A[token]=4B, B[accumulated]=8B
- RU45: work 3, input 8 B; ops [op0:matmul x4]; imports A[token]=4B, B[accumulated]=4B

**mm_3x3x2** (mm) F=inf G=W/4: L=24, I*=[50, 56]
- RU13: work 6, input 12 B; ops [op0:matmul x8]; imports A[token]=4B, B[accumulated]=8B
- RU21: work 6, input 12 B; ops [op0:matmul x8]; imports A[token]=8B, B[accumulated]=4B
- RU25: work 6, input 12 B; ops [op0:matmul x8]; imports A[token]=4B, B[accumulated]=8B
- RU37: work 6, input 12 B; ops [op0:matmul x8]; imports A[token]=4B, B[accumulated]=8B
- RU45: work 3, input 8 B; ops [op0:matmul x4]; imports A[token]=4B, B[accumulated]=4B

**chain_1x1x1x1** (chain) F=W/4 G=W/4: L=6, I*=16
- RU0: work 1, input 4 B; ops [op0:matmul x2]; imports A[token]=2B, B1[accumulated]=2B
- RU1: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
- RU2: work 1, input 4 B; ops [op1:matmul x2]; imports B2[accumulated]=2B, op0:matmul.out=2B
- RU3: work 1, input 4 B; ops [op1:matmul x1]; imports op1:matmul.out=4B

**chain_1x1x1x1** (chain) F=W/4 G=W/2: L=6, I*=16
- RU0: work 1, input 4 B; ops [op0:matmul x2]; imports A[token]=2B, B1[accumulated]=2B
- RU1: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
- RU2: work 1, input 4 B; ops [op1:matmul x2]; imports B2[accumulated]=2B, op0:matmul.out=2B
- RU3: work 1, input 4 B; ops [op1:matmul x1]; imports op1:matmul.out=4B

**chain_1x1x1x1** (chain) F=W/4 G=inf: L=6, I*=16
- RU0: work 1, input 4 B; ops [op0:matmul x2]; imports A[token]=2B, B1[accumulated]=2B
- RU1: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
- RU2: work 1, input 4 B; ops [op1:matmul x2]; imports B2[accumulated]=2B, op0:matmul.out=2B
- RU3: work 1, input 4 B; ops [op1:matmul x1]; imports op1:matmul.out=4B

**chain_1x1x1x1** (chain) F=W/2 G=W/4: L=6, I*=16
- RU0: work 1, input 4 B; ops [op0:matmul x2]; imports A[token]=2B, B1[accumulated]=2B
- RU1: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
- RU2: work 1, input 4 B; ops [op1:matmul x2]; imports B2[accumulated]=2B, op0:matmul.out=2B
- RU3: work 1, input 4 B; ops [op1:matmul x1]; imports op1:matmul.out=4B

**chain_1x1x1x1** (chain) F=inf G=W/4: L=6, I*=16
- RU0: work 1, input 4 B; ops [op0:matmul x2]; imports A[token]=2B, B1[accumulated]=2B
- RU1: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
- RU2: work 1, input 4 B; ops [op1:matmul x2]; imports B2[accumulated]=2B, op0:matmul.out=2B
- RU3: work 1, input 4 B; ops [op1:matmul x1]; imports op1:matmul.out=4B

**fixacc_1x1x1** (fixacc) F=W/4 G=W/4: L=4, I*=14
- RU0: work 1, input 2 B; ops [op0:matmul x2]; imports A[token]=2B
- RU1: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
- RU2: work 1, input 4 B; ops [op1:matmul x2]; imports Wa[accumulated]=2B, op0:matmul.out=2B
- RU3: work 1, input 4 B; ops [op1:matmul x1]; imports op1:matmul.out=4B

**fixacc_1x1x1** (fixacc) F=W/4 G=W/2: L=4, I*=14
- RU0: work 1, input 2 B; ops [op0:matmul x2]; imports A[token]=2B
- RU1: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
- RU2: work 1, input 4 B; ops [op1:matmul x2]; imports Wa[accumulated]=2B, op0:matmul.out=2B
- RU3: work 1, input 4 B; ops [op1:matmul x1]; imports op1:matmul.out=4B

**fixacc_1x1x1** (fixacc) F=W/4 G=inf: L=4, I*=14
- RU0: work 1, input 2 B; ops [op0:matmul x2]; imports A[token]=2B
- RU1: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
- RU2: work 1, input 4 B; ops [op1:matmul x2]; imports Wa[accumulated]=2B, op0:matmul.out=2B
- RU3: work 1, input 4 B; ops [op1:matmul x1]; imports op1:matmul.out=4B

**fixacc_1x1x1** (fixacc) F=W/2 G=W/4: L=4, I*=14
- RU0: work 1, input 2 B; ops [op0:matmul x2]; imports A[token]=2B
- RU1: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
- RU2: work 1, input 4 B; ops [op1:matmul x2]; imports Wa[accumulated]=2B, op0:matmul.out=2B
- RU3: work 1, input 4 B; ops [op1:matmul x1]; imports op1:matmul.out=4B

**fixacc_1x1x1** (fixacc) F=inf G=W/4: L=4, I*=14
- RU0: work 1, input 2 B; ops [op0:matmul x2]; imports A[token]=2B
- RU1: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
- RU2: work 1, input 4 B; ops [op1:matmul x2]; imports Wa[accumulated]=2B, op0:matmul.out=2B
- RU3: work 1, input 4 B; ops [op1:matmul x1]; imports op1:matmul.out=4B

**fixacc_2x2x1** (fixacc) F=W/4 G=W/4: L=12, I*=32
- RU0: work 5, input 8 B; ops [op0:matmul x3, op1:matmul x4]; imports A[token]=2B, Wa[accumulated]=4B, op0:matmul.out=2B
- RU1: work 5, input 8 B; ops [op0:matmul x3, op1:matmul x4]; imports A[token]=2B, Wa[accumulated]=4B, op0:matmul.out=2B
- RU2: work 5, input 8 B; ops [op0:matmul x3, op1:matmul x4]; imports A[token]=2B, Wa[accumulated]=4B, op0:matmul.out=2B
- RU3: work 5, input 8 B; ops [op0:matmul x3, op1:matmul x4]; imports A[token]=2B, Wa[accumulated]=4B, op0:matmul.out=2B

**fixacc_2x2x1** (fixacc) F=W/2 G=W/4: L=12, I*=32
- RU0: work 5, input 8 B; ops [op0:matmul x3, op1:matmul x4]; imports A[token]=2B, Wa[accumulated]=4B, op0:matmul.out=2B
- RU1: work 5, input 8 B; ops [op0:matmul x3, op1:matmul x4]; imports A[token]=2B, Wa[accumulated]=4B, op0:matmul.out=2B
- RU2: work 5, input 8 B; ops [op0:matmul x3, op1:matmul x4]; imports A[token]=2B, Wa[accumulated]=4B, op0:matmul.out=2B
- RU3: work 5, input 8 B; ops [op0:matmul x3, op1:matmul x4]; imports A[token]=2B, Wa[accumulated]=4B, op0:matmul.out=2B

**fixacc_2x2x1** (fixacc) F=inf G=W/4: L=12, I*=32
- RU0: work 5, input 8 B; ops [op0:matmul x3, op1:matmul x4]; imports A[token]=2B, Wa[accumulated]=4B, op0:matmul.out=2B
- RU1: work 5, input 8 B; ops [op0:matmul x3, op1:matmul x4]; imports A[token]=2B, Wa[accumulated]=4B, op0:matmul.out=2B
- RU2: work 5, input 8 B; ops [op0:matmul x3, op1:matmul x4]; imports A[token]=2B, Wa[accumulated]=4B, op0:matmul.out=2B
- RU3: work 5, input 8 B; ops [op0:matmul x3, op1:matmul x4]; imports A[token]=2B, Wa[accumulated]=4B, op0:matmul.out=2B

**a2_fixed_chain_F_split** (a) F=W/4 G=W/4: L=4, I*=12
- RU0: work 3, input 4 B; ops [op0:matmul x4]; imports x[token]=4B
- RU1: work 3, input 4 B; ops [op0:matmul x4]; imports x[token]=4B
- RU2: work 3, input 4 B; ops [op1:matmul x4]; imports op0:matmul.out=4B

**a2_fixed_chain_F_split** (a) F=W/4 G=W/2: L=4, I*=12
- RU0: work 5, input 4 B; ops [op0:matmul x7]; imports x[token]=4B
- RU1: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
- RU2: work 3, input 4 B; ops [op1:matmul x4]; imports op0:matmul.out=4B

**a2_fixed_chain_F_split** (a) F=W/2 G=W/4: L=4, I*=12
- RU0: work 3, input 4 B; ops [op0:matmul x4]; imports x[token]=4B
- RU1: work 3, input 4 B; ops [op0:matmul x4]; imports x[token]=4B
- RU2: work 3, input 4 B; ops [op1:matmul x4]; imports op0:matmul.out=4B

**a2_fixed_chain_F_split** (a) F=W/2 G=W/2: L=4, I*=10
- RU0: work 5, input 4 B; ops [op0:matmul x7]; imports x[token]=4B
- RU1: work 4, input 6 B; ops [op0:matmul x1, op1:matmul x4]; imports op0:matmul.out=6B

**a2_fixed_chain_F_split** (a) F=inf G=W/4: L=4, I*=12
- RU0: work 3, input 4 B; ops [op0:matmul x4]; imports x[token]=4B
- RU1: work 3, input 4 B; ops [op0:matmul x4]; imports x[token]=4B
- RU2: work 3, input 4 B; ops [op1:matmul x4]; imports op0:matmul.out=4B

**a2_fixed_chain_F_split** (a) F=inf G=W/2: L=4, I*=10
- RU0: work 5, input 4 B; ops [op0:matmul x7]; imports x[token]=4B
- RU1: work 4, input 6 B; ops [op0:matmul x1, op1:matmul x4]; imports op0:matmul.out=6B

**b1_compact_precursor_fanout** (b) F=W/4 G=W/4: L=4, I*=24
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports h[accumulated]=4B
- RU1: work 6, input 4 B; ops [op0:matmul x8]; imports h[accumulated]=4B
- RU2: work 5, input 8 B; ops [op1:matmul x6]; imports op0:matmul.out=8B
- RU3: work 5, input 8 B; ops [op1:matmul x6]; imports op0:matmul.out=8B

**b1_compact_precursor_fanout** (b) F=W/4 G=W/2: L=4, I*=14
- RU0: work 11, input 4 B; ops [op0:matmul x15]; imports h[accumulated]=4B
- RU1: work 11, input 10 B; ops [op0:matmul x1, op1:matmul x12]; imports op0:matmul.out=10B

**b1_compact_precursor_fanout** (b) F=W/4 G=inf: L=4, I*=12
- RU0: work 12, input 4 B; ops [op0:matmul x16]; imports h[accumulated]=4B
- RU1: work 10, input 8 B; ops [op1:matmul x12]; imports op0:matmul.out=8B

**b1_compact_precursor_fanout** (b) F=W/2 G=W/4: L=4, I*=24
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports h[accumulated]=4B
- RU1: work 6, input 4 B; ops [op0:matmul x8]; imports h[accumulated]=4B
- RU2: work 5, input 8 B; ops [op1:matmul x6]; imports op0:matmul.out=8B
- RU3: work 5, input 8 B; ops [op1:matmul x6]; imports op0:matmul.out=8B

**b1_compact_precursor_fanout** (b) F=W/2 G=W/2: L=4, I*=14
- RU0: work 11, input 4 B; ops [op0:matmul x15]; imports h[accumulated]=4B
- RU1: work 11, input 10 B; ops [op0:matmul x1, op1:matmul x12]; imports op0:matmul.out=10B

**b1_compact_precursor_fanout** (b) F=W/2 G=inf: L=4, I*=12
- RU0: work 12, input 4 B; ops [op0:matmul x16]; imports h[accumulated]=4B
- RU1: work 10, input 8 B; ops [op1:matmul x12]; imports op0:matmul.out=8B

**b1_compact_precursor_fanout** (b) F=inf G=W/4: L=4, I*=24
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports h[accumulated]=4B
- RU1: work 6, input 4 B; ops [op0:matmul x8]; imports h[accumulated]=4B
- RU2: work 5, input 8 B; ops [op1:matmul x6]; imports op0:matmul.out=8B
- RU3: work 5, input 8 B; ops [op1:matmul x6]; imports op0:matmul.out=8B

**b1_compact_precursor_fanout** (b) F=inf G=W/2: L=4, I*=14
- RU0: work 11, input 4 B; ops [op0:matmul x15]; imports h[accumulated]=4B
- RU1: work 11, input 10 B; ops [op0:matmul x1, op1:matmul x12]; imports op0:matmul.out=10B

**e1_shared_weights_fixed_acts** (e) F=W/4 G=W/4: L=8, I*=24
- RU0: work 5, input 4 B; ops [op0:matmul x4, op1:matmul x3]; imports W[accumulated]=4B
- RU1: work 5, input 8 B; ops [op0:matmul x4, op1:matmul x3]; imports W[accumulated]=4B, op1:matmul.out=4B
- RU2: work 5, input 8 B; ops [op1:matmul x2, op2:matmul x4]; imports W[accumulated]=4B, op1:matmul.out=4B
- RU3: work 3, input 4 B; ops [op2:matmul x4]; imports W[accumulated]=4B

**e1_shared_weights_fixed_acts** (e) F=W/2 G=W/4: L=8, I*=24
- RU0: work 5, input 4 B; ops [op0:matmul x4, op1:matmul x3]; imports W[accumulated]=4B
- RU1: work 5, input 8 B; ops [op0:matmul x4, op1:matmul x3]; imports W[accumulated]=4B, op1:matmul.out=4B
- RU2: work 5, input 8 B; ops [op1:matmul x2, op2:matmul x4]; imports W[accumulated]=4B, op1:matmul.out=4B
- RU3: work 3, input 4 B; ops [op2:matmul x4]; imports W[accumulated]=4B

**e1_shared_weights_fixed_acts** (e) F=inf G=W/4: L=8, I*=24
- RU0: work 5, input 4 B; ops [op0:matmul x4, op1:matmul x3]; imports W[accumulated]=4B
- RU1: work 5, input 8 B; ops [op0:matmul x4, op1:matmul x3]; imports W[accumulated]=4B, op1:matmul.out=4B
- RU2: work 5, input 8 B; ops [op1:matmul x2, op2:matmul x4]; imports W[accumulated]=4B, op1:matmul.out=4B
- RU3: work 3, input 4 B; ops [op2:matmul x4]; imports W[accumulated]=4B

**e2_shared_weights_token_acts** (e) F=W/4 G=W/4: L=20, I*=48
- RU0: work 5, input 12 B; ops [op0:matmul x7]; imports A1[token]=4B, W[accumulated]=8B
- RU1: work 5, input 14 B; ops [op0:matmul x1, op1:matmul x6]; imports A2[token]=4B, W[accumulated]=6B, op0:matmul.out=4B
- RU2: work 5, input 14 B; ops [op1:matmul x2, op2:matmul x4]; imports A2[token]=2B, A3[token]=4B, W[accumulated]=4B, op1:matmul.out=4B
- RU3: work 3, input 8 B; ops [op2:matmul x4]; imports A3[token]=4B, W[accumulated]=4B

**e2_shared_weights_token_acts** (e) F=W/2 G=W/4: L=20, I*=48
- RU0: work 5, input 12 B; ops [op0:matmul x7]; imports A1[token]=4B, W[accumulated]=8B
- RU1: work 5, input 14 B; ops [op0:matmul x1, op1:matmul x6]; imports A2[token]=4B, W[accumulated]=6B, op0:matmul.out=4B
- RU2: work 5, input 14 B; ops [op1:matmul x2, op2:matmul x4]; imports A2[token]=2B, A3[token]=4B, W[accumulated]=4B, op1:matmul.out=4B
- RU3: work 3, input 8 B; ops [op2:matmul x4]; imports A3[token]=4B, W[accumulated]=4B

**e2_shared_weights_token_acts** (e) F=inf G=W/4: L=20, I*=48
- RU0: work 5, input 12 B; ops [op0:matmul x7]; imports A1[token]=4B, W[accumulated]=8B
- RU1: work 5, input 14 B; ops [op0:matmul x1, op1:matmul x6]; imports A2[token]=4B, W[accumulated]=6B, op0:matmul.out=4B
- RU2: work 5, input 14 B; ops [op1:matmul x2, op2:matmul x4]; imports A2[token]=2B, A3[token]=4B, W[accumulated]=4B, op1:matmul.out=4B
- RU3: work 3, input 8 B; ops [op2:matmul x4]; imports A3[token]=4B, W[accumulated]=4B

**f1_lora_fixed_x** (f) F=W/4 G=W/4: L=6, I*=16
- RU0: work 4, input 2 B; ops [op0:matmul x4, op3:add x1]; imports op2:matmul.out=2B
- RU1: work 4, input 2 B; ops [op0:matmul x4, op3:add x1]; imports op2:matmul.out=2B
- RU2: work 5, input 4 B; ops [op1:matmul x7]; imports A[accumulated]=4B
- RU3: work 5, input 8 B; ops [op1:matmul x1, op2:matmul x6]; imports Bm[accumulated]=2B, op1:matmul.out=6B

**f1_lora_fixed_x** (f) F=W/2 G=W/4: L=6, I*=16
- RU0: work 4, input 2 B; ops [op0:matmul x4, op3:add x1]; imports op2:matmul.out=2B
- RU1: work 4, input 2 B; ops [op0:matmul x4, op3:add x1]; imports op2:matmul.out=2B
- RU2: work 5, input 4 B; ops [op1:matmul x7]; imports A[accumulated]=4B
- RU3: work 5, input 8 B; ops [op1:matmul x1, op2:matmul x6]; imports Bm[accumulated]=2B, op1:matmul.out=6B

**f1_lora_fixed_x** (f) F=inf G=W/4: L=6, I*=16
- RU0: work 4, input 2 B; ops [op0:matmul x4, op3:add x1]; imports op2:matmul.out=2B
- RU1: work 4, input 2 B; ops [op0:matmul x4, op3:add x1]; imports op2:matmul.out=2B
- RU2: work 5, input 4 B; ops [op1:matmul x7]; imports A[accumulated]=4B
- RU3: work 5, input 8 B; ops [op1:matmul x1, op2:matmul x6]; imports Bm[accumulated]=2B, op1:matmul.out=6B

**f2_lora_token_x** (f) F=W/4 G=W/4: L=14, I*=32
- RU0: work 5, input 8 B; ops [op0:matmul x4, op1:matmul x3]; imports A[accumulated]=4B, x[token]=4B
- RU1: work 5, input 8 B; ops [op0:matmul x4, op1:matmul x3]; imports A[accumulated]=4B, x[token]=4B
- RU2: work 4, input 8 B; ops [op1:matmul x1, op2:matmul x3, op3:add x1]; imports Bm[accumulated]=2B, op0:matmul.out=2B, op1:matmul.out=4B
- RU3: work 4, input 8 B; ops [op1:matmul x1, op2:matmul x3, op3:add x1]; imports Bm[accumulated]=2B, op0:matmul.out=2B, op1:matmul.out=4B

**f2_lora_token_x** (f) F=W/2 G=W/4: L=14, I*=32
- RU0: work 5, input 8 B; ops [op0:matmul x4, op1:matmul x3]; imports A[accumulated]=4B, x[token]=4B
- RU1: work 5, input 8 B; ops [op0:matmul x4, op1:matmul x3]; imports A[accumulated]=4B, x[token]=4B
- RU2: work 4, input 8 B; ops [op1:matmul x1, op2:matmul x3, op3:add x1]; imports Bm[accumulated]=2B, op0:matmul.out=2B, op1:matmul.out=4B
- RU3: work 4, input 8 B; ops [op1:matmul x1, op2:matmul x3, op3:add x1]; imports Bm[accumulated]=2B, op0:matmul.out=2B, op1:matmul.out=4B

**f2_lora_token_x** (f) F=inf G=W/4: L=14, I*=32
- RU0: work 5, input 8 B; ops [op0:matmul x4, op1:matmul x3]; imports A[accumulated]=4B, x[token]=4B
- RU1: work 5, input 8 B; ops [op0:matmul x4, op1:matmul x3]; imports A[accumulated]=4B, x[token]=4B
- RU2: work 4, input 8 B; ops [op1:matmul x1, op2:matmul x3, op3:add x1]; imports Bm[accumulated]=2B, op0:matmul.out=2B, op1:matmul.out=4B
- RU3: work 4, input 8 B; ops [op1:matmul x1, op2:matmul x3, op3:add x1]; imports Bm[accumulated]=2B, op0:matmul.out=2B, op1:matmul.out=4B

**g1_residual_add** (g) F=W/4 G=W/4: L=12, I*=28
- RU0: work 2, input 8 B; ops [op0:matmul x3]; imports W[accumulated]=4B, x[carried]=4B
- RU1: work 2, input 6 B; ops [op0:matmul x1, op1:add x1]; imports op0:matmul.out=4B, x[carried]=2B
- RU2: work 2, input 8 B; ops [op0:matmul x3]; imports W[accumulated]=4B, x[carried]=4B
- RU3: work 2, input 6 B; ops [op0:matmul x1, op1:add x1]; imports op0:matmul.out=4B, x[carried]=2B

**g1_residual_add** (g) F=W/2 G=W/4: L=12, I*=28
- RU0: work 2, input 8 B; ops [op0:matmul x3]; imports W[accumulated]=4B, x[carried]=4B
- RU1: work 2, input 6 B; ops [op0:matmul x1, op1:add x1]; imports op0:matmul.out=4B, x[carried]=2B
- RU2: work 2, input 8 B; ops [op0:matmul x3]; imports W[accumulated]=4B, x[carried]=4B
- RU3: work 2, input 6 B; ops [op0:matmul x1, op1:add x1]; imports op0:matmul.out=4B, x[carried]=2B

**g1_residual_add** (g) F=inf G=W/4: L=12, I*=28
- RU0: work 2, input 8 B; ops [op0:matmul x3]; imports W[accumulated]=4B, x[carried]=4B
- RU1: work 2, input 6 B; ops [op0:matmul x1, op1:add x1]; imports op0:matmul.out=4B, x[carried]=2B
- RU2: work 2, input 8 B; ops [op0:matmul x3]; imports W[accumulated]=4B, x[carried]=4B
- RU3: work 2, input 6 B; ops [op0:matmul x1, op1:add x1]; imports op0:matmul.out=4B, x[carried]=2B

**h4_gram_matrix** (h) F=W/4 G=W/4: L=8, I*=24
- RU0: work 3, input 4 B; ops [op0:matmul x4]; imports x[carried]=4B
- RU1: work 3, input 8 B; ops [op0:matmul x4]; imports x[carried]=8B
- RU2: work 3, input 8 B; ops [op0:matmul x4]; imports x[carried]=8B
- RU3: work 3, input 4 B; ops [op0:matmul x4]; imports x[carried]=4B

**h4_gram_matrix** (h) F=W/2 G=W/4: L=8, I*=24
- RU0: work 3, input 4 B; ops [op0:matmul x4]; imports x[carried]=4B
- RU1: work 3, input 8 B; ops [op0:matmul x4]; imports x[carried]=8B
- RU2: work 3, input 8 B; ops [op0:matmul x4]; imports x[carried]=8B
- RU3: work 3, input 4 B; ops [op0:matmul x4]; imports x[carried]=4B

**h4_gram_matrix** (h) F=inf G=W/4: L=8, I*=24
- RU0: work 3, input 4 B; ops [op0:matmul x4]; imports x[carried]=4B
- RU1: work 3, input 8 B; ops [op0:matmul x4]; imports x[carried]=8B
- RU2: work 3, input 8 B; ops [op0:matmul x4]; imports x[carried]=8B
- RU3: work 3, input 4 B; ops [op0:matmul x4]; imports x[carried]=4B

**t2_repeated_row_view_B** (t) F=W/4 G=W/4: L=8, I*=20
- RU0: work 2, input 4 B; ops [op0:matmul x4]; imports W[accumulated]=2B, x[token]=2B
- RU1: work 2, input 8 B; ops [op0:matmul x2]; imports W[accumulated]=2B, op0:matmul.out=4B, x[token]=2B
- RU2: work 2, input 8 B; ops [op0:matmul x2]; imports W[accumulated]=2B, op0:matmul.out=4B, x[token]=2B

**t2_repeated_row_view_B** (t) F=W/4 G=W/2: L=8, I*=20
- RU0: work 3, input 8 B; ops [op0:matmul x5]; imports W[accumulated]=4B, x[token]=4B
- RU1: work 3, input 12 B; ops [op0:matmul x3]; imports W[accumulated]=2B, op0:matmul.out=8B, x[token]=2B

**t2_repeated_row_view_B** (t) F=W/2 G=W/4: L=8, I*=20
- RU0: work 2, input 4 B; ops [op0:matmul x4]; imports W[accumulated]=2B, x[token]=2B
- RU1: work 2, input 8 B; ops [op0:matmul x2]; imports W[accumulated]=2B, op0:matmul.out=4B, x[token]=2B
- RU2: work 2, input 8 B; ops [op0:matmul x2]; imports W[accumulated]=2B, op0:matmul.out=4B, x[token]=2B

**t2_repeated_row_view_B** (t) F=inf G=W/4: L=8, I*=20
- RU0: work 2, input 4 B; ops [op0:matmul x4]; imports W[accumulated]=2B, x[token]=2B
- RU1: work 2, input 8 B; ops [op0:matmul x2]; imports W[accumulated]=2B, op0:matmul.out=4B, x[token]=2B
- RU2: work 2, input 8 B; ops [op0:matmul x2]; imports W[accumulated]=2B, op0:matmul.out=4B, x[token]=2B

**t5_rowfull_epilogue_F** (t) F=W/4 G=W/4: L=16, I*=34
- RU0: work 4, input 8 B; ops [op0:matmul x4, op1:rmsnorm x1]; imports W[accumulated]=4B, x[token]=4B
- RU1: work 3, input 8 B; ops [op0:matmul x4]; imports W[accumulated]=4B, x[token]=4B
- RU2: work 4, input 6 B; ops [op1:rmsnorm x5]; imports op0:matmul.out=2B, op1:rmsnorm.out=4B
- RU3: work 4, input 12 B; ops [op1:rmsnorm x4]; imports g[accumulated]=4B, op0:matmul.out=4B, op1:rmsnorm.out=4B

**t5_rowfull_epilogue_F** (t) F=W/2 G=W/4: L=16, I*=34
- RU0: work 4, input 8 B; ops [op0:matmul x4, op1:rmsnorm x1]; imports W[accumulated]=4B, x[token]=4B
- RU1: work 3, input 8 B; ops [op0:matmul x4]; imports W[accumulated]=4B, x[token]=4B
- RU2: work 4, input 6 B; ops [op1:rmsnorm x5]; imports op0:matmul.out=2B, op1:rmsnorm.out=4B
- RU3: work 4, input 12 B; ops [op1:rmsnorm x4]; imports g[accumulated]=4B, op0:matmul.out=4B, op1:rmsnorm.out=4B

**t5_rowfull_epilogue_F** (t) F=inf G=W/4: L=16, I*=34
- RU0: work 4, input 8 B; ops [op0:matmul x4, op1:rmsnorm x1]; imports W[accumulated]=4B, x[token]=4B
- RU1: work 3, input 8 B; ops [op0:matmul x4]; imports W[accumulated]=4B, x[token]=4B
- RU2: work 4, input 6 B; ops [op1:rmsnorm x5]; imports op0:matmul.out=2B, op1:rmsnorm.out=4B
- RU3: work 4, input 12 B; ops [op1:rmsnorm x4]; imports g[accumulated]=4B, op0:matmul.out=4B, op1:rmsnorm.out=4B

**t6_softmax_epilogue_F** (t) F=W/4 G=W/4: L=12, I*=36
- RU0: work 6, input 12 B; ops [op0:matmul x8]; imports W[accumulated]=8B, x[token]=4B
- RU1: work 4, input 4 B; ops [op1:softmax x5]; imports op0:matmul.out=4B
- RU2: work 6, input 8 B; ops [op1:softmax x4]; imports op0:matmul.out=4B, op1:softmax.out=4B
- RU3: work 5, input 12 B; ops [op1:softmax x5]; imports op1:softmax.out=12B

**t6_softmax_epilogue_F** (t) F=W/4 G=W/2: L=12, I*=32
- RU0: work 8, input 12 B; ops [op0:matmul x8, op1:softmax x3]; imports W[accumulated]=8B, x[token]=4B
- RU1: work 8, input 8 B; ops [op1:softmax x6]; imports op0:matmul.out=4B, op1:softmax.out=4B
- RU2: work 5, input 12 B; ops [op1:softmax x5]; imports op1:softmax.out=12B

**t6_softmax_epilogue_F** (t) F=W/4 G=inf: L=12, I*=32
- RU0: work 8, input 12 B; ops [op0:matmul x8, op1:softmax x3]; imports W[accumulated]=8B, x[token]=4B
- RU1: work 8, input 8 B; ops [op1:softmax x6]; imports op0:matmul.out=4B, op1:softmax.out=4B
- RU2: work 5, input 12 B; ops [op1:softmax x5]; imports op1:softmax.out=12B

**t6_softmax_epilogue_F** (t) F=W/2 G=W/4: L=12, I*=36
- RU0: work 6, input 12 B; ops [op0:matmul x8]; imports W[accumulated]=8B, x[token]=4B
- RU1: work 4, input 4 B; ops [op1:softmax x5]; imports op0:matmul.out=4B
- RU2: work 6, input 8 B; ops [op1:softmax x4]; imports op0:matmul.out=4B, op1:softmax.out=4B
- RU3: work 5, input 12 B; ops [op1:softmax x5]; imports op1:softmax.out=12B

**t6_softmax_epilogue_F** (t) F=inf G=W/4: L=12, I*=36
- RU0: work 6, input 12 B; ops [op0:matmul x8]; imports W[accumulated]=8B, x[token]=4B
- RU1: work 4, input 4 B; ops [op1:softmax x5]; imports op0:matmul.out=4B
- RU2: work 6, input 8 B; ops [op1:softmax x4]; imports op0:matmul.out=4B, op1:softmax.out=4B
- RU3: work 5, input 12 B; ops [op1:softmax x5]; imports op1:softmax.out=12B

**t9_scalar_read_by_two_matmuls** (t) F=W/4 G=W/4: L=2, I*=12
- RU0: work 1, input 2 B; ops [op0:matmul x2]; imports p[accumulated]=2B
- RU1: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
- RU2: work 1, input 2 B; ops [op1:matmul x2]; imports p[accumulated]=2B
- RU3: work 1, input 4 B; ops [op1:matmul x1]; imports op1:matmul.out=4B

**t9_scalar_read_by_two_matmuls** (t) F=W/4 G=W/2: L=2, I*=10
- RU0: work 2, input 2 B; ops [op0:matmul x2, op1:matmul x2]; imports p[accumulated]=2B
- RU1: work 2, input 8 B; ops [op0:matmul x1, op1:matmul x1]; imports op0:matmul.out=4B, op1:matmul.out=4B

**t9_scalar_read_by_two_matmuls** (t) F=W/4 G=inf: L=2, I*=10
- RU0: work 2, input 2 B; ops [op0:matmul x2, op1:matmul x2]; imports p[accumulated]=2B
- RU1: work 2, input 8 B; ops [op0:matmul x1, op1:matmul x1]; imports op0:matmul.out=4B, op1:matmul.out=4B

**t9_scalar_read_by_two_matmuls** (t) F=W/2 G=W/4: L=2, I*=12
- RU0: work 1, input 2 B; ops [op0:matmul x2]; imports p[accumulated]=2B
- RU1: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
- RU2: work 1, input 2 B; ops [op1:matmul x2]; imports p[accumulated]=2B
- RU3: work 1, input 4 B; ops [op1:matmul x1]; imports op1:matmul.out=4B

**t9_scalar_read_by_two_matmuls** (t) F=inf G=W/4: L=2, I*=12
- RU0: work 1, input 2 B; ops [op0:matmul x2]; imports p[accumulated]=2B
- RU1: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
- RU2: work 1, input 2 B; ops [op1:matmul x2]; imports p[accumulated]=2B
- RU3: work 1, input 4 B; ops [op1:matmul x1]; imports op1:matmul.out=4B

**t11_reprow_B_deep** (t) F=W/4 G=W/4: L=10, I*=24
- RU0: work 4, input 8 B; ops [op0:matmul x4, op1:add x1]; imports W1[accumulated]=4B, x[carried]=4B
- RU1: work 4, input 4 B; ops [op2:matmul x6]; imports W[accumulated]=2B, op1:add.out=2B
- RU2: work 4, input 4 B; ops [op2:matmul x6]; imports W[accumulated]=2B, op1:add.out=2B
- RU3: work 5, input 8 B; ops [op3:matmul x6]; imports op2:matmul.out=8B

**t11_reprow_B_deep** (t) F=W/2 G=W/4: L=10, I*=24
- RU0: work 4, input 8 B; ops [op0:matmul x4, op1:add x1]; imports W1[accumulated]=4B, x[carried]=4B
- RU1: work 4, input 4 B; ops [op2:matmul x6]; imports W[accumulated]=2B, op1:add.out=2B
- RU2: work 4, input 4 B; ops [op2:matmul x6]; imports W[accumulated]=2B, op1:add.out=2B
- RU3: work 5, input 8 B; ops [op3:matmul x6]; imports op2:matmul.out=8B

**t11_reprow_B_deep** (t) F=inf G=W/4: L=10, I*=24
- RU0: work 4, input 8 B; ops [op0:matmul x4, op1:add x1]; imports W1[accumulated]=4B, x[carried]=4B
- RU1: work 4, input 4 B; ops [op2:matmul x6]; imports W[accumulated]=2B, op1:add.out=2B
- RU2: work 4, input 4 B; ops [op2:matmul x6]; imports W[accumulated]=2B, op1:add.out=2B
- RU3: work 5, input 8 B; ops [op3:matmul x6]; imports op2:matmul.out=8B

**t12_attn_decode_kv_undercount** (t) F=W/4 G=W/4: L=20, I*=44
- RU0: work 7, input 12 B; ops [op0:matmul x8, op1:softmax x1]; imports Kc[carried]=8B, q[token]=4B
- RU1: work 7, input 6 B; ops [op1:softmax x7]; imports op0:matmul.out=2B, op1:softmax.out=4B
- RU2: work 7, input 14 B; ops [op1:softmax x6]; imports op0:matmul.out=2B, op1:softmax.out=12B
- RU3: work 6, input 12 B; ops [op2:matmul x8]; imports Vt[carried]=8B, op1:softmax.out=4B

**t12_attn_decode_kv_undercount** (t) F=W/2 G=W/4: L=20, I*=44
- RU0: work 7, input 12 B; ops [op0:matmul x8, op1:softmax x1]; imports Kc[carried]=8B, q[token]=4B
- RU1: work 7, input 6 B; ops [op1:softmax x7]; imports op0:matmul.out=2B, op1:softmax.out=4B
- RU2: work 7, input 14 B; ops [op1:softmax x6]; imports op0:matmul.out=2B, op1:softmax.out=12B
- RU3: work 6, input 12 B; ops [op2:matmul x8]; imports Vt[carried]=8B, op1:softmax.out=4B

**t12_attn_decode_kv_undercount** (t) F=inf G=W/4: L=20, I*=44
- RU0: work 7, input 12 B; ops [op0:matmul x8, op1:softmax x1]; imports Kc[carried]=8B, q[token]=4B
- RU1: work 7, input 6 B; ops [op1:softmax x7]; imports op0:matmul.out=2B, op1:softmax.out=4B
- RU2: work 7, input 14 B; ops [op1:softmax x6]; imports op0:matmul.out=2B, op1:softmax.out=12B
- RU3: work 6, input 12 B; ops [op2:matmul x8]; imports Vt[carried]=8B, op1:softmax.out=4B

**i1_inference_mlp** (i) F=W/4 G=W/4: L=4, I*=24
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports x[token]=4B
- RU2: work 6, input 8 B; ops [op2:swiglu x1, op3:matmul x4, op4:add x1]; imports op0:matmul.out=2B, op1:matmul.out=2B, op2:swiglu.out=2B, x[token]=2B
- RU3: work 6, input 8 B; ops [op2:swiglu x1, op3:matmul x4, op4:add x1]; imports op0:matmul.out=2B, op1:matmul.out=2B, op2:swiglu.out=2B, x[token]=2B

**i1_inference_mlp** (i) F=W/4 G=W/2: L=4, I*=18
- RU0: work 12, input 4 B; ops [op0:matmul x8, op1:matmul x8]; imports x[token]=4B
- RU1: work 10, input 10 B; ops [op2:swiglu x1, op3:matmul x8, op4:add x2]; imports op0:matmul.out=2B, op1:matmul.out=2B, op2:swiglu.out=2B, x[token]=4B
- RU2: work 2, input 4 B; ops [op2:swiglu x1]; imports op0:matmul.out=2B, op1:matmul.out=2B

**i1_inference_mlp** (i) F=W/4 G=inf: L=4, I*=18
- RU0: work 12, input 4 B; ops [op0:matmul x8, op1:matmul x8]; imports x[token]=4B
- RU1: work 10, input 10 B; ops [op2:swiglu x1, op3:matmul x8, op4:add x2]; imports op0:matmul.out=2B, op1:matmul.out=2B, op2:swiglu.out=2B, x[token]=4B
- RU2: work 2, input 4 B; ops [op2:swiglu x1]; imports op0:matmul.out=2B, op1:matmul.out=2B

**i1_inference_mlp** (i) F=W/2 G=W/4: L=4, I*=24
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports x[token]=4B
- RU2: work 6, input 8 B; ops [op2:swiglu x1, op3:matmul x4, op4:add x1]; imports op0:matmul.out=2B, op1:matmul.out=2B, op2:swiglu.out=2B, x[token]=2B
- RU3: work 6, input 8 B; ops [op2:swiglu x1, op3:matmul x4, op4:add x1]; imports op0:matmul.out=2B, op1:matmul.out=2B, op2:swiglu.out=2B, x[token]=2B

**i1_inference_mlp** (i) F=W/2 G=W/2: L=4, I*=12
- RU0: work 12, input 6 B; ops [op0:matmul x4, op1:matmul x4, op2:swiglu x1, op3:matmul x4, op4:add x1]; imports op2:swiglu.out=2B, x[token]=4B
- RU1: work 12, input 6 B; ops [op0:matmul x4, op1:matmul x4, op2:swiglu x1, op3:matmul x4, op4:add x1]; imports op2:swiglu.out=2B, x[token]=4B

**i1_inference_mlp** (i) F=W/2 G=inf: L=4, I*=10
- RU0: work 16, input 6 B; ops [op0:matmul x4, op1:matmul x4, op2:swiglu x1, op3:matmul x8, op4:add x2]; imports op2:swiglu.out=2B, x[token]=4B
- RU1: work 8, input 4 B; ops [op0:matmul x4, op1:matmul x4, op2:swiglu x1]; imports x[token]=4B

**i1_inference_mlp** (i) F=inf G=W/4: L=4, I*=24
- RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
- RU1: work 6, input 4 B; ops [op1:matmul x8]; imports x[token]=4B
- RU2: work 6, input 8 B; ops [op2:swiglu x1, op3:matmul x4, op4:add x1]; imports op0:matmul.out=2B, op1:matmul.out=2B, op2:swiglu.out=2B, x[token]=2B
- RU3: work 6, input 8 B; ops [op2:swiglu x1, op3:matmul x4, op4:add x1]; imports op0:matmul.out=2B, op1:matmul.out=2B, op2:swiglu.out=2B, x[token]=2B

**i1_inference_mlp** (i) F=inf G=W/2: L=4, I*=12
- RU0: work 12, input 6 B; ops [op0:matmul x4, op1:matmul x4, op2:swiglu x1, op3:matmul x4, op4:add x1]; imports op2:swiglu.out=2B, x[token]=4B
- RU1: work 12, input 6 B; ops [op0:matmul x4, op1:matmul x4, op2:swiglu x1, op3:matmul x4, op4:add x1]; imports op2:swiglu.out=2B, x[token]=4B

**r1_train_linear_sgd** (r) F=W/4 G=W/4: L=20, I*=44
- RU0: work 5, input 12 B; ops [op0:matmul x4, op2:sgdupdate x2]; imports W[accumulated]=4B, op2:sgdupdate.out=4B, x[token]=4B
- RU1: work 5, input 12 B; ops [op0:matmul x4, op2:sgdupdate x2]; imports W[accumulated]=4B, op2:sgdupdate.out=4B, x[token]=4B
- RU2: work 6, input 10 B; ops [op1:matmul x6, op2:sgdupdate x2]; imports dy[token]=2B, lr[seed]=4B, x[token]=4B
- RU3: work 6, input 10 B; ops [op1:matmul x6, op2:sgdupdate x2]; imports dy[token]=2B, lr[seed]=4B, x[token]=4B

**r1_train_linear_sgd** (r) F=W/2 G=W/4: L=20, I*=44
- RU0: work 5, input 12 B; ops [op0:matmul x4, op2:sgdupdate x2]; imports W[accumulated]=4B, op2:sgdupdate.out=4B, x[token]=4B
- RU1: work 5, input 12 B; ops [op0:matmul x4, op2:sgdupdate x2]; imports W[accumulated]=4B, op2:sgdupdate.out=4B, x[token]=4B
- RU2: work 6, input 10 B; ops [op1:matmul x6, op2:sgdupdate x2]; imports dy[token]=2B, lr[seed]=4B, x[token]=4B
- RU3: work 6, input 10 B; ops [op1:matmul x6, op2:sgdupdate x2]; imports dy[token]=2B, lr[seed]=4B, x[token]=4B

**r1_train_linear_sgd** (r) F=inf G=W/4: L=20, I*=44
- RU0: work 5, input 12 B; ops [op0:matmul x4, op2:sgdupdate x2]; imports W[accumulated]=4B, op2:sgdupdate.out=4B, x[token]=4B
- RU1: work 5, input 12 B; ops [op0:matmul x4, op2:sgdupdate x2]; imports W[accumulated]=4B, op2:sgdupdate.out=4B, x[token]=4B
- RU2: work 6, input 10 B; ops [op1:matmul x6, op2:sgdupdate x2]; imports dy[token]=2B, lr[seed]=4B, x[token]=4B
- RU3: work 6, input 10 B; ops [op1:matmul x6, op2:sgdupdate x2]; imports dy[token]=2B, lr[seed]=4B, x[token]=4B

**fuzz7-13** (matmul|view) F=inf G=W/4: L=24, I*=56
- RU0: work 4, input 16 B; ops [op0:matmul x5]; imports p0[token]=8B, p1[carried]=8B
- RU1: work 4, input 16 B; ops [op0:matmul x4]; imports op0:matmul.out=8B, p0[token]=4B, p1[carried]=4B
- RU2: work 4, input 12 B; ops [op0:matmul x6]; imports p0[token]=4B, p1[carried]=8B
- RU3: work 3, input 12 B; ops [op0:matmul x3]; imports op0:matmul.out=4B, p0[token]=4B, p1[carried]=4B

**fuzz7-13** (matmul|view) F=W/2 G=W/4: L=24, I*=56
- RU0: work 4, input 16 B; ops [op0:matmul x5]; imports p0[token]=8B, p1[carried]=8B
- RU1: work 4, input 16 B; ops [op0:matmul x4]; imports op0:matmul.out=8B, p0[token]=4B, p1[carried]=4B
- RU2: work 4, input 12 B; ops [op0:matmul x6]; imports p0[token]=4B, p1[carried]=8B
- RU3: work 3, input 12 B; ops [op0:matmul x3]; imports op0:matmul.out=4B, p0[token]=4B, p1[carried]=4B

**fuzz7-18** (add+matmul+mul+rowscale|shared,chain) F=W/2 G=W/4: L=8, I*=20
- RU0: work 5, input 2 B; ops [op0:rowscale x3, op1:matmul x3]; imports p1[token]=2B
- RU1: work 4, input 8 B; ops [op1:matmul x2, op2:mul x1, op4:add x1]; imports op0:rowscale.out=2B, op1:matmul.out=4B, p2[accumulated]=2B
- RU2: work 4, input 4 B; ops [op2:mul x2, op4:add x2]; imports p2[accumulated]=4B
- RU3: work 4, input 6 B; ops [op3:matmul x5]; imports op0:rowscale.out=6B

**fuzz7-18** (add+matmul+mul+rowscale|shared,chain) F=W/4 G=W/4: L=8, I*=20
- RU0: work 5, input 2 B; ops [op0:rowscale x3, op1:matmul x3]; imports p1[token]=2B
- RU1: work 4, input 8 B; ops [op1:matmul x2, op2:mul x1, op4:add x1]; imports op0:rowscale.out=2B, op1:matmul.out=4B, p2[accumulated]=2B
- RU2: work 4, input 4 B; ops [op2:mul x2, op4:add x2]; imports p2[accumulated]=4B
- RU3: work 4, input 6 B; ops [op3:matmul x5]; imports op0:rowscale.out=6B

**fuzz7-18** (add+matmul+mul+rowscale|shared,chain) F=inf G=W/4: L=8, I*=20
- RU0: work 5, input 2 B; ops [op0:rowscale x3, op1:matmul x3]; imports p1[token]=2B
- RU1: work 4, input 8 B; ops [op1:matmul x2, op2:mul x1, op4:add x1]; imports op0:rowscale.out=2B, op1:matmul.out=4B, p2[accumulated]=2B
- RU2: work 4, input 4 B; ops [op2:mul x2, op4:add x2]; imports p2[accumulated]=4B
- RU3: work 4, input 6 B; ops [op3:matmul x5]; imports op0:rowscale.out=6B

**fuzz7-23** (add+colsum+mul|shared,view,chain) F=inf G=W/2: L=2, I*=8
- RU0: work 5, input 2 B; ops [op0:mul x1, op1:colsum x5]; imports p0[accumulated]=2B
- RU1: work 4, input 6 B; ops [op1:colsum x3, op2:add x1]; imports op0:mul.out=2B, op1:colsum.out=4B

**fuzz7-23** (add+colsum+mul|shared,view,chain) F=inf G=W/4: L=2, I*=14
- RU0: work 3, input 2 B; ops [op0:mul x1, op1:colsum x3]; imports p0[accumulated]=2B
- RU1: work 3, input 6 B; ops [op1:colsum x2, op2:add x1]; imports op0:mul.out=2B, op1:colsum.out=4B
- RU2: work 3, input 6 B; ops [op1:colsum x3]; imports op0:mul.out=2B, op1:colsum.out=4B

**fuzz7-23** (add+colsum+mul|shared,view,chain) F=W/4 G=W/4: L=2, I*=14
- RU0: work 3, input 2 B; ops [op0:mul x1, op1:colsum x3]; imports p0[accumulated]=2B
- RU1: work 3, input 6 B; ops [op1:colsum x2, op2:add x1]; imports op0:mul.out=2B, op1:colsum.out=4B
- RU2: work 3, input 6 B; ops [op1:colsum x3]; imports op0:mul.out=2B, op1:colsum.out=4B

**fuzz7-23** (add+colsum+mul|shared,view,chain) F=W/4 G=inf: L=2, I*=14
- RU0: work 6, input 2 B; ops [op0:mul x1, op1:colsum x5, op2:add x1]; imports p0[accumulated]=2B
- RU1: work 3, input 12 B; ops [op1:colsum x3]; imports op1:colsum.out=12B

**fuzz7-25** (embed+matmul|shared,chain) F=inf G=W/2: L=6, I*=14
- RU0: work 12, input 6 B; ops [op0:matmul x15, op1:embed x1]; imports p0[token]=6B
- RU1: work 12, input 8 B; ops [op0:matmul x12, op1:embed x2]; imports op0:matmul.out=4B, p0[token]=4B

**fuzz7-25** (embed+matmul|shared,chain) F=W/4 G=W/4: L=6, I*=24
- RU0: work 6, input 4 B; ops [op0:matmul x9]; imports p0[token]=4B
- RU1: work 6, input 8 B; ops [op0:matmul x6, op1:embed x1]; imports op0:matmul.out=4B, p0[token]=4B
- RU2: work 6, input 6 B; ops [op0:matmul x6, op1:embed x1]; imports op0:matmul.out=2B, p0[token]=4B
- RU3: work 6, input 6 B; ops [op0:matmul x6, op1:embed x1]; imports op0:matmul.out=2B, p0[token]=4B

**fuzz7-25** (embed+matmul|shared,chain) F=W/2 G=W/4: L=6, I*=24
- RU0: work 6, input 4 B; ops [op0:matmul x9]; imports p0[token]=4B
- RU1: work 6, input 8 B; ops [op0:matmul x6, op1:embed x1]; imports op0:matmul.out=4B, p0[token]=4B
- RU2: work 6, input 6 B; ops [op0:matmul x6, op1:embed x1]; imports op0:matmul.out=2B, p0[token]=4B
- RU3: work 6, input 6 B; ops [op0:matmul x6, op1:embed x1]; imports op0:matmul.out=2B, p0[token]=4B

**fuzz7-29** (gain+matmul+mul|shared,view,chain) F=W/4 G=W/2: L=4, I*=14
- RU0: work 4, input 0 B; ops [op0:mul x2, op1:matmul x3]; imports nothing charged
- RU1: work 4, input 6 B; ops [op0:mul x1, op1:matmul x2, op2:gain x1]; imports op1:matmul.out=4B, p2[token]=2B
- RU2: work 9, input 8 B; ops [op1:matmul x5, op2:gain x1, op3:matmul x5]; imports op0:mul.out=6B, p2[token]=2B

**fuzz7-29** (gain+matmul+mul|shared,view,chain) F=inf G=W/4: L=4, I*=20
- RU0: work 5, input 2 B; ops [op0:mul x2, op1:matmul x4]; imports op0:mul.out=2B
- RU1: work 5, input 4 B; ops [op0:mul x1, op3:matmul x5]; imports op0:mul.out=4B
- RU2: work 2, input 6 B; ops [op1:matmul x1, op2:gain x1]; imports op1:matmul.out=4B, p2[token]=2B
- RU3: work 5, input 8 B; ops [op1:matmul x5, op2:gain x1]; imports op0:mul.out=6B, p2[token]=2B

**fuzz7-29** (gain+matmul+mul|shared,view,chain) F=W/4 G=W/4: L=4, I*=20
- RU0: work 5, input 2 B; ops [op0:mul x2, op1:matmul x4]; imports op0:mul.out=2B
- RU1: work 5, input 4 B; ops [op0:mul x1, op3:matmul x5]; imports op0:mul.out=4B
- RU2: work 2, input 6 B; ops [op1:matmul x1, op2:gain x1]; imports op1:matmul.out=4B, p2[token]=2B
- RU3: work 5, input 8 B; ops [op1:matmul x5, op2:gain x1]; imports op0:mul.out=6B, p2[token]=2B

**fuzz7-33** (colsum) F=W/4 G=inf: L=4, I*=12
- RU0: work 4, input 4 B; ops [op0:colsum x6]; imports p0[accumulated]=4B
- RU1: work 2, input 8 B; ops [op0:colsum x2]; imports op0:colsum.out=8B

**fuzz7-33** (colsum) F=W/4 G=W/4: L=4, I*=12
- RU0: work 2, input 2 B; ops [op0:colsum x3]; imports p0[accumulated]=2B
- RU1: work 2, input 6 B; ops [op0:colsum x2]; imports op0:colsum.out=4B, p0[accumulated]=2B
- RU2: work 2, input 4 B; ops [op0:colsum x3]; imports op0:colsum.out=4B

**fuzz7-34** (embed) F=W/2 G=W/4: L=4, I*=12
- RU0: work 1, input 4 B; ops [op0:embed x1]; imports p1[token]=4B
- RU1: work 1, input 4 B; ops [op0:embed x1]; imports p1[token]=4B
- RU2: work 1, input 4 B; ops [op0:embed x1]; imports p1[token]=4B

**fuzz7-37** (embed+matmul|shared,view) F=inf G=W/4: L=6, I*=30
- RU0: work 7, input 6 B; ops [op0:matmul x9]; imports p0[token]=6B
- RU1: work 6, input 12 B; ops [op0:matmul x2, op1:embed x2]; imports op0:matmul.out=8B, p2[token]=4B
- RU2: work 7, input 6 B; ops [op0:matmul x9]; imports p0[token]=6B
- RU3: work 6, input 6 B; ops [op1:embed x2, op2:matmul x3]; imports p2[token]=4B, p3[accumulated]=2B

**fuzz7-37** (embed+matmul|shared,view) F=W/4 G=W/2: L=6, I*=18
- RU0: work 12, input 6 B; ops [op0:matmul x15]; imports p0[token]=6B
- RU1: work 12, input 10 B; ops [op0:matmul x5, op1:embed x4]; imports p0[token]=6B, p2[token]=4B
- RU2: work 2, input 2 B; ops [op2:matmul x3]; imports p3[accumulated]=2B

**fuzz7-37** (embed+matmul|shared,view) F=W/2 G=W/4: L=6, I*=30
- RU0: work 7, input 6 B; ops [op0:matmul x9]; imports p0[token]=6B
- RU1: work 6, input 12 B; ops [op0:matmul x2, op1:embed x2]; imports op0:matmul.out=8B, p2[token]=4B
- RU2: work 7, input 6 B; ops [op0:matmul x9]; imports p0[token]=6B
- RU3: work 6, input 6 B; ops [op1:embed x2, op2:matmul x3]; imports p2[token]=4B, p3[accumulated]=2B

**fuzz7-41** (swiglu|shared,view) F=W/2 G=inf: L=0, I*=6
- RU0: work 6, input 6 B; ops [op0:swiglu x3]; imports p0[token]=6B

**fuzz7-41** (swiglu|shared,view) F=W/2 G=W/2: L=0, I*=6
- RU0: work 2, input 2 B; ops [op0:swiglu x1]; imports p0[token]=2B
- RU1: work 2, input 2 B; ops [op0:swiglu x1]; imports p0[token]=2B
- RU2: work 2, input 2 B; ops [op0:swiglu x1]; imports p0[token]=2B

**fuzz7-41** (swiglu|shared,view) F=inf G=W/4: L=0, I*=6
- RU0: work 2, input 2 B; ops [op0:swiglu x1]; imports p0[token]=2B
- RU1: work 2, input 2 B; ops [op0:swiglu x1]; imports p0[token]=2B
- RU2: work 2, input 2 B; ops [op0:swiglu x1]; imports p0[token]=2B

**fuzz7-41** (swiglu|shared,view) F=inf G=W/2: L=0, I*=6
- RU0: work 2, input 2 B; ops [op0:swiglu x1]; imports p0[token]=2B
- RU1: work 2, input 2 B; ops [op0:swiglu x1]; imports p0[token]=2B
- RU2: work 2, input 2 B; ops [op0:swiglu x1]; imports p0[token]=2B

**fuzz7-49** (gain+matmul|shared) F=W/4 G=W/4: L=24, I*=52
- RU0: work 6, input 12 B; ops [op0:matmul x8]; imports p0[accumulated]=4B, p1[token]=8B
- RU1: work 6, input 12 B; ops [op0:matmul x8]; imports p0[accumulated]=4B, p1[token]=8B
- RU2: work 6, input 12 B; ops [op0:matmul x8]; imports p0[accumulated]=4B, p1[token]=8B
- RU3: work 6, input 16 B; ops [op1:gain x6]; imports p0[accumulated]=12B, p2[token]=4B

**fuzz7-51** (mul+rowscale|shared,view,chain) F=W/2 G=W/4: L=14, I*=32
- RU0: work 6, input 8 B; ops [op0:rowscale x3, op1:mul x3]; imports p0[accumulated]=2B, p1[accumulated]=6B
- RU1: work 6, input 8 B; ops [op0:rowscale x3, op1:mul x3]; imports p0[accumulated]=2B, p1[accumulated]=6B
- RU2: work 6, input 8 B; ops [op0:rowscale x3, op1:mul x3]; imports p0[accumulated]=2B, p1[accumulated]=6B
- RU3: work 6, input 8 B; ops [op0:rowscale x3, op1:mul x3]; imports p0[accumulated]=2B, p1[accumulated]=6B

**fuzz7-51** (mul+rowscale|shared,view,chain) F=W/4 G=W/4: L=14, I*=32
- RU0: work 6, input 8 B; ops [op0:rowscale x3, op1:mul x3]; imports p0[accumulated]=2B, p1[accumulated]=6B
- RU1: work 6, input 8 B; ops [op0:rowscale x3, op1:mul x3]; imports p0[accumulated]=2B, p1[accumulated]=6B
- RU2: work 6, input 8 B; ops [op0:rowscale x3, op1:mul x3]; imports p0[accumulated]=2B, p1[accumulated]=6B
- RU3: work 6, input 8 B; ops [op0:rowscale x3, op1:mul x3]; imports p0[accumulated]=2B, p1[accumulated]=6B

**fuzz7-56** (matmul|shared) F=W/4 G=W/4: L=6, I*=18
- RU0: work 4, input 4 B; ops [op0:matmul x6]; imports p0[token]=4B
- RU1: work 4, input 4 B; ops [op0:matmul x6]; imports p0[token]=4B
- RU2: work 4, input 4 B; ops [op0:matmul x6]; imports p0[token]=4B
- RU3: work 4, input 4 B; ops [op0:matmul x6]; imports p0[token]=4B
- RU4: work 2, input 2 B; ops [op0:matmul x3]; imports p0[token]=2B

**fuzz7-60** (matmul|shared) F=W/2 G=W/4: L=18, I*=[40, 58]
- RU0: work 9, input 12 B; ops [op0:matmul x12]; imports p0[accumulated]=12B
- RU1: work 2, input 6 B; ops [op0:matmul x2]; imports op0:matmul.out=4B, p0[accumulated]=2B
- RU2: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
- RU3: work 8, input 12 B; ops [op0:matmul x10]; imports p0[accumulated]=12B
- RU4: work 8, input 12 B; ops [op0:matmul x10]; imports p0[accumulated]=12B
- RU5: work 8, input 12 B; ops [op0:matmul x10]; imports p0[accumulated]=12B

**fuzz7-60** (matmul|shared) F=inf G=W/4: L=18, I*=[40, 58]
- RU0: work 9, input 12 B; ops [op0:matmul x12]; imports p0[accumulated]=12B
- RU1: work 2, input 6 B; ops [op0:matmul x2]; imports op0:matmul.out=4B, p0[accumulated]=2B
- RU2: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
- RU3: work 8, input 12 B; ops [op0:matmul x10]; imports p0[accumulated]=12B
- RU4: work 8, input 12 B; ops [op0:matmul x10]; imports p0[accumulated]=12B
- RU5: work 8, input 12 B; ops [op0:matmul x10]; imports p0[accumulated]=12B

**fuzz7-61** (embed+matmul+mul+rowscale|shared,view,chain) F=W/2 G=W/4: L=16, I*=34
- RU0: work 5, input 14 B; ops [op0:embed x2, op1:mul x1]; imports p0[token]=8B, p1[token]=6B
- RU1: work 5, input 6 B; ops [op1:mul x1, op2:rowscale x2, op4:matmul x3]; imports p1[token]=4B, p2[accumulated]=2B
- RU2: work 4, input 8 B; ops [op1:mul x1, op2:rowscale x1, op4:matmul x2]; imports op4:matmul.out=4B, p1[token]=2B, p2[accumulated]=2B
- RU3: work 4, input 6 B; ops [op3:matmul x5]; imports p1[token]=6B

**fuzz7-61** (embed+matmul+mul+rowscale|shared,view,chain) F=W/4 G=W/4: L=16, I*=34
- RU0: work 5, input 14 B; ops [op0:embed x2, op1:mul x1]; imports p0[token]=8B, p1[token]=6B
- RU1: work 5, input 6 B; ops [op1:mul x1, op2:rowscale x2, op4:matmul x3]; imports p1[token]=4B, p2[accumulated]=2B
- RU2: work 4, input 8 B; ops [op1:mul x1, op2:rowscale x1, op4:matmul x2]; imports op4:matmul.out=4B, p1[token]=2B, p2[accumulated]=2B
- RU3: work 4, input 6 B; ops [op3:matmul x5]; imports p1[token]=6B

**fuzz7-68** (matmul|view) F=W/2 G=W/4: L=16, I*=40
- RU0: work 3, input 12 B; ops [op0:matmul x4]; imports p0[accumulated]=6B, p1[seed]=6B
- RU1: work 2, input 8 B; ops [op0:matmul x2]; imports op0:matmul.out=4B, p0[accumulated]=2B, p1[seed]=2B
- RU2: work 3, input 12 B; ops [op0:matmul x4]; imports p0[accumulated]=6B, p1[seed]=6B
- RU3: work 2, input 8 B; ops [op0:matmul x2]; imports op0:matmul.out=4B, p0[accumulated]=2B, p1[seed]=2B

**fuzz7-68** (matmul|view) F=W/4 G=W/2: L=16, I*=34
- RU0: work 5, input 16 B; ops [op0:matmul x7]; imports p0[accumulated]=10B, p1[seed]=6B
- RU1: work 5, input 18 B; ops [op0:matmul x5]; imports op0:matmul.out=8B, p0[accumulated]=6B, p1[seed]=4B

**fuzz7-69** (add+embed+gain+matmul|shared,view,chain) F=W/2 G=W/4: L=12, I*=30
- RU0: work 5, input 8 B; ops [op0:matmul x6, op2:gain x1]; imports op1:add.out=2B, p0[token]=2B, p1[accumulated]=2B, p3[accumulated]=2B
- RU1: work 5, input 4 B; ops [op0:matmul x6, op3:embed x1]; imports p0[token]=2B, p1[accumulated]=2B
- RU2: work 5, input 6 B; ops [op0:matmul x6, op1:add x1]; imports p0[token]=2B, p1[accumulated]=2B, p2[accumulated]=2B
- RU3: work 5, input 12 B; ops [op3:embed x5]; imports op0:matmul.out=12B

**fuzz7-73** (matmul_tt|shared) F=W/2 G=W/4: L=6, I*=18
- RU0: work 4, input 4 B; ops [op0:matmul x6]; imports p0[carried]=4B
- RU1: work 4, input 4 B; ops [op0:matmul x6]; imports p0[carried]=4B
- RU2: work 4, input 4 B; ops [op0:matmul x6]; imports p0[carried]=4B
- RU3: work 4, input 4 B; ops [op0:matmul x6]; imports p0[carried]=4B
- RU4: work 2, input 2 B; ops [op0:matmul x3]; imports p0[carried]=2B

**fuzz7-73** (matmul_tt|shared) F=W/4 G=W/4: L=6, I*=18
- RU0: work 4, input 4 B; ops [op0:matmul x6]; imports p0[carried]=4B
- RU1: work 4, input 4 B; ops [op0:matmul x6]; imports p0[carried]=4B
- RU2: work 4, input 4 B; ops [op0:matmul x6]; imports p0[carried]=4B
- RU3: work 4, input 4 B; ops [op0:matmul x6]; imports p0[carried]=4B
- RU4: work 2, input 2 B; ops [op0:matmul x3]; imports p0[carried]=2B

**fuzz7-75** (matmul|shared,view) F=W/2 G=W/4: L=8, I*=32
- RU0: work 5, input 4 B; ops [op0:matmul x7]; imports p0[accumulated]=4B
- RU1: work 5, input 12 B; ops [op0:matmul x6]; imports op0:matmul.out=8B, p0[accumulated]=4B
- RU2: work 5, input 8 B; ops [op0:matmul x7]; imports p0[accumulated]=8B
- RU3: work 3, input 8 B; ops [op0:matmul x4]; imports p0[accumulated]=8B

**fuzz7-75** (matmul|shared,view) F=inf G=W/4: L=8, I*=32
- RU0: work 5, input 4 B; ops [op0:matmul x7]; imports p0[accumulated]=4B
- RU1: work 5, input 12 B; ops [op0:matmul x6]; imports op0:matmul.out=8B, p0[accumulated]=4B
- RU2: work 5, input 8 B; ops [op0:matmul x7]; imports p0[accumulated]=8B
- RU3: work 3, input 8 B; ops [op0:matmul x4]; imports p0[accumulated]=8B

**fuzz7-79** (gain+matmul_tt|shared) F=W/4 G=W/4: L=8, I*=20
- RU0: work 5, input 8 B; ops [op0:matmul x5, op1:gain x1]; imports op0:matmul.out=4B, p0[accumulated]=4B
- RU1: work 5, input 8 B; ops [op0:matmul x7]; imports p0[accumulated]=8B
- RU2: work 5, input 4 B; ops [op0:matmul x4, op1:gain x2]; imports p0[accumulated]=4B
- RU3: work 5, input 0 B; ops [op1:gain x5]; imports nothing charged

**fuzz7-79** (gain+matmul_tt|shared) F=inf G=W/4: L=8, I*=20
- RU0: work 5, input 8 B; ops [op0:matmul x5, op1:gain x1]; imports op0:matmul.out=4B, p0[accumulated]=4B
- RU1: work 5, input 8 B; ops [op0:matmul x7]; imports p0[accumulated]=8B
- RU2: work 5, input 4 B; ops [op0:matmul x4, op1:gain x2]; imports p0[accumulated]=4B
- RU3: work 5, input 0 B; ops [op1:gain x5]; imports nothing charged

**fuzz7-85** (gain+matmul|chain) F=inf G=W/2: L=10, I*=22
- RU0: work 9, input 6 B; ops [op0:matmul x12]; imports p1[token]=6B
- RU1: work 9, input 16 B; ops [op0:matmul x6, op1:gain x3]; imports op0:matmul.out=12B, p1[token]=2B, p2[accumulated]=2B

**fuzz7-85** (gain+matmul|chain) F=W/4 G=W/4: L=10, I*=32
- RU0: work 5, input 8 B; ops [op0:matmul x6]; imports p1[token]=8B
- RU1: work 5, input 8 B; ops [op0:matmul x6]; imports p1[token]=8B
- RU2: work 5, input 8 B; ops [op0:matmul x6]; imports p1[token]=8B
- RU3: work 3, input 8 B; ops [op1:gain x3]; imports op0:matmul.out=6B, p2[accumulated]=2B

**fuzz7-100** (softmax) F=W/4 G=W/2: L=8, I*=38
- RU0: work 6, input 6 B; ops [op0:softmax x7]; imports p0[accumulated]=6B
- RU1: work 14, input 12 B; ops [op0:softmax x10]; imports op0:softmax.out=4B, p0[accumulated]=8B
- RU2: work 9, input 20 B; ops [op0:softmax x9]; imports op0:softmax.out=20B

**fuzz7-100** (softmax) F=W/4 G=inf: L=8, I*=36
- RU0: work 4, input 4 B; ops [op0:softmax x5]; imports p0[accumulated]=4B
- RU1: work 16, input 12 B; ops [op0:softmax x12]; imports op0:softmax.out=4B, p0[accumulated]=8B
- RU2: work 9, input 20 B; ops [op0:softmax x9]; imports op0:softmax.out=20B

**fuzz7-100** (softmax) F=W/2 G=W/2: L=8, I*=28
- RU0: work 15, input 8 B; ops [op0:softmax x14]; imports p0[accumulated]=8B
- RU1: work 14, input 20 B; ops [op0:softmax x12]; imports op0:softmax.out=16B, p0[accumulated]=4B

**fuzz7-100** (softmax) F=W/4 G=W/4: L=8, I*=46
- RU0: work 6, input 6 B; ops [op0:softmax x7]; imports p0[accumulated]=6B
- RU1: work 8, input 8 B; ops [op0:softmax x6]; imports op0:softmax.out=4B, p0[accumulated]=4B
- RU2: work 7, input 12 B; ops [op0:softmax x5]; imports op0:softmax.out=8B, p0[accumulated]=4B
- RU3: work 8, input 20 B; ops [op0:softmax x8]; imports op0:softmax.out=20B

**fuzz7-106** (colsum+gain|view) F=inf G=W/2: L=8, I*=18
- RU0: work 6, input 12 B; ops [op0:gain x3, op1:colsum x3]; imports op1:colsum.out=4B, p0[accumulated]=6B, p2[token]=2B
- RU1: work 6, input 6 B; ops [op1:colsum x7]; imports p2[token]=6B

**fuzz7-106** (colsum+gain|view) F=inf G=W/4: L=8, I*=26
- RU0: work 3, input 6 B; ops [op0:gain x3]; imports p0[accumulated]=6B
- RU1: work 3, input 4 B; ops [op1:colsum x4]; imports p2[token]=4B
- RU2: work 3, input 10 B; ops [op1:colsum x3]; imports op1:colsum.out=8B, p2[token]=2B
- RU3: work 3, input 6 B; ops [op1:colsum x3]; imports op1:colsum.out=4B, p2[token]=2B

**fuzz7-106** (colsum+gain|view) F=W/4 G=W/4: L=8, I*=26
- RU0: work 3, input 6 B; ops [op0:gain x3]; imports p0[accumulated]=6B
- RU1: work 3, input 4 B; ops [op1:colsum x4]; imports p2[token]=4B
- RU2: work 3, input 10 B; ops [op1:colsum x3]; imports op1:colsum.out=8B, p2[token]=2B
- RU3: work 3, input 6 B; ops [op1:colsum x3]; imports op1:colsum.out=4B, p2[token]=2B

**fuzz7-111** (colsum|view) F=W/4 G=inf: L=0, I*=16
- RU0: work 4, input 4 B; ops [op0:colsum x6]; imports p0[carried]=4B
- RU1: work 6, input 12 B; ops [op0:colsum x6]; imports op0:colsum.out=8B, p0[carried]=4B

**fuzz7-111** (colsum|view) F=W/4 G=W/2: L=0, I*=16
- RU0: work 4, input 4 B; ops [op0:colsum x6]; imports p0[carried]=4B
- RU1: work 3, input 6 B; ops [op0:colsum x3]; imports op0:colsum.out=4B, p0[carried]=2B
- RU2: work 3, input 6 B; ops [op0:colsum x3]; imports op0:colsum.out=4B, p0[carried]=2B

**fuzz7-111** (colsum|view) F=W/2 G=W/4: L=0, I*=16
- RU0: work 2, input 2 B; ops [op0:colsum x3]; imports p0[carried]=2B
- RU1: work 3, input 6 B; ops [op0:colsum x3]; imports op0:colsum.out=4B, p0[carried]=2B
- RU2: work 2, input 2 B; ops [op0:colsum x3]; imports p0[carried]=2B
- RU3: work 3, input 6 B; ops [op0:colsum x3]; imports op0:colsum.out=4B, p0[carried]=2B

**fuzz7-111** (colsum|view) F=W/4 G=W/4: L=0, I*=16
- RU0: work 2, input 2 B; ops [op0:colsum x3]; imports p0[carried]=2B
- RU1: work 3, input 6 B; ops [op0:colsum x3]; imports op0:colsum.out=4B, p0[carried]=2B
- RU2: work 2, input 2 B; ops [op0:colsum x3]; imports p0[carried]=2B
- RU3: work 3, input 6 B; ops [op0:colsum x3]; imports op0:colsum.out=4B, p0[carried]=2B

**fuzz7-112** (matmul|shared) F=inf G=W/4: L=16, I*=44
- RU0: work 5, input 8 B; ops [op0:matmul x6]; imports p0[accumulated]=8B
- RU1: work 5, input 12 B; ops [op0:matmul x7]; imports p0[accumulated]=12B
- RU2: work 5, input 16 B; ops [op0:matmul x5]; imports op0:matmul.out=8B, p0[accumulated]=8B
- RU3: work 5, input 8 B; ops [op0:matmul x6]; imports p0[accumulated]=8B

**fuzz7-112** (matmul|shared) F=W/4 G=W/4: L=16, I*=44
- RU0: work 5, input 8 B; ops [op0:matmul x6]; imports p0[accumulated]=8B
- RU1: work 5, input 12 B; ops [op0:matmul x7]; imports p0[accumulated]=12B
- RU2: work 5, input 16 B; ops [op0:matmul x5]; imports op0:matmul.out=8B, p0[accumulated]=8B
- RU3: work 5, input 8 B; ops [op0:matmul x6]; imports p0[accumulated]=8B

**fuzz7-121** (matmul|shared) F=inf G=W/4: L=16, I*=44
- RU0: work 5, input 8 B; ops [op0:matmul x6]; imports p0[accumulated]=8B
- RU1: work 5, input 12 B; ops [op0:matmul x7]; imports p0[accumulated]=12B
- RU2: work 5, input 16 B; ops [op0:matmul x5]; imports op0:matmul.out=8B, p0[accumulated]=8B
- RU3: work 5, input 8 B; ops [op0:matmul x6]; imports p0[accumulated]=8B

**fuzz7-123** (embed) F=W/2 G=W/4: L=32, I*=72
- RU0: work 2, input 12 B; ops [op0:embed x1]; imports p0[token]=4B, p1[accumulated]=8B
- RU1: work 2, input 12 B; ops [op0:embed x1]; imports p0[token]=4B, p1[accumulated]=8B
- RU2: work 2, input 12 B; ops [op0:embed x1]; imports p0[token]=4B, p1[accumulated]=8B
- RU3: work 2, input 12 B; ops [op0:embed x1]; imports p0[token]=4B, p1[accumulated]=8B
- RU4: work 2, input 12 B; ops [op0:embed x1]; imports p0[token]=4B, p1[accumulated]=8B
- RU5: work 2, input 12 B; ops [op0:embed x1]; imports p0[token]=4B, p1[accumulated]=8B

**fuzz7-128** (colsum+softmax|shared) F=W/2 G=W/4: L=4, I*=24
- RU0: work 6, input 2 B; ops [op0:softmax x6]; imports p0[token]=2B
- RU1: work 4, input 10 B; ops [op0:softmax x4]; imports op0:softmax.out=8B, p0[token]=2B
- RU2: work 6, input 8 B; ops [op0:softmax x6]; imports op0:softmax.out=8B
- RU3: work 5, input 4 B; ops [op1:colsum x6]; imports p0[token]=4B

**fuzz7-128** (colsum+softmax|shared) F=W/4 G=W/2: L=4, I*=20
- RU0: work 6, input 8 B; ops [op0:softmax x6]; imports op0:softmax.out=8B
- RU1: work 9, input 4 B; ops [op0:softmax x4, op1:colsum x6]; imports p0[token]=4B
- RU2: work 6, input 8 B; ops [op0:softmax x6]; imports op0:softmax.out=8B

**fuzz7-128** (colsum+softmax|shared) F=W/4 G=W/4: L=4, I*=24
- RU0: work 6, input 2 B; ops [op0:softmax x6]; imports p0[token]=2B
- RU1: work 4, input 10 B; ops [op0:softmax x4]; imports op0:softmax.out=8B, p0[token]=2B
- RU2: work 6, input 8 B; ops [op0:softmax x6]; imports op0:softmax.out=8B
- RU3: work 5, input 4 B; ops [op1:colsum x6]; imports p0[token]=4B

**fuzz7-135** (gain+matmul+mul|shared,chain) F=inf G=W/4: L=6, I*=16
- RU0: work 4, input 6 B; ops [op0:gain x2, op1:matmul x3]; imports p0[accumulated]=4B, p1[token]=2B
- RU1: work 4, input 4 B; ops [op1:matmul x3, op2:mul x2]; imports op0:gain.out=2B, op1:matmul.out=2B
- RU2: work 3, input 4 B; ops [op1:matmul x3, op2:mul x1]; imports op0:gain.out=4B
- RU3: work 3, input 2 B; ops [op1:matmul x3, op2:mul x1]; imports op0:gain.out=2B

**fuzz7-135** (gain+matmul+mul|shared,chain) F=W/4 G=W/4: L=6, I*=16
- RU0: work 4, input 6 B; ops [op0:gain x2, op1:matmul x3]; imports p0[accumulated]=4B, p1[token]=2B
- RU1: work 4, input 4 B; ops [op1:matmul x3, op2:mul x2]; imports op0:gain.out=2B, op1:matmul.out=2B
- RU2: work 3, input 4 B; ops [op1:matmul x3, op2:mul x1]; imports op0:gain.out=4B
- RU3: work 3, input 2 B; ops [op1:matmul x3, op2:mul x1]; imports op0:gain.out=2B

**fuzz7-141** (embed+matmul+rowscale|shared) F=inf G=W/4: L=14, I*=34
- RU0: work 3, input 14 B; ops [op0:embed x1, op1:rowscale x1]; imports p0[token]=4B, p1[accumulated]=8B, p2[token]=2B
- RU1: work 3, input 8 B; ops [op1:rowscale x3]; imports p1[accumulated]=6B, p2[token]=2B
- RU2: work 3, input 6 B; ops [op2:matmul x4]; imports p1[accumulated]=6B
- RU3: work 2, input 6 B; ops [op2:matmul x2]; imports op2:matmul.out=4B, p1[accumulated]=2B

**fuzz7-141** (embed+matmul+rowscale|shared) F=W/2 G=W/4: L=14, I*=34
- RU0: work 3, input 14 B; ops [op0:embed x1, op1:rowscale x1]; imports p0[token]=4B, p1[accumulated]=8B, p2[token]=2B
- RU1: work 3, input 8 B; ops [op1:rowscale x3]; imports p1[accumulated]=6B, p2[token]=2B
- RU2: work 3, input 6 B; ops [op2:matmul x4]; imports p1[accumulated]=6B
- RU3: work 2, input 6 B; ops [op2:matmul x2]; imports op2:matmul.out=4B, p1[accumulated]=2B

**fuzz7-151** (matmul+rmsnorm|view,chain) F=W/2 G=W/4: L=24, I*=52
- RU0: work 6, input 6 B; ops [op0:rmsnorm x7]; imports p0[carried]=6B
- RU1: work 7, input 12 B; ops [op0:rmsnorm x7]; imports op0:rmsnorm.out=4B, p0[carried]=8B
- RU2: work 7, input 16 B; ops [op0:rmsnorm x3, op1:matmul x6]; imports op0:rmsnorm.out=6B, p1[carried]=6B, p2[token]=4B
- RU3: work 7, input 18 B; ops [op0:rmsnorm x1, op1:matmul x6]; imports op0:rmsnorm.out=4B, op1:matmul.out=8B, p1[carried]=2B, p2[token]=4B

**fuzz7-151** (matmul+rmsnorm|view,chain) F=W/4 G=W/4: L=24, I*=52
- RU0: work 6, input 6 B; ops [op0:rmsnorm x7]; imports p0[carried]=6B
- RU1: work 7, input 12 B; ops [op0:rmsnorm x7]; imports op0:rmsnorm.out=4B, p0[carried]=8B
- RU2: work 7, input 16 B; ops [op0:rmsnorm x3, op1:matmul x6]; imports op0:rmsnorm.out=6B, p1[carried]=6B, p2[token]=4B
- RU3: work 7, input 18 B; ops [op0:rmsnorm x1, op1:matmul x6]; imports op0:rmsnorm.out=4B, op1:matmul.out=8B, p1[carried]=2B, p2[token]=4B

**fuzz7-151** (matmul+rmsnorm|view,chain) F=inf G=W/4: L=24, I*=52
- RU0: work 6, input 6 B; ops [op0:rmsnorm x7]; imports p0[carried]=6B
- RU1: work 7, input 12 B; ops [op0:rmsnorm x7]; imports op0:rmsnorm.out=4B, p0[carried]=8B
- RU2: work 7, input 16 B; ops [op0:rmsnorm x3, op1:matmul x6]; imports op0:rmsnorm.out=6B, p1[carried]=6B, p2[token]=4B
- RU3: work 7, input 18 B; ops [op0:rmsnorm x1, op1:matmul x6]; imports op0:rmsnorm.out=4B, op1:matmul.out=8B, p1[carried]=2B, p2[token]=4B

**fuzz7-152** (mul|view) F=W/2 G=W/4: L=0, I*=4
- RU0: work 1, input 2 B; ops [op0:mul x1]; imports p0[accumulated]=2B
- RU1: work 1, input 2 B; ops [op0:mul x1]; imports p0[accumulated]=2B

**fuzz7-152** (mul|view) F=inf G=W/2: L=0, I*=4
- RU0: work 1, input 2 B; ops [op0:mul x1]; imports p0[accumulated]=2B
- RU1: work 1, input 2 B; ops [op0:mul x1]; imports p0[accumulated]=2B

**fuzz7-152** (mul|view) F=inf G=W/4: L=0, I*=4
- RU0: work 1, input 2 B; ops [op0:mul x1]; imports p0[accumulated]=2B
- RU1: work 1, input 2 B; ops [op0:mul x1]; imports p0[accumulated]=2B

**fuzz7-152** (mul|view) F=W/4 G=W/4: L=0, I*=4
- RU0: work 1, input 2 B; ops [op0:mul x1]; imports p0[accumulated]=2B
- RU1: work 1, input 2 B; ops [op0:mul x1]; imports p0[accumulated]=2B

**fuzz7-156** (gain+matmul+rmsnorm|shared,view) F=W/4 G=W/2: L=8, I*=20
- RU0: work 7, input 8 B; ops [op0:rmsnorm x5, op1:matmul x1, op2:gain x2]; imports op1:matmul.out=4B, p0[accumulated]=4B
- RU1: work 7, input 12 B; ops [op0:rmsnorm x5, op1:matmul x3]; imports op0:rmsnorm.out=4B, p0[accumulated]=4B, p1[token]=4B

**fuzz7-156** (gain+matmul+rmsnorm|shared,view) F=W/4 G=W/4: L=8, I*=26
- RU0: work 3, input 2 B; ops [op0:rmsnorm x3, op2:gain x1]; imports p0[accumulated]=2B
- RU1: work 4, input 6 B; ops [op0:rmsnorm x4]; imports op0:rmsnorm.out=4B, p0[accumulated]=2B
- RU2: work 3, input 8 B; ops [op0:rmsnorm x2, op1:matmul x2]; imports op0:rmsnorm.out=4B, p0[accumulated]=2B, p1[token]=2B
- RU3: work 4, input 10 B; ops [op0:rmsnorm x1, op1:matmul x2, op2:gain x1]; imports op0:rmsnorm.out=2B, op1:matmul.out=4B, p0[accumulated]=2B, p1[token]=2B

**fuzz7-157** (softmax) F=W/4 G=inf: L=8, I*=32
- RU0: work 22, input 8 B; ops [op0:softmax x20]; imports p0[accumulated]=8B
- RU1: work 8, input 24 B; ops [op0:softmax x8]; imports op0:softmax.out=24B

**fuzz7-157** (softmax) F=inf G=W/4: L=8, I*=36
- RU0: work 8, input 4 B; ops [op0:softmax x8]; imports p0[accumulated]=4B
- RU1: work 7, input 14 B; ops [op0:softmax x6]; imports op0:softmax.out=12B, p0[accumulated]=2B
- RU2: work 8, input 4 B; ops [op0:softmax x8]; imports p0[accumulated]=4B
- RU3: work 7, input 14 B; ops [op0:softmax x6]; imports op0:softmax.out=12B, p0[accumulated]=2B

**fuzz7-157** (softmax) F=W/4 G=W/2: L=8, I*=32
- RU0: work 15, input 16 B; ops [op0:softmax x14]; imports op0:softmax.out=12B, p0[accumulated]=4B
- RU1: work 15, input 16 B; ops [op0:softmax x14]; imports op0:softmax.out=12B, p0[accumulated]=4B

**fuzz7-159** (rmsnorm) F=W/4 G=W/4: L=12, I*=32
- RU0: work 7, input 6 B; ops [op0:rmsnorm x8]; imports p0[token]=6B
- RU1: work 6, input 10 B; ops [op0:rmsnorm x6]; imports op0:rmsnorm.out=4B, p0[token]=6B
- RU2: work 7, input 6 B; ops [op0:rmsnorm x8]; imports p0[token]=6B
- RU3: work 6, input 10 B; ops [op0:rmsnorm x6]; imports op0:rmsnorm.out=4B, p0[token]=6B

**fuzz7-159** (rmsnorm) F=W/2 G=W/4: L=12, I*=32
- RU0: work 7, input 6 B; ops [op0:rmsnorm x8]; imports p0[token]=6B
- RU1: work 6, input 10 B; ops [op0:rmsnorm x6]; imports op0:rmsnorm.out=4B, p0[token]=6B
- RU2: work 7, input 6 B; ops [op0:rmsnorm x8]; imports p0[token]=6B
- RU3: work 6, input 10 B; ops [op0:rmsnorm x6]; imports op0:rmsnorm.out=4B, p0[token]=6B

**fuzz7-162** (matmul|shared,view) F=W/4 G=W/4: L=12, I*=46
- RU0: work 6, input 12 B; ops [op0:matmul x8]; imports p0[accumulated]=6B, p1[token]=6B
- RU1: work 6, input 16 B; ops [op0:matmul x6]; imports op0:matmul.out=12B, p0[accumulated]=2B, p1[token]=2B
- RU2: work 6, input 8 B; ops [op0:matmul x6, op1:matmul x3]; imports p0[accumulated]=4B, p1[token]=4B
- RU3: work 6, input 10 B; ops [op1:matmul x7]; imports op1:matmul.out=4B, p0[accumulated]=6B

**fuzz7-164** (add+matmul+rowscale+softmax|shared,chain) F=W/4 G=inf: L=2, I*=22
- RU0: work 12, input 2 B; ops [op0:matmul x6, op1:rowscale x1, op2:softmax x7]; imports p1[token]=2B
- RU1: work 9, input 4 B; ops [op2:softmax x6]; imports op2:softmax.out=4B
- RU2: work 10, input 16 B; ops [op2:softmax x7, op3:add x3]; imports op2:softmax.out=16B

**fuzz7-164** (add+matmul+rowscale+softmax|shared,chain) F=W/4 G=W/2: L=2, I*=22
- RU0: work 12, input 2 B; ops [op0:matmul x6, op1:rowscale x1, op2:softmax x7]; imports p1[token]=2B
- RU1: work 9, input 4 B; ops [op2:softmax x6]; imports op2:softmax.out=4B
- RU2: work 10, input 16 B; ops [op2:softmax x7, op3:add x3]; imports op2:softmax.out=16B

**fuzz7-164** (add+matmul+rowscale+softmax|shared,chain) F=W/2 G=W/2: L=2, I*=14
- RU0: work 16, input 2 B; ops [op0:matmul x6, op1:rowscale x1, op2:softmax x10]; imports p1[token]=2B
- RU1: work 15, input 12 B; ops [op2:softmax x10, op3:add x3]; imports op2:softmax.out=12B

**fuzz7-164** (add+matmul+rowscale+softmax|shared,chain) F=W/4 G=W/4: L=2, I*=30
- RU0: work 6, input 2 B; ops [op0:matmul x6, op1:rowscale x1]; imports p1[token]=2B
- RU1: work 6, input 0 B; ops [op2:softmax x7]; imports nothing charged
- RU2: work 6, input 4 B; ops [op2:softmax x4]; imports op2:softmax.out=4B
- RU3: work 3, input 4 B; ops [op2:softmax x2]; imports op2:softmax.out=4B
- RU4: work 8, input 16 B; ops [op2:softmax x7, op3:add x1]; imports op2:softmax.out=16B
- RU5: work 1, input 2 B; ops [op3:add x1]; imports op2:softmax.out=2B
- RU6: work 1, input 2 B; ops [op3:add x1]; imports op2:softmax.out=2B

**fuzz7-166** (add+matmul_tt|shared,view) F=inf G=W/4: L=2, I*=6
- RU0: work 3, input 2 B; ops [op0:matmul x3, op1:add x1]; imports p1[accumulated]=2B
- RU1: work 3, input 2 B; ops [op0:matmul x3, op1:add x1]; imports p1[accumulated]=2B
- RU2: work 3, input 2 B; ops [op0:matmul x3, op1:add x1]; imports p1[accumulated]=2B
- RU3: work 3, input 0 B; ops [op1:add x3]; imports nothing charged

**fuzz7-169** (matmul+matmul_tt|shared,view,chain) F=W/2 G=W/4: L=8, I*=26
- RU0: work 4, input 8 B; ops [op0:matmul x6]; imports p0[accumulated]=4B, p1[token]=4B
- RU1: work 4, input 8 B; ops [op0:matmul x2, op1:matmul x3]; imports op0:matmul.out=4B, p0[accumulated]=2B, p1[token]=2B
- RU2: work 4, input 6 B; ops [op1:matmul x6]; imports op0:matmul.out=2B, p1[token]=4B
- RU3: work 2, input 4 B; ops [op1:matmul x3]; imports op0:matmul.out=2B, p1[token]=2B

**fuzz7-171** (matmul|shared,view) F=W/4 G=W/4: L=8, I*=32
- RU0: work 5, input 8 B; ops [op0:matmul x6]; imports p0[accumulated]=8B
- RU1: work 5, input 8 B; ops [op0:matmul x6]; imports p0[accumulated]=8B
- RU2: work 5, input 8 B; ops [op0:matmul x6]; imports p0[accumulated]=8B
- RU3: work 5, input 8 B; ops [op0:matmul x6]; imports p0[accumulated]=8B

**fuzz7-171** (matmul|shared,view) F=inf G=W/4: L=8, I*=32
- RU0: work 5, input 8 B; ops [op0:matmul x6]; imports p0[accumulated]=8B
- RU1: work 5, input 8 B; ops [op0:matmul x6]; imports p0[accumulated]=8B
- RU2: work 5, input 8 B; ops [op0:matmul x6]; imports p0[accumulated]=8B
- RU3: work 5, input 8 B; ops [op0:matmul x6]; imports p0[accumulated]=8B

**fuzz7-175** (matmul|shared) F=W/2 G=W/4: L=16, I*=44
- RU0: work 5, input 8 B; ops [op0:matmul x6]; imports p0[accumulated]=8B
- RU1: work 5, input 12 B; ops [op0:matmul x7]; imports p0[accumulated]=12B
- RU2: work 5, input 16 B; ops [op0:matmul x5]; imports op0:matmul.out=8B, p0[accumulated]=8B
- RU3: work 5, input 8 B; ops [op0:matmul x6]; imports p0[accumulated]=8B

**fuzz7-180** (matmul+mul|shared,view,chain) F=inf G=W/4: L=4, I*=16
- RU0: work 5, input 4 B; ops [op0:matmul x6, op1:mul x1]; imports p0[carried]=4B
- RU1: work 5, input 2 B; ops [op0:matmul x6, op1:mul x1]; imports p0[carried]=2B
- RU2: work 5, input 4 B; ops [op0:matmul x6, op1:mul x1]; imports p0[carried]=4B
- RU3: work 3, input 6 B; ops [op1:mul x3]; imports op0:matmul.out=6B

**fuzz7-180** (matmul+mul|shared,view,chain) F=W/4 G=W/4: L=4, I*=16
- RU0: work 5, input 4 B; ops [op0:matmul x6, op1:mul x1]; imports p0[carried]=4B
- RU1: work 5, input 2 B; ops [op0:matmul x6, op1:mul x1]; imports p0[carried]=2B
- RU2: work 5, input 4 B; ops [op0:matmul x6, op1:mul x1]; imports p0[carried]=4B
- RU3: work 3, input 6 B; ops [op1:mul x3]; imports op0:matmul.out=6B

**fuzz7-184** (matmul|shared) F=inf G=W/4: L=6, I*=18
- RU0: work 4, input 4 B; ops [op0:matmul x6]; imports p0[accumulated]=4B
- RU1: work 4, input 4 B; ops [op0:matmul x6]; imports p0[accumulated]=4B
- RU2: work 4, input 4 B; ops [op0:matmul x6]; imports p0[accumulated]=4B
- RU3: work 4, input 4 B; ops [op0:matmul x6]; imports p0[accumulated]=4B
- RU4: work 2, input 2 B; ops [op0:matmul x3]; imports p0[accumulated]=2B

**fuzz7-184** (matmul|shared) F=W/2 G=W/4: L=6, I*=18
- RU0: work 4, input 4 B; ops [op0:matmul x6]; imports p0[accumulated]=4B
- RU1: work 4, input 4 B; ops [op0:matmul x6]; imports p0[accumulated]=4B
- RU2: work 4, input 4 B; ops [op0:matmul x6]; imports p0[accumulated]=4B
- RU3: work 4, input 4 B; ops [op0:matmul x6]; imports p0[accumulated]=4B
- RU4: work 2, input 2 B; ops [op0:matmul x3]; imports p0[accumulated]=2B

**fuzz7-185** (softmax) F=W/2 G=W/2: L=6, I*=24
- RU0: work 11, input 6 B; ops [op0:softmax x11]; imports p0[token]=6B
- RU1: work 11, input 18 B; ops [op0:softmax x9]; imports op0:softmax.out=16B, p0[token]=2B

**fuzz7-185** (softmax) F=inf G=W/2: L=6, I*=24
- RU0: work 11, input 6 B; ops [op0:softmax x11]; imports p0[token]=6B
- RU1: work 11, input 18 B; ops [op0:softmax x9]; imports op0:softmax.out=16B, p0[token]=2B

**fuzz7-185** (softmax) F=W/2 G=inf: L=6, I*=22
- RU0: work 16, input 6 B; ops [op0:softmax x14]; imports p0[token]=6B
- RU1: work 6, input 16 B; ops [op0:softmax x6]; imports op0:softmax.out=16B

**fuzz7-185** (softmax) F=W/4 G=W/4: L=6, I*=40
- RU0: work 6, input 6 B; ops [op0:softmax x7]; imports p0[token]=6B
- RU1: work 4, input 10 B; ops [op0:softmax x3]; imports op0:softmax.out=8B, p0[token]=2B
- RU2: work 6, input 8 B; ops [op0:softmax x4]; imports op0:softmax.out=4B, p0[token]=4B
- RU3: work 6, input 16 B; ops [op0:softmax x6]; imports op0:softmax.out=16B

**fuzz7-190** (colsum) F=W/4 G=W/2: L=4, I*=12
- RU0: work 3, input 4 B; ops [op0:colsum x4]; imports p0[token]=4B
- RU1: work 3, input 8 B; ops [op0:colsum x4]; imports op0:colsum.out=8B

**fuzz7-190** (colsum) F=inf G=W/4: L=4, I*=12
- RU0: work 2, input 2 B; ops [op0:colsum x3]; imports p0[token]=2B
- RU1: work 2, input 6 B; ops [op0:colsum x2]; imports op0:colsum.out=4B, p0[token]=2B
- RU2: work 2, input 4 B; ops [op0:colsum x3]; imports op0:colsum.out=4B

**fuzz7-190** (colsum) F=W/4 G=W/4: L=4, I*=12
- RU0: work 2, input 2 B; ops [op0:colsum x3]; imports p0[token]=2B
- RU1: work 2, input 6 B; ops [op0:colsum x2]; imports op0:colsum.out=4B, p0[token]=2B
- RU2: work 2, input 4 B; ops [op0:colsum x3]; imports op0:colsum.out=4B

**fuzz7-195** (gain+softmax|shared) F=W/4 G=inf: L=4, I*=18
- RU0: work 3, input 4 B; ops [op0:gain x1, op1:softmax x3]; imports p0[accumulated]=2B, p1[accumulated]=2B
- RU1: work 3, input 6 B; ops [op1:softmax x2]; imports op1:softmax.out=4B, p0[accumulated]=2B
- RU2: work 3, input 8 B; ops [op1:softmax x3]; imports op1:softmax.out=8B

**fuzz7-195** (gain+softmax|shared) F=W/4 G=W/2: L=4, I*=18
- RU0: work 3, input 4 B; ops [op0:gain x1, op1:softmax x3]; imports p0[accumulated]=2B, p1[accumulated]=2B
- RU1: work 3, input 6 B; ops [op1:softmax x2]; imports op1:softmax.out=4B, p0[accumulated]=2B
- RU2: work 3, input 8 B; ops [op1:softmax x3]; imports op1:softmax.out=8B

**fuzz7-195** (gain+softmax|shared) F=W/2 G=inf: L=4, I*=12
- RU0: work 6, input 4 B; ops [op0:gain x1, op1:softmax x5]; imports p0[accumulated]=2B, p1[accumulated]=2B
- RU1: work 3, input 8 B; ops [op1:softmax x3]; imports op1:softmax.out=8B

**fuzz7-195** (gain+softmax|shared) F=inf G=W/2: L=4, I*=14
- RU0: work 4, input 12 B; ops [op0:gain x1, op1:softmax x3]; imports op1:softmax.out=8B, p0[accumulated]=2B, p1[accumulated]=2B
- RU1: work 5, input 2 B; ops [op1:softmax x5]; imports p0[accumulated]=2B

**fuzz7-198** (embed) F=inf G=W/4: L=28, I*=64
- RU0: work 3, input 16 B; ops [op0:embed x3]; imports p0[token]=4B, p1[token]=12B
- RU1: work 3, input 16 B; ops [op0:embed x3]; imports p0[token]=8B, p1[token]=8B
- RU2: work 3, input 16 B; ops [op0:embed x3]; imports p0[token]=8B, p1[token]=8B
- RU3: work 3, input 16 B; ops [op0:embed x3]; imports p0[token]=4B, p1[token]=12B

**fuzz7-199** (matmul|shared,view) F=W/4 G=W/2: L=2, I*=10
- RU0: work 2, input 2 B; ops [op0:matmul x4]; imports p0[token]=2B
- RU1: work 2, input 8 B; ops [op0:matmul x2]; imports op0:matmul.out=8B

