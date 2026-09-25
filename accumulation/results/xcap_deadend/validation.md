# Validation table (SPEC §6): exact `I*` vs certified `L` and constructive `U`

Generated 2026-09-15 16:09:51 by `accumulation.redteam.validation`; `I*` from `exact_min_input(flat, F, X, G)` (SPEC §4, brute force / HiGHS MILP, time box None s per point on torture rows). Policy `(F, G, X)`: `F` = per-gate upstream-work cap, `G` = per-RU total-work cap (`-` = unbounded), `X` = per-RU runtime-input cap in bytes; `inf` = unbounded.  `U` is the literal partition (`recompute=False`), the object `I*` matches.  All byte quantities are bytes for the whole micro circuit (`Workload.tokens` is 1--3 here; per-token normalisation is not meaningful at this scale).

Module versions (sha256[:10] of the file at evaluation time; several hashes = the module changed while the campaign ran):
  - `bounds/lower.py`: `e97fcdbecf` x4668
  - `bounds/upper.py`: `bffcd7b8d8` x4326, `5c1dbc6803` x342
  - `exact/solve.py`: `048007c023` x4471, `e2e7433a44` x197
  - `graph/opgraph.py`: `c0a6c64a99` x4668

## Aggregate

**All instances**: 151/3983 soundness violations (L <= I* and I* <= U) over 3983 judged instances (4668 evaluated; 3983 with I* proved optimal, 0 found-not-proved, 338 solver timeouts, 347 infeasible policies).
  L/I*: median 1.000, 5th percentile 0.333, min 0.091, fraction exactly 1: 0.624, fraction < 0.5: 0.110 (n = 3983).
  U/I* (recompute=False): median 1.000, 95th percentile 2.000, max 6.667, fraction exactly 1: 0.633 (n = 2888); headline U/I* median 1.000.
  Robustness gaps (upper_bound raised although a legal partition exists): 1095; binding-G points: 2313 (L computed at G=None on 0 of them); violation kinds: {'U_unavailable': 1095, 'L_gt_Istar': 143, 'Lnoprog_gt_Istar': 143, 'Ulit_lt_Istar': 8, 'Ulit_illegal': 7, 'Urec_lt_L': 6, 'Ulit_lt_L': 6, 'U_but_infeasible': 1}.
  Violating rows (source, circuit, F, G, X, L, I*, U_lit, violations):
    - ('torture', 'f1_lora_fixed_x', 6, None, 1048576, 6, 10, 6, ['Ulit_illegal', 'Ulit_lt_Istar'])
    - ('torture', 't1_same_row_read_twice', 1073741824, None, 1048576, 16, 16, 12, ['Ulit_lt_Istar', 'Urec_lt_L', 'Ulit_lt_L'])
    - ('torture', 't2_repeated_row_view_B', 1073741824, None, 1048576, 12, 8, 8, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'Urec_lt_L', 'Ulit_lt_L'])
    - ('torture', 't2_repeated_row_view_B', 1073741824, None, 8, 12, 8, 8, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'Urec_lt_L', 'Ulit_lt_L'])
    - ('torture', 't3_repeated_row_view_A', 1073741824, None, 6, 14, 12, 32, ['L_gt_Istar', 'Lnoprog_gt_Istar'])
    - ('torture', 't3_repeated_row_view_A', 1073741824, None, 4, 20, 14, 40, ['L_gt_Istar', 'Lnoprog_gt_Istar'])
    - ('torture', 't4_joint_epilogue_F', 8, None, 1048576, 30, 38, 30, ['Ulit_illegal', 'Ulit_lt_Istar'])
    - ('torture', 't4_joint_epilogue_F', 5, None, 1048576, 30, 38, 30, ['Ulit_illegal', 'Ulit_lt_Istar'])
    - ('torture', 't5_rowfull_epilogue_F', 12, None, 1048576, 16, 20, 16, ['Ulit_illegal', 'Ulit_lt_Istar'])
    - ('torture', 't5_rowfull_epilogue_F', 9, None, 1048576, 16, 20, 16, ['Ulit_illegal', 'Ulit_lt_Istar'])
    - ('torture', 't6_softmax_epilogue_F', 16, None, 1048576, 12, 16, 12, ['Ulit_illegal', 'Ulit_lt_Istar'])
    - ('torture', 't6_softmax_epilogue_F', 12, None, 1048576, 12, 22, 12, ['Ulit_illegal', 'Ulit_lt_Istar'])
    - ('torture', 't11_reprow_B_deep', 1073741824, None, 1048576, 16, 10, 10, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'Urec_lt_L', 'Ulit_lt_L'])
    - ('torture', 't11_reprow_B_deep', 1073741824, None, 10, 16, 10, 10, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'Urec_lt_L', 'Ulit_lt_L'])
    - ('torture', 't11_reprow_B_deep', 1073741824, None, 8, 16, 12, 12, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'Urec_lt_L', 'Ulit_lt_L'])
    - ('fuzz', 's21c63', 8, None, 6, 12, 6, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c63', 8, 2, 6, 12, 6, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c63', 8, None, 22, 12, 6, 12, ['L_gt_Istar', 'Lnoprog_gt_Istar'])
    - ('fuzz', 's21c63', 8, 2, 22, 12, 6, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c63', 8, None, 1152921504606846976, 12, 6, 12, ['L_gt_Istar', 'Lnoprog_gt_Istar'])
    - ('fuzz', 's21c63', 8, 2, 1152921504606846976, 12, 6, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c63', 1152921504606846976, None, 6, 12, 6, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c63', 1152921504606846976, 2, 6, 12, 6, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c63', 1152921504606846976, None, 22, 12, 6, 12, ['L_gt_Istar', 'Lnoprog_gt_Istar'])
    - ('fuzz', 's21c63', 1152921504606846976, 2, 22, 12, 6, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c63', 1152921504606846976, None, 1152921504606846976, 12, 6, 12, ['L_gt_Istar', 'Lnoprog_gt_Istar'])
    - ('fuzz', 's21c63', 1152921504606846976, 2, 1152921504606846976, 12, 6, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c65', 4, None, 16, 24, 8, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c65', 4, 8, 16, 24, 8, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c65', 4, None, 48, 24, 8, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])

**Torture / hand suite**: 15/182 soundness violations (L <= I* and I* <= U) over 182 judged instances (197 evaluated; 182 with I* proved optimal, 0 found-not-proved, 4 solver timeouts, 11 infeasible policies).
  L/I*: median 0.800, 5th percentile 0.300, min 0.143, fraction exactly 1: 0.385, fraction < 0.5: 0.170 (n = 182).
  U/I* (recompute=False): median 1.000, 95th percentile 1.500, max 2.857, fraction exactly 1: 0.704 (n = 152); headline U/I* median 1.000.
  Robustness gaps (upper_bound raised although a legal partition exists): 30; binding-G points: 81 (L computed at G=None on 0 of them); violation kinds: {'U_unavailable': 30, 'Ulit_lt_Istar': 8, 'Ulit_illegal': 7, 'L_gt_Istar': 7, 'Lnoprog_gt_Istar': 7, 'Urec_lt_L': 6, 'Ulit_lt_L': 6}.
  Violating rows (source, circuit, F, G, X, L, I*, U_lit, violations):
    - ('torture', 'f1_lora_fixed_x', 6, None, 1048576, 6, 10, 6, ['Ulit_illegal', 'Ulit_lt_Istar'])
    - ('torture', 't1_same_row_read_twice', 1073741824, None, 1048576, 16, 16, 12, ['Ulit_lt_Istar', 'Urec_lt_L', 'Ulit_lt_L'])
    - ('torture', 't2_repeated_row_view_B', 1073741824, None, 1048576, 12, 8, 8, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'Urec_lt_L', 'Ulit_lt_L'])
    - ('torture', 't2_repeated_row_view_B', 1073741824, None, 8, 12, 8, 8, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'Urec_lt_L', 'Ulit_lt_L'])
    - ('torture', 't3_repeated_row_view_A', 1073741824, None, 6, 14, 12, 32, ['L_gt_Istar', 'Lnoprog_gt_Istar'])
    - ('torture', 't3_repeated_row_view_A', 1073741824, None, 4, 20, 14, 40, ['L_gt_Istar', 'Lnoprog_gt_Istar'])
    - ('torture', 't4_joint_epilogue_F', 8, None, 1048576, 30, 38, 30, ['Ulit_illegal', 'Ulit_lt_Istar'])
    - ('torture', 't4_joint_epilogue_F', 5, None, 1048576, 30, 38, 30, ['Ulit_illegal', 'Ulit_lt_Istar'])
    - ('torture', 't5_rowfull_epilogue_F', 12, None, 1048576, 16, 20, 16, ['Ulit_illegal', 'Ulit_lt_Istar'])
    - ('torture', 't5_rowfull_epilogue_F', 9, None, 1048576, 16, 20, 16, ['Ulit_illegal', 'Ulit_lt_Istar'])
    - ('torture', 't6_softmax_epilogue_F', 16, None, 1048576, 12, 16, 12, ['Ulit_illegal', 'Ulit_lt_Istar'])
    - ('torture', 't6_softmax_epilogue_F', 12, None, 1048576, 12, 22, 12, ['Ulit_illegal', 'Ulit_lt_Istar'])
    - ('torture', 't11_reprow_B_deep', 1073741824, None, 1048576, 16, 10, 10, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'Urec_lt_L', 'Ulit_lt_L'])
    - ('torture', 't11_reprow_B_deep', 1073741824, None, 10, 16, 10, 10, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'Urec_lt_L', 'Ulit_lt_L'])
    - ('torture', 't11_reprow_B_deep', 1073741824, None, 8, 16, 12, 12, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'Urec_lt_L', 'Ulit_lt_L'])

**Adversarial fuzz**: 136/3801 soundness violations (L <= I* and I* <= U) over 3801 judged instances (4471 evaluated; 3801 with I* proved optimal, 0 found-not-proved, 334 solver timeouts, 336 infeasible policies).
  L/I*: median 1.000, 5th percentile 0.333, min 0.091, fraction exactly 1: 0.636, fraction < 0.5: 0.107 (n = 3801).
  U/I* (recompute=False): median 1.000, 95th percentile 2.000, max 6.667, fraction exactly 1: 0.629 (n = 2736); headline U/I* median 1.000.
  Robustness gaps (upper_bound raised although a legal partition exists): 1065; binding-G points: 2232 (L computed at G=None on 0 of them); violation kinds: {'U_unavailable': 1065, 'L_gt_Istar': 136, 'Lnoprog_gt_Istar': 136, 'U_but_infeasible': 1}.
  Violating rows (source, circuit, F, G, X, L, I*, U_lit, violations):
    - ('fuzz', 's21c63', 8, None, 6, 12, 6, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c63', 8, 2, 6, 12, 6, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c63', 8, None, 22, 12, 6, 12, ['L_gt_Istar', 'Lnoprog_gt_Istar'])
    - ('fuzz', 's21c63', 8, 2, 22, 12, 6, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c63', 8, None, 1152921504606846976, 12, 6, 12, ['L_gt_Istar', 'Lnoprog_gt_Istar'])
    - ('fuzz', 's21c63', 8, 2, 1152921504606846976, 12, 6, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c63', 1152921504606846976, None, 6, 12, 6, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c63', 1152921504606846976, 2, 6, 12, 6, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c63', 1152921504606846976, None, 22, 12, 6, 12, ['L_gt_Istar', 'Lnoprog_gt_Istar'])
    - ('fuzz', 's21c63', 1152921504606846976, 2, 22, 12, 6, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c63', 1152921504606846976, None, 1152921504606846976, 12, 6, 12, ['L_gt_Istar', 'Lnoprog_gt_Istar'])
    - ('fuzz', 's21c63', 1152921504606846976, 2, 1152921504606846976, 12, 6, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c65', 4, None, 16, 24, 8, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c65', 4, 8, 16, 24, 8, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c65', 4, None, 48, 24, 8, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c65', 4, 8, 48, 24, 8, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c65', 4, None, 88, 24, 8, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c65', 4, 8, 88, 24, 8, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c65', 1152921504606846976, None, 16, 24, 8, 32, ['L_gt_Istar', 'Lnoprog_gt_Istar'])
    - ('fuzz', 's21c65', 1152921504606846976, 8, 16, 24, 8, 32, ['L_gt_Istar', 'Lnoprog_gt_Istar'])
    - ('fuzz', 's21c65', 1152921504606846976, None, 48, 24, 8, 24, ['L_gt_Istar', 'Lnoprog_gt_Istar'])
    - ('fuzz', 's21c65', 1152921504606846976, 8, 48, 24, 8, 32, ['L_gt_Istar', 'Lnoprog_gt_Istar'])
    - ('fuzz', 's21c65', 1152921504606846976, None, 88, 24, 8, 24, ['L_gt_Istar', 'Lnoprog_gt_Istar'])
    - ('fuzz', 's21c65', 1152921504606846976, 8, 88, 24, 8, 32, ['L_gt_Istar', 'Lnoprog_gt_Istar'])
    - ('fuzz', 's21c77', 2, None, 10, 56, 32, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c77', 2, None, 24, 56, 32, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c77', 2, None, 104, 56, 32, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c77', 17, None, 10, 56, 32, None, ['L_gt_Istar', 'Lnoprog_gt_Istar', 'U_unavailable'])
    - ('fuzz', 's21c77', 17, None, 24, 56, 32, 64, ['L_gt_Istar', 'Lnoprog_gt_Istar'])
    - ('fuzz', 's21c77', 17, None, 104, 56, 32, 56, ['L_gt_Istar', 'Lnoprog_gt_Istar'])
  Fuzz side records: {'build_error': 0, 'eval_error': 0, 'mono': 0}.

### By family

| family | n | soundness violations | L/I* median | L/I* p05 | L/I* min | U/I* median | U unavailable |
|---|---|---|---|---|---|---|---|
| a (independent generation) | 13 | 0 | 0.500 | 0.222 | 0.182 | 1.000 | 0 |
| add | 126 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 52 |
| add+colsum | 12 | 0 | 0.600 | 0.333 | 0.333 | 1.000 | 6 |
| add+colsum+embed+gain+swiglu\|shared,view,chain | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.083 | 0 |
| add+colsum+embed+matmul_tt\|shared,view,chain | 12 | 0 | 0.800 | 0.800 | 0.800 | 1.000 | 1 |
| add+colsum+embed\|shared | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| add+colsum+matmul+mul+swiglu\|shared,view,chain | 12 | 0 | 1.000 | 0.500 | 0.500 | 1.000 | 8 |
| add+colsum+matmul+rowscale\|shared,view | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| add+colsum+matmul_tt+mul\|shared,view,chain | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| add+colsum+matmul_tt+rowscale\|shared,view,chain | 12 | 1 | 1.000 | 1.000 | 1.000 | 1.125 | 0 |
| add+colsum+matmul\|shared,chain | 12 | 0 | 1.000 | 0.500 | 0.500 | 1.000 | 6 |
| add+colsum+matmul\|shared,view | 24 | 0 | 1.000 | 0.889 | 0.889 | 2.000 | 2 |
| add+colsum+matmul\|shared,view,chain | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| add+colsum+rowscale\|shared,chain | 12 | 0 | 1.000 | 0.714 | 0.714 | 1.200 | 6 |
| add+colsum\|shared,view | 12 | 0 | 1.000 | 0.833 | 0.833 | 1.200 | 0 |
| add+embed+gain+mul\|shared | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| add+embed+gain+swiglu\|shared,chain | 12 | 0 | 0.842 | 0.432 | 0.432 | 1.474 | 8 |
| add+embed+matmul+matmul_tt\|shared,chain | 24 | 0 | 0.871 | 0.568 | 0.568 | 1.182 | 0 |
| add+embed+matmul_tt+rowscale\|shared,view,chain | 12 | 0 | 0.529 | 0.360 | 0.360 | 2.040 | 0 |
| add+embed+matmul\|shared,chain | 12 | 0 | 0.840 | 0.568 | 0.568 | 1.541 | 0 |
| add+embed+rowscale\|shared | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.429 | 0 |
| add+embed+rowscale\|shared,chain | 12 | 0 | 1.000 | 1.000 | 1.000 | 3.500 | 0 |
| add+embed\|shared,view,chain | 12 | 0 | 1.000 | 1.000 | 1.000 | 2.647 | 0 |
| add+embed\|view,chain | 12 | 0 | 0.833 | 0.714 | 0.714 | 1.167 | 0 |
| add+gain | 12 | 0 | 1.000 | 0.750 | 0.750 | 1.000 | 0 |
| add+gain+matmul+matmul_tt\|shared,view,chain | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.308 | 0 |
| add+gain+matmul+rowscale\|shared,view | 12 | 0 | 0.880 | 0.550 | 0.550 | 1.682 | 4 |
| add+gain+matmul_tt\|shared | 12 | 0 | 1.000 | 1.000 | 1.000 | 2.000 | 4 |
| add+gain+matmul\|shared | 12 | 0 | 0.842 | 0.842 | 0.842 | 1.000 | 0 |
| add+gain+matmul\|shared,view | 12 | 0 | 0.909 | 0.500 | 0.500 | 1.182 | 0 |
| add+gain+matmul\|shared,view,chain | 36 | 0 | 1.000 | 0.233 | 0.233 | 1.000 | 10 |
| add+gain+mul+rmsnorm\|shared | 12 | 0 | 1.000 | 0.562 | 0.562 | 1.778 | 8 |
| add+gain+mul+rowscale\|shared | 12 | 0 | 1.000 | 0.654 | 0.654 | 1.118 | 2 |
| add+gain+mul\|shared,chain | 12 | 0 | 1.000 | 0.643 | 0.643 | 1.444 | 8 |
| add+gain+mul\|shared,view | 12 | 12 | 1.200 | 1.200 | 1.200 | 1.600 | 0 |
| add+gain+mul\|shared,view,chain | 12 | 0 | 1.000 | 0.875 | 0.875 | 1.750 | 0 |
| add+gain+rowscale+swiglu\|shared | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.216 | 8 |
| add+gain+rowscale+swiglu\|shared,chain | 12 | 0 | 0.875 | 0.875 | 0.875 | 1.250 | 0 |
| add+gain+rowscale\|shared | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.273 | 4 |
| add+gain\|chain | 12 | 0 | 1.000 | 1.000 | 1.000 | 2.000 | 8 |
| add+matmul | 24 | 0 | 1.000 | 0.657 | 0.657 | 1.000 | 6 |
| add+matmul+matmul_tt+mul\|shared,chain | 12 | 0 | 1.000 | 0.800 | 0.800 | 1.250 | 0 |
| add+matmul+matmul_tt+rowscale\|chain | 12 | 0 | 1.000 | 0.125 | 0.125 | 1.000 | 0 |
| add+matmul+matmul_tt+rowscale\|shared,view,chain | 12 | 0 | 1.000 | 0.778 | 0.778 | 1.000 | 0 |
| add+matmul+matmul_tt\|shared,view,chain | 12 | 0 | 0.600 | 0.273 | 0.273 | 1.200 | 6 |
| add+matmul+mul+rowscale\|shared,chain | 24 | 0 | 1.000 | 0.522 | 0.522 | 1.000 | 6 |
| add+matmul+mul+rowscale\|shared,view | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.314 | 0 |
| add+matmul+mul\|shared | 24 | 0 | 1.000 | 0.273 | 0.273 | 1.000 | 6 |
| add+matmul+rowscale\|shared,view,chain | 36 | 12 | 1.429 | 0.500 | 0.500 | 1.429 | 0 |
| add+matmul+swiglu\|shared,chain | 12 | 0 | 0.600 | 0.375 | 0.375 | 1.400 | 0 |
| add+matmul+swiglu\|shared,view | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| add+matmul_tt | 12 | 0 | 0.778 | 0.700 | 0.700 | 1.000 | 0 |
| add+matmul_tt\|shared | 24 | 0 | 0.714 | 0.636 | 0.636 | 1.500 | 8 |
| add+matmul_tt\|shared,chain | 12 | 0 | 0.600 | 0.500 | 0.500 | 3.000 | 4 |
| add+matmul_tt\|shared,view | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.250 | 0 |
| add+matmul\|shared | 84 | 0 | 1.000 | 0.186 | 0.186 | 1.000 | 18 |
| add+matmul\|shared,chain | 12 | 0 | 0.500 | 0.333 | 0.333 | 2.667 | 0 |
| add+matmul\|shared,view | 24 | 2 | 1.000 | 0.583 | 0.583 | 1.000 | 6 |
| add+matmul\|shared,view,chain | 36 | 0 | 0.889 | 0.333 | 0.333 | 1.000 | 2 |
| add+mul+rowscale+softmax\|shared,chain | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| add+mul+softmax\|shared,view,chain | 12 | 0 | 0.632 | 0.400 | 0.400 | 1.000 | 8 |
| add+mul+swiglu\|shared,view | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.222 | 0 |
| add+mul\|chain | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.214 | 4 |
| add+mul\|shared | 24 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 4 |
| add+mul\|shared,chain | 24 | 0 | 1.000 | 0.500 | 0.500 | 1.000 | 4 |
| add+mul\|shared,view,chain | 12 | 0 | 1.000 | 1.000 | 1.000 | 5.000 | 8 |
| add+rmsnorm+rowscale\|shared,view,chain | 12 | 0 | 0.556 | 0.556 | 0.556 | - | 6 |
| add+rmsnorm\|shared | 12 | 0 | 1.000 | 0.750 | 0.750 | 1.500 | 0 |
| add+rowscale\|shared | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| add+rowscale\|view | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 6 |
| add+swiglu\|shared | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 10 |
| add+swiglu\|shared,chain | 12 | 0 | 1.000 | 1.000 | 1.000 | 2.000 | 8 |
| add\|shared | 252 | 0 | 1.000 | 1.000 | 1.000 | 2.000 | 110 |
| add\|shared,view | 36 | 12 | 1.000 | 1.000 | 1.000 | 2.000 | 8 |
| add\|view | 36 | 12 | 1.000 | 1.000 | 1.000 | 1.000 | 14 |
| b (shared compact precursor) | 6 | 0 | 0.400 | 0.188 | 0.188 | 1.143 | 1 |
| c (independent accumulated precursors) | 9 | 0 | 1.000 | 0.500 | 0.500 | 1.000 | 2 |
| colsum | 216 | 0 | 1.000 | 0.200 | 0.200 | 1.000 | 100 |
| colsum+embed | 12 | 0 | 0.833 | 0.556 | 0.556 | 1.000 | 9 |
| colsum+embed+gain+mul\|shared,chain | 12 | 0 | 0.800 | 0.750 | 0.750 | 1.250 | 4 |
| colsum+embed+matmul\|shared | 12 | 0 | 1.000 | 0.647 | 0.647 | 1.029 | 0 |
| colsum+embed+mul\|shared | 24 | 0 | 1.000 | 0.524 | 0.524 | 1.000 | 6 |
| colsum+embed+rmsnorm\|shared,view | 12 | 0 | 1.000 | 1.000 | 1.000 | - | 2 |
| colsum+gain\|shared | 12 | 0 | 1.000 | 0.333 | 0.333 | 1.000 | 9 |
| colsum+gain\|shared,chain | 12 | 0 | 1.000 | 1.000 | 1.000 | 3.000 | 8 |
| colsum+matmul+mul+swiglu\|shared,view | 12 | 0 | 1.000 | 0.222 | 0.222 | 1.000 | 9 |
| colsum+matmul+mul\|shared,chain | 12 | 0 | 1.000 | 0.095 | 0.095 | 2.500 | 4 |
| colsum+matmul+mul\|shared,view,chain | 12 | 12 | 2.000 | 2.000 | 2.000 | 3.000 | 0 |
| colsum+matmul_tt+mul\|shared,view,chain | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| colsum+matmul_tt\|shared | 12 | 0 | 0.500 | 0.400 | 0.400 | 1.000 | 6 |
| colsum+matmul\|shared | 24 | 0 | 1.000 | 0.296 | 0.296 | 1.000 | 4 |
| colsum+matmul\|shared,view | 48 | 0 | 1.000 | 0.125 | 0.125 | 1.273 | 14 |
| colsum+matmul\|view | 12 | 12 | 3.000 | 3.000 | 3.000 | 4.000 | 4 |
| colsum+mul+rowscale+swiglu\|shared,chain | 12 | 0 | 0.923 | 0.429 | 0.429 | 2.000 | 4 |
| colsum+mul\|chain | 12 | 0 | 1.000 | 0.167 | 0.167 | 2.000 | 6 |
| colsum+mul\|shared | 36 | 0 | 1.000 | 0.400 | 0.400 | 1.000 | 25 |
| colsum+mul\|shared,view | 24 | 0 | 0.667 | 0.400 | 0.400 | 1.500 | 9 |
| colsum+rowscale | 12 | 0 | 1.000 | 0.833 | 0.833 | 1.000 | 4 |
| colsum+rowscale\|shared,view | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| colsum+swiglu\|shared | 12 | 0 | 0.667 | 0.500 | 0.500 | 1.000 | 6 |
| colsum\|view | 12 | 6 | 2.000 | 0.500 | 0.500 | 2.000 | 4 |
| d (two-layer dense mixing) | 15 | 0 | 0.714 | 0.500 | 0.455 | 1.000 | 0 |
| e (shared weights across branches) | 11 | 0 | 0.625 | 0.333 | 0.333 | 1.000 | 4 |
| embed | 108 | 0 | 0.778 | 0.500 | 0.400 | 1.000 | 0 |
| embed+gain | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| embed+gain+matmul\|shared,chain | 12 | 0 | 0.583 | 0.583 | 0.583 | 1.000 | 0 |
| embed+gain+matmul\|shared,view,chain | 12 | 0 | 1.000 | 0.250 | 0.250 | 2.500 | 6 |
| embed+gain+rowscale\|chain | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| embed+gain+rowscale\|view | 12 | 0 | 1.000 | 0.818 | 0.818 | 1.000 | 0 |
| embed+gain\|view | 12 | 6 | 1.750 | 0.583 | 0.583 | 1.750 | 10 |
| embed+matmul | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| embed+matmul+matmul_tt+swiglu\|shared,view | 12 | 0 | 0.615 | 0.500 | 0.500 | - | 12 |
| embed+matmul+mul\|shared,view,chain | 12 | 0 | 0.688 | 0.688 | 0.688 | 1.000 | 0 |
| embed+matmul\|chain | 12 | 0 | 0.852 | 0.689 | 0.689 | 1.081 | 0 |
| embed+matmul\|shared,chain | 12 | 0 | 0.429 | 0.333 | 0.333 | 1.111 | 0 |
| embed+matmul\|shared,view,chain | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| embed+matmul\|view,chain | 12 | 0 | 1.000 | 0.462 | 0.462 | 1.000 | 0 |
| embed+mul | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| embed+mul\|shared,chain | 12 | 0 | 0.643 | 0.545 | 0.545 | 1.179 | 6 |
| embed+rowscale | 24 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| f (LoRA compact state) | 15 | 1 | 0.500 | 0.412 | 0.333 | 1.000 | 0 |
| g (residual / copy / embedding) | 20 | 0 | 0.667 | 0.417 | 0.417 | 1.000 | 2 |
| gain | 132 | 0 | 1.000 | 0.500 | 0.333 | 1.000 | 44 |
| gain+matmul+matmul_tt+rowscale\|shared,view,chain | 12 | 0 | 1.000 | 0.750 | 0.750 | 1.000 | 0 |
| gain+matmul+rmsnorm\|shared,chain | 12 | 0 | 1.000 | 0.600 | 0.600 | 1.000 | 6 |
| gain+matmul+rmsnorm\|shared,view | 12 | 0 | 1.000 | 0.684 | 0.684 | 1.000 | 6 |
| gain+matmul+rowscale\|shared | 12 | 0 | 0.833 | 0.500 | 0.500 | 1.800 | 0 |
| gain+matmul+swiglu\|shared,chain | 12 | 0 | 1.000 | 0.375 | 0.375 | 1.875 | 0 |
| gain+matmul_tt+rowscale | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| gain+matmul_tt+rowscale\|shared | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.194 | 0 |
| gain+matmul_tt\|shared | 36 | 0 | 1.000 | 0.100 | 0.100 | 1.000 | 0 |
| gain+matmul\|shared | 24 | 0 | 1.000 | 0.714 | 0.714 | 1.000 | 0 |
| gain+matmul\|shared,chain | 12 | 0 | 0.333 | 0.333 | 0.333 | 1.333 | 0 |
| gain+mul+rowscale\|shared,view,chain | 1 | 0 | 0.538 | 0.538 | 0.538 | - | 1 |
| gain+mul\|shared,chain | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 6 |
| gain+mul\|view | 24 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| gain+rowscale | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| gain+swiglu\|shared,chain | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 3 |
| h (weight gradient / reduction) | 20 | 0 | 1.000 | 0.429 | 0.333 | 1.000 | 2 |
| i (dense inference) | 11 | 0 | 0.588 | 0.143 | 0.143 | 1.000 | 2 |
| k (G forced split (hand)) | 7 | 0 | 0.667 | 0.500 | 0.500 | 1.000 | 1 |
| matmul | 228 | 0 | 0.750 | 0.333 | 0.143 | 1.000 | 16 |
| matmul+matmul_tt+mul\|shared,chain | 12 | 0 | 1.000 | 0.538 | 0.538 | 1.308 | 0 |
| matmul+matmul_tt\|shared | 24 | 0 | 1.000 | 0.556 | 0.556 | 1.000 | 0 |
| matmul+matmul_tt\|shared,view | 48 | 4 | 1.000 | 0.600 | 0.600 | 1.000 | 4 |
| matmul+mul+rowscale\|shared | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| matmul+mul\|shared | 36 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 13 |
| matmul+mul\|shared,chain | 24 | 0 | 1.000 | 0.571 | 0.571 | 1.571 | 0 |
| matmul+mul\|shared,view,chain | 24 | 0 | 1.000 | 0.833 | 0.833 | 1.800 | 0 |
| matmul+rmsnorm\|shared,chain | 12 | 0 | 0.800 | 0.400 | 0.400 | 1.000 | 8 |
| matmul+rowscale | 24 | 0 | 1.000 | 0.500 | 0.500 | 1.000 | 0 |
| matmul+rowscale\|shared | 48 | 0 | 1.000 | 0.500 | 0.500 | 1.000 | 10 |
| matmul+rowscale\|shared,view,chain | 24 | 0 | 0.667 | 0.143 | 0.143 | 1.000 | 10 |
| matmul_tt | 48 | 0 | 1.000 | 0.477 | 0.125 | 1.000 | 2 |
| matmul_tt+mul | 12 | 0 | 1.000 | 0.800 | 0.800 | 1.000 | 0 |
| matmul_tt+mul\|shared,chain | 12 | 0 | 1.000 | 0.500 | 0.500 | 1.500 | 8 |
| matmul_tt+mul\|shared,view | 24 | 0 | 1.000 | 0.423 | 0.423 | 1.154 | 0 |
| matmul_tt+rmsnorm | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| matmul_tt+rowscale | 12 | 0 | 0.818 | 0.529 | 0.529 | 1.000 | 0 |
| matmul_tt+rowscale\|shared | 24 | 0 | 1.000 | 0.619 | 0.619 | 1.615 | 0 |
| matmul_tt+rowscale\|shared,chain | 12 | 0 | 1.000 | 0.750 | 0.750 | 1.875 | 0 |
| matmul_tt\|shared | 60 | 0 | 0.500 | 0.091 | 0.091 | 1.500 | 6 |
| matmul_tt\|view | 12 | 6 | 2.000 | 0.400 | 0.400 | 4.000 | 4 |
| matmul\|shared | 228 | 0 | 0.600 | 0.200 | 0.091 | 1.538 | 38 |
| matmul\|shared,chain | 12 | 0 | 0.833 | 0.455 | 0.455 | 1.111 | 4 |
| matmul\|shared,view | 48 | 3 | 0.500 | 0.250 | 0.103 | 2.000 | 8 |
| matmul\|shared,view,chain | 48 | 0 | 0.800 | 0.200 | 0.200 | 1.400 | 4 |
| matmul\|view | 24 | 0 | 0.333 | 0.133 | 0.133 | 1.000 | 6 |
| mul | 84 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 35 |
| mul+rmsnorm\|shared,chain | 12 | 0 | 1.000 | 0.235 | 0.235 | 1.000 | 8 |
| mul+rowscale\|shared | 24 | 0 | 1.000 | 0.500 | 0.500 | 1.000 | 0 |
| mul+rowscale\|shared,chain | 12 | 0 | 1.000 | 0.778 | 0.778 | 1.000 | 6 |
| mul+rowscale\|shared,view | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 0 |
| mul+softmax\|shared | 12 | 0 | 0.667 | 0.583 | 0.583 | 1.000 | 9 |
| mul+swiglu\|shared,view | 12 | 12 | 3.000 | 3.000 | 3.000 | 4.000 | 6 |
| mul+swiglu\|shared,view,chain | 12 | 0 | 1.000 | 1.000 | 1.000 | 2.333 | 6 |
| mul\|shared | 48 | 0 | 1.000 | 1.000 | 1.000 | 1.727 | 19 |
| mul\|shared,view | 24 | 12 | 2.000 | 1.000 | 1.000 | 2.000 | 16 |
| mul\|view | 36 | 12 | 1.000 | 1.000 | 1.000 | 1.500 | 8 |
| mul\|view,chain | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.500 | 0 |
| r (dense training) | 10 | 0 | 0.700 | 0.357 | 0.357 | 1.222 | 1 |
| rmsnorm | 72 | 0 | 0.750 | 0.222 | 0.222 | 1.000 | 33 |
| rowscale | 84 | 0 | 1.000 | 1.000 | 0.800 | 1.000 | 4 |
| rowscale\|shared,chain | 12 | 0 | 1.000 | 1.000 | 1.000 | 2.000 | 4 |
| rowscale\|view | 24 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 6 |
| softmax | 24 | 0 | 1.000 | 0.200 | 0.200 | 1.000 | 4 |
| swiglu\|shared | 24 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 12 |
| swiglu\|shared,view | 24 | 0 | 1.000 | 0.600 | 0.600 | 1.000 | 8 |
| swiglu\|view | 12 | 0 | 1.000 | 1.000 | 1.000 | 1.000 | 6 |
| t (torture (overlapping reads, epilogues)) | 60 | 14 | 0.857 | 0.421 | 0.167 | 1.000 | 15 |

### By policy regime

| regime | n | soundness violations | L/I* median | L/I* p05 | U/I* median | U unavailable |
|---|---|---|---|---|---|---|
| F<inf, G<work, X<inf | 1128 | 26 | 0.818 | 0.222 | 1.000 | 376 |
| F<inf, G<work, X=inf | 390 | 10 | 0.769 | 0.200 | 1.125 | 119 |
| F<inf, G=None, X<inf | 1186 | 38 | 1.000 | 0.400 | 1.000 | 267 |
| F<inf, G=None, X=inf | 371 | 21 | 1.000 | 0.500 | 1.000 | 37 |
| F=inf, G<work, X<inf | 632 | 18 | 1.000 | 0.235 | 1.000 | 172 |
| F=inf, G<work, X=inf | 163 | 6 | 0.833 | 0.235 | 1.067 | 42 |
| F=inf, G=None, X<inf | 634 | 23 | 1.000 | 0.714 | 1.000 | 82 |
| F=inf, G=None, X=inf | 164 | 9 | 1.000 | 1.000 | 1.000 | 0 |

## Table: torture suite, dense inference / training micro circuits, G forced split

Rows are grouped by family; `hand` is the analytically derived `I*` (`eq`), bound (`ge`/`le`) or infeasibility (`inf`) from the case docstring in `redteam/torture.py`.  `L@G=None`: the certificate ran without the binding `G`.

### i. dense inference

| circuit | (F, G, X) | I* | L | U (recompute=False) | L/I* | U/I* | hand | note |
|---|---|---|---|---|---|---|---|---|
| i1_inference_mlp | (inf, -, inf) | 4 | 4 | 4 | 1.000 | 1.000 | eq 4 |  |
| i1_inference_mlp | (inf, 12, inf) | 12 | 4 | 16 | 0.333 | 1.333 | eq 12 |  |
| i1_inference_mlp | (inf, 6, inf) | 24 | 4 | 28 | 0.167 | 1.167 | eq 24 |  |
| i1_inference_mlp | (inf, -, 4) | 4 | 4 | unavailable | 1.000 | - | eq 4 | U_unavailable |
| i1_inference_mlp | (inf, 12, 4) | 20 | 4 | unavailable | 0.200 | - |  | U_unavailable |
| i1_inference_mlp | (4, -, inf) | 16 | 4 | 20 | 0.250 | 1.250 |  |  |
| i1_inference_mlp | (4, 6, inf) | 28 | 4 | 28 | 0.143 | 1.000 |  |  |
| i2_inference_attn_decode | (inf, -, inf) | 20 | 20 | 20 | 1.000 | 1.000 | eq 20 |  |
| i2_inference_attn_decode | (inf, -, 12) | 32 | 20 | 32 | 0.625 | 1.000 |  |  |
| i2_inference_attn_decode | (inf, 20, 12) | 34 | 20 | 36 | 0.588 | 1.059 |  |  |
| i2_inference_attn_decode | (inf, 20, inf) | 32 | 20 | 32 | 0.625 | 1.000 |  |  |

### r. dense training

| circuit | (F, G, X) | I* | L | U (recompute=False) | L/I* | U/I* | hand | note |
|---|---|---|---|---|---|---|---|---|
| r1_train_linear_sgd | (inf, -, inf) | 20 | 20 | 20 | 1.000 | 1.000 | eq 20 |  |
| r1_train_linear_sgd | (inf, 11, inf) | 28 | 20 | 40 | 0.714 | 1.429 | eq 28 |  |
| r1_train_linear_sgd | (inf, 5, inf) | timeout | 20 | 52 | - | - |  |  |
| r1_train_linear_sgd | (inf, -, 12) | 36 | 20 | 44 | 0.556 | 1.222 |  |  |
| r1_train_linear_sgd | (inf, 11, 12) | 42 | 20 | 44 | 0.476 | 1.048 |  |  |
| r1_train_linear_sgd | (inf, -, 8) | 56 | 20 | unavailable | 0.357 | - |  | U_unavailable |
| r2_train_two_layer | (inf, -, inf) | 28 | 28 | 28 | 1.000 | 1.000 | eq 28 |  |
| r2_train_two_layer | (inf, 14, inf) | 40 | 28 | 48 | 0.700 | 1.200 |  |  |
| r2_train_two_layer | (inf, -, 16) | 40 | 28 | 52 | 0.700 | 1.300 |  |  |
| r2_train_two_layer | (inf, 14, 16) | 40 | 28 | 52 | 0.700 | 1.300 |  |  |

### k. G forced split (hand)

| circuit | (F, G, X) | I* | L | U (recompute=False) | L/I* | U/I* | hand | note |
|---|---|---|---|---|---|---|---|---|
| k1_matmul_G_forced_split | (inf, -, inf) | 16 | 16 | 16 | 1.000 | 1.000 | eq 16 |  |
| k1_matmul_G_forced_split | (inf, 12, inf) | 16 | 16 | 16 | 1.000 | 1.000 | eq 16 |  |
| k1_matmul_G_forced_split | (inf, 11, inf) | 20 | 16 | 24 | 0.800 | 1.200 | eq 20 |  |
| k1_matmul_G_forced_split | (inf, 6, inf) | 24 | 16 | 24 | 0.667 | 1.000 | eq 24 |  |
| k1_matmul_G_forced_split | (inf, 3, inf) | 32 | 18 | 32 | 0.562 | 1.000 | eq 32 |  |
| k1_matmul_G_forced_split | (inf, 1, inf) | 64 | 32 | unavailable | 0.500 | - | eq 64 | U_unavailable |
| k1_matmul_G_forced_split | (inf, 0, inf) | infeasible | error | unavailable | - | - | inf inf |  |

### d. two-layer dense mixing

| circuit | (F, G, X) | I* | L | U (recompute=False) | L/I* | U/I* | hand | note |
|---|---|---|---|---|---|---|---|---|
| d_two_layer_M1 | (inf, -, inf) | 20 | 20 | 20 | 1.000 | 1.000 | eq 20 |  |
| d_two_layer_M1 | (inf, -, 12) | 24 | 20 | 24 | 0.833 | 1.000 | eq 24 |  |
| d_two_layer_M1 | (inf, -, 8) | 28 | 20 | 28 | 0.714 | 1.000 | eq 28 |  |
| d_two_layer_M1 | (inf, 7, inf) | 24 | 20 | 24 | 0.833 | 1.000 |  |  |
| d_two_layer_M1 | (inf, 3, inf) | 36 | 20 | 36 | 0.556 | 1.000 |  |  |
| d_two_layer_M2 | (inf, -, inf) | 20 | 20 | 20 | 1.000 | 1.000 | eq 20 |  |
| d_two_layer_M2 | (inf, -, 12) | 36 | 20 | 36 | 0.556 | 1.000 | eq 36 |  |
| d_two_layer_M2 | (inf, -, 8) | 44 | 20 | 44 | 0.455 | 1.000 | eq 44 |  |
| d_two_layer_M2 | (inf, 14, inf) | 28 | 20 | 28 | 0.714 | 1.000 |  |  |
| d_two_layer_M2 | (inf, 7, inf) | 40 | 20 | 48 | 0.500 | 1.200 |  |  |
| d_two_layer_M3 | (inf, -, inf) | 20 | 20 | 20 | 1.000 | 1.000 | eq 20 |  |
| d_two_layer_M3 | (inf, -, 12) | timeout | 20 | 48 | - | - | eq 48 |  |
| d_two_layer_M3 | (inf, -, 8) | timeout | 20 | 60 | - | - | eq 60 |  |
| d_two_layer_M3 | (inf, 21, inf) | 32 | 20 | 32 | 0.625 | 1.000 |  |  |
| d_two_layer_M3 | (inf, 10, inf) | timeout | 20 | 56 | - | - |  |  |

### a. independent generation

| circuit | (F, G, X) | I* | L | U (recompute=False) | L/I* | U/I* | hand | note |
|---|---|---|---|---|---|---|---|---|
| a1_independent_rows | (inf, -, inf) | 12 | 12 | 12 | 1.000 | 1.000 | eq 12 |  |
| a1_independent_rows | (3, -, 4) | 12 | 12 | 12 | 1.000 | 1.000 | eq 12 |  |
| a1_independent_rows | (2, -, inf) | 36 | 12 | 36 | 0.333 | 1.000 | eq 36 |  |
| a1_independent_rows | (inf, -, 3) | infeasible | 12 | unavailable | - | - | inf inf |  |
| a1_independent_rows | (inf, 9, inf) | 12 | 12 | 12 | 1.000 | 1.000 |  |  |
| a1_independent_rows | (inf, 4, inf) | 24 | 12 | 24 | 0.500 | 1.000 |  |  |
| a2_fixed_chain_F_split | (inf, -, inf) | 4 | 4 | 4 | 1.000 | 1.000 | eq 4 |  |
| a2_fixed_chain_F_split | (9, -, inf) | 4 | 4 | 4 | 1.000 | 1.000 | eq 4 |  |
| a2_fixed_chain_F_split | (6, -, inf) | 8 | 4 | 8 | 0.500 | 1.000 | eq 8 |  |
| a2_fixed_chain_F_split | (3, -, inf) | 8 | 4 | 8 | 0.500 | 1.000 | eq 8 |  |
| a2_fixed_chain_F_split | (2, -, inf) | 18 | 4 | 20 | 0.222 | 1.111 | eq 18 |  |
| a2_fixed_chain_F_split | (inf, 4, inf) | 12 | 4 | 12 | 0.333 | 1.000 |  |  |
| a2_fixed_chain_F_split | (inf, 2, inf) | 22 | 4 | 24 | 0.182 | 1.091 |  |  |

### b. shared compact precursor

| circuit | (F, G, X) | I* | L | U (recompute=False) | L/I* | U/I* | hand | note |
|---|---|---|---|---|---|---|---|---|
| b1_compact_precursor_fanout | (inf, -, inf) | 4 | 4 | 4 | 1.000 | 1.000 | eq 4 |  |
| b1_compact_precursor_fanout | (17, -, 4) | 4 | 4 | unavailable | 1.000 | - | eq 4 | U_unavailable |
| b1_compact_precursor_fanout | (16, -, inf) | 10 | 4 | 12 | 0.400 | 1.200 | eq 10 |  |
| b1_compact_precursor_fanout | (inf, -, 3) | infeasible | 20 | unavailable | - | - | inf inf |  |
| b1_compact_precursor_fanout | (inf, 11, inf) | 14 | 4 | 16 | 0.286 | 1.143 |  |  |
| b1_compact_precursor_fanout | (inf, 5, inf) | 32 | 6 | 32 | 0.188 | 1.000 |  |  |

### c. independent accumulated precursors

| circuit | (F, G, X) | I* | L | U (recompute=False) | L/I* | U/I* | hand | note |
|---|---|---|---|---|---|---|---|---|
| c1_independent_precursors | (inf, -, inf) | 24 | 24 | 24 | 1.000 | 1.000 | eq 24 |  |
| c1_independent_precursors | (1, -, 4) | 24 | 24 | unavailable | 1.000 | - | eq 24 | U_unavailable |
| c1_independent_precursors | (inf, -, 3) | infeasible | 24 | unavailable | - | - | inf inf |  |
| c1_independent_precursors | (inf, 3, inf) | 24 | 24 | 24 | 1.000 | 1.000 |  |  |
| c1_independent_precursors | (inf, 1, inf) | 24 | 24 | unavailable | 1.000 | - |  | U_unavailable |
| c2_accumulated_rows_fixed_weights | (inf, -, inf) | 12 | 12 | 12 | 1.000 | 1.000 | eq 12 |  |
| c2_accumulated_rows_fixed_weights | (3, -, 4) | 12 | 12 | 12 | 1.000 | 1.000 | eq 12 |  |
| c2_accumulated_rows_fixed_weights | (inf, 9, inf) | 12 | 12 | 12 | 1.000 | 1.000 |  |  |
| c2_accumulated_rows_fixed_weights | (inf, 4, inf) | 24 | 12 | 24 | 0.500 | 1.000 |  |  |

### e. shared weights across branches

| circuit | (F, G, X) | I* | L | U (recompute=False) | L/I* | U/I* | hand | note |
|---|---|---|---|---|---|---|---|---|
| e1_shared_weights_fixed_acts | (inf, -, inf) | 8 | 8 | 8 | 1.000 | 1.000 | eq 8 |  |
| e1_shared_weights_fixed_acts | (3, -, 8) | 8 | 8 | 8 | 1.000 | 1.000 | eq 8 |  |
| e1_shared_weights_fixed_acts | (3, -, 4) | 8 | 8 | 8 | 1.000 | 1.000 | eq 8 |  |
| e1_shared_weights_fixed_acts | (inf, -, 3) | infeasible | 8 | unavailable | - | - | inf inf |  |
| e1_shared_weights_fixed_acts | (inf, 9, inf) | 8 | 8 | 8 | 1.000 | 1.000 |  |  |
| e1_shared_weights_fixed_acts | (inf, 4, inf) | 24 | 8 | unavailable | 0.333 | - |  | U_unavailable |
| e2_shared_weights_token_acts | (inf, -, inf) | 20 | 20 | 20 | 1.000 | 1.000 | eq 20 |  |
| e2_shared_weights_token_acts | (inf, -, 12) | 36 | 20 | unavailable | 0.556 | - | eq 36 | U_unavailable |
| e2_shared_weights_token_acts | (inf, -, 8) | 48 | 20 | unavailable | 0.417 | - | eq 48 | U_unavailable |
| e2_shared_weights_token_acts | (inf, 9, inf) | 32 | 20 | 32 | 0.625 | 1.000 |  |  |
| e2_shared_weights_token_acts | (inf, 4, inf) | 48 | 20 | unavailable | 0.417 | - |  | U_unavailable |

### f. LoRA compact state

| circuit | (F, G, X) | I* | L | U (recompute=False) | L/I* | U/I* | hand | note |
|---|---|---|---|---|---|---|---|---|
| f1_lora_fixed_x | (inf, -, inf) | 6 | 6 | 6 | 1.000 | 1.000 | eq 6 |  |
| f1_lora_fixed_x | (inf, -, 6) | 6 | 6 | 6 | 1.000 | 1.000 | eq 6 |  |
| f1_lora_fixed_x | (inf, -, 5) | 12 | 6 | 20 | 0.500 | 1.667 | eq 12 |  |
| f1_lora_fixed_x | (inf, -, 4) | 12 | 6 | 20 | 0.500 | 1.667 | eq 12 |  |
| f1_lora_fixed_x | (inf, -, 3) | infeasible | 11 | unavailable | - | - | inf inf |  |
| f1_lora_fixed_x | (4, -, inf) | 14 | 6 | 14 | 0.429 | 1.000 | eq 14 |  |
| f1_lora_fixed_x | (6, -, inf) | 10 | 6 | 6 ILLEGAL | 0.600 | 0.600 | eq 10 | Ulit_illegal, Ulit_lt_Istar |
| f1_lora_fixed_x | (9, -, inf) | 6 | 6 | 6 | 1.000 | 1.000 | eq 6 |  |
| f1_lora_fixed_x | (inf, 9, inf) | 12 | 6 | 12 | 0.500 | 1.000 |  |  |
| f1_lora_fixed_x | (inf, 4, inf) | 18 | 6 | 38 | 0.333 | 2.111 |  |  |
| f2_lora_token_x | (inf, -, inf) | 14 | 14 | 14 | 1.000 | 1.000 | eq 14 |  |
| f2_lora_token_x | (inf, -, 14) | 14 | 14 | 14 | 1.000 | 1.000 | eq 14 |  |
| f2_lora_token_x | (inf, -, 8) | 28 | 14 | 30 | 0.500 | 1.071 |  |  |
| f2_lora_token_x | (inf, 9, inf) | 20 | 14 | 20 | 0.700 | 1.000 |  |  |
| f2_lora_token_x | (inf, 4, inf) | 34 | 14 | 46 | 0.412 | 1.353 |  |  |

### g. residual / copy / embedding

| circuit | (F, G, X) | I* | L | U (recompute=False) | L/I* | U/I* | hand | note |
|---|---|---|---|---|---|---|---|---|
| g1_residual_add | (inf, -, inf) | 12 | 12 | 12 | 1.000 | 1.000 | eq 12 |  |
| g1_residual_add | (inf, -, 12) | 12 | 12 | 12 | 1.000 | 1.000 | eq 12 |  |
| g1_residual_add | (inf, -, 8) | 16 | 12 | 24 | 0.750 | 1.500 | eq 16 |  |
| g1_residual_add | (inf, -, 7) | infeasible | 12 | unavailable | - | - | inf inf |  |
| g1_residual_add | (inf, 4, inf) | 16 | 12 | 20 | 0.750 | 1.250 |  |  |
| g1_residual_add | (inf, 2, inf) | 28 | 12 | 32 | 0.429 | 1.143 |  |  |
| g2_embedding_gather | (inf, -, inf) | 20 | 20 | 20 | 1.000 | 1.000 | eq 20 |  |
| g2_embedding_gather | (inf, -, 16) | 28 | 20 | 28 | 0.714 | 1.000 | eq 28 |  |
| g2_embedding_gather | (inf, -, 8) | 48 | 20 | 48 | 0.417 | 1.000 | eq 48 |  |
| g2_embedding_gather | (inf, -, 7) | infeasible | 20 | unavailable | - | - | inf inf |  |
| g2_embedding_gather | (inf, 3, inf) | 32 | 20 | 32 | 0.625 | 1.000 |  |  |
| g2_embedding_gather | (inf, 1, inf) | 48 | 20 | 48 | 0.417 | 1.000 |  |  |
| g3_embedding_then_matmul | (inf, -, inf) | 28 | 28 | 28 | 1.000 | 1.000 | eq 28 |  |
| g3_embedding_then_matmul | (inf, -, 16) | 56 | 28 | 56 | 0.500 | 1.000 | ge 28 |  |
| g3_embedding_then_matmul | (inf, 12, inf) | 44 | 28 | 48 | 0.636 | 1.091 |  |  |
| g3_embedding_then_matmul | (inf, 6, inf) | 56 | 28 | 56 | 0.500 | 1.000 |  |  |
| g4_branching_intermediate | (inf, -, inf) | 16 | 16 | 16 | 1.000 | 1.000 | eq 16 |  |
| g4_branching_intermediate | (inf, -, 8) | 24 | 16 | unavailable | 0.667 | - | eq 24 | U_unavailable |
| g4_branching_intermediate | (inf, 6, inf) | 20 | 16 | 20 | 0.800 | 1.000 |  |  |
| g4_branching_intermediate | (inf, 3, inf) | 24 | 16 | unavailable | 0.667 | - |  | U_unavailable |

### h. weight gradient / reduction

| circuit | (F, G, X) | I* | L | U (recompute=False) | L/I* | U/I* | hand | note |
|---|---|---|---|---|---|---|---|---|
| h1_weight_gradient_tt | (inf, -, inf) | 16 | 16 | 16 | 1.000 | 1.000 | eq 16 |  |
| h1_weight_gradient_tt | (inf, -, 16) | 16 | 16 | 16 | 1.000 | 1.000 | eq 16 |  |
| h1_weight_gradient_tt | (inf, -, 8) | 32 | 16 | 32 | 0.500 | 1.000 | eq 32 |  |
| h1_weight_gradient_tt | (inf, -, 7) | infeasible | 16 | unavailable | - | - | inf inf |  |
| h1_weight_gradient_tt | (inf, 6, inf) | 24 | 16 | 24 | 0.667 | 1.000 |  |  |
| h1_weight_gradient_tt | (inf, 3, inf) | 32 | 18 | 32 | 0.562 | 1.000 |  |  |
| h2_colsum | (inf, -, inf) | 12 | 12 | 12 | 1.000 | 1.000 | eq 12 |  |
| h2_colsum | (inf, -, 6) | 12 | 12 | 12 | 1.000 | 1.000 | eq 12 |  |
| h2_colsum | (inf, -, 5) | infeasible | 12 | unavailable | - | - | inf inf |  |
| h2_colsum | (inf, 7, inf) | 12 | 12 | 12 | 1.000 | 1.000 |  |  |
| h2_colsum | (inf, 3, inf) | 28 | 12 | unavailable | 0.429 | - |  | U_unavailable |
| h3_colsum_into_gain | (inf, -, inf) | 20 | 20 | 20 | 1.000 | 1.000 | eq 20 |  |
| h3_colsum_into_gain | (inf, -, 8) | 24 | 20 | 28 | 0.833 | 1.167 | eq 24 |  |
| h3_colsum_into_gain | (inf, 9, inf) | 20 | 20 | 24 | 1.000 | 1.200 |  |  |
| h3_colsum_into_gain | (inf, 4, inf) | 32 | 20 | unavailable | 0.625 | - |  | U_unavailable |
| h4_gram_matrix | (inf, -, inf) | 8 | 8 | 8 | 1.000 | 1.000 | eq 8 |  |
| h4_gram_matrix | (inf, -, 8) | 8 | 8 | 8 | 1.000 | 1.000 | eq 8 |  |
| h4_gram_matrix | (inf, -, 6) | infeasible | 8 | unavailable | - | - | inf inf |  |
| h4_gram_matrix | (inf, 6, inf) | 16 | 8 | 24 | 0.500 | 1.500 |  |  |
| h4_gram_matrix | (inf, 3, inf) | 24 | 8 | 32 | 0.333 | 1.333 |  |  |

### t. torture (overlapping reads, epilogues)

| circuit | (F, G, X) | I* | L | U (recompute=False) | L/I* | U/I* | hand | note |
|---|---|---|---|---|---|---|---|---|
| t1_same_row_read_twice | (inf, -, inf) | 16 | 16 | 12 | 1.000 | 0.750 | eq 12 MISMATCH | Ulit_lt_Istar, Urec_lt_L, Ulit_lt_L |
| t1_same_row_read_twice | (inf, -, 8) | 16 | 16 | unavailable | 1.000 | - | eq 16 | U_unavailable |
| t1_same_row_read_twice | (inf, 3, inf) | 16 | 16 | unavailable | 1.000 | - |  | U_unavailable |
| t1_same_row_read_twice | (inf, 1, inf) | 32 | 16 | unavailable | 0.500 | - |  | U_unavailable |
| t2_repeated_row_view_B | (inf, -, inf) | 8 | 12 | 8 | 1.500 | 1.000 | eq 8 | L_gt_Istar, Lnoprog_gt_Istar, Urec_lt_L, Ulit_lt_L |
| t2_repeated_row_view_B | (inf, -, 8) | 8 | 12 | 8 | 1.500 | 1.000 | eq 8 | L_gt_Istar, Lnoprog_gt_Istar, Urec_lt_L, Ulit_lt_L |
| t2_repeated_row_view_B | (inf, 3, inf) | 16 | 12 | 16 | 0.750 | 1.000 |  |  |
| t2_repeated_row_view_B | (inf, 1, inf) | 32 | 16 | unavailable | 0.500 | - |  | U_unavailable |
| t3_repeated_row_view_A | (inf, -, inf) | 12 | 12 | 12 | 1.000 | 1.000 | eq 12 |  |
| t3_repeated_row_view_A | (inf, -, 6) | 12 | 14 | 32 | 1.167 | 2.667 | eq 12 | L_gt_Istar, Lnoprog_gt_Istar |
| t3_repeated_row_view_A | (inf, -, 4) | 14 | 20 | 40 | 1.429 | 2.857 | eq 14 | L_gt_Istar, Lnoprog_gt_Istar |
| t3_repeated_row_view_A | (inf, 12, inf) | 14 | 12 | 24 | 0.857 | 1.714 |  |  |
| t3_repeated_row_view_A | (inf, 6, inf) | 20 | 13 | 32 | 0.650 | 1.600 |  |  |
| t4_joint_epilogue_F | (inf, -, inf) | 30 | 30 | 30 | 1.000 | 1.000 | eq 30 |  |
| t4_joint_epilogue_F | (9, -, inf) | 30 | 30 | 30 | 1.000 | 1.000 | eq 30 |  |
| t4_joint_epilogue_F | (8, -, inf) | 38 | 30 | 30 ILLEGAL | 0.789 | 0.789 | ge 31 | Ulit_illegal, Ulit_lt_Istar |
| t4_joint_epilogue_F | (5, -, inf) | 38 | 30 | 30 ILLEGAL | 0.789 | 0.789 | ge 31 | Ulit_illegal, Ulit_lt_Istar |
| t4_joint_epilogue_F | (inf, -, 12) | 56 | 30 | unavailable | 0.536 | - |  | U_unavailable |
| t4_joint_epilogue_F | (inf, 9, inf) | 36 | 30 | 36 | 0.833 | 1.000 |  |  |
| t4_joint_epilogue_F | (inf, 4, inf) | 56 | 30 | 76 | 0.536 | 1.357 |  |  |
| t5_rowfull_epilogue_F | (inf, -, inf) | 16 | 16 | 16 | 1.000 | 1.000 | eq 16 |  |
| t5_rowfull_epilogue_F | (13, -, inf) | 16 | 16 | 16 | 1.000 | 1.000 | eq 16 |  |
| t5_rowfull_epilogue_F | (12, -, inf) | 20 | 16 | 16 ILLEGAL | 0.800 | 0.800 | eq 20 | Ulit_illegal, Ulit_lt_Istar |
| t5_rowfull_epilogue_F | (9, -, inf) | 20 | 16 | 16 ILLEGAL | 0.800 | 0.800 | eq 20 | Ulit_illegal, Ulit_lt_Istar |
| t5_rowfull_epilogue_F | (inf, 7, inf) | 24 | 16 | unavailable | 0.667 | - |  | U_unavailable |
| t5_rowfull_epilogue_F | (inf, 3, inf) | 38 | 16 | unavailable | 0.421 | - |  | U_unavailable |
| t6_softmax_epilogue_F | (inf, -, inf) | 12 | 12 | 12 | 1.000 | 1.000 | eq 12 |  |
| t6_softmax_epilogue_F | (20, -, inf) | 12 | 12 | 12 | 1.000 | 1.000 | eq 12 |  |
| t6_softmax_epilogue_F | (16, -, inf) | 16 | 12 | 12 ILLEGAL | 0.750 | 0.750 | eq 16 | Ulit_illegal, Ulit_lt_Istar |
| t6_softmax_epilogue_F | (12, -, inf) | 22 | 12 | 12 ILLEGAL | 0.545 | 0.545 | eq 22 | Ulit_illegal, Ulit_lt_Istar |
| t6_softmax_epilogue_F | (inf, 10, inf) | 28 | 12 | unavailable | 0.429 | - |  | U_unavailable |
| t6_softmax_epilogue_F | (inf, 5, inf) | 40 | 12 | unavailable | 0.300 | - |  | U_unavailable |
| t7_single_leaf_overlap_share | (inf, -, inf) | 10 | 10 | 10 | 1.000 | 1.000 | eq 10 |  |
| t7_single_leaf_overlap_share | (inf, -, 6) | 12 | 10 | 12 | 0.833 | 1.000 | eq 12 |  |
| t7_single_leaf_overlap_share | (inf, 4, inf) | 12 | 10 | 12 | 0.833 | 1.000 |  |  |
| t7_single_leaf_overlap_share | (inf, 2, inf) | 16 | 10 | 16 | 0.625 | 1.000 |  |  |
| t9_scalar_read_by_two_matmuls | (inf, -, inf) | 2 | 2 | 2 | 1.000 | 1.000 | eq 2 |  |
| t9_scalar_read_by_two_matmuls | (2, -, 2) | 2 | 2 | 2 | 1.000 | 1.000 | eq 2 |  |
| t9_scalar_read_by_two_matmuls | (inf, 2, inf) | 4 | 2 | unavailable | 0.500 | - |  | U_unavailable |
| t9_scalar_read_by_two_matmuls | (inf, 1, inf) | 12 | 2 | unavailable | 0.167 | - |  | U_unavailable |
| t10_last_leaf_overlap | (inf, -, inf) | 6 | 6 | 6 | 1.000 | 1.000 | eq 6 |  |
| t10_last_leaf_overlap | (inf, -, 6) | 6 | 6 | 6 | 1.000 | 1.000 | eq 6 |  |
| t10_last_leaf_overlap | (inf, 4, inf) | 6 | 6 | 8 | 1.000 | 1.333 |  |  |
| t10_last_leaf_overlap | (inf, 2, inf) | 8 | 6 | 8 | 0.750 | 1.000 |  |  |
| t11_reprow_B_deep | (inf, -, inf) | 10 | 16 | 10 | 1.600 | 1.000 | eq 10 | L_gt_Istar, Lnoprog_gt_Istar, Urec_lt_L, Ulit_lt_L |
| t11_reprow_B_deep | (inf, -, 10) | 10 | 16 | 10 | 1.600 | 1.000 | eq 10 | L_gt_Istar, Lnoprog_gt_Istar, Urec_lt_L, Ulit_lt_L |
| t11_reprow_B_deep | (inf, -, 8) | 12 | 16 | 12 | 1.333 | 1.000 | eq 12 | L_gt_Istar, Lnoprog_gt_Istar, Urec_lt_L, Ulit_lt_L |
| t11_reprow_B_deep | (inf, 8, inf) | 20 | 16 | 20 | 0.800 | 1.000 |  |  |
| t11_reprow_B_deep | (inf, 4, inf) | 28 | 16 | 32 | 0.571 | 1.143 |  |  |
| t8_add_same_tensor_twice | (inf, -, inf) | 8 | 8 | 8 | 1.000 | 1.000 | eq 8 |  |
| t8_add_same_tensor_twice | (inf, -, 8) | 8 | 8 | 8 | 1.000 | 1.000 | eq 8 |  |
| t8_add_same_tensor_twice | (inf, -, 2) | 8 | 8 | unavailable | 1.000 | - | eq 8 | U_unavailable |
| t8_add_same_tensor_twice | (inf, 2, inf) | 8 | 8 | 16 | 1.000 | 2.000 |  |  |
| t8_add_same_tensor_twice | (inf, 1, inf) | 8 | 8 | unavailable | 1.000 | - |  | U_unavailable |
| t12_attn_decode_kv_undercount | (inf, -, inf) | 20 | 20 | 20 | 1.000 | 1.000 | eq 20 |  |
| t12_attn_decode_kv_undercount | (inf, -, 16) | 24 | 20 | 24 | 0.833 | 1.000 | eq 24 |  |
| t12_attn_decode_kv_undercount | (inf, -, 12) | 24 | 20 | 24 | 0.833 | 1.000 | eq 24 |  |
| t12_attn_decode_kv_undercount | (20, -, inf) | 24 | 20 | 24 | 0.833 | 1.000 | eq 24 |  |
| t12_attn_decode_kv_undercount | (inf, 13, inf) | 34 | 20 | unavailable | 0.588 | - |  | U_unavailable |
| t12_attn_decode_kv_undercount | (inf, 6, inf) | 48 | 20 | unavailable | 0.417 | - |  | U_unavailable |

## Attacks the certificate misses (L/I* < 0.5)

For every such instance: the family, the policy, and the exact optimal partition -- the RUs, their work, their charged input and which values they import.  This is *what the adversary does* that `L` does not charge for.

438 instances with L/I* < 0.5 (families: colsum: 48, matmul|shared: 42, matmul: 32, matmul_tt|shared: 25, rmsnorm: 17, matmul|shared,view: 14, add+matmul|shared: 14, matmul|view: 13, gain+matmul|shared,chain: 12, matmul+rowscale|shared,view,chain: 10, matmul|shared,view,chain: 10, gain+matmul_tt|shared: 9, colsum+matmul|shared,view: 8, add+matmul+matmul_tt+rowscale|chain: 6, colsum+matmul+mul+swiglu|shared,view: 6, add+gain+matmul|shared,view,chain: 6, mul+rmsnorm|shared,chain: 6, add+matmul+matmul_tt|shared,view,chain: 6, colsum+gain|shared: 6, gain: 6, add+matmul|shared,chain: 6, add+embed+matmul_tt+rowscale|shared,view,chain: 6, gain+matmul+swiglu|shared,chain: 6, colsum+mul|shared,view: 6, matmul+rmsnorm|shared,chain: 6, colsum+mul|shared: 6, matmul_tt: 5, i: 5, t: 5, colsum+matmul+mul|shared,chain: 4, colsum+mul|chain: 4, a: 4, softmax: 4, embed+gain+matmul|shared,view,chain: 4, add+matmul+mul|shared: 4, colsum+matmul|shared: 4, add+colsum: 4, add+matmul|shared,view,chain: 4, embed: 4, add+mul+softmax|shared,view,chain: 4, colsum+matmul_tt|shared: 4, colsum+mul+rowscale+swiglu|shared,chain: 4, add+embed+gain+swiglu|shared,chain: 4, matmul|shared,chain: 4, embed+matmul|view,chain: 4, b: 3, e: 3, f: 3, embed+matmul|shared,chain: 3, add+matmul+swiglu|shared,chain: 3, g: 3, h: 2, r: 2, matmul_tt|view: 2, matmul_tt+mul|shared,view: 2, d: 1).

### s22c0  [fuzz; family `matmul_tt|shared`]  policy (F, G, X) = (17, 1, 12)
L = 6  I* = 66  U = unavailable  L/I* = 0.091; gates 27, work 18; optimal partition has 18 RUs:
  - 3 RUs (RU0, RU8, ...): work 1, input 2 B; ops [op0:matmul x2]; imports p0[accumulated]=2B
  - 9 RUs (RU1, RU3, ...): work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
  - 6 RUs (RU2, RU4, ...): work 1, input 4 B; ops [op0:matmul x2]; imports p0[accumulated]=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (1, 3), 'accumulated'),
    ],
    ops=[
        Op('matmul_tt', (('p', 'p0'), ('p', 'p0'))),  # 0
    ],
    name='s22c0',
)
~~~

### s22c3  [fuzz; family `matmul|shared`]  policy (F, G, X) = (2, 1, 10)
L = 12  I* = 132  U = unavailable  L/I* = 0.091; gates 36, work 27; optimal partition has 27 RUs:
  - 3 RUs (RU0, RU12, ...): work 1, input 2 B; ops [op0:matmul x2]; imports p0[accumulated]=2B
  - 3 RUs (RU1, RU13, ...): work 1, input 6 B; ops [op0:matmul x1]; imports op0:matmul.out=4B, p0[accumulated]=2B
  - 9 RUs (RU2, RU5, ...): work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
  - 6 RUs (RU3, RU6, ...): work 1, input 4 B; ops [op0:matmul x2]; imports p0[accumulated]=4B
  - 6 RUs (RU4, RU7, ...): work 1, input 8 B; ops [op0:matmul x1]; imports op0:matmul.out=4B, p0[accumulated]=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (3, 2), 'accumulated'),
    ],
    ops=[
        Op('matmul', (('p', 'p0'), ('p', 'p0'))),  # 0
    ],
    name='s22c3',
)
~~~

### s22c137  [fuzz; family `colsum+matmul+mul|shared,chain`]  policy (F, G, X) = (13, 1, 12)
L = 8  I* = 84  U = unavailable  L/I* = 0.095; gates 30, work 26; optimal partition has 26 RUs:
  - 6 RUs (RU0, RU1, ...): work 1, input 0 B; ops [op0:mul x1]; imports nothing charged
  - 3 RUs (RU6, RU11, ...): work 1, input 4 B; ops [op1:colsum x2]; imports op1:colsum.out=4B
  - 6 RUs (RU7, RU8, ...): work 1, input 2 B; ops [op1:colsum x1]; imports op0:mul.out=2B
  - 3 RUs (RU9, RU14, ...): work 1, input 8 B; ops [op1:colsum x1]; imports op1:colsum.out=8B
  - 3 RUs (RU10, RU15, ...): work 1, input 4 B; ops [op1:colsum x1]; imports op1:colsum.out=4B
  - RU21: work 1, input 2 B; ops [op2:matmul x2]; imports p1[token]=2B
  - 3 RUs (RU22, RU23, ...): work 1, input 6 B; ops [op2:matmul x1]; imports op2:matmul.out=4B, p1[token]=2B
  - RU25: work 1, input 4 B; ops [op2:matmul x1]; imports op2:matmul.out=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (2, 3), 'fixed'),
        Param('p1', (1, 4), 'token'),
    ],
    ops=[
        Op('mul', (('p', 'p0'), ('p', 'p0'))),  # 0
        Op('colsum', (('o', 0),)),  # 1
        Op('matmul', (('p', 'p1'), ('p', 'p1'))),  # 2
    ],
    name='s22c137',
)
~~~

### s22c89  [fuzz; family `gain+matmul_tt|shared`]  policy (F, G, X) = (3, 2, 8)
L = 2  I* = 20  U = 36  L/I* = 0.100; gates 23, work 19; optimal partition has 10 RUs:
  - 4 RUs (RU0, RU2, ...): work 2, input 0 B; ops [op0:matmul x3]; imports nothing charged
  - 4 RUs (RU1, RU3, ...): work 2, input 4 B; ops [op0:matmul x2]; imports op0:matmul.out=4B
  - RU8: work 2, input 2 B; ops [op1:gain x2]; imports p2[accumulated]=2B
  - RU9: work 1, input 2 B; ops [op1:gain x1]; imports p2[accumulated]=2B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (3, 1), 'fixed'),
        Param('p1', (3, 4), 'fixed'),
        Param('p2', (1,), 'accumulated'),
    ],
    ops=[
        Op('matmul_tt', (('p', 'p0'), ('p', 'p1'))),  # 0
        Op('gain', (('p', 'p0'), ('p', 'p2'))),  # 1
    ],
    name='s22c89',
)
~~~

### s22c213  [fuzz; family `matmul|shared,view`]  policy (F, G, X) = (1, -, 10)
L = 6  I* = 58  U = unavailable  L/I* = 0.103; gates 20, work 16; optimal partition has 6 RUs:
  - RU0: work 6, input 10 B; ops [op0:matmul x10]; imports op0:matmul.out=8B, p0[token]=2B
  - 4 RUs (RU1, RU2, ...): work 2, input 10 B; ops [op0:matmul x2]; imports op0:matmul.out=8B, p0[token]=2B
  - RU5: work 2, input 8 B; ops [op0:matmul x2]; imports op0:matmul.out=8B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (1, 3), 'token'),
    ],
    ops=[
        Op('matmul', (('p', 'p0'), ('p', 'p0', 'reprow', 0, 4))),  # 0
    ],
    name='s22c213',
)
~~~

### s21c99  [fuzz; family `add+matmul+matmul_tt+rowscale|chain`]  policy (F, G, X) = (10, 5, 4)
L = 2  I* = 16  U = 16  L/I* = 0.125; gates 36, work 28; optimal partition has 8 RUs:
  - RU0, RU2: work 5, input 4 B; ops [op0:matmul x4, op1:rowscale x2]; imports op0:matmul.out=2B, p2[token]=2B
  - RU1, RU3: work 3, input 0 B; ops [op0:matmul x4]; imports nothing charged
  - 4 RUs (RU4, RU5, ...): work 3, input 2 B; ops [op2:add x1, op3:matmul x3]; imports op1:rowscale.out=2B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (1, 2), 'fixed'),
        Param('p1', (4, 2), 'fixed'),
        Param('p2', (1,), 'token'),
        Param('p3', (1, 4), 'fixed'),
        Param('p4', (1, 1), 'fixed'),
    ],
    ops=[
        Op('matmul', (('p', 'p0'), ('p', 'p1'))),  # 0
        Op('rowscale', (('o', 0), ('p', 'p2'))),  # 1
        Op('add', (('o', 1), ('p', 'p3'))),  # 2
        Op('matmul_tt', (('p', 'p4'), ('o', 2))),  # 3
    ],
    name='s21c99',
)
~~~

### s22c13  [fuzz; family `colsum+matmul|shared,view`]  policy (F, G, X) = (13, 4, 12)
L = 2  I* = 16  U = unavailable  L/I* = 0.125; gates 16, work 14; optimal partition has 4 RUs:
  - RU0: work 4, input 0 B; ops [op0:matmul x5]; imports nothing charged
  - RU1: work 2, input 8 B; ops [op0:matmul x1, op1:colsum x1]; imports op0:matmul.out=4B, op1:colsum.out=4B
  - RU2: work 4, input 2 B; ops [op1:colsum x5]; imports p1[token]=2B
  - RU3: work 4, input 6 B; ops [op1:colsum x4]; imports op1:colsum.out=4B, p1[token]=2B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (1, 4), 'fixed'),
        Param('p1', (1, 1), 'token'),
    ],
    ops=[
        Op('matmul', (('p', 'p0'), ('p', 'p0'))),  # 0
        Op('colsum', (('p', 'p1', 'reprow', 0, 4),)),  # 1
    ],
    name='s22c13',
)
~~~

### s22c217  [fuzz; family `matmul_tt`]  policy (F, G, X) = (1, -, 6)
L = 8  I* = 64  U = unavailable  L/I* = 0.125; gates 24, work 18; optimal partition has 12 RUs:
  - RU0, RU3: work 4, input 6 B; ops [op0:matmul x7]; imports op0:matmul.out=4B, p0[accumulated]=2B
  - 6 RUs (RU1, RU2, ...): work 1, input 6 B; ops [op0:matmul x1]; imports op0:matmul.out=4B, p0[accumulated]=2B
  - 4 RUs (RU5, RU7, ...): work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (2, 2), 'accumulated'),
        Param('p1', (2, 3), 'fixed'),
    ],
    ops=[
        Op('matmul_tt', (('p', 'p0'), ('p', 'p1'))),  # 0
    ],
    name='s22c217',
)
~~~

### s22c135  [fuzz; family `matmul|view`]  policy (F, G, X) = (1, -, 10)
L = 4  I* = 30  U = unavailable  L/I* = 0.133; gates 12, work 9; optimal partition has 4 RUs:
  - RU0: work 5, input 10 B; ops [op0:matmul x8]; imports op0:matmul.out=8B, p1[token]=2B
  - RU1: work 2, input 10 B; ops [op0:matmul x2]; imports op0:matmul.out=8B, p1[token]=2B
  - RU2: work 1, input 6 B; ops [op0:matmul x1]; imports op0:matmul.out=4B, p1[token]=2B
  - RU3: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (3, 2), 'fixed'),
        Param('p1', (1, 2), 'token'),
    ],
    ops=[
        Op('matmul', (('p', 'p0'), ('p', 'p1', 'reprow', 0, 1))),  # 0
    ],
    name='s22c135',
)
~~~

### i1_inference_mlp  [torture; family `i`]  policy (F, G, X) = (4, 6, inf)
L = 4  I* = 28  U = 28  L/I* = 0.143; gates 28, work 24; optimal partition has 4 RUs:
  - RU0: work 6, input 4 B; ops [op0:matmul x8]; imports x[token]=4B
  - RU1: work 6, input 4 B; ops [op1:matmul x8]; imports x[token]=4B
  - RU2: work 6, input 16 B; ops [op2:swiglu x2, op4:add x2]; imports op0:matmul.out=4B, op1:matmul.out=4B, op3:matmul.out=4B, x[token]=4B
  - RU3: work 6, input 4 B; ops [op3:matmul x8]; imports op2:swiglu.out=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('x', (1, 2), 'token'),
        Param('W1', (2, 2), 'fixed'),
        Param('W3', (2, 2), 'fixed'),
        Param('W2', (2, 2), 'fixed'),
    ],
    ops=[
        Op('matmul', (('p', 'x'), ('p', 'W1'))),  # 0
        Op('matmul', (('p', 'x'), ('p', 'W3'))),  # 1
        Op('swiglu', (('o', 0), ('o', 1))),  # 2
        Op('matmul', (('o', 2), ('p', 'W2'))),  # 3
        Op('add', (('p', 'x'), ('o', 3))),  # 4
    ],
    name='i1_inference_mlp',
)
~~~

### s22c200  [fuzz; family `colsum+matmul|shared,view`]  policy (F, G, X) = (1, 1, 22)
L = 12  I* = 84  U = unavailable  L/I* = 0.143; gates 28, work 23; optimal partition has 23 RUs:
  - 3 RUs (RU0, RU3, ...): work 1, input 0 B; ops [op0:matmul x2]; imports nothing charged
  - 6 RUs (RU1, RU2, ...): work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
  - RU9, RU16: work 1, input 4 B; ops [op1:colsum x2]; imports op1:colsum.out=4B
  - 6 RUs (RU10, RU11, ...): work 1, input 2 B; ops [op1:colsum x1]; imports p1[accumulated]=2B
  - 4 RUs (RU12, RU14, ...): work 1, input 8 B; ops [op1:colsum x1]; imports op1:colsum.out=8B
  - RU15, RU22: work 1, input 4 B; ops [op1:colsum x1]; imports op1:colsum.out=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (3, 2), 'fixed'),
        Param('p1', (3, 2), 'accumulated'),
    ],
    ops=[
        Op('matmul', (('p', 'p0'), ('p', 'p0', 'rows', 2, 3))),  # 0
        Op('colsum', (('p', 'p1'),)),  # 1
    ],
    name='s22c200',
)
~~~

### s22c45  [fuzz; family `matmul`]  policy (F, G, X) = (1, -, 14)
L = 8  I* = 56  U = unavailable  L/I* = 0.143; gates 18, work 15; optimal partition has 4 RUs:
  - RU0: work 6, input 14 B; ops [op0:matmul x9]; imports op0:matmul.out=12B, p1[carried]=2B
  - 3 RUs (RU1, RU2, ...): work 3, input 14 B; ops [op0:matmul x3]; imports op0:matmul.out=12B, p1[carried]=2B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (3, 4), 'fixed'),
        Param('p1', (1, 4), 'carried'),
    ],
    ops=[
        Op('matmul', (('p', 'p0'), ('p', 'p1'))),  # 0
    ],
    name='s22c45',
)
~~~

### s22c93  [fuzz; family `matmul+rowscale|shared,view,chain`]  policy (F, G, X) = (1, -, 8)
L = 4  I* = 28  U = unavailable  L/I* = 0.143; gates 16, work 12; optimal partition has 4 RUs:
  - RU0: work 6, input 8 B; ops [op0:matmul x8, op1:rowscale x2]; imports op0:matmul.out=4B, p0[carried]=4B
  - RU1, RU2: work 2, input 8 B; ops [op0:matmul x2]; imports op0:matmul.out=8B
  - RU3: work 2, input 4 B; ops [op1:rowscale x2]; imports op0:matmul.out=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (2, 1), 'carried'),
        Param('p1', (2,), 'fixed'),
    ],
    ops=[
        Op('matmul', (('p', 'p0', 'reprow', 0, 2), ('p', 'p0'))),  # 0
        Op('rowscale', (('o', 0), ('p', 'p1'))),  # 1
    ],
    name='s22c93',
)
~~~

### s22c18  [fuzz; family `colsum+mul|chain`]  policy (F, G, X) = (10, 1, 20)
L = 8  I* = 48  U = unavailable  L/I* = 0.167; gates 24, work 20; optimal partition has 20 RUs:
  - 4 RUs (RU0, RU1, ...): work 1, input 2 B; ops [op0:mul x1]; imports p2[token]=2B
  - 4 RUs (RU4, RU7, ...): work 1, input 4 B; ops [op1:colsum x2]; imports op1:colsum.out=4B
  - 4 RUs (RU5, RU8, ...): work 1, input 2 B; ops [op1:colsum x1]; imports op0:mul.out=2B
  - 4 RUs (RU6, RU9, ...): work 1, input 4 B; ops [op1:colsum x1]; imports op1:colsum.out=4B
  - 4 RUs (RU16, RU17, ...): work 1, input 0 B; ops [op2:mul x1]; imports nothing charged

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (1, 4), 'fixed'),
        Param('p2', (1, 4), 'token'),
        Param('p3', (1, 4), 'fixed'),
        Param('p4', (1, 4), 'fixed'),
    ],
    ops=[
        Op('mul', (('p', 'p0'), ('p', 'p2'))),  # 0
        Op('colsum', (('o', 0),)),  # 1
        Op('mul', (('p', 'p3'), ('p', 'p4'))),  # 2
    ],
    name='s22c18',
)
~~~

### t9_scalar_read_by_two_matmuls  [torture; family `t`]  policy (F, G, X) = (inf, 1, inf)
L = 2  I* = 12  U = unavailable  L/I* = 0.167; gates 6, work 4; optimal partition has 4 RUs:
  - RU0: work 1, input 2 B; ops [op0:matmul x2]; imports p[accumulated]=2B
  - RU1: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
  - RU2: work 1, input 2 B; ops [op1:matmul x2]; imports p[accumulated]=2B
  - RU3: work 1, input 4 B; ops [op1:matmul x1]; imports op1:matmul.out=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p', (1, 1), 'accumulated'),
        Param('Wf1', (1, 1), 'fixed'),
        Param('Wf2', (1, 1), 'fixed'),
    ],
    ops=[
        Op('matmul', (('p', 'p'), ('p', 'Wf1'))),  # 0
        Op('matmul', (('p', 'p'), ('p', 'Wf2'))),  # 1
    ],
    name='t9_scalar_read_by_two_matmuls',
)
# OpGraph.share = {'p': 2} (true per-element multiplicity of the shared leaf = 2)
~~~

### a2_fixed_chain_F_split  [torture; family `a`]  policy (F, G, X) = (inf, 2, inf)
L = 4  I* = 22  U = 24  L/I* = 0.182; gates 12, work 9; optimal partition has 5 RUs:
  - RU0, RU2: work 2, input 4 B; ops [op0:matmul x3]; imports x[token]=4B
  - RU1: work 2, input 4 B; ops [op0:matmul x1, op1:matmul x2]; imports op0:matmul.out=4B
  - RU3: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
  - RU4: work 2, input 6 B; ops [op1:matmul x2]; imports op0:matmul.out=2B, op1:matmul.out=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('x', (1, 2), 'token'),
        Param('Wf1', (2, 2), 'fixed'),
        Param('Wf2', (1, 2), 'fixed'),
    ],
    ops=[
        Op('matmul', (('p', 'x'), ('p', 'Wf1'))),  # 0
        Op('matmul', (('o', 0), ('p', 'Wf2'))),  # 1
    ],
    name='a2_fixed_chain_F_split',
)
~~~

### s22c10  [fuzz; family `add+matmul|shared`]  policy (F, G, X) = (7, 1, 24)
L = 22  I* = 118  U = unavailable  L/I* = 0.186; gates 27, work 23; optimal partition has 23 RUs:
  - RU0, RU15: work 1, input 2 B; ops [op0:matmul x2]; imports p0[token]=2B
  - 6 RUs (RU1, RU2, ...): work 1, input 6 B; ops [op0:matmul x1]; imports op0:matmul.out=4B, p0[token]=2B
  - 4 RUs (RU4, RU9, ...): work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
  - RU5, RU10: work 1, input 4 B; ops [op0:matmul x2]; imports p0[token]=4B
  - 6 RUs (RU6, RU7, ...): work 1, input 8 B; ops [op0:matmul x1]; imports op0:matmul.out=4B, p0[token]=4B
  - 3 RUs (RU20, RU21, ...): work 1, input 2 B; ops [op1:add x1]; imports p1[token]=2B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (2, 4), 'token'),
        Param('p1', (3, 1), 'token'),
    ],
    ops=[
        Op('matmul', (('p', 'p0'), ('p', 'p0'))),  # 0
        Op('add', (('p', 'p1'), ('p', 'p1'))),  # 1
    ],
    name='s22c10',
)
~~~

### b1_compact_precursor_fanout  [torture; family `b`]  policy (F, G, X) = (inf, 5, inf)
L = 6  I* = 32  U = 32  L/I* = 0.188; gates 28, work 22; optimal partition has 5 RUs:
  - RU0: work 5, input 4 B; ops [op0:matmul x7]; imports h[accumulated]=4B
  - RU1: work 4, input 8 B; ops [op0:matmul x5]; imports h[accumulated]=4B, op0:matmul.out=4B
  - RU2: work 3, input 4 B; ops [op0:matmul x4]; imports h[accumulated]=4B
  - RU3, RU4: work 5, input 8 B; ops [op1:matmul x6]; imports op0:matmul.out=8B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('h', (1, 2), 'accumulated'),
        Param('Wf', (4, 2), 'fixed'),
        Param('Wf2', (2, 4), 'fixed'),
    ],
    ops=[
        Op('matmul', (('p', 'h'), ('p', 'Wf'))),  # 0
        Op('matmul', (('o', 0), ('p', 'Wf2'))),  # 1
    ],
    name='b1_compact_precursor_fanout',
)
~~~

### s21c72  [fuzz; family `matmul|shared`]  policy (F, G, X) = (1, -, 16)
L = 8  I* = 42  U = unavailable  L/I* = 0.190; gates 16, work 12; optimal partition has 4 RUs:
  - RU0: work 7, input 16 B; ops [op0:matmul x11]; imports op0:matmul.out=12B, p0[accumulated]=4B
  - RU1: work 3, input 16 B; ops [op0:matmul x3]; imports op0:matmul.out=12B, p0[accumulated]=4B
  - RU2: work 1, input 6 B; ops [op0:matmul x1]; imports op0:matmul.out=4B, p0[accumulated]=2B
  - RU3: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (2, 2), 'accumulated'),
    ],
    ops=[
        Op('matmul', (('p', 'p0'), ('p', 'p0'))),  # 0
    ],
    name='s21c72',
)
~~~

### s21c55  [fuzz; family `matmul|shared,view,chain`]  policy (F, G, X) = (6, 3, 4)
L = 4  I* = 20  U = unavailable  L/I* = 0.200; gates 24, work 17; optimal partition has 7 RUs:
  - RU0: work 3, input 4 B; ops [op0:matmul x4]; imports p0[accumulated]=4B
  - 4 RUs (RU1, RU2, ...): work 2, input 2 B; ops [op1:matmul x3]; imports op0:matmul.out=2B
  - RU5, RU6: work 3, input 4 B; ops [op2:matmul x4]; imports p0[accumulated]=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (1, 2), 'accumulated'),
        Param('p1', (4, 1), 'fixed'),
    ],
    ops=[
        Op('matmul', (('p', 'p0'), ('p', 'p0'))),  # 0
        Op('matmul', (('o', 0), ('p', 'p1'))),  # 1
        Op('matmul', (('p', 'p0'), ('p', 'p0', 'reprow', 0, 2))),  # 2
    ],
    name='s21c55',
)
~~~

### s22c177  [fuzz; family `colsum`]  policy (F, G, X) = (5, 1, 10)
L = 4  I* = 20  U = unavailable  L/I* = 0.200; gates 6, work 5; optimal partition has 5 RUs:
  - RU0: work 1, input 4 B; ops [op0:colsum x2]; imports op0:colsum.out=4B
  - RU1, RU2: work 1, input 2 B; ops [op0:colsum x1]; imports p0[token]=2B
  - RU3: work 1, input 8 B; ops [op0:colsum x1]; imports op0:colsum.out=8B
  - RU4: work 1, input 4 B; ops [op0:colsum x1]; imports op0:colsum.out=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (2, 1), 'token'),
    ],
    ops=[
        Op('colsum', (('p', 'p0'),)),  # 0
    ],
    name='s22c177',
)
~~~

### s22c189  [fuzz; family `softmax`]  policy (F, G, X) = (4, -, 20)
L = 8  I* = 40  U = unavailable  L/I* = 0.200; gates 28, work 30; optimal partition has 2 RUs:
  - RU0, RU1: work 15, input 20 B; ops [op0:softmax x14]; imports op0:softmax.out=12B, p0[accumulated]=8B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (2, 2), 'accumulated'),
    ],
    ops=[
        Op('softmax', (('p', 'p0'),)),  # 0
    ],
    name='s22c189',
)
~~~

### s22c211  [fuzz; family `colsum`]  policy (F, G, X) = (4, 1, 4)
L = 2  I* = 10  U = unavailable  L/I* = 0.200; gates 4, work 3; optimal partition has 3 RUs:
  - RU0: work 1, input 4 B; ops [op0:colsum x2]; imports op0:colsum.out=4B
  - RU1: work 1, input 2 B; ops [op0:colsum x1]; imports p0[accumulated]=2B
  - RU2: work 1, input 4 B; ops [op0:colsum x1]; imports op0:colsum.out=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (1, 1), 'accumulated'),
    ],
    ops=[
        Op('colsum', (('p', 'p0'),)),  # 0
    ],
    name='s22c211',
)
~~~

### s22c27  [fuzz; family `colsum`]  policy (F, G, X) = (2, 1, 4)
L = 4  I* = 20  U = unavailable  L/I* = 0.200; gates 8, work 6; optimal partition has 6 RUs:
  - RU0, RU3: work 1, input 4 B; ops [op0:colsum x2]; imports op0:colsum.out=4B
  - RU1, RU4: work 1, input 2 B; ops [op0:colsum x1]; imports p0[carried]=2B
  - RU2, RU5: work 1, input 4 B; ops [op0:colsum x1]; imports op0:colsum.out=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (1, 2), 'carried'),
    ],
    ops=[
        Op('colsum', (('p', 'p0'),)),  # 0
    ],
    name='s22c27',
)
~~~

### s22c40  [fuzz; family `colsum`]  policy (F, G, X) = (4, 1, 10)
L = 8  I* = 40  U = unavailable  L/I* = 0.200; gates 12, work 10; optimal partition has 10 RUs:
  - RU0, RU5: work 1, input 4 B; ops [op0:colsum x2]; imports op0:colsum.out=4B
  - 4 RUs (RU1, RU2, ...): work 1, input 2 B; ops [op0:colsum x1]; imports p0[accumulated]=2B
  - RU3, RU8: work 1, input 8 B; ops [op0:colsum x1]; imports op0:colsum.out=8B
  - RU4, RU9: work 1, input 4 B; ops [op0:colsum x1]; imports op0:colsum.out=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (2, 2), 'accumulated'),
    ],
    ops=[
        Op('colsum', (('p', 'p0'),)),  # 0
    ],
    name='s22c40',
)
~~~

### s21c139  [fuzz; family `colsum+matmul+mul+swiglu|shared,view`]  policy (F, G, X) = (4, 2, 12)
L = 4  I* = 18  U = unavailable  L/I* = 0.222; gates 19, work 19; optimal partition has 10 RUs:
  - 3 RUs (RU0, RU1, ...): work 2, input 0 B; ops [op0:swiglu x1]; imports nothing charged
  - RU3: work 2, input 4 B; ops [op1:colsum x3]; imports op1:colsum.out=4B
  - RU4: work 2, input 2 B; ops [op1:colsum x1, op2:matmul x2]; imports p1[token]=2B
  - RU5: work 2, input 2 B; ops [op1:colsum x3]; imports p1[token]=2B
  - RU6: work 2, input 4 B; ops [op1:colsum x1, op3:mul x1]; imports op1:colsum.out=4B
  - RU7: work 2, input 6 B; ops [op2:matmul x2]; imports op2:matmul.out=4B, p1[token]=2B
  - RU8: work 2, input 0 B; ops [op3:mul x2]; imports nothing charged
  - RU9: work 1, input 0 B; ops [op3:mul x1]; imports nothing charged

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (1, 3), 'fixed'),
        Param('p1', (1, 2), 'token'),
        Param('p2', (2, 2), 'fixed'),
    ],
    ops=[
        Op('swiglu', (('p', 'p0'), ('p', 'p0'))),  # 0
        Op('colsum', (('p', 'p1'),)),  # 1
        Op('matmul', (('p', 'p1', 'reprow', 0, 1), ('p', 'p1'))),  # 2
        Op('mul', (('p', 'p2'), ('p', 'p2'))),  # 3
    ],
    name='s21c139',
)
~~~

### s22c4  [fuzz; family `rmsnorm`]  policy (F, G, X) = (10, 3, 16)
L = 8  I* = 36  U = unavailable  L/I* = 0.222; gates 20, work 18; optimal partition has 6 RUs:
  - RU0, RU3: work 3, input 6 B; ops [op0:rmsnorm x4]; imports op0:rmsnorm.out=4B, p0[accumulated]=2B
  - 4 RUs (RU1, RU2, ...): work 3, input 6 B; ops [op0:rmsnorm x3]; imports op0:rmsnorm.out=4B, p0[accumulated]=2B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (2, 2), 'accumulated'),
        Param('p1', (2,), 'fixed'),
    ],
    ops=[
        Op('rmsnorm', (('p', 'p0'), ('p', 'p1'))),  # 0
    ],
    name='s22c4',
)
~~~

### s22c186  [fuzz; family `matmul|shared`]  policy (F, G, X) = (2, 2, 10)
L = 12  I* = 52  U = 80  L/I* = 0.231; gates 20, work 16; optimal partition has 8 RUs:
  - RU0, RU6: work 2, input 4 B; ops [op0:matmul x3]; imports p0[accumulated]=4B
  - RU1, RU7: work 2, input 6 B; ops [op0:matmul x2]; imports op0:matmul.out=4B, p0[accumulated]=2B
  - RU2, RU4: work 2, input 8 B; ops [op0:matmul x3]; imports p0[accumulated]=8B
  - RU3, RU5: work 2, input 8 B; ops [op0:matmul x2]; imports op0:matmul.out=4B, p0[accumulated]=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (2, 3), 'accumulated'),
    ],
    ops=[
        Op('matmul', (('p', 'p0'), ('p', 'p0'))),  # 0
    ],
    name='s22c186',
)
~~~

### s21c4  [fuzz; family `add+gain+matmul|shared,view,chain`]  policy (F, G, X) = (25, 1, 6)
L = 28  I* = 120  U = unavailable  L/I* = 0.233; gates 38, work 30; optimal partition has 30 RUs:
  - 8 RUs (RU0, RU3, ...): work 1, input 2 B; ops [op0:matmul x2]; imports p1[token]=2B
  - 8 RUs (RU1, RU4, ...): work 1, input 6 B; ops [op0:matmul x1]; imports op0:matmul.out=4B, p1[token]=2B
  - 8 RUs (RU2, RU5, ...): work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
  - 3 RUs (RU24, RU25, ...): work 1, input 4 B; ops [op1:gain x1]; imports p2[carried]=2B, p3[token]=2B
  - 3 RUs (RU27, RU28, ...): work 1, input 4 B; ops [op2:add x1]; imports op1:gain.out=2B, p2[carried]=2B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (2, 2), 'fixed'),
        Param('p1', (4, 2), 'token'),
        Param('p2', (1, 3), 'carried'),
        Param('p3', (3,), 'token'),
    ],
    ops=[
        Op('matmul', (('p', 'p0'), ('p', 'p1'))),  # 0
        Op('gain', (('p', 'p2'), ('p', 'p3'))),  # 1
        Op('add', (('o', 1, 'reprow', 0, 1), ('p', 'p2'))),  # 2
    ],
    name='s21c4',
)
~~~

### s21c84  [fuzz; family `mul+rmsnorm|shared,chain`]  policy (F, G, X) = (25, 4, 12)
L = 8  I* = 34  U = unavailable  L/I* = 0.235; gates 22, work 21; optimal partition has 6 RUs:
  - RU0: work 4, input 4 B; ops [op0:rmsnorm x5]; imports p0[accumulated]=4B
  - RU1: work 4, input 8 B; ops [op0:rmsnorm x4]; imports op0:rmsnorm.out=4B, p0[accumulated]=4B
  - RU2: work 4, input 10 B; ops [op0:rmsnorm x4]; imports op0:rmsnorm.out=4B, p0[accumulated]=6B
  - RU3: work 3, input 6 B; ops [op0:rmsnorm x2, op1:mul x1]; imports op0:rmsnorm.out=4B, p0[accumulated]=2B
  - RU4: work 4, input 4 B; ops [op0:rmsnorm x2, op1:mul x2]; imports op0:rmsnorm.out=4B
  - RU5: work 2, input 2 B; ops [op0:rmsnorm x1, op1:mul x1]; imports op0:rmsnorm.out=2B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (1, 4), 'accumulated'),
        Param('p1', (4,), 'fixed'),
    ],
    ops=[
        Op('rmsnorm', (('p', 'p0'), ('p', 'p1'))),  # 0
        Op('mul', (('o', 0), ('o', 0))),  # 1
    ],
    name='s21c84',
)
~~~

### s22c201  [fuzz; family `rmsnorm`]  policy (F, G, X) = (3, -, 6)
L = 8  I* = 34  U = unavailable  L/I* = 0.235; gates 18, work 15; optimal partition has 6 RUs:
  - RU0: work 4, input 6 B; ops [op0:rmsnorm x5]; imports op0:rmsnorm.out=2B, p0[token]=2B, p1[accumulated]=2B
  - 3 RUs (RU1, RU3, ...): work 1, input 6 B; ops [op0:rmsnorm x1]; imports op0:rmsnorm.out=4B, p0[token]=2B
  - RU2: work 6, input 4 B; ops [op0:rmsnorm x8]; imports p0[token]=4B
  - RU4: work 2, input 6 B; ops [op0:rmsnorm x2]; imports op0:rmsnorm.out=4B, p1[accumulated]=2B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (3, 1), 'token'),
        Param('p1', (1,), 'accumulated'),
    ],
    ops=[
        Op('rmsnorm', (('p', 'p0'), ('p', 'p1'))),  # 0
    ],
    name='s22c201',
)
~~~

### s21c109  [fuzz; family `matmul+rowscale|shared,view,chain`]  policy (F, G, X) = (17, 1, 12)
L = 8  I* = 32  U = unavailable  L/I* = 0.250; gates 10, work 8; optimal partition has 8 RUs:
  - RU0: work 1, input 2 B; ops [op0:matmul x2]; imports p0[accumulated]=2B
  - RU1: work 1, input 6 B; ops [op0:matmul x1]; imports op0:matmul.out=4B, p0[accumulated]=2B
  - RU2, RU5: work 1, input 4 B; ops [op0:matmul x1]; imports op0:matmul.out=4B
  - RU3: work 1, input 4 B; ops [op0:matmul x2]; imports p0[accumulated]=4B
  - RU4: work 1, input 8 B; ops [op0:matmul x1]; imports op0:matmul.out=4B, p0[accumulated]=4B
  - RU6, RU7: work 1, input 2 B; ops [op1:rowscale x1]; imports op0:matmul.out=2B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (2, 2), 'accumulated'),
        Param('p1', (2,), 'fixed'),
    ],
    ops=[
        Op('matmul', (('p', 'p0'), ('p', 'p0', 'reprow', 0, 1))),  # 0
        Op('rowscale', (('o', 0), ('p', 'p1'))),  # 1
    ],
    name='s21c109',
)
~~~

### s22c56  [fuzz; family `matmul|shared,view`]  policy (F, G, X) = (10, 3, 4)
L = 2  I* = 8  U = 16  L/I* = 0.250; gates 12, work 8; optimal partition has 4 RUs:
  - 4 RUs (RU0, RU1, ...): work 2, input 2 B; ops [op0:matmul x3]; imports p0[accumulated]=2B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (1, 1), 'accumulated'),
    ],
    ops=[
        Op('matmul', (('p', 'p0', 'reprow', 0, 4), ('p', 'p0'))),  # 0
    ],
    name='s22c56',
)
~~~

### s22c62  [fuzz; family `embed+gain+matmul|shared,view,chain`]  policy (F, G, X) = (7, 1, 10)
L = 12  I* = 48  U = unavailable  L/I* = 0.250; gates 14, work 12; optimal partition has 12 RUs:
  - 4 RUs (RU0, RU1, ...): work 1, input 2 B; ops [op0:gain x1]; imports p1[accumulated]=2B
  - RU4, RU5: work 1, input 4 B; ops [op1:embed x1]; imports p2[token]=4B
  - RU6, RU9: work 1, input 4 B; ops [op2:matmul x2]; imports op0:gain.out=2B, p1[accumulated]=2B
  - RU7, RU10: work 1, input 8 B; ops [op2:matmul x1]; imports op0:gain.out=2B, op2:matmul.out=4B, p1[accumulated]=2B
  - RU8, RU11: work 1, input 4 B; ops [op2:matmul x1]; imports op2:matmul.out=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (2, 2), 'fixed'),
        Param('p1', (2,), 'accumulated'),
        Param('p2', (1,), 'token', width=32),
    ],
    ops=[
        Op('gain', (('p', 'p0'), ('p', 'p1'))),  # 0
        Op('embed', (('p', 'p2'), ('p', 'p0'))),  # 1
        Op('matmul', (('p', 'p1', 'as2d'), ('o', 0))),  # 2
    ],
    name='s22c62',
)
~~~

### s21c0  [fuzz; family `rmsnorm`]  policy (F, G, X) = (4, -, 6)
L = 24  I* = 92  U = unavailable  L/I* = 0.261; gates 36, work 34; optimal partition has 16 RUs:
  - RU0, RU10: work 4, input 4 B; ops [op0:rmsnorm x5]; imports p0[token]=4B
  - RU1, RU11: work 2, input 6 B; ops [op0:rmsnorm x2]; imports op0:rmsnorm.out=4B, p0[token]=2B
  - RU2, RU12: work 4, input 6 B; ops [op0:rmsnorm x4]; imports op0:rmsnorm.out=4B, p0[token]=2B
  - 6 RUs (RU3, RU4, ...): work 1, input 6 B; ops [op0:rmsnorm x1]; imports op0:rmsnorm.out=4B, p0[token]=2B
  - 4 RUs (RU6, RU7, ...): work 2, input 6 B; ops [op0:rmsnorm x2]; imports op0:rmsnorm.out=4B, p1[accumulated]=2B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (2, 4), 'token'),
        Param('p1', (4,), 'accumulated'),
    ],
    ops=[
        Op('rmsnorm', (('p', 'p0'), ('p', 'p1'))),  # 0
    ],
    name='s21c0',
)
~~~

### s22c146  [fuzz; family `matmul`]  policy (F, G, X) = (1, 4, 14)
L = 10  I* = 38  U = unavailable  L/I* = 0.263; gates 18, work 12; optimal partition has 4 RUs:
  - RU0: work 4, input 8 B; ops [op0:matmul x8]; imports p0[seed]=4B, p1[accumulated]=4B
  - RU1: work 3, input 12 B; ops [op0:matmul x3]; imports op0:matmul.out=12B
  - RU2: work 3, input 10 B; ops [op0:matmul x5]; imports op0:matmul.out=4B, p0[seed]=2B, p1[accumulated]=4B
  - RU3: work 2, input 8 B; ops [op0:matmul x2]; imports op0:matmul.out=8B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (3, 1), 'seed'),
        Param('p1', (2, 1), 'accumulated'),
    ],
    ops=[
        Op('matmul', (('p', 'p0'), ('p', 'p1'))),  # 0
    ],
    name='s22c146',
)
~~~

### s22c119  [fuzz; family `add+matmul+mul|shared`]  policy (F, G, X) = (13, 2, 16)
L = 6  I* = 22  U = 22  L/I* = 0.273; gates 22, work 17; optimal partition has 9 RUs:
  - RU0: work 2, input 0 B; ops [op0:matmul x3]; imports nothing charged
  - RU1: work 2, input 4 B; ops [op1:add x1, op2:mul x1]; imports p1[carried]=2B, p2[accumulated]=2B
  - RU2: work 2, input 2 B; ops [op2:mul x1, op3:matmul x2]; imports p2[accumulated]=2B
  - RU3, RU6: work 2, input 4 B; ops [op3:matmul x2]; imports op3:matmul.out=4B
  - RU4, RU7: work 2, input 0 B; ops [op3:matmul x3]; imports nothing charged
  - RU5: work 2, input 4 B; ops [op3:matmul x3]; imports op3:matmul.out=4B
  - RU8: work 1, input 4 B; ops [op3:matmul x1]; imports op3:matmul.out=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (1, 1), 'fixed'),
        Param('p1', (1, 1), 'carried'),
        Param('p2', (2, 1), 'accumulated'),
        Param('p3', (2, 2), 'fixed'),
    ],
    ops=[
        Op('matmul', (('p', 'p0'), ('p', 'p0'))),  # 0
        Op('add', (('p', 'p0'), ('p', 'p1'))),  # 1
        Op('mul', (('p', 'p2'), ('p', 'p2'))),  # 2
        Op('matmul', (('p', 'p3'), ('p', 'p3'))),  # 3
    ],
    name='s22c119',
)
~~~

### s23c2  [fuzz; family `add+matmul+matmul_tt|shared,view,chain`]  policy (F, G, X) = (1, -, 4)
L = 12  I* = 44  U = unavailable  L/I* = 0.273; gates 16, work 11; optimal partition has 11 RUs:
  - RU0: work 1, input 4 B; ops [op0:add x1]; imports p0[accumulated]=2B, p1[token]=2B
  - 4 RUs (RU1, RU3, ...): work 1, input 4 B; ops [op1:matmul x2]; imports op0:add.out=2B, p0[accumulated]=2B
  - 4 RUs (RU2, RU4, ...): work 1, input 4 B; ops [op1:matmul x1]; imports op1:matmul.out=4B
  - RU9: work 1, input 4 B; ops [op2:matmul x2]; imports p1[token]=2B, p2[seed]=2B
  - RU10: work 1, input 4 B; ops [op2:matmul x1]; imports op2:matmul.out=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (4, 1), 'accumulated'),
        Param('p1', (1, 1), 'token'),
        Param('p2', (1, 1), 'seed'),
    ],
    ops=[
        Op('add', (('p', 'p0', 'reprow', 3, 1), ('p', 'p1'))),  # 0
        Op('matmul', (('p', 'p0'), ('o', 0))),  # 1
        Op('matmul_tt', (('p', 'p1'), ('p', 'p2'))),  # 2
    ],
    name='s23c2',
)
~~~

### s21c75  [fuzz; family `matmul_tt|shared`]  policy (F, G, X) = (3, -, 10)
L = 16  I* = 56  U = 112  L/I* = 0.286; gates 24, work 20; optimal partition has 7 RUs:
  - RU0: work 9, input 10 B; ops [op0:matmul x13]; imports p0[carried]=10B
  - RU1: work 2, input 6 B; ops [op0:matmul x2]; imports op0:matmul.out=4B, p0[carried]=2B
  - RU2, RU4: work 1, input 8 B; ops [op0:matmul x1]; imports op0:matmul.out=4B, p0[carried]=4B
  - RU3, RU5: work 2, input 8 B; ops [op0:matmul x2]; imports op0:matmul.out=4B, p0[carried]=4B
  - RU6: work 3, input 8 B; ops [op0:matmul x3]; imports op0:matmul.out=4B, p0[carried]=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (4, 2), 'carried'),
    ],
    ops=[
        Op('matmul_tt', (('p', 'p0'), ('p', 'p0'))),  # 0
    ],
    name='s21c75',
)
~~~

### s23c4  [fuzz; family `matmul_tt|shared`]  policy (F, G, X) = (2, -, 12)
L = 16  I* = 56  U = 112  L/I* = 0.286; gates 24, work 20; optimal partition has 5 RUs:
  - RU0: work 9, input 12 B; ops [op0:matmul x13]; imports op0:matmul.out=4B, p0[accumulated]=8B
  - RU1: work 3, input 12 B; ops [op0:matmul x3]; imports op0:matmul.out=8B, p0[accumulated]=4B
  - RU2: work 2, input 12 B; ops [op0:matmul x2]; imports op0:matmul.out=8B, p0[accumulated]=4B
  - RU3: work 4, input 12 B; ops [op0:matmul x4]; imports op0:matmul.out=8B, p0[accumulated]=4B
  - RU4: work 2, input 8 B; ops [op0:matmul x2]; imports op0:matmul.out=4B, p0[accumulated]=4B

~~~python
from accumulation.redteam.circuits import Circuit, Param, Op
circ = Circuit(
    params=[
        Param('p0', (4, 2), 'accumulated'),
    ],
    ops=[
        Op('matmul_tt', (('p', 'p0'), ('p', 'p0'))),  # 0
    ],
    name='s23c4',
)
~~~

... 1 more circuits in validation.json (`rows[*].optimal_partition`).
