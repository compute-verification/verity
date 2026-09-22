# backends/gkr/packed: base-field-first ("packed") GKR c-round sumcheck on INT8 tensor cores

Lane `a-packed` (track A).  Candidate A's eq-weighted degree-3 sumcheck over BabyBear with a `F_{p^6}` challenge
(`2^-128`), with the first `k` rounds computed from the untouched base-field tables (Bagad-Domb-Thaler 2025/1117)
and the eq factor never folded (Gruen 2024/108, Dao-Thaler 2024/1210).  Design, cost model, soundness: `DESIGN.md`.
Labels on every number: `ARITHMETIC_DIAGNOSTIC` / `COMPONENT`, `authentication excluded`, `security.target = -128`,
random tables, kernel only.  Not a proof.

~~~text
reference.py        BabyBear^6 in Python ints; naive three-table prover, packed prover (eq-trick + k base rounds), verifier;
                    the two provers emit bit-identical transcripts (tests/test_reference.py, laptop, < 1 s)
sumcheck_packed.py  the torch kernel: Kernel(mode="eager" | "compile" | "triton"); packed_round, fold_k, ext_round, fold_pair;
                    every matmul is torch._int_mm (cuBLAS INT8 -> INT32), K1 = 2^13 rows per inner product
kernels_triton.py   fused SIMT passes (Hadamard + limbs, both folds) -- the Hadamard kernels are generated unrolled source
bench.py            measurement (CUDA graphs, CUDA events, median of --repeats), layer model, MAC model, research envelope,
                    --reprice (laptop: explore.tensor design A with the measured bucket substituted)
tests/              kernel == reference bit for bit: CPU eager (laptop), CUDA eager + triton (pod, PACKED_DEVICE=cuda)
results/            fetched bench.json / result.json / logs of the runs below
~~~

## Run

~~~sh
# laptop (CPU torch): reference + kernel agreement
uv run python -m pytest backends/gkr/packed/tests -q

# pod vy-g4 (H100): agreement on CUDA in both modes, then the sweep as a recorded run (commit first)
PACKED_DEVICE=cuda /workspace/venv312/bin/python -m pytest backends/gkr/packed/tests -q
research run --on vy-g4 --project verity --source . --require-result -- \
    /workspace/venv312/bin/python -m backends.gkr.packed.bench --mode triton --log2 20 22 24 --repeats 9 --warm 3
research fetch <run> --all

# laptop: re-price
PYTHONPATH=backends/numerical/python python backends/gkr/packed/bench.py --reprice <run>/bench.json \
    --report backends/numerical/reports/packed_sumcheck_h100
~~~

## Recorded runs (vy-g4: NVIDIA H100 80GB HBM3 SXM, driver 580.126.20, CUDA 12.4, torch 2.6.0+cu124, Triton 3.2)

| run | what | agreement |
|---|---|---|
| **r20260922-062108-4b25** | **the table**: `--mode triton` (fused kernels), t = 2^20, 2^22, 2^24, 2^26, k = 0..4, median of 9 after 3 warm | 36 tests passed on the H100 (CPU, CUDA eager, CUDA triton; k = 0..4, n <= 10) |
| r20260922-055336-4080 | earlier state: Triton SIMT passes + cuBLAS `_int_mm`, torch reduce/contract; t <= 2^24 | 36 passed |
| (unrecorded, `results/eager_l22_24`) | `--mode eager`: the torch-op baseline | 22 passed |

## The table (r20260922-062108-4b25; one layer table would be 2^28.6 elements)

Whole sumcheck, all rounds, each round replayed as a CUDA graph (device time, no launch gaps):

| k | 2^24 ms | 2^26 ms | frac. INT8 peak (2^26, whole round) | MACs issued (2^26) | peak mem (2^26) | fit ns/elem + ms | per batch (6 layers, padded / un-padded) | per VU |
|---|---|---|---|---|---|---|---|---|
| 0 | 9.20 | 30.87 | 1.39% | 4.25e11 | 34.9 GB | 0.435 + 1.73 | 973 / 581 ms | **238 us** (142) |
| 1 | 6.06 | 18.47 | 1.36% | 2.48e11 | 18.0 GB | 0.251 + 1.68 | 566 / 339 ms | 138 us (83) |
| 2 | 4.35 | 11.73 | 1.39% | 1.61e11 | 9.7 GB | 0.152 + 1.61 | 345 / 209 ms | 84 us (51) |
| 3 | 3.71 | 9.08 | 1.63% | 1.47e11 | 5.3 GB | 0.112 + 1.64 | 257 / 156 ms | **62.7 us** (38.2) |
| 4 | 7.30 | 23.37 | 0.85% | 1.97e11 | 5.0 GB | 0.325 + 1.63 | 729 / 436 ms | 178 us (106) |

**A100 rows** (anchor-a100 lane, NVIDIA A100-SXM4-80GB, torch 2.6 cu124, Triton 3.2; run `r20260922-073828-153f`, same
flags, 36 tests bit-exact on CUDA; `backends/numerical/reports/anchor_a100.md`).  Whole sumcheck, CUDA-graphed rounds:

| k | 2^24 ms | 2^26 ms | frac. A100 INT8 peak (2^26) | peak mem (2^26) | fit ns/elem + ms | per batch (padded / un-padded) | per VU | vs H100 |
|---|---|---|---|---|---|---|---|---|
| 0 | 16.15 | 50.22 | 2.71% | 34.1 GB | 0.704 + 3.25 | 1,578 / 943 ms | 385 us | 0.62x |
| 1 | 10.63 | 30.73 | 2.59% | 17.2 GB | 0.415 + 3.03 | 937 / 563 ms | 229 us | 0.60x |
| 2 | 7.42 | 20.21 | 2.55% | 8.9 GB | 0.262 + 2.71 | 596 / 360 ms | 146 us | 0.58x |
| 3 | 6.96 | 16.58 | 2.83% | 4.5 GB | 0.208 + 2.81 | 476 / 289 ms | **116 us** | 0.54x |
| 4 | 18.27 | 64.95 | 0.97% | 4.1 GB | 0.928 + 2.69 | 2,071 / 1,234 ms | 506 us | 0.35x |

The plain k = 0 sumcheck scales exactly with bandwidth (0.62x the H100 = 2,039 / 3,350): it is HBM-bound on both cards.  The
packed k = 3 rounds scale a little worse (0.54x) -- their base x base Hadamards are INT32 work, and the A100 has 0.47x the
H100's INT32 rate -- so packing saves 3.3x on the A100 (3.8x on the H100).  k = 4's two-pass path is 0.35x.  The fixed
per-sumcheck intercept is 2.7-3.3 ms on the A100 vs 1.6-1.7 ms on the H100 (launch/tail latency at 1.41 GHz).

* "per batch" = the six c-round sumchecks of the batch (tables `393,216 x 2^read`, `LAYER_READ_LOG2`) from the linear
  fit over t = 2^20..2^26; "per VU" = per batch / 4096; un-padded widths (`LAYER_WIDTHS`) in parentheses.
* **Measured saving of packing on the dominant bucket: k = 3 vs k = 0 = 3.79x (padded), 3.72x (un-padded).**  The MAC
  model predicts 2.9x for the same pair (1.47e11 vs 4.25e11 at 2^26) and 8.4x against tensor.md's plain 18,432-MAC/element
  model; the kernels beat the MAC ratio because the packed rounds' base x base Hadamards are cheaper per MAC than the
  extension rounds' `F_{p^6}` Hadamards and folds, which the plain algorithm runs on every element.
* `k = 4` loses: the `4^(i-1)` blow-up makes round 4 a 192-row Hadamard block per pair (768 limb rows, too large for the
  fused register tile; it runs the two-pass path through HBM and cuBLAS, 3.2 GB of limbs at 2^26).
* **Utilisation.**  Whole-round fraction of the H100's 989.5 T MAC/s INT8 peak: 1.4% plain, 1.6% packed k = 3 at 2^26
  (0.35% / 0.16% before the fused kernels).  The tensor cores are not the bottleneck: the rounds are bound by the SIMT
  modular arithmetic (INT64 `%` for the Hadamard products and the `F_{p^6}` coefficient products) and by HBM in the
  folds.  Lane `tc-sumcheck`'s plain kernel on the RTX 4090: 2.2% matmul-only batch-weighted, 0.56% for the tensor bucket.
  Both are far below the 17.2% that `tensor.md` assumed (TensorZKP's end-to-end calibration).  The bench's
  `utilisation.matmul_only` column is not meaningful for the fused kernels (the "matmul" is the whole fused kernel).
* Where the time goes (2^26, k = 3, CUDA-graphed rounds): packed rounds 1-3 = 0.76 + 0.69 + 1.15 ms (fused: Hadamard
  limbs in registers -> `tl.dot`, BLOCK_K = 128), fold_k 1.03 ms, extension rounds 4-7 = 1.90 + 1.03 + 0.57 + 0.34 ms
  (fused Hadamard+dot ~50%, the `[x_e | x_o] x [C_(1-r); C_r]` fold ~35%), then a floor of 0.07 ms per round.
* History of the same measurement at 2^24 (k = 0 / k = 3): eager torch 253 / 81 ms; Triton SIMT passes + cuBLAS +
  torch reduce 30.4 / 22.5 ms (r20260922-055336-4080, 0.35% / 0.16% of peak); fused inner products 14.1 / 4.7 ms;
  + Triton reduce/contract and finish 12.6 / 4.5 ms; + one reduction per limb sum 9.2 / 3.7 ms (the table).

## Re-priced Candidate A (`backends/numerical/reports/packed_sumcheck_h100.json`)

`explore.tensor` design A on the H100 with the `gkr_checker` matmul bucket (5.4e9 MACs/VU at 0.172) and its SIMT
Hadamard bucket replaced by the measured six-layer sumcheck; every other bucket (logUp GKR, Ligero encoding, row
combination, hashing, hint generation) stays modelled at 0.172.

| | overhead vs native peak | t.total per batch | checker sumcheck |
|---|---|---|---|
| model @ 0.172 | 1.07e7 | 0.431 s | 127 ms matmul + 130 ms Hadamards (model) |
| measured k = 0 (padded / un-padded) | 2.84e7 / 1.87e7 | 1.147 / 0.755 s | 973 / 581 ms (85% / 77% of total) |
| measured k = 3 (padded / un-padded) | **1.07e7 / 8.19e6** | 0.431 / 0.330 s | 257 / 156 ms (60% / 47%) |

Read: the measured packed kernel lands exactly on the number the model assumed for this bucket (257 ms per batch both
ways), but for a different reason -- 3.8x fewer/cheaper operations at 1.6% utilisation instead of the plain algorithm
at 17.2%.  The plain kernel as measured would have put A at 2.84e7.  The sumcheck is still 47-60% of A's re-priced
total; the remaining buckets are modelled, not measured.

## Ledger (`backends/reports/ledger/a-packed.jsonl`)

Final entries (run r20260922-062108-4b25): `plain c-round sumcheck k=0, fused Triton+tensor cores` -- component,
238 us/VU; `packed sumcheck k=3: 3.8x on the dominant bucket (measured)` -- component, 62.7 us/VU, breakthrough;
`A re-priced w/ packed sumcheck, model` -- `ARITHMETIC_DIAGNOSTIC`, scope vu, overhead 1.07e7, omitted: every modelled
bucket.  Earlier entries from r20260922-055336-4080 (509 / 247 us per VU, 2.94e7) are the pre-fusion state.

## Second pass (lane `a-packed2`): 32-bit Montgomery, direct 2^28, fused k = 4, logUp

Full write-up: `backends/numerical/reports/a_kernels_h100.md`.  Runs (all vy-g4 / H100):

| run | what |
|---|---|
| r20260922-064917-c293 | Montgomery kernels, k = 0..4, t <= 2^26 (36 CUDA tests bit-exact) |
| r20260922-065412-73be | **direct 2^28** (one full layer table), k = 2, 3, 4: 19.78 / **17.69** / 30.96 ms |
| r20260922-073439-1f15 | **logUp GKR at the full-VU shape** (393,216 x 306 lookups, 65,536-row table): k = 3 **427 ms = 104 us/VU**, k = 0 447 ms |
| r20260922-073702-b11f | fused (split) k = 4 vs k = 3: 2^26 7.57 vs 6.13 ms, 2^28 22.88 vs 17.61 ms. **k = 4 fused recovers 1.3x (9.25 -> 7.57) but still loses; default stays k = 3**; kept as a memory lever (11.0 vs 18.2 GB at 2^28) |

| **A100** r20260922-075147-91a5 (anchor-a100 lane, NVIDIA A100-SXM4-80GB, same flags; 46 CUDA tests bit-exact) | Montgomery 2^26: k = 0..4 **21.4 / 17.8 / 11.2 / 11.5 / 15.2 ms** (0.52-0.61x the H100 = the bandwidth ratio 0.61); direct 2^28 k = 2, 3, 4: 34.3 / **31.9** / 43.8 ms (0.55x; 5.9% of the A100 INT8 peak) = **64.3 us/VU, 263 ms per six-layer batch** padded (156 un-padded); logUp GKR full-VU k = 3 **412 ms = 101 us/VU** (H100 427: 1.04x -- launch-bound, the card does not matter yet), k = 0 478 ms. Table: `backends/numerical/reports/anchor_a100.md` section 4b |

The k-table before -> after Montgomery (2^26 ms; per VU = six padded layers / 4096): k = 0 30.9 -> 12.6 (238 -> 90 us), k = 1 18.5 -> 10.7,
k = 2 11.7 -> 6.4, **k = 3 9.08 -> 5.97 (62.7 -> 37.5 us; 35.6 us from the direct 2^28 point)**, k = 4 23.4 -> 9.25.
3.35% of INT8 peak and 3.3x the HBM floor at 2^28: still SIMT-bound (limb decomposition + `tl.join` feeding `tl.dot`).

### logUp on the GPU (`logup_gpu.py`, `logup_reference.py`, `logup_bench.py`)

The whole fractional GKR for one table: `bincount` multiplicities; the level below the leaves formed straight from the base
queries (`leaf_level`), the tree by `tree_level` (levels stored de-interleaved `[evens | odds]` so a level *is* the
`p0, p1, q0, q1` operand of the sumcheck below it); the 2^26-pair leaf level with **k packed base rounds** on the witness
(`A_x = 2z - s_x`, `B_x = z^2 - z s_x + pi_x`, `s = w_e + w_o`, `pi = w_e w_o` base: `logup_ip_{half}` = `half^2` Hadamards
+ `half` linear rows per `x`), the table region on the generic kernel with its own eq slice, the padding analytic; all
other levels on the generic four-table kernel `logup_ext_ip` (`A = p0 q1 + p1 q0`, `B = q0 q1` per pair and `x`, 144 limb
rows), `reduce_contract`, `finish_round` with `coeff = [1, lambda]`, `fold_ext4`; levels under 128 pairs per chunk in one
fused SIMT kernel (`logup_small_round`).  Bit-exact against `logup_reference.py` (`logup.rs` with low-bit-first binding).

Split at k = 3 (427 ms): histogram 3.7, tree values 12.6, leaf level 8.2 (packed) + 2.4 (fold_k) + 29.4 (23 extension
rounds), level 25 48, level 24 37, ..., the 20 smallest levels 171 ms.  **Launch-bound**: ~1 ms of Python/Triton launch
overhead per round x 351 rounds; the device work is ~90-200 ms.  Peak memory 12.5 GB.

### For a-gpu: `kernels.py`

`from backends.gkr.packed import kernels` (or `from packed import kernels` with `backends/gkr` on `sys.path`).  Everything
that crosses the module boundary is **plain** (not Montgomery), `torch.int32`, values in `[0, p)`:

* base table `(t,)`; extension table `(t, 6)` **interleaved** (row = the 6 coefficients of one element).  If your prover is
  planar `(6, t)`, transpose at the boundary (`x.t().contiguous()`).  `X^6 = W` (`reference.W`), BabyBear `p = 2^31 - 2^27 + 1`.
* host-side extension elements are `reference.Ext` = 6-tuples of Python ints.
* binding order: round `i` binds bit `i - 1` of the table index (pairs `(2y, 2y+1)`); `eq(gamma, b)` has `gamma[0]` on bit 0.
  `logup.rs` binds the top bit first; same protocol under bit-reversal (see `logup_reference.py`'s docstring).

~~~python
from backends.gkr.packed import kernels as K
kern = K.make_kernel("cuda")                                   # once; compiles the generated Triton module on first use
rounds, rs, (eq_at_rs, g_at_rs, h_at_rs) = K.sumcheck_layer(kern, gammas, g, h, k=3, challenger=fs_challenge)
                                                               # g, h: (2^n,) int32; challenger(i, S_i) -> r_i (Fiat-Shamir here)
gE = K.fold(kern, g, rs[:3])                                   # base (t,) -> extension (t/8, 6) by eq(rs) (fold_k)
prover = K.LogUpProver("cuda", k=3)
tr, claim, timing = K.logup_prove(prover, w, table, z, coins)  # w: (Nw,) int32 queries in natural order; coins.next() -> Ext
S = K.logup_round(prover, T4, rho, rs, i, lam)                 # one generic level round on T4 (4, t_i, 6) = p0, p1, q0, q1
T4 = K.logup_fold(prover, T4, r)
pq = K.leaf_level(prover, w, z, n); pq = K.tree_level(prover, pq)
~~~

Shapes: `sumcheck_layer` needs `t` a multiple of `2^k . 8192` for the fused packed rounds (`k1_log2` lowers the 8192);
`logup_prove` needs `Nw` and `Nt` aligned so every region (witness / table / pad) is a multiple of `2^(k+1) . 128` leaves
(`393,216 x 306` and `65,536` are).  The coins object is a stand-in for the verifier: `lambda_0`, then per level
`r_1..r_k, mu, lambda` -- replace `coins.next()` with the sponge; the transcript (`tr.q_root`, per level `rounds` (4 `Ext`
each) and `vals = [p0, p1, q0, q1]`) is what `logup.rs` sends, modulo the bit-reversal convention.  Test vectors:
`python -m backends.gkr.packed.kernels --vectors` (reference SHA-256s of two fixed cases: `sumcheck_t4096_k3_seed7` =
`a6a12f3b...1afd780`, `logup_Nw12288_Nt2048_seed1` = `2a8fe70f...bee37bda`; the GPU path is checked message
by message in `tests/test_kernel_agreement.py` and `tests/test_logup_gpu.py` (`PACKED_DEVICE=cuda pytest backends/gkr/packed/tests`).

Launch overhead is the a-gpu prover's problem to solve (it owns the round loop): ~10 launches + 1 H2D per round here.

## Not done / next

1. **Univariate skip** (Gruen 2024/108): a protocol change (round degree `3 (2^s - 1)`, new soundness term); needs
   `security.accounting` and a flagged re-account.  Not started.
2. **SIMT modular arithmetic.**  The rounds are bound by INT64 `%` (emulated 64-bit division) in the Hadamards and
   `F_{p^6}` products: a 32-bit Montgomery or Barrett reduction in the Triton kernels (BabyBear fits: `p < 2^31`) should
   give 2-4x on the Hadamard passes; the fold matrices' INT64 recompositions the same.
3. **k = 4 fused**: split the 768-row Hadamard block across programs (two 384-row tiles) so round 4 does not go through
   HBM; and a fused fold_k (currently a SIMT kernel with `2^k . 6` INT64 products per output: 1.03 ms at 2^26).
4. The layer model extrapolates linearly from t <= 2^26 to 2^28.6 (one layer table); a direct 2^28 measurement needs
   k >= 2 (memory: 34.9 GB at 2^26 for k = 0, 5.3 GB for k = 3).
