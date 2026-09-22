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
| r20260922-055336-4080 | `--mode triton`, t = 2^20, 2^22, 2^24, k = 0..4, median of 9 after 3 warm | 36 tests passed on the H100 (CPU, CUDA eager, CUDA triton; k = 0..4, n <= 10) |
| (unrecorded, `results/eager_l22_24`) | `--mode eager`, t = 2^22, 2^24 -- the torch-op baseline the Triton kernels replace | 22 passed |

## The table (r20260922-055336-4080, whole sumcheck at t = 2^24, one layer table would be 2^28.6)

| k | sumcheck ms | matmul-only ms | MACs issued | frac. INT8 peak, whole | frac. peak, matmul-only | peak mem | fit ns/elem | per batch (6 layers, padded) | per VU |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 30.36 | 14.79 | 1.06e11 | 0.35% | 0.73% | 7.75 GB | 0.899 | 2084 ms | **509 us** |
| 1 | 25.39 | 14.76 | 6.20e10 | 0.25% | 0.42% | 4.43 GB | 0.600 | 1421 ms | 347 us |
| 2 | 22.90 | 8.73 | 4.03e10 | 0.18% | 0.47% | 2.77 GB | 0.443 | 1075 ms | 262 us |
| 3 | 22.54 | 5.82 | 3.66e10 | 0.16% | 0.64% | 2.05 GB | 0.414 | 1011 ms | **247 us** |
| 4 | 26.70 | 5.54 | 4.93e10 | 0.19% | 0.90% | 2.87 GB | 0.644 | 1523 ms | 372 us |

* "per batch" = the six c-round sumchecks of the batch (tables `393,216 x 2^read`, `LAYER_READ_LOG2`) from the linear
  fit over t = 2^20, 2^22, 2^24 (`fit ns/elem` x table + ~15.5 ms intercept per sumcheck); "per VU" = per batch / 4096.
  Un-padded widths (`LAYER_WIDTHS`): 1273 / 880 / 675 / 637 / 942 ms, i.e. 311 / 215 / 165 / 156 / 230 us per VU.
* **Measured saving of packing on the dominant bucket: k = 3 vs k = 0 = 2.06x (padded), 2.00x (un-padded).**  The
  MAC model predicts 2.9x for the same pair (3.66e10 vs 1.06e11) and 8.4x against tensor.md's plain 18,432-MAC/element
  model; the kernels realise less because they are not MAC-bound (next bullet).
* **Utilisation.**  The plain kernel (k = 0) runs its matmuls at 0.73% of the H100's INT8 peak and the whole round at
  0.35%; packed k = 3 at 0.64% / 0.16%.  Lane `tc-sumcheck`'s plain kernel on the RTX 4090 reports 2.2% matmul-only
  batch-weighted (3.6-4.6% in the large rounds, HBM-bound), 0.56% for the tensor bucket -- same order.  Both are two
  orders below the 17.2% that `tensor.md` assumed (TensorZKP's end-to-end calibration).  Where the time goes (t = 2^24,
  k = 3, per-round CUDA graphs): packed rounds 1-3 = 1.27 + 1.50 + 1.98 ms (the `4^(i-1)` limb matrix of round 3 is
  400 MB, written by SIMT and read by cuBLAS; `_int_mm` with N = 24 runs at ~0.6 TB/s), fold_k 0.28 ms, extension rounds
  4-6 = 1.66 + 1.27 + 1.05 ms, then a floor of ~0.72 ms per round for rounds with t < 2^17 (the reduce/contract chain of
  ~150 small torch kernels: 13 tail rounds ~ 9.4 ms of the 15.5 ms intercept).
* Eager torch (`results/eager_l22_24`): 253 ms at 2^24 k = 0, 81 ms at k = 3 -- the fused Triton kernels are 8.3x / 3.6x
  faster; the eager per-VU figures were 7.47 ms / 1.95 ms.

## Re-priced Candidate A (`backends/numerical/reports/packed_sumcheck_h100.json`)

`explore.tensor` design A on the H100 with the `gkr_checker` matmul bucket (5.4e9 MACs/VU at 0.172) and its SIMT
Hadamard bucket replaced by the measured six-layer sumcheck; every other bucket (logUp GKR, Ligero encoding, row
combination, hashing, hint generation) stays modelled at 0.172.

| | overhead vs native peak | t.total per batch | checker sumcheck |
|---|---|---|---|
| model @ 0.172 | 1.07e7 | 0.431 s | 127 ms matmul + 130 ms Hadamards (model) |
| measured k = 0 (padded / un-padded) | 5.60e7 / 3.59e7 | 2.258 / 1.447 s | 2084 / 1273 ms (92% / 88% of total) |
| measured k = 3 (padded / un-padded) | **2.94e7 / 2.01e7** | 1.184 / 0.811 s | 1011 / 637 ms (85% / 79%) |

Read: packing halves the measured dominant bucket (ledger `--breakthrough`, 2.06x), but the measured kernel -- even
packed -- is 4x slower than the model priced it, because the assumed 17.2% tensor-core utilisation is not what these
kernels get (0.2-0.7%).  The sumcheck is still 80-90% of A's re-priced total.

## Ledger (`backends/reports/ledger/a-packed.jsonl`)

* `plain INT8-limb c-round sumcheck k=0 (eq-trick, Triton)` -- component, 509 us/VU.
* `packed sumcheck k=3: 2.06x on the dominant bucket (measured)` -- component, 247 us/VU, breakthrough.
* `A re-priced w/ packed sumcheck, model` -- `ARITHMETIC_DIAGNOSTIC`, scope vu, overhead 2.94e7, omitted: every modelled bucket.

## Not done / next

1. **Univariate skip** (Gruen 2024/108): a protocol change (round degree `3 (2^s - 1)`, new soundness term); needs
   `security.accounting` and a flagged re-account.  Not started.
2. **Make the kernel MAC-bound.**  The A operand of every inner product is materialised in HBM (limbs written by one
   kernel, read by cuBLAS with N = 24).  One Triton kernel that forms the Hadamard limbs in registers and feeds `tl.dot`
   directly (as `fold_ext` already does for the fold) removes the 400 MB round trip of the packed rounds and the
   0.6 TB/s cuBLAS pass; the reduce/contract tail should be one kernel per round (removes the 0.72 ms floor).
3. The layer model extrapolates linearly from t <= 2^24 to 2^28.6; measure one 2^28 sumcheck directly (memory: 7.75 GB
   at 2^24 for k = 0 -> ~124 GB naive; needs the chunked A operand of step 2 or k >= 2, 2.05-2.77 GB at 2^24).
