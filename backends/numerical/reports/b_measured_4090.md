# Candidate B on the RTX 4090, every prover bucket measured (b-commit lane, 2026-09-22)

**Label: `ARITHMETIC_DIAGNOSTIC`-level re-pricing, scope vu.**  vu-level re-pricing: tensor.py cost model with the commitment, IRS row combination, linear and quadratic tests, helper inversions and column openings replaced by standalone measurements on the RTX 4090 (B = 4096, K = 1536, 2^-128, BabyBear / BabyBear^6, HVZK rows included, authentication excluded).  Device: NVIDIA GeForce RTX 4090 (128 SMs, driver 580.159.04, CUDA runtime 12090, cupy 14.2.0).  Runs: `r20260922-064722-07c7`, `r20260922-065537-279b`.  CUDA-event timings, warm, median of 7; every kernel bit-exact against a numpy / Python reference before timing.

## Measured buckets, B shape (310,602 Ligero rows x [16384, 4096] BabyBear; 33,485 product rows; 125.9M helpers; t = 192)

| bucket | kernel | ms / batch | us / row | floor | fraction of floor | bit-exact |
|---|---|---|---|---|---|---|
| commitment (encode -> blake3 column hash -> root) | `commit_stream` fused, 512-row L2 slices, CUDA graph, cosets kept (15.3 GB store) | **49.4** | 0.159 | encode 32.2 + hash unfused = 88.8 ms; HBM read floor 5.0 ms | 1.80x faster than unfused | yes (vs unfused path + CPU) |
| IRS row combination v = r^T W (k-wide, F_c scalar per row) | `combine_scalar`, chunk 1024 | **5.7** | 0.018 | read 5.09 GB once = 5.0 ms | 0.89 of HBM floor (1351 G mont-mul/s) | yes |
| quadratic test p_0 (8,192-point domain) | 3 encoder calls (coset 1) per product row + `combine_quadratic`, 512-row slices | **7.8** | 0.232 per product row | 6 transforms per product row | -- | yes |
| linear test q_add (8,192-point domain) | 7 encoder calls per row (6 coordinate rows of r^T A + the witness row, coset 1) + `combine_linear`, 256-row slices | **149.4** | 0.481 | 14 transforms per row = 3.5x the commitment's encoding | 3.0x the fused commitment per row | yes (r^T A generation NOT included) |
| helper inversions (125.9M F_c elements) | `batch_inverse`, Montgomery trick, 262144 threads x 481 | **28.8** | 0.23 ns / element | traffic 15.0 ms | 0.52 of HBM floor | yes |
| column openings (t = 192 columns + paths) | `open_columns_gather_from_store` + host Merkle paths | **5.1** | -- | 239 MB opened | -- | yes |

## Re-pricing (all rows: `explore.tensor` with the buckets above measured; `t.modelled` = what the model still prices)

~~~text
design gpu      leaf    open      total ms  commit tests+irs    inv   open  model  meas%  overhead  label
B      RTX4090  blake3  store        258.7    49.4     162.9   28.8    5.1   12.5    95%  6.41e+06  MEASURED buckets on RTX 4090 (commitment, tests, inversions, openings); r^T A generation, witness generation, PRG modelled
B      A100     blake3  store        564.1   104.6     345.0   61.1   10.8   42.6    92%   1.4e+07  SCALING: measured buckets / 0.47 (INT32 issue-rate ratio; they are compute-bound on the 4090), modelled buckets at A100 peaks
B      H100     blake3  store        318.5    61.0     201.0   35.6    6.3   14.7    95%   7.9e+06  SCALING: measured buckets / 0.81 (INT32 issue-rate ratio; they are compute-bound on the 4090), modelled buckets at H100 peaks
B      RTX4090  blake3  reencode     284.5    46.7     162.9   28.8   33.5   12.5    96%  7.05e+06  MEASURED buckets on RTX 4090 (commitment, tests, inversions, openings); r^T A generation, witness generation, PRG modelled
B      A100     blake3  reencode     618.6    99.0     345.0   61.1   71.0   42.6    93%  1.53e+07  SCALING: measured buckets / 0.47 (INT32 issue-rate ratio; they are compute-bound on the 4090), modelled buckets at A100 peaks
B      H100     blake3  reencode     350.3    57.7     201.0   35.6   41.4   14.7    96%  8.69e+06  SCALING: measured buckets / 0.81 (INT32 issue-rate ratio; they are compute-bound on the 4090), modelled buckets at H100 peaks
B      RTX4090  sha256  store        270.0    60.7     162.9   28.8    5.1   12.5    95%   6.7e+06  MEASURED buckets on RTX 4090 (commitment, tests, inversions, openings); r^T A generation, witness generation, PRG modelled
B      A100     sha256  store        588.1   128.7     345.0   61.1   10.8   42.6    93%  1.46e+07  SCALING: measured buckets / 0.47 (INT32 issue-rate ratio; they are compute-bound on the 4090), modelled buckets at A100 peaks
B      H100     sha256  store        332.5    75.0     201.0   35.6    6.3   14.7    96%  8.24e+06  SCALING: measured buckets / 0.81 (INT32 issue-rate ratio; they are compute-bound on the 4090), modelled buckets at H100 peaks
B      RTX4090  sha256  reencode     296.1    58.3     162.9   28.8   33.5   12.5    96%  7.34e+06  MEASURED buckets on RTX 4090 (commitment, tests, inversions, openings); r^T A generation, witness generation, PRG modelled
B      A100     sha256  reencode     643.2   123.6     345.0   61.1   71.0   42.6    93%  1.59e+07  SCALING: measured buckets / 0.47 (INT32 issue-rate ratio; they are compute-bound on the 4090), modelled buckets at A100 peaks
B      H100     sha256  reencode     364.6    72.0     201.0   35.6   41.4   14.7    96%  9.04e+06  SCALING: measured buckets / 0.81 (INT32 issue-rate ratio; they are compute-bound on the 4090), modelled buckets at H100 peaks
B      RTX4090  sha256  n/a          107.0                                   75.4    30%  2.65e+06  PREVIOUS (encode_gpu.json): measured encoder 31.6 ms + measured SHA-256 446 GB/s, other buckets modelled
A      RTX4090  blake3  store        637.7     8.4       1.0    0.0    2.1  626.2     2%  1.58e+07  MEASURED buckets on RTX 4090 (commitment, tests, inversions, openings); r^T A generation, witness generation, PRG modelled
A      A100     blake3  store       1162.5    17.8       2.1    0.0    4.4 1138.2     2%  2.88e+07  SCALING: measured buckets / 0.47 (INT32 issue-rate ratio; they are compute-bound on the 4090), modelled buckets at A100 peaks
A      H100     blake3  store        358.0    10.4       1.2    0.0    2.6  343.9     4%  8.88e+06  SCALING: measured buckets / 0.81 (INT32 issue-rate ratio; they are compute-bound on the 4090), modelled buckets at H100 peaks
A      RTX4090  blake3  reencode     641.2     7.4       1.0    0.0    6.7  626.2     2%  1.59e+07  MEASURED buckets on RTX 4090 (commitment, tests, inversions, openings); r^T A generation, witness generation, PRG modelled
A      A100     blake3  reencode    1170.1    15.7       2.1    0.0   14.1 1138.2     3%   2.9e+07  SCALING: measured buckets / 0.47 (INT32 issue-rate ratio; they are compute-bound on the 4090), modelled buckets at A100 peaks
A      H100     blake3  reencode     362.4     9.1       1.2    0.0    8.2  343.9     5%  8.99e+06  SCALING: measured buckets / 0.81 (INT32 issue-rate ratio; they are compute-bound on the 4090), modelled buckets at H100 peaks
A      RTX4090  sha256  store        639.8    10.5       1.0    0.0    2.1  626.2     2%  1.59e+07  MEASURED buckets on RTX 4090 (commitment, tests, inversions, openings); r^T A generation, witness generation, PRG modelled
A      A100     sha256  store       1167.0    22.3       2.1    0.0    4.4 1138.2     2%  2.89e+07  SCALING: measured buckets / 0.47 (INT32 issue-rate ratio; they are compute-bound on the 4090), modelled buckets at A100 peaks
A      H100     sha256  store        360.6    13.0       1.2    0.0    2.6  343.9     5%  8.94e+06  SCALING: measured buckets / 0.81 (INT32 issue-rate ratio; they are compute-bound on the 4090), modelled buckets at H100 peaks
A      RTX4090  sha256  reencode     643.1     9.3       1.0    0.0    6.7  626.2     3%  1.59e+07  MEASURED buckets on RTX 4090 (commitment, tests, inversions, openings); r^T A generation, witness generation, PRG modelled
A      A100     sha256  reencode    1174.1    19.6       2.1    0.0   14.1 1138.2     3%  2.91e+07  SCALING: measured buckets / 0.47 (INT32 issue-rate ratio; they are compute-bound on the 4090), modelled buckets at A100 peaks
A      H100     sha256  reencode     364.8    11.4       1.2    0.0    8.2  343.9     6%  9.04e+06  SCALING: measured buckets / 0.81 (INT32 issue-rate ratio; they are compute-bound on the 4090), modelled buckets at H100 peaks
A      RTX4090  sha256  n/a          640.1                                  635.1     1%  1.59e+07  PREVIOUS (encode_gpu.json): measured encoder 31.6 ms + measured SHA-256 446 GB/s, other buckets modelled
~~~

B on the 4090 with every bucket measured: **259 ms per batch, overhead 6.41e+06** (leaf blake3, openings from the kept store), 95% of the time measured; the previous state (encoder + SHA-256 measured, rest modelled) said 107 ms / 2.65e+06.  Modelled remainder: simt:constraint_combination 8.9 ms, hint_gen 0.4 ms, prg 3.2 ms.

## What the measurements say

* **The linear test is B's bucket, not hashing or encoding.**  q_add needs the deg < k interpolant of every row of r^T A (an F_c row: 6 BabyBear coordinate rows), evaluated on 8,192 points, times the re-encoded witness row: 14 NTT-size transforms per row against the commitment's 4.  Measured 149 ms = 58% of the batch.  The model priced the whole row-combination bucket as one tensor-core matmul of 5.4e11 MACs (9.5 ms at utilisation 0.172) and did not count the r^T A encodings at all.
* **Commitment: 49.4 ms fused** (1.80x vs unfused encode + hash), the codeword never in HBM (peak device memory 20.6 GiB).  Blake3 with an unrolled schedule hashes the 20.4 GB codeword in 15.4 ms; SHA-256 in 30 ms at the INT32 issue rate.  The SIMT NTT encoder (34-36 ms in L2 slices) is now the commitment's bottleneck; its SIMT floor is ~14 ms.
* IRS row combination 5.7 ms (0.89 of the read floor: memory-bound), quadratic test 7.8 ms, helper inversions 28.8 ms (0.52 of traffic floor), openings 5.1 ms: small buckets; no tensor-core operation anywhere in B.
* Still modelled: r^T A generation (the sparse constraint combination), witness generation (no GPU kernel exists), prover randomness.

## Files

`backends/direct/encode/commit_stream.py` (fused commitment), `buckets.py` (bucket kernels), `buckets_ref.py` (references), `commit_bench.py`, `buckets_bench.py`, `reprice_measured.py` (this report), raw `results_commit_*.json` / `results_buckets_*.json`; ledger `ledger/b-commit.jsonl`.
