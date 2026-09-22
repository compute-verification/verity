# encode -- Reed-Solomon encoders, the fused streaming commitment and B's prover buckets on an RTX 4090 (b-encode + b-commit lanes, 2026-09-22)

Sections: the encoder (b-encode lane, below) -- then **the fused encode -> column-hash commitment and B's remaining buckets
(b-commit lane, second half of this file)**.

**Scope: component.** Encoding the committed matrix of the protocols' `[n = 16384, k = 4096]` RS code over BabyBear
(`backends/direct/PROTOCOL.md` section 3), nothing else of the prover.  Every kernel is bit-exact against the numpy
reference (`babybear.encode_rows`) on tiny sizes (laptop tests) and on 64 full rows before it is timed (pod).
Roofline: `ROOFLINE.md`.  Recorded run: **`r20260922-062016-6bdf`** (`research run --on vy-g2 --source .`, tree
`bdef01a..403cbfa`; CUDA-event timings, warm, median of 7; raw `results_r20260922-062016-6bdf.json`, envelope
`result.json` in the run, contract-valid).  Device: RTX 4090 (128 SMs, 24 GB, 1,008 GB/s), driver 580.159.04,
CUDA runtime 12.9, cupy 14.2.0, torch 2.4.1+cu124, Triton 3.0.0.  Earlier run `r20260922-061346-71f8` (v1 kernel
only) is superseded.

## Roofline vs measured, Candidate B shape (310,602 rows x 4096 -> 16384; 5.09 GB in, 20.36 GB out)

~~~text
encoder                                 floor (roofline)                    measured ms/batch   us/row   codeword GB/s   fraction of floor
HBM: read input + write codeword        25.2 ms  (write alone 20.2)         --
(a) NTT, SIMT, v2 t512  (best)          25.2 ms HBM  / 14 ms SIMT ops       32.2               0.104    632             0.78 of HBM floor   948 G butterflies/s, 790 GB/s
(a) NTT, SIMT, v2 t512, cosets only     20.2 ms HBM                         31.6               0.102    644             0.64
(a) NTT, SIMT, v1 t512                  25.2 ms HBM                         38.6               0.124    527             0.65
(b) INT8 matrix DFT, Triton fused       31.6 ms TC@1.0 (coset form)         171.7              0.553    119             0.184 of INT8 peak  60.7 T MAC/s
(b) INT8 matrix DFT, cuBLAS unfused     1,373 ms (1.4 TB INT32 traffic)     15,225 (scaled from 8,192 rows)  49.0   1   0.002 of INT8 peak
(c) dense generator matrix              757 ms TC@1.0                       not built (30x the memory floor)
(d) Brakedown expander (20 MAC/symbol)  39 ms SIMT / 3.2 s HBM gather       not built (more multiplies than the NTT at k = 4096; changes the Ligero test)
~~~

Candidate A shape (48,873 rows; 0.80 GB in, 3.20 GB out; HBM floor 4.0 ms): NTT v2 t512 **5.3 ms** (0.76 of floor),
Triton matrix DFT 26.8 ms, cuBLAS 2,396 ms (scaled).  Per-row times are shape-independent (0.10-0.11 us/row SIMT).
Peak device memory for the B batch: 5.5 GiB with 65,536-row slices (1.07 GB in + 4.29 GB out per slice; the
20.36 GB codeword is never resident -- see the streaming plan).  All SIMT variants in the run: v1 t256 40.2, v1 t512
38.6, v2 t128 34.8, v2 t256 39.6 (register spill at 16 elements/thread), v2 t512 32.2, v2 t1024 32.7 ms.

**A100 rows** (anchor-a100 lane, NVIDIA A100-SXM4-80GB, torch 2.6.0+cu124, cupy 14.2.0, Triton 3.2; run
`r20260922-064413-0508`, same bench and flags, all 8 variants bit-exact; floors at the A100's *measured* copy bandwidth
1,754 GB/s and datasheet 312 T INT8 MAC/s, `backends/numerical/reports/anchor_a100.md`):

~~~text
encoder (A100)                          B ms/batch   us/row   codeword GB/s   fraction of A100 HBM floor (14.5 ms)   vs 4090   A ms/batch
(a) NTT, SIMT, v2 t512                  60.7         0.195    336             0.24                                    0.53x     11.6
(a) NTT, SIMT, v2 t1024 (best here)     57.4         0.185    355             0.25                                    0.57x     11.0
(a) NTT, SIMT, v1 t512                  73.2         0.236    278             0.20                                    0.53x     13.8
(b) INT8 matrix DFT, Triton fused       607.1        1.955    34              0.055 of INT8 peak                      0.28x     94.6
(b) INT8 matrix DFT, cuBLAS unfused     9,146 (scaled from 8,192 rows)        0.004 of INT8 peak                      1.66x     1,439
~~~

The encoder does **not** scale with bandwidth: on the A100 it is 0.53x the 4090 (the bandwidth ratio is 2.02x) and sits at
0.24 of the HBM floor -- INT32/shared-memory-bound (the A100's INT32 rate and shared-memory bandwidth are ~0.47x the 4090's;
503 vs 948 G butterflies/s).  The 4090's 0.78 of *its* HBM floor was that card's higher ALU:bandwidth ratio, not a
memory-bound kernel.  Recommendation 5 below ("scales with bandwidth") and the A100/H100 scaling rows are superseded.

**Verdict.**  The SIMT NTT beats the fused tensor-core matrix DFT by **5.3x** and the cost model's tensor-core encoder
(`rs_matrix_ntt` at 0.172: 413 ms per B batch) by **13x**, at 0.78 of the read+write roofline.  The tensor-core kernel
reaches the utilisation the model assumed (0.184 vs 0.172) and still loses, because its compute floor at 100% of the
INT8 peak (31.6 ms) is already above the whole problem's memory floor: the O(N sqrt N) matrix DFT performs 340x the
INT32-equivalent multiplies of the O(N log N) NTT for a 16x faster pipe.  "Expressible as a matmul" was the wrong
criterion for this operation.

## Re-pricing (`reprice.py` -> `backends/numerical/reports/encode_gpu.json`; utilisation 0.172 for the still-modelled matmuls)

~~~text
design  gpu       enc ms  hash ms  matmul ms  simt ms  total ms  enc%  hash%  overhead   label
B       RTX4090    412.8     45.7        9.5     24.2     492.5   84%     9%  1.22e+07   model encoder rs_matrix_ntt@0.172 + measured SHA-256 (hash_gpu.md)
B       RTX4090     31.6     45.7        9.5     16.7     103.8   30%    44%  2.57e+06   MEASURED encoder + MEASURED SHA-256 (this card)
B       A100        15.6     22.6       10.1     70.6     120.5   13%    19%  2.99e+06   SCALING: encoder and SHA-256 x2.02 (bandwidth), other buckets at A100 peaks
B       H100         9.5     13.7        3.2     20.5      47.4   20%    29%  1.18e+06   SCALING: encoder and SHA-256 x3.32 (bandwidth), other buckets at H100 peaks
A       RTX4090     64.9      7.5      477.4    150.4     700.7    9%     1%  1.74e+07   model encoder + measured SHA-256
A       RTX4090      5.2      7.5      477.4    149.2     639.7    1%     1%  1.59e+07   MEASURED encoder + MEASURED SHA-256
~~~

B's overhead on the 4090 drops **4.7x** (1.22e7 -> 2.57e6) with the encoding bucket measured; A's by 9% (encoding
was 9% of A).  B's batch is now 104 ms: SHA-256 hashing 44%, encoding 30%, SIMT tests 16%, row combination 9% --
**B is hash+encode bound, and both are memory-streaming SIMT work, not tensor-core work.**  Labels: this is
`explore.tensor`'s cost model with two inputs replaced by measurements (`ARITHMETIC_DIAGNOSTIC`-level, scope `vu`,
2^-128, B = 4096, K = 1536, BabyBear^6, authentication excluded); witness generation, row combination, the
linear/quadratic tests, helper inversions, openings and serialisation remain modelled (ledger `omitted`).  The A100
and H100 rows are scalings by the bound that applies (bandwidth for the memory-bound encoder and for SHA-256, where it
is an upper bound since that kernel is compute-bound; INT8 / FP32 peaks for the modelled buckets), not measurements.
Note the A100 row gets *worse* than the 4090 because the model's SIMT bucket scales with FP32 peak (19.5 vs 82.6
TFLOPS); that bucket is unmeasured.  For comparison, A's tensor-core sumcheck measured at 0.014 utilisation
(`tc_sumcheck_4090.md`) puts A at ~1e8 on this card; B at 2.6e6 with two of four buckets measured.

## Recommendation for the b-ligero prover

1. **Encoder: the SIMT NTT (`encode_ntt_simt.RSEncoderSIMT`, v2, 512 threads), not a tensor-core matrix DFT.**
   Coset formulation: one DIF INTT over the k-domain, three DIT NTTs on the coefficient row shifted by
   `k^-1 omega_n^(c i)`; 98,304 butterflies + 12,288 twiddle multiplies per row; Montgomery R = 2^32 with twiddles
   in Montgomery form and data in plain form.  Port to Rust/CUDA as is (the kernel is 120 lines).
2. **Layout: coset-major codeword** `[c][j] = f(omega^c omega_k^j)`, i.e. column `c k + j` is evaluation point
   `omega^(4 j + c)`; coset 0 is the input row (systematic).  It is a public permutation of the natural order
   (`babybear.coset_to_natural`); Ligero's column sampling is uniform so either labelling is a valid domain, and the
   verifier's interpolation constants are computed for it.  The natural layout would cost a stride-4 scatter or a
   64 KiB shared-memory transpose per row (occupancy halves).
3. **Streaming plan: never materialise the 20 GB codeword.**  Encode in row slices (65,536 rows = 4.3 GB out; 5.5 GiB
   peak) and feed each slice to the column hash: SHA-256 over a column is sequential, so 16,384 column-hash states
   (each 32 B + a 64 B block buffer) are updated per slice and the slice is discarded.  With the hash lane's
   sequential-leaf SHA-256 (446 GB/s) that is ~46 ms of hashing + 32 ms of encoding per B batch, and the 20 GB write
   + the hash lane's 20 GB read both disappear (the encoder then runs at its SIMT floor, ~15-25 ms).  Keeping the
   systematic quarter (the payload is already in memory) saves a further 5 GB of writes: `write_systematic=False`.
   If the codeword must be resident for the quadratic test / openings, 1.07 GB in-slices + the 20.36 GB codeword
   = 21.4 GB fits the 24 GB card; the hash lane ran the full tree at 19.6 GiB.
4. **What not to do:** the two-stage INT8 matrix DFT (even fused at 18% of peak it is 5x slower and its floor is
   above the memory floor); cuBLASLt with INT32 intermediates (1.4 TB of traffic, 15 s); a Brakedown expander at
   this k (more multiplies than the NTT, a gather, and a worse Ligero test).
5. **A100/H100:** the encoder scales with bandwidth (x2.0 / x3.3), not with tensor peak; do not buy tensor FLOPS for
   encoding.

## Layout

| file | what |
|---|---|
| `babybear.py` | field constants (Montgomery `p^-1 mod 2^32`), roots, numpy NTT/INTT, reference `encode_rows` (coset / natural layouts), naive Vandermonde check, `coset_to_natural` |
| `encode_ntt_simt.py` | cupy `RawKernel` SIMT encoder: `v1` (two 16 KiB smem buffers, all stages in smem), `v2` (default: register-resident stages `m >= threads`, one 16 KiB buffer, coalesced strided ownership, Plonky3-style Montgomery reduce); `twiddle_tables` |
| `matdft_ref.py` | numpy reference of the two-stage limb-GEMM DFT (balanced int8 limbs, INT32 dots, INT64 recombination with `2^(8s) mod p`) |
| `encode_matdft_int8.py` | `RSEncoderMatDFT(variant="cublas")`: `torch._int_mm`, DFT-matrix limbs decomposed once and resident; `variant="triton"` dispatches to `matdft_triton.py` |
| `matdft_triton.py` | fused Triton kernel: one program per row, 4 DFTs x (split, 16 `tl.dot`, recombine, twiddle, `tl.trans`, split, 16 `tl.dot`, recombine) |
| `roofline.py`, `ROOFLINE.md` | floors per candidate shape |
| `bench.py` | the recorded measurement (slices, bit-exact check, CUDA events, median of `--reps`) -> `bench.json`; `envelope.py` wraps it in the `contract` result envelope (python 3.12) |
| `reprice.py` | measured encoder + measured hash into `explore.tensor` (`Priced.times(encoder_seconds_per_row=...)`, CLI `--encoder-seconds-per-row`) -> `reports/encode_gpu.json` |
| `tests/test_encode.py` | reference checks (laptop, < 1 s) and GPU bit-exactness for every kernel variant (skipped without CUDA) |

## Running

~~~bash
# laptop: reference tests (GPU tests skip)
uv run --frozen --package verity-numerical python -m pytest backends/direct/encode/tests -q

# pod (vy-g2: system python3 has cupy 14 + torch 2.4.1 + triton 3.0; venv312 for the envelope); commit first
uv run --frozen research run --on vy-g2 --source . --project verity --stage b_encode.bench --require-result -- \
  bash -c 'export PATH=/usr/local/cuda-12.4/bin:$PATH; cd backends/direct && python3 -m encode.bench --reps 7 --out "$RESEARCH_RUN_DIR/bench.json" && \
           PYTHONPATH=../numerical/python:../../packages/verity/src /workspace/venv312/bin/python encode/envelope.py --bench "$RESEARCH_RUN_DIR/bench.json" --out "$RESEARCH_RUN_DIR/result.json"'
uv run --frozen research fetch <run-id> --all

# re-price
uv run --frozen --package verity-numerical python backends/direct/encode/reprice.py --bench backends/direct/encode/results_<run>.json --run-id <run> --out backends/numerical/reports/encode_gpu.json
uv run --frozen --package verity-numerical python -m verity_numerical.explore.tensor --hash-rate 446 --encoder-seconds-per-row 1.04e-7
~~~

## API

~~~python
import sys; sys.path.insert(0, "backends/direct")
import cupy as cp
from encode.encode_ntt_simt import RSEncoderSIMT

enc = RSEncoderSIMT()                                   # k = 4096, n = 16384, v2, 512 threads
rows = cp.asarray(...)                                  # (R, 4096) uint32, values < p, on device
code = enc(rows)                                        # (R, 16384) uint32, coset layout; code[:, :4096] == rows
enc(rows, out, write_systematic=False)                  # only cosets 1..3 written into out[:, 4096:]
~~~

Ledger: `backends/numerical/reports/ledger/b-encode.jsonl` (4 component entries, 1 `ARITHMETIC_DIAGNOSTIC` vu entry
flagged breakthrough); plot `backends/numerical/reports/plots/overhead_b-encode.png`.

Not done: a Rust port; the fused encode->column-hash kernel (recommendation 3, the next lever); the natural-order
layout variant; radix-4 stages (the kernel is at 0.78 of the memory floor, so the remaining gain is <= 1.3x);
A100/H100 measurements (scaled only).


---

# b-commit lane: fused commitment and B's remaining buckets (2026-09-22, runs `r20260922-064722-07c7` commit, `r20260922-065537-279b` buckets)

**Scope: component measurements** of every remaining B-Ligero prover bucket on the same RTX 4090 (driver 580.159.04, CUDA
runtime 12.9, cupy 14.2.0), CUDA-event timings, warm, median of 7, each kernel **bit-exact** against a numpy / Python
reference before it is timed (`tests/test_commit_stream.py`, `tests/test_buckets.py`; the bench re-checks on a small
shape).  Inputs are resident pseudo-random data (data-independent cost).  Re-pricing and the honest label:
`backends/numerical/reports/b_measured_4090.md` (`ARITHMETIC_DIAGNOSTIC`-level: not a proof, buckets measured separately).

## Candidate B shape (310,602 rows x [16384, 4096]; 33,485 product rows; 125.9M helpers; t = 192), ms per batch

~~~text
bucket                                    kernel / configuration                                              ms/batch   us/row    floor                                      of floor   bit-exact
commitment, Blake3 leaf, fused            commit_stream: v2 encode -> 512-row L2 slice -> Blake3 chunk CVs      46.7       0.150    unfused encode 32.2 + Blake3 56.6 = 88.8  1.90x      yes (vs unfused + CPU)
  ... + cosets kept (15.3 GB store)       same, cosets streamed to a resident (R, 3k) store for tests/openings  49.4       0.159    -- (+2.7 ms for the 15.3 GB write)
  encoder half alone (L2 slices)          v2, 1024 threads, 64 regs, 1 CTA/SM, 512-row slices                   35.7       0.115    SIMT ~14 ms (ROOFLINE.md)                  0.39
  Blake3 half alone                       blake3_cols_chunks, schedule unrolled, 32,768 chunks per slice        15.4       0.050    20.4 GB hashed = 1.32 TB/s
commitment, SHA-256 leaf, fused           commit_stream: 16,384 sequential column states, 1024-row slices       58.3       0.188    unfused 78.0                               1.34x      yes
  SHA-256 half alone                      sha256_cols_update, one thread per column                             30.0       0.097    ~2,000 INT32 ops / 64 B block = INT32 issue rate: at the ALU floor
IRS row combination v = r^T W             combine_scalar, k-wide rows, F_c scalar per row, chunk 1024            5.7       0.018    read 5.09 GB once = 5.0 ms                 0.89       yes
quadratic test p_0 (8,192 points)         3 encoder calls (coset 1) per product row + combine_quadratic, s512    7.8       0.232*   6 transforms per product row                          yes
linear test q_add (8,192 points)          7 encoder calls per row (6 coord. rows of r^T A + witness) + combine  149.4      0.481    14 transforms per row                                 yes (r^T A generation NOT included)
helper inversions, 125.9M BabyBear^6      batch_inverse: Montgomery trick, 262,144 threads x 481, Fermat inv.   28.8       0.23 ns  traffic 15 ms                              0.52       yes (Python-int F_c)
column openings, t = 192                  gather_columns from the kept store (+1.3 ms host Merkle paths)          3.8 (+1.3)         239 MB opened                                         yes
  ... without a store                     gather-mode v2 encoder (encode every row, write the 192 positions)    32.2                (a full re-encode's compute)                          yes
                                                                                                     sum      246 measured + 12.5 modelled (r^T A, witness, PRG) = 258.7 ms -> overhead 6.41e6
~~~

\* per product row.  Candidate A shape (48,873 rows): fused Blake3 commitment 7.4 ms (1.96x vs unfused), SHA-256 9.3 ms;
IRS row combination 1.0 ms; openings 0.66 ms from the store / 5.3 ms re-encoded (+1.4 ms paths).  Slice sweep (B, Blake3):
256 rows 49.7-52.2 ms, 512 47.2, 1024 46.7-47.1, 4096 58.7-59.3 (slice leaves L2), 65,536 (HBM-sized) 55.4-55.7.

**What changed.**  (1) The commitment is 1.9x faster fused and needs 6.3 GiB instead of 20.6 GiB: the codeword never
leaves L2.  Blake3 with the 7-round message schedule unrolled to literal indices hashes the 20.4 GB codeword in 15.4 ms
(the reference-style `m[MSG_PERM[i]]` loop of `hash_gpu` indexes a register array with a constant-memory value and spills:
359 GB/s there, 1,320 GB/s here); SHA-256 sits at the 4090's INT32 issue rate (30 ms) and cannot go faster on this card.
The encoder (34-36 ms in L2 slices) is now the commitment's bottleneck, 0.39 of its SIMT floor.  Two-stream overlap of
encoder and hash gains nothing: both are ALU-bound, and the v2 encoder (120 registers x 512 threads, or 64 x 1024) owns
every register of the SM; capping it to leave room made the sum slower.  (2) **B's dominant bucket is the linear test,
not hashing or encoding**: `q_add` needs the `deg < k` interpolant of every row of `r^T A` -- an `F_c` row, i.e. six
BabyBear coordinate rows, each an INTT + a coset NTT -- times the re-encoded witness row: 14 size-4096 transforms per row
against the commitment's 4, 149 ms = 58% of the batch.  The cost model priced the whole row-combination bucket as one
tensor-core matmul of 5.4e11 MACs (9.5 ms at utilisation 0.172) and did not count the `r^T A` encodings.  (3) With every
bucket measured B is **259 ms per batch on the 4090 = 63 us/VU, overhead 6.4e6** (was 2.65e6 with the encoder and hash
measured and the rest modelled); H100 scaling 7.9e6, A100 1.4e7 (INT32 issue-rate ratios 0.81 / 0.47, labelled scalings).
No bucket of B is a tensor-core operation; the IRS combination and the inversions are memory-streaming, the rest is
integer SIMT arithmetic.

**Still modelled**: `r^T A` generation (the sparse per-unit constraint combination, 8.9 ms in the model), witness
generation (0.4 ms in the model; no GPU witness kernel exists on main -- the b-ligero lane's generator is Rust/CPU),
prover randomness (3.2 ms at an assumed 100 GB/s).

## Conventions for the b-ligero Rust/CUDA port

* **Codeword layout: coset-major systematic** (`babybear.encode_rows(layout="coset")`): column `c k + j` (`c < 4`,
  `j < k`) is the evaluation at `omega^(4 j + c)`, `omega` = `31^((p-1)/16384)`; columns `< k` are the message values
  (the input row).  `coset_to_natural` is the public permutation to the natural order; Ligero's column sampling is
  uniform so either labelling is a valid domain (the verifier evaluates its constants for this one).  The 8,192-point
  domain of the tests is cosets 0 and 1 = the first `2k` columns.  Randomizer positions are the last `k - l = 256`
  message symbols of a row (values, like the payload: the INTT is part of the protocol).
* **Arithmetic**: Montgomery `R = 2^32`, `p^-1 mod 2^32 = babybear.P_INV_MOD_R`; twiddles and challenge coefficients in
  Montgomery form, data plain, so one `mont_mul(data, coeff_R)` is the plain product.  Stage twiddles
  `tw[m + t] = omega_(2m)^(+-t)`, coset tables `k^-1 omega_n^(c i)` stored at the bit-reversed coefficient position
  (`encode_ntt_simt.twiddle_tables`).  `F_c = F_p[X] / (X^6 - 31)`, coordinate-major storage `(6, N)`; `F_c x F_t`
  products are six independent `F_t` products (the tests never multiply two `F_c` elements; only the inversion does).
* **Column hash / leaf**: leaf `x` = H(column `x` of the coset-major codeword, each element as 4 little-endian bytes,
  row 0 first).  Blake3: the spec's chunk tree (1 KiB chunk = 256 rows; chunk counter = global chunk index; `ROOT` on the
  final node), digest = 8 little-endian words.  SHA-256: FIPS padding with the bit length `32 R`, digest = 8 big-endian
  words (`sha256_cuda.digests_to_bytes`).  Streaming state: SHA-256 one 8-word state per column, folded 16 rows at a
  time (slices multiples of 16 rows; the `< 16`-row tail and padding in the final kernel); Blake3 one chaining value per
  (column, chunk) in a `(nchunks, 16384, 8)` table (slices multiples of 256 rows), then `blake3_cols_parents`.
* **Tree**: `hash_gpu.merkle`: node = SHA-256(left || right) / Blake3 parent-node compression (`PARENT`, counter 0,
  `ROOT` on the last), leaves in column order `0 .. 16383`, an unpaired node promoted unchanged, depth 14.  Opening of
  column `x`: the column (`R` u32) + 14 sibling digests, leaf level first (`buckets.merkle_paths`).
* **Slice pipeline**: encode 512-1024 rows into a `(rows, 3k)` L2-resident buffer (`write_systematic=False`, coset `c`
  at column `(c - 1) k`), hash the slice reading coset 0 from the input rows, discard; optionally also stream the cosets
  to a resident `(R, 3k)` store (15.3 GB at B: `keep_codeword=True`) for the quadratic test and the openings -- at 2.7 ms
  it is cheaper than re-encoding for the openings (32 ms) and for the tests.  Capture the loop as a CUDA graph (600
  slices x 3 launches otherwise cost more than the GPU work at these sizes).
* **Tests** (`buckets.py`): thread per (column, 32-row chunk) accumulating six `F_t` lanes, chunk partials reduced once
  at the end; `p_0` term `r_i (x y - z)` with `r_i` supplied as `(r R, r R^2)` so the plain `x y R^-1` product needs no
  extra reduction; `q_add` term `R_c[i][j] u[i][j]` with `u` lifted by `R^2`.  Inversion: strided ownership
  (`idx = s T + t`), prefix products in a global scratch, one `a^(p^6 - 2)` per thread.

## Layout (b-commit files)

| file | what |
|---|---|
| `commit_stream.py` | `StreamingCommitter(hash, slice_rows, overlap, graph, keep_codeword)`: fused encode -> column hash -> root; SHA-256 column states / Blake3 chunk CVs; `reference_leaves` (CPU) |
| `buckets.py` | `Combiner` (`combine_scalar` / `combine_quadratic` / `combine_linear` / `reduce_chunks`), `BatchInverter`, `ColumnGather`, `merkle_paths` |
| `buckets_ref.py` | numpy / Python-int references (`F_c` arithmetic, `is_irreducible_binomial`) |
| `encode_ntt_simt.py` | v2 gains `ncosets`, cosets-only `(R, 3k)` output, `min_blocks_per_sm` (register cap), `gather=True` + `.gather(rows, columns)` |
| `commit_bench.py`, `buckets_bench.py` | the recorded measurements (raw `results_commit_*.json`, `results_buckets_*.json`) |
| `reprice_measured.py` | `Priced.times(measured=...)` re-pricing -> `reports/b_measured_4090.json` / `.md` |

~~~bash
# pod (commit first):
uv run --frozen research run --on vy-g2 --source . --project verity --stage b_commit.bench --require-result -- \
  bash -c 'export PATH=/usr/local/cuda-12.4/bin:$PATH; cd backends/direct && python3 -m encode.commit_bench --reps 7 --out "$RESEARCH_RUN_DIR/bench.json" && \
           PYTHONPATH=../numerical/python:../../packages/verity/src /workspace/venv312/bin/python encode/envelope.py --bench "$RESEARCH_RUN_DIR/bench.json" --out "$RESEARCH_RUN_DIR/result.json"'
#   (buckets: python3 -m encode.buckets_bench --reps 7 ...)
uv run --frozen --package verity-numerical python backends/direct/encode/reprice_measured.py --commit backends/direct/encode/results_commit_<run>.json \
  --buckets backends/direct/encode/results_buckets_<run>.json --out backends/numerical/reports/b_measured_4090.json --md backends/numerical/reports/b_measured_4090.md
~~~

**Not done**: a Rust port; a faster encoder (the commitment's bottleneck at 0.39 of its SIMT floor: radix-4 stages and
fewer shared-memory passes are the levers); the `r^T A` generation kernel (needs the constraint system on the GPU);
a witness-generation kernel; a cheaper linear test (the 6 coordinate encodings per row are inherent to `q_add` over
`F_c` with `F_t` rows; only a structural shortcut through the per-unit constraint pattern could remove them --
**done by the b-lintest lane, next section**).
Ledger `backends/numerical/reports/ledger/b-commit.jsonl` (5 entries), plot `plots/overhead_b-commit.png`.


---

# b-lintest lane: the linear test through the structure of r^T A (2026-09-22, runs `r20260922-071843-ac9a`, `r20260922-072315-6497`, `r20260922-073017-0924` = M 1-4 with the lazy kernel)

**Scope: component** (the linear-test bucket, same RTX 4090, CUDA events, warm, median of 7, bit-exact against the
b-commit GPU path `Combiner.linear` and numpy in every run) and the vu-level re-pricing (`ARITHMETIC_DIAGNOSTIC`,
`backends/numerical/reports/b_measured_4090.md`).  The algebra, the structure of `A`, the soundness note and the roofline
are in **`LINTEST.md`**; the kernels in `lintest_structured.py`.

The protocol combines the linear constraints with POWERS of one element (`r_add = (1, rho, rho^2, ...)`, `PROTOCOL.md`
section 6), and every unit satisfies the same local constraint pattern, so with a unit-major constraint enumeration
`r^T A` on a Ligero row is a per-row `F_c` scalar times one of `M` fixed rows (`M = 3` for the chain layout: witness
pattern, helper pattern, the sum constraint's constant row) except on ~130 exceptional rows (unit 0, the last unit, the
table-side helper equations, the multiplicities).  Interpolation is linear, so `q = sum_m enc(E_m) * enc(sum_i s_im W_i)`
+ the general path on the exceptional rows: **one memory-bound pass over the witness instead of 14 transforms per row**.
Same test, same `r`, same `q`, same transcript -- a prover-side algorithm.

## Before / after, B shape (310,602 rows), ms per batch

~~~text
linear test q_add                                      ms/batch   transforms/row   transforms/batch   floor                 of floor   bit-exact
general path (b-commit; re-timed in the same run)      149.9-150.5   14              4.35M              35 ms HBM (r^T A 30.5 GB + W)   0.23   yes
structured, M = 1                                        5.66       0               12                 5.05 ms (read W once)   0.89     yes (vs general GPU path + numpy)
structured, M = 1 + 256 exceptional rows                 6.37       0               12 + 3,584         5.05                    0.79     yes
structured, M = 2 + 256 exceptional rows                 7.03       0               24 + 3,584         5.05                    0.72     yes
structured, M = 3 + 256 exceptional rows                 9.72       0               36 + 3,584         5.05                    0.52     yes   (INT32-issue-bound: 2.3e10 mont-muls)
structured, M = 4 + 256 exceptional rows                13.67       0               48 + 3,584         5.05                    0.37     yes
structured, lazy reduction (4 rows of 64-bit products per REDC), M = 2 / 3 / 4 + 256:   5.90 / 7.15 / 7.68           0.86 / 0.71 / 0.66   yes
  split (M = 3 lazy, the chain layout's configuration): combine 6.95 + encode V 0.07 + F_c pointwise 0.05 + exceptional rows 0.08; G = enc(E_m) once per batch 0.16
~~~

**Speed-up 26x (M = 1) / 21x (M = 3, lazy) / 11x (M = 4, plain)**; the linear test falls from 58% of the B batch to 6%.
The per-row cost of the exceptional rows is the general path's 0.48 us/row (256 rows = 0.12 ms).  The lazily reduced
kernel (`combine_multi_lazy`: 1 IMAD.WIDE + one 64-bit add per product, one Montgomery reduction per 4 rows) is what
keeps `M = 3, 4` near the memory floor; at `M = 1` the plain kernel is already there.

## Tensor cores vs SIMT for the pass over W (LINTEST.md section 4)

The pass is `V[m][c][j] = sum_i s[i][m][c] W[i][j]`: a `(24 M + 1)`-row INT8 limb GEMV against the 310,602-deep witness.
`torch._int_mm` (cuBLASLt INT8 -> INT32, exact recombination, bit-exact): **10.3 ms at M = 1 (0.49 of the read floor, 2.4%
of the INT8 peak), 12.4 at M = 2, 16.7 at M = 4**, plus an 11 ms byte-shift pass (`W ^ 0x80`, a 5 GB write) unless the
witness is stored shifted.  The SIMT kernel: 5.66 / 6.99 / 13.5 ms.  **SIMT wins at every M**: the operation is
memory-bound and cuBLASLt's skinny kernel reaches half the bandwidth the SIMT kernel does; the tensor cores' 0.25 M ms of
arithmetic never mattered.  A bandwidth-efficient hand-written tensor-core GEMV could gain at most ~8 ms at M = 4 (3% of
B) -- the only tensor-core opportunity left in B.  The dense `[rows x |P|] . [|P| x n]` formulation for a uniform `r`
would cost 170-330 ms at 100% of INT8 peak (LINTEST.md 2.4): the powers combination, not the tensor core, is what makes
the linear test cheap.

## B re-priced (reprice_measured.py --lintest; b_measured_4090.md)

~~~text
B, RTX 4090, Blake3 leaves, openings from the store:  116.4 ms per 4096-VU batch = 28.4 us/VU, overhead 2.89e6   (was 258.7 ms / 6.41e6: 2.2x)
  commitment 49.4 | linear test 7.15 (M = 3 lazy + 256 exc.; 13.7 with the plain M = 4 kernel -> 122.9 ms / 3.05e6) | IRS 5.7 | quadratic 7.8 | inversions 28.8 | openings 5.1 | modelled 12.5 (89% measured)
  scalings (labels, not measurements): H100 142.9 ms / 3.54e6, A100 262.7 ms / 6.51e6 by the INT32 issue-rate ratio -- the structured
  linear test is memory-bound, so for that bucket bandwidth scaling (A100 x0.65, H100 x0.30 of the 4090 time) is the right rule
~~~

B's buckets are now the fused commitment (49.4 ms, encoder-bound at 0.39 of its SIMT floor) and the helper inversions
(28.8 ms, 0.52 of traffic floor); the linear test is 6%.  The modelled `r^T A` generation (8.9 ms) is kept in the total although the structure
makes it `O(rows + |pattern|)` (LINTEST.md section 5).

## Conventions for the b-ligero Rust/CUDA port (in addition to the b-commit list above)

* **Constraint enumeration is unit-major**: block `u` holds every linear constraint that involves only unit `u` and its
  successor (the WP2 identities and splits, the alignment copies, the transition differences, unit `u`'s 306 helper-side
  `F_c`-linear equations and copies), in a fixed order `c < C`; index `u C + c`.  Boundary constraints (`row0_*`) and the
  table-side constraints (`h_v (z_T - t_v) = m_T(v)`, the five sums) follow the blocks.  `r_add[idx] = rho^idx`.  Any order
  gives the same `(#lin - 1)/|F_c|` bound; this one gives the structure.
* **Rows hold whole units.**  Witness/triple rows: `S = 3` units of 1 064 slots per 3 840-symbol payload (or the transposed
  layout "one AIR column x 3 840 consecutive units"); helper rows: whole units' `306 x 6` coordinates (12 units of 306 per
  coordinate row, or 2 units of 1 836 interleaved).  Randomizer positions `>= l` carry 0 in `r^T A`.
* **Prover algorithm** (`StructuredLinearTest`): per batch compute `pi = sum_c rho^c P_c + rho^(-C) sum_c rho^c P'_c`
  (own + neighbour pattern), `E_m` (`(M, 6, k)` plain), `G_m = enc(E_m)` on cosets 0 and 1; per-row scalars
  `s[i][m] = rho^(i S C)` etc. in Montgomery form `(rows, M, 6)`; `combine_multi` (thread per column, `chunk_rows` rows,
  `6 M` accumulators, `acc (nchunk, 6M, k)` then `reduce_multi`); `V_m` on coset 1 (`6 M` encoder calls, `ncosets = 1`);
  `fc_pointwise_accum` (`F_c x F_c`, plain inputs: `mont_mul` twice with `R^2`); exceptional rows through
  `Combiner.linear` and `add_mod_inplace`.  Output `q (6, 2k)` plain, identical to the general path's.
* **Verifier**: `sum_i R_i(eta) U_i(eta) = sum_m G_m(eta) sum_i s_im U_i(eta)` + exceptional rows: `M` scalar combinations of
  the opened columns (the IRS test's work) and `M` evaluations of the fixed rows per opened column; it need not
  materialise `r^T A` either.
* **Not a tensor-core operation**: implement the pass as the SIMT kernel; store `s` as 24 M bytes per row, never `r^T A`.

Files: `lintest_structured.py`, `tests/test_lintest_structured.py` (the structure test runs on the laptop), `LINTEST.md`,
`results_lintest_<run>.json`, `reprice_measured.py --lintest results_lintest_r20260922-073017-0924.json --lintest-config linear_test_structured_M3_X256_lazy`; ledger
`backends/numerical/reports/ledger/b-lintest.jsonl`, plot `plots/overhead_b-lintest.png`.

**Not done**: the `pi`/scalar generation kernel from the real constraint system (the toy `UnitPatternSystem` proves the
algebra; the b-ligero port's constraint emitter must adopt the enumeration); merging the helper-row classes to `M = 2`;
a bandwidth-efficient tensor-core GEMV for `M >= 3`; the radix-4 encoder (the remaining encoder gain is <= 1.3x on a bucket
that is now 40% of B).
