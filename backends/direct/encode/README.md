# encode -- Reed-Solomon encoders for the Ligero witness matrix, measured on an RTX 4090 (b-encode lane, 2026-09-22)

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
