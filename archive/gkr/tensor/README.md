# archive/gkr/tensor: A-GKR's GKR round on INT8 tensor cores

One kernel family, measured: the degree-3 GKR c-round over BabyBear with a degree-6 extension challenge (2^-128), as
INT8 tensor-core matrix multiplications in TensorZKP's limb layout.  Labels on every number: `ARITHMETIC_DIAGNOSTIC`,
`authentication excluded`, `security.target = -128`; the hardware is in the fingerprint.  Not a proof.

~~~text
reference.py       BabyBear^6 (X^6 - 22) in Python ints; the naive round (S(0..3) and the folds)
sumcheck_int8.py   the kernel: balanced signed 8-bit limbs, round = 4 Hadamards (SIMT) + limb decomposition +
                   8 triple inner products as one split-K INT8 matmul (INT32 over <= 32768-row chunks, INT64 chunk
                   sums, one reduction per inner product) + 3 folds as limb matmuls (scalar convolution matrix, X^6 = W
                   applied inside the matrix) + recomposition; numpy / torch-CPU / torch-CUDA from the same code
bench.py           the measurement driver (bench.json + research/result/v0.1 envelope) and --reprice (laptop)
tests/             kernel == reference bit-for-bit on t <= 64 (laptop, numpy, < 1 s); CUDA + Triton paths on the pod
~~~

## Run

~~~sh
# laptop: agreement test (numpy only; torch CPU too when installed)
uv run --package verity-numerical --with numpy --with pytest pytest archive/gkr/tensor/tests -q

# pod vy-sp1 (venv: uv venv /workspace/tc/venv; uv pip install torch numpy pytest --index-url https://download.pytorch.org/whl/cu124)
uv run --package research research run --on vy-sp1 --project verity --source . --stage tc.sumcheck.4090 --require-result -- \
    /workspace/tc/venv/bin/python archive/gkr/tensor/bench.py --compile --repeats 7
uv run --package research research fetch <run> --all

# laptop: re-price explore.tensor with the measured utilisation
uv run --package verity-numerical --with numpy python archive/gkr/tensor/bench.py \
    --reprice ~/.research/runs/<run>/bench.json --report backends/numerical/reports/tc_sumcheck_4090
~~~

`bench.py` options: `--inner triton|int_mm` (split-K Triton kernel vs a chunk loop of `torch._int_mm`), `--compile`
(torch.compile the Hadamard / limb passes into one kernel each), `--layout reduced|paper` (48- or 96-column fold output),
`--chunk`, `--ext-max-log2`, `--base-log2`, `--ceiling`.

## Recorded runs (RTX 4090, driver 580.159.04, CUDA 12.4, torch 2.6.0+cu124)

| run | what | layer-model kernel time | graphed wall | notes |
|---|---|---|---|---|
| r20260922-051730-15d4 | the table: Triton split-K inner product, compiled SIMT passes, median of 7 | 87.4 ms / 130 rounds | 91.4 ms | agreement test 22 passed on CUDA |
| r20260922-052015-a62d | `--inner int_mm`: cuBLAS chunk loop for the inner product | 30.2 ms for the 2^21 round alone (vs 4.6) | | `_int_mm(48, 32768, 96)` = 821 us: no split-K |
| **A100** r20260922-074359-964b (anchor-a100 lane, NVIDIA A100-SXM4-80GB, torch 2.6.0+cu124, Triton 3.2, same flags) | the table on the anchor | 115.5 ms / 130 rounds (0.76x the 4090) | 115.6 ms (0.79x) | 22 passed on CUDA; 2^21 deg-6 round 4,894 us (0.95x), 2^18 1,044 us (0.58x); matmul kernels 98.2 ms (0.82x); `_int_mm` 8192^3 = 14.5 ms = **0.121 of the A100 INT8 peak** (4090: 0.253), `(48, 32768, 96)` 1,547 us (0.53x); table in `note:r20-proof/anchor-a100/20260922T0820Z-report-anchor-a100` |

Per-round results, ceilings, effective utilisation and the re-priced overheads: `note:r20-proof/tc-sumcheck/20260922T0524Z-report-tc-sumcheck-4090`
(`.json` beside it).  Headline: matmul kernels reach 2.2% of the INT8 peak batch-weighted over the 130 rounds (3.6-4.6% in
the large rounds, HBM-bound at 0.8-0.9 TB/s), the tensor bucket 0.56%; model-equivalent utilisation 0.056 / 0.014 /
0.012 against the assumed 0.172; `_int_mm` 8192^3 reaches 0.253 of peak on this card.

## Layout notes

- Limbs are *signed balanced* (`l in [-128, 128)`, 4 per BabyBear coefficient, 24 per extension element) because
  `_int_mm` / cuBLAS / `tl.dot` multiply signed INT8; a product is < 2^14 so the INT32 bound is N < 2^17 (the paper's
  UINT8 bound is 33,025).  Chunks stay at 32768 rows as in the paper.
- Inner product: `F = [f_e | f_o]` (t/2 x 48) against `GH = [g_e o h_e | g_e o h_o | g_o o h_e | g_o o h_o]` (t/2 x 96):
  the 48 x 96 INT32 block holds all eight `T_abc`; `S(0..3)` are integer combinations of them (points 0, 1, 2, 3).
- Fold: `[x_e | x_o] x [C_(1-r); C_r]` with the scalar convolution matrix of TensorZKP 4.1; the `X^6 = W` wrap is folded
  into the matrix (the wrapped blocks carry the limbs of `W c_t mod p`), 48 output columns instead of the paper's 96
  (`--layout paper` reproduces the 96).
- The round's model-vs-performed MAC count: `explore.tensor.sumcheck_macs` prices 32 t dl^2 per round of t elements,
  the kernel performs 16 t dl^2 (inner products and folds are over t/2 pairs); the report gives both utilisations.
- The `c` rounds are what this measures (3 tables `eq, V_l, V_r` over units x wires).  Round 1 is base x base x base
  as specified by the lane; an extension-valued `eq` table would make round 1 cost like an extension round for that
  factor.  The per-circuit `l, r` rounds and the logUp fractional GKR are not run.
