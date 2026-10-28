# tc_probe_fp4: the Blackwell FP4 path against its models

Verity's circuits for FP4 arithmetic use the exact semantics of Blackwell's block-scaled tensor-core instructions
(`verity.ml.circuits.silicon.models`: `BLACKWELL_SM120_NVF4`, `BLACKWELL_SM120_MXF4`). These probes run the instructions
on the GPU, capture the raw output bits, and compare every word with the pinned model or with the candidate models a
capture decides between. The rest of the directory prices the FP4 path: what one MMA costs, and whether any other route
computes the same words for less.

Every script's docstring gives its command lines, the instructions it covers and what its record holds. Captures run on a
GPU of the architecture with nvcc and `cuobjdump`, as `research run` jobs on a node (the writing-runs skill). CPU halves
(`verify`, `eval`) and fake-silicon dry runs need no GPU.

## sm_120 (RTX 5090, RTX PRO 6000): `mma.sync`

- `probe.py`: the probe. One `mma.sync` per tile (kernels in `mma_fp4.cu`, rows in `rows_fp4.cuh`), raw bits in and
  out, chains driven from the host by feeding D back as C. `--instruction` picks the row (`nvf4`, `mxf4`, the K = 32
  kinds, the conversions, the sparse rows; `--help` lists them); `--sass-check` confirms each tile kernel holds exactly
  one MMA; `--sweep` runs the input families (random, uniform encodings, every scale code, the alignment window's edges,
  signed zero) against the models. It writes `result.json`, whose validation passes only when every mismatch count is
  zero, beside the full capture.

  ```text
  uv run python benchmarks/tc_probe_fp4/probe.py --instruction nvf4 --out DIR --sass-check --sweep
  ```
- `model5.py`: an earlier five-way align-add model of the NVFP4 tile, which the probe refuted on sm_120 and keeps as its
  control.
- `recheck.py`: fresh-seed rechecks beyond the probe's gate, split into a GPU `capture` and a CPU `verify`.
- `lean_vectors.py`: named NVFP4 conformance vectors from a generated Lean vector file, captured on the card.
- `cast.py` (kernels in `cvt_fp4.cu`): the FP4-path conversions (`cvt` to and from E2M1 and UE8M0) against exact
  references.
- `k32.py`: the K = 32 rows with an E2M1 operand, which lower to the FP8 datapath (QMMA) rather than the FP4 one (OMMA):
  their encodings and the candidate models the capture decides between.
- `sparse.py`: the 2:4 sparse FP4 rows (`mma.sp::ordered_metadata` m16n8k128) against candidate models; `meta_rule.py`:
  what the device does with each of the 16 values a metadata nibble can hold, the 10 invalid ones included.
- `sp_e4m3.py` (kernels in `sp_e4m3.cu`): the 2:4 sparse E4M3 row against two dense K = 32 atoms on the same operands.
- `t1_merged.py`: how many words one sparse m16n8k128 instruction reproduces against two chained dense m16n8k64 atoms, on
  the operands the FP4 emulation script feeds its merged route. `fp4_emulation_1ad1aaa2.py` is that script, verbatim (it
  asks which non-native routes reproduce the NVFP4 chain's words, on the CPU), its origin in `t1_merged_origin.json` and
  its bytes checked by `t1_merged.py`; `emulation.py` imports it under the module name it was written against.
- `stage0_sm120.py`: which FP4-path mnemonics ptxas accepts for sm_120a and what each lowers to in SASS, before any
  semantics. Nothing executes.

## sm_100 and sm_103 (B200, B300): `tcgen05.mma`

- `tcgen05_fp4.cu`: one `tcgen05.mma kind::mxf4nvf4` per tile, run through `probe.py` (`--instruction tcgen05_nvf4`,
  `tcgen05_mxf4`, `tcgen05_mxf4kind`), to ask whether the datacenter part computes sm_120's words.
- `f8_sm100.py` (kernels in `tcgen05_f8.cu`, driven by `f8_sm100.sh`): the FP8 step of `kind::f8f6f4` and
  `kind::mxf8f6f4`, in three phases: `gen` (CPU), `capture` (GPU) and `eval` (CPU). `f8_model.py` is the vectorized
  candidate step `eval` scores 10^8 captured words against.

## Prices and cheaper routes, on sm_120a

- `w1_price.py` (timing loops in `w1_price.cu`): what one MMA of each FP4-path row costs on the card, sustained issue rate
  and dependent-chain latency, on the instruction the probe's capture gates.
- `f1_rates.py`: every multiplier ptxas accepts for sm_120a, each `mma.sync` and `mma.sp` kind, shape, type and sparsity
  and the CUDA-core multipliers, priced per nonzero product against dense NVFP4 in the same process.
- `f2_strassen.py` (`f2_strassen.cu`): one- and two-level Strassen over one NVFP4 group on the TF32 tensor core, against
  dense NVFP4, word for word.
- `f3_lut.py` (`f3_lut.cu`): LUT-GEMM over the 16-code E2M1 alphabet with tables from the weights, against dense NVFP4.
- `nvf4_step.cuh`: the NVFP4 atom's align-add on integers, shared by the two CUDA-core routes.

## Records and tests

- `tool.py`: the store declarations, which say which parameters key each result: `tc_probe_fp4`, `tc_recheck_fp4`,
  `tc_sparse_e4m3`, `tc_price_fp4`, `tc_rates_fp4`, `tc_strassen_fp4` and `tc_lut_fp4`.
- `tests/`: the CPU side of each probe, without a GPU: the candidate models against verity's own reference steps, the
  cast references against brute force, the sparse rows' stored-to-logical mapping, and fake-silicon dry runs in which each
  candidate in turn is the device's truth and must be the only one reproducing every word.
