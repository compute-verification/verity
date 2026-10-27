# units: the frozen decoder's non-matmul units on the H200 and the RTX PRO 6000

GPU kernels for the units between the matmuls of the frozen shape (MQA decoder, width 16384, 52 layers, 256 query heads of
64, one KV head, MLP 65536, byte vocabulary, context 32), each word for word its CPU twin, both in circuits' forms (the
CMP-96 replay's templates, CIR-100's quantizer, CIR-104's row units). Linear HASH-19, for the walking skeleton (CMP-96).
`build.sh` builds `units.cu` for sm_90a (the H200) and sm_120a (the RTX PRO 6000), and `lib.py` loads the cubin of the
device's arch.

| unit | GPU (units.cu) | twin and its op sequence | circuits' templates |
|---|---|---|---|
| RMSNorm | `units_rmsnorm`, a CTA of 128 a row, staged in shared memory | `rmsnorm.py` | RmsSumSqPart_v1, F32SumPart_v1, RmsNormRecip_v1, RmsNormOut_v1 |
| s8 quantizer | `units_quant`, a CTA a row in registers; `units_quant4`, a cluster of four at 65536 | `quant.py` | RowAbsMax_v1, QuantS8Scale_v1, QuantS8Pair_v1 |
| softmax | `units_softmax`, a thread a row, 128 rows a CTA through shared memory | `softmax.py` | SoftmaxMax_v1, SoftmaxRecip_v1, SoftmaxOut_v1 |
| argmax | `units_argmax`, a thread a row | `argmax.py` | Argmax_v1; ArgmaxChunkBest_v1, ArgmaxChunkIdx_v1, ArgmaxCombine_v1 |
| ReLU | `units_relu`, uint4s | `relu.py` | relu16 (folded into the quantizer at down_quant) |
| residual add | `units_add` | `add.py` | Bf16AddRow_v1 (Bf16AddF2fp_v1 per element) |

Each module's docstring states its OPS: every f32 operation an IEEE round-to-nearest-even instruction with no flush, every
reduction sequential in ascending order from +0, no atomics. The one approximate operation is softmax's exponential,
`ex2.approx.ftz.f32` (MUFU.EX2, `verity.ml.circuits.ops.mufu.MufuEx2Ftz`). `words.py` holds the word operations the twins
are made of; `gpu()` in each module launches its kernel on torch tensors through `lib.py`, and with `units=True` also
returns the intermediate unit outputs (chunk sums, rs, the absmax tree, m, r, chunk bests).

`sweep.py` runs the kernels' helpers on every operand word (ex2, the s8 conversion, sqrt.rn, 1/b, a/16384, the quantizer's
division by every scale, the bf16 rounding, the scale, the residual add of every pair of bf16 words) and reduces each
chunk of 2^24 operands to a digest, so MUFU.EX2 is held to `MufuEx2Ftz` (measured on sm_89) on all 2^32 inputs on each
card, and the quantizer's quotient (RN(1 / scale) once a row and one FMA correction, div.rn outside its window) to div.rn on
every bf16 word and every amax. The sm_120a cubin also sweeps MUFU.RSQ, which no unit reads: the bare
`rsqrt.approx.ftz.f32` against `mufu.mufu_rsq_words` over circuits' `rsq` table, and `rsqrt.approx.f32` against
`mufu.RsqrtApprox`; its `units_sweep_words` gives a disagreeing chunk's words, and `check.py` names the first that differs.

`bench.py` times each unit on the GPU's ruler (a sleep ahead of the start event, so the host's launch time isn't counted)
and the host-inclusive one, and a decode step's units both summed and launched as one; `check.py` runs it after the check,
and `session.sh` also runs it on another ship's modules (`units-old.tar`) in the same lease.

## Running

```text
# on a CPU node with CUDA 13.0: both cubins, build_check.py, the harness SASS gate's verdicts, and the fixture (pools,
# words, sweep digests)
research run --on vy-nebius-cpu-1 --pool research --project verity --source . --cwd source --env PROCS=16 \
  -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && bash benchmarks/torch_gpu/units/build.sh "$OUT/ship"'
# on the H200: the sweeps, the check at B = 1, 7, 128, 4096, 16384 and 20480, then timing at 4096 and 16384 (with
# units-old.tar, an earlier build's ship, timed beside it)
tar -C BUILD_RUN/out/ship -cf /tmp/units-ship.tar .
research run --on vy-dc-h200 --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
  --send /tmp/units-ship.tar [--send /tmp/units-old.tar] -- bash benchmarks/torch_gpu/units/session.sh
# on the RTX PRO 6000 (vy-nebius-2): the sweeps alone, ops 0 to 10
research run --on vy-nebius-2 --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE --env SWEEP_ONLY=1 \
  --send /tmp/units-ship.tar -- bash benchmarks/torch_gpu/units/session.sh
```

The fixture holds every pool word of the twins to circuits' references and refuses to write on a disagreement:
`--replay` to the CMP-96 replay's (`benchmarks/e2e_h200/replay.py`, which `build.sh` passes unless REPLAY names another),
and `--rows` to `ops.rows`' `argmax` and `add_row` (CIR-104; ROWS names a file to load in its place). `tests/` run on a
CPU: the word operations against the fp32 primitives' evaluators, each twin against a scalar composition of them, the
sweep digests and models, and `build_check.py`.

The harness SASS gate (`benchmarks/pouw/harness/sass_gate.py`) passes div.rn's and sqrt.rn's sequences under its pins, and
excepts MUFU.EX2 only under 13.0.1-sm_90a, on the H200 sweeps' evidence; `build.sh` records its verdict on each cubin
without refusing on it, so sm_120a's MUFU.EX2 and the sweep's bare MUFU.RSQ are listed there as flagged.
