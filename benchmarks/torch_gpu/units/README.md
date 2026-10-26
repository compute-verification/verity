# units: the frozen decoder's non-matmul units on the H200

GPU kernels for the units between the matmuls of the frozen shape (MQA decoder, width 16384, 52 layers, 256 query heads of
64, one KV head, MLP 65536, byte vocabulary, context 32), each word for word its CPU twin, both in circuits' forms (the
CMP-96 replay's templates, CIR-100's quantizer). Linear HASH-19, for the walking skeleton (CMP-96).

| unit | GPU (units.cu) | twin and its op sequence | circuits' templates |
|---|---|---|---|
| RMSNorm | `units_rmsnorm`, a block a row | `rmsnorm.py` | RmsSumSqPart_v1, F32SumPart_v1, RmsNormRecip_v1, RmsNormOut_v1 |
| s8 quantizer | `units_quant`, a block a row | `quant.py` | RowAbsMax_v1, QuantS8Scale_v1, QuantS8Pair_v1 |
| softmax | `units_softmax`, a thread a row | `softmax.py` | SoftmaxMax_v1, SoftmaxRecip_v1, SoftmaxOut_v1 |
| argmax | `units_argmax`, a thread a row | `argmax.py` | ArgmaxChunkBest_v1, ArgmaxChunkIdx_v1, ArgmaxCombine_v1 |
| ReLU | `units_relu` | `relu.py` | relu16 (folded into the quantizer at down_quant) |

Each module's docstring states its OPS: every f32 operation an IEEE round-to-nearest-even instruction with no flush, every
reduction sequential in ascending order from +0, no atomics. The one approximate operation is softmax's exponential,
`ex2.approx.ftz.f32` (MUFU.EX2, `verity.ml.circuits.ops.mufu.MufuEx2Ftz`). `words.py` holds the word operations the twins
are made of; `gpu()` in each module launches its kernel on torch tensors through `lib.py`, and with `units=True` also
returns the intermediate unit outputs (chunk sums, rs, the absmax tree, m, r, chunk bests).

`sweep.py` runs the kernels' helpers on every operand word (ex2, the s8 conversion, sqrt.rn, 1/b, a/16384, the quantizer's
division by every scale, the bf16 rounding, the scale) and reduces each block of 2^24 to a digest, so the H200's MUFU.EX2
is held to `MufuEx2Ftz` (measured on sm_89) on all 2^32 inputs.

## Running

```text
# on a CPU node: the cubin, build_check.py, the harness SASS gate's verdict, and the fixture (pools, words, sweep digests)
research run --on auto --cwd source --cpus 16 --mem-gb 32 --send replay.py --env PROCS=16 \
  -- bash -lc 'uv sync -q --all-packages --extra torch-cpu && bash benchmarks/torch_gpu/units/build.sh "$OUT/ship"'
# on the H200: the sweeps, the check at B = 1, 7, 128 and 20480, then timing
tar -C BUILD_RUN/out/ship -cf /tmp/units-ship.tar .
research run --on vy-dc-h200 --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
  --send /tmp/units-ship.tar -- bash benchmarks/torch_gpu/units/session.sh
```

`fixture.py --replay` holds every pool word of the twins to the CMP-96 replay's references (e2e_h200's replay.py, #4460;
`build.sh` passes it a sent replay.py). `tests/` run on a CPU: the word operations against the fp32 primitives'
evaluators, each twin against a scalar composition of them, the sweep digest, and `build_check.py`.

The harness SASS gate (`benchmarks/pouw/harness/sass_gate.py`) refuses MUFU.EX2, and has no sm_90a pin for div.rn's and
sqrt.rn's sequences; `build.sh` records its verdict without refusing on it. Every division takes runtime operands and no
kernel divides integers, so a pinned sm_90a toolkit would leave only softmax's MUFU.EX2 flagged.
