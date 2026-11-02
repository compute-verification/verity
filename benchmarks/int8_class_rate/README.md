# int8_class_rate: the H200's sustained int8 rate per operand class

What one H200 sustains on int8 tensor-core work for each operand class (`Verity.OperandClass`: s8 uniform on
[-128, 127], s2 on [-2, 1], u2 on [0, 3], u4 on [0, 15], zero), alone and beside SHA-512 paced at a ratio of
compressions a T MAC, at the declared clock (1830 MHz, locked under the lease). It is the measured reading of the
capacity a card registers per class; the bound reads the data sheet until a reading is accepted. Timings only: every
rate and clock comes from the kernels' own `clock64` over `%globaltimer`, and the one `nvidia-smi` query is of clocks.

- `class_rate.cu`: the kernels. `wgmma` is one warpgroup an SM issuing wgmma m64n256k32
  back to back on shared-memory operands, in three operand encodings: s8·s8, and the offset re-encoding as u8·u8 or
  s8·u8. `lt` is cuBLASLt's int8 GEMM at 8192³, its output checked at sampled entries. Beside each, SHA-512 runs on
  a second stream, each warp paced to the target ratio of the alone arm's rate, and the arm reports the ratio it
  achieved. Where the hash can't keep up beside the whole tensor, the tensor is paced down to leave it room. cuBLASLt
  runs alone only, because the hash's resident CTAs crowd out its GEMM's.
- `wgmma_i8.cuh`: the wgmma wrappers, which `gen_wgmma.py` writes.
- `run.sh`: one card. It builds, takes one gpu-lease, locks the clock, runs every round, class, kernel and encoding,
  and resets the clock at exit.
- `summarize.py`: the reading over one or more runs' `out/` directories. It is the maximum over every attempt and
  card, per class, alone and at each ratio. An arm beside the hash counts at its ratio only if the hash reached that
  ratio.

~~~text
research run --on NODE --project verity --source . --cwd source --env CARD=NODE \
  --env GPU_LEASE_WHO=compute-accounting -- bash benchmarks/int8_class_rate/run.sh
python3 benchmarks/int8_class_rate/summarize.py ~/.research/runs/RUN1/out ~/.research/runs/RUN2/out [--json FILE]
~~~

`run.sh`'s header lists its knobs: classes, encodings, kernels, ratios, seconds an arm, rounds and clock. Nothing here
is a proof, and there are no tests: a run's rates are its record.
