# tensor_duty: the H200's tensor MAC rate by duty cycle

Whether one H200, with its power limit at the board's maximum, ever moves more dense tensor MACs a second than 132 SMs x
4096 MACs a clock at 1830 MHz (9.894e14), the tensor clock a card can declare only if nothing beats it. A full tensor
pipe is not the only case: a pipe busy part of the time, with the rest idle or non-tensor work, could run its tensor
phases at a higher clock. So each point holds the dense tensor MMA at a duty cycle from 50% to 100% (5% steps to 90%, 1%
steps above), for an operand class, a fill and a burst (SM-cycles of tensor work a period), with the SM clock locked to
1980 MHz (`nvidia-smi -lgc`) and again at the driver's default. The reading compared with the bound is MACs over the
host's wall clock (cudaEvents around a kernel of at least a second); clock64 over `%globaltimer`, for the whole kernel
and for its tensor phases alone, is the second reading. Timings only.

- `duty.cu`: the kernels and the driver. Arms: wgmma m64n256k32 s8 (`wg_s8`) and e4m3 (`wg_e4m3`), one warpgroup an
  SM on shared-memory operands, as `int8_class_rate/`'s wgmma kernel; and mma.sync m16n8k32 s8 (`sync_s8`) on register
  operands. Classes: zero, a constant, uniform bytes, and `Verity.OperandClass`'s s2, u2 and u4. Fills: idle
  (`__nanosleep`) or FFMA chains. Its header gives the period loop and the JSON fields.
- `gen_wgmma.py` writes `wgmma_e4m3.cuh`; the s8 wrappers are `int8_class_rate/wgmma_i8.cuh`.
- `run.sh`: one card. It builds outside the lease, then under one gpu-lease sets the power limit and, for each clock,
  locks it, warms the card and runs every arm, class, fill, burst and duty; at exit it resets the clock and the power
  limit. Its header lists the knobs.
- `summarize.py`: the verdict, the maximum by clock, arm and class with both readings, each clock's and arm's duty x
  class table, and the fills and bursts compared, over one or more runs' `out/` directories.

~~~text
research run --on NODE --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
  -- bash benchmarks/tensor_duty/run.sh
python3 benchmarks/tensor_duty/summarize.py ~/.research/runs/RUN/out
~~~

Nothing here is a proof, and there are no tests: a run's points are its record.
