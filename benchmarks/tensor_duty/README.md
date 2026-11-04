# tensor_duty: the H200's tensor MAC rate by duty cycle

The falsification test of the H200 card's tensor clock (`clock.tensor` in `verity/hardware/cards/h200-sxm-141gb.toml`,
`tensorClock` in `Security/Definitions/Core/Accel/H200.lean`): whether one H200, with its power limit at the board's
maximum, ever moves more dense tensor MACs a second than the card's tensor rate, 132 SMs x 4096 MACs a clock at
`clock.tensor`'s 1830 MHz (9.894e14; 1.0705e15 at the 1980 MHz SM clock, the card's second reading). A run above it
refutes the card. A full tensor pipe is not the only case: a pipe busy part of the time, with the rest idle or
non-tensor work, could run its tensor phases at a higher clock. So each point holds the dense tensor MMA at a duty
cycle from 50% to 100% (5% steps to 90%, 1% steps above), for an operand class, a fill and a burst (SM-cycles of tensor
work a period), with the SM clock locked to 1980 MHz (`nvidia-smi -lgc`) and again at the driver's default.

Throughput rests on the host's wall clock alone: a point's MACs over the host's interval around its kernel of at least
a second. The host stamps are printed to the millisecond, so `summarize.py` refutes the card only on a point over the
card's rate even over its interval lengthened by a stamp, and passes it only when no point is over it over its interval
shortened by one; anything between is undecided, which doesn't pass. The GPU's own timers (cudaEvents, `%globaltimer`,
clock64 for the whole kernel and for its tensor phases) are diagnostics. Timings only.

- `duty.cu`: the kernels and the driver. Arms: wgmma m64n256k32 s8 (`wg_s8`) and e4m3 (`wg_e4m3`), one warpgroup an
  SM on shared-memory operands, as `int8_class_rate/`'s wgmma kernel; and mma.sync m16n8k32 s8 (`sync_s8`) on register
  operands. Classes: zero, a constant, uniform bytes, and `Verity.OperandClass`'s s2, u2 and u4. Fills: idle
  (`__nanosleep`) or FFMA chains. Its header gives the period loop and the JSON fields.
- `gen_wgmma.py` writes `wgmma_e4m3.cuh`; the s8 wrappers are `int8_class_rate/wgmma_i8.cuh`.
- `run.sh`: one card. It builds outside the lease, then under one gpu-lease sets the power limit and, for each clock,
  locks it, warms the card and runs every arm, class, fill, burst and duty; at exit it resets the clock and the power
  limit. Last it runs `summarize.py`, and exits nonzero unless the card passes. Its header lists the knobs.
- `summarize.py`: the verdict, PASS, UNDECIDED or REFUTED against the card's rate (`verity.hardware.rate`), exiting 0
  only on PASS; then the maximum by clock, arm and class against both readings, each clock's and arm's duty x class
  table, and the fills and bursts compared, over one or more runs' `out/` directories. `tests/` holds its verdict to
  the card on made-up points.

~~~text
research run --on NODE --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
  -- bash benchmarks/tensor_duty/run.sh
PYTHONPATH=. python3 benchmarks/tensor_duty/summarize.py ~/.research/runs/RUN/out
~~~

The card's `clock.tensor` source cites the run of record by its `art:` id.
