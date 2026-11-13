# tensor_duty: the H200's tensor MAC rate by duty cycle

The falsification test of the H200 card's tensor clock (`clock.tensor` in `hardware/cards/h200-sxm-141gb.toml`,
`tensorClock` in `security/Definitions/Core/Accel/H200.lean`): whether one H200, with its power limit at the board's
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
- `gen_wgmma.py` writes `wgmma_e4m3.cuh` (a test holds the file to it); the s8 wrappers are
  `int8_class_rate/wgmma_i8.cuh`.
- `run.sh`: one card. It builds outside the lease, then under one gpu-lease sets the power limit and, for each clock,
  locks it, warms the card and runs every arm, class, fill, burst and duty; at exit it resets the clock and the power
  limit. Last it runs `summarize.py`, and exits nonzero unless the card passes. Its header lists the knobs.
- `summarize.py`: the verdict, PASS, UNDECIDED or REFUTED against the card's rate (`verity.hardware.cards.rate`), exiting 0
  only on PASS; then the maximum by clock, arm and class against both readings, each clock's and arm's duty x class
  table, and the fills and bursts compared, over one or more runs' `out/` directories. `tests/` holds its verdicts on
  made-up points.

`run.sh --arms fp32-beside` asks a second question of the same card: how many FP32 adds it retires per tensor MAC,
per second over the host's wall clock, when both pipes are busy (`price.fp32_loop` in the card file; PRF-612's clock
falsifier). Its arms are duty.cu's A (the e4m3 wgmma alone, R_t), B (a loop of 64 FADDs and 3 loop instructions alone,
on two warps a scheduler), C (both in one kernel, the adds in the wgmma's shadow on warps of their own, at add:MAC
ratios 1/128, 1/32 and 1/8) and D (the two alone in alternating phases of 1, 10 and 100 ms), on zero, constant and
uniform operands, at each clock, with nvidia-smi sampling every 10 ms and `nvidia-smi -q -d CLOCK,PERFORMANCE` around
each lock. `summarize.py --arms fp32-beside` writes `summary.json`: each point's rates and the clock it held (measured,
never the one asked for), and p = R_t,alone / R_f,mixed, the time price of an add in tensor units, for C and D. The run
prices nothing: it tests each reading of the price (per clock 32; the sheet's clocks 32 x 1830 / 1980; a cautious base
clock over 1980), which survives only if every C and D point has p >= 0.995 of it and arm B reaches 95% of the loop's
ideal rate. `--no-lock` runs only the default clock and calls no sudo; a clock whose lock is refused is skipped either
way.

`verity bench tensor_duty` runs `run.sh`, which writes under a run's `$OUT` and refuses without one, so run it as:

~~~text
research run --on NODE --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
  -- bash benchmarks/tensor_duty/run.sh
PYTHONPATH=. python3 benchmarks/tensor_duty/summarize.py ~/.research/runs/RUN/out
~~~

The card's `clock.tensor` source cites the run of record by its `art:` id. Nothing imports this directory:
`summarize.py` is a command, and it reads the card through `verity.hardware.cards`, never a copy of its numbers.
