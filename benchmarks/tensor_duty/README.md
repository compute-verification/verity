# tensor_duty: the H200's tensor MAC rate by duty cycle

Can an H200 move more tensor MACs a second than its card says it can? The H200's card
(`hardware/cards/h200-sxm-141gb.toml`) gives its tensor rate at `clock.tensor`, and this benchmark is the falsification
test of that clock: a run above the card's rate refutes the card.

A full tensor pipe is not the only case: a pipe busy part of the time, with the rest idle or non-tensor work, could run
its tensor phases at a higher clock. So each point holds the dense tensor MMA at a duty cycle from 50% to 100%, for an
operand class, a fill and a burst, with the SM clock locked to 1980 MHz and again at the driver's default, and the power
limit at the board's maximum. Throughput rests on the host's wall clock alone.

## Run it

It needs one H200, under `research run`:

~~~text
research run --on NODE --project verity --source . --cwd source --env GPU_LEASE_WHO=LANE \
  -- bash benchmarks/tensor_duty/run.sh
PYTHONPATH=. python3 benchmarks/tensor_duty/summarize.py ~/.research/runs/RUN/out
~~~

`run.sh` exits nonzero unless the card passes. `summarize.py` gives the verdict, PASS, UNDECIDED or REFUTED against the
card's rate, then the most each clock, arm and class reached against both of the card's readings. Its tests check its
verdicts on made-up points, on a CPU:

~~~sh
uv run tools/check/suites.py benchmarks/tensor_duty
~~~

`run.sh --arms fp32-beside` asks a second question of the same card: whether a pipe beats its ceiling when both are
busy. [`DESIGN.md`](DESIGN.md) says what each kernel runs, how a point is judged against the card's rate, and what the
second question's arms are.
