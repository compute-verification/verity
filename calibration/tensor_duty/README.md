# tensor_duty: an sm_90a GPU's tensor MAC rate by duty cycle

Can an sm_90a GPU move more tensor MACs a second than its card says it can? Its card, the data sheet under
`calibration/hardware/cards/` whose `driver` holds the name nvidia-smi gives the GPU, gives its tensor rate at its
ceiling reading (`clock.tensor` where the card marks that clock a ceiling, else `clock.sm`), and this benchmark is the
falsification test of that ceiling: a run above the card's rate refutes the card.

A full tensor pipe is not the only case: a pipe busy part of the time, with the rest idle or non-tensor work, could run
its tensor phases at a higher clock. So each point holds the dense tensor MMA at a duty cycle from 50% to 100%, for an
operand class, a fill and a burst, with the SM clock locked to the GPU's highest and again at the driver's default, and the power
limit at the board's maximum. Throughput rests on the host's wall clock alone.

## Run it

It needs one sm_90a GPU (an H200 or an H100), under `research run`:

~~~text
research run --on NODE --project verity --source . --cwd source --mem-gb 1 --env GPU_LEASE_WHO=LANE \
  -- bash calibration/tensor_duty/run.sh
PYTHONPATH=. python3 calibration/tensor_duty/summarize.py ~/.research/runs/RUN/out
~~~

`run.sh` exits nonzero unless the card passes. `summarize.py` gives the verdict, PASS, UNDECIDED or REFUTED against the
card's rate, then the most each clock, arm and class reached against both of the card's readings. Its tests check its
verdicts on made-up points, on a CPU:

~~~sh
uv run tools/check/suites.py calibration/tensor_duty
~~~

`run.sh --arms fp32-beside` asks a second question of the same card: whether a pipe beats its ceiling when both are
busy.
