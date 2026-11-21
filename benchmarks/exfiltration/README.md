# exfiltration

`verity bench exfiltration` prints GLM-5.3's exfiltration capacity in GB over a monitoring period at 10% of total
compute: the compute audit's part beside the network attester's, every input marked. It composes the two calculators, so
it needs no GPU and no node. Its code is `benchmarks/harness/verity_bench/exfiltration*.py`.

## What you may change

Each input the print lists with its mark, by its flag (`--share`, `--row`, `--step`, `--response-ids`, `--scan-delta`),
and the record the rows come from: the store's runs of record, or one run's directory with `--from`.

## What is fixed

The two calculators and how the command reads them (the fit over the pinned budgets, λ, the errors summed, the marks),
and the kit's `report`, which says what the run ran with and that it is a measurement, not an audit.

## How a cheating change is refused

Parts that cover different GPU-seconds, requests or weights, or two runs of record for one scope, are refused; a figure
whose error is 1 or more prints no amount; a made-up record writes no measurement.

Try it, with no store and no key (hand-written rows, printed as made up):
~~~sh
uv run verity bench exfiltration --from benchmarks/exfiltration/made_up
~~~
