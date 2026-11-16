# sampling_plan

SPOT's cost model (`verity/protocols/proofs/spot/prover/cost/`) turns a budget for proving into a sample and the
bound it certifies, and it is only as good as the numbers it is fed. This directory holds those numbers, each with
its source and whether it is measured or assumed, and plans one served workload with them. It is a model, not a
measurement.

## Try it

~~~sh
uv run python benchmarks/sampling_plan/demo_a.py --out /tmp/sampling_plan --no-frontier
~~~

It plans Demo A on one H200 with the cost model's planner and prints, for an execution phase of one server block, the
least length that affords a plan, a week, a month and a year: the budget, the sample, and the free compute and leakage
it certifies at δ = 1% and 1e-6. It writes `plan_of_record-all.json` and `m3.json` to `/tmp/sampling_plan` in about
twenty seconds; without `--no-frontier` it also plans the frontier over budgets. Its docstring says what the plan
assumes.

`inputs.json` holds the models' shapes, the inputs with a low and a high each, and the proving observations the cost
model is fitted from, with the builds and hardware they key. The suite
(`tests/benchmarks/sampling_plan/`) holds the cost model's rows to them.
