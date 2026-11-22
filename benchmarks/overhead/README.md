# overhead

`verity bench overhead` is the one front door for γ, a PoUW scheme's prover overhead over stock: the busy GPU time, with
the host CPU time it adds at a weight, the scheme spends beyond stock's for the same output, at a card's two operating
points for a model, each a number of rows m over one layer's four linears. **C** is the smallest m reaching 90% of the
best sampled throughput; **L** is the largest m whose time stays within 10% of the time at m = 1. The rules and the
sweep are `benchmarks/harness/verity_bench/overhead.py`'s; this directory holds the runner.

~~~text
verity bench overhead points --on NODE [--model M]   sweep m on one GPU of NODE for C and L (a research run)
verity bench overhead run --on NODE --scheme S       time S against stock at the card's points (a research run)
verity bench overhead table [--json]                 γ per scheme, point and card, from runs of record alone
~~~

## Try it

~~~sh
uv run verity bench overhead table
~~~

Prints γ for each scheme, card and point from runs of record, with its 95% interval, its A/A control and its ε, then γ
over a week, a month and a year. It reads the evidence store, so it needs the store's credentials and no GPU. `points`
and `run` need a GPU node and are the commands of a `research run`.
