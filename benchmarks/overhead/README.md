# Overhead

γ is a PoUW scheme's prover overhead over stock: how much more busy GPU time, with the host CPU time it adds at a
weight, the scheme spends than stock does for the same useful output, at a card's operating points.
`verity bench overhead` is the one front door for it: it gives γ from labelled runs of record alone. Its code is the
harness's ([`benchmarks/harness`](../harness/README.md)); this directory holds its runner.

A card has two operating points, each a number of rows m over one layer's four linears of Llama-3.1-70B:

- **C**, the smallest m at which the layer's throughput reaches 90% of the best sampled, and stays there above it;
- **L**, the largest m whose time stays within 10% of the time at m = 1.

## Try it

~~~sh
uv run verity bench overhead table
~~~

This prints γ for each scheme, card and point from runs of record, with its 95% interval, its A/A control and its ε,
then each γ over a monitoring period of a week, a month and a year, over a datacenter's compute and the world's. It
reads the evidence store, so it needs the store's credentials, and no GPU. `--json` gives each row with `of_record` and
`why`.

Every γ covers the four linears of one layer and leaves the LM head out, and `table` says so. Pearl+'s rows carry the
note `without y's commitment (y tree not timed)`: its arm commits no output y yet, so its γ leaves y's hash out.

## Commands

```text
verity bench overhead table [--json]                          γ per scheme, point and card, from runs of record alone
```
