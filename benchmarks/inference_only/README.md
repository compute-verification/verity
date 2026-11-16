# Inference-only

How much training could a developer hide inside the inference blocks that Llama 3.2 1B's inference-only policy lets it
serve? This benchmark answers with one number: the least slowdown that training would suffer, under the min-cut bound.
A number of 1 says the bound shows no slowdown. The served policy's number is 1.

Given a served run's record, it follows the core benchmarks' interface: a measurement, not an audit, with a number only
where the record holds a PoUW accept of record and the policy's check ran and both its rules comply. Given none, it
prices the policy that a file in this tree states, and says under its first line that it reads no run.

## Try it

~~~sh
uv sync --all-packages --extra torch-cpu
uv run verity bench inference_only --stated-only     # the served policy and its number, from its stated numbers alone
~~~

This needs nothing but a CPU. It prints `A STATED POLICY PRICED, NOT A MEASUREMENT OF A RUN`, then the served policy
(`AS SERVED: 1`), its `c`, `k` and `n` with what `k` is made of, the same policy over monitoring periods, and one JSON
object last. It exits 0.

On a served run's record, where `DIR` is the record's directory, its `out/`, or a run id:

~~~sh
uv run verity bench inference_only --stated-only --from DIR
~~~

Its first lines are the interface's:

~~~text
Inference-only: the least slowdown of training run inside the policy's inference blocks
ran with: coins local challenger; parameters development (scheme SCHEME)
a measurement, not an audit
The served run's record read: DIR/out
~~~

The `ran with` line says which coins and parameters the record states PoUW's piece was made with, or `ran with: not
stated` where it names none. Where the record has no PoUW accept of record, or the check refuses it, the first line is
`inference_only: no number: WHY` and the command exits 1. The record's check is Lean, built with `lake build` the first
time it runs, so this form needs `lake`.

Two more forms:

~~~sh
uv run verity bench inference_only --stated-only --uncredited-share 0.1%   # and the served class under the value tie
uv run verity bench inference_only                   # and the class this tree builds, with the Lean count on it
~~~

The last builds the Lean count and runs it on the class this tree builds, which is heavy work.

## What you may change

The record: any served run's files, its class and policy, its limits and counts, its PoUW verdict. And the options:
`--policy FILE`, a stated policy to price; `--model`, `--class` and `--division`, the built class; `--uncredited-share
S`, a typed share, whose figures print under a label saying they rest on no PoUW verdict.

## What is fixed

By this tree, whatever a record says: the check (the policy's Lean record check, `inference-only-record-check`, with
its two rules, the rate rule and the count), the bound (`min_cut`) with its premises (H) and (E) at the slowdown
calculator's β (`benchmarks/slowdown/data/premises.toml`), the rule that only PoUW's accept of record
(`protocols.work.verdict`) credits work, the tie between that verdict and the record the check read, `tied.toml`, the
fleets and periods, and that a number is given only where the record holds a PoUW accept of record and both rules
comply.

## How a cheating change is refused

A record with a weight left unregistered is refused by the count rule, and by the rate rule, and gives no number. So
does a record whose PoUW verdict is a development accept or no accept, and one with no Program, which the check can't
run on. A malicious developer who gives up credit raises the uncredited share `s`, which limits the number to `1 / (s +
1/slowdown)`. A verdict crediting below 0, above the blocks' work or above the MACs it counts, or counting other MACs
than the class's linears, is refused; so are a verdict tied to another record than the check's, a record holding
counts of two classes with no `--served-class`, and a record missing a file, which is named. The coins line never
says the deployed challenger from a record's own word. `tests/benchmarks/inference_only/test_served_record.py` holds each.

## What it does not mean

The number is a block's ratio under two assumptions, on a check that Lean does not yet accept for any class.

- **Two assumptions.** The bound holds under (H), the cut's hardness, and (E), its exports, conjectures
  (`min_cut.HARDNESS` and `min_cut.EXPORTS` are their Lean names). The values of (H)'s β by `k` are the slowdown calculator's
  (`benchmarks/slowdown/data/premises.toml`).
- **No class passes the check.** The Lean count is given the session Program with no θ and no dataflow, so it counts
  every row a block reads as fresh and refuses the policy on `k`. A built policy's `c` and `k` are the Python count of
  the record's description, which the Lean check does not establish; a stated policy's are not checked here at all.
- **A block, not a deployment.** The periods table multiplies one block's ratio through the blocks a fleet holds at its
  data sheet's rate. What a deployment registers, what it leaves free and what its proofs miss are the slowdown
  calculator's (`uv run python benchmarks/slowdown/periods.py`).
- **Units, not converted.** `N` and `limit` are in the policy's own MACs, which count a MAC of the layers' linears as one
  and a BF16 MAC (attention, the LM head) as two, the card's dense int8 rate against its BF16 rate. The bound is at
  `F = 1`, which charges at most two of a training step's three kinds of product; the two thirds that leaves of a step
  where the weight-gradient products are not charged is not applied.
