# The inference-only policy's literature index and slowdown board

This directory is data and a calculator for the question negotiators will ask about the
[inference-only policy](../../verity/protocols/workload_classification/training_detection/inference_only/verifier/README.md):
if both sides adopt it, how fast can a malicious developer still advance frontier capability? It holds every source the
answer relies on, each claim quoted verbatim; the parameters the bounds depend on, each estimate saying which findings
it merges and how; the policies they are evaluated under; and the board, one training method in one configuration per
row, each with a slowdown bracket. Nothing in `verity` reads this directory.

## Try it

From the repository root, print the board's brackets under one calibration:

~~~sh
python benchmarks/nci/bracket.py --calibration kimi-k2-1m-raw
~~~

It prints one line a row: `thread`, the proven lower bound under the row's rates and the theorem's assumptions;
`upper`, the cost of the cheapest construction the search finds; and that construction's basis. Both are computed from
the parameters, not measured: the constructions are analytic, and none has been run.

## What the committed parameters may hold

Today the policy puts no limit on θ, the registered information. That matters most for seed-based training.
A developer may commit a noise table, one entry per parameter per direction, and then decode an update with one
multiply-add per parameter instead of a PRNG sample.

- **Without a limit** (`kimi-k2-1m-raw-uncapped`), `zo-seed-pool`, `es-population-30` and `mezo-seed-finetune` cost
  about 1.2, 2 and 3.
- **With a limit**, which every other calibration assumes (`theta_cap`), they cost about 3, 2,200 and 2,400. A fixed seed
  pool fits in one block even then: its decoding, 4.3×10¹⁶ flops, is under `Gamma`.

**Design constraint.** The policy serves an agreement between rivals, so a rule on θ must neither reveal nor narrow the
architecture. Checks run inside the zero-knowledge proof, and the verifier learns only that they pass.

**Recommended rule: two public budgets.**

- At most `B_matrix` bits of θ may be read only as a multiplication's operand against a value not in θ (matrix weights,
  in any architecture).
- At most `B_other` bits may be anything else.

Both budgets are set alike for all parties and generously, a few times the largest permitted model, so they reveal
nothing about any party's model. A noise table for a 1T-parameter model costs about 125 GB per direction, so a budget
of a few TB leaves room for too few directions to train with. Frontier corpora don't fit either.

**Rejected** because they would reveal or narrow the architecture: typed roles, declared shapes or layers, and a
per-model registry. A rule that committed text enter only through the embedding lookup is optional. It is generic, but
it needs the proof to identify that operation.

**Not closed.** Structure can't tell weights from data: a one-hot activation copies any θ entry. A corpus can also hide
in a checkpoint's slack bits, about 2×10¹¹ tokens for a 1T-parameter model requantized within 1% loss. So small-model
training on smuggled data stays bounded, not closed.

**The proof gap.** The thread guarantees say nothing about a developer whose data is in θ, since their premise
`ThetaRegistered` fails. The fix to pursue states the guarantee with a term for the committed data, degrading by
at most the bits of data the budgets allow.

## Rendering

~~~sh
python benchmarks/nci/render.py > /tmp/literature-index.md     # never commit the output
python benchmarks/nci/bracket.py --calibration kimi-k2-1m-raw
~~~

`render.py` reads `sources/`, a fetched fixture: `research data fetch-fixtures benchmarks/nci/sources` brings it.

[`DESIGN.md`](DESIGN.md) has the theorem it cites, the files, what loads them, the rules for adding to them, how
estimates merge findings, how slowdown is measured, the bracket, the argument map, what the literature says now, the
algorithm library, and what is not yet here.
