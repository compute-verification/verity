# benchmarks/hash/dither

This benchmark asks what PoUW's receipts cost a served model. It serves int8-exact Llama-3.1-8B in compiled vLLM with
each way of computing the receipts, called an arm, and times every arm against stock vLLM int8 on the same GPU. An arm
replaces the int8-exact linear with kernels that serve y and, from the same operands, compute the execution phase's
tables, seeds and receipts. Before an arm is timed, its own run gates it: its roots, receipts and z against its CPU
twin and the Definition.

The kernels it times are frozen. They live with Dither's prover, in
`verity/protocols/experimental/dither/prover/kernels/`, and move with compute accounting's protocol.

Despite the names, what the arms serve are H1's and J10-R's folds, not Dither, PoUW's no-fold design. The arms keep
names such as `dither-a` because the runs' records carry them. Nothing a verifier runs reads this directory.

## Try it

On an RTX PRO 6000 under a GPU lease, from the tree's root:

~~~sh
RESEARCH_RUN_DIR=DIR GPU_LEASE_WHO=WHO bash benchmarks/hash/dither/sbench.sh \
  --arms stock,exact,dither-a-hbr-zy-aomn-so4 --gate --ops decode --t 64
~~~

It builds the libraries the arms load for the card (`SBENCH_ARCH=100a` on a B200, `90a` on an H200), starts each arm
as a resident vLLM engine in its own worker process on that GPU, and alternates reps between them. `--gate` first holds
each arm's greedy tokens to its reference arm's; then each rep times decode at batch 32 (`tpot_b32_ms`). The run writes
`summary.json` and `stop.json` under `$RESEARCH_RUN_DIR/out`. `sbench.py`'s docstring lists every arm and flag.

One family of arms, labelled `form=A (provisional)`, is the arm of record. Every other arm is labelled `beside`:
measured beside it, never credited. [`DESIGN.md`](DESIGN.md) says which arms those are, what each one serves, the
kernels and gates behind them, and which files the rest of the tree reads.
