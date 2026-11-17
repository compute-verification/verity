# Pearl-C4's runs, ships and replays

Pearl-C4 is Pearl-C on FP4 operands that carry one scale a scale group (`pearl-c-nvfp4-v0` and `pearl-c-mxfp4-v0`), an
experimental protocol (`verity/protocols/experimental/pearl_c4/`), whose device kernels for the RTX PRO 6000 are in the
protocol's `prover/kernels/`. This directory holds what ships a build of them as a run's source, and the reference replay,
on a CPU, of what a device ran. Its timing is self-timing for development; a number of record is the GPU harness's
([`../harness/`](../harness/README.md)).

These recompute rows and words in the clear, a development shortcut: the protocol's check is a zero-knowledge proof,
and a replay's ACCEPT is never its verdict.

## Try it

Make and verify a replay's transcripts on a CPU:

~~~sh
uv run pytest tests/benchmarks/pouw/test_pearl_c4_replay.py -q
~~~

It prints `5 passed`.

On the card, the device check builds off the GPU machine, ships the tree and runs it under `gpu-lease 1`:

~~~text
bash verity/protocols/experimental/pearl_c4/prover/kernels/build.sh [OUT]        # with CUDA 12.9's nvcc; OUT defaults to /tmp/pearl-c4-ship
bash benchmarks/pouw/pearl_c4/ship.sh [OUT] [REPO]  # OUT as a one-commit repository for `research run --source`
bash run.sh [check]                                 # in the shipped tree, on the card
bash run.sh time [MxNxK ...]
~~~

It exits 3 when the GPU's identity is refused, 5 when the tree does not match its manifest and 4 when a buffer differs,
in which case nothing is timed.
