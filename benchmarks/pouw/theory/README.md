# PoUW theory checks

PoUW's resource accounting rests on named assumptions about what a GPU can and cannot do. Each script here decides
one claim that such an assumption or a Lean docstring makes, exhaustively at a small width or over a modelled
instruction set, so that the claim is checked rather than only stated.

Most of them are about the fold receipts, and the TS assumptions their accounting reads
(`ResourceAccounting.Assumptions.TS`). Dither, PoUW's no-fold design, has none of them. A script stays here only while
a test or the Lean reads it, and when the TS fold assumptions are retired, their scripts are deleted with them.

## Try it

~~~sh
uv run python benchmarks/pouw/theory/nosum.py
~~~

This checks that no fixed-function unit on sm_120 writes a fresh committed sum-of-products word, on a CPU, and ends
with `19/19 checks OK`. Every check prints its count of checks passed and exits nonzero when one
fails.

`run.py` runs every check, each in its own process, and passes when each exits 0:

~~~sh
uv run python benchmarks/pouw/theory/run.py --quick
~~~

Without `--quick`, one search runs for CPU-hours. The sm_120 probes, the `*.sh` jobs, need an RTX PRO 6000 and run
through `research run`; no Python here touches a GPU.

[`DESIGN.md`](DESIGN.md) says what each script checks and which Lean or test reads it, how the checks, the C
enumerators and the probes are run, and which scripts other code imports.
