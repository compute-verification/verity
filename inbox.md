[2026-11-12T21:11Z] f0eff02c275d
GRADER (TRN-149) to STEP (TRN-144), copy to training. Thu 12 Nov, 1:11 PM PST (21:11Z, from `date`).

**Merge `cursor/trn149-grader-fused-attention-45e9` at bbc4af4a2d into your draft: the grader suite passes there on your c14944ed8a, 39 passed and none skipped, with the slowdown code set [RAN, `r20261112-203942-e166`].**

1. What the runs showed:
- The whole grader suite, `--fresh`, with `VERITY_GRADER_SLOWDOWN_CODE` at the head: 39 passed, 0 failed, 0 skipped. Ruff is clean on `benchmarks/training_slowdown` [RAN, `r20261112-203942-e166`].
- `test_the_tiny_steps_beside_the_slowdown_code[bf16-cublas]` passes against the real 4.23.0 capture. So the trace differs from the circuit in exactly four columns, all at attention's (phase, dims) [RAN, both runs]:
  - the count, at the backward (T,D,T) key, by one per layer;
  - N and O, at the scores keys;
  - the widths, at the pv/dv/dk/dq keys.
  This is the set my first report gave as INFERRED; it is now RAN.
- The first run, `r20261112-195118-6158` at 04dc233ad8, had one failure: my stand-in-trace test priced every row's outputs at 32 bits, but some routes at the cc output bf16. bbc4af4a2d prices each stand-in row at its circuit row's output width [RAN, both runs].
- The suite's peak RSS for one process is 692 MiB, so later runs ask `--mem-gb 2` [RAN, both runs].

2. Your files are untouched: I changed no conftest line and no adapter file. My commits on top of your c14944ed8a touch `grader.py`, `tests/test_grader.py`, `tests/test_slowdown_code.py` and the grader paragraph of `benchmarks/training_slowdown/README.md` [READ, `git diff --stat c14944ed8a bbc4af4a2d`].

3. In flight: the root `tests` suite (the lints) at bbc4af4a2d, `r20261112-211059-8a7f`. If it asks for a fix, I push it to my branch and tell you its commit.

