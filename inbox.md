[2026-11-13T05:20Z] a5b9d4485c27
BACKWARD2, Thu 12 Nov 9:20 PM PST

**The head is `dd8b446dc8`, pushed: all four items are done, and the suites and circuit-check passed at it. #9417 stays a draft.**

What changed since `3b384b35f29e` (one commit):
- 1 (MUST), CONFORMANCE: "and the second's eight, in each of which one dim's product is 2^128, which overflows only if that product is rounded alone".
- 3: "whose inputs give values no captured word has".
- 4: "two probes on a card of the same model".
- 2: `ROWS` gains `"s/wide": ("bwd/s/det/wide", 3, (2, 8), (43,))` and `"l16/wide": ("bwd/l16/det/wide", 3, (2, 16), (14,))`.
  - `cut` zeroes every row entry's k and v past its last row (no row sees them), and its docstring says so. The rule covers `xl/wide` too, whose rows all still give the card's words.
  - `test_each_row_s_dq_is_the_captured_kernels_words` adds both rows to `seen`. Through the Definitions they give the card's words. With each row's dP sum taken from the test's `_dpsum` in each order the row tells (`ORDER_TOLD`), the dQ is other words.
  - The fixture was regenerated with its generator: 240,308 B, under the 262,144 B limit. It is registered as `art:3b4e100e16ec5f11bccadf73cab4d002d370ffe50815ac9eb4f3d37cbd36ec57`, with `art:253bd5f4...` and `art:cf13269b...` under `previous`.

RAN:
- The generator twice on the capture (sha256 `487ec182`, the same file TILES used): the same bytes both times. Before the change it rebuilt the old tracked file exactly.
- The three mutations of `Fa2TrainBwdPre`, each against the whole test file (VM). Each fails the new rows' card-words assertion, `assert ... == d[dq]`, and `test_the_dp_sum_is_fmul_then_ffma`:
  - dims 2 and 3 swapped: AssertionError ('rows/s/wide', 43);
  - dim 0 fused after dim 2: AssertionError ('rows/l16/wide', 14);
  - dims 2 to 7 reversed: AssertionError ('rows/s/wide', 43).
  - At the head itself: 9 passed, slow test included.
- r20261113-035735-9b9b, suites at `dd8b446dc8` (`--changed 3b384b35f29e --quick`, `--cwd source`, rc 0): "57 suites: 57 passed (0 cached), 0 failed; 110.1 min of test time".
- r20261113-035801-62a0, circuit-check on both sidecars at `dd8b446dc8` (rc 0): "circuit-check: 16 target(s), 0 failure(s), 14 unchecked (warnings, which never fail) on 14 Definition(s)".
- `git merge-tree` of `dd8b446dc8` against origin/main `8ef93f0196`, with the `fixture-map` driver configured: clean, rc 0. No merge was made.
- `pytest tests/test_repository.py` on the VM, which holds the blob limit: 17 passed. `ruff check` on the changed files: passed.

READ (`gh pr view 9417`): the body is replaced and still a draft, with the Lane, Refs and Goal lines kept. The three statements stay. It adds the new rows, the fixture's size and id, the three order mutations in the mutation table, and these checks. RAN and READ are kept apart. The forward's choice is now stated with TILES's two corrections (O of the largest-LSE split -0 or within about 2^-148 of zero; only rows whose four splits all hold keys), along with your ruling of no card job.

The handoff note is updated the same way. No run id, date, name, count or probe word is in code, comments, docstrings or CONFORMANCE.

Nothing is left that I could not do.

