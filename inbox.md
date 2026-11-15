[2026-11-15T07:46Z] 205ea013847d
ROWSPROTO2, TOP-424 tie mend PUSHED (11:47 PM PST). Branch cursor/na-prove-path-tie-b51f, one commit b8ffef108f1 on architecture 009698aa9c0a. Nothing pushed to architecture or collect.
Cause confirmed. The subprocess's sys.path named verity/protocols/proofs/spot/verifier/tests, which holds none of test_pearltie, test_remotetie, window_tie or settings.py (none of them is anywhere under it). All four live in benchmarks/prove_path/tests (HERE). 00dac86f8bf moved the two tie tests there, and 2bac3457712 moved window_tie; neither updated the path. records_audit is in HERE.parent, which stays. Nothing else is needed from spot/verifier/tests, so I dropped that entry.
benchmarks/prove_path/tests/test_records_audit.py:718
before: sys.path[:0] = [{str(Path(__file__).resolve().parents[3] / "verity" / "protocols" / "proofs" / "spot" / "verifier" / "tests")!r}, {str(HERE.parent)!r}]
after:  sys.path[:0] = [{str(HERE)!r}, {str(HERE.parent)!r}]
Ran on this VM (CPU, no Lean needed), ci-like path on the worktree, fixtures fetched:
- the test (-k): 1 passed in 25 s; without the mend, ModuleNotFoundError 'test_pearltie', as in ci's run.
- the whole file: 52 passed.
- py_compile ok; ruff 178 -> 177 (one long line fewer).
- tests/test_repository.py: 17 passed.
Read guard: test_records_audit.py is in the suite's `untraced` list (keyed on the suite's whole tree), and the mend points only at benchmarks/prove_path/tests and benchmarks/prove_path, both inside the suite. Fine.

