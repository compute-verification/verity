[2026-11-15T07:57Z] 1c88843a7016
POLICYROWS2 to lead (11:57 PM PST): record-difftest pins, pushed to scratch branch cursor/na-record-difftest-pins-b51f only; one commit, 92cfb0854004 on 009698aa9c0a.
- Count 708, unchanged, so no case to read.
- Generator (uv run, scripts/record_difftest_vectors.py), two runs, identical: cases.jsonl f9e04b6ebd274e69437ab19c4df8661d154b8cd7f228acd8d808df0a0b77e52e, python.jsonl 0a275f9aa8d0a3dd122339a9441cc08676027766c145f469af152fdfe3cfa507 (the same as 3:50 PM). The lock pinned 01deee7f…/5adb8c2d….
- Lock: only runs["InferenceOnlyCheck.RecordDifftestMain"].data changed. check.py's tests() writes entry | {"data": sha256 of each generated file}; --update never writes args, which stay 708. The test and the check hash the same bytes: both call write(out, None), uncompressed.
- dependencies (only "toolchain" in that lock): not written. It is olean_digest over the .olean files Facts.lean reports loaded, which needs a build.
- Ran: test_inference_only_record_difftest.py, all 20 including test_the_data_is_what_the_audit_pins, passed. tests/test_lean_packages.py + tests/test_repository.py: 51 passed, with 4 bookkeeping warnings on owners, md caps and basenames, none from this change. check.load takes the lock. git merge-tree with origin/cursor/lean-arch-lock-layers-fc8d (9477aac4534) is clean; that branch leaves this lock untouched.
TIMER: none

[2026-11-15T07:57Z] c1965b6bb9b1
ROWSPROTO2, TOP-424 read-guard mend PUSHED (11:57 PM PST). Branch cursor/na-prove-path-tie-b51f, new head 5046f4b5202, one commit on b8ffef108f1. Nothing pushed to architecture or collect.
Cause: test_kept_scan.py:42-45 puts benchmarks/prove_path on sys.path and imports prove_path, and prove_path.py:210 imports records_audit. 00dac86f8bf moved both files back to benchmarks/prove_path. The suite's inputs named only benchmarks/prove_path/sources.json from that directory.
Boundary rule: tests/test_boundaries.py says "nothing outside benchmarks/ imports benchmarks/". test_kept_scan is inside benchmarks/, so BENCHMARKS_IMPORTERS and the other lists stay as they are.
Added lines, benchmarks/network_traces/pyproject.toml [tool.verity.tests] inputs, beside "benchmarks/prove_path/sources.json":
+          "benchmarks/prove_path/prove_path.py", "benchmarks/prove_path/records_audit.py",
The comment now reads "SPOT's prove_path, which test_kept_scan imports with the records_audit it imports, reads its sources' list." I rewrapped the next two comment lines. The test itself is unchanged.
Guard, run with the runner, `uv run tools/check/suites.py network_traces --fresh -- tests/test_kept_scan.py`, on this VM in a worktree with its own synced .venv and fixtures fetched. This is test_kept_scan alone, not the whole suite: there wasn't time for the whole suite before 12:10.
- with the mend: passed, 8 passed; the report's "outside" is {}. The two files read nothing else outside the inputs.
- without the mend: failed, "its tests read 2 files outside its inputs ... prove_path.py ... records_audit.py (read by test_kept_scan.py)", ci's failure exactly.
Also: the TOML parses (51 inputs); tests/test_repository.py 17 passed.

