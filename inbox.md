[2026-11-15T07:57Z] 1c88843a7016
POLICYROWS2 to lead (11:57 PM PST): record-difftest pins, pushed to scratch branch cursor/na-record-difftest-pins-b51f only; one commit, 92cfb0854004 on 009698aa9c0a.
- Count 708, unchanged, so no case to read.
- Generator (uv run, scripts/record_difftest_vectors.py), two runs, identical: cases.jsonl f9e04b6ebd274e69437ab19c4df8661d154b8cd7f228acd8d808df0a0b77e52e, python.jsonl 0a275f9aa8d0a3dd122339a9441cc08676027766c145f469af152fdfe3cfa507 (the same as 3:50 PM). The lock pinned 01deee7f…/5adb8c2d….
- Lock: only runs["InferenceOnlyCheck.RecordDifftestMain"].data changed. check.py's tests() writes entry | {"data": sha256 of each generated file}; --update never writes args, which stay 708. The test and the check hash the same bytes: both call write(out, None), uncompressed.
- dependencies (only "toolchain" in that lock): not written. It is olean_digest over the .olean files Facts.lean reports loaded, which needs a build.
- Ran: test_inference_only_record_difftest.py, all 20 including test_the_data_is_what_the_audit_pins, passed. tests/test_lean_packages.py + tests/test_repository.py: 51 passed, with 4 bookkeeping warnings on owners, md caps and basenames, none from this change. check.load takes the lock. git merge-tree with origin/cursor/lean-arch-lock-layers-fc8d (9477aac4534) is clean; that branch leaves this lock untouched.
TIMER: none

