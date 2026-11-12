[2026-11-12T20:41Z] 06f36ffcd837
SORT, TRN-146: carry of #8563, plus #8177's records.

#8563. What it is for: the Lean verifier opens a supplied row-seg value under its row-seg read, and round 2 reads a linked port at one; the PR adds difftests for both. READ (the PR title).
- Branch `cursor/carry-8563-dc90`, head c67169ef6275e0ad791c2ee44864a9ba7189d1da: #8563's head with `main` 30d6fcc92ff7 merged in, across the three moves. Restack stopped at the base merge, so I merged by hand. RAN.
- The diff against `main` is tests only: `verity/protocols/spot/verifier/lean/tests/test_difftest.py` (+66 lines, a net count) and `verity/protocols/flock/prover/tests/test_lean_registered_bit_rows.py` (+26), 83 lines added and 9 removed in all. In `test_difftest.py` I resolved four hunks by keeping both sides: `_read_linked` takes both sides' parameters (`value`, `root`, `kept`), all the helpers stay (`_supplied_linked`, `_row_schema`, `_shadowing`), and both new tests stay. RAN (`git diff --stat`).
- Quick suites `r20261112-190809-3ca0`, rc 0, last line: `60 suites: 60 passed (6 cached), 0 failed; 176.3 min of test time`. RAN (fetched stdout.log).
- No Lean changed, so there is no records run. Question: none.

#8177's records run, `r20261112-185903-4d95`, rc 0. It audited `verity/protocols/spot/verifier/lean`, then its requirers: `verity/protocols/certifier/verifier/lean`, `security` (with `security/Proofs`) and `security/Properties`. Every one is `AUDIT … PASS` with axioms propext, Classical.choice and Quot.sound only. The run kept no `lean-audit/record/`, so #8177 has no lock record to commit. RAN (fetched stdout.log and `lean-audit/`).

