[2026-11-13T00:20Z] 7d2af016e855
WALKER, TRN-145. The draft is open: https://github.com/compute-verification/verity/pull/9558. It is a draft, unmarked, based on main, linked to TRN-145, head 102ad1d702 (centre 2^-3, with 1(a)'s and 1(b)'s tests).

All of these RAN at 102ad1d702:
- Suites, gate and ruff, r20261112-233904-3665:
  - 3 suites passed, 0 failed.
  - Ratchet gate 4520abe708b0..102ad1d7021c: 0 failed.
  - Ruff: clean.
- Count and dump, r20261112-233844-46e6:
  - circuit_properties: 81 passed.
  - training_slowdown: 577 passed.
  - grader: 28 passed, 3 skipped.
  - The lf dump is identical to the base, and no library entry is refused.
  - Every point-check record is identical to 53c7b497f2's except the 2 new tests, so no pinned figure or pass moved.
- lf at 2^-2, 2^-3 and 2^-4, both loops (r20261112-233844-46e6):
  - 0 origins, 8 nonfinite nodes (the mask's), 0 NaN nodes.
  - 23 claims, 0 refused, N 6,160,384 and 12,320,768.
- Reverts, r20261112-233931-95e0: all 11 caught.
  - The nan-rule revert fails both NaN tests.
  - compare-by-difference fails the inf-rows test and the specific-overflow test.

Open:
- s-t20 at 2^-3 is r20261112-233958-dacc, still running. The old-centre run r20261112-222055-84cf is still running too. I report both when they end.
- The centre was chosen on lf alone. e4m3 sits off-centre, as it already did before the change.
- The 464/116/29/7 numbers are in the PR body, not beside the constant.
- The draft is stacked on #9315.

[2026-11-13T00:20Z] edcae8d3d11e
PROBE, Thu 12 Nov 4:20 PM PST. **Every check you listed passes at `b23c853a35bc`, and the ratchet shows exactly the three module-name sites; #9474's body has them all.**

Branch `cursor/nci-record-check-6111`, head `b23c853a35bc566197f506851381ac04545f8e57`, pushed. It merges onto today's `main` (4175265019) without conflict (RAN locally, `git merge-tree`).

- The package's Lean check, r20261112-231419-57c6, rc 0 (RAN): `AUDIT: PASS; reports in …/r20261112-231419-57c6/lean-audit`. The package line: `PASS 1730 declarations in 17 modules; axioms propext, Classical.choice, Quot.sound; 0 guarantees`. The record difftest's run log: `record difftest: 670 cases compared, 0 differences`.
- Lean try of `RecordCheckMain.lean`, r20261112-231447-1238, rc 0 (RAN): `lean exit 0 in 50 s`; last line `axiom probe: no theorems in it`.
- The suites, `--changed 5f0c008aa0 --quick`, 57 suites in five groups, all rc 0 (RAN):
  - r20261112-231420-42b6: `6 suites: 6 passed (0 cached), 0 failed`.
  - r20261112-231447-1b28: `12 suites: 12 passed (0 cached), 0 failed`. It includes the network suite, `88 passed` with no skips, with `test_nci_record_check.py` run whole, the compiled command's test included.
  - r20261112-231513-5d06: `1 suites: 1 passed (0 cached), 0 failed` (verity-vllm).
  - r20261112-231525-4494: `19 suites: 19 passed (0 cached), 0 failed`.
  - r20261112-231552-7fea: `19 suites: 19 passed (0 cached), 0 failed`.
- The ratchet, run locally against today's origin/main (RAN): `5f0c008aa040..b23c853a35bc (b23c853a35bc): failed: adds 3 retired-term site(s) … (…/NciCheck/RecordCheckMain.lean 0 -> 2; …/lean/lakefile.toml 35 -> 36)`. The sites are `RecordCheckMain.lean:3` (`public import NciCheck.RecordCount`), `RecordCheckMain.lean:15` (`open NciCheck in`) and `lakefile.toml:57` (`root = "NciCheck.RecordCheckMain"`), and nothing else.
- The served record, r20261112-204835-6b1e, was at fae1cf1b8d (RAN). The reference and the compiled command both give `{"error":"inputs"}` for check and count. Since then the exe has changed only its name and one `open`, and the script only its docstring (READ: the diff fae1cf1b8d..b23c853a35). I didn't rerun it.
- PR #9474 is a draft (READ: `gh pr view`). Its body has the "Retired terms" paragraph, with the three sites, why each stays and that the rename removes them, and "Run and read" with every run above, its last line and the head.

The diff from fae1cf1b8d is one commit, b23c853a35. It covers the target's name in the lakefile, the README, the test's `D.build`, the script's docstring and the lock's `exempt` text; the `NciCheck.` prefix dropped from prose; and `open NciCheck in` in the Lean.

Nothing of mine is in flight. The handoff note's id follows.

