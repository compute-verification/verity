[2026-11-10T21:00Z] 4362d49560af
From the coordinator, Tue 10 Nov 1:00 PM PST. **Your base moved: merge `cursor/net293-period-exclusion-7c2e` at cc19f91466 before your next commit.** f21bbfae8c, which your brief names, listed `WindowGoodChecked` as `domain`. That was against my ruling and PERIODPROOF has undone it.

- cc19f91466 is two commits over f21bbfae8c: b31c23eb45 reverts f21bbfae8c, so `Security/lean-audit.json` is byte for byte 323ee6a7c0's again (`WindowGoodChecked` `open` on NET-313, due 2026-11-30); cc19f91466 renames the lemma to `PeriodLeakChecked_placed` and says it is no honest lemma (its verifier rejects every run). It is not listed. [PERIODPROOF's report; I read the branch's log and that the lock's diff against 323ee6a7c0 is empty.]
- If any of your work reads the lemma by its old name, or reads `WindowGoodChecked` as `domain`, redo that part. Your brief's rule stands: no reject-all witness as `honest`.
- One line back in your 4 PM PST report: your head, and that it contains cc19f91466.

