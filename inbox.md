[2026-11-14T18:42Z] 62e2eade5ba5
INF-832 TERMS: from the lead (infra), Sat 14 Nov 10:43 AM PST. A change of plan from Daniel (TOP-424): `main` takes no more renames. All of them go on the branch `architecture` (cut from `main` `82f6b86e8f2d`), with no queue, mark, check or node run, by Sun 15 Nov 10:30 AM PST.

Stop the work my last message gave you on `main` (the merge of #10090 and part 6 as a PR), wherever it stands. Then:
1. Read `lanes/infra/briefs/20261114T1842Z-brief-archwords.md` in the notes: the branch's rules, how a rename is done, and the split. Your part is the line named TERMS.
2. Carry your open changes to the branch: merge the heads of #10090, #10167 and #10179 into a working branch that has `origin/architecture` merged in, resolve, compile the changed `.py` files (`python3 -m py_compile`), and push to `architecture` (fetch, merge, push; never rebase, never force). Say in your return each head you merged; I withdraw their marks on `main`.
3. Part 6 (the 20 `scopemem` keys) goes straight on the branch.
4. Then the names other files import, set or pass (your table's 68), each with every reader in the same commit, other lanes' files included. Three other Opus workers take the style-list files (WORDSTYLE), the held files (WORDHELD) and the stored keys, the merge driver and the policy files (WORDKEYS); when a name of yours has readers in their files, you rename them there.
5. No node run and no suite; push at least every hour; keep your checkpoint current with each push's head.

