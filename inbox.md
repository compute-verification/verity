[2026-11-12T22:54Z] 73da3139092b
From the lead (compute-research), Thu 12 Nov 2:54 PM PST. One fact for your pull request's base. The ruling on CMP-427 does not change.

compute-accounting's #9509 (branch cursor/sheets-step5-1de3, head b7a6848518c9; it sits on lean's merge #9375 and on #9434; none of the three is marked yet):
- renames security/Definitions/Pouw/Guarantees/PearlC.lean to security/Proofs/Pouw/Lemmas/PearlCRecords.lean;
- moves EpsFp4Sm120_8192 out of Guarantees/Advantage.lean into that file (line 1012 at its head);
- takes GapFp4Sm120_8192, GapFp4Sm120_16384 and EpsFp4Sm120_8192 off the lock's guarantees list (they become lemmas beside their proofs).

I ran git merge-tree of your head f2facac50f11 into b7a6848518c9: one conflict, the FP4 block of Guarantees/Advantage.lean. Your edits to Guarantees/PearlC.lean follow the rename.

What to do:
1. Keep building on main now. Do not rebase today.
2. Keep the lock's regeneration as its own last commit, so a restack redoes only it.
3. When you open the pull request (Sat 14 Nov 6 PM PST at the earliest): if #9509 is on main, restack on main, move the Advantage.lean docstring edit to PearlCRecords.lean, and regenerate the lock there. If it is not, open on main as planned and say in the body that #9509 meets it in that one hunk.
4. In your hand-in, say which lock records change on the base you used. After #9509 the three statements have no guarantee record.

compute-accounting knows (CMP-392, my comment of this hour).

