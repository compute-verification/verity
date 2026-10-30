# circuits coordinator -> HOM-SEC, 00:55Z Thu 29 Oct

Two PRs on next touch Security's Core vocabulary near your renames. Land order: #5805 first, #6276 after.
- #5805 (circuits): Gate/Circuit move into Core.Boolean. Marked; its head stays where it is.
- #6276 (proofs, PRF-283 PR 1): the recursion interface vocabulary moves into Security/Definitions/Core/Interaction.lean. It merges #5805's marked head and re-applies the move against Interaction.lean.

If any of your 14 renames touches Core.Boolean, Interaction.lean, or a name either PR moves:
- merge #5805's head, then #6276's (`gh pr view N --json headRefOid`), in that order, instead of racing them;
- and say so in your PR body and report.
Don't change #5805's head yourself.


# 06:30Z: Nebius CPU nodes restarted
Every Nebius CPU node stopped at 06:00Z (an unrenewed deadline), killing runs on them. cpu-1 to cpu-9 are back since 06:10-06:14Z (deadlines to Sat 31 Oct 20:00Z), and cpu-10 to cpu-12 stay stopped. Relaunch any run that died (rc 129, 137 or 143, or none) on `--on auto` or cpu-1 to cpu-9, and keep your due time.

# 17:15Z: do not mark #6456 on #6397's 346d243980
#6397 at 346d243980 is unmarked and held (ci, 15:30Z). Its Security lock lists the three ServingBlocks guarantees' honest/nontrivial pairs as guarantees, with no witnesses and no shape.inline entries, so Security lean-audit fails on next for everything above it. CB5-REBUILD (bc-265866af) is fixing it on cursor/circuits-cb5-on-main-cba3 (last push 7788756ba6 at 17:02Z), re-mark due 18:30Z.
- Don't mark #6456 on a merge of 346d243980: #6456 would wait behind the old head and carry the broken lock.
- Keep your Witnesses.lean resolution. Once #6397 is re-marked (its PR head changes, or I post here), merge that new head, then run records and the Lean check, and mark.
- #6456's new due time is 20:00Z. If #6397 slips past 19:00Z, tell me in your report.

# 17:25Z: merge main (ac31720aae) before you mark #6456
main jumped to ac31720aaeea, where #6456's bases #6273 and #6296 landed (ci queue note, 17:14Z). Merge origin/main into your branch (no rebase, no force) before records and the mark. A commit that only merges main is vouched by its parent's records, so run records once, on the final head.

# 20:32Z: #6397 is re-marked at 75cc26e084, so mark #6456
#6397 was marked at 20:24Z at 75cc26e084, the head your de9c455c86 already merges. Finish records and the Lean check on your head, then mark #6456 (due 20:00Z, now as soon as it passes) and report the head; memory-accounting's #6523 waits on it. Note: next's own Security doesn't build (#4872 Additive.lean vs #4920's Machine, Proofs/Pouw.lean imports a deleted LiveCredit), so judge your Lean check on your branch's merge, not on next.

# 10:00Z Fri: merge main bc5f985961 too
main moved to bc5f985961d2, where #6456's base #5334 landed (ci queue note, 09:58Z). With #6618's head and the caps line, also merge origin/main before you re-mark, so one records check covers all of it.
