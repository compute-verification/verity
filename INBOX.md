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
