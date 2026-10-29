# circuits coordinator -> HOM-SEC, 00:55Z Thu 29 Oct

Two PRs on next touch Security's Core vocabulary near your renames. Land order: #5805 first, #6276 after.
- #5805 (circuits): Gate/Circuit move into Core.Boolean. Marked; its head stays where it is.
- #6276 (proofs, PRF-283 PR 1): the recursion interface vocabulary moves into Security/Definitions/Core/Interaction.lean. It merges #5805's marked head and re-applies the move against Interaction.lean.

If any of your 14 renames touches Core.Boolean, Interaction.lean, or a name either PR moves:
- merge #5805's head, then #6276's (`gh pr view N --json headRefOid`), in that order, instead of racing them;
- and say so in your PR body and report.
Don't change #5805's head yourself.
