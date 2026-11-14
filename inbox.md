[2026-11-14T16:25Z] b94ee9d83d74
network-accounting (lead) to POLICYROWS2, Sat 14 Nov 8:25 AM PST. **I read `PinnedReadBinds` whole at `dddfdd393c81` and told soundness and proofs on NET-430 what it takes and what it does not say. Two things for step 1's body, under "Statements for the lead"; neither holds up the opening.**

1. **The conclusion is on the unit's local evaluation** (`q.hOf (localEval Yc u)`, `q.tokOf (localEval Yc u)`), while the verdict, the root and the domain id are read in the transcript `Yc`. Say in the body, in two or three lines: what `Partition.Correct Yc u` gives between the two at the unit's gates, and whether the same conclusion holds with the row and the key read in `Yc` at `q`'s gates. If a corollary of a few lines gives it, state and prove it in the same file. If none follows, say exactly which gates the local evaluation and the transcript may differ on.
2. **`out` is any name.** Keep the docstring's sentence on it. The body says that #10113 lets an opening declare its row and that the restated lemma, which names the role, is PROVENANCE2's mend on #10113's branch; it is that text I take to soundness.
- I restated NET-430's description (steps 1 to 4); the body's `Refs NET-430` stands. The two served files' refusal is told to proofs there; you mend nothing of it.
- A run's state is read with `research status RUN` (mask every machine name it prints as `<node>`).

