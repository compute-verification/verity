[2026-11-16T22:19Z] af9c2ab0eb4c
RECORD to the training lead. #10570 (serving blocks' Lean), no mark, as asked.

- Head 6c3d28c4d68b06f82aa709d71e79325c7aebf141: main's tip at the branch point plus my lock commit, which hand-removes the deleted guarantees' entries (guarantees 9, witnesses 3, shape.frozen 3, shape.inline 4, open 6, reads 4, meaning 2, and the two ServingBlocks modules from the layers' imports). claim_ids.json unchanged.
- r20261116-215004-e354 done: `lean_changed.py --records security --update` at that head. security and security/Properties both AUDIT PASS (axioms propext, Classical.choice, Quot.sound), OUTCOME pass, "0 files changed": the regenerated records equal the committed lock, so nothing more to commit [ran].
- Not marked. Core's `Verity.*` frozen reasons that mention "serving-block" in prose are left as they are (outside the deleted declarations) [read].

Also from the same order: #10607 was already linked to TRN-177 (the link command said so) [ran]. #10556's Lean check passed at e082926 (r20261116-215914-106d done, AUDIT PASS) [ran]; its 95-suite run r20261116-215836-155e is still running, so #10556 isn't marked yet. #10553's run r20261116-213739-a542 has the research suite red (2 test_treelock.py tests); #10553 touches four READMEs and nothing under tools/research, and I'm running main's research suite fresh (r20261116-220844-9735) to see whether main shares it before I decide the mark.

