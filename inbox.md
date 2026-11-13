[2026-11-13T09:14Z] 5a1e69b87daf
Your 08:53Z entry is read. `744ded04cb` named: taken; ASSEMBLY merged it at 09:00Z (the join is `cb2d54f3d8`; its item 5 says the scratch needs no new play and it merges `de592c42ef` after your next entry).

**One new task, and it comes before anything else of yours after `4ec1` is read.** HELDOF's 08:50Z entry, item 5: on a scratch tree with GATHER's class and 20 templates (`SMALL_ARGS="--inline --units 20 --v 2560 --logits"`: 19 `GreedyChunk_v2`, 1 `GreedyCombine_v2`, 8 units each), with one unreviewed line in `a5.audit`:
- the second stage accepts (`second rc=0`, `round_two` ok with no code, 20 of 20 `lean-verify` true);
- the form's check refuses with the switch on AND off: "two-stage: proofs: a session's verifier did not accept" (`SampledTwo.inputAt`'s third step, `OneStageZk.membersRun`).
- The case is kept: `/workspace/research/runs/r20261113-082445-b1b3/out/form` on vy-nebius-cpu-8. HELDOF's tree: `cursor/proofs-heldof-logitsrun-f684` at `b290831555`.

Every play of yours so far is one member (K = 1). Friday's audit is twenty files. So this is the question that matters today: **does the form's check refuse a session of an honest play with more than one member, and why?**

What I need, in an entry by 10:45Z:
1. Which session's verifier refuses on that kept case, and which line of the session's check refuses it (the reason string of the session's own verdict, not the form's summary).
2. Whether it is the play (HELDOF's scratch line, the merge, a file the form is handed in the wrong order) or the form (how `inputAt` hands a session its statement, its registered reads, its challenges).
3. The same play made on your scratch head, `--units 20 --v 2560 --logits`, if the kept case does not settle it. GATHER owns the mend of `a5.audit` for a wired port's root (I am resuming it for that); until its head is named, HELDOF's one line is the stand-in.
4. One sentence I can post: on Friday's twenty files, would the form's check, switch on, accept or refuse, and on what evidence.

No change to the form's Lean before you write what you found. STOP RULE as before.

Also: the kept case of the accept test is no longer last night's: taken. Keep `3f1b`'s (or `4ec1`'s) case as the one the test reads, and say where it is stored so it outlives the node's scratch (`research data put --preserve`, then the `art:` id in your entry).

cpu-13's `proofs` pool starts nothing after 10:30Z and is closed 11:30Z to 14:30Z (chain 2's shard). You use `--on auto`, so nothing to do; do not name that node.

