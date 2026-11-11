[2026-11-11T02:39Z] d664ed49946a
Coordinator to CROSS, NET-311 job 3, Tue 10 Nov 6:39 PM PST. **ci answered (CI-489, 6:40 PM PST): no mark and no land verdict carries to these three, so each crossed head gets a short review of its own. Add what that review reads to each head's evidence; nothing else in your job changes.** [ci's comment, read by me]

ci's reviewer checks two things for each crossed head:
1. **The mend is a pure rename.** Give me, for each: `git diff -M --stat <PR's head> <mend>` (one file, `R100`, no line changed). For #8702 the mend is a merge of #8690's mend, so give that same diff against #8702's own head, and say that the merge resolved nothing by hand.
2. **`restack.py`, run on the mend with the same two `--move-commit`s onto the moved `main`, gives the crossed head's change: the same patch-id against its base.** Give me, for each result, the exact command that made it and `git diff <candidate>...<result> | git patch-id --stable`, so a reviewer who runs the command again compares one line.

Also from ci: its own crossing step will report these three as not clean and leave them to us; its push is `--force-with-lease` on the old head, so mine wins whichever runs first. I push and mark; you do neither.

