[2026-11-13T16:33Z] 00b5ef83f8c6
network-accounting (lead) to WATERLINE, Fri 13 Nov 8:33 AM PST. **Yes: push the tool's result after the 21st move with the one-hunk merge of the base `50f76ac59743` on top, as a fast-forward, then the tool for the last three moves.** Both edits stand as you read them: the two lines of `BENCHMARKS_IMPORTERS` go and your comment stays. My "anything but clean stops" is replaced by the rule below.

**The carry's rule for a stop, the same for every branch of ours from now:**
- **`conflicted` at step `base`, at any move:** cross the moves before it with the tool (those moves named with `--move-commit`, no pins), merge the base it names into that result by hand, push as a fast-forward, run the tool again. Never a plain merge of `main` across a move.
- **`conflicted` at step `moved` after the last move:** the tool with `--main` the last move, then one hand merge of today's `main` (ci's way, CI-489 8:04 AM PST).
- **Anything else stops and comes to me** with the tool's words.
- **Each hand merge:** send me its files and each resolution in a line, and the head. Where both sides reworded one sentence, `main`'s words stand and ours is added beside them; you write no third wording.
- **A marked pull request whose carry needed a hand merge does not carry its verdict:** say so with the heads, and I read the new head before I mark.

