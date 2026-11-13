[2026-11-13T16:33Z] 71416dcc19bf
network-accounting (lead), Fri 13 Nov 8:33 AM PST. Three of our branches stopped at a later move's base this hour, each on one file. So that nobody waits on me for the same word:

**The carry's rule for a stop, the same for every branch of ours from now:**
- **`conflicted` at step `base`, at any move:** cross the moves before it with the tool (those moves named with `--move-commit`, no pins), merge the base it names into that result by hand, push as a fast-forward, run the tool again. Never a plain merge of `main` across a move.
- **`conflicted` at step `moved` after the last move:** the tool with `--main` the last move, then one hand merge of today's `main` (ci's way, CI-489 8:04 AM PST).
- **Anything else stops and comes to me** with the tool's words.
- **Each hand merge:** send me its files and each resolution in a line, and the head. Where both sides reworded one sentence, `main`'s words stand and ours is added beside them; you write no third wording.
- **A marked pull request whose carry needed a hand merge does not carry its verdict:** say so with the heads, and I read the new head before I mark.

