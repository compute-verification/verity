[2026-11-13T16:33Z] 16821c88d28a
network-accounting (lead), Fri 13 Nov 8:33 AM PST. Three of our branches stopped at a later move's base this hour, each on one file. So that nobody waits on me for the same word:

**The carry's rule for a stop, the same for every branch of ours from now:**
- **`conflicted` at step `base`, at any move:** cross the moves before it with the tool (those moves named with `--move-commit`, no pins), merge the base it names into that result by hand, push as a fast-forward, run the tool again. Never a plain merge of `main` across a move.
- **`conflicted` at step `moved` after the last move:** the tool with `--main` the last move, then one hand merge of today's `main` (ci's way, CI-489 8:04 AM PST).
- **Anything else stops and comes to me** with the tool's words.
- **Each hand merge:** send me its files and each resolution in a line, and the head. Where both sides reworded one sentence, `main`'s words stand and ours is added beside them; you write no third wording.
- **A marked pull request whose carry needed a hand merge does not carry its verdict:** say so with the heads, and I read the new head before I mark.

[2026-11-13T17:14Z] 48d746ef5923
network-accounting (lead), Fri 13 Nov 9:14 AM PST, to the owners of our marked pull requests that crossed the moves (#9369, #9547, #9639, #9664). One rule, so nobody waits on me for a word.

**The job's generated commit on a marked, carried head: my word is given now.**
- If the job at the carried head rewrites nothing, send me "nothing rewritten" and the run's id. I check the carry from the old head and mark again.
- If it rewrites pins or a lock's records, and nothing else: push that as one commit of generated content only, as a fast-forward, and send me its head with the run's id and the files. I check the carry from the old head to that head and mark it once.
- If it rewrites anything else, or a suite fails: push nothing, and send me the list.
- After that push, nothing more goes onto the pull request until it has landed.

A carried head whose lock records are stale by name is expected: the moves' script renames the hand-written sections and leaves the records to the records run.

