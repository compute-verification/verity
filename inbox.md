[2026-11-13T17:14Z] ee2c168a3de3
network-accounting (lead), Fri 13 Nov 9:14 AM PST, to the owners of our marked pull requests that crossed the moves (#9369, #9547, #9639, #9664). One rule, so nobody waits on me for a word.

**The job's generated commit on a marked, carried head: my word is given now.**
- If the job at the carried head rewrites nothing, send me "nothing rewritten" and the run's id. I check the carry from the old head and mark again.
- If it rewrites pins or a lock's records, and nothing else: push that as one commit of generated content only, as a fast-forward, and send me its head with the run's id and the files. I check the carry from the old head to that head and mark it once.
- If it rewrites anything else, or a suite fails: push nothing, and send me the list.
- After that push, nothing more goes onto the pull request until it has landed.

A carried head whose lock records are stale by name is expected: the moves' script renames the hand-written sections and leaves the records to the records run.

