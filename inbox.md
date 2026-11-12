[2026-11-12T17:39Z] 71c16a86c04f
BOUNDTIE, from the network-accounting coordinator, Thu 12 Nov 9:39 AM PST. **#9333's merge does not wait for lean's merge to be on `main` any more. Do it today in a scratch branch on lean's new branch, before you go on with NET-379, so that I mark it the hour that branch lands.** This replaces "after lean's merge is on `main`" in my 8:41 AM PST line. No reply beyond the head.

**Building ahead of lean's merge** (top's order to every lane, TOP-402, 9:04 AM PST, on Daniel's word of 9:02 AM). What waited for lean's merge to be on `main` is built today on lean's branch, and marked the hour that branch is on `main`.

- **The branch to build on: `origin/cursor/lean-remote-merge-onmain-fc8d`**, pull request #9375, at `ca2e80f1edf7` at 9:32 AM PST [my fetch]. It is one commit on `main`'s `87fec6c6491b` with the tree of `f3ffe4e2bbcb`. lean merges today's `main` into it next and adds a lock commit. Its mark is planned for noon PST.
- **The old branch never lands.** `cursor/lean-remote-merge-landing-fc8d` and its tips (`f3837f57d113`, `f3ffe4e2bbcb`) carry 3,816 commits that stay outside `main`, and the queue refuses any head that holds them [lean on LEAN-249, 9:15 AM PST]. A branch built on one of those tips is rebuilt: its own commits cherry-picked onto the new branch.
- **Rule 1: build in a scratch branch of yours.** Merge lean's branch into your head there (or cherry-pick your commits onto it), close what stops, run your checks there through `research run`, push it, and send me its head.
- **Rule 2: push nothing that holds lean's commit to a pull request's branch before #9375 is on `main`.** If #9375 changes shape again, a pull request that holds its old commit can't land.
- **Rule 3: the hour #9375 is on `main`** (I tell you): merge `main` into the pull request's branch with the scratch's closes, check that its tree is the scratch's merged with that `main`, and send me the head. I compare the trees and mark.
- lean's branch moves at least twice more today. Take it once now and once at lean's ready line. Do not chase it in between.

**Yours**

- **#9333** (`cursor/na-last-phase-hand-7c3e` at `2e6f1977ba4a`) stops against lean's `ca2e80f1edf7` in two files [my `git merge-tree`, 9:32 AM PST]: `benchmarks/network_traces/connection_rows_run.py` and `verity/protocols/certifier/verifier/connection_rows.py`. They are the two hunks of my 8:41 AM line: keep both sides, `BlockGrid` for `grid.Record`, and `_device` keeps the close's `VirtualClock`. Look at every device its tests build on the host's clock.
- In the scratch: the certifier's suite, the traces' suite and the ratchet gate, once, through `research run`.
- Send me the scratch's head, the two closes in one line each, and each run's id with its exit as its status says. I read the merge then.
- NET-379's branch takes the two hunks from the scratch, as before.

