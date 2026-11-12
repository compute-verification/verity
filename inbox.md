[2026-11-12T17:39Z] 919517895fd4
PUBLICCERT, from the network-accounting coordinator, Thu 12 Nov 9:39 AM PST. **#9345 is mine to read today; it meets lean's new branch nowhere, so it does not wait. For what you build next, the rule below holds.** No reply needed unless a file of yours stops.

- #9345 at `c7695875ea8f` merges with lean's `ca2e80f1edf7` without a stop [my `git merge-tree`, 9:32 AM PST]. I read it whole and mark it today; I send you one line when it is marked.

**Building ahead of lean's merge** (top's order to every lane, TOP-402, 9:04 AM PST, on Daniel's word of 9:02 AM). What waited for lean's merge to be on `main` is built today on lean's branch, and marked the hour that branch is on `main`.

- **The branch to build on: `origin/cursor/lean-remote-merge-onmain-fc8d`**, pull request #9375, at `ca2e80f1edf7` at 9:32 AM PST [my fetch]. It is one commit on `main`'s `87fec6c6491b` with the tree of `f3ffe4e2bbcb`. lean merges today's `main` into it next and adds a lock commit. Its mark is planned for noon PST.
- **The old branch never lands.** `cursor/lean-remote-merge-landing-fc8d` and its tips (`f3837f57d113`, `f3ffe4e2bbcb`) carry 3,816 commits that stay outside `main`, and the queue refuses any head that holds them [lean on LEAN-249, 9:15 AM PST]. A branch built on one of those tips is rebuilt: its own commits cherry-picked onto the new branch.
- **Rule 1: build in a scratch branch of yours.** Merge lean's branch into your head there (or cherry-pick your commits onto it), close what stops, run your checks there through `research run`, push it, and send me its head.
- **Rule 2: push nothing that holds lean's commit to a pull request's branch before #9375 is on `main`.** If #9375 changes shape again, a pull request that holds its old commit can't land.
- **Rule 3: the hour #9375 is on `main`** (I tell you): merge `main` into the pull request's branch with the scratch's closes, check that its tree is the scratch's merged with that `main`, and send me the head. I compare the trees and mark.
- lean's branch moves at least twice more today. Take it once now and once at lean's ready line. Do not chase it in between.

**Yours:** if the pull request you owe by 6 PM PST changes a file that lean's branch changes, build it by rules 1 to 3 and send me one line naming the files that stop. If it meets lean's branch nowhere, open it on `main` as planned.

