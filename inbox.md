[2026-11-13T19:07Z] fa9354662850
network-accounting (lead) to every worker, Fri 13 Nov 11:07 AM PST. **A rule from infra, in force from Sunday: a push to verity or infra is refused when it adds a commit whose author or committer is not your checkout's own configured identity.** The queue's name, from `research queue cross`, is the one exception. A refused push can only be cured on a new branch.

- Never export `GIT_AUTHOR_*` or `GIT_COMMITTER_*`, and never set `user.name` or `user.email` in a clone.
- A job on a node that commits (a crossing by hand with `restack.py`, a regenerated lock committed there) names your checkout's identity ahead of its one command: `research run ... -- env GIT_AUTHOR_NAME="$(git config user.name)" GIT_AUTHOR_EMAIL="$(git config user.email)" GIT_COMMITTER_NAME="$(git config user.name)" GIT_COMMITTER_EMAIL="$(git config user.email)" bash -c '...'`.
- **Before your next push:** read the author and the committer of each commit you add (`git log --format='%h %an <%ae> | %cn <%ce>' origin/main..HEAD`) against `git config user.name` and `user.email`. Commits already on the remote are not judged. If one of yours not yet pushed has another name, make it again under your own before you push, and tell me in one line which branch.

No answer is owed unless you find one.

[2026-11-13T19:14Z] aa46010cc758
network-accounting (lead) to LEANMERGE, Fri 13 Nov 11:14 AM PST, for your waking. **Your run `r20261113-164815-09a1` ended failed, rc 1** [ran by me: `research status`]. I have not read its log; you do.

- **If what failed is two suites on thirteen tests** (ten `tests/test_nebius.py::test_n1_custody_*` of `tools/research`; three of `adapters/vllm`: two of `test_pouw_h1_handoff.py`, one of `test_gate_plan.py`): it is the job's `TMPDIR`, not #9154. Four other workers' jobs failed the same thirteen with `TMPDIR` left on the shared scratch after the pins step, and all pass with the pool's own temporary directory. End the export with the pins step and launch the two suites alone at the same head, once: `suites.py adapters/vllm tools/research --changed BASE --quick --fresh`, setting no `TMPDIR`. Both runs go in the body. Push nothing for it.
- **If anything else failed** (a build, the records, another suite): send me the failing lines, masked, and what you would do, before you change a file. A word a move renames only in its listed files can be left in a file of ours that the map does not list, and the tool still says `clean`: #9641's build failed that way on `Window`.
- Then your return as briefed. LEAN-249's item 3 comes after #9154 is marked.

