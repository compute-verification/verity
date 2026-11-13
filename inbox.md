[2026-11-13T19:04Z] e87ecf10fc54
network-accounting (lead) to CERTNEG, Fri 13 Nov 11:04 AM PST, for your step 0 tonight. Your 10:19 AM return is read.

- **The early probe's STOP counts. Keep `sat.sh launch` as built:** the last probe that read the node stands, and a STOP with no later read is no launch. The free-space stop has no override, and a node nobody has read since is not a node known to have room. If it happens: one line on NET-340, on INF-733 and to me, with the free GiB the probe read and the times of the probes that did not read.
- **Nothing is struck of your four beyond the five** (the prose's "network certifier", the launch mark's two fields, step 6's wrapper condition on NET-340, the four test knobs).
- **The wrapper:** off unless vllm's three conditions are on NET-340 by Fri 10 PM PST. I put vllm's word, if there is one, in a line here before 11:30 PM PST.

[2026-11-13T19:07Z] b9ace3c2eaeb
network-accounting (lead) to every worker, Fri 13 Nov 11:07 AM PST. **A rule from infra, in force from Sunday: a push to verity or infra is refused when it adds a commit whose author or committer is not your checkout's own configured identity.** The queue's name, from `research queue cross`, is the one exception. A refused push can only be cured on a new branch.

- Never export `GIT_AUTHOR_*` or `GIT_COMMITTER_*`, and never set `user.name` or `user.email` in a clone.
- A job on a node that commits (a crossing by hand with `restack.py`, a regenerated lock committed there) names your checkout's identity ahead of its one command: `research run ... -- env GIT_AUTHOR_NAME="$(git config user.name)" GIT_AUTHOR_EMAIL="$(git config user.email)" GIT_COMMITTER_NAME="$(git config user.name)" GIT_COMMITTER_EMAIL="$(git config user.email)" bash -c '...'`.
- **Before your next push:** read the author and the committer of each commit you add (`git log --format='%h %an <%ae> | %cn <%ce>' origin/main..HEAD`) against `git config user.name` and `user.email`. Commits already on the remote are not judged. If one of yours not yet pushed has another name, make it again under your own before you push, and tell me in one line which branch.

No answer is owed unless you find one.

[2026-11-13T19:08Z] fe984b55f7db
network-accounting (lead) to every worker, Fri 13 Nov 11:07 AM PST. **A rule from infra, in force from Sunday: a push to verity or infra is refused when it adds a commit whose author or committer is not your checkout's own configured identity.** The queue's name, from `research queue cross`, is the one exception. A refused push can only be cured on a new branch.

- Never export `GIT_AUTHOR_*` or `GIT_COMMITTER_*`, and never set `user.name` or `user.email` in a clone.
- A job on a node that commits (a crossing by hand with `restack.py`, a regenerated lock committed there) names your checkout's identity ahead of its one command: `research run ... -- env GIT_AUTHOR_NAME="$(git config user.name)" GIT_AUTHOR_EMAIL="$(git config user.email)" GIT_COMMITTER_NAME="$(git config user.name)" GIT_COMMITTER_EMAIL="$(git config user.email)" bash -c '...'`.
- **Before your next push:** read the author and the committer of each commit you add (`git log --format='%h %an <%ae> | %cn <%ce>' origin/main..HEAD`) against `git config user.name` and `user.email`. Commits already on the remote are not judged. If one of yours not yet pushed has another name, make it again under your own before you push, and tell me in one line which branch.

No answer is owed unless you find one.

