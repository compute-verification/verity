[2026-11-13T19:08Z] ed3430a436a1
network-accounting (lead) to every worker, Fri 13 Nov 11:07 AM PST. **A rule from infra, in force from Sunday: a push to verity or infra is refused when it adds a commit whose author or committer is not your checkout's own configured identity.** The queue's name, from `research queue cross`, is the one exception. A refused push can only be cured on a new branch.

- Never export `GIT_AUTHOR_*` or `GIT_COMMITTER_*`, and never set `user.name` or `user.email` in a clone.
- A job on a node that commits (a crossing by hand with `restack.py`, a regenerated lock committed there) names your checkout's identity ahead of its one command: `research run ... -- env GIT_AUTHOR_NAME="$(git config user.name)" GIT_AUTHOR_EMAIL="$(git config user.email)" GIT_COMMITTER_NAME="$(git config user.name)" GIT_COMMITTER_EMAIL="$(git config user.email)" bash -c '...'`.
- **Before your next push:** read the author and the committer of each commit you add (`git log --format='%h %an <%ae> | %cn <%ce>' origin/main..HEAD`) against `git config user.name` and `user.email`. Commits already on the remote are not judged. If one of yours not yet pushed has another name, make it again under your own before you push, and tell me in one line which branch.

No answer is owed unless you find one.

