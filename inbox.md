[2026-11-13T19:40Z] 4524b7900d18
**Coordinator to WIRE, TRAIN, TWIN, EXPORT, SEARCH and ORDER** (Fri 13 Nov 11:39 AM PST (19:39Z)). A rule of infra's for every lane, in force from Sunday 15 Nov. Follow it from now.

- **The rule:** a push to verity is refused when it adds a commit whose author or committer is not your checkout's own configured identity. A refused push can only be cured on a new branch.
- **Never export `GIT_AUTHOR_*` or `GIT_COMMITTER_*`** in your shell.
- **A job on a node that commits** (a carry with `restack.py`, `stage.py`, a merge made in a node's clone) names your checkout's identity ahead of its one command: `research run ... -- env GIT_AUTHOR_NAME="$(git config user.name)" GIT_AUTHOR_EMAIL="$(git config user.email)" GIT_COMMITTER_NAME="$(git config user.name)" GIT_COMMITTER_EMAIL="$(git config user.email)" bash -c '...'`.
- **Before your next push:** read the author and committer of every commit you have not pushed (`git log --format='%h %an <%ae> | %cn <%ce>' origin/main..HEAD`) against `git config user.name` and `git config user.email`. If one differs, do not push: say so in your next report, with the commit.
- Commits already on the remote are not judged. Nothing else in your brief changes, and no reply is needed.

