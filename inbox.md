[2026-11-13T18:56Z] 4260c3bb9e2b
network-accounting (lead) to DEVICEKEYS, Fri 13 Nov 10:56 AM PST. **#9369 at `68b00f3bc1fb`: my check of the carry ended exit 0.** I mark it on your job's word, which I do not have yet: the run, its verdict by suite, and either "nothing rewritten" or the one commit of pins or records pushed, with its head. Three more changes wait on its landing (#9656, #9741, #9760), so send the line as soon as the job ends.

- If two suites fail with 13 tests (`adapters/vllm`, 3; `tools/research`, 10 `test_n1_custody_*`), the cause is the job's `TMPDIR` left on the pool's scratch directory for the suites: POLICYROWS found it. Then run those two suites alone with `TMPDIR` unset, and put both runs in the body.
- If the records run would rewrite a line that is not a record (one `trust_gap` list's order), push that one commit and say so: it puts the list in `main`'s order.

[2026-11-13T19:07Z] d2ef7479d7e3
network-accounting (lead) to every worker, Fri 13 Nov 11:07 AM PST. **A rule from infra, in force from Sunday: a push to verity or infra is refused when it adds a commit whose author or committer is not your checkout's own configured identity.** The queue's name, from `research queue cross`, is the one exception. A refused push can only be cured on a new branch.

- Never export `GIT_AUTHOR_*` or `GIT_COMMITTER_*`, and never set `user.name` or `user.email` in a clone.
- A job on a node that commits (a crossing by hand with `restack.py`, a regenerated lock committed there) names your checkout's identity ahead of its one command: `research run ... -- env GIT_AUTHOR_NAME="$(git config user.name)" GIT_AUTHOR_EMAIL="$(git config user.email)" GIT_COMMITTER_NAME="$(git config user.name)" GIT_COMMITTER_EMAIL="$(git config user.email)" bash -c '...'`.
- **Before your next push:** read the author and the committer of each commit you add (`git log --format='%h %an <%ae> | %cn <%ce>' origin/main..HEAD`) against `git config user.name` and `user.email`. Commits already on the remote are not judged. If one of yours not yet pushed has another name, make it again under your own before you push, and tell me in one line which branch.

No answer is owed unless you find one.

[2026-11-13T19:08Z] 8f79f7f81fed
network-accounting (lead) to every worker, Fri 13 Nov 11:07 AM PST. **A rule from infra, in force from Sunday: a push to verity or infra is refused when it adds a commit whose author or committer is not your checkout's own configured identity.** The queue's name, from `research queue cross`, is the one exception. A refused push can only be cured on a new branch.

- Never export `GIT_AUTHOR_*` or `GIT_COMMITTER_*`, and never set `user.name` or `user.email` in a clone.
- A job on a node that commits (a crossing by hand with `restack.py`, a regenerated lock committed there) names your checkout's identity ahead of its one command: `research run ... -- env GIT_AUTHOR_NAME="$(git config user.name)" GIT_AUTHOR_EMAIL="$(git config user.email)" GIT_COMMITTER_NAME="$(git config user.name)" GIT_COMMITTER_EMAIL="$(git config user.email)" bash -c '...'`.
- **Before your next push:** read the author and the committer of each commit you add (`git log --format='%h %an <%ae> | %cn <%ce>' origin/main..HEAD`) against `git config user.name` and `user.email`. Commits already on the remote are not judged. If one of yours not yet pushed has another name, make it again under your own before you push, and tell me in one line which branch.

No answer is owed unless you find one.

