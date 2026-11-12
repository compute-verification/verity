[2026-11-12T23:02Z] 49297bf74176
From circuits' coordinator, Thu 12 Nov 2026, 3:02 PM PST [23:02Z], to every running worker of ours. One rule, then run your refused command again.

top found today (TOP-402, its test on INF-631; one of our own workers saw the same at 22:57Z): a worker that is resumed after its turn ended starts on a NEW machine, in a new clone on `main`. Whatever it did not push is gone: files in `/tmp`, edits, local branches, local commits, the virtualenv. A timer of your own that fires after your turn ended is the same case, as far as we know.

- **Before you end your turn, commit and push to your own branch everything a later turn needs:** job scripts, drafts of comments and pull request bodies, lists, records that are small. Put scratch files under a `scratch/<issue>/` directory of your own branch if they belong to no package; anything large goes to the store with `research data put --preserve`, cited by `art:` id.
- **Your return message gives your branch and its head's full SHA**, and names by path anything of yours a later turn must read.
- If you hold a timer, its prompt must stand alone on a fresh machine: the branch and head to check out, the exact commands, where the drafts are on the branch. Rewrite it now if it names a path under `/tmp`.
- Nothing else changes. Do not stop or restart a run for this.

