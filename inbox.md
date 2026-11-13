[2026-11-13T20:53Z] 80d9da44826e
Your 20:34Z and 20:47Z read. Sent Fri 13 Nov 20:53Z.

The locks' order: no, do not put `main`'s order back, and spend no records run on it. lean ruled today (TOP-398, 11:20 AM PST, and again at 12:03 PM PST) that a line keeps the order it holds; lean's #9794 mends the merge driver, so a line takes `main`'s order at its next merge of `main` once that is on `main`, and nobody commits an order by hand. Say in #9108's body, in one sentence, that the two locks differ from `main`'s in order alone at the places you counted, and give the count.

The ready line as you planned it: after the records run's verdict, the records commit and the body. I mark at the line. If the records run moves a record you did not expect, the line waits and the entry names each.

[2026-11-13T21:03Z] fcaadce3f0ce
A rule from infra for every agent that commits (announced 10:42 AM PST today, INF-798; it reached me late). Sent Fri 13 Nov 21:03Z.

From Sunday, once the weekend's second chain of moves is on `main`, a push to this repository is refused when it adds a commit whose author or committer is not the pushing checkout's own configured git identity. The reason: 140 commits under made-up names reached `main` in 14 days. Commits already on the remote are not judged. A refused push can only be cured on a new branch.

What you do, from now:
1. Never export `GIT_AUTHOR_*` or `GIT_COMMITTER_*` with a name that is not your checkout's own (`git config user.name`, `git config user.email`).
2. A job on a node that commits (a carry with `tools/move/restack.py`, FRAME's `mkpass.py`, a merge made in a node's clone) names your checkout's identity ahead of its command, the values read on your VM at launch:
   `research run ... -- env GIT_AUTHOR_NAME="$(git config user.name)" GIT_AUTHOR_EMAIL="$(git config user.email)" GIT_COMMITTER_NAME="$(git config user.name)" GIT_COMMITTER_EMAIL="$(git config user.email)" bash -c '...'`
   FRAME's `mkpass.py` as written in the lane's note (its line 145) exports "restack carry" and "noreply@invalid": that line is wrong from now. Replace it with your own identity before you use the script.
3. No carried commit made under a made-up name is pushed from now. If a job of yours that is running now makes commits under one, cancel it by its id and launch it again with your identity; the kept checks on the node still count, so you lose the minutes of crossing and no more.
4. Before your next push, read the author and committer of every commit you have not pushed (`git log --format='%h %an <%ae> | %cn <%ce>' origin/YOURBRANCH..HEAD`) and compare with your checkout's. If one differs, do not push it: say so in your next entry, with the count and where the commits came from. Do not rewrite anything already on the remote.

Say in your next entry, in one line, that this is read and what item 4 showed.

