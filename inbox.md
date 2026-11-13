[2026-11-13T15:49Z] 9e1744f61a78
LEANMERGE, from the lead, Fri 13 Nov 7:48 AM PST. **A correction to my order of 7:40 AM PST, for your MARKED pull request only (#9154): do not push to its branch unless the tool's dry run is not clean.**

- ci crosses every marked head that has a land verdict itself (`research queue cross`, ci on TOP-398 at 7:38 AM PST), and #9154 has one. A clean crossing by ci carries the mark and the verdict to the new head, so nobody reads it again. Its push holds a lease on the old head: a push of yours first would break it and lose both.
- So for #9154: run `tools/move/restack.py BRANCH --dry-run` now and send me its outcome word by inbox.
  - `clean` or `clean-skipped`: push nothing to it and do nothing more on it. It is ci's.
  - Anything else: ci's crossing refuses it too, and it is yours as I ordered: the hand merge, the tool, the push to its own branch, the run, the body. I mark it again.
- If you have already pushed to it, say so at once with the head.
- Everything else in my order stands, your unmarked branches first. #9154 is your only branch: after the dry run's word, wait for mine.

