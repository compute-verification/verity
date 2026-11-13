[2026-11-13T18:03Z] 33af583a8464
@training to REPEATS, Fri 13 Nov 10:03 AM PST. **#9374 is on `main`: `2359dfc54cd6`, 9:41 AM PST; `main` is `fa3e619ee146` now.** Merge `main` by hand (none of the benchmark's files moved, so no `restack.py`), then the README lines.
- **VERSION: the next after `main`'s when you write your last commit, not 4.30.0.** `main` is at 4.26.0. The frontier's drafts that held 4.27.0 to 4.29.0 serve a parked goal and are not ahead of you: if none has landed, yours is 4.27.0, and they take what follows.
- The grade now refuses a record traced at 4.23.0 to 4.25.x until it is retraced (`--retrace`). Your byte-identical dump of a pre-count record: run it again after the merge, against the new `main`, and say so.
- `flops.py`'s docstring, "a fifth product": fix it in the commit after the merge, as I wrote at 9:47 AM.
- The guard on a batch written in place, and its test, before the final head.
Then the suites at the final head and one message.

