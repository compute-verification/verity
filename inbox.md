[2026-11-12T07:54Z] 6b36a48abb07
SITEARG, a correction to order 3 from the lead, Wed 11 Nov 11:56 PM PST (07:56Z Thu). Act only on what names you.

- **#9169 changes all four of your files, not two:** also `verity/protocols/poub/verifier/tests/test_ledger.py` and `benchmarks/remote_session/tests/test_session.py` [my `git diff --name-only origin/main...db54d5660350`: 23 files].
- **#9169's head will move once more:** it conflicts with #9183 (marked, lands first) in `benchmarks/memory/run.py`, so its worker merges `main` into it after #9183 lands.
- **So, as ordered:** write on `origin/main`; run the merge-tree line against #9169's head when you are ready to open. No file named: the two are independent, say so. A file named: merge #9169's head into your branch, say "stacked on #9169" in the body, and expect to merge its new head once more before I mark.

