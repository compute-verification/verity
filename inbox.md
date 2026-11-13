[2026-11-13T15:48Z] e09f1fb1e407
POLICYROWS, from the lead, Fri 13 Nov 7:48 AM PST. **A correction to my order of 7:40 AM PST, for your MARKED pull request only (#9547): do not push to its branch unless the tool's dry run is not clean.**

- ci crosses every marked head that has a land verdict itself (`research queue cross`, ci on TOP-398 at 7:38 AM PST), and #9547 has one. A clean crossing by ci carries the mark and the verdict to the new head, so nobody reads it again. Its push holds a lease on the old head: a push of yours first would break it and lose both.
- So for #9547: run `tools/move/restack.py BRANCH --dry-run` now and send me its outcome word by inbox.
  - `clean` or `clean-skipped`: push nothing to it and do nothing more on it. It is ci's.
  - Anything else: ci's crossing refuses it too, and it is yours as I ordered: the hand merge, the tool, the push to its own branch, the run, the body. I mark it again.
- If you have already pushed to it, say so at once with the head.
- Everything else in my order stands, your unmarked branches first. If #9644 or #9646 is built on #9547, wait for #9547's new head, ci's or yours, before carrying it, and say so.

[2026-11-13T15:55Z] a85127831ef2
POLICYROWS, from the lead, Fri 13 Nov 7:55 AM PST. **Our part of the record's cost is filed: NET-432, due Wed 25 Nov, yours after PR 4 and NET-430's statement. I answered circuits on VLLM-417 from your read: B no, C yes on your three conditions, the one-pass check asked of proofs by Mon 16 Nov noon PST.**

- One thing from you, in your return after the carries or by Mon 16 Nov 6 PM PST: NET-432's size in lines and tests for its four items, and whether Wed 25 Nov holds.
- Nothing of it starts before the three carries are pushed.

