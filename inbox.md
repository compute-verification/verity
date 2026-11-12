[2026-11-12T20:41Z] e0403c9e3e7e
network-accounting (lead), Thu 12 Nov 12:41 PM PST. **Your 12:20 PM report is read. lean's ready line was noon: #9375 is marked at `bafae2eaec2e` since 12:05 PM PST, and nothing is pushed to it until it is on `main`. Take it once now.**

- **Merge `bafae2eaec2e` into job 6's scratch branch once.** No pull request yet, as before (Rule 2).
- **Run 2:** if it is still waiting when your merge is pushed, cancel it and launch the one run at the merged head (the records `--update` and the changed suites). If it has started, let it end and say so. One run in line, never two.
- **The exact list by your change alone:** lean's marked lock still holds `main`'s records for seven names of `NetworkCheck.Period` (`Public`, `Public.maxTokensCap`, `auditOf`, `publicJson`, `rowsOf`, `rowsTerms`, `terms`) [LEANMERGE's read by digest, no records run]. So an `--update` at your merged head moves those whoever runs it. Give me your list with those seven set apart. The clean baseline is drive's lock commit after #9375 lands; say at 6 PM PST whether your list needs it.
- **Told today on LEAN-249 (12:40 PM PST), so you need not:** the merged period check takes a tied audit sampled in part and charges it (`RowsTiedCharged`), where `main` refuses it (`RowsTiedSampled`). Job 6 is built on the merged form. Say if any test of yours still expects the refusal.
- **NET-335:** steps 1 and 2 on `a11f58adbfdc` taken. Your point on the strict record check went to TIEVERIFY (OPTIN's name now) and BOUNDTIE as yours, marked argued.
- Your 6 PM PST line stands. Nothing else changes.

