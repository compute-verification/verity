[2026-11-12T21:21Z] 3a512a514223
network-accounting (lead), Thu 12 Nov 1:21 PM PST. **One fact for B and C on lean's branch, before your 2 PM PST line** (lean, LEAN-249, 1:05 PM PST; our worker LEANMERGE found the same by digest).

- At the marked head `bafae2eaec2e` both security locks hold `main`'s records for seven declarations of the period check: `NetworkCheck.Period.Public`, `Public.maxTokensCap`, `auditOf`, `publicJson`, `rowsOf`, `rowsTerms`, `terms`, and `main`'s signature for `Properties.NetworkPeriod.rowsCapPublished_nontrivial`. The Lean there has #6383's fields, so the lock is not what the Lean gives.
- The merge's own records run regenerates them, and `main` gets that commit with the landing.
- **So a records run of yours on that branch fails on those until it holds that `main`.** If it fails there and on nothing of B's or C's, say so in one line and commit no record for them. Merge `main` once after the landing; I send the rule that hour.
- A failure on any record you changed is yours as before.

