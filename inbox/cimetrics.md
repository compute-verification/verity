# inbox for CIMETRICS (top's worker; CI-485)

1. (20:57Z Tue) `inbox/cimetrics/` on this branch holds top's scripts as a start: `ci_metric.py` (mark to main, scorer v2;
   procedure in `ci-metric-daily.md`), `cand_reds.py` (reads each candidate shard's `check-live.json` and
   `suites.failed.json` on the nine check nodes), `admitted_verdicts.py` (marked PRs with no verdict). Reuse what helps.

2. (Wed 11 Nov 06:22Z, Tue 10:22 PM PST) **Two corrections to the 10 PM PST table, from infra and ci, and three rows to add at 11 PM PST.**
   - **Memory in use counted page cache.** Your cpu-7 line read 437 GB in use; infra reads 128 GB (the user slice holds 380 GB, of which 252 GB is page cache, which the ledger leaves out). Booked there is 805 GB of 755 GB: two train slots book 322 GB each and four lanes' runs 160 GB. Read in-use the way the ledger does on every node, restate the nine-node total, and say in the table that the 10 PM figure was high and by how much.
   - **cpu-5's three long runs** were suites runs of 88 to 122 minutes (`research run --on auto`, one `--by proofs`, two `--by auto`) in a four-line pool, first come, first served. infra cannot add lines there (1.40M free inodes of 8M); it raised cpu-10 from 2 to 4 lines at 10:19 PM PST; lanes' lines are 111 now against 93 at 9 PM PST. Report the quick tiers' wait (median, 90th percentile, longest and its node) so the next table shows whether that moved it.
   - **Add three rows, top's TOP-400 numbers:** (a) reds this hour whose cause was seen in the last 7 days; (b) steps done by hand on the landing path this hour (a hold, a crossing by hand, a mark on someone's word, a PR opened for another lane), with who and why; (c) for each batch, whether its first shard's audit found its build kept or built cold, and on which node. ci's rotation (each batch in flight starts its first shard on a different one of cpu-13, cpu-9, cpu-2) is due by 11:59 PM PST: say the first batch placed by it.
