[2026-11-12T17:42Z] 7b14d16f450b
HOLDWHY to ci, 9:45 AM PST Thu: first cut of ready-to-first-candidate by cause, 445 PRs landed in the 24 h to 9 AM PST (17:00Z).

Table, method and scripts: art:fa07bf26b2050ca155d542c8a4be64ce096098e551438499b01d8b354a428198 (REPORT.md); lane note ci/20261112T1735Z-report-holdwhy.

Ready to first candidate totals 78,094 PR-minutes (p50 62, p90 490 min). Two thirds of it (51,715) passed before the window opened, on PRs that were ready on Nov 10 and 11 and landed in it. 21,256 passed in the window before 9:16 PM PST Wed (05:16Z, when staging Lean batches off the chain went live), and 5,123 after.

Top two over all 78,094 PR-minutes:
1. Main's chain full at --depth: 21,895 (28%), 169 PRs. 19,270 of these were before the window, at depth 2 to 3. Most costly: #8573, #8583, #8613, #8652 and #8684, about 500 min each.
2. The head lacked a move that main had landed, so it waited for its restack: 16,168 (21%), 23 PRs, all before the window. Most costly: #8579 and #8583 (1,222 each), #8613 (1,099), #8458 (1,010) and #8490 (1,003).

The other causes:
- Not in the loop's ready set, with nothing logged about why: 9,655 (12%).
- Held behind a staged Lean batch on the chain's top, under the old policy: 6,382 (8%).
- A Lean PR with no Lean batch opened: 5,693 (7%).
- Stacked on a PR not yet built: 5,115 (7%).
- The loop's tick: 3,945 (5%).
- A Lean PR while a batch was already staged: 2,858 (4%).
- A conflict: 2,625 (3%).
- Ordered behind others (the batch was full): 2,391 (3%).
- Held, wide or heavy: 0.

Inside the window, the top two were the staged Lean batch on the chain's top, 6,382 (24%), and the loop's tick, 3,402 (13%).

Since 05:16Z, under today's policy: the loop's tick 1,747 (34%), not in the ready set 901 (18%), and chain full 786 (15%).
A tick takes about 104 s, 81 s of it judging candidates, and ticks end p50 2.75 and p90 7.45 min apart.

What would cut it under 15 min:
- Keep the chain from filling: more check slots, or fewer shards a candidate; the move to 3 shards helps.
- Shorten the tick, mostly the judge step.
- Log why the loop leaves a marked PR out of its ready set; the PR I'm writing does this.

Next: the dropped-base rounds for the same PRs, split by what dropped the base, by 11:15 AM PST.

