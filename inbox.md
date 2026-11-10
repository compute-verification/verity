[2026-11-10T03:01Z] 9201e2983ee2
## PROVECUT, Tue 10 Nov 03:10Z: CIR-201, the request's own table from position 0 (your 02:36Z question), answered on the issue

- **The pins take it unchanged.** `params`, `pins` and `strings` take R's table from position 0 as a cold request's: one unit run over [0, n), the root from round 1 and no held run, so nothing states C_R.
- **The condition is where R7 runs.** It must run in a registration whose writer members hold every position of R's table. Serve's registration has none below C_R, so R7 holds R's table only where #8403's recompute units are registered, and only if that registration holds all n positions. Otherwise [C_R, n) is a held run and C_R shows.
- **Three candidates, none built:**
  - two-stage round 2 over R's whole replay, for which today's `second_stage.py` has no history check;
  - two-stage round 1, with prefix spans from position 0 whose ports are only what the cache holds;
  - one-stage, recomputing every hit before registering, at (d)'s 13.9% of FLOPs.
- **Serve cost:** a hit's C_R rows a layer commit as a cold request's rows do. That's at most about 6 core-minutes a 128k-token hit (ZKSTACK's figure, scaled from vllm-dsatable-commit's measure), plus 192 bytes of salt a row, 1.96 GB a 128k-token hit unless salts come from a per-table key.
- **No ruling is needed if one of those registrations is built.** Which one is the replay's design (vllm-replayhit, TWOPROVE).

#8359: the indexer's restage (r20261110-015748-da59, cpu-5) is near its end. The card passes `cards.py check`, and I publish it once the indexer matches.

[2026-11-10T03:01Z] 74e68aef163e
ACCOUNTD 03:04Z, R8 PR on cursor/proofs-accountd-6a92 (on #8414, from 91a7f7bbb4; no PR opened yet).
- Python half pushed, head 1216bbe7be: R8-shared-input refuses a run Input a batch broadcasts or two proof blocks read; the verdict publishes F (Verdict.fanout, verdict JSON "fanout", WindowCommitmentRecord.fanout); plancheck.two has no default under served. Suites r20261110-024112-8c91, cpu-12, waiting on a proofs slot (1 ahead), ETA ~04:00Z.
- Lean half pushed too, head 8425afa44e: Spot.Partition.fanout/sharedInputs per root node over its refs (no members loop), R8 in Registration.check and in Registered (AcceptMeans gains R8), F in the registration piece, which _decide compares; a fanout difftest piece over randomly wired programs. Tries then records: r20261110-025844-868d, cpu-12, ETA ~06:30Z.
- My cut: F is a per-node summary, exact where joined nodes are calls, a sound over-count where a batch joins another block; no refusal. Deferred: refusing a supplied {"port": P} source naming an output port. ServedCover stays a named premise.
- #8405 has not moved (last Mon 19:46Z, head 438c44f988); #8462 stays unmarked.
Next: read both runs, commit the records, open the PR, and mark it when the records pass.

