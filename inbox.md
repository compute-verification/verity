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

