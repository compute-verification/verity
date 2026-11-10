[2026-11-10T06:55Z] 99b87ff4ebe6
HELDINLINE, done. Note: note:lean/20261110T0650Z-finding-compile-tail (evidence art:c42dbcbcce59c33e7c2aa5cc6f2e326c3594a1697b7fdf90f430469fd28ef7ce); note:lean/20261110T0544Z-finding-time-outside-build corrected twice, LEAN-190 commented.
Corrected: cut 1 saves at most 12% of the sampled recompile (58 of 497 min in 22 h); cut 2 saves records 4.3/4.5 min of work and the check nothing; check on a one-edit train with all cuts, p95 16.0 (not 15.5).
Tail: no compile change moves that p95 by over 0.2 min (30 of the 37 chain files stay over 9 min under any import cut); with A (sampled recompile run on the seal beside the replay), B (proved-in fix, 10 s) and C (Hm2 defs split out, 3 files under) the p95 is 15.6, under 15 needs 40 s more from unprofiled rows 3/11/16. But the model counts commits: per landing, a one-PR train is p95 about 55 min (40 for one-file PRs).
Recommendation: write A now (it also covers every check now carrying the recompile, 17.8 to 15.9), and re-base goal 1g on landings, which only ONCE handing a train its PR records-run build can move (LEAN-236 to measure).
Loose ends: unlanded cut-1 patch at /tmp/around/lean246-cut1-unlanded.patch; PR 8493 still open with review-1 land.

