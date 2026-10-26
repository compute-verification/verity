
## 26 Oct 01:25Z, from the coordinator: the per-call-noise tile is being built for the H200
Daniel (6:05 PM PDT, via top): kernels has started the sm_90a build of `PouwH1Tile_v5/v6` (per-call noise, J10-R) with KCOMMIT-FB: the recompile, the SASS gate and the exactness gates, on vy-dc-h200 (not the demo card). Whether M3's credit cites it is still Daniel's decision. Treat v5/v6 on the H200 as the near-term baseline row: get its SHA-512 compressions per credited MAC from the J10-R code (`benchmarks/hash/dither`, h1u), and say what coverage it reaches at γ ≤ 10. When kernels posts measured numbers, I'll relay them.

## 26 Oct 01:35Z, from the coordinator: URGENT, do the v5/v6 row first
Daniel (6:17 PM PDT, via top): the per-call-noise tile `PouwH1Tile_v5/v6` (J10-R) is Demo A's PoUW, on M2–M3's critical path (CMP-126). top: "BET-COVERAGE's (CMP-124) v5/v6 row is this leg's input, so do that row first." Before anything else, compute the v5/v6 row for the frozen shape on the H200, with the verifier's range covering every static INT8 matmul of the sampling block:
- coverage (share of the step's static MACs);
- γ, including SHA-512 compressions per credited MAC at 4.79 G/s;
- commit bytes per sampling block;
- the proof unit's size against C-Flock's limit.

Post the row as a comment on **both CMP-124 and CMP-126** as soon as you have it, even rough, with the assumptions stated, and aim for 03:00Z. compute-accounting builds the block's PoUW leg from it. Then continue with the rest of your table.
