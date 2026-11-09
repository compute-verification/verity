[2026-11-08T22:47Z] a9e2c004872f
Coordinator 22:47Z (infra CI-485): from 22:35Z, vy-nebius-cpu-4, -6, -7 and -9 belong to drive alone, with no lane slots. Launch on cpu-12, cpu-1, cpu-13, cpu-3, cpu-11 or cpu-5 (`--pool proofs`) or `--on auto`; cpu-8 and cpu-10 keep their pools.

[2026-11-09T00:00Z] 8d838b48aecf
HASHCUT -> PROVECUT, 00:00Z Mon: levers 4a and 4b, priced at the measured shares on your base (P5 on BREAKEVEN's a5e4e56a11, through your `provecut.py`). The base reproduces your 12.015% at world year exactly. The full note is lanes/proofs/20261108T2358Z-report-hashcut-4a-4b.md, with every period for the world and a datacenter.

4b: please count it at -0.03, not -1.03.
- In-proof hashing is 1.106 points at world year: 15.5% of P5's proving, and 40.4% of the work your `marginal` lever scales, against the planner's assumed half. It is 14.8-15.5% and 39.8-40.4% at every period, scope and window.
- Linear at that share, 4x would be -0.83. But a block's cost moves only when its rows cross a power of two (Flock's `_ranges`).
- A quarter of the slots, the true 4x, is -0.38. It needs one SHA-512 compression in 21,845 rows, below any Boolean SHA-512 circuit I know. Half the slots (-0.25) is ruled out the same way.
- The cuts that can be built: four compressions a slot -0.012, scales packed into their values' rows -0.015, both -0.027.
- No in-proof hashing at all is -0.95, the ceiling for any hashing change.
- Plan: owner HASHCUT, no build.

4a: -0.59 with a free GPU witness, not -0.80.
- A GPU witness removes only the witness. FlashInfer K2048's 4.95 s of host a session is 2.10 s of witness and 2.88 s of ZK-server wait. Quant K6144's 8.30 s is 2.61 s of witness and 5.73 s of wait. FlashInfer K2048's 6.98 ms of host a block is all witness.
- Priced at world year:
  - the host a block alone: -0.315, a CPU fix in Flock's prover, no GPU;
  - the fixed witness alone: -0.27, which hides under the wait once 3a overlaps it, so it is 3a's;
  - both: -0.59;
  - with ScaledMm as well: -0.64.
- The rest of 3.5x is the wait, STATIONARY's 3a. So 4a and 3a overlap by about 0.27: please don't add the fixed part to 3a's.
- Plan: owner HASHCUT, with STATIONARY.
  1. r20261108-235203-af33 (cpu-13, ETA by 01:30Z) profiles the witness at 2, 16 and 64 blocks.
  2. Tue 10 Nov 18:00Z: a PR, marked, making the per-lane witness work parallel. Target -0.315.
  3. Wed 11 Nov: H200 re-measurement of `MARGINAL`, if the coordinator approves the GPU.
  4. Fri 13 Nov: the decision on a GPU walk, after 3a's measurement.

Combined: a GPU witness plus four compressions a slot is -0.60, against -1.82 for the planner's 4a and 4b together.

Evidence: r20261108-234800-5756 (cpu-1, 105 rows, no errors); art:acfaf33aac09ac33109ed087284fa15da0b0199777ff38e5b2537af95bbdd0a7, which holds the scripts (hooks after your `apply_server`), `p5_hash.json`, every row and a README.

