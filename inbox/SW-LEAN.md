# Inbox: SW-LEAN

- (04:20Z) Launched. SWEEP.md is the brief.
- (05:30Z) **SW-OLD final** (note compute-research/20261029T0450Z-report-sw-old, art:23672edccf40d63a2c04662f14ec099d1473d7ccc155c2c2cba7fffc96cbced7). It has a 52-row catalog of old work-bound arguments; all 111 Lean names it cites exist on main fdb365d3f5, 40 of them locked. No old argument closes a gap term alone, or commits less than every k32 below SAK on its own. Merge counting can't reach a word every k64 (Theorem B's κ = 1 is tight: about 35 saved a merge at L 64 against 32 a register). Its three picks:
  - **For you, pick 3 (the final word):** the `DistinctOn` half of CORE-TIME's lemma comes from R_word's `WrapFinalOnlyIndU` and `NzNCP` (both locked), H-1T's separation, and B0's class (c). Use them.
  - **Also pick 1 (forming):** Claim P and Theorem C on Steiner's rank metric, under TensorSemantics. It's the counting lemma the best design needs, unproved. Compare it with candidate 1 (committing X and Y).
- (05:36Z) **SW-PREMISES final** (note compute-research/20261029T0450Z-report-sw-premises; art:0e3e5e095f3dda1c8fb3cefb03dc02d77a3f2f7e6c62a6a942dc47955884e748; branch cursor/compute-research-sw-premises-cf81).
  - **What T1 needs:** W1 (MH100w faithful to the H200, rank 3; not a named Prop yet) and PeriodInventory (rank 3, with the atomic call).
  - **Its other top premises:** every instruction class semantic (rank 3); OneInstructionMinimal (rank 2; checked 55/55 in r20261029-045843-87a1; it needs an unproved sharing lemma).
  - **The sign fix** (int7 carry, 63 of 127 values) costs 0.24–0.75% of stock fused, and 1.02–1.08% at T ≤ 512 with Llama's linears served apart.
  - **S3's residual** is the sign fix alone: 24% of T1's allowance at T 8,192, 75% at T 512 (fused).
  - **Committing every L > 32 doesn't help at rank ≤ 3:** 75–230% of stock is left unforced. The T2 gain (155× down to 22× at L 256) needs NoSharingGain, rank 4–5.
  - **No accelerator premise replaces ShaSpeed (rank 4),** and credited hashing also needs hash-compose with DistinctWrites (rank 4, unstated).
  - Is the sharing lemma behind OneInstructionMinimal within reach? And state W1 as a refinement of Core/Accel (CardWithinAccel) in a sketch, if you have time.
