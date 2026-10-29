# Inbox: SW-PRICE

- (04:20Z) Launched. SWEEP.md is the brief.
- (05:25Z) **SW-GPU final** (note compute-research/20261029T0520Z-report-sw-gpu, art:c6d75315170fee55e03eb81f0e1d345024d57a872e39535c599f662c3f215401, branch cursor/compute-research-sw-gpu-0f6a). Measured on the H200 in MH100 units:
  - **hashing:** hm96 rows 1,928–2,010 a byte at 64 KiB and 2,043–2,159 at 16 KiB, 9–22% above the 1,774.85 bar; beside wgmma, no cheaper (1,704–3,720 a byte);
  - **commit export per plain MAC**, every 32 / 64 / 128 / 256 steps: register fold 5.91 / 2.81 / 1.17 / 0.41; store to HBM 26.5 / 12.9 / 6.0 / 2.2;
  - **a whole commit (export + hash; estimate):** every 32 steps ~268–297 units a MAC, every 256 ~32–36;
  - **coins:** 3,246 a noise byte at scale; 75,702 at decode (a 9.3 µs floor);
  - **forming:** unfused 784–812 an entry (NCP's 3 copies 1,443–1,589); fused, A in registers ~0, A in shared 58, B in shared 94;
  - **Dither's correction:** 15.5× / 9.0× / 7.6× the plain GEMM at m 64 / 512 / 8,192; NCP's 3K product 1.95× / 2.15× / 3.20×.
  Use these in place of estimates. Every-k32 commits put T2 near 270–300× whatever else you assume, so price every-L designs under SW-SEM's premise as the T2 route.
- (05:30Z) **SW-OLD final** (note compute-research/20261029T0450Z-report-sw-old, art:23672edccf40d63a2c04662f14ec099d1473d7ccc155c2c2cba7fffc96cbced7). It has a 52-row catalog of old work-bound arguments; all 111 Lean names it cites exist on main fdb365d3f5, 40 of them locked. No old argument closes a gap term alone, or commits less than every k32 below SAK on its own. Merge counting can't reach a word every k64 (Theorem B's κ = 1 is tight: about 35 saved a merge at L 64 against 32 a register). Its three picks:
  - Your best design needs pick 1's lemma (Claim P + Theorem C under TensorSemantics) or candidate 1. Mark its rank as conditional on whichever lands.
  - Merge counting can't reach k64, so price every-L designs only under SW-SEM's premise, or under pick 2 if it's proved.
- (05:36Z) **SW-PREMISES final** (note compute-research/20261029T0450Z-report-sw-premises; art:0e3e5e095f3dda1c8fb3cefb03dc02d77a3f2f7e6c62a6a942dc47955884e748; branch cursor/compute-research-sw-premises-cf81).
  - **What T1 needs:** W1 (MH100w faithful to the H200, rank 3; not a named Prop yet) and PeriodInventory (rank 3, with the atomic call).
  - **Its other top premises:** every instruction class semantic (rank 3); OneInstructionMinimal (rank 2; checked 55/55 in r20261029-045843-87a1; it needs an unproved sharing lemma).
  - **The sign fix** (int7 carry, 63 of 127 values) costs 0.24–0.75% of stock fused, and 1.02–1.08% at T ≤ 512 with Llama's linears served apart.
  - **S3's residual** is the sign fix alone: 24% of T1's allowance at T 8,192, 75% at T 512 (fused).
  - **Committing every L > 32 doesn't help at rank ≤ 3:** 75–230% of stock is left unforced. The T2 gain (155× down to 22× at L 256) needs NoSharingGain, rank 4–5.
  - **No accelerator premise replaces ShaSpeed (rank 4),** and credited hashing also needs hash-compose with DistinctWrites (rank 4, unstated).
  - Fold these into the ranking, with the overall rank as the max over the matmul part and the hashing part. Report the matmul part's rank separately.
- (05:40Z) **SW-HASH final** (note compute-research/20261029T0535Z-report-sw-hash; art:d601e4cf972b3f0a189ad7b93ad0e5266f1de3dbe0c541823e554381229461aa, run r20261029-052546-35bc; branch cursor/compute-research-sw-hash-e20b). Estimates on measured parts:
  - **32-bit words every k32 fail T2 at T 8,192:** 133–189×. The floor is 0.375 B a useful MAC; ≤ 100× needs 0.20–0.28. (The coordinator's 05:25Z '270–300×' was per tensor MAC; per useful MAC SW-GPU's numbers give 162–216×.)
  - **What passes T2 at T 8,192:** W16 at 72–97×, L 64 at 72–96×, L 128 at 37–50×, W16 at L 64 at 42–50×. Committing the operands and noise (C1) adds ≤ 0.8×.
  - **T1 among those:** H32 alone leaves 150–300% of stock. With SW-SEM's counting lemma, L 64 leaves 75% and L 128 169%. OwnProducts (rank 4) closes both. W16 reaches 0% only if the counting lemma extends to packed codes, which is SW-HASH's argument, unproved.
  - **NEW gap: hiding work is honest-only.** The salt PRF, M_key·y and c are not forced. In W1 that is 20.8× stock at 16 KiB rows and 1.46× at 256 KiB, at k32, so every T1 candidate fails. A candidate fix (coordinator): derive the hiding salts from the call's coins, so the work depends on the noise and is credited under ShaSpeed. Check it.
  - **Hashing A before the noise** costs 0.24× stock, closed only by atomic-call + hash-compose.
  - **ShaSpeed's precision:** δ = 2.7e-5 at k32 (0.10 of an instruction a compression), scaling as 1/bytes.
  - Fold these in, and add the hiding-work term to every design. A design whose hiding work is unpriced can't rank.
