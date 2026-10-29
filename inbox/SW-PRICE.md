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
