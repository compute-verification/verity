[2026-11-13T00:11Z] 10c6e5a900f7
BACKWARD2, Thu 12 Nov 4:11 PM PST.

**The second probe ended with rc 0, and exactly one case of the eight is non-finite: fma1. So the card's dP sum is FFMA, but each thread's FMUL is its second dim's product (dim 1), not its first (dim 0), so must 1 is right about FFMA and the Definition is wrong about which product is the FMUL.**

- RAN r20261113-000631-529d, rc 0 (lease on GPU 0, 15 s). Its last line: "gpu-lease: GPU 0 (GPU-7833ad2c) held 0m15s, busy 0m00s of 0m15s sampled (0%)". Two backward calls agree on every word in every case.
  - fma1: "nan_or_inf_words": {"dq": 3145728, "dk": 3145728, "dv": 0}. Every one is NaN, and all 49,152 dQ rows are non-finite.
  - fma0 and fma2 to fma7: 0 non-finite words in dQ, dK and dV.
- The reading this fits (INFERRED): `dot_do_o`'s first add, `do0*o0 + do1*o1`, is contracted on dim 0's product. That is FMUL of dim 1's product, then FFMA of dim 0's into it, then FFMA of dims 2 to 7 into the sum, presumably in order.
  - At normal magnitudes it rounds p0 + p1 once, exactly as "FMUL dim 0, then FFMA dim 1" does, since a bf16 product is exact in f32. So the 64 captured heads can't tell the two apart, and both heads fit every word.
  - It explains both probes. In the first probe's `ovf`, dim 1's 2^128 is the FMUL, so it overflows. In fma0, the 2^128 product is fused into -2^125 and stays finite. In fma1 it is the FMUL, so it overflows.
- Not settled by the probes: the order of dims 2 to 7 after the first two. Neither probe tells it. The 64 heads' every-word fit with dims 2 to 7 in order is the evidence for it (READ r20261112-212456-3c41). INFERRED: a different order would round some dP sums differently, so some dS words would likely move.
- Proposed fix, awaiting your ruling, nothing pushed: in `Fa2TrainBwdPre`, each thread's sum is `F32MulV2` on its dim 1, `F32FmaV2` of its dim 0 into that, then `F32FmaV2` of dims 2 to 7. The probe test changes to an overflow row: dim 0 -2^127 and dim 1 2^128 gives inf under the fix, and swapped it gives a finite 2^127. The twin's `dot_do_o` changes to match. Then CPU fits of both probes' heads at that head, so the probes' words are the record, and CONFORMANCE says the FFMA order is told by the probes' words.
- The fits of the first probe at both heads are still running. The table follows when they end.

