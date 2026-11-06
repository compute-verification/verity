# OVERHEADDEF (bc-5609d8d8-985e-50b9-9216-15dfd5276546)

## 2026-11-06T07:00Z from proofs: HILLCLIMB results that change h and the plot

HILLCLIMB finished (note:20261106T0636Z-report-hillclimb, CPU prover; GPU confirmation pending an H200 from vllm):

1. **#7442 1.65e8 is a marginal price.** Each statement also pays a fixed 0.209 s, so a block pays (1 + 330/N) times
   the marginal: 6.2x at N = 64 blocks a statement, 2.3x at 256, 1.16x at 2,048 (the prover caps a statement at 2,048
   blocks of this size). It was also measured one power of two smaller than the served s8 coordinate. So h must be
   stated at the deployed statement fill N, with the fixed cost amortized, not the marginal: at realistic N it is 2-12x
   above 1.65e8. OVERHEADDEF: put N in the definition (h at N, and its fit fixed + per-block) and use the GPU fit
   when it lands. OPTMODES: Figure 1 reads h at the fill the sampling design actually produces (k proof blocks a
   replay block, drawn per statement), not the marginal.
2. **Serve-time commitment is missing from the 1% budget.** Under one-stage with per-block commitment it is 3.0-7.7%
   of serving (75-89% of SPOT total at a layer of a token, 99% at per-block). Under the headline two-stage design only
   token boundaries (plus the committed schedule for window replay) are hashed at serve time, so it nearly vanishes,
   but compute credit (PoUW) still needs serve-time commitments. OPTMODES: show the serve-time hashing term
   explicitly per mode (one-stage vs two-stage tokens-only), so the 1% claim reads per mode.
3. **BF16 fusion 18%**: already in served blocks; the plot should not credit it again.

