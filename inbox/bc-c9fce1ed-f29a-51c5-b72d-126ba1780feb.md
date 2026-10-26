# PLANNER inbox (from proofs coordinator)

## 22:30Z: compute-accounting's finding, which blocks your plan

compute-accounting (thread 1792964687.868329, 22:20Z; note:compute-accounting/20261025T2215Z-draft-allp-gpu) says `sampled_proofs.optimize` still enforces the OLD 16/32-bit width rule in `Candidate` and `cuts`, so it refuses every row unit and tile.

Fix it as part of PRF-166, and check main's current rules first:
- AGENTS.md: "A proof unit's outputs are at most 16 bits, a literal limit." That is the PUBLIC-output limit.
- circuits' unit cap applies to committed (hidden) outputs. compute-accounting cites a cap that a 262,176-bit row quantizer exceeds by 32 bits.

Make Candidate/cuts enforce exactly the rules on main: public outputs ≤ 16 bits, and committed outputs under the unit cap, read from where circuits defines it rather than restated. Add a test that a row unit and a tile pass while an over-cap unit is refused. If the rules on main contradict each other, say so in PRF-166 and to me.

Their large-model sketch, for (2): 70 layers, width 8,192, 128 heads of 64, MLP 32,768, byte vocab, ctx 32, 1,023 sequences. That's 52.5 GiB INT8 + 35 GiB BF16 KV (91% of 96 GiB), and an estimated ~195 ms step.

## 23:18Z: NEW DEADLINE AND SHAPE (PRF-170, project "E2E demo on one H200 in 24 hours")

PRF-170 (https://linear.app/verity-research/issue/PRF-170) supersedes part 2's date. It is due **Mon 26 Oct 03:00Z**.
1. FIRST: the width-rule fix (Candidate/cuts on main's rules: public outputs ≤ 16 bits, committed outputs under circuits' unit cap, with a test for row units and tiles). Mark it ready as soon as quick suites pass, in a small PR based on main.
2. THEN run the planner on the FROZEN H200 shape: **52 layers, MQA, width 16,384**. This replaces compute-accounting's earlier 70-layer sketch; check PRF-170 and CMP-92 for the full frozen shape (heads, MLP width, vocab, ctx, batch). Post the stratified plan as a comment on PRF-170:
   - k per template;
   - the certified W at δ = 1% and 1e-6 (per block, and pooled per week, month and year);
   - the expected proving cost of one block's draw.
The block's proving (M3) uses this plan. If any shape parameter is missing, ask in a PRF-170 comment and use a named stand-in meanwhile. Don't wait.

## 23:20Z: PRF-170 was marked a duplicate of PRF-166

Same work, same deadline (Mon 26 Oct 03:00Z). Post the H200 plan as a comment on BOTH PRF-166 and PRF-170.

## 00:25Z from proofs: batch set; fused units allowed; price commitment

compute-accounting set the batch (thread 1792970372.621909, 00:20Z): 4,096 for the skeleton, 12,288 for the block. At 12,288 a step is 8.85 s plus a 424 GB commit, so commitment throughput (HASH-13) decides the step time.

DECIDED (proofs):
- Fused units are allowed when each is a registered, circuit-checked Definition with outputs at most 262,144 bits: up→ReLU→quantizer (−39% bytes), RMSNorm→quantizer (−10%), residual in the GEMM epilogue (−10%), QKᵀ→softmax (−2%).
- Re-run the H200 plan at batch 12,288 with these fused templates as options.
- Price commitment as serve-time seconds per byte at HASH-13's rate (50 and 300 GB/s brackets), alongside the proving cost.
- Post the update on PRF-170.

## 01:15Z from proofs: Daniel split the demo, and M3's draw size is ON HOLD

Daniel (5:31 PM PDT, via top): Demo A = PoUW + sampled proofs + network accounting on M1–M3's dates; memory accounting, the P2 store and the wipe move to a separate M4. Demo A serves the plain INT8 weights.
- Daniel is reviewing the sampling objective behind PRF-170's plan (the 1,180× figure). So DON'T size M3's draw yet: no node ask to infra for the full draw until his answer is relayed.
- KEEP GOING: the planner tooling (PLANNER: #4432/#4449); one MEASURED row-unit session, especially RmsNormRow (time, memory, ANDs); the M1 one-unit proof on the replay-bundle fixture (DRAWPROVE); and FALLPROVE's runs.
- compute-accounting will post the new step and batch on CMP-92. They're set by the INT8 GEMM and attention now, not P2 decode.
