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
