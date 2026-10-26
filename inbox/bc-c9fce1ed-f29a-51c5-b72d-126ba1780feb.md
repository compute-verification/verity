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

## 00:55Z from proofs: also post a circuit-size variant of the 50/50 rerun

Top (00:53Z, thread 1792970372.621909): Daniel is weighing whether to measure the work half by circuit size. Post that variant on PRF-166 beside the PoUW-work one, so his next answer doesn't cost a second run.
- harm_work = each unit's own ANDs (`Candidate.ands`, not its row openings), for every template. No PoUW work table and no influence.
- Use the same exchange rate (normalize each half by its best bound at budget B) and the same budgets as the PoUW-work variant.
- Cite `free_compute` and `leak` from the same draw, through `certify`.

## 01:00Z from proofs: DANIEL RULED that the work half is circuit size; this replaces the 00:55Z entry

Daniel (5:55 PM PDT, via top):
- harm_work = each unit's own ANDs (`Candidate.ands`, not its row openings), every subcircuit alike. No PoUW work table and no `influence`.
- The 50/50 split with leaked bits and top's exchange rate stay as they were.
- The ANDs plan is the plan of record. DROP the PoUW-work variant from the 02:30Z rerun: post only the ANDs frontier on PRF-166.
- Daniel wants to hear about any weirdness circuit size causes, for example a subcircuit whose ANDs badly misstate its GPU cost. Post each one on PRF-166 with numbers (ANDs against GPU-s), and tell me in your final message so I can ring top.

Words (Daniel):
- "Sampling block" means the proof units one draw covers, committed and registered before that draw. It replaces "window" in that sense.
- "Subcircuit" replaces "template" in prose. A subcircuit is a template with every parameter bound; each stratum is one subcircuit's units.
- The planner's "block kind" (`Candidate.kind`) now collides with "sampling block", so rename it in prose.
- Code identifiers and record keys (`template` fields, `verity.ml.windows`) stay as they are. Change only prose, docstrings and READMEs, in the PRs you're already touching (#4432, #4449).
