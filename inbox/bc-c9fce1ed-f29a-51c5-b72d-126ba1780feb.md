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

## 01:09Z from proofs: DANIEL RULED the budget is 1% of the block's compute, so M3's draw sizing is UNBLOCKED

Daniel (6:05 PM PDT, via top): the budget is 1% of the block's compute. That point on PLANNER's 02:30Z frontier is the demo's plan of record: ANDs for work, 50/50 with leaked bits, one stratified `certify` bound. The Lean form of the joint bound is still with Daniel.
- PLANNER: mark the 1% point on PRF-166 as the plan of record, and give its draw: units per subcircuit, total GPU-s per block, and `free_compute` and `leak` at δ = 1% and 1e-6, with week, month and year pooled.
- DRAWPROVE: size M3's draw from that point once PLANNER posts it (about 02:30Z), then send the node ask to infra (PRF-171; sizing to infra by Mon 13:00Z).

## 01:45Z from proofs: C-Flock's size cap affects which units the frontier can use

DRAWPROVE measured (r20261026-004208-a28d, PRF-170) that C-Flock refuses any statement past k_log 27, and that limit is a soundness bound. RmsNormRow at the planner's size is k_log 28, so it can't be one proof unit.
- In the 02:30Z frontier, every unit must be at most k_log 27. Use the measured RMSNorm Definition on main, `RMSNormFusedCuda_v2` (N=4096: 21.5M ANDs, k_log 26, about 259 s to prove), or a two-part split. Two m30 halves measured 384 s at 42 threads and 107 GiB, with a 277 s Lean verify.
- Say on PRF-166 which units you used, so DRAWPROVE's `cpu_budget.py` can size M3's draw at the 1% point straight from your table.

## 01:56Z from proofs: DANIEL (6:36 PM PDT): PoUW's tiles are one more subcircuit in the draw

- PoUW's draw is no longer set by hand. Drop K′ = 22 at ε = 1/2.
- Add the PoUW tile units (PouwH1Tile_v5/v6, per-call noise) to the 1% frontier as one more subcircuit, with their ANDs in the work half, under the same objective and budget.
- Report the tile stratum's ε at δ (`eps_at`) from the same draw as PoUW's number, beside the joint `free_compute`/`leak`.
- If the v5/v6 sizes aren't measured yet, use the PouwH1Tile sizes on main and say so.
- Proving and serving overhead are SEPARATE numbers. The 1% is the proving budget. Commits (1.0 s of a ~6.35 s step at M = 16,384, HASH-20) and per-call noise go beside it, never folded into it.

## 02:20Z from proofs: for the post after 9a8f, from top

- One line for the demo's operating point: what M3's proving slots afford (DRAWPROVE's CPU sizing). Give proving cost as a multiple of the block's GPU time, and the certified ANDs and bits at δ = 1% and 1e-6. Beside it, the 1% point at a one-week sampling block, with its condition: 843 serving blocks (58.7 h) is the shortest that affords any plan.
- Add the PoUW tiles (the 01:56Z entry).
- NEW, owned by you, due Mon 20:00Z: a difftest holding `certify` to Lean's executable stratified bound, on the plan-of-record draw and a few random draws, in `verity_difftest`. Start it after the frontier is posted.

## 04:40Z from proofs: circuits mapped every subcircuit (use it in the next post)

Circuits (04:32Z, thread 1792970372.621909; FAMILIES' table on PRF-166, db928eb0): every subcircuit in Demo A's plan now maps to a circuits Definition that stages in today's C-Flock. Row ops drop from ~1.13M units a token to ~144k.
- The residual stream is 128 chunk rows of 256 B: Bf16AddRow_v2{128} x128. RMSNorm, the quantizer and the gathers read those rows.
- The O projection's input goes through AllGather_v1{2,64}.
When 9a8f ends, recompute the plan of record and the 1% week point on FAMILIES' table instead of your stubs, and say what changed. M3 itself is an 8-unit live draw now (PRF-171), so this table matters for the report's production point.

## 05:58Z from proofs: one change to FAMILIES (circuits 05:33Z)

RMSNorm's 128 chunk sums are committed 32 to a row: 4 rows of 128 B. Per RMSNorm that's 4 x RmsSumSqChunks_v1{4096,128} (k_log 25) + 128 x RmsNormOutSums_v1 (k_log 22). The summed block drops from ~17.7 to 0.67 Gbit at the same ANDs. Use this in the recompute.

## 06:45Z from proofs: warm-copy rules (PRF-187/188), read before seeding

1. NEVER seed from /workspace/research/src/<sha>: vy-store-evict deletes src trees no running process names, and RECL's seed vanished that way. Seeds live only under /workspace/research/lean-warm.
2. Do NOT make a new warm copy if you already have one on any node: reuse it (git checkout your new head inside it, then lake build).
3. If you truly need one on cpu-3, which is at 91% inodes: copy from an existing copy that's already built (yours, or ask in your note), not a cold seed. Delete it the moment your piece is delivered.
4. cpu-2 is at 84% and cpu-3 at 91%; the guards cancel at 95%. Never cpu-5.

## 07:18Z from proofs: cpu-3 is FULL (93% inodes, PRF-189); route runs to cpu-2 or cpu-1

- NEW runs go to vy-nebius-cpu-2 (slots p8-p15, 2.6M inodes free) or cpu-1. NOT cpu-3 until it drains, and never cpu-5.
- Every research run ships a source tree that holds inodes until it ends, so launch FEWER, bigger runs: batch your warm checks.
- If your warm copy is on cpu-3, keep using it if it's built. Don't make a new one there. Delete it the moment your piece is delivered.

## 07:35Z from proofs: hint on run 453e

Your sampled_proofs Lean audit failed with 'object file … SampledProofs.olean does not exist (a root that was not built?)'. Pass `--build` to check.py, or `lake build` first. Also, cpu-3 is full (93% inodes), so new runs go to cpu-2 or cpu-1.

## 07:38Z from proofs: the PoUW tile unit's size for the frontier (compute-accounting, CMP-126)

A v5 lane chunk (4×8 outputs over 131,072 MACs) is ~42.9M-44.0M ANDs (~329 ANDs/MAC), an ESTIMATE; the measured figure comes ~18:00Z. Output bits: 1 per statement (ok); serving commits ~257 bits per 131,072 MACs. Details: https://linear.app/verity-research/issue/CMP-126#comment-a02dc893
Use these for the PoUW tile subcircuit in the 1% plan, mark them as estimates, and update when the measurement lands.

## 07:47Z from proofs: the difftest target is now concrete (LEAN-56)

lean has STATED and PROVED `Flock.Guarantees.JointStratified` (#4560 statement, #4590 proofs, stacked on #4488): per sampling block, one budget Σ_s k_s·|B∩s|/n_s ≤ ln(1/δ), with harm in ANDs and in output bits each ≤ its exact supremum over that confidence set, both from one draw at one δ. File: Security/Proofs/Flock/EndToEnd/Joint/Statement.lean.
- Your certify-to-Lean difftest (due Mon 20:00Z) should hold Python's `certify` to that EXACT supremum: same budget, same harm weights.
- Cite JointStratified, not the per-template windows, as the plan of record's bound.

## 07:15Z from proofs
Until cursor/lean-records-reuse-key-061a lands, every `check.py`/`lean_changed.py --records` run passes `--rebuild`. A kept pass isn't keyed by package, so a pass of Security can be reused for Security/Properties (or the reverse), which runs nothing and writes no record (seen in r20261026-050354-87cc). If a records run since 05:00Z wrote no records for a package it should have covered, rerun it with --rebuild before marking.
