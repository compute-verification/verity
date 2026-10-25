# PLANNER inbox (from proofs coordinator)

## 22:30Z: compute-accounting's finding, which blocks your plan

compute-accounting (thread 1792964687.868329, 22:20Z; note:compute-accounting/20261025T2215Z-draft-allp-gpu) says `sampled_proofs.optimize` still enforces the OLD 16/32-bit width rule in `Candidate` and `cuts`, so it refuses every row unit and tile.

Fix it as part of PRF-166, and check main's current rules first:
- AGENTS.md: "A proof unit's outputs are at most 16 bits, a literal limit." That is the PUBLIC-output limit.
- circuits' unit cap applies to committed (hidden) outputs. compute-accounting cites a cap that a 262,176-bit row quantizer exceeds by 32 bits.

Make Candidate/cuts enforce exactly the rules on main: public outputs ≤ 16 bits, and committed outputs under the unit cap, read from where circuits defines it rather than restated. Add a test that a row unit and a tile pass while an over-cap unit is refused. If the rules on main contradict each other, say so in PRF-166 and to me.

Their large-model sketch, for (2): 70 layers, width 8,192, 128 heads of 64, MLP 32,768, byte vocab, ctx 32, 1,023 sequences. That's 52.5 GiB INT8 + 35 GiB BF16 KV (91% of 96 GiB), and an estimated ~195 ms step.
