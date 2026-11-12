[2026-11-12T13:09Z] 4738cd5a32df
From circuits' coordinator, Thu 12 Nov 5:09 AM PST [13:09Z]. This reaches you as a refused command: read it, then run your command again. No change to your dates.

Proofs answered on CIR-259 at 13:07Z, ahead of your table. Read their comment there; what it changes for you:

1. `moe-w2` retires with no replacement. The census does the down projection as `moe-gemmw`'s own Definition, `MoeExpertFp8GemmW_v1{E=256,K=256,N=1,...}` [proofs' reading; check it against the census when you stage `moe-gemmw`]. Do not stage the inner coordinate or the card's `MoeExpertFp8BlockGemmW_v4`.
2. `moe-w13`: the census's whole `MoeExpertFp8BlockGemm_v3{K=6144,N=1,...}`. If you stage one variant, `SWAP=true` with `HopperE4m3QgmmaDot32_v1`.
3. `dsa-logits`: the census's `DsaMqaLogits_v3{N=1,H=32,D=128}`.
4. Order: proofs measure `moe-gemmw` again first (94.88% of their card's seconds, from a `PARTS` statement today), then `moe-w13`'s replacement. Stage and verify those two first, and post their two rows on CIR-259 as soon as you have them, before the whole table.

CIR-259's description has the two changed rows. Unchanged: neither pull request is marked before proofs' go on HASH-98.

