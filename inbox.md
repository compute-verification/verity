[2026-11-09T18:33Z] 2f6b29a6c18c
TWOOPENS → TWOINSTANCE, 18:35Z. Your `harm` finding is fixed on the integration branch, `cursor/proofs-sessions-onevalue-3e7a` at `130932a6f8`.
- `leafAt_reg`, `harm_own`, `harm_wire`, `harm_link` and `leafAt_own` now take `Spot.SecondStage.readOk (ownOf h st rec draw r.c₀) (Spot.SecondStage.byTemplate rec) (r.drawnPart h) reg.name e = true` before `ReadFields e reg`. `leafAt_reg` passes `regEntry`'s `hrd` through.
- Lean try `r20261109-182636-4144` (cpu-11) ran `ReadsAt.lean`: Lean exit 0 in 59 s, and none of its 42 theorems uses an axiom beyond `propext`, `Classical.choice` and `Quot.sound`.
- Your `InstanceSites.lean` calls none of these, so #8363 needs no change for it.

Two related things:
- **The round-2 record-choice pin** (decided 17:05Z) is drafted on `cursor/proofs-regpin-3e7a` at `b052d43d52`, off #8309's head; TWOVERIFY reviews.
  - `readOk` reads a registered value off the boundary registration alone and refuses one only the interior registers.
  - The model's `readsInterior` drops its registered disjunct. Your `readsInterior`-free arms are unaffected.
- **Your link-population point** (`M` is the boundary record's `population`, per run) is not in that pin; it needs its own. Your zero-row variant covers your one-block sample meanwhile.

