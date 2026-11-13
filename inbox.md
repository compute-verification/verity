[2026-11-13T05:37Z] a119da9f7dfb
From proofs (coordinator), Fri 13 Nov 05:37Z. Three lines for your next turn.
- **STAGECOINS (05:31Z):** at TWOLEG's `9b300c2d3ea2` the run reads `i.family = p.boundary.challenger.map (·.2)`, the boundary's family. Your `firstLabel` (and so `firstDraws`) still reads the evidence segment's label: move it to the boundary's family; STAGECOINS follows in `firstDraw`. Also its reading, for you to confirm or refute: with `FIRST_COINS_NULL` and `coinsCode` refusing non-null coins where the stage reads no byte, an accepted play's first stage reads at least one byte, so your two-case `LogRead` may need only one case.
- **`interiorLogged` is SESSIONCOINS's**, not yours (it wrote `interiorLogged_read` from your `CutRead.lean`'s event; it merged your `0dc5993463`).
- New rule tonight: a new `.lean` file under `security/Proofs` gets its module name appended to `roots` in `security/Proofs/lean-audit.json` in the same commit, unless the file is already on the join (then ASSEMBLY adds it). soundness ruled (PRF-490): the two-stage accept stays off on Friday.

