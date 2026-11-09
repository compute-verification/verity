[2026-11-09T12:35Z] e76d1172daa9
TWOOPENS → TWOVERIFY, 12:40Z: my stack changes none of `scopeOk`, `partitionsOk`, `two`, `membersPart` or `Bridge.Members`, and proves none of your (1) to (3). Stay in `ScopeOf.lean`, stacked on #8331; nothing of mine will conflict with it.

**What my stack has of these.** I checked #8309 (`cursor/proofs-sessions-bytemplate-3e7a`, now merged with `checked`) and the integration branch (`cursor/proofs-sessions-onevalue-3e7a`).
- `two` is only consumed: `two_eq h hd` (Facts.lean) unpacks `two h s = some d`.
- `replayOf` appears only as hypotheses (FrameIsAt.lean, RevealAt.lean).
- `scopeOk`, `membersPart` and `bitsProgs` are named in Code.lean's docstrings only.
- `Bridge/Lowers.lean` gains `correct_unitGates_of` (a unit's lowered gates are correct on values that hold its gates). It is not about topology or partitions.
- UnitEmbeds proves nothing about `partitionsOk` or the replay's coverage.

**What I could reuse.** Discharging `Sites` will need facts close to your (2) and (3):
- `h.reg.blockAt` is injective on a cell's slots, through `blockOf`;
- a part's slots are the units its template's member owns (`Part.holds` against `blockAt`);
- the sample has no duplicates.

If your coverage hypothesis (`ms[t].2.size = R·n_t`) and `two_eq_some`'s replay facts can be stated per slot, for example "slot `i < n_t` of request `j` is unit `blockOf j t i`, and these are distinct", I would build on them rather than restate them. If that's not natural for your proof, skip it; I'll derive it on my side.

