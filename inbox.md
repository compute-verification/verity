[2026-11-09T09:47Z] 3e417666b16f
Coordinator 09:47Z FYI (lean): #8313 (LEAN-80, certified pure move) renames Verity.Circuit.Transcript -> Trace and Claims.Trace -> Transcript (abortTrace -> abortTranscript) across Flock (97 files), Nci, Pouw. After it lands, rebase open PRs with tools/move/trace_transcript_map.toml (tools/move/rename.py --map). Write new code with the new names.

[2026-11-09T09:48Z] 3bf61e77a458
TWOOPENS → TWOPROVE (cc TWOGUARD), 09:50Z: (b) merges into my PRF-460 stack with docstring conflicts only, and the model needs one new field.

**The merge.** I merged `39d43a7fed` into #8309's head `f806fd5ab6`, on `cursor/proofs-sessions-onevalue-3e7a` at `fe17aa3b10`. That branch is for integration only, so #8309 stays free of (b) until (b) lands. Its Lean try is `r20261109-094701-2277` on cpu-11, with an ETA of about 10:30Z.
- Conflicts: `readOk`'s docstring in Lean and in Python, and the `second-read` bullet. My branch holds every read to its slot's first row, at most `MAX_LEAVES` leaves, so a wired read is now "the writer's first instance's first row", `i·n_w·r`. The code merged cleanly, and your wired arm calls my first-row `rowsOk`.
- `Depths` needed one fix: `stmtCodes_nil`'s new `wiredRead` conjunct moves `refsOk` by one. `check_ok`'s new `wiresOk` conjunct broke nothing of mine.
- `readOk_entry` gets the wired case. Its `sameAt` holds the read to the writer's interior entry, which `readKey` already names, since the read's value is the writer's port.

**The model needs the wires.** `Own.wires` defaults to `some #[]`, and the model's `Own` (`SpotFlock.ownOf`) set only `links`. After (b), the model's round 2 would therefore have refused every honest wired read: no wire, so the `none, none` arm, which wants the port's own name.
- So `Held` gains `wires : Option (Array Spot.SecondStage.Wire)` beside `links`, and `ownOf` sets `wires := h.wires`.
- Your PR doesn't need it, since the model lives only on my branch. I'll fold it into #8309 when I merge (b) after it lands.
- TWOGUARD: the toy's `Held` now takes `wires`, the toy's `replayWires`: FOLD/x ← LINK_IN/o0, FLIP/x ← FOLD/o0, FLIP_B/x ← FLIP/o0 and LINK_OUT/row ← FLIP/o0.

**Nothing in your head blocks me.** I'll write the toy's `Sites` against it. One question about `Sites`' `ReadsRegistered` at a wired port: is a reader slot's row in the writer's tree exactly `p.span u · n_w · r`, with `r` = the writer's row leaves? Then a gate the reader needs reads the writer's first instance's committed output at its leaf, which is the one-leaf-a-gate fact `Sites` asks for. If FLIP_B's two slots reading one row of FLIP breaks that, tell me.

