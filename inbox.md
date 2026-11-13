[2026-11-13T01:21Z] 2bd94a90d42e
From the proofs coordinator: thank you, `ownReads_of` is taken as done at `90bbf657c8`. Two next pieces, in this order, both small and both without an owner:

1. **`RefReadsWf (heldOf …)`**, the one premise your lemma names. HELDOF (`bc-3ddfb587-549a-5964-9ddf-80547065f684`) is busy until 02:00Z on the cost of deciding the design's conditions, so it is yours. Its files are on `origin/cursor/proofs-heldof-4c7b` at `7db84a3bc0` (not yet compiled; its try `r20261113-002713-b5d8` on vy-nebius-cpu-12): `security/Proofs/Flock/SpotFlock/HeldOf.lean` (`heldOf i it u : Option Held`, `refStmt u.xs`), `HeldOfFiles.lean` (`refStmtsOf p bs`, each template's reference statement built from its member and the blobs with `memberRaw`) and `HeldOfCheck.lean`. State and prove: where `heldOf i it u = some h` and the reference statements are `refStmtsOf p bs`, `RefReadsWf h`, by `FlockComplete.JsonRT.parse_wf` where the reads come from `Flock.JsonParse.parse`. Branch from HELDOF's head onto a branch of your own; if HELDOF's files do not compile yet, write against their statements and say so. Write an entry naming the piece at once, so HELDOF does not write it too.
2. **`Apart` at the two-session device history** (HISTORY's need, its 00:39Z and 01:00Z entries): the condition that each session's ledger lines open its run only on its own window, proved at HISTORY's two-session deployment `dev₂` in `security/Proofs/Properties/Verity/SampledTwo/Device.lean` on `origin/cursor/proofs-history-d2e7` at `0c7bd6c2f4`. Read HISTORY's entries for the exact statement; say its size in your entry before you write it.

Machines: vy-nebius-cpu-12 as you have been using. Daniel ruled at 5:02 PM PST that Friday's run is a demo and where the parties run does not matter yet; nothing here changes. An entry by 02:30Z whose first line says what compiled.

[2026-11-13T01:45Z] c6c1bd095bb7
From the proofs coordinator, an update to my 01:21Z message (your two pieces stand):
- **`RefReadsWf` at `heldOf` is yours, and HELDOF is told not to write it.** HELDOF's three files now compile on `origin/cursor/proofs-heldof-4c7b` (its 01:34Z entry): `HeldOf.lean` (`r20261113-010607-9ddf` at `e310d9d623`), `HeldOfFiles.lean` (`9e74`), `HeldOfCheck.lean` (`8c63`). Branch from its head.
- **`Apart` at `dev₂`:** HISTORY's branch is at `b272f37a46bd` (I merged TWOLEG's compiled `Steps.lean` and CUT's fixed files; try `r20261113-012543-4d04` on vy-nebius-cpu-12). HISTORY is running again since 01:32Z and is told the piece is yours unless it writes an entry first saying it takes it. Read its latest entry before you start.
- Say in your entry the exact call FRAME's assembly (`AddressesGive.lean` on `cursor/proofs-frame-l1-9c4e`) makes to get `OwnReads h` at an accepted run once `RefReadsWf` is in.
- An entry by 02:30Z whose first line says what compiled.

