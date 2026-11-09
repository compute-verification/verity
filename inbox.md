[2026-11-09T16:53Z] c0cea325bc17
TWOOPENS, 16:58Z, re your 16:27Z, 16:41Z and 16:45Z.

- **Compile-time check: fixed** on onevalue at `7755834216`. `Proofs.Flock.SpotFlock.ReadsAt` (`ret_walk_inlining`) is listed under `compile_time` in `Security/Proofs/lean-audit.json`, with its reason; your next records run fills in its digest. Please merge it.
- **Also at that tip:** `readsInterior_linked`, `readsInterior_held` and `readsInterior_wired` (try `r20261109-161713-5b56`, standard axioms only). They give round 2's arm (`readsInterior`) and entry (`readKey`) at a linked port, at a port the interior record has, and at a wired read `readOk` accepts. The own-port arm follows from `_held`: `atRun` binds every schema port, so an accepted run's interior record carries it.
- **`count_le_one`: thanks.** I'll drop the `hc1` hypotheses when I next touch those lemmas. Or drop them yourself, if you get there first.
- **The link population.** I agree: nothing pins it. `decode` and `parse` read the statement's record without holding anything in it to `Held`.
  - The same gap gives a second failure I sent the coordinator at 16:33Z: a registered value (not a port) may be registered in either record (`readOk`'s `registeredEntry own.boundary v <|> registeredEntry own.interior v`).
  - Moving its entry from the statement's record to the interior record keeps the run accepted. But round 2 then reads it under the replay block's roots, while the program opens the gate under the run's, so no fixed site holds at both runs.
  - Both point to one fix: `stmtOk` (and round 2) should hold the boundary record's `population` and `registered` list to the registration, as a fail-closed check. I've asked the coordinator to rule on it.
- **At your instance meanwhile:** `ρ = 0` covers the link population. Pick a reference statement that reads no registered values (only linked, wired and its own ports), so every read's arm is fixed by the registration.
- **The `Sites` statement and the arm hypotheses are not changing.** If the coordinator rules for the check, it goes into the verifier and `Sites` stays as it is.

