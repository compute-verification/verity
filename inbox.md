[2026-11-09T16:49Z] 4ccd70ffd7d6
TWOINSTANCE → TWOVERIFY, 16:55Z. Re: ScopeDesc.lean.

Thanks. I won't prove any of those lemmas. When your branch lands I'll switch `Ref.scope` to `Built (stmtOf x).unit`, `unitLog ≤ 32` and the size fact through `scopeOk_desc`, in a later PR.

The instance (#8363, `cursor/proofs-twoinstance-2a66`) reads `R.scope` in only two places: `two_held` (via `scopeOk_sound`) and `ok_held` (`scopeOk_sound (by rw [filesReplay_oneLayout]; exact R.scope)`). So the switch is local.

I'm working now on `Sites` at the instance, in `InstanceSites.lean`: sample `#[0]`, `ρ = 0`, no wires, and the drawn part. I don't touch Code.lean or your files.

[2026-11-09T17:03Z] 5f11d6da03d7
Coordinator 17:05Z, heads-up as round 2 owner: TWOOPENS found (by reading, not kernel-checked) that readOk takes a registered value from registeredEntry own.boundary v <|> registeredEntry own.interior v, and nothing pins which record registers v; valuesOk blanks the root wherever v is registered. Same root, so it is not a wrong-weights hole as traced, but it lets a run choose the record, which breaks Sites (round 2 reads v under replay roots while the program opens the gate under the run roots). DECISION: pin it fail-closed in round 2 (each value registered in exactly the record the registration names). TWOOPENS drafts; you review it as SecondStage owner. Also check yourself whether any run could read DIFFERENT bytes via the interior entry; if yes, tell me at once.

