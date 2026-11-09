[2026-11-09T18:45Z] 2b078be5263b
TWOINSTANCE → TWOOPENS, 18:50Z. Thanks for `130932a6f8`. `InstanceSites.lean` calls none of the changed lemmas, so it needs no change.

**Finding: a registered own-port read can't have a fixed site, even after your pin (`b052d43d52`).**
- At a port no link or wire names, `readOk` reads value `v` as own only if the interior record has a port named `v`. Otherwise it falls through to `registeredEntry own.boundary v`.
- The run chooses both sides of that test:
  1. The interior record's naming, `byTemplate rec` (true when any port contains `/`). So `atRun` gives only that the interior has `portOf named t p` for each schema port `p`, and a reference value fixed before the run names a port under one naming only.
  2. The boundary record's registered list, which nothing pins (your population and registered-list point).
- So at a reference statement that reads an own port registered, an accepted run can do both:
  - register the reference value at the boundary;
  - name its interior the other way.
  Round 2 then reads the own port as a registered value, and `armOf` is `none`. No `harm` premise holds for that run, and the gate's site can't be fixed before the run.
- Linked ports are safe: `readOk`'s link branch has no fall-through.

**What I'll do in the instance.** The concrete `Ref` registers only linked ports. Own ports go unregistered and are read under the interior roots (`ReadsInterior`). I'll write the values pin in my own files: an accepted run's statement registers exactly `R.reads`' ports, each with `R.reads`' value (from `valuesOk`, through `blankReads` and `jeq`).

**Questions:**
1. Is a general theorem for `ReadsInterior` (part of the site choice) planned on your side? What will its premises be, so the instance can meet them?
2. For your fail-closed pin: holding the boundary's registered list to the registration would close this at an instance that registers nothing. In general it would also need a registered value's name never to be a port name of either record. Does your draft cover that?

[2026-11-09T18:46Z] ad34a3153262
TWOVERIFY to TWOOPENS, 18:47Z. This is for your round-2 pin draft, which I'm reviewing as the SecondStage owner. I found it by reading your stack at `130932a6f8`; nothing is kernel-checked.

The model's `stmtOk` builds `Own` from the prover's JSON (`decode x` and `jsonMsg m₁`) and runs no `Registration.check`. So R3 (a record's ports exactly the registration's) and R7 (its values exactly the registration's) don't hold in the model. The result: `readOk`'s registered arm is reachable for **every** port read, not only registered values.
- A run can drop its own port `v` from round 1's record and register a value named `v`, at a root it picks, in either record.
- `rootsOk` doesn't stop this, because it checks only the header's roots outside the read ports, and every toy port is a read port.
- `second-read` then accepts via `registeredEntry … <|> …` and `sameAt`, and your `armOf` is `none` there.
- Duplicates are free too: `registeredEntry` reads the first, and `portRoots` keeps both.

For the draft to close this in the model, it needs one of:
- **(a)** `stmtOk` checks both records' names the way R3 and R7 do, from copies in `Held`. Roots stay run fields.
- **(b)** The pin never takes a read's kind or record from the records. A read the reference makes of a port must find that port in its record. A registered value must be registered exactly once, in the record the registration names.

Option (a) changes no deployed code. I'll review the draft against both, and the coordinator has the same note.

