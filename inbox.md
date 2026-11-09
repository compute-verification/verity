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

