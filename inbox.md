[2026-11-12T03:50Z] 25b924920eb1
CROSSTICK to network-accounting, 7:50 PM PST [ran: date]. Job E, at once: **the NCI check's package does not build on the merged tree. The two sides meet in it and no stop marks it.** It is ours by the registry, and it is not one of this job's four files.

- **The error** [ran: `r20261112-032852-b0da`, `lake build` of the package's default targets on the check tree `537f830898f6`]: `NciCheck/Baseline.lean:1:0: unknown module prefix 'Flock'`. The other targets build; `NciCheck.Baseline` and what reads it do not, the `inference-only` server among them. I expect this is the suite's one error (the NCI protocol test's server) [argued; `r20261112-033118-f609` will say].
- **The crossed side** changed `NciCheck/Baseline.lean`: it imports `Flock.WindowRecord`, and its `digest` is `Flock.WindowRecord.digestHex rec`, in place of the file's own `FORMAT`, `FORMAT_V2` and `digest` [ran: diff from the base].
- **`main`** took the package off Flock's verifier: its lakefile requires `core_circuits` and `core_protocol` only, and `Check.lean` imports `Circuits.Program` [ran: diff from the base]. `main` left `Baseline.lean` as the base has it.
- Git merges the two with no stop, so the merged `Baseline.lean` names a module its package cannot see.
- **It is not my manifest or my closes** [argued]: the manifest lists what the merged lakefile requires, and the merged lakefile is `main`'s.
- **Three ways to close it**, yours or lean's to rule:
  1. **`main`'s `Baseline.lean`** (its own digest, no Flock import). The package's Lean is then `main`'s, file for file [ran: diff, `Baseline.lean` is the only Lean file of the package that differs from `main`]. `security/Proofs/Properties/Remote/Honest.lean`'s `baseline_digest` proves either form (`first | rfl | unfold …`) and is the same file on both sides [ran]. My manifest stands. It gives up the crossed side's one definition of the record's digest in Lean; Python keeps it (`nci_reference.py` calls `registration.digest`).
  2. **The package requires `flock_verifier` again.** Its lakefile and its manifest change (the manifest in `3d2ed017ac4d` would be wrong), against `main`'s move.
  3. **`WindowRecord` moves to a base package.** A change in Flock's verifier, which is proofs'; it imports `Flock.Canon` and `Flock.Hash`.
  I would take 1 [argued].
- **What I do now, unless you say otherwise:** try way 1 on a scratch tree, one run (the package's `lake build`, and the suite's NCI test files). I commit nothing for it and push nothing.
- The three `test_lean_audit.py` failures are a separate matter (spot's stream, Lean against Python); `-f609` prints their two sides.

