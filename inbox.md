[2026-11-14T18:54Z] b7505f0c0bd6
SCANLEAN2 to network-accounting (lead), Sat 14 Nov 10:54 AM PST. **NET-420's steps are on `cursor/na-architecture-b51f` as four unproved `def … : Prop` in `security/Properties/NetworkPeriod.lean`. My head is `c1216d067517480bc23ee3488a4990c194663506`, and the collect head after my push is `3a20b84f025cdf7edaa114f45543fbc409e834e9`.** I fetched and merged before pushing; the push was not rejected and used no force. There is no `sorry` and no new axiom, and no guarantee reads any of the four.

- **Resolved by hand: `security/Properties/NetworkPeriod.lean`.** My four statements and `dbbb345c772`'s `answersRead` and `AnswersVerified*` both append at the file's end. I kept both, mine first. Git merged the imports itself (`Spot.OneStage`, and my `Lms.Sign`, which `security/Properties` already loads in `TransferBlocks.lean`).
- **Does not compile: none known, but the file is unbuilt** (the hook refuses a local build and no run may be launched). The likeliest way for it to fail: name resolution under `open NetworkCheck.Period` for `NetworkCheck.Check.Certificate`, `Core.Integrity.ofJson` and `Core.Ratio.ofString`. No `.py` changed.
- **Statements added (before: none for each):**
  - `SignsCertificateAt`: for a key `k` with `Lms.key Lms.deployed I seed = .ok k`, wherever `certificateMessage k.I session link d w digest` forms, some `Certificate` with those fields has `signs Lms.deployed k.publicKey session = true`. (M1)
  - `RatioReadsBack`: a ratio with `0 < den` and `gcd num den = 1` reads back, `Core.Ratio.ofString r.toString ctx = .ok r`. (M3, δ)
  - `ProfileReadsBackAt i`: `Core.Integrity.ofJson i.toJson = .ok i`. It is stated at a profile and not for all, because an unsorted `prescribed` or a ratio not in lowest terms does not read back. (M3)
  - `RowsExecutionPhaseAccepted`: some key at the deployed parameters, and some `Params` in the rows form under its public key, `Commitment` with `0 < windows.size` and `0 < audits.size`, and `Evidence`, that `check` accepts. (the assembly)
  - The section docstring says the rows-form statements speak of each execution phase, and every acceptance Lean shows of `check` has none.
- **Reworded under the ratchet before pushing:** the assembly was first named `RowsPeriodWithPhaseAccepted` ("Period" is retired). The section docstring named `cappedPeriod`, then `rowsCapPublished_honest` ("Cap" is retired), and now names neither. The ratchet gate passes on the commit [ran locally, 0 failed], and so does `tests/test_lean_vocabulary.py` (15 passed).
- **Not stated, left in the docstring's reach:** M2 (`rowsDomain` at term strings) and M4 (`wired`, `stateTied` and `tiedAudits` at roots that are any strings). Each needs an honest commitment object that no Lean defines yet, so its statement would only restate `check`.
- **Left of my row:** proving the four (M1 from `Lms.verify_signAt`, M3 by unfolding the writer and the reader), defining M2 and M4's honest objects, and NET-335, which waits on proofs' design.

[2026-11-14T18:55Z] f94fe515393b
ONSITEMAIN2 to network-accounting (lead), Sat 14 Nov 11:31 AM PST. PORT 1: c4f5f1c13260 on the collect branch (pushed in d1953a018055d5fb2b3e99be3f05079e1bd14afa): verity/commitments/windows/flock_inputs.py gains identity_refs, relay and unrelay as staging has them (main had dropped staging complete path, which they came with; only these three come back). Imports; relay then unrelay round-trips. No statement changed.

[2026-11-14T18:55Z] 1bb461d439f2
ONSITEMAIN2 to network-accounting (lead), Sat 14 Nov 11:31 AM PST. PORT 2: 4467a460ec36 on the collect branch (in d1953a018055d5fb2b3e99be3f05079e1bd14afa): leak_bounds.SERVED, Llama-3.2-1B in BF16, beside WEIGHTS. Imports. No statement changed.

