# Premises the lock lists by name

The Lean check reads premises through `let`, in conclusions and in packed hypotheses, and each of these has its record
in `Security/lean-audit.json`'s `assumptions`. Each is one of the following: one of `ASSUMPTIONS.md`'s assumptions at
the guarantee's own finders or inputs; an obligation on our code; or a definition the conclusion reads, which the check
lists as a premise because it sits left of an arrow in the conclusion. None adds a hash premise beyond `cr/sha-512` and
`ecr/sha-512`. Each docstring states its record.

| `Prop` | Claim id | Reduces to | Taken by |
|---|---|---|---|
| `ZkLink.LinkCRZC` | `ecr/sha-512` | `SHA512CRExpected` at each commit string's link finder (`Claims.ExpectedCR`) | `EndToEnd*`, `PublicAudit`, `Recursive*`, `flock_verify_sound` |
| `ZkLink.RowsCRZC` | `ecr/sha-512` | `SHA512CRExpected` at each registered read's rows finder (`Claims.ExpectedCR`) | `EndToEndRegisteredRows`, `EndToEndRegisteredRowsFlat` |
| `ZkLink.SessCRZ` (through `TableCRZ`) | `cr/sha-512` | `Finder.CR` at each of one table's knowledge finders (`Claims.StrictBirthday`) | as `LinkCRZC` |
| `StrictCR.Finder.CR` | `cr/sha-512` | `SHA512CRStrict` at the constant cost `q` (`Claims.StrictBirthday`); the budget isn't checked (above) | as `LinkCRZC`, and `qR`'s registered-path finder |
| `Recursion.StackedTopAt` | `registrant-holds-openings` | the registrant holds the strings `bcs'` it registered: the registered entry is `bcs'`'s, and `TopOf`; `TopColl.collision` reads the entry conjuncts at `bcs'`, which the verifier checks only at a transcript's own top | `RecursiveAuditC3` |
| `Recursion.VStarParsesScoped` | none: an obligation (gap 4) | fixes the bound `Vs`, which `vstarParsesScoped_of_list` builds at every inner shape and binding; weaker than `VStarParses` (`vstarParsesScoped_of`) | `PublicAudit`, `RecursiveAuditC3` |
| `ZkReg.OwnReadsHJ` | none | in the conclusion, it scopes `∀ rd, … → rd.Checked`; as `ZkOuterD.hown`, a fact about how `rd` is built from `own` (`ownReadsHJ_of_served`, `ownReads_cover`) | `EndToEndRegistered*` and `flock_verify_sound` (conclusion); `PublicAudit`, `Recursive*` (`hown`) |
| `FailClosed.FlockVerifyAcceptsR` | none: not an assumption | the acceptance event, from the verifier's own Lean functions (`AcceptsZKR`, `ScopeZR`); `verifyCmd`'s `IO` wrapper is read, as for `FlockVerifyAccepts` | `flock_verify_sound`, `flock_verify_sound_205`, `flock_verify_public_ties` (conclusion) |
| `Partitioning.CertifiesLeak` | none: not an assumption | the exfiltration target the bounds compare laws by, as `MeetsTarget` | `Partitioning.SecurityProofs.*`, `Witness.*` (conclusion) |
| `RecDeploy.Coined` | none: not an assumption | scopes a liveness witness to transcripts of coin bytes | `ZKStop_meets_ready` (conclusion) |
| `RecDeploy.PlainLeavesAdmitted` | none: a completeness fact, open | one inner statement passes the reader's checks and is in scope and true (measured, `r20261018-053343-99cf`; `Json.parse` is `partial`) | `ZKStop_meets_ready` (antecedent of its first conjunct) |
