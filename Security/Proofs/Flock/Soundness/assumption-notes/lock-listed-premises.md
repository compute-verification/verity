# Premises the lock lists by name

The guarantees take these after a `let`, or packed in `∃` at a hypothesis. The Lean check reads premises through both,
and each of these has its record in `Security/lean-audit.json`'s `assumptions`. Each is one of `ASSUMPTIONS.md`'s
assumptions at the guarantee's own finders or inputs, an obligation on our code, or a scope condition on the statement.
None adds a hash premise beyond `cr/sha-512` and `ecr/sha-512`. Each docstring states its record.

| `Prop` | Claim id | Reduces to | Taken by |
|---|---|---|---|
| `ZkLink.LinkCRZC` | `ecr/sha-512` | `SHA512CRExpected` at each commit string's link finder (`Claims.ExpectedCR`) | `EndToEnd*`, `PublicAudit`, `Recursive*`, `flock_verify_sound` |
| `ZkLink.RowsCRZC` | `ecr/sha-512` | `SHA512CRExpected` at each registered read's rows finder (`Claims.ExpectedCR`) | `EndToEndRegisteredRows`, `EndToEndRegisteredRowsFlat` |
| `ZkLink.SessCRZ` (through `TableCRZ`) | `cr/sha-512` | `Finder.CR` at each of one table's knowledge finders (`Claims.StrictBirthday`) | as `LinkCRZC` |
| `StrictCR.Finder.CR` | `cr/sha-512` | `SHA512CRStrict` at the constant cost `q` (`Claims.StrictBirthday`); the budget isn't checked (above) | as `LinkCRZC`, and `qR`'s registered-path finder |
| `Recursion.StackedTopAt` | `registrant-holds-openings` | `TopOf`, the prover's held tree and salts; its other two conjuncts are the verifier's checks (`stackedTop_of_innerCommits`) | `RecursiveAuditC3` |
| `Recursion.VStarParsesScoped` | none: an obligation (gap 4) | implied by `VStarParses` (`vstarParsesScoped_of`), a fact about our code | `PublicAudit`, `RecursiveAuditC3` |
| `RecDeploy.PartIn` | none: a scope condition, as `FlockDeploy.PlainIn` | evaluating `setupO` at each staged V\* part (1–200 tables, at most two link points, `25 ≤ m ≤ 35`, `InRange`); a scope refusal would discharge it | `RecDeploy.lamLin2_deployed_ge`, `lamLin2_deployed_ge_dep` (`hO`) |
