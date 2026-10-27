# Premises the lock lists by name

The guarantees take these inside their statements: after a `let`, packed in `∃` or a field at a hypothesis, or named in
an Assumptions module (`Assumptions/Finders.lean`, `Assumptions/CustodyZk.lean`, `Claims/Recursion/Assumptions/`). The
Lean check reads premises through each, and each of these has its record in `Security/lean-audit.json`'s `assumptions`. Each is one of `ASSUMPTIONS.md`'s
assumptions at the guarantee's own finders or inputs, an obligation on our code, or a scope condition on the statement.
None adds a hash premise beyond `cr/sha-512` and `ecr/sha-512`. Each docstring states its record.

| `Prop` | Claim id | Reduces to | Taken by |
|---|---|---|---|
| `RecDeploy.PartIn` | none: a scope condition, as `FlockDeploy.PlainIn` | evaluating `setupO` at each staged V\* part (1–200 tables, at most two link points, `25 ≤ m ≤ 35`, `InRange`); a scope refusal would discharge it | `RecDeploy.lamLin2_deployed_ge`, `lamLin2_deployed_ge_dep` (`hO`) |
| `ZkLink.LinkCRZC` | `ecr/sha-512` | `SHA512CRExpected` at each commit string's link finder (`Claims.ExpectedCR`) | `EndToEnd*`, `PublicAudit`, `Recursive*`, `flock_verify_sound` |
| `ZkLink.RowsCRZC` | `ecr/sha-512` | `SHA512CRExpected` at each registered read's rows finder (`Claims.ExpectedCR`) | `EndToEndRegisteredRows`, `EndToEndRegisteredRowsFlat` |
| `ZkLink.SessCRZ` (through `TableCRZ`) | `cr/sha-512` | `Finder.CR` at each of one table's knowledge finders (`Claims.StrictBirthday`) | as `LinkCRZC` |
| `StrictCR.Finder.CR` | `cr/sha-512` | `SHA512CRStrict` at the constant cost `q` (`Claims.StrictBirthday`); the budget isn't checked (above) | as `LinkCRZC`, and `qR`'s registered-path finder |
| `Recursion.StackedTopAt` | `registrant-holds-openings` | the registrant holds the strings `bcs'` it registered: the registered entry is `bcs'`'s, and `TopOf`; `TopColl.collision` reads the entry conjuncts at `bcs'`, which the verifier checks only at a transcript's own top | `RecursiveAuditC3` |
| `Recursion.VStarParsesScoped` | none: an obligation (gap 4) | fixes the bound `Vs`, which `vstarParsesScoped_of_list` builds at every inner shape and binding; weaker than `VStarParses` (`vstarParsesScoped_of`) | `PublicAudit`, `RecursiveAuditC3` |
| `Assumptions.TableCRAt`, `TableCRZAt` | `cr/sha-512` | `Exec.TableCRL` (`SessCRL`), `ZkLink.TableCRZ` (`SessCRZ`): as `SessCRZ` | `flock_verify_sound`'s `hKS`, `hCR` |
| `Assumptions.LinkCRAt` | `ecr/sha-512` | `Exec.LinkCRL`: `SHA512CRExpected` at each commit string's link finder | `flock_verify_sound`'s `hCR` |
| `Assumptions.Zk.RecordsCustodyZKJ`, `Zk.DrawCustodyZKHJ` | `live-verifier` | `RecordCustodyZKHJLaw`'s two halves, `RecordsLiveZKJ` and `DrawOkZKHJ` at every draw | `Recursive*` (`ZkOuter.hRec`, `ZkOuterD.hDraw`, `ZkOuterR.hDraw`) |
| `Assumptions.Zk.DrawAtZKJ`, `DrawAtZKHJ` | `live-verifier` | `DrawOkZKJ`, `DrawOkZKHJ` at one draw; `drawOkZK(H)J_of_file` proves it from `DrawFileZK` | `setupZJ_of_acceptsZK`, `setupZHJ_of_acceptsZK` (`hok`) |
| `RecDeploy.PlainLeavesAdmitted` | none: a fact about one inner statement, which a recorded run measured | some inner statement passes the checked reader, fits Flock's completeness, and is in scope and true | `ZKStop_meets_ready` |
