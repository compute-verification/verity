# Premises the lock lists by name

The guarantees take these after a `let`. The Lean check reads premises through `let`, and each of these has its record
in `Security/lean-audit.json`'s `assumptions`. Each is one of `ASSUMPTIONS.md`'s assumptions at the guarantee's own
finders or inputs, or an obligation on our code. None adds a hash premise beyond `cr/sha-512` and `ecr/sha-512`. Each docstring
states its record.

| `Prop` | Claim id | Reduces to | Taken by |
|---|---|---|---|
| `ZkLink.LinkCRZC` | `ecr/sha-512` | `SHA512CRExpected` at each commit string's link finder (`Claims.ExpectedCR`) | `EndToEnd*`, `PublicAudit`, `Recursive*`, `flock_verify_sound` |
| `ZkLink.RowsCRZC` | `ecr/sha-512` | `SHA512CRExpected` at each registered read's rows finder (`Claims.ExpectedCR`) | `EndToEndRegisteredRows`, `EndToEndRegisteredRowsFlat` |
| `ZkLink.SessCRZ` (through `TableCRZ`) | `cr/sha-512` | `Finder.CR` at each of one table's knowledge finders (`Claims.StrictBirthday`) | as `LinkCRZC` |
| `StrictCR.Finder.CR` | `cr/sha-512` | `SHA512CRStrict` at the constant cost `q` (`Claims.StrictBirthday`); the budget isn't checked (above) | as `LinkCRZC`, and `qR`'s registered-path finder |
| `Recursion.StackedTopAt` | `registrant-holds-openings` | the registrant holds the strings `bcs'` it registered: the registered entry is `bcs'`'s, and `TopOf`; `TopColl.collision` reads the entry conjuncts at `bcs'`, which the verifier checks only at a transcript's own top | `RecursiveAuditC3` |
| `Recursion.VStarParsesScoped` | none: an obligation (gap 4) | fixes the bound `Vs`, which `vstarParsesScoped_of_list` builds at every inner shape and binding; weaker than `VStarParses` (`vstarParsesScoped_of`) | `PublicAudit`, `RecursiveAuditC3` |
