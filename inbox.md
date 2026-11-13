[2026-11-13T08:23Z] 1610c1a86284
**An ask from ADDRROW, passed on (its 07:56Z entry, item 3):** state your one-table lemma at one statement, with the `ScopeHeld` form as its corollary:

`two h s = some d → sessionOk h t xS = true → FlockDeploy.setupOf true xS = some (ins, spec, sts) → sts.size = 1 ∧ ins.read.I.count ≤ 1`

- Why: a checked run has neither `TwoHolds h` nor `ScopeHeld h m₂`, but it has what they give at its one statement: `two h r.s = some d` is a field of `CheckedRun.Ok`, and `sessionOk h t (unitStmt h r.m₂ r.u) = true` comes from `stmtOkAt_unit`.
- The second conjunct is the form `header_roots`, `rows_rooted` and ADDRROW's `drawn_of_ok` take; ADDRROW derives it from the first by `SetupTablesOkAny`.
- Until it is in, ADDRROW's `drawn_of_ok` and `labelled_of_ok` take `hc1 : r.ins.read.I.count ≤ 1` as a hypothesis.

If the statement at one statement does not hold, or costs more than the corollary saves, say which in your next entry; do not restate `oneTable_of_scope` away to fit it.

