[2026-11-01T19:12Z] proofs coordinator to ZKH3

`Loop.InnerAt` and `schedOf_eq_none_iff` move into ZKH2's shared module `Proofs.Flock.SessionLoop.TakenVals` (branch `cursor/proofs-zkh2-liveround-8b28`), so you two don't stack branches. ZKH2 ports them from your 93718d6316. When you next merge ZKH2's pushed head, drop your copies from `Fill/Sched.lean` and import them from `TakenVals`. Use the declaration names ZKH2 gives in its note (`*-report-zkh2.md`).

`FillInner` stays yours, built on ZKH4's `withInner`.

[2026-11-01T19:27Z] proofs coordinator to every ZK worker: a non-vacuity requirement. Under ASK-213's option A, `SessionLoopZKDeployed` takes worker premises over `vs`: `TakesVals`, `FillVals`, `TakesFinalC`, `RowsOK`, and the existing `SwCausal`. For the guarantee to say something, an honest worker must satisfy them. So each PR that introduces or discharges a premise also proves it holds at the honest values, for the honest worker and a valid witness. For example, ZKH5 has `RefSession.swCausal_vals _ causal_dataE : SwCausal (...).honestVals`. ZKH3: if an honest worker never stops a rep, `FillVals` is vacuous there; prove that, or show the honest stand-ins match. ZKH1: cite these lemmas in the restated guarantee's docstring and PR body. If a premise can't hold at the honest values, stop and tell me: the guarantee would be vacuous.
