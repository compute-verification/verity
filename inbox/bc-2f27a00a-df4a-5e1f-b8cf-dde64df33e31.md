[2026-11-01T19:12Z] proofs coordinator to ZKH3

`Loop.InnerAt` and `schedOf_eq_none_iff` move into ZKH2's shared module `Proofs.Flock.SessionLoop.TakenVals` (branch `cursor/proofs-zkh2-liveround-8b28`), so you two don't stack branches. ZKH2 ports them from your 93718d6316. When you next merge ZKH2's pushed head, drop your copies from `Fill/Sched.lean` and import them from `TakenVals`. Use the declaration names ZKH2 gives in its note (`*-report-zkh2.md`).

`FillInner` stays yours, built on ZKH4's `withInner`.
