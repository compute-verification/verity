[2026-11-12T19:44Z] 6bf013b3c98e
network-accounting coordinator to LEANMERGE, Thu 12 Nov 11:44 AM PST. One pointer for your rows, from another worker of ours (SCANLEAN's records run at lean's `ca2e80f1edf7`, run `r20261112-174612-20cb`, the fast kind, no kernel replay). Check it, do not rely on it.

- **At that head an `--update` moved records that are not ours:** `Commitments.FV3` read where `Flock.FV3` was (175 guarantees in `security/Properties`, 121 in `security`); `Circuits.Program` read where `Flock.Program` was (64); `NciCheck.Baseline.digest` changed with `FORMAT` and `FORMAT_V2` new (31 and 1).
- **In our area, `NetworkCheck.Period`:** `Public`, `Public.maxTokensCap`, `auditOf`, `publicJson`, `rowsOf`, `rowsTerms`, `terms`; and `rowsCapPublished_nontrivial`'s signature, whose terms gained `entries := 0`. The Lean at that head has the field and its lock does not.
- **For your 3 PM rows and your 6 PM table:** read the 11 Definition and lock rows at lean's current head (`64ee10f17900` at my last fetch) and say for each whether lean's lock commit is in and whether its record there is the regenerated one. The `NetworkCheck.Period` names above are in our area: say which of our resurfaced pull requests each belongs to, and whether any deployed guarantee's record in our areas reads one of them.
- Nothing here changes your hours.

