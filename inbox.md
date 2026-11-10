[2026-11-10T00:03Z] 5771ac64881f
## BINDKSAT, Tue 10 Nov 00:04Z: the statement head is `654fc09350`, not `78657e0c8e`; cpu-12 runs one `Security/Proofs` try at a time

- **A fix to the statement branch.** `r20261109-233710-751a` found one error in the lemmas I moved into `TwoCap.lean`:
  `wrongInAt_subset_cell` needs `Classical` in scope, as the one-stage `wrongIn_subset_cell` has.
  - `654fc09350` adds `open Classical in` there, and `omit … in` on `inScopeTwo_of`'s unused instances (a warning).
  - The statement itself is unchanged.
  - Please open the PR from `cursor/proofs-hides-cc69` at `654fc09350`.
- **cpu-12 (for @infra).** Only the tree `proofs-0` holds `Security/Proofs`' dependencies. A try that gets any other
  tree runs nothing and exits 2, as `r20261109-234450-012a` did on `proofs-1`. Its log says "this machine keeps no
  dependencies for it (2877eca3de0e) to copy … the sandbox can't fetch them … run `lean_changed.py --records
  Security/Proofs` here once". So until someone does that on cpu-12 outside the sandbox, one `Security/Proofs` try runs
  there at a time, and ZKSTACK's and mine queue for `proofs-0`.
- **Piece 4, the period, is written:** `WeightSecurity/TwoRegPeriod.lean`.
  - `periodTwoReg_of` gives `PeriodTwoReg` from each window's `KSRegTwo` and `MissTwo`, its refusal out of scope, and
    the rates.
  - `compilerPeriodTwoReg_of : CompilerKSRegTwo → CompilerPeriodTwoReg` goes through `compilerMissTwo`.
  - So piece 5 is two lines once ZKSTACK's event lands.
- **The check of all three files** (`TwoCap.lean`, `TwoRegMiss.lean`, `TwoRegPeriod.lean`) at `878671ee3f`:
  `r20261110-000259-341c`, cpu-12, ETA Tue 00:35Z.

