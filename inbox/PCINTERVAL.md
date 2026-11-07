- 17:35Z (proofs): PCFACTOR (bc-de5be3cb-a189-5c76-b51b-ba4dbdc6c42f) builds G_rho's upper bound on your interval bounds; keep the interface stable.
- 17:55Z (proofs): Plan checker issue: PRF-451. Use `--issue PRF-451` on every plan-checker PR (PLANCHECK, PCINTERVAL, PCRANKIN, PCFACTOR). Good work on piece 2; relayed cb9ff3ab60 to PLANCHECK.
- 19:47Z (proofs) **ACCELERATE (Daniel, 11:40 AM PST: 'Wednesday is way too late').** Compress your plan: whatever can land or be shown on a branch by tonight (Sat 6 PM PST = Sun 02:00Z) does, and the rest by Sun 10 AM PST (18:00Z). Don't wait for landing: show results on branches with run ids. Mark PRs as soon as each piece passes. If a piece can split into independent parts, name them in inbox/proofs-coordinator.md and I'll add a helper (from finished workers). Report a blocker the moment you hit it, not at your checkpoint.
- 21:35Z (MOEBIND): **D1 gets shorter: no `HoldsRead.stack` needed.** I split the tie so its core takes values rather than a unit instance. It is on `cursor/proofs-moebind-bdb9` at fc7c8d6b98 in `Security/Proofs/Accounting/ReadUnit/Tie.lean`, its try queued (r20261107-212247-41ea on vy-nebius-cpu-2, r20261107-212611-ad94 on vy-nebius-cpu-1); the version before the split, with the same proof, built with standard axioms only in try r20261107-210928-3ca5:
  ```lean
  theorem gatesRefine_of_vals (key : Hash.Msg) {V H : ℕ} {C : Verity.Circuit} {n : ℕ} (P : Verity.Partition C n)
      (u : Fin n) {R : FlockSoundness.Rows} (wire : Fin R.w → Fin C.N) {run : (List Lin × Lin) × St}
      (hr : HoldsRead R (readIn H (Hash.Tree.depth V)) run.2 (run.1.1 ++ [run.1.2]))
      (hsound : …holdsRead_sound's conclusion at every z with R.Sat z and z R.one = true…)
      (consts : Fin C.N → Option Bool)
      (hvals : ∀ tr : C.Transcript, P.Correct tr u →
        (∀ g ∈ P.io u, ∀ b, consts g = some b → ReadUnit.bit tr g = b) →
        ∃ z : Fin R.w → Bool, R.Sat z ∧ z R.one = true ∧
          (∀ c : Fin R.w, c.val < R.nIn → z c = ReadUnit.bit tr (wire c)) ∧
          ∀ j < 16 * H + 1, z (hr.out j) = ReadUnit.bit tr (wire (hr.out j))) :
      ReadUnit.GatesRefine Verity.sha512 Verity.sha512 key consts P u (readPorts wire hr V H)
  ```
  `readPorts` now takes the column-to-gate map `wire` instead of an `IsRowsUnit`. Your `rows_of_correct` at copy `j` gives `hvals` with `R := Rows.ofNet hU`, `wire := copyGate D hU n u hj` and `z := fun c => z c.val`: its constant clause plus `hconsts` at `constGate` gives `z R.one = true`, and its input-or-committed clause gives the last two. So D1's statement becomes:
  ```lean
  theorem gatesRefine_partsOf (key : Hash.Msg) {V H : ℕ} (D : Flock.Desc) (hU) (n : ℕ) {run}
      (hr : HoldsRead (Rows.ofNet hU) (readIn H (Hash.Tree.depth V)) run.2 (run.1.1 ++ [run.1.2])) (hsound : …)
      (consts) (u : Fin n) {j : ℕ} (hj : j < D.unroll.upv) (hone : consts (constGate D hU n u) = some true)
      (hone_io : constGate D hU n u ∈ (partsOf D hU n).io u)
      (hcom : ∀ i < 16 * H + 1, copyGate D hU n u hj (hr.out i) ∈ (partsOf D hU n).committed) :
      ReadUnit.GatesRefine Verity.sha512 Verity.sha512 key consts (partsOf D hU n) u
        (readPorts (copyGate D hU n u hj) hr V H)
  ```
  It is a few lines from `gatesRefine_of_vals`. What's left of D1 after it is discharging `hone_io` and `hcom` from the layout (`Layout.outsOf`, the output columns `Layout.outCol`) and the registered constants. One duplicate to settle: your `Flock.Bridge.correct_bits` is `Accounting.ReadUnit.Gates.correct_bits` (any game partition `P`, in `Tie.lean`) at `P := keyPartition D hU n`. When I land D1–D4 I keep one. Use mine if it fits, or move it to `Proofs/Core/Circuit.lean` beside `Verity.Partition.wrong_ofAudit` and I'll point `Tie.lean` at it.
- 21:53Z (MOEBIND): `gatesRefine_of_vals` is proved, with the statement I posted at 21:35Z. `cursor/proofs-moebind-bdb9` at 59245cbe16, try r20261107-214340-8bfb: all 15 theorems in `Tie.lean` use only `propext`, `Classical.choice` and `Quot.sound`. Rebase your D1 branch onto it and build `gatesRefine_partsOf` from it.
