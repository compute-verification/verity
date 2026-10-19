import Models.Accounting.Assumptions.TS

/-!
# FILL-OPT for honest runs: a menu route's share meets FILL-OPT per share filled

FILL-OPT as `Accounting.Assumptions.TS.FillOpt` states it asks every run's filler for the menu's whole `χ*`, so it is
false at `χ* > 0` for a run that fills less of its window: an honest prover that aborts after a failed round, or one
whose filler is load dependent (RT-LOGIC-1, RT-LOGIC-2; note:20261019T0515Z-draft-rt-logic). Its per-share form,
`(1 − μ_x)·filled·χ*(π) ≤ π·n_fill` at every certificate (RT-LOGIC-1's fix), holds for every run whose filler runs its
`filled` share of one route of the filler menu, facet by facet (`fillFilled_of_route`): at `filled = 1` that is
`FillOpt` (`fillOpt_of_route`), and at `filled = 0`, an abort, every split run meets it (`fillFilled_zero`).

These are proofs-only witnesses about the premise's form, like `Accounting.SecurityProofs.Checkpoint.crossRead`, and
carry no lock record. They say nothing about a cheater's filler: that it costs at least the menu's `χ*` is FILL-OPT's
content, which `Accounting.Guarantees.TS.FillOptOfFanIn` gives at the fan-in menu's price. No design of record credits
a filler (#2699's 23c8696410 runs the H200 at `χ* = 0`), so no headline reads either form today.
-/

namespace Accounting.SecurityProofs.FillRoute

open Finset Accounting.TS

variable {K R T L U F X : Type} [Fintype K] [Fintype R]

/-- **A run that runs its `filled` share of a filler route meets FILL-OPT per share filled.** In a well-formed model,
if the run's filler loads every facet at least `filled ≥ 0` times route `ψ`'s demand, then at every certificate its
filler is priced at least `(1 − μ_x)·filled·χ*(π)`. -/
theorem fillFilled_of_route (Z : Model K R T L U F X) (P : Labelled K) (A : Anatomy K T L U) (filled : ℝ) (ψ : X)
    (hZ : Z.Wf P) (hf : 0 ≤ filled) (hload : ∀ r, filled * Z.fill ψ r ≤ Z.G.load A.nfill r) :
    ∀ lam : R → ℝ, IsCert lam → (1 - Z.muX) * filled * Z.chiStar lam ≤ Z.G.price lam A.nfill := by
  intro lam hlam
  obtain ⟨-, -, -, -, -, -, -, -, -, -, hfill, -, -, hmu, -⟩ := hZ
  have hd : ∀ ψ' : X, 0 ≤ demandPrice lam (Z.fill ψ') := fun ψ' =>
    sum_nonneg fun r _ => mul_nonneg (hlam.1 r) (hfill ψ' r)
  have : Nonempty X := ⟨ψ⟩
  have hchi : Z.chiStar lam ≤ demandPrice lam (Z.fill ψ) :=
    ciInf_le ⟨0, by rintro _ ⟨ψ', rfl⟩; exact hd ψ'⟩ ψ
  have hchi0 : 0 ≤ Z.chiStar lam := le_ciInf hd
  have hfc : 0 ≤ filled * Z.chiStar lam := mul_nonneg hf hchi0
  calc (1 - Z.muX) * filled * Z.chiStar lam ≤ filled * Z.chiStar lam := by nlinarith
    _ ≤ filled * demandPrice lam (Z.fill ψ) := mul_le_mul_of_nonneg_left hchi hf
    _ = ∑ r, lam r * (filled * Z.fill ψ r) := by
      unfold demandPrice
      rw [mul_sum]
      exact sum_congr rfl fun r _ => by ring
    _ ≤ ∑ r, lam r * Z.G.load A.nfill r := sum_le_sum fun r _ => mul_le_mul_of_nonneg_left (hload r) (hlam.1 r)
    _ = Z.G.price lam A.nfill := rfl

/-- **A run that runs one whole filler route meets `FillOpt`**: `fillFilled_of_route` at `filled = 1`. -/
theorem fillOpt_of_route [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X] (Z : Model K R T L U F X)
    (P : Labelled K) (A : Anatomy K T L U) (ψ : X) (hZ : Z.Wf P) (hload : ∀ r, Z.fill ψ r ≤ Z.G.load A.nfill r) :
    Accounting.Assumptions.TS.FillOpt Z A := by
  intro lam hlam
  simpa using fillFilled_of_route Z P A 1 ψ hZ zero_le_one (fun r => by simpa using hload r) lam hlam

/-- **An abort meets FILL-OPT per share filled**: at `filled = 0` every split run's filler is priced at least
`(1 − μ_x)·0·χ*(π) = 0`, whatever it ran. -/
theorem fillFilled_zero [Fintype T] [Fintype L] [Fintype U] (Z : Model K R T L U F X) (P : Labelled K)
    (A : Anatomy K T L U) (hZ : Z.Wf P) (hA : A.Split) :
    ∀ lam : R → ℝ, IsCert lam → (1 - Z.muX) * 0 * Z.chiStar lam ≤ Z.G.price lam A.nfill := by
  intro lam hlam
  rw [mul_zero, zero_mul]
  exact sum_nonneg fun r _ => mul_nonneg (hlam.1 r) (sum_nonneg fun k _ => mul_nonneg (hZ.1 r k) (hA.2.2.1 k))

end Accounting.SecurityProofs.FillRoute
