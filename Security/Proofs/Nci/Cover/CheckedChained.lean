import Proofs.Nci.Cover.Checked
import Proofs.Nci.Cover.Chained

/-!
# A verdict of compliance carries the covering bound's factor on the chained multiplication blocks

`CoverTrainedChained` is `Nci.Cover.Guarantees.CoverTrainingChained`'s conclusion on a verdict, at `Γ = gamma` and
`α = num / den`, as `CoverTrained` (`Proofs.Nci.Cover.Checked`) is `CoverTraining`'s. `coverTrainedChained` proves it of
every verdict of the gate-level check `NciCheck.evaluate`, from the verdict's compliance (`Witnesses`), and
`CheckedCoverTrainingChained` states it of every verdict of the check of record, `NciCheck.Baseline.evaluate`, on every
expansion of the instance, and `checkedCoverTrainingChained` proves it from `SecurityProofs.DescAgrees`. It is stated
here in `Proofs/`, so it is no guarantee.
-/

namespace NciCheck.Properties.SecurityProofs

open Lean (Json)
open Flock.Program (Prog)

/-- When the verdict `out` on `j` has `complies` `true`, every declaration `d` the Program lays out with the θ
`theta` names, containing a reference training circuit `c` with its θ registered (`ThetaRegistered`), pays the
covering bound's two-cost factor on the multiplication blocks of `c`'s bits chained at `(D, c_a)`, at `Γ = gamma` and
`α = num / den` (`Nci.Cover.Guarantees.CoverTrainingChained`'s conclusion, under its premises on `bitDecl c`). -/
def CoverTrainedChained (j out : Json) : Prop :=
  ∀ (i : Instance) (P : Prog) (d : Verity.RegisteredCircuit),
    Parses j i → Flock.Program.decode i.program.toUTF8 = .ok P → LaysOut P d.D → Names i.theta d.θ →
    out.getObjVal? "complies" = .ok (.bool true) →
    let α : ℝ := (i.num : ℝ) / i.den
    ∀ (c : Verity.RegisteredCircuit) (M : Verity.Containment c.D d.D)
      (B : Nci.Covering.MulBlocks (Nci.Cover.bitDecl c)) (s D w₀ : ℕ) (cw ca : ℝ),
      Nci.Assumptions.ThetaRegistered c d (Nci.PartialContainment.ofContainment M) →
      Nci.Assumptions.ParameterHardAt (Nci.Cover.bitDecl c) B s cw → α * i.gamma < cw * D → 0 < ca →
      (∀ g, c.D.w g ≤ w₀) → w₀ * i.gamma + α * i.gamma + c.D.width c.θ ≤ s →
        cw * ca * (Nci.Assumptions.chainedBlocksAt (Nci.Cover.bitDecl c) B D ca).card ≤
            (α * α * i.gamma + cw * ca * α) * d.D.size d.D.computed ∧
          ((Nci.Assumptions.chainedBlocksAt (Nci.Cover.bitDecl c) B D ca).card : ℝ) / c.D.size c.D.computed *
              i.gamma / (α * i.gamma / cw * (α * i.gamma / ca) + α * i.gamma) ≤
            (d.D.size d.D.computed : ℝ) / c.D.size c.D.computed

/-- **A verdict of compliance of the gate-level check carries the covering bound's factor on the chained blocks.** -/
theorem coverTrainedChained {j out : Json} (hev : evaluate j = .ok out) : CoverTrainedChained j out :=
  fun i P d hpar hdec hL hθ hc c M B s D w₀ cw ca hB hH hk hca hw hs =>
    Nci.Cover.SecurityProofs.CoverTrainingChained i.gamma _ c d M B s D w₀ cw ca
      (div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _))
      (Witnesses j out hev i P d hpar hdec hL hθ hc).1 hB hH hk hca hw hs

/-- On every verdict `out` of the check of record, `Baseline.evaluate`, the instance `j` has an expansion gate by gate,
and on each expansion `out` is `CoverTrainedChained`. -/
def CheckedCoverTrainingChained : Prop :=
  ∀ (j tr out : Json), Baseline.evaluate j tr = .ok out →
    (∃ j', Expands j j') ∧ ∀ j', Expands j j' → CoverTrainedChained j' out

/-- **`CheckedCoverTrainingChained` from `DescAgrees`.** -/
theorem checkedCoverTrainingChained_of_descAgrees (h : Properties.DescAgrees) : CheckedCoverTrainingChained :=
  fun j _ out hev =>
    have hd := h j out (baseline_desc hev)
    ⟨hd.1, fun j' hj' => coverTrainedChained (hd.2 j' hj')⟩

/-- **A verdict of compliance of the check of record carries the covering bound's factor on the chained blocks.** -/
theorem checkedCoverTrainingChained : CheckedCoverTrainingChained :=
  checkedCoverTrainingChained_of_descAgrees DescAgrees

end NciCheck.Properties.SecurityProofs
