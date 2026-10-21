import NciCheck.Baseline
import Properties.Proofs.Nci.DescAgree
import Proofs.Nci.Cover.Training

/-!
# A verdict of compliance carries the cover theorem's factor

`CoverTrained` is `Nci.Cover.Guarantees.CoverTraining`'s conclusion on a verdict, at `Γ = gamma` and `α = num / den`, in
the shape of `Properties.Nci.Threads`'s statements. `coverTrained` proves it of every verdict of the gate-level check
`NciCheck.evaluate`, from the verdict's compliance (`Witnesses`) and `CoverTraining`.

The check of record is `NciCheck.Baseline.evaluate`: `Desc.evaluate`, θ and the owners by root node, then θ's provenance.
Its verdict is `Desc.evaluate`'s (`baseline_desc`), and `Properties.DescAgrees` makes that the gate-level check's on every
expansion of the instance, gate by gate. `CheckedCoverTraining` states the cover theorem's factor of every verdict of the
check of record, on every expansion, of which there is one. It is not yet a guarantee: `DescAgrees` is proved from two
open statements (`GatesLocated`, `NovelExact`, `Properties.Proofs.Nci.DescAgree`), and `checkedCoverTraining_of` proves
`CheckedCoverTraining` from them.
-/

namespace NciCheck.Properties.SecurityProofs

open Lean (Json)
open Flock.Program (Prog)

/-- When the verdict `out` on `j` has `complies` `true`, every declaration `d` the Program lays out with the baseline
`theta` names, containing a reference training circuit `c` with its baseline registered (`BaselineRegistered`), pays the
cover theorem's factor at `Γ = gamma` and `α = num / den` (`Nci.Cover.Guarantees.CoverTraining`'s conclusion, under its
premises on `c`'s bit view). -/
def CoverTrained (j out : Json) : Prop :=
  ∀ (i : Instance) (P : Prog) (d : Verity.Declaration),
    Parses j i → Flock.Program.decode i.program.toUTF8 = .ok P → LaysOut P d.D → Names i.theta d.θ →
    out.getObjVal? "complies" = .ok (.bool true) →
    let α : ℝ := (i.num : ℝ) / i.den
    ∀ (b α₀ : ℝ) (c : Verity.Declaration) (M : Verity.Containment c.D d.D)
      {J L : Type} [DecidableEq J] [DecidableEq L] (Mb : Nci.Cover.MulBlocks (Nci.Cover.Circuit.ofDecl c) J L)
      (A : J → Finset (Fin (Nci.Cover.Circuit.ofDecl c).N)) (s D w₀ : ℕ),
      0 < b → Nci.Assumptions.BaselineRegistered c d (Nci.PartialContainment.ofContainment M) →
      Nci.Cover.Assumptions.UpdateHardness Mb s b → Mb.Upstream A D → α * i.gamma < b * D →
      (∀ g, c.D.w g ≤ w₀) → w₀ * i.gamma + α * i.gamma + c.D.width c.θ ≤ s →
      α₀ * c.D.size c.D.computed ≤ Mb.P.card →
        α₀ * b * c.D.size c.D.computed ≤ (α * α * i.gamma + b * α) * d.D.size d.D.computed

/-- **A verdict of compliance of the gate-level check carries the cover theorem's factor.** -/
theorem coverTrained {j out : Json} (hev : evaluate j = .ok out) : CoverTrained j out :=
  fun i P d hpar hdec hL hθ hc b α₀ c M _ _ _ _ Mb A s D w₀ hb hB hH hU hk hw hs hα₀ =>
    Nci.Cover.SecurityProofs.CoverTraining i.gamma _ b α₀ c d M Mb A s D w₀ hb
      (div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _))
      (Witnesses j out hev i P d hpar hdec hL hθ hc) hB hH hU hk hw hs hα₀

/-- `Baseline.evaluate`'s verdict is `Desc.evaluate`'s: θ's provenance is checked after it, and the verdict returned
unchanged. -/
theorem baseline_desc {j tr out : Json} (h : Baseline.evaluate j tr = .ok out) : Desc.evaluate j = .ok out := by
  unfold Baseline.evaluate at h
  rcases h1 : Desc.evaluate j with e | o <;> simp only [h1, bind, Except.bind] at h
  · cases h
  rcases h2 : Baseline.provenance j tr (fun _ k => k) with e | u <;> simp only [h2, pure, Except.pure] at h
  · cases h
  · rw [h]

/-- On every verdict `out` of the check of record, `Baseline.evaluate`, the instance `j` has an expansion gate by gate,
and on each expansion `out` is `CoverTrained`. -/
def CheckedCoverTraining : Prop :=
  ∀ (j tr out : Json), Baseline.evaluate j tr = .ok out →
    (∃ j', Expands j j') ∧ ∀ j', Expands j j' → CoverTrained j' out

/-- **`CheckedCoverTraining` from `DescAgrees`.** -/
theorem checkedCoverTraining_of_descAgrees (h : Properties.DescAgrees) : CheckedCoverTraining :=
  fun j _ out hev =>
    have hd := h j out (baseline_desc hev)
    ⟨hd.1, fun j' hj' => coverTrained (hd.2 j' hj')⟩

/-- **A verdict of compliance of the check of record carries the cover theorem's factor, given `DescAgrees`'s open
statements.** -/
theorem checkedCoverTraining_of (hG : GatesLocated) (hN : NovelExact) : CheckedCoverTraining :=
  checkedCoverTraining_of_descAgrees (descAgrees_of hG hN)

end NciCheck.Properties.SecurityProofs
