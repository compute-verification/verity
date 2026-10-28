import Properties.NciDesc.Cover
import Proofs.Properties.Nci.Agrees
import Proofs.Properties.Nci.DescVerdict
import Proofs.Nci.Cover.Training

/-!
# A verdict of compliance carries the cover theorem's factor

`CoverTrained` and `CheckedCoverTraining` are stated in `Properties.NciDesc.Cover`. `coverTrained` proves `CoverTrained`
of every verdict of the gate-level check `NciCheck.evaluate`, from the verdict's compliance (`Witnesses`) and
`CoverTraining`.

The check of record is `NciCheck.Baseline.evaluate`: `Desc.evaluate`, θ and the owners by root node, then θ's provenance.
Its verdict is `Desc.evaluate`'s (`baseline_desc`), and `Properties.DescAgrees` makes that the gate-level check's on every
expansion of the instance, gate by gate, so `checkedCoverTraining_of_descAgrees` proves `CheckedCoverTraining` from it,
and `SecurityProofs.CheckedCoverTraining` from `SecurityProofs.DescAgrees`.
-/

namespace NciCheck.Properties.SecurityProofs

open Lean (Json)

/-- **A verdict of compliance of the gate-level check carries the cover theorem's factor.** -/
theorem coverTrained {j out : Json} (hev : evaluate j = .ok out) : Properties.CoverTrained j out :=
  fun i P d hpar hdec hL hθ hc b α₀ c M _ _ _ _ Mb A s D w₀ hb hB hH hU hk hw hs hα₀ =>
    Nci.Cover.SecurityProofs.CoverTraining i.gamma _ b α₀ c d M Mb A s D w₀ hb
      (div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _))
      (Witnesses j out hev i P d hpar hdec hL hθ hc).1 hB hH hU hk hw hs hα₀

/-- `Baseline.evaluate`'s verdict is `Desc.evaluate`'s: θ's provenance is checked after it, and the verdict returned
unchanged. -/
theorem baseline_desc {j tr out : Json} (h : Baseline.evaluate j tr = .ok out) : Desc.evaluate j = .ok out := by
  unfold Baseline.evaluate at h
  rcases h1 : Desc.evaluate j with e | o <;> simp only [h1, bind, Except.bind] at h
  · cases h
  rcases h2 : Baseline.provenance j tr (fun _ k => k) with e | u <;> simp only [h2, pure, Except.pure] at h
  · cases h
  · rw [h]

/-- **`CheckedCoverTraining` from `DescAgrees`.** -/
theorem checkedCoverTraining_of_descAgrees (h : Properties.DescAgrees) : Properties.CheckedCoverTraining :=
  fun _ _ _ hev => onExpansions_of h (fun _ _ => coverTrained) (baseline_desc hev)

/-- **A verdict of compliance of the check of record carries the cover theorem's factor**, for every reference training
circuit the declaration contains, under the cover theorem's premises on its bit view and its baseline registered. -/
theorem CheckedCoverTraining : Properties.CheckedCoverTraining :=
  checkedCoverTraining_of_descAgrees DescAgrees

end NciCheck.Properties.SecurityProofs
