import Definitions.InferenceOnly.BitView
import Definitions.InferenceOnly.Assumptions.Baseline
import Definitions.InferenceOnly.Assumptions.Cover
import NciCheck.Baseline
import Properties.InferenceOnlyDesc.Verdict

/-!
# A verdict of compliance of the check of record carries the cover theorem's factor

`CoverTrained` is `InferenceOnly.Cover.Guarantees.CoverTraining`'s conclusion on a verdict, at `Γ = gamma` and `α = num / den`, in
the shape of `Properties.InferenceOnly.Threads`'s statements: its premises on the reference training circuit's bit view
(`UpdateHardness`, the upstream certificate `MulBlocks.Upstream`) and the baseline's (`BaselineRegistered`) are named
assumptions, which the check does not see.

The check of record is `NciCheck.Baseline.evaluate`: `Desc.evaluate`, θ and the owners by root node, then θ's
provenance, its verdict `Desc.evaluate`'s. `CheckedCoverTraining` states the cover theorem's factor of every verdict of
the check of record, on every expansion of the instance gate by gate (`OnExpansions`), of which there is one.

`SecurityProofs.CheckedCoverTraining` (`Proofs.InferenceOnly.Cover.Checked`, package `security_proofs`, whose proof needs
`InferenceOnly.Cover.SecurityProofs.CoverTraining`) proves it from `DescAgrees` (`SecurityProofs.DescAgrees`,
`Proofs.Properties.InferenceOnly.Agrees`).
-/

namespace NciCheck.Properties

open Lean (Json)
open Flock.Program (Prog)

/-- When the verdict `out` on `j` has `complies` `true`, every declaration `d` the Program lays out with the baseline
`theta` names, containing a reference training circuit `c` with its baseline registered (`BaselineRegistered`), pays the
cover theorem's factor at `Γ = gamma` and `α = num / den` (`InferenceOnly.Cover.Guarantees.CoverTraining`'s conclusion, under its
premises on `c`'s bit view). -/
def CoverTrained (j out : Json) : Prop :=
  ∀ (i : Instance) (P : Prog) (d : Verity.Declaration),
    Parses j i → Flock.Program.decode i.program.toUTF8 = .ok P → LaysOut P d.D → Names i.theta d.θ →
    out.getObjVal? "complies" = .ok (.bool true) →
    let α : ℝ := (i.num : ℝ) / i.den
    ∀ (b α₀ : ℝ) (c : Verity.Declaration) (M : Verity.Containment c.D d.D)
      {J L : Type} [DecidableEq J] [DecidableEq L] (Mb : InferenceOnly.Cover.MulBlocks (InferenceOnly.Cover.Circuit.ofDecl c) J L)
      (A : J → Finset (Fin (InferenceOnly.Cover.Circuit.ofDecl c).N)) (s D w₀ : ℕ),
      0 < b → InferenceOnly.Assumptions.BaselineRegistered c d (InferenceOnly.PartialContainment.ofContainment M) →
      InferenceOnly.Cover.Assumptions.UpdateHardness Mb s b → Mb.Upstream A D → α * i.gamma < b * D →
      (∀ g, c.D.w g ≤ w₀) → w₀ * i.gamma + α * i.gamma + c.D.width c.θ ≤ s →
      α₀ * c.D.size c.D.computed ≤ Mb.P.card →
        α₀ * b * c.D.size c.D.computed ≤ (α * α * i.gamma + b * α) * d.D.size d.D.computed

/-- **A verdict of compliance of the check of record carries the cover theorem's factor**, for every reference training
circuit the declaration contains, under the cover theorem's premises on its bit view and its baseline registered. On every
verdict of `NciCheck.Baseline.evaluate`, the verdict is `CoverTrained` on every expansion. -/
def CheckedCoverTraining : Prop :=
  ∀ (j tr out : Json), Baseline.evaluate j tr = .ok out → OnExpansions CoverTrained j out

end NciCheck.Properties
