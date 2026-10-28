import Definitions.InferenceOnly.BitView
import Definitions.InferenceOnly.Assumptions.Baseline
import Definitions.InferenceOnly.Assumptions.Cover
import NciCheck.Baseline
import Properties.InferenceOnlyDesc.Verdict

/-!
# A verdict of compliance of the check of record carries the cover theorem's factor

`CoverTrained` is `InferenceOnly.Cover.Guarantees.CoverTraining`'s conclusion on a verdict, at the gate budget `c = gamma`
and the fresh-input budget `k = α c`, `α = num / den`, in the shape of `Properties.InferenceOnly.Threads`'s statements:
its premises on the reference training circuit's bit view (`UpdateHardness`, the upstream certificate
`MulBlocks.Upstream`) and on the registered information (`BaselineRegistered`) are named assumptions, which the check
does not see.

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

/-- When the verdict `out` on `j` has `complies` `true`, every declaration `d` the Program lays out with the registered
information `theta` names, containing a reference training circuit `r` with none of its fresh input registered
(`BaselineRegistered`), pays the cover theorem's factor at `c = gamma`, `α = num / den` and `k = α c`,
`α₀ b c size(r) ≤ (k² + b k) size(d)` (`InferenceOnly.Cover.Guarantees.CoverTraining`'s conclusion, under its premises on
`r`'s bit view). -/
def CoverTrained (j out : Json) : Prop :=
  ∀ (i : Instance) (P : Prog) (d : Verity.Declaration),
    Parses j i → Flock.Program.decode i.program.toUTF8 = .ok P → LaysOut P d.D → Names i.theta d.θ →
    out.getObjVal? "complies" = .ok (.bool true) →
    let c : ℕ := i.gamma
    let α : ℝ := (i.num : ℝ) / i.den
    let k : ℝ := α * c
    ∀ (b α₀ : ℝ) (r : Verity.Declaration) (M : Verity.Containment r.D d.D)
      {J L : Type} [DecidableEq J] [DecidableEq L]
      (Mb : InferenceOnly.Cover.MulBlocks (InferenceOnly.Cover.Circuit.ofDecl r) J L)
      (A : J → Finset (Fin (InferenceOnly.Cover.Circuit.ofDecl r).N)) (s D w₀ : ℕ),
      0 < b → InferenceOnly.Assumptions.BaselineRegistered r d (InferenceOnly.PartialContainment.ofContainment M) →
      InferenceOnly.Cover.Assumptions.UpdateHardness Mb s b → Mb.Upstream A D → k < b * D →
      (∀ g, r.D.w g ≤ w₀) → w₀ * c + k + r.D.width r.θ ≤ s →
      α₀ * r.D.size r.D.computed ≤ Mb.P.card →
        α₀ * b * c * r.D.size r.D.computed ≤ (k * k + b * k) * d.D.size d.D.computed

/-- **A verdict of compliance of the check of record carries the cover theorem's factor**, for every reference training
circuit the declaration contains, under the cover theorem's premises on its bit view and its baseline registered. On every
verdict of `NciCheck.Baseline.evaluate`, the verdict is `CoverTrained` on every expansion. -/
def CheckedCoverTraining : Prop :=
  ∀ (j tr out : Json), Baseline.evaluate j tr = .ok out → OnExpansions CoverTrained j out

end NciCheck.Properties
