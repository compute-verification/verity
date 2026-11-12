module

public import Definitions.Nci.BitView
public import Definitions.Nci.Assumptions.Baseline
public import Definitions.Nci.Assumptions.Cover
public import NciCheck.Baseline
public import Properties.Nci.Desc.Verdict

/-!
# A verdict of compliance of the check of record carries the cover theorem's factor

`CoverTrained` is `Nci.Cover.Guarantees.CoverTraining`'s conclusion on a verdict, multiplied out by the gate budget
`c = gamma`, at the fresh-input budget `k = α c`, `α = num / den`, in the shape of `Properties.Nci.Threads`'s
statements: its premises on the reference training circuit's bit view (`UpdateHardness`, the upstream certificate
`MulBlocks.Upstream`) and on the registered information (`ThetaRegistered`) are named assumptions, which the check
does not see.

The check of record is `NciCheck.Baseline.evaluate`: `Desc.evaluate`, θ and the owners by root node, then θ's
provenance, its verdict `Desc.evaluate`'s. `CheckedCoverTraining` states the cover theorem's factor of every verdict of
the check of record, on every expansion of the instance gate by gate (`OnExpansions`), of which there is one.

`SecurityProofs.CheckedCoverTraining` (`Proofs.Nci.Cover.Checked`, package `security_proofs`, whose proof needs
`Nci.Cover.SecurityProofs.CoverTraining`) proves it from `DescAgrees` (`SecurityProofs.DescAgrees`,
`Proofs.Properties.Nci.Agrees`).
-/

@[expose] public section

namespace NciCheck.Properties

open Lean (Json)
open Flock.Program (Prog)

/-- When the verdict `out` on `j` has `complies` `true`, every declaration `d` the Program lays out with the registered
information `theta` names, containing a reference training circuit `r` with none of its fresh input registered
(`ThetaRegistered`), pays the cover theorem's factor at `c = gamma`, `α = num / den` and `k = α c`,
`α₀ b c size(r) ≤ (k² + b k) size(d)` (`c` times `Nci.Cover.Guarantees.CoverTraining`'s conclusion
`α₀ b size(r) ≤ (α² c + b α) size(d)`, under its premises on `r`'s bit view). -/
def CoverTrained (j out : Json) : Prop :=
  ∀ (i : Instance) (P : Prog) (d : Verity.RegisteredCircuit),
    Parses j i → Flock.Program.decode i.program.toUTF8 = .ok P → LaysOut P d.D → Names i.theta d.θ →
    out.getObjVal? "complies" = .ok (.bool true) →
    let c : ℕ := i.gamma
    let α : ℝ := (i.num : ℝ) / i.den
    let k : ℝ := α * c
    ∀ (b α₀ : ℝ) (r : Verity.RegisteredCircuit) (M : Verity.Containment r.D d.D)
      {J L : Type} [DecidableEq J] [DecidableEq L]
      (Mb : Nci.Cover.MulBlocks (Nci.Cover.Circuit.ofDecl r) J L)
      (A : J → Finset (Fin (Nci.Cover.Circuit.ofDecl r).N)) (s D w₀ : ℕ),
      0 < b → Nci.Assumptions.ThetaRegistered r d (Nci.PartialContainment.ofContainment M) →
      Nci.Cover.Assumptions.UpdateHardness Mb s b → Mb.Upstream A D → k < b * D →
      (∀ g, r.D.w g ≤ w₀) → w₀ * c + k + r.D.width r.θ ≤ s →
      α₀ * r.D.size r.D.computed ≤ Mb.P.card →
        α₀ * b * c * r.D.size r.D.computed ≤ (k * k + b * k) * d.D.size d.D.computed

/-- **A verdict of compliance of the check of record carries the cover theorem's factor**, for every reference training
circuit the declaration contains, under the cover theorem's premises on its bit view and its registered information free
of the reference circuit's fresh input. On every verdict of `NciCheck.Baseline.evaluate`, the verdict is `CoverTrained`
on every expansion. -/
def CheckedCoverTraining : Prop :=
  ∀ (j tr out : Json), Baseline.evaluate j tr = .ok out → OnExpansions CoverTrained j out

end NciCheck.Properties
