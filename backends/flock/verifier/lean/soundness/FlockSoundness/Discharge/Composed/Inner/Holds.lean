import FlockSoundness.Discharge.Composed.Defs

/-!
# `innerOf_holds`: the honest prover's pads satisfy the inner proof's constraint at the draw
-/

namespace FlockSoundness.Discharge.Composed

open Game Audit Exec Refine Flock Model ZkSession ZkLink ZkExec DecodeZL FlockLevel3 ZkView

variable {mPts n : ℕ} {Reg : Type}

/-- **The inner relation at the draw** (open, inner lane): at every coin vector the honest prover doesn't refuse, every
valid witness whose registered values are the public ones satisfies `Table.InnerHolds` (§2.4 line 16's ten
constraints). -/
theorem innerOf_holds (I : Inputs) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) (S : Finset (Fin n)) (R : Reg) (y : DrawSetupZK I mPts S (dj S)) :
    let C := coinedOf (tablesAtZK (Reg := Reg) dj y₀ S R) y.x.st
    ∀ c, C.Admits c → ∀ w, C.Valid w → (C.T c).tab.sReg w = (C.T c).tab.pubReg → (C.T c).tab.InnerHolds w := by
  sorry

end FlockSoundness.Discharge.Composed
