module

public import Mathlib.Basic.Real.Basic
import Mathlib.Data.Finset.Attr
import Mathlib.Tactic.SetLike

/-!
# The chain's parameters (trusted)
-/

@[expose] public section

namespace Pouw

/-- The salt: λ = 128 uniform bits. -/
abbrev Salt : Type := Fin (2 ^ 128)

/-- The reduction's gap `g = 1 − (1 − g₀)/Ω_max` (§4.4, the Lean split, item 1). -/
noncomputable def reductionGap (g₀ Ωmax : ℝ) : ℝ := 1 - (1 - g₀) / Ωmax

namespace Params

/-- TT's target, `g₀ = 0.5%`. -/
noncomputable def gap0 : ℝ := 1 / 200

end Params

end Pouw
