import Definitions.Pouw.Game.Defs

/-!
# The chain's parameters (trusted)
-/

namespace Pouw

/-- The salt: λ = 128 uniform bits. -/
abbrev Salt : Type := Fin (2 ^ 128)

/-- The reduction's gap `γ = 1 − (1 − γ₀)/Ω_max` (§4.4, the Lean split, item 1). -/
noncomputable def reductionGamma (γ₀ Ωmax : ℝ) : ℝ := 1 - (1 - γ₀) / Ωmax

namespace Params

/-- TT's target, `γ₀ = 0.5%`. -/
noncomputable def γ0 : ℝ := 1 / 200

end Params

end Pouw
