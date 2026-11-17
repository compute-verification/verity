module

public import Definitions.Slowdown.BitView
public import Definitions.Slowdown.Dark
public import Definitions.InferenceOnly.Assumptions.Theta
public import Definitions.InferenceOnly.Assumptions
public import Definitions.Slowdown.Assumptions.Cover
public import Definitions.Slowdown.Assumptions.CoverTraining

/-! The cover theorem joined to the inference-only policy through a reference training circuit that the declared circuit contains. -/

@[expose] public section

namespace Slowdown.Cover.Guarantees

/-- **Training through a compliant declaration pays the cover theorem's factor**: a declaration `d` that contains a
reference training circuit `r` and complies at `(Γ, α)` (`CompliesRate`), with its registered information free of the
reference training algorithm's fresh input (`ThetaRegistered`), has `α₀ b size(r) ≤ (α² Γ + b α) size(d)` when the counted
products of `r`'s bit view are a fraction `α₀` of `r`'s gates, under `UpdateHardness` at the block size
`s ≥ w₀ Γ + α Γ + width(θ)` (no gate of `r` wider than `w₀`) and the upstream certificate at depth `D` with
`α Γ < b D`. -/
def CoverTraining : Prop :=
  ∀ (Γ : ℕ) (α b α₀ : ℝ) (r d : Verity.RegisteredCircuit) (M : Verity.Containment r.D d.D)
    {J L : Type} [DecidableEq J] [DecidableEq L] (Mb : MulBlocks (Circuit.ofDecl r) J L)
    (A : J → Finset (Fin (Circuit.ofDecl r).N)) (s D w₀ : ℕ),
    0 < b → 0 ≤ α → InferenceOnly.CompliesRate Γ α d →
    InferenceOnly.Assumptions.ThetaRegistered r d (Slowdown.PartialContainment.ofContainment M) →
    Assumptions.UpdateHardness Mb s b → Mb.Upstream A D → Assumptions.CapBelowDepth (α * Γ) b D →
    Assumptions.BlockFits r Γ (α * Γ) w₀ s → InferenceOnly.Assumptions.Coverage α₀ r Mb.P.card →
      α₀ * b * r.D.size r.D.computed ≤ (α * α * Γ + b * α) * d.D.size d.D.computed

end Slowdown.Cover.Guarantees
