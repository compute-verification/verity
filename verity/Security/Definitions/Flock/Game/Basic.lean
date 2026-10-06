import Definitions.Core.Game

/-! The definitions of `Proofs.Flock.Soundness.Game.Basic` that C-Flock's guarantees read, or that a
definition here uses. The lemmas about them stay there. -/

namespace FlockSoundness

open scoped ENNReal

export Verity (Game)

namespace Game

export Verity.Game (ret send coin bind recv draw map Strategy prob value)

variable {α β γ : Type}

/-- The uniform average over a finite type. -/
noncomputable def avg {C : Type} [Fintype C] (f : C → ℝ≥0∞) : ℝ≥0∞ :=
  ∑ c : C, (Fintype.card C : ℝ≥0∞)⁻¹ * f c

open Classical in
/-- The probability that a uniform coin satisfies `p`, as an average of indicators. -/
noncomputable def prCoin {C : Type} [Fintype C] (p : C → Prop) : ℝ≥0∞ :=
  avg fun c => if p c then 1 else 0

end Game

end FlockSoundness
