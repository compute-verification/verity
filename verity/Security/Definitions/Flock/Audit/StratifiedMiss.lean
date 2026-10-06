import Definitions.Flock.Audit.Stratified

/-! The definitions of `Proofs.Flock.Soundness.Audit.StratifiedMiss` that C-Flock's guarantees read, or that a
definition here uses. The lemmas about them stay there. -/

namespace FlockSoundness.Audit

open Finset
open scoped ENNReal

namespace Law

variable {n m : ℕ}

/-- Stratum `s`'s escape with `c` of its units wrong: `C(n_s − c, k_s) / C(n_s, k_s)`. -/
noncomputable def stratumEscape (σ : Fin n → Fin m) (k : Fin m → ℕ) (s : Fin m) (c : ℕ) : ℝ≥0∞ :=
  (((stratum σ s).card - c).choose (k s) : ℝ≥0∞) / ((stratum σ s).card.choose (k s) : ℝ≥0∞)

/-- The factor by which stratum `s`'s escape falls when its `(c + 1)`-th unit is wrong: `(n_s − c − k_s)/(n_s − c)`. -/
noncomputable def stratumRatio (σ : Fin n → Fin m) (k : Fin m → ℕ) (s : Fin m) (c : ℕ) : ℝ≥0∞ :=
  (((stratum σ s).card - c - k s : ℕ) : ℝ≥0∞) / (((stratum σ s).card - c : ℕ) : ℝ≥0∞)

end Law

end FlockSoundness.Audit
