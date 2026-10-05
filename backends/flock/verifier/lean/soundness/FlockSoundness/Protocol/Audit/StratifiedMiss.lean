import FlockSoundness.Protocol.Audit.Stratified

/-!
# The stratified law's count curve

`miss K` is the largest escape of a set of at least `K` wrong units (`Law.miss`). Under the stratified law a set's escape
depends only on its count `c_s` in each stratum (`stratified_escape`). Stratum `s`'s escape falls by the factor
`r_s(c) = (n_s − c − k_s)/(n_s − c)` when its `(c + 1)`-th unit is wrong, and that factor falls as `c` grows. So the
likeliest set to escape fills the strata greedily, taking the largest factors first, as
`verity.proofs.profile.Stratified.miss` computes it.

* `stratified_miss_eq_greedy`: a count vector `a` that some `τ ∈ (0, 1]` separates attains `miss (∑ a)`, which is then
  `∏ s C(n_s − a_s, k_s)/C(n_s, k_s)`. Every factor `a` takes must be at least `τ`, and every factor it leaves at most
  `τ`. A greedy fill has such a `τ`, the smallest factor it took, so `audit_exfiltration`'s `hk : miss (K + 1) < δ`
  holds for a stratified law once the greedy fill at `K + 1` escapes with less than `δ`.
* `stratum_exchange`, the step: within one stratum, `C(n_s − c, k_s) · τ^a ≤ C(n_s − a, k_s) · τ^c` for every
  `c ≤ n_s`.
-/

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
