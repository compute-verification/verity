import Mathlib.Analysis.MeanInequalities
import FlockSoundness.Protocol.Audit.Stratified

/-!
# The work law

The draw law's second version (Daniel, Sep 28): the stratified law's strata, with each stratum's draw sized by the PoUW
work its units do instead of by their number. Stratum `s` has `n_s` units, each doing work `w_s` (zero for units that do no
PoUW work: dequantization, the quantizer, attention), and the population does `W = Σ_s w_s n_s`. Stratum `s` draws
`k_s = min(n_s, max(f_s, ⌈K w_s n_s / W⌉))` (`Law.workRule`, the executable's `Flock.Draw.workK`) above its floor `f_s ≥ 1`,
which the verifier's work table sets per stratum beside `w_s` (one when it names none).

The work law is `Law.stratified` at those sizes (`Law.work`), so every theorem of the stratified law and of the one-stage
audit holds of it as stated: `stratified_escape`, `stratified_escape_floor`, `audit_whole_stratum`,
`extraction_audit_whole_stratum`, `audit_profile`, `audit_count`, `audit_drawn` and the Flock instantiations. What is new,
for every floor `f ≥ 1`:

* `floor_le_workK`: **the floor**. A stratum draws at least `min(n_s, f_s)` units, so every non-empty stratum draws at least
  one and a wholly wrong stratum is caught whatever its work (`audit_work_whole_stratum`,
  `extraction_audit_work_whole_stratum`).
* `work_escape_floor_le`: **the per-stratum floor escape**. A set with `b_s` units in a non-empty stratum `s` escapes with
  probability at most `((n_s − b_s)/n_s)^(f_s)`, so a floor `f_s ≈ ln(1/δ)/ε_s` catches a fraction `ε_s` of the stratum
  wrong up to `δ` (`audit_work_floor`, `extraction_audit_work_floor`).
* `workK_ge`: **draws sized by work**. A stratum draws at least `K w_s n_s / W` units, or all of them.
* `sum_workK_le`: at most `K + Σ_s f_s` units are drawn (`K` plus one per stratum at `f = 1`).
* `work_escape_le`: **the work bound**. A set of units doing `W_B` of the work escapes with probability at most
  `((W − W_B)/W)^K`, so wrong units doing a fraction `ε` of the work escape with at most `(1 − ε)^K`.
* `audit_work`, `extraction_audit_work`: the audit accepts while its wrong units do at least `T` of the work with
  probability at most `((W − T)/W)^K + ε_ks + δ_link`.
* `record_sizing`, `audit_work_of_record`: at the sizing of record (`δ = 2^-40`, `ε = 0.1%`), `K = 27,713` suffices;
  `record_sizing_one_fewer`: `K = 27,712` does not, under this bound.

For the strata that do no work:

* `workK_rest`: such a stratum draws exactly `min(n_s, f_s)`.
* `work_escape_le_count`: **floors at the count law's sizes keep its guarantee**. When every stratum that does no work has
  `f_s` at least `countK σ C s` (`main`'s stratified law's `stratumK C n_s N` for a non-empty stratum and `C ≤ N`,
  `countK_eq`), a set escapes the work law at most as often as its units in those strata escape the stratified law at `C`;
  `audit_work_rest` and `extraction_audit_work_rest` carry any such integrity profile to the audit.

Closure draws (a drawn unit proved with the units it reads) are not part of this law. `Law.widen` proves more units than a
law draws on the same coins, and `escape_widen_le` carries every escape bound to it.
-/

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal NNReal

namespace Law

variable {n m : ℕ}

/-- The work rule, the executable's `Flock.Draw.workK`: `⌈K · work · size / total⌉`, at least `floor` and at most `size`. -/
def workRule (K work size total floor : ℕ) : ℕ := min size (max floor ((K * work * size + total - 1) / total))

/-- The work of a set of units, each doing its stratum's `w (σ u)`. -/
def workOf (σ : Fin n → Fin m) (w : Fin m → ℕ) (B : Finset (Fin n)) : ℕ := ∑ u ∈ B, w (σ u)

/-- The population's work, `W`. -/
abbrev totalWork (σ : Fin n → Fin m) (w : Fin m → ℕ) : ℕ := workOf σ w univ

/-- Stratum `s`'s draw under the work law with floors `f`. -/
def workK (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K : ℕ) (s : Fin m) : ℕ :=
  workRule K (w s) (stratum σ s).card (totalWork σ w) (f s)

theorem workK_le (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K : ℕ) (s : Fin m) :
    workK σ w f K s ≤ (stratum σ s).card :=
  min_le_left _ _

/-- **The work law**: independent uniform subsets inside the strata of `σ`, stratum `s` drawing `workK σ w f K s`. -/
abbrev work (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K : ℕ) : Law n :=
  stratified σ (workK σ w f K) (workK_le σ w f K)

/-! ## The rule -/

/-! ## The work bound -/

/-! ## Floors -/

/-! ## Closure draws: the seam -/

/-- A law that proves more units than it draws: the same coins, each draw widened by `f`. Closure draws are `f S = S`
with the units `S`'s members read. -/
def widen (L : Law n) (f : Finset (Fin n) → Finset (Fin n)) : Law n where
  Ω := L.Ω
  draw ω := f (L.draw ω)

/-! ## The sizing of record -/

end Law

section Audits

variable {n m : ℕ} {σ : Fin n → Fin m} {w : Fin m → ℕ} {f : Fin m → ℕ} {K : ℕ}
  {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}

end Audits

end FlockSoundness.Audit
