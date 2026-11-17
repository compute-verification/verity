module

public import Proofs.Flock.Soundness.Audit.Periods
public import Definitions.Flock.Audit.SamplingDesign
import Proofs.Flock.Soundness.Audit.SamplingDesign

/-! The pooled missProb of incorrect proof blocks that escape every period's sample, stated in `prCoin` only: no theorem states it in `Claims.Pr` or at `δ`. -/

@[expose] public section

namespace FlockSoundness.Audit

open Game
open scoped ENNReal

namespace SamplingDesign

variable {N kd : ℕ}

def past {α : Type} {T : ℕ} (ω : Fin T → α) (t : Fin T) : Fin t → α :=
  fun i => ω (Fin.castLE t.isLt.le i)

abbrev Pooled (N kd : ℕ) (hk : kd ≤ N) (T : ℕ) : Type :=
  (t : Fin T) → (Fin t → (subset N kd hk).Ω) → Finset (Fin N)

namespace Pooled

variable {hk : kd ≤ N} {T : ℕ}

def incorrect (A : Pooled N kd hk T) (ω : Fin T → (subset N kd hk).Ω) (t : Fin T) : Finset (Fin N) :=
  A t (past ω t)

def MissProb (A : Pooled N kd hk T) (ω : Fin T → (subset N kd hk).Ω) : Prop :=
  ∀ t, Disjoint (ω t).1 (A.incorrect ω t)

def total (A : Pooled N kd hk T) (ω : Fin T → (subset N kd hk).Ω) : ℕ :=
  ∑ t, (A.incorrect ω t).card

theorem predictable (A : Pooled N kd hk T) :
    Periods.Predictable (Ω := fun _ => (subset N kd hk).Ω) fun t ω => A.incorrect ω t :=
  fun t _ _ h => congrArg (A t) (funext fun i => h _ (show (Fin.castLE t.isLt.le i).val < t.val from i.isLt))

theorem adapted_escapes (A : Pooled N kd hk T) :
    Periods.Adapted (Ω := fun _ => (subset N kd hk).Ω) fun t ω => Disjoint (ω t).1 (A.incorrect ω t) :=
  fun t ω ω' h => congrArg₂ (fun x y => Disjoint (Subtype.val x) y) (h t le_rfl) (A.predictable.adapted t ω ω' h)

theorem escape_update (A : Pooled N kd hk T) (ω : Fin T → (subset N kd hk).Ω) (t : Fin T) :
    prCoin (fun a => Disjoint (Function.update ω t a t).1 (A.incorrect (Function.update ω t a) t)) =
      (subset N kd hk).missProb (A.incorrect ω t) := by
  have hw : ∀ a, A.incorrect (Function.update ω t a) t = A.incorrect ω t := fun a =>
    A.predictable t _ _ fun i hi => Function.update_of_ne (ne_of_lt hi) _ _
  simp only [Function.update_self, hw]
  rfl

end Pooled

noncomputable def Pooled.ofAdapted {hk : kd ≤ N} {T : ℕ}
    (W : Fin T → (Fin T → (subset N kd hk).Ω) → Finset (Fin N)) : Pooled N kd hk T :=
  fun t h => W t fun i => if hi : i.val < t.val then h ⟨i.val, hi⟩ else Classical.arbitrary _

theorem Pooled.incorrect_ofAdapted {hk : kd ≤ N} {T : ℕ}
    {W : Fin T → (Fin T → (subset N kd hk).Ω) → Finset (Fin N)}
    (hW : Periods.Predictable (Ω := fun _ => (subset N kd hk).Ω) W)
    (ω : Fin T → (subset N kd hk).Ω) (t : Fin T) : (Pooled.ofAdapted W).incorrect ω t = W t ω := by
  refine hW t _ _ fun i hi => ?_
  have hi' : i.val < t.val := hi
  show (if h : i.val < t.val then past ω t ⟨i.val, h⟩ else _) = ω i
  rw [dite_eq_left hi']
  rfl

theorem choose_div_le_one_sub_pow (hk : kd ≤ N) {w : ℕ} (hw : w ≤ N) :
    ((N - w).choose kd : ℝ≥0∞) / (N.choose kd : ℝ≥0∞) ≤ (1 - (kd : ℝ≥0∞) / N) ^ w := by
  rcases Nat.eq_zero_or_pos N with rfl | hN
  · obtain rfl : w = 0 := by omega
    obtain rfl : kd = 0 := by omega
    simp
  have hN0 : (N : ℝ≥0∞) ≠ 0 := by exact_mod_cast hN.ne'
  have hq : (1 - (kd : ℝ≥0∞) / N) = ((N - kd : ℕ) : ℝ≥0∞) / (N : ℝ≥0∞) := by
    rw [ENNReal.natCast_sub, ENNReal.sub_div (fun _ _ => hN0), ENNReal.div_self hN0 (ENNReal.natCast_ne_top _)]
  rw [hq, div_eq_mul_inv ((N - kd : ℕ) : ℝ≥0∞), mul_pow, ← ENNReal.inv_pow, ← div_eq_mul_inv]
  have hc : (N.choose kd : ℝ≥0∞) ≠ 0 := by exact_mod_cast (Nat.choose_pos hk).ne'
  have hp : ((N : ℝ≥0∞)) ^ w ≠ 0 := pow_ne_zero _ hN0
  rw [ENNReal.div_le_iff hc (ENNReal.natCast_ne_top _), mul_comm, ← mul_div_assoc,
    ENNReal.le_div_iff_mul_le (Or.inl hp) (Or.inl (ENNReal.pow_ne_top (ENNReal.natCast_ne_top _)))]
  exact_mod_cast choose_sub_mul_pow_le N kd w hw

theorem one_sub_div_le_one : (1 - (kd : ℝ≥0∞) / N) ≤ 1 := tsub_le_self

theorem prod_choose_div_le_pow (hk : kd ≤ N) {T : ℕ} (w : Fin T → ℕ) (hw : ∀ t, w t ≤ N) {M : ℕ}
    (hM : M ≤ ∑ t, w t) :
    ∏ t, ((N - w t).choose kd : ℝ≥0∞) / (N.choose kd : ℝ≥0∞) ≤ (1 - (kd : ℝ≥0∞) / N) ^ M :=
  calc ∏ t, ((N - w t).choose kd : ℝ≥0∞) / (N.choose kd : ℝ≥0∞)
      ≤ ∏ t, (1 - (kd : ℝ≥0∞) / N) ^ w t := Finset.prod_le_prod fun t _ => choose_div_le_one_sub_pow hk (hw t)
    _ = (1 - (kd : ℝ≥0∞) / N) ^ ∑ t, w t := Finset.prod_pow_eq_pow_sum _ _ _
    _ ≤ (1 - (kd : ℝ≥0∞) / N) ^ M := pow_le_pow_of_le_one zero_le one_sub_div_le_one hM

theorem pooled_escape_prod {hk : kd ≤ N} {T : ℕ} (A : Pooled N kd hk T) (w : Fin T → ℕ)
    (hw : ∀ t h, w t ≤ (A t h).card) :
    prCoin (fun ω => A.MissProb ω) ≤ ∏ t, ((N - w t).choose kd : ℝ≥0∞) / (N.choose kd : ℝ≥0∞) :=
  Periods.prCoin_forall_le_prod (Ω := fun _ => (subset N kd hk).Ω) _ A.adapted_escapes _ fun t ω => by
    rw [A.escape_update, subset_escape]
    exact ENNReal.div_le_div_right (Nat.cast_le.2 (Nat.choose_le_choose kd (Nat.sub_le_sub_left (hw t _) N))) _

theorem pooled_escape {hk : kd ≤ N} {T : ℕ} (A : Pooled N kd hk T) (M : ℕ) :
    prCoin (fun ω => A.MissProb ω ∧ M ≤ A.total ω) ≤ (1 - (kd : ℝ≥0∞) / N) ^ M :=
  Periods.prCoin_forall_sum_le_pow (Ω := fun _ => (subset N kd hk).Ω) _ (fun t ω => (A.incorrect ω t).card)
    A.adapted_escapes (fun t ω ω' h => congrArg Finset.card (A.predictable t ω ω' h)) one_sub_div_le_one
    (fun t ω => by
      rw [A.escape_update, subset_escape]
      exact choose_div_le_one_sub_pow hk (by simpa using Finset.card_le_univ (A.incorrect ω t))) M

theorem pooled_escape_adapted {hk : kd ≤ N} {T : ℕ}
    (W : Fin T → (Fin T → (subset N kd hk).Ω) → Finset (Fin N))
    (hW : Periods.Predictable (Ω := fun _ => (subset N kd hk).Ω) W) (M : ℕ) :
    prCoin (fun ω => (∀ t, Disjoint (ω t).1 (W t ω)) ∧ M ≤ ∑ t, (W t ω).card) ≤
      (1 - (kd : ℝ≥0∞) / N) ^ M := by
  have h := pooled_escape (Pooled.ofAdapted W) M
  simp only [Pooled.MissProb, Pooled.total, Pooled.incorrect_ofAdapted hW] at h
  exact h

end SamplingDesign

end FlockSoundness.Audit
