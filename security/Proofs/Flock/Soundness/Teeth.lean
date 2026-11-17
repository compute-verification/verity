module

public import Proofs.Flock.Soundness.Assumptions
import Mathlib.Tactic.Positivity.Finset
import Proofs.Flock.Soundness.Game

/-! Both forms of SHA-512's collision resistance have teeth: they hold for every finder of an injective hash and fail
for a constant hash. -/

@[expose] public section

namespace FlockSoundness.Teeth

open Game

theorem noColl_of_injective {D β : Type} {H : List UInt8 → D} (hH : Function.Injective H)
    (out : β → Option (List UInt8 × List UInt8)) :
    (fun b => ∃ x y, out b = some (x, y) ∧ x ≠ y ∧ H x = H y) = fun _ => False :=
  funext fun _ => propext ⟨fun ⟨_, _, _, hne, he⟩ => hne (hH he), False.elim⟩

theorem strict_of_injective {D β : Type} {H : List UInt8 → D} (hH : Function.Injective H) (g : Game β)
    (s : Strategy g) (out : β → Option (List UInt8 × List UInt8)) (cost : β → ℝ) (q : ℝ) :
    Assumptions.SHA512CRStrict H g s out cost q := by
  intro _
  rw [noColl_of_injective hH out, prob_false, ENNReal.toReal_zero]
  positivity

theorem expected_of_injective {D β : Type} {H : List UInt8 → D} (hH : Function.Injective H) (g : Game β)
    (s : Strategy g) (out : β → Option (List UInt8 × List UInt8)) (cost : β → ℝ) (hc : ∀ b, 0 ≤ cost b) :
    Assumptions.SHA512CRExpected H g s out cost := by
  unfold Assumptions.SHA512CRExpected
  rw [noColl_of_injective hH out, prob_false, ENNReal.toReal_zero]
  exact div_nonneg (expect_nonneg hc g s) (by positivity)

theorem prob_const_coll {D : Type} (d : D) (s : Strategy (Game.ret () : Game Unit)) :
    prob (fun _ : Unit => ∃ x y, (some ([0], []) : Option (List UInt8 × List UInt8)) = some (x, y) ∧ x ≠ y ∧ d = d)
      (.ret ()) s = 1 :=
  le_antisymm (prob_le_one _ _ s) ((prob_true _ s).symm.le.trans
    (prob_mono (fun _ _ => ⟨[0], [], rfl, List.cons_ne_nil _ _, rfl⟩) _ s))

theorem not_strict_const {D : Type} (d : D) (s : Strategy (Game.ret () : Game Unit)) :
    ¬ Assumptions.SHA512CRStrict (fun _ => d) (.ret ()) s (fun _ => some ([0], [])) (fun _ => 0) 0 := by
  intro h
  have h1 := h fun _ => le_rfl
  rw [prob_const_coll d s] at h1
  norm_num at h1

theorem not_expected_const {D : Type} (d : D) (s : Strategy (Game.ret () : Game Unit)) :
    ¬ Assumptions.SHA512CRExpected (fun _ => d) (.ret ()) s (fun _ => some ([0], [])) (fun _ => 0) := by
  intro h
  unfold Assumptions.SHA512CRExpected at h
  rw [prob_const_coll d s, expect_const 0 _ s] at h
  norm_num at h

end FlockSoundness.Teeth
