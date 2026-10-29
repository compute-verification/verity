import Proofs.Flock.Soundness.Assumptions
import Proofs.Flock.Soundness.Game.Prob

/-!
# The collision-resistance assumptions have teeth

Each form of SHA-512's collision resistance is a hypothesis about one finder, so a mis-stated one could be true of every
hash, and then assume nothing, or of none, and then make every theorem that takes it vacuous. Neither is the case:
* both hold for every finder of an injective hash (`strict_of_injective`, `expected_of_injective`; the expected form for
  a nonnegative cost);
* both fail for a constant hash, at the finder that outputs two different strings after no evaluations
  (`not_strict_const`, `not_expected_const`).
-/

namespace FlockSoundness.Teeth

open Game

theorem noColl_of_injective {D β : Type} {H : List UInt8 → D} (hH : Function.Injective H)
    (out : β → Option (List UInt8 × List UInt8)) :
    (fun b => ∃ x y, out b = some (x, y) ∧ x ≠ y ∧ H x = H y) = fun _ => False :=
  funext fun _ => propext ⟨fun ⟨_, _, _, hne, he⟩ => hne (hH he), False.elim⟩

/-- **`SHA512CRStrict` holds for every finder of an injective hash.** -/
theorem strict_of_injective {D β : Type} {H : List UInt8 → D} (hH : Function.Injective H) (g : Game β)
    (s : Strategy g) (out : β → Option (List UInt8 × List UInt8)) (cost : β → ℝ) (q : ℝ) :
    Assumptions.SHA512CRStrict H g s out cost q := by
  intro _
  rw [noColl_of_injective hH out, prob_false, ENNReal.toReal_zero]
  positivity

/-- **`SHA512CRExpected` holds for every finder of an injective hash** whose cost is nonnegative. -/
theorem expected_of_injective {D β : Type} {H : List UInt8 → D} (hH : Function.Injective H) (g : Game β)
    (s : Strategy g) (out : β → Option (List UInt8 × List UInt8)) (cost : β → ℝ) (hc : ∀ b, 0 ≤ cost b) :
    Assumptions.SHA512CRExpected H g s out cost := by
  unfold Assumptions.SHA512CRExpected
  rw [noColl_of_injective hH out, prob_false, ENNReal.toReal_zero]
  exact div_nonneg (expect_nonneg hc g s) (by positivity)

/-- The finder that outputs `[0]` and `[]` after no evaluations finds a collision of a constant hash with probability 1. -/
theorem prob_const_coll {D : Type} (d : D) (s : Strategy (Game.ret () : Game Unit)) :
    prob (fun _ : Unit => ∃ x y, (some ([0], []) : Option (List UInt8 × List UInt8)) = some (x, y) ∧ x ≠ y ∧ d = d)
      (.ret ()) s = 1 :=
  le_antisymm (prob_le_one _ _ s) ((prob_true _ s).symm.le.trans
    (prob_mono (fun _ _ => ⟨[0], [], rfl, List.cons_ne_nil _ _, rfl⟩) _ s))

/-- **`SHA512CRStrict` fails for a constant hash**: a finder with no evaluations finds a collision. -/
theorem not_strict_const {D : Type} (d : D) (s : Strategy (Game.ret () : Game Unit)) :
    ¬ Assumptions.SHA512CRStrict (fun _ => d) (.ret ()) s (fun _ => some ([0], [])) (fun _ => 0) 0 := by
  intro h
  have h1 := h fun _ => le_rfl
  rw [prob_const_coll d s] at h1
  norm_num at h1

/-- **`SHA512CRExpected` fails for a constant hash**: a finder with no evaluations finds a collision. -/
theorem not_expected_const {D : Type} (d : D) (s : Strategy (Game.ret () : Game Unit)) :
    ¬ Assumptions.SHA512CRExpected (fun _ => d) (.ret ()) s (fun _ => some ([0], [])) (fun _ => 0) := by
  intro h
  unfold Assumptions.SHA512CRExpected at h
  rw [prob_const_coll d s, expect_const 0 _ s] at h
  norm_num at h

end FlockSoundness.Teeth
