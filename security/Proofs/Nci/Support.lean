module

public import Proofs.Core
public import Proofs.Nci.Lemmas

/-!
# What an IU's reads determine: support lemmas

Every value an IU `R` holds is a function of the values on `in(R)` (`eval_agree_of_into`), and a set of gates takes at
most `2^width` distinct values (`support_le_width`).
-/

@[expose] public section

namespace Nci.SecurityProofs

open Verity Finset

/-- The values on `R` are determined by the values on `in(R)`. -/
theorem eval_agree_of_into {D : Circuit} {R : Finset (Fin D.N)} (hR : R ⊆ D.computed) {a b : D.Trace}
    (h : ∀ g ∈ D.into R, D.eval a g = D.eval b g) : ∀ g ∈ R, D.eval a g = D.eval b g := by
  intro g
  induction g using WellFoundedLT.induction with
  | _ g ih => ?_
  intro hg
  have hc : D.isInput g = false := by simpa [Circuit.computed] using hR hg
  rw [D.eval_op a hc, D.eval_op b hc]
  refine D.op_congr g _ _ fun x hx => ?_
  by_cases hxR : x ∈ R
  · exact ih x (D.reads_lt g x hx) hxR
  · exact h x (by simpa [Circuit.into] using ⟨hxR, g, hg, hx⟩)

theorem card_bitVec (n : ℕ) : Fintype.card (BitVec n) = 2 ^ n :=
  (Fintype.card_congr (⟨BitVec.toFin, BitVec.ofFin, fun _ => by simp, fun _ => by simp⟩ : BitVec n ≃ Fin (2 ^ n))).trans
    (Fintype.card_fin _)

/-- `f`'s image has at most as many elements as `r`'s when `f` is a function of `r` on `s`. -/
theorem card_image_le_of_factor {α β γ : Type} [DecidableEq β] [DecidableEq γ] {s : Finset α} {f : α → β} {r : α → γ}
    (h : ∀ a ∈ s, ∀ b ∈ s, r a = r b → f a = f b) : (s.image f).card ≤ (s.image r).card := by
  classical
  calc (s.image f).card = ((s.image fun a => (r a, f a)).image Prod.snd).card := by rw [image_image]; rfl
    _ ≤ (s.image fun a => (r a, f a)).card := card_image_le
    _ ≤ (s.image r).card := by
      refine card_le_card_of_injOn Prod.fst (fun x hx => ?_) fun x hx y hy hxy => ?_
      · obtain ⟨a, ha, rfl⟩ := mem_image.mp hx
        exact mem_image_of_mem r ha
      · obtain ⟨a, ha, rfl⟩ := mem_image.mp (mem_coe.mp hx)
        obtain ⟨b, hb, rfl⟩ := mem_image.mp (mem_coe.mp hy)
        exact Prod.ext hxy (h a ha b hb hxy)

/-- A set of gates takes at most `2^width` distinct values. -/
theorem support_le_width (d : RegisteredCircuit) (θv : d.D.Trace) (T : Finset (Fin d.D.N)) :
    support d θv T ≤ 2 ^ d.D.width T := by
  classical
  calc support d θv T ≤ Fintype.card ((g : T) → BitVec (d.D.w g)) := card_le_univ _
    _ = 2 ^ d.D.width T := by
        rw [Fintype.card_pi]
        simp only [card_bitVec]
        rw [prod_coe_sort T fun g => 2 ^ d.D.w g, prod_pow_eq_pow_sum]
        rfl

end Nci.SecurityProofs

