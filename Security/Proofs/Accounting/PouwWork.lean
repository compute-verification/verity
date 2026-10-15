import Models.Accounting.Guarantees.PouwWork

/-!
# PoUW's work lower bound on an isolated node (proof)

`Accounting.SecurityProofs.PouwLocalOff`: an accepted coin whose node is short of `Wb · Nmin` credits is short of
`Nmin` right blocks (A2), right by luck (A2b), in the floor's skip event, or under the floor, which A5 at 0 rules out;
the union bound sums the first three.
-/

namespace Accounting.SecurityProofs

open Accounting.PouwWork Finset

namespace PouwWork

theorem pr_mono {α : Type} [Fintype α] {E F : α → Prop} (h : ∀ x, E x → F x) : Pouw.pr E ≤ Pouw.pr F := by
  classical
  unfold Pouw.pr
  refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
  have hsub : (univ.filter E) ⊆ univ.filter F := fun x hx => by
    simp only [mem_filter, mem_univ, true_and] at hx ⊢
    exact h x hx
  exact_mod_cast card_le_card hsub

theorem pr_or_le {α : Type} [Fintype α] (E F : α → Prop) :
    Pouw.pr (fun x => E x ∨ F x) ≤ Pouw.pr E + Pouw.pr F := by
  classical
  unfold Pouw.pr
  rw [← add_div]
  refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
  have hsub : (univ.filter fun x => E x ∨ F x) ⊆ univ.filter E ∪ univ.filter F := fun x hx => by
    simp only [mem_filter, mem_union, mem_univ, true_and] at hx ⊢
    exact hx
  have key : ((univ.filter fun x => E x ∨ F x).card : ℝ) ≤ ((univ.filter E).card : ℝ) + (univ.filter F).card := by
    exact_mod_cast (card_le_card hsub).trans (card_union_le _ _)
  convert key using 3
  exact Finset.filter_congr_decidable _ _ _

end PouwWork

theorem PouwLocalOff : Accounting.Guarantees.PouwLocalOff := by
  intro Ω _ L S P X π m Wb Nmin ρ δdet δluck δskip hdet hluck hfl hH
  obtain ⟨Z, hZ, hfloor⟩ := hfl
  have hsub : ∀ r, (Accepts S P r ∧ credits S P X π m r < Wb * Nmin) →
      ((Accepts S P r ∧ good S P r < Nmin) ∨ (Accepts S P r ∧ Nmin ≤ good S P r ∧ goodAll S P < Nmin)) ∨ Z r := by
    intro r ⟨hacc, hlt⟩
    by_cases hg : good S P r < Nmin
    · exact .inl (.inl ⟨hacc, hg⟩)
    by_cases hall : goodAll S P < Nmin
    · exact .inl (.inr ⟨hacc, Nat.le_of_not_lt hg, hall⟩)
    by_cases hz : Z r
    · exact .inr hz
    have h1 := hfloor r hz (Nat.le_of_not_lt hall)
    have h2 := hH r hacc
    have h3 : help S P X r = 0 := Nat.le_zero.mp h2
    rw [h3, Nat.mul_zero, Nat.add_zero] at h1
    omega
  calc Pouw.pr (fun r => Accepts S P r ∧ credits S P X π m r < Wb * Nmin)
      ≤ Pouw.pr (fun r => ((Accepts S P r ∧ good S P r < Nmin) ∨
          (Accepts S P r ∧ Nmin ≤ good S P r ∧ goodAll S P < Nmin)) ∨ Z r) := PouwWork.pr_mono hsub
    _ ≤ Pouw.pr (fun r => (Accepts S P r ∧ good S P r < Nmin) ∨
          (Accepts S P r ∧ Nmin ≤ good S P r ∧ goodAll S P < Nmin)) + Pouw.pr Z := PouwWork.pr_or_le _ _
    _ ≤ (Pouw.pr (fun r => Accepts S P r ∧ good S P r < Nmin) +
          Pouw.pr (fun r => Accepts S P r ∧ Nmin ≤ good S P r ∧ goodAll S P < Nmin)) + Pouw.pr Z :=
        add_le_add_left (PouwWork.pr_or_le _ _) _
    _ ≤ δdet + δluck + δskip := add_le_add (add_le_add hdet hluck) hZ

end Accounting.SecurityProofs
