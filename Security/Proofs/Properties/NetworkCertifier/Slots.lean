import Mathlib.Order.Interval.Finset.Nat
import Mathlib.Data.Finset.Prod

/-!
# Counting slots

The Finset arithmetic of the circuit count's proof (`Proofs.Properties.NetworkCertifier.Counts`): a slot is a template,
one of its units and a bit of its input rows; a box is a template's units and bits in two ranges (`box`, `card_box`,
`box_disjoint`); the union of a list of sets (`bigU`) has its members' sizes summed when they are pairwise disjoint
(`card_bigU`), and at most that sum always (`card_bigU_le`), a sum over a list's indices being one over its members
(`sum_range_eq`); and a set covered by the images of a set's slots outside a subset and of another set has at most
their difference's size plus the other's (`card_le_of_cover`).
-/

namespace Properties.NetworkCertifier.Counts

/-- A slot: a template, one of its units and a bit of its input rows. -/
abbrev Slot := ℕ × ℕ × ℕ

/-- Template `i`'s units `[lo, hi)` at bits `[b0, b1)`. -/
def box (i lo hi b0 b1 : ℕ) : Finset Slot := {i} ×ˢ Finset.Ico lo hi ×ˢ Finset.Ico b0 b1

theorem mem_box {i lo hi b0 b1 : ℕ} {s : Slot} :
    s ∈ box i lo hi b0 b1 ↔ s.1 = i ∧ lo ≤ s.2.1 ∧ s.2.1 < hi ∧ b0 ≤ s.2.2 ∧ s.2.2 < b1 := by
  simp only [box, Finset.mem_product, Finset.mem_singleton, Finset.mem_Ico, and_assoc]

theorem card_box (i lo hi b0 b1 : ℕ) : (box i lo hi b0 b1).card = (hi - lo) * (b1 - b0) := by
  simp only [box, Finset.card_product, Finset.card_singleton, Nat.card_Ico, Nat.one_mul]

theorem box_disjoint {i lo hi b0 b1 i' lo' hi' b0' b1' : ℕ}
    (h : i ≠ i' ∨ hi ≤ lo' ∨ hi' ≤ lo ∨ b1 ≤ b0' ∨ b1' ≤ b0) :
    Disjoint (box i lo hi b0 b1) (box i' lo' hi' b0' b1') := by
  rw [Finset.disjoint_left]
  intro s hs hs'
  rw [mem_box] at hs hs'
  omega

theorem box_subset {i lo hi b0 b1 lo' hi' b0' b1' : ℕ} (h1 : lo' ≤ lo) (h2 : hi ≤ hi') (h3 : b0' ≤ b0)
    (h4 : b1 ≤ b1') : box i lo hi b0 b1 ⊆ box i lo' hi' b0' b1' := by
  intro s hs
  rw [mem_box] at hs ⊢
  omega

/-- Template `i`'s units below `U` at the columns `cols`. -/
def pbox (i U : ℕ) (cols : Finset ℕ) : Finset Slot := {i} ×ˢ Finset.range U ×ˢ cols

theorem mem_pbox {i U : ℕ} {cols : Finset ℕ} {s : Slot} : s ∈ pbox i U cols ↔ s.1 = i ∧ s.2.1 < U ∧ s.2.2 ∈ cols := by
  simp only [pbox, Finset.mem_product, Finset.mem_singleton, Finset.mem_range]

theorem card_pbox (i U : ℕ) (cols : Finset ℕ) : (pbox i U cols).card = U * cols.card := by
  simp only [pbox, Finset.card_product, Finset.card_singleton, Finset.card_range, Nat.one_mul]

/-- The union of a list of sets. -/
def bigU {α : Type} [DecidableEq α] (L : List (Finset α)) : Finset α := L.foldr (· ∪ ·) ∅

theorem bigU_cons {α : Type} [DecidableEq α] (A : Finset α) (L : List (Finset α)) : bigU (A :: L) = A ∪ bigU L :=
  rfl

theorem mem_bigU {α : Type} [DecidableEq α] {a : α} : ∀ {L : List (Finset α)}, a ∈ bigU L ↔ ∃ A ∈ L, a ∈ A
  | [] => by simp [bigU]
  | A :: L => by
    rw [bigU_cons, Finset.mem_union, mem_bigU]
    simp only [List.mem_cons, exists_eq_or_imp]

theorem bigU_subset {α : Type} [DecidableEq α] {S : Finset α} {L : List (Finset α)} (h : ∀ A ∈ L, A ⊆ S) :
    bigU L ⊆ S := by
  intro a ha
  obtain ⟨A, hA, ha⟩ := mem_bigU.1 ha
  exact h A hA ha

theorem subset_bigU {α : Type} [DecidableEq α] {A : Finset α} {L : List (Finset α)} (h : A ∈ L) : A ⊆ bigU L :=
  fun _ ha => mem_bigU.2 ⟨A, h, ha⟩

theorem card_bigU {α : Type} [DecidableEq α] : ∀ {L : List (Finset α)}, L.Pairwise Disjoint →
    (bigU L).card = (L.map Finset.card).sum
  | [], _ => by simp [bigU]
  | A :: L, h => by
    rw [List.pairwise_cons] at h
    have hd : Disjoint A (bigU L) := by
      rw [Finset.disjoint_left]
      intro a ha hb
      obtain ⟨B, hB, hb⟩ := mem_bigU.1 hb
      exact Finset.disjoint_left.1 (h.1 B hB) ha hb
    rw [bigU_cons, Finset.card_union_of_disjoint hd, card_bigU h.2, List.map_cons, List.sum_cons]

theorem card_bigU_le {α : Type} [DecidableEq α] : ∀ (L : List (Finset α)), (bigU L).card ≤ (L.map Finset.card).sum
  | [] => by simp [bigU]
  | A :: L => by
    rw [bigU_cons, List.map_cons, List.sum_cons]
    exact (Finset.card_union_le _ _).trans (Nat.add_le_add_left (card_bigU_le L) _)

theorem sum_range_eq {α : Type} (g : α → ℕ) (W : List α) (F : ℕ → ℕ) (h : ∀ i (hi : i < W.length), F i = g W[i]) :
    ((List.range W.length).map F).sum = (W.map g).sum := by
  congr 1
  apply List.ext_getElem (by simp)
  intro i h1 h2
  simp only [List.getElem_map, List.getElem_range]
  exact h i (by simpa using h1)

/-- **A cover's bound**: if every member of `X` is the image under `ψ` of a member of `S` outside its subset `C`, or the
image under `ψd` of a member of `D`, then `X` has at most `#S − #C + #D` members. -/
theorem card_le_of_cover {α β : Type} [DecidableEq α] [DecidableEq β] {X : Finset β} {S C D : Finset α}
    (ψ ψd : α → β) (hCS : C ⊆ S) (h : ∀ y ∈ X, (∃ s ∈ S, s ∉ C ∧ ψ s = y) ∨ ∃ d ∈ D, ψd d = y) :
    X.card ≤ S.card - C.card + D.card := by
  have hX : X ⊆ (S \ C).image ψ ∪ D.image ψd := by
    intro y hy
    rw [Finset.mem_union, Finset.mem_image, Finset.mem_image]
    rcases h y hy with ⟨s, hs, hc, rfl⟩ | ⟨d, hd, rfl⟩
    · exact .inl ⟨s, Finset.mem_sdiff.2 ⟨hs, hc⟩, rfl⟩
    · exact .inr ⟨d, hd, rfl⟩
  calc X.card ≤ ((S \ C).image ψ ∪ D.image ψd).card := Finset.card_le_card hX
    _ ≤ ((S \ C).image ψ).card + (D.image ψd).card := Finset.card_union_le _ _
    _ ≤ (S \ C).card + D.card := Nat.add_le_add Finset.card_image_le Finset.card_image_le
    _ = S.card - C.card + D.card := by rw [Finset.card_sdiff_of_subset hCS]

end Properties.NetworkCertifier.Counts
