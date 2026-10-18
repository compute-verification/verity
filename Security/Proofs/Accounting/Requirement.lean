import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Order.Interval.Finset.Nat
import Models.Accounting.Guarantees.Requirement

/-!
# A declared circuit's requirement, from what the audit credits

The values a layout packs in one block total at most the block's `B` bits (`bits_filter_block_le`): their bit ranges are
disjoint intervals of `[0, B)`. So the values of a set `S`, each in a block of `R`, outside the blocks `H` total at most
`|R \ H|·B` bits (`bits_le_bitsAt_add`). `Accounting.SecurityProofs.RequirementHeld` applies it to the live values and
the resident blocks, `Accounting.SecurityProofs.RequirementMoved` to the values read and the plan, and
`Accounting.SecurityProofs.RequirementDense` sums the live bits block by block. An execution that follows its wiring
reads only live values (`reads_subset_live`).
-/

namespace Accounting.Requirement

variable {G : Type} [DecidableEq G]

/-- A step reads only values live when it begins. -/
theorem reads_subset_live {W : Wiring G} {e : Execution G} (h : e.Follows W) (t : ℕ) :
    reads W e t ⊆ live W e t := by
  intro v hv
  simp only [reads, live, Finset.mem_filter] at hv ⊢
  obtain ⟨hv, hb, g, hg, hgt, hu⟩ := hv
  exact ⟨hv, hb, g, hg, hgt.ge, (h.2 g hg).1 hu⟩

variable (W : Wiring G) (ℓ : Layout G)

omit [DecidableEq G] in
/-- The values of a packed set in one block total at most the block's `B` bits. -/
theorem bits_filter_block_le {B : ℕ} {S T : Finset G} (hT : T ⊆ S) (h : Packs W ℓ B S) (j : ℕ) :
    bits W (T.filter fun v => ℓ.block v = j) ≤ B := by
  set U := T.filter fun v => ℓ.block v = j
  let I : G → Finset ℕ := fun v => Finset.Ico (ℓ.offset v) (ℓ.offset v + W.width v)
  have hdisj : (U : Set G).PairwiseDisjoint I := by
    intro u hu v hv huv
    have hu' := Finset.mem_filter.mp (Finset.mem_coe.mp hu)
    have hv' := Finset.mem_filter.mp (Finset.mem_coe.mp hv)
    refine Finset.disjoint_left.mpr fun x hx hx' => ?_
    simp only [I, Finset.mem_Ico] at hx hx'
    rcases (h u (hT hu'.1)).2 v (hT hv'.1) huv (hu'.2.trans hv'.2.symm) with hle | hle <;> omega
  calc bits W U = ∑ v ∈ U, (I v).card := by simp [bits, I]
    _ = (U.biUnion I).card := (Finset.card_biUnion hdisj).symm
    _ ≤ (Finset.range B).card := by
        refine Finset.card_le_card fun x hx => ?_
        obtain ⟨v, hv, hx⟩ := Finset.mem_biUnion.mp hx
        have hfit := (h v (hT (Finset.mem_filter.mp hv).1)).1
        simp only [I, Finset.mem_Ico] at hx
        exact Finset.mem_range.mpr (by omega)
    _ = B := Finset.card_range B

omit [DecidableEq G] in
/-- The values of a packed set `S`, each in a block of `R`, total at most their bits in the blocks `H` and `B` for each
block of `R` outside `H`. -/
theorem bits_le_bitsAt_add {B : ℕ} {S : Finset G} {R : Finset ℕ} (H : Finset ℕ) (hR : ∀ v ∈ S, ℓ.block v ∈ R)
    (h : Packs W ℓ B S) : bits W S ≤ bitsAt W ℓ S H + (R \ H).card * B := by
  have hmaps : ∀ v ∈ S.filter (fun v => ℓ.block v ∉ H), ℓ.block v ∈ R \ H := fun v hv =>
    Finset.mem_sdiff.mpr ⟨hR v (Finset.mem_filter.mp hv).1, (Finset.mem_filter.mp hv).2⟩
  have hout : bits W (S.filter fun v => ℓ.block v ∉ H) ≤ (R \ H).card * B := by
    calc bits W (S.filter fun v => ℓ.block v ∉ H)
        = ∑ j ∈ R \ H, bits W ((S.filter fun v => ℓ.block v ∉ H).filter fun v => ℓ.block v = j) :=
          (Finset.sum_fiberwise_of_maps_to hmaps _).symm
      _ ≤ ∑ _j ∈ R \ H, B := Finset.sum_le_sum fun j _ => bits_filter_block_le W ℓ (Finset.filter_subset _ _) h j
      _ = (R \ H).card * B := Finset.sum_const_nat fun _ _ => rfl
  calc bits W S = bitsAt W ℓ S H + bits W (S.filter fun v => ℓ.block v ∉ H) :=
        (Finset.sum_filter_add_sum_filter_not S _ _).symm
    _ ≤ bitsAt W ℓ S H + (R \ H).card * B := Nat.add_le_add_left hout _

end Accounting.Requirement

namespace Accounting.SecurityProofs

open Accounting.Requirement

theorem RequirementHeld : Accounting.Guarantees.RequirementHeld := by
  intro G _ W e ℓ s B t H ⟨hres, hpack, _⟩
  exact bits_le_bitsAt_add W ℓ H (fun v hv => hres ▸ Finset.mem_image_of_mem _ hv) hpack

theorem RequirementDense : Accounting.Guarantees.RequirementDense := by
  intro G _ W e ℓ s B t D ⟨hres, _, _⟩ hD
  have hmaps : ∀ v ∈ live W e t, ℓ.block v ∈ s.resident t := fun v hv => hres ▸ Finset.mem_image_of_mem _ hv
  calc (s.resident t).card * D = ∑ _j ∈ s.resident t, D := (Finset.sum_const_nat fun _ _ => rfl).symm
    _ ≤ ∑ j ∈ s.resident t, liveIn W e ℓ t j := Finset.sum_le_sum hD
    _ = liveBits W e t := Finset.sum_fiberwise_of_maps_to hmaps _

theorem RequirementMoved : Accounting.Guarantees.RequirementMoved := by
  intro G _ W e ℓ plan B t M ⟨hplan, hpack⟩
  exact bits_le_bitsAt_add W ℓ M (fun v hv => hplan ▸ Finset.mem_image_of_mem _ hv) hpack

end Accounting.SecurityProofs
