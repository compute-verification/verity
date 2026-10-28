import Definitions.Nci
import Definitions.Nci.BitView
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.Fin
import Proofs.Nci.Cover.Paths

/-!
# A declaration's circuit, bit by bit: blocks of word gates

The block of a set `R` of a declaration's gates (`block`) is the bits of `R` and of the gates `R` reads from outside,
`in(R)`, in the bit view (`Circuit.ofDecl`). The translation (`translation`) is what the cover theorem needs of it:

* (T1) the block computes every bit of `R` that reads something (`computed_block`): such a bit reads only bits of its
  gate's reads, which lie in `R ∪ in(R)`;
* (T2) the bits the block is handed outside the free bits are bits of `in(R) \ θ` (`inputs_block_subset`), so they are
  at most `novelReads R`: a bit of `R` is handed only when it reads nothing, and then it is free, and a bit of a baseline
  gate is free;
* (T3) the block has `width(R ∪ in(R))` bits (`card_block`), at most `w₀ #R + novelReads R + width(θ)` when every gate is
  at most `w₀` bits wide (`card_block_le`).

Every bit that reads something is a real bit of a computed gate (`computed_of_pred`), so the blocks of a division of the
computed gates cover the bit view (`mem_computed_block_of_pred`).

`width_aligned`: a block of whole words that is handed a bit of a region of whole words, each at least `f` bits wide,
is handed at least `f` of the region's bits, since all bits of one gate read the same bits.
-/

namespace Nci.Cover.SecurityProofs

open Finset BitView

section Words

variable (C : Verity.Circuit)

theorem w_le_W (g : Fin C.N) : C.w g ≤ W C := le_sup (f := C.w) (mem_univ g)

theorem enc_val (x : Bit C) : ((enc C x : Fin (C.N * W C)) : ℕ) = x.2 + W C * x.1 := by
  simp [enc, finProdFinEquiv]

variable {C}

theorem mem_bits {S : Finset (Fin C.N)} {x : Bit C} : x ∈ bits C S ↔ x.1 ∈ S ∧ (x.2 : ℕ) < C.w x.1 := by
  simp [bits]

theorem bits_mono {S T : Finset (Fin C.N)} (h : S ⊆ T) : bits C S ⊆ bits C T := fun _ hx =>
  mem_bits.2 ⟨h (mem_bits.1 hx).1, (mem_bits.1 hx).2⟩

theorem reads_subset {R : Finset (Fin C.N)} {g : Fin C.N} (hg : g ∈ R) : C.reads g ⊆ R ∪ C.into R := by
  intro a ha
  by_cases haR : a ∈ R
  · exact mem_union_left _ haR
  · exact mem_union_right _ (mem_filter.2 ⟨mem_univ _, haR, g, hg, ha⟩)

variable (C) in
theorem card_bits (S : Finset (Fin C.N)) : (bits C S).card = C.width S := by
  unfold bits Verity.Circuit.width
  rw [card_filter, sum_product]
  refine sum_congr rfl fun g _ => ?_
  show ∑ i : Fin (W C), (if (i : ℕ) < C.w g then 1 else 0) = C.w g
  rw [← card_filter, Fin.card_filter_val_lt, min_eq_right (w_le_W C g)]

/-- All real bits of one gate read the same bits. -/
theorem bitPred_eq {x y : Bit C} (hx : (x.2 : ℕ) < C.w x.1) (hy : (y.2 : ℕ) < C.w y.1) (h : x.1 = y.1) :
    bitPred C x = bitPred C y := by
  obtain ⟨x1, x2⟩ := x
  obtain ⟨y1, y2⟩ := y
  simp only at h hx hy
  subst h
  simp [bitPred, hx, hy]

end Words

section Blocks

variable (d : Verity.Registered)

/-- The numbering of `d`'s bits (`enc`), as the bit view's gates. -/
def benc : Bit d.D ≃ Fin (Circuit.ofDecl d).N := enc d.D

theorem ofDecl_pred (x : Bit d.D) :
    (Circuit.ofDecl d).pred (benc d x) = (bitPred d.D x).map (benc d).toEmbedding := by
  change (bitPred d.D ((enc d.D).symm (enc d.D x))).map (enc d.D).toEmbedding =
    (bitPred d.D x).map (enc d.D).toEmbedding
  rw [Equiv.symm_apply_apply]

theorem mem_ofDecl_θ {x : Bit d.D} : benc d x ∈ (Circuit.ofDecl d).θ ↔ x ∈ free d.D d.θ := by
  change (enc d.D).toEmbedding x ∈ (free d.D d.θ).map (enc d.D).toEmbedding ↔ _
  exact mem_map' _

/-- **The block of a set `R` of gates**: the bits of `R` and of the gates `R` reads from outside. -/
def block (R : Finset (Fin d.D.N)) : Finset (Fin (Circuit.ofDecl d).N) :=
  (bits d.D (R ∪ d.D.into R)).map (benc d).toEmbedding

variable {d}

/-- A real bit of `R` reads only bits of `R`'s block. -/
theorem pred_subset_block {R : Finset (Fin d.D.N)} {x : Bit d.D} (hx : x.1 ∈ R) :
    (Circuit.ofDecl d).pred (benc d x) ⊆ block d R := by
  intro y hy
  rw [ofDecl_pred] at hy
  obtain ⟨x', hx', rfl⟩ := mem_map.1 hy
  unfold bitPred at hx'
  split_ifs at hx'
  · exact mem_map_of_mem _ (bits_mono (reads_subset hx) hx')
  · exact absurd hx' (notMem_empty _)

/-- **(T1) The block computes `R`**: every real bit of `R` that reads something is computed by `R`'s block. -/
theorem computed_block (R : Finset (Fin d.D.N)) (x : Bit d.D) (hx : x ∈ bits d.D R)
    (hne : ((Circuit.ofDecl d).pred (benc d x)).Nonempty) :
    benc d x ∈ (Circuit.ofDecl d).computed (block d R) := by
  have hxR : x.1 ∈ R := (mem_bits.1 hx).1
  have hmem : benc d x ∈ block d R := mem_map_of_mem _ (bits_mono subset_union_left hx)
  refine mem_computed.2 ⟨hmem, fun hin => ?_⟩
  rcases (mem_inputs.1 hin).2 with h | h
  · exact hne.ne_empty h
  · exact h (pred_subset_block hxR)

/-- **(T2) The block is handed at most `R`'s novel reads**: for `R` of computed gates, the bits its block is handed
outside the free bits are bits of `in(R) \ θ`. -/
theorem inputs_block_subset (R : Finset (Fin d.D.N)) (hR : R ⊆ d.D.computed) :
    (Circuit.ofDecl d).inputs (block d R) \ (Circuit.ofDecl d).θ ⊆
      (bits d.D (d.D.into R \ d.θ)).map (benc d).toEmbedding := by
  intro y hy
  obtain ⟨hyin, hyθ⟩ := mem_sdiff.1 hy
  obtain ⟨hyb, hyc⟩ := mem_inputs.1 hyin
  obtain ⟨x, hx, rfl⟩ := mem_map.1 hyb
  obtain ⟨hx1, hxr⟩ := mem_bits.1 hx
  have hfree : x ∈ free d.D d.θ → False := fun hf => hyθ (mem_ofDecl_θ d |>.2 hf)
  rcases mem_union.1 hx1 with hxR | hxI
  · -- a bit of `R` that is handed reads nothing, and its gate is computed, so it is free
    have hnil : bitPred d.D x = ∅ := by
      rcases hyc with h | h
      · change (Circuit.ofDecl d).pred (benc d x) = ∅ at h
        rwa [ofDecl_pred, map_eq_empty] at h
      · exact absurd (pred_subset_block hxR) h
    have hcomp : d.D.isInput x.1 = false := (mem_filter.1 (hR hxR)).2
    exact (hfree (mem_filter.2 ⟨mem_univ _, hnil, Or.inr (Or.inl hcomp)⟩)).elim
  · by_cases hxθ : x.1 ∈ d.θ
    · have hnil : bitPred d.D x = ∅ := by
        simp [bitPred, d.D.input_reads _ (d.θ_input _ hxθ), bits]
      exact (hfree (mem_filter.2 ⟨mem_univ _, hnil, Or.inl hxθ⟩)).elim
    · exact mem_map_of_mem _ (mem_bits.2 ⟨mem_sdiff.2 ⟨hxI, hxθ⟩, hxr⟩)

/-- **(T3) The block's size** is the width of `R` and of what it reads. -/
theorem card_block (R : Finset (Fin d.D.N)) : (block d R).card = d.D.width (R ∪ d.D.into R) := by
  rw [block, card_map, card_bits]

/-- **The translation**: the block of `R` has `width(R ∪ in(R))` bits, computes every bit of `R` that reads something,
and is handed at most `novelReads R` bits outside the free bits. -/
theorem translation (R : Finset (Fin d.D.N)) (hR : R ⊆ d.D.computed) :
    (block d R).card = d.D.width (R ∪ d.D.into R) ∧
    (∀ x ∈ bits d.D R, ((Circuit.ofDecl d).pred (benc d x)).Nonempty →
      benc d x ∈ (Circuit.ofDecl d).computed (block d R)) ∧
    ((Circuit.ofDecl d).inputs (block d R) \ (Circuit.ofDecl d).θ).card ≤ novelReads d R := by
  refine ⟨card_block R, computed_block R, ?_⟩
  calc ((Circuit.ofDecl d).inputs (block d R) \ (Circuit.ofDecl d).θ).card
      ≤ ((bits d.D (d.D.into R \ d.θ)).map (benc d).toEmbedding).card := card_le_card (inputs_block_subset R hR)
    _ = novelReads d R := by rw [card_map, card_bits]; rfl

/-- A bit that reads something is a real bit of a computed gate. -/
theorem real_of_pred {y : Fin (Circuit.ofDecl d).N} (hne : ((Circuit.ofDecl d).pred y).Nonempty) :
    (((benc d).symm y).2 : ℕ) < d.D.w ((benc d).symm y).1 ∧ ((benc d).symm y).1 ∈ d.D.computed := by
  obtain ⟨x, rfl⟩ := (benc d).surjective y
  rw [Equiv.symm_apply_apply]
  rw [ofDecl_pred, map_nonempty] at hne
  unfold bitPred at hne
  split_ifs at hne with hr
  · refine ⟨hr, mem_filter.2 ⟨mem_univ _, ?_⟩⟩
    cases h : d.D.isInput x.1
    · rfl
    · obtain ⟨z, hz⟩ := hne
      rw [d.D.input_reads _ h] at hz
      exact absurd (mem_bits.1 hz).1 (notMem_empty _)
  · exact absurd hne (by simp)

theorem computed_of_pred {y : Fin (Circuit.ofDecl d).N} (hne : ((Circuit.ofDecl d).pred y).Nonempty) :
    ((benc d).symm y).1 ∈ d.D.computed := (real_of_pred hne).2

/-- **Blocks cover the bits that read something**: such a bit is computed by the block of any `R` holding its gate. -/
theorem mem_computed_block_of_pred {R : Finset (Fin d.D.N)} {y : Fin (Circuit.ofDecl d).N}
    (hne : ((Circuit.ofDecl d).pred y).Nonempty) (hR : ((benc d).symm y).1 ∈ R) :
    y ∈ (Circuit.ofDecl d).computed (block d R) := by
  have hr := (real_of_pred hne).1
  obtain ⟨x, rfl⟩ := (benc d).surjective y
  rw [Equiv.symm_apply_apply] at hR hr
  exact computed_block R x (mem_bits.2 ⟨hR, hr⟩) hne

/-- **The block's size, bounded**: when every gate is at most `w₀` bits wide, `R`'s block has at most
`w₀ #R + novelReads R + width(θ)` bits. -/
theorem card_block_le {w₀ : ℕ} (hw : ∀ g, d.D.w g ≤ w₀) (R : Finset (Fin d.D.N)) :
    (block d R).card ≤ w₀ * R.card + novelReads d R + d.D.width d.θ := by
  rw [card_block]
  have hdisj : Disjoint R (d.D.into R) :=
    disjoint_left.2 fun g hg hgi => (mem_filter.1 hgi).2.1 hg
  have hR : d.D.width R ≤ w₀ * R.card := by
    unfold Verity.Circuit.width
    rw [mul_comm, ← smul_eq_mul]
    exact sum_le_card_nsmul R d.D.w w₀ fun g _ => hw g
  have hI : d.D.width (d.D.into R) ≤ novelReads d R + d.D.width d.θ := by
    unfold novelReads Verity.Circuit.width
    calc ∑ g ∈ d.D.into R, d.D.w g ≤ ∑ g ∈ (d.D.into R \ d.θ) ∪ d.θ, d.D.w g :=
          sum_le_sum_of_subset (by rw [sdiff_union_self_eq_union]; exact subset_union_left)
      _ ≤ ∑ g ∈ d.D.into R \ d.θ, d.D.w g + ∑ g ∈ d.θ, d.D.w g := by
          rw [sum_union sdiff_disjoint]
  calc d.D.width (R ∪ d.D.into R) = d.D.width R + d.D.width (d.D.into R) := by
        unfold Verity.Circuit.width
        rw [sum_union hdisj]
    _ ≤ w₀ * R.card + (novelReads d R + d.D.width d.θ) := Nat.add_le_add hR hI
    _ = w₀ * R.card + novelReads d R + d.D.width d.θ := by rw [Nat.add_assoc]

/-- **Width on word-aligned blocks**: a block made of whole words that is handed any bit of a region made of whole
words, each at least `f` bits wide, is handed at least `f` of the region's bits. -/
theorem width_aligned (X Aw : Finset (Fin d.D.N)) (f : ℕ) (hf : ∀ a ∈ Aw, f ≤ d.D.w a)
    (hne : ((Circuit.ofDecl d).inputs ((bits d.D X).map (benc d).toEmbedding) ∩
      (bits d.D Aw).map (benc d).toEmbedding).Nonempty) :
    f ≤ ((Circuit.ofDecl d).inputs ((bits d.D X).map (benc d).toEmbedding) ∩
      (bits d.D Aw).map (benc d).toEmbedding).card := by
  obtain ⟨y, hy⟩ := hne
  obtain ⟨hyS, hyA⟩ := mem_inter.1 hy
  obtain ⟨hyb, hyc⟩ := mem_inputs.1 hyS
  obtain ⟨x, hx, rfl⟩ := mem_map.1 hyb
  obtain ⟨hxX, hxr⟩ := mem_bits.1 hx
  obtain ⟨x', hx', hxx'⟩ := mem_map.1 hyA
  have hx'eq : x' = x := (benc d).injective hxx'
  subst hx'eq
  have hxA := (mem_bits.1 hx').1
  set a := x'.1
  let bit : Fin (W d.D) ↪ Fin (Circuit.ofDecl d).N :=
    ⟨fun i => benc d (a, i), fun i j h => by simpa using (benc d).injective h⟩
  have hsub : (univ.filter fun i : Fin (W d.D) => (i : ℕ) < d.D.w a).map bit ⊆
      (Circuit.ofDecl d).inputs ((bits d.D X).map (benc d).toEmbedding) ∩
        (bits d.D Aw).map (benc d).toEmbedding := by
    intro z hz
    obtain ⟨i, hi, rfl⟩ := mem_map.1 hz
    have hir : (i : ℕ) < d.D.w a := (mem_filter.1 hi).2
    have hpe : (Circuit.ofDecl d).pred (bit i) = (Circuit.ofDecl d).pred ((benc d).toEmbedding x') := by
      change (Circuit.ofDecl d).pred (benc d (a, i)) = (Circuit.ofDecl d).pred (benc d x')
      rw [ofDecl_pred, ofDecl_pred, bitPred_eq (x := (a, i)) hir hxr rfl]
    have h1 : ((a, i) : Bit d.D) ∈ bits d.D X := mem_bits.2 ⟨hxX, hir⟩
    have h2 : ((a, i) : Bit d.D) ∈ bits d.D Aw := mem_bits.2 ⟨hxA, hir⟩
    have hbS : bit i ∈ (bits d.D X).map (benc d).toEmbedding := mem_map_of_mem _ h1
    have hbA : bit i ∈ (bits d.D Aw).map (benc d).toEmbedding := mem_map_of_mem _ h2
    exact mem_inter.2 ⟨mem_inputs.2 ⟨hbS, by rwa [hpe]⟩, hbA⟩
  calc f ≤ d.D.w a := hf a hxA
    _ = ((univ.filter fun i : Fin (W d.D) => (i : ℕ) < d.D.w a).map bit).card := by
        rw [card_map, Fin.card_filter_val_lt, min_eq_right (w_le_W d.D a)]
    _ ≤ _ := card_le_card hsub

end Blocks

end Nci.Cover.SecurityProofs
