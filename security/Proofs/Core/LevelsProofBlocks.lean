module

public import Proofs.Core.LevelsCount
public import Proofs.Core.LevelsTwo
public import Definitions.Core.LevelsProofBlocks

/-!
# A claims partition's proof blocks as the last level's blocks: what carries across

Where `blk` places a claims partition `P`'s proof blocks as the blocks of `L`'s last level (`Levels.SlotsAreBlocks`):

* `Levels.SlotsAreBlocks.image_subset`, `card_image`: a set of slots goes to a set of the last level's blocks of its
  size, and `faulty_image_subset`, `card_faulty_image`: so does the incorrect set `Claims.Sampling.faulty`.
* `Levels.SlotsAreBlocks.image_filter`: every set of the last level's blocks is the image of the slots it holds.
* `Levels.MissesAsDesign.missProb_image`, `missProb_faulty`: where `L`'s sample misses the slots as the claims design
  `d` on `P` does (`Levels.MissesAsDesign`), `L.design` misses the image of a set of slots, the incorrect set among
  them, with probability `Design.Pa d P` of the set, and `relS_iff` reads the sampling relation `RelS` at `δ` as
  `δ` below that miss.
* `Levels.countAt_eq_slots`: under both, the count at `δ` (`Levels.countAt`) is the largest set of slots `d` accepts
  with probability at least `δ`, at any number of levels; `missesAsDesign_one` reduces the miss to the level's design
  at one level (`Levels.design_one`).
-/

@[expose] public section

namespace Verity

namespace Levels

open Claims.Sampling
open scoped ENNReal

variable {G : Type} [Fintype G] [DecidableEq G] {L : Levels G} {P : Partition} {blk : ℕ → Finset G}

theorem SlotsAreBlocks.image_subset (h : L.SlotsAreBlocks P blk) {B : Finset ℕ}
    (hB : B ⊆ Finset.range P.blocks.length) : B.image blk ⊆ L.blocks := fun X hX => by
  obtain ⟨v, hv, rfl⟩ := Finset.mem_image.1 hX
  exact h.2.1 v (Finset.mem_range.1 (hB hv))

theorem SlotsAreBlocks.card_image (h : L.SlotsAreBlocks P blk) {B : Finset ℕ}
    (hB : B ⊆ Finset.range P.blocks.length) : (B.image blk).card = B.card :=
  Finset.card_image_of_injOn (h.1.mono (Finset.coe_subset.2 hB))

/-- Every set of the last level's blocks is the image of the slots whose blocks it holds. -/
theorem SlotsAreBlocks.image_filter (h : L.SlotsAreBlocks P blk) {T : Finset (Finset G)} (hT : T ⊆ L.blocks) :
    ((Finset.range P.blocks.length).filter (blk · ∈ T)).image blk = T := by
  ext X
  simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_range]
  constructor
  · rintro ⟨v, ⟨-, hv⟩, rfl⟩
    exact hv
  · intro hX
    obtain ⟨v, hv, rfl⟩ := h.2.2 X (hT hX)
    exact ⟨v, ⟨hv, hX⟩, rfl⟩

open Classical in
theorem faulty_subset_range (H : Claims.Msg → Claims.Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement)
    (w : Wit) : faulty H Cm s P w ⊆ Finset.range P.blocks.length := by
  intro v hv
  unfold faulty at hv
  exact (Finset.mem_filter.1 hv).1

/-- **The incorrect set goes to a set of the last level's blocks.** -/
theorem SlotsAreBlocks.faulty_image_subset (h : L.SlotsAreBlocks P blk) (H : Claims.Msg → Claims.Msg)
    (Cm : CommitmentScheme Bool) (s : AuditStatement) (w : Wit) : (faulty H Cm s P w).image blk ⊆ L.blocks :=
  h.image_subset (faulty_subset_range H Cm s w)

/-- **The incorrect set's image has its size**, so the count form `RCount` reads the same on blocks. -/
theorem SlotsAreBlocks.card_faulty_image (h : L.SlotsAreBlocks P blk) (H : Claims.Msg → Claims.Msg)
    (Cm : CommitmentScheme Bool) (s : AuditStatement) (w : Wit) :
    ((faulty H Cm s P w).image blk).card = (faulty H Cm s P w).card :=
  h.card_image (faulty_subset_range H Cm s w)

theorem missProb_ne_top (L : Levels G) (T : Finset (Finset G)) : L.design.missProb T ≠ ⊤ :=
  ne_top_of_le_ne_top ENNReal.one_ne_top (SamplingDesign.missProb_le_one _ _)

/-- **The image of a set of slots is missed with the claims design's acceptance probability.** -/
theorem MissesAsDesign.missProb_image {d : Design} (hm : L.MissesAsDesign d P blk) {B : Finset ℕ}
    (hB : B ⊆ Finset.range P.blocks.length) : L.design.missProb (B.image blk) = ENNReal.ofReal (d.Pa P B) := by
  rw [← hm B hB, ENNReal.ofReal_toReal (L.missProb_ne_top _)]

/-- **The incorrect set's image is missed with its acceptance probability** under the claims design `d` on `P`. -/
theorem MissesAsDesign.missProb_faulty {d : Design} (hm : L.MissesAsDesign d P blk)
    (H : Claims.Msg → Claims.Msg) (Cm : CommitmentScheme Bool) (s : AuditStatement) (w : Wit) :
    L.design.missProb ((faulty H Cm s P w).image blk) = ENNReal.ofReal (d.Pa P (faulty H Cm s P w)) :=
  hm.missProb_image (faulty_subset_range H Cm s w)

/-- **The sampling relation at `δ` on blocks**: the incorrect set is in the confidence set `M_δ` of the statement's
design exactly when `L.design` misses its image with probability above `δ`. -/
theorem MissesAsDesign.relS_iff {H : Claims.Msg → Claims.Msg} {Cm : CommitmentScheme Bool} {s : AuditStatement}
    (hm : L.MissesAsDesign s.design P blk) (δ : ℝ) (w : Wit) :
    RelS H Cm δ s P w ↔ δ < (L.design.missProb ((faulty H Cm s P w).image blk)).toReal := by
  rw [hm _ (faulty_subset_range H Cm s w)]
  exact Iff.rfl

/-- At one level, the levels' sample misses the slots as `d` does exactly when the level's design does. -/
theorem missesAsDesign_one {l : Level G} (hL : L.levels = [l]) (d : Design) :
    L.MissesAsDesign d P blk ↔ ∀ B ⊆ Finset.range P.blocks.length,
      ((l.design Finset.univ).missProb (B.image blk)).toReal = d.Pa P B := by
  unfold MissesAsDesign
  simp only [L.missProb_one hL]

open Classical in
/-- **The count at `δ` over slots is `Levels.countAt`**: where `blk` places `P`'s proof blocks as `L`'s last level's
blocks and `L`'s sample misses them as the claims design `d` on `P` does, the largest set of the last level's blocks
missed with probability at least `δ` has as many blocks as the largest set of slots `d` accepts with probability at
least `δ`. -/
theorem countAt_eq_slots {d : Design} (h : L.SlotsAreBlocks P blk) (hm : L.MissesAsDesign d P blk) (δ : ℝ≥0∞) :
    L.countAt δ =
      ((Finset.range P.blocks.length).powerset.filter fun B => δ ≤ ENNReal.ofReal (d.Pa P B)).sup Finset.card := by
  unfold countAt
  refine le_antisymm (Finset.sup_le fun T hT => ?_) (Finset.sup_le fun B hB => ?_)
  · obtain ⟨hT, hδ⟩ := Finset.mem_filter.1 hT
    have hT := Finset.mem_powerset.1 hT
    set B := (Finset.range P.blocks.length).filter (blk · ∈ T)
    have hB : B ⊆ Finset.range P.blocks.length := Finset.filter_subset _ _
    have himg : B.image blk = T := h.image_filter hT
    refine le_trans ?_ (Finset.le_sup (f := Finset.card) (Finset.mem_filter.2 ⟨Finset.mem_powerset.2 hB, ?_⟩))
    · rw [← himg, h.card_image hB]
    · rw [← hm.missProb_image hB, himg]
      exact hδ
  · obtain ⟨hB, hδ⟩ := Finset.mem_filter.1 hB
    have hB := Finset.mem_powerset.1 hB
    rw [← h.card_image hB]
    refine Finset.le_sup (f := Finset.card) (Finset.mem_filter.2 ⟨Finset.mem_powerset.2 (h.image_subset hB), ?_⟩)
    rw [hm.missProb_image hB]
    exact hδ

end Levels

end Verity
