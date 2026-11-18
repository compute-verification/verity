module

public import Proofs.Core.LevelsCount
public import Proofs.Core.LevelsTwo
public import Proofs.Core.Game.Prob
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
* `Levels.missesAsDesign_all`: the first instance. One level whose design draws every block, against a claims design
  whose every rule is `all` (`classesPa_all`): both miss a set of slots with probability `1` when it is empty and `0`
  otherwise. `missesAsDesign_all_design`: the same at a level whose design is core's `SamplingDesign.all` of its
  blocks, the design the claims' `all` rule is a product of (`Claims.Sampling.Design.Pa_all_prod`).
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

/-- `window o n B` is empty exactly when `B` has no member in `[o, o + n)`. -/
theorem window_eq_empty_iff (o n : ℕ) (B : Finset ℕ) :
    window o n B = ∅ ↔ ∀ v ∈ B, ¬ (o ≤ v ∧ v < o + n) := by
  unfold window
  rw [Finset.image_eq_empty, Finset.filter_eq_empty_iff]

open Classical in
/-- **Every class sampled `all`**: the classes `cs` from slot `o` accept `B` with probability `1` when `B` has no
member among their slots and `0` otherwise. -/
theorem classesPa_all : ∀ (cs : List (List (List (List ℕ)))) (o : ℕ) (B : Finset ℕ),
    classesPa (cs.map fun _ => OneStage.all) cs o B =
      if ∀ v ∈ B, ¬ (o ≤ v ∧ v < o + cs.flatten.flatten.length) then 1 else 0
  | [], o, B => by
    rw [if_pos fun v _ h => by simp only [List.flatten_nil, List.length_nil, Nat.add_zero] at h; omega]
    rfl
  | c :: cs, o, B => by
    have hlen : (c :: cs).flatten.flatten.length = c.flatten.length + cs.flatten.flatten.length := by
      simp [List.flatten_cons, List.flatten_append]
    have hE : (∀ v ∈ B, ¬ (o ≤ v ∧ v < o + (c :: cs).flatten.flatten.length)) ↔
        (∀ v ∈ B, ¬ (o ≤ v ∧ v < o + c.flatten.length)) ∧
          ∀ v ∈ B, ¬ (o + c.flatten.length ≤ v ∧ v < o + c.flatten.length + cs.flatten.flatten.length) := by
      rw [hlen]
      constructor
      · intro h
        exact ⟨fun v hv hc => h v hv ⟨hc.1, by omega⟩, fun v hv hc => h v hv ⟨by omega, by omega⟩⟩
      · rintro ⟨h1, h2⟩ v hv hc
        by_cases hm : v < o + c.flatten.length
        · exact h1 v hv ⟨hc.1, hm⟩
        · exact h2 v hv ⟨by omega, by omega⟩
    simp only [List.map_cons, classesPa, OneStage.Pa]
    rw [classesPa_all cs]
    by_cases h1 : ∀ v ∈ B, ¬ (o ≤ v ∧ v < o + c.flatten.length)
    · rw [if_pos ((window_eq_empty_iff _ _ _).2 h1), one_mul]
      by_cases h2 :
          ∀ v ∈ B, ¬ (o + c.flatten.length ≤ v ∧ v < o + c.flatten.length + cs.flatten.flatten.length)
      · rw [if_pos h2, if_pos (hE.2 ⟨h1, h2⟩)]
      · rw [if_neg h2, if_neg fun h => h2 (hE.1 h).2]
    · rw [if_neg fun h => h1 ((window_eq_empty_iff _ _ _).1 h), zero_mul, if_neg fun h => h1 (hE.1 h).1]

/-- **The first instance: a level that draws every block misses the slots as the claims design that samples every
class `all`.** One level whose design at the whole circuit draws every block, with `blk` placing `P`'s proof blocks as
its blocks, against a claims design whose rule for each class of `P` is `all`. -/
theorem missesAsDesign_all {l : Level G} (hL : L.levels = [l]) (h : L.SlotsAreBlocks P blk)
    (hall : ∀ ω, (l.design Finset.univ).draw ω = l.blocks.parts) {d : Design}
    (hd : d.rules = P.classes.map fun _ => OneStage.all) : L.MissesAsDesign d P blk := by
  classical
  intro B hB
  have hblocks : L.blocks = l.blocks.parts := by simp [Levels.blocks, hL]
  have hdis : ∀ ω, Disjoint ((l.design Finset.univ).draw ω) (B.image blk) ↔ B = ∅ := fun ω => by
    rw [hall ω]
    constructor
    · intro hdj
      by_contra hB0
      obtain ⟨v, hv⟩ := Finset.nonempty_iff_ne_empty.2 hB0
      have hmem := h.2.1 v (Finset.mem_range.1 (hB hv))
      rw [hblocks] at hmem
      exact Finset.disjoint_left.1 hdj hmem (Finset.mem_image_of_mem blk hv)
    · rintro rfl
      simp
  have hPa : d.Pa P B = if B = ∅ then 1 else 0 := by
    unfold Design.Pa
    rw [hd, classesPa_all]
    by_cases hB0 : B = ∅
    · rw [if_pos (fun v hv => by simp [hB0] at hv), if_pos hB0]
    · obtain ⟨v, hv⟩ := Finset.nonempty_iff_ne_empty.2 hB0
      have hlt : v < 0 + P.classes.flatten.flatten.length := by
        simpa [Partition.blocks] using Finset.mem_range.1 (hB hv)
      rw [if_neg (fun hc => hc v hv ⟨Nat.zero_le v, hlt⟩), if_neg hB0]
  rw [hPa, L.missProb_one hL]
  show (Game.prCoin fun ω => Disjoint ((l.design Finset.univ).draw ω) (B.image blk)).toReal = _
  by_cases hB0 : B = ∅
  · have h1 : ∀ ω, Disjoint ((l.design Finset.univ).draw ω) (B.image blk) := fun ω => (hdis ω).2 hB0
    rw [if_pos hB0]
    unfold Game.prCoin
    simp only [h1, ite_true]
    rw [Game.avg_const, ENNReal.toReal_one]
  · have h0 : ∀ ω, ¬ Disjoint ((l.design Finset.univ).draw ω) (B.image blk) := fun ω hω =>
      hB0 ((hdis ω).1 hω)
    rw [if_neg hB0, le_antisymm ((Game.prCoin_mono fun c hc => (h0 c hc).elim).trans_eq Game.prCoin_false) zero_le,
      ENNReal.toReal_zero]

/-- **The first instance at core's `all`**: `missesAsDesign_all` at a level whose design at the whole circuit is
`SamplingDesign.all` of its blocks, the one design both sides read for `all`. -/
theorem missesAsDesign_all_design {l : Level G} (hL : L.levels = [l]) (h : L.SlotsAreBlocks P blk)
    (hl : l.design Finset.univ = SamplingDesign.all l.blocks.parts) {d : Design}
    (hd : d.rules = P.classes.map fun _ => OneStage.all) : L.MissesAsDesign d P blk :=
  missesAsDesign_all hL h (by generalize l.design Finset.univ = M at hl ⊢; subst hl; exact fun _ => rfl) hd

end Levels

end Verity
