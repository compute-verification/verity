module

public import Definitions.Flock.Audit.Levels

/-! # An audit's units sampled by levels: the transfer lemma, and every design as one level -/

@[expose] public section

namespace FlockSoundness.Audit.SamplingDesign

open Verity

variable {n : ℕ}

theorem disjoint_ofLevels {G : Type} [Fintype G] [DecidableEq G] (Lv : Levels G) (idx : Fin n → Finset G)
    (ω : (ofLevels Lv idx).Ω) (B : Finset (Fin n)) :
    Disjoint ((ofLevels Lv idx).draw ω) B ↔ Disjoint (Lv.design.draw ω) (B.image idx) := by
  rw [Finset.disjoint_left, Finset.disjoint_left]
  constructor
  · intro h b hb hbB
    obtain ⟨u, hu, rfl⟩ := Finset.mem_image.1 hbB
    exact h (Finset.mem_filter.2 ⟨Finset.mem_univ _, hb⟩) hu
  · intro h u hu huB
    exact h (Finset.mem_filter.1 hu).2 (Finset.mem_image_of_mem idx huB)

theorem missProb_ofLevels {G : Type} [Fintype G] [DecidableEq G] (Lv : Levels G) (idx : Fin n → Finset G)
    (B : Finset (Fin n)) : (ofLevels Lv idx).missProb B = Lv.design.missProb (B.image idx) :=
  congrArg Game.prCoin (funext fun ω => propext (disjoint_ofLevels Lv idx ω B))

theorem miss_ofLevels {G : Type} [Fintype G] [DecidableEq G] (Lv : Levels G) (idx : Fin n → Finset G) (K : ℕ) :
    (ofLevels Lv idx).miss K = ⨆ (B : Finset (Fin n)) (_ : K ≤ B.card), Lv.design.missProb (B.image idx) := by
  simp only [Verity.SamplingDesign.miss, missProb_ofLevels]

theorem miss_ofLevels_le {G : Type} [Fintype G] [DecidableEq G] (Lv : Levels G) {idx : Fin n → Finset G}
    (hidx : Function.Injective idx) (K : ℕ) : (ofLevels Lv idx).miss K ≤ Lv.design.miss K := by
  rw [miss_ofLevels]
  exact iSup₂_le fun B hB => le_iSup₂_of_le (f := fun (B : Finset (Finset G)) (_ : K ≤ B.card) =>
    Lv.design.missProb B) (B.image idx) (by rwa [Finset.card_image_of_injective _ hidx]) le_rfl

/-- `L` as one level of singleton blocks. -/
def toLevels (L : SamplingDesign n) : Levels (Fin n) where
  levels := [⟨⊥, fun _ => { Ω := L.Ω, draw := fun ω => (L.draw ω).map ⟨singleton, Finset.singleton_injective⟩ }⟩]
  nested := by
    simp only [Level.NestedBelow, and_true]
    exact fun b _ => ⟨_, Finset.mem_singleton_self _, Finset.subset_univ _⟩
  inside := by
    simp only [Level.InsideBelow, and_true]
    intro X hX ω b hb
    obtain ⟨u, -, rfl⟩ := Finset.mem_map.1 hb
    rw [Finset.mem_singleton.1 hX]
    exact ⟨Finpartition.mem_bot_iff.2 ⟨u, Finset.mem_univ _, rfl⟩, Finset.subset_univ _⟩

theorem ofLevels_toLevels (L : SamplingDesign n) : ofLevels L.toLevels (fun u => {u}) = L := by
  obtain ⟨Ω, draw⟩ := L
  show Verity.SamplingDesign.mk Ω
    (fun ω => Finset.univ.filter fun u => {u} ∈ (draw ω).map ⟨singleton, Finset.singleton_injective⟩) = _
  congr 1
  funext ω
  ext u
  simp

theorem missProb_toLevels (L : SamplingDesign n) (B : Finset (Fin n)) :
    L.toLevels.design.missProb (B.image fun u => {u}) = L.missProb B := by
  rw [← missProb_ofLevels, ofLevels_toLevels]

end FlockSoundness.Audit.SamplingDesign
