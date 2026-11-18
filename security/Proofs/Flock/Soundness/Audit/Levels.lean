module

public import Definitions.Flock.Audit.Levels

/-! # An audit's units sampled by levels: the transfer lemma, and every design as one level -/

@[expose] public section

namespace FlockSoundness.Audit.SamplingDesign

open Verity

variable {n : ℕ}

theorem missProb_ofLevels {G : Type} [Fintype G] [DecidableEq G] (Lv : Levels G) (idx : Fin n → Finset G)
    (B : Finset (Fin n)) : (ofLevels Lv idx).missProb B = Lv.design.missProb (B.image idx) := by
  show Game.prCoin (fun ω : Lv.design.Ω => Disjoint ((ofLevels Lv idx).draw ω) B) =
    Game.prCoin fun ω => Disjoint (Lv.design.draw ω) (B.image idx)
  congr 1
  funext ω
  apply propext
  rw [Finset.disjoint_left, Finset.disjoint_left]
  constructor
  · intro h b hb hbB
    obtain ⟨u, hu, rfl⟩ := Finset.mem_image.1 hbB
    exact h (Finset.mem_filter.2 ⟨Finset.mem_univ _, hb⟩) hu
  · intro h u hu huB
    exact h (Finset.mem_filter.1 hu).2 (Finset.mem_image_of_mem idx huB)

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
