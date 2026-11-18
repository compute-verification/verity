module

public import Definitions.Core.LevelsKinds
public import Proofs.Core.Draw
public import Proofs.Core.Levels

/-!
# The first level drawn by kind: proof

Taking for the first level's design a product of designs by kind whose units read as that level's blocks keeps both
conditions of levels: nesting reads only the blocks, the first level's condition reads its design only at the whole
circuit, and there the new design draws only blocks of the first level (`Levels.exists_firstByKind`). At one level the sample is that design, and read through an
injective `blockOf` its miss probability is the product's, the product over kinds (`Levels.missProb_firstByKind_one`).
-/

@[expose] public section

namespace Verity

namespace SamplingDesign

/-- Read through an injective `f`, a design misses the image of a set exactly when it misses the set. -/
theorem missProb_map_image {U V : Type} [DecidableEq V] {f : U → V} (hf : Function.Injective f) (D : SamplingDesign U)
    (S : Finset U) : (D.map f).missProb (S.image f) = D.missProb S := by
  show Game.prCoin (fun ω : D.Ω => Disjoint ((D.draw ω).image f) (S.image f)) =
    Game.prCoin fun ω : D.Ω => Disjoint (D.draw ω) S
  simp only [Finset.disjoint_image hf]

end SamplingDesign

namespace Levels

variable {G : Type} [Fintype G] [DecidableEq G]

/-- **Any levels may have their first level drawn by kind**: for designs `D k` by kind whose units read as blocks of
the first level, there are levels with the same first blocks and the same later levels whose first level's design at
the whole circuit is their product. -/
theorem exists_firstByKind (L : Levels G) {l : Level G} {ls : List (Level G)} (hL : L.levels = l :: ls)
    {K : Type} [Fintype K] [DecidableEq K] {V : K → Type} (blockOf : (k : K) → V k → Finset G)
    (D : (k : K) → SamplingDesign (V k)) (hb : ∀ k u, blockOf k u ∈ l.blocks.parts) :
    ∃ L' : Levels G, L'.FirstByKind blockOf D ∧ ∃ l', L'.levels = l' :: ls ∧ l'.blocks = l.blocks := by
  let P := (SamplingDesign.prod D).map fun u : (k : K) × V k => blockOf u.1 u.2
  let l' : Level G := { blocks := l.blocks, design := fun _ => P }
  have hn : Level.NestedBelow {Finset.univ} (l :: ls) := hL ▸ L.nested
  have hi : Level.InsideBelow {Finset.univ} (l :: ls) := hL ▸ L.inside
  simp only [Level.NestedBelow, Level.InsideBelow] at hn hi
  have hn' : Level.NestedBelow {Finset.univ} (l' :: ls) := by
    simp only [Level.NestedBelow]
    exact hn
  have hi' : Level.InsideBelow {Finset.univ} (l' :: ls) := by
    simp only [Level.InsideBelow]
    refine ⟨fun X hX ω b hbX => ?_, hi.2⟩
    rw [Finset.mem_singleton] at hX
    subst hX
    simp only [l', P, SamplingDesign.map, SamplingDesign.prod, Finset.mem_image, Finset.mem_sigma] at hbX
    obtain ⟨⟨k, u⟩, -, rfl⟩ := hbX
    exact ⟨hb k u, Finset.subset_univ _⟩
  exact ⟨⟨l' :: ls, hn', hi'⟩, ⟨l', ls, rfl, rfl⟩, l', rfl, rfl⟩

/-- **Drawn by kind, the first level misses as drawn by kind.** -/
theorem FirstByKind.missesByKind {L : Levels G} {K : Type} [Fintype K] [DecidableEq K] {V : K → Type}
    {blockOf : (k : K) → V k → Finset G} {D : (k : K) → SamplingDesign (V k)} (h : L.FirstByKind blockOf D) :
    L.FirstMissesByKind blockOf D := by
  obtain ⟨l, ls, hL, hd⟩ := h
  exact ⟨l, ls, hL, fun T => by rw [hd]⟩

/-- **At one level that misses as drawn by kind, a set's miss probability is the product over kinds**: for `blockOf`
injective, the sample misses the blocks of `⋃ₖ {k} × T k` with probability `∏ₖ missProb (D k) (T k)`. -/
theorem missProb_firstMissesByKind_one (L : Levels G) {l : Level G} (hL : L.levels = [l]) {K : Type} [Fintype K]
    [DecidableEq K] {V : K → Type} {blockOf : (k : K) → V k → Finset G} {D : (k : K) → SamplingDesign (V k)}
    (h : L.FirstMissesByKind blockOf D) (hinj : Function.Injective fun u : (k : K) × V k => blockOf u.1 u.2)
    (T : (k : K) → Finset (V k)) :
    L.design.missProb ((Finset.univ.sigma T).image fun u => blockOf u.1 u.2) = ∏ k, (D k).missProb (T k) := by
  obtain ⟨l₀, ls, h₀, hd⟩ := h
  obtain ⟨rfl, -⟩ := List.cons.inj (hL.symm.trans h₀)
  rw [L.design_one hL, hd, SamplingDesign.missProb_map_image hinj, SamplingDesign.missProb_prod]

/-- **At one level drawn by kind, a set's miss probability is the product over kinds** (`missProb_firstMissesByKind_one`
at `FirstByKind.missesByKind`). -/
theorem missProb_firstByKind_one (L : Levels G) {l : Level G} (hL : L.levels = [l]) {K : Type} [Fintype K]
    [DecidableEq K] {V : K → Type} {blockOf : (k : K) → V k → Finset G} {D : (k : K) → SamplingDesign (V k)}
    (h : L.FirstByKind blockOf D) (hinj : Function.Injective fun u : (k : K) × V k => blockOf u.1 u.2)
    (T : (k : K) → Finset (V k)) :
    L.design.missProb ((Finset.univ.sigma T).image fun u => blockOf u.1 u.2) = ∏ k, (D k).missProb (T k) :=
  L.missProb_firstMissesByKind_one hL h.missesByKind hinj T

end Levels

end Verity
