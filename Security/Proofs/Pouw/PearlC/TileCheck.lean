import Definitions.Pouw.PearlC.TileCheck

/-!
# L3 from its parts, and on a transcript

`TileCheck.computes_of`: L3 is its cap and its words; the tile's weight rows are the check's own.
`TileCheck.mem_of_computes`: under L3, a tile whose unit `τ` commits as `c`, with a witness its check accepts, is in
`goodTiles ∪ blankTiles`, or the registered values fail to bind the circuit's words; `TileGame.good_of_accept`'s
conclusion when `Collides` is in the proof system's bad event.
-/

namespace Pouw.PearlC

variable {Q R S : Type}

theorem tileGood_iff (P : Protocol Q R S) (TR : TileRules Q R S) (U : Workload) (H : Q → R) (s : S) (g : ℕ)
    (c : Commit) :
    TileGood P TR U H s g c ↔
      TR.capOK H s U g c.act ∧
        (∀ j ∈ (TR.tiling U.layout).cols g,
          P.weightOK (U.layout.shape ((TR.tiling U.layout).unit g))
            (U.weight (U.layout.wid ((TR.tiling U.layout).unit g))) j) ∧
        TileWordsGood P TR U H s g c :=
  Iff.rfl

theorem TileCheck.computes_of (TC : TileCheck Q R S) (P : Protocol Q R S) (TR : TileRules Q R S)
    (D : UnitLayout → Prop) (hc : TC.ComputesCap P TR D) (hw : TC.ComputesWords P TR D) : TC.Computes P TR D := by
  intro U hU H s g hg c X hX
  rcases hw U hU H s g hg c X hX with h | h
  · exact Or.inl ⟨hc U hU H s g hg c X hX, hX.2.2.2.2, h⟩
  · exact Or.inr h

theorem TileCheck.mem_of_computes (TC : TileCheck Q R S) (P : Protocol Q R S) (TR : TileRules Q R S)
    (D : UnitLayout → Prop) (h : TC.Computes P TR D) (U : Workload) (hU : U.InDomain P D) (H : Q → R) (s : S)
    (τ : CommitLog) (g : ℕ) (hg : g < (TR.tiling U.layout).NT) (c : Commit)
    (hc : τ ((TR.tiling U.layout).unit g) = some c)
    (X : (TC.C (U.layout.shape ((TR.tiling U.layout).unit g))).Transcript) (hX : TC.Accepts P TR H s U g c X) :
    g ∈ goodTiles P TR U H s τ ∪ blankTiles TR U τ ∨
      TC.Collides U ((TR.tiling U.layout).unit g) ((TR.tiling U.layout).rows g) ((TR.tiling U.layout).cols g) c X := by
  rcases h U hU H s g hg c X hX with h | h
  · left
    simp only [Finset.mem_union, goodTiles, Finset.mem_filter, Finset.mem_range]
    exact Or.inl ⟨hg, c, hc, h⟩
  · exact Or.inr h

/-- A tile of a blank unit is blank. -/
theorem TileCheck.mem_blank (TR : TileRules Q R S) (U : Workload) (τ : CommitLog) (g : ℕ)
    (hg : g < (TR.tiling U.layout).NT) (hb : τ ((TR.tiling U.layout).unit g) = none) : g ∈ blankTiles TR U τ := by
  simp only [blankTiles, Finset.mem_filter, Finset.mem_range]
  exact ⟨hg, hb⟩

/-- The registered values of a committed unit are its commitment's. -/
theorem TileCheck.valueAt_some (TC : TileCheck Q R S) (U : Workload) (τ : CommitLog) (u : ℕ) (c : Commit)
    (hc : τ u = some c) : TC.valueAt U τ u = TC.rowsOf U u c := by
  simp only [TileCheck.valueAt, hc]

end Pouw.PearlC
