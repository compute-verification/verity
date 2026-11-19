module

public import Proofs.Core.Levels
public import Proofs.Core.Draw
public import Definitions.Claims.SamplingTwo

/-!
# Two levels agree with `Design.Two`

`Claims.Sampling.Design.Two` gives its outer sample only by its acceptance probability `P_a` (`Design.Pa`, a formula
with no coins) and its inner sample as a uniform `min(k_t, n_{ρ,t})`-subset of each cell, read through its escape
`e_ρ` (`Design.Two.esc`). Its acceptance probability at a footprint is `P̄_a = Σ_T P_a(T)·∏_{ρ∈T}(1 − e_ρ)·∏_{ρ∉T} e_ρ`
(`Design.Two.Pa`).

`Levels.missProb_two_toReal`: for two levels whose outer blocks are the replay blocks (`ρ`, a bijection onto their
indices), whose first design misses each set of outer blocks with the probability `P_a` gives it, and whose design in
each outer block `X` misses `T_X` with the escape `e_{ρ(X)}` at the footprint `fp`, the level-by-level sample misses `T`
with probability `P̄_a(fp)`. So `Design.Two.Pa` is the miss of `Levels.design` wherever the levels realize its outer
and inner samples; `Levels.missProb_two_toReal_cells` takes the inner designs to be its uniform subsets of each cell.
-/

@[expose] public section

namespace Verity

open Game Claims.Sampling
open scoped ENNReal

/-- A uniform `min(k_t, |C_t|)`-subset of each cell `C_t`, `t < m`, each on coins of its own: `Design.Two`'s inner
sample in one outer block. -/
def SamplingDesign.cellSubsets {U : Type} [DecidableEq U] (m : ℕ) (C : ℕ → Finset U) (k : ℕ → ℕ) :
    SamplingDesign U where
  Ω := (t : Fin m) → {S // S ∈ (C t).powersetCard (min (k t) (C t).card)}
  nonempty := ⟨fun t =>
    let h := Finset.exists_subset_card_eq (min_le_right (k t) (C t).card)
    ⟨h.choose, Finset.mem_powersetCard.2 h.choose_spec⟩⟩
  draw ω := Finset.univ.biUnion fun t => (ω t).1

/-- A uniform `j`-subset of `C` misses `T` with probability `C(|C| − |T ∩ C|, j)/C(|C|, j)`. -/
theorem SamplingDesign.prCoin_subset_disjoint {U : Type} [DecidableEq U] (C : Finset U) (j : ℕ) (T : Finset U) :
    prCoin (fun S : {S // S ∈ C.powersetCard j} => Disjoint S.1 T) =
      (((C.card - (T ∩ C).card).choose j : ℕ) : ℝ≥0∞) / ((C.card.choose j : ℕ) : ℝ≥0∞) := by
  classical
  have hsdiff : (C \ T).card = C.card - (T ∩ C).card := by
    have := Finset.card_sdiff_add_card_inter C T
    rw [Finset.inter_comm] at this
    omega
  have hfilter : (Finset.univ.filter fun S : {S // S ∈ C.powersetCard j} => Disjoint S.1 T).card =
      ((C \ T).powersetCard j).card := by
    refine Finset.card_bij (fun S _ => S.1) ?_ ?_ ?_
    · intro S hS
      have hS' := Finset.mem_powersetCard.1 S.2
      have hd := (Finset.mem_filter.1 hS).2
      exact Finset.mem_powersetCard.2 ⟨fun x hx => Finset.mem_sdiff.2 ⟨hS'.1 hx, Finset.disjoint_left.1 hd hx⟩,
        hS'.2⟩
    · intro S₁ _ S₂ _ h
      exact Subtype.ext h
    · intro b hb
      have hb' := Finset.mem_powersetCard.1 hb
      refine ⟨⟨b, Finset.mem_powersetCard.2 ⟨fun x hx => (Finset.mem_sdiff.1 (hb'.1 hx)).1, hb'.2⟩⟩,
        Finset.mem_filter.2 ⟨Finset.mem_univ _, Finset.disjoint_left.2 fun x hx hxT =>
          (Finset.mem_sdiff.1 (hb'.1 hx)).2 hxT⟩, rfl⟩
  rw [prCoin_eq_card, hfilter, Finset.card_powersetCard, hsdiff, Fintype.card_coe, Finset.card_powersetCard]

theorem SamplingDesign.missProb_cellSubsets {U : Type} [DecidableEq U] (m : ℕ) (C : ℕ → Finset U) (k : ℕ → ℕ)
    (T : Finset U) :
    (cellSubsets m C k).missProb T = ∏ t : Fin m,
      ((((C t).card - (T ∩ C t).card).choose (min (k t) (C t).card) : ℕ) : ℝ≥0∞) /
        (((C t).card.choose (min (k t) (C t).card) : ℕ) : ℝ≥0∞) := by
  show prCoin (fun ω : (t : Fin m) → {S // S ∈ (C t).powersetCard (min (k t) (C t).card)} =>
    Disjoint (Finset.univ.biUnion fun t => (ω t).1) T) = _
  simp only [Finset.disjoint_biUnion_left]
  haveI : ∀ t : Fin m, Nonempty {S // S ∈ (C t).powersetCard (min (k t) (C t).card)} := fun t =>
    let h := Finset.exists_subset_card_eq (min_le_right (k t) (C t).card)
    ⟨⟨h.choose, Finset.mem_powersetCard.2 h.choose_spec⟩⟩
  rw [prCoin_pi_forall Finset.univ
    (fun (t : Fin m) (S : {S // S ∈ (C t).powersetCard (min (k t) (C t).card)}) => Disjoint S.1 T)]
  exact Finset.prod_congr rfl fun t _ => prCoin_subset_disjoint (C t) _ T

namespace Levels

variable {G : Type} [Fintype G] [DecidableEq G]

/-- **Two levels agree with `Design.Two`** where they realize its outer and inner samples: the outer blocks are its replay blocks
under `ρ`, the first design's miss is `P_a` (`h₁`), and the design in each outer block misses `T`'s blocks inside it with
the escape at `fp` (`h₂`). -/
theorem missProb_two_toReal (L : Levels G) {l₁ l₂ : Level G} (hL : L.levels = [l₁, l₂]) (d : Design.Two)
    (Pt : Partition) (ρ : Finset G → ℕ)
    (hρ : Set.BijOn ρ (l₁.blocks.parts : Set (Finset G)) (Finset.range d.part.blocks.length : Set ℕ))
    (T : Finset (Finset G)) (fp : ℕ → ℕ → ℕ)
    (h₁ : ∀ R ⊆ l₁.blocks.parts, ((l₁.design Finset.univ).missProb R).toReal = d.replay.Pa d.part (R.image ρ))
    (h₂ : ∀ X ∈ l₁.blocks.parts, ((l₂.design X).missProb (T.filter (· ⊆ X))).toReal = d.esc Pt fp (ρ X)) :
    (L.design.missProb T).toReal = d.Pa Pt fp := by
  classical
  have hi := L.inside
  rw [hL] at hi
  simp only [Level.InsideBelow, Finset.mem_singleton, forall_eq, and_true] at hi
  obtain ⟨hin, -⟩ := hi
  have key := SamplingDesign.avg_prod_draw (l₁.design Finset.univ) l₁.blocks.parts
    (fun ω b hb => (hin ω b hb).1) (fun X => (l₂.design X).missProb (T.filter (· ⊆ X)))
    (fun X _ => SamplingDesign.missProb_le_one _ _)
  rw [L.missProb_two hL, key]
  have hfin : ∀ X, (l₂.design X).missProb (T.filter (· ⊆ X)) ≠ ⊤ := fun X =>
    ne_top_of_le_ne_top ENNReal.one_ne_top (SamplingDesign.missProb_le_one _ _)
  rw [ENNReal.toReal_sum fun R _ => ENNReal.mul_ne_top
    (ne_top_of_le_ne_top ENNReal.one_ne_top (SamplingDesign.missProb_le_one _ _))
    (ENNReal.mul_ne_top (ENNReal.prod_ne_top fun X _ => ENNReal.sub_ne_top ENNReal.one_ne_top)
      (ENNReal.prod_ne_top fun X _ => hfin X))]
  simp only [ENNReal.toReal_mul, ENNReal.toReal_prod]
  have hsub : ∀ X, (1 - (l₂.design X).missProb (T.filter (· ⊆ X))).toReal =
      1 - ((l₂.design X).missProb (T.filter (· ⊆ X))).toReal := fun X => by
    rw [ENNReal.toReal_sub_of_le (SamplingDesign.missProb_le_one _ _) ENNReal.one_ne_top, ENNReal.toReal_one]
  simp only [hsub]
  unfold Design.Two.Pa
  set A := l₁.blocks.parts
  set n := d.part.blocks.length
  have hinj : Set.InjOn ρ (A : Set (Finset G)) := hρ.injOn
  have himage_sdiff : ∀ R ⊆ A, (A \ R).image ρ = Finset.range n \ R.image ρ := by
    intro R hR
    ext i
    simp only [Finset.mem_image, Finset.mem_sdiff, Finset.mem_range]
    constructor
    · rintro ⟨X, ⟨hXA, hXR⟩, rfl⟩
      refine ⟨by simpa using hρ.mapsTo (Finset.mem_coe.2 hXA), ?_⟩
      rintro ⟨Y, hYR, hY⟩
      exact hXR (hinj (Finset.mem_coe.2 (hR hYR)) (Finset.mem_coe.2 hXA) hY ▸ hYR)
    · rintro ⟨hi, hnot⟩
      obtain ⟨X, hXA, rfl⟩ := hρ.surjOn (by simpa using hi)
      exact ⟨X, ⟨Finset.mem_coe.1 hXA, fun hXR => hnot ⟨X, hXR, rfl⟩⟩, rfl⟩
  refine Finset.sum_nbij' (fun R => R.image ρ) (fun S => A.filter fun X => ρ X ∈ S) ?_ ?_ ?_ ?_ ?_
  · intro R hR
    simp only [Finset.mem_powerset] at hR ⊢
    intro i hi
    obtain ⟨X, hX, rfl⟩ := Finset.mem_image.1 hi
    simpa using hρ.mapsTo (Finset.mem_coe.2 (hR hX))
  · intro S _
    simp only [Finset.mem_powerset]
    exact Finset.filter_subset _ _
  · intro R hR
    simp only [Finset.mem_powerset] at hR
    ext X
    simp only [Finset.mem_filter, Finset.mem_image]
    constructor
    · rintro ⟨hXA, Y, hYR, hY⟩
      exact hinj (Finset.mem_coe.2 (hR hYR)) (Finset.mem_coe.2 hXA) hY ▸ hYR
    · intro hXR
      exact ⟨hR hXR, X, hXR, rfl⟩
  · intro S hS
    simp only [Finset.mem_powerset] at hS
    ext i
    simp only [Finset.mem_image, Finset.mem_filter]
    constructor
    · rintro ⟨X, ⟨-, hX⟩, rfl⟩
      exact hX
    · intro hi
      obtain ⟨X, hXA, rfl⟩ := hρ.surjOn (by simpa using hS hi)
      exact ⟨X, ⟨Finset.mem_coe.1 hXA, hi⟩, rfl⟩
  · intro R hR
    simp only [Finset.mem_powerset] at hR
    rw [h₁ R hR, ← himage_sdiff R hR, Finset.prod_image (hinj.mono (Finset.coe_subset.2 hR)),
      Finset.prod_image (hinj.mono (Finset.coe_subset.2 Finset.sdiff_subset))]
    congr 2
    · exact Finset.prod_congr rfl fun X hX => by rw [h₂ X (hR hX)]
    · exact Finset.prod_congr rfl fun X hX => by rw [h₂ X (Finset.mem_sdiff.1 hX).1]

/-- **Two levels agree with `Design.Two` at its uniform inner subsets**: as `missProb_two_toReal`, with the design in each outer
block `X` a uniform `min(k_t, n_{ρ(X),t})`-subset of each of its cells `C X t`, which have `Design.Two`'s cell sizes
and hold `fp(ρ(X), t)` blocks of `T`. -/
theorem missProb_two_toReal_cells (L : Levels G) {l₁ l₂ : Level G} (hL : L.levels = [l₁, l₂]) (d : Design.Two)
    (Pt : Partition) (ρ : Finset G → ℕ)
    (hρ : Set.BijOn ρ (l₁.blocks.parts : Set (Finset G)) (Finset.range d.part.blocks.length : Set ℕ))
    (T : Finset (Finset G)) (fp : ℕ → ℕ → ℕ)
    (h₁ : ∀ R ⊆ l₁.blocks.parts, ((l₁.design Finset.univ).missProb R).toReal = d.replay.Pa d.part (R.image ρ))
    (C : Finset G → ℕ → Finset (Finset G))
    (hdesign : ∀ X ∈ l₁.blocks.parts,
      l₂.design X = SamplingDesign.cellSubsets Pt.strata.length (C X) fun t => d.k.getD t 0)
    (hcard : ∀ X ∈ l₁.blocks.parts, ∀ t < Pt.strata.length, (C X t).card = (cell Pt d.part (ρ X) t).card)
    (hcount : ∀ X ∈ l₁.blocks.parts, ∀ t < Pt.strata.length, (T.filter (· ⊆ X) ∩ C X t).card = fp (ρ X) t) :
    (L.design.missProb T).toReal = d.Pa Pt fp := by
  refine L.missProb_two_toReal hL d Pt ρ hρ T fp h₁ fun X hX => ?_
  rw [hdesign X hX, SamplingDesign.missProb_cellSubsets, ENNReal.toReal_prod]
  unfold Design.Two.esc
  rw [← Fin.prod_univ_eq_prod_range]
  refine Finset.prod_congr rfl fun t _ => ?_
  rw [ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast, hcard X hX t t.2, hcount X hX t t.2]
  rfl

end Levels

end Verity
