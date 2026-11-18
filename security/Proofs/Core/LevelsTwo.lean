module

public import Proofs.Core.Levels
public import Definitions.Claims.SamplingTwo

/-!
# Two levels agree with the two-stage design

`Claims.Sampling.Design.Two` gives its first stage only by its acceptance probability `P_a` (`Design.Pa`, a formula
with no coins) and its second stage as a uniform `min(k_t, n_{ρ,t})`-subset of each cell, read through its escape
`e_ρ` (`Design.Two.esc`). Its acceptance probability at a footprint is `P̄_a = Σ_T P_a(T)·∏_{ρ∈T}(1 − e_ρ)·∏_{ρ∉T} e_ρ`
(`Design.Two.Pa`).

`Levels.missProb_two_toReal`: for two levels whose outer blocks are the replay blocks (`ρ`, a bijection onto their
indices), whose first design misses each set of outer blocks with the probability `P_a` gives it, and whose design in
each outer block `X` misses `T_X` with the escape `e_{ρ(X)}` at the footprint `fp`, the level-by-level sample misses `T`
with probability `P̄_a(fp)`. So `Design.Two.Pa` is the miss of `Levels.design` wherever the levels realize its two
stages.
-/

@[expose] public section

namespace Verity

open Game Claims.Sampling
open scoped ENNReal

theorem SamplingDesign.missProb_le_one {U : Type} (D : SamplingDesign U) (T : Finset U) : D.missProb T ≤ 1 := by
  classical
  unfold missProb prCoin
  refine (avg_mono fun ω => ?_).trans (avg_const (C := D.Ω) 1).le
  split_ifs <;> simp

namespace Levels

variable {G : Type} [Fintype G] [DecidableEq G]

/-- **Two levels agree with `Design.Two`** where they realize its two stages: the outer blocks are its replay blocks
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
    simp only [Finset.mem_coe, Finset.mem_powerset] at hR ⊢
    intro i hi
    obtain ⟨X, hX, rfl⟩ := Finset.mem_image.1 hi
    simpa using hρ.mapsTo (Finset.mem_coe.2 (hR hX))
  · intro S _
    simp only [Finset.mem_coe, Finset.mem_powerset]
    exact Finset.filter_subset _ _
  · intro R hR
    simp only [Finset.mem_coe, Finset.mem_powerset] at hR
    ext X
    simp only [Finset.mem_filter, Finset.mem_image]
    constructor
    · rintro ⟨hXA, Y, hYR, hY⟩
      exact hinj (Finset.mem_coe.2 (hR hYR)) (Finset.mem_coe.2 hXA) hY ▸ hYR
    · intro hXR
      exact ⟨hR hXR, X, hXR, rfl⟩
  · intro S hS
    simp only [Finset.mem_coe, Finset.mem_powerset] at hS
    ext i
    simp only [Finset.mem_image, Finset.mem_filter]
    constructor
    · rintro ⟨X, ⟨-, hX⟩, rfl⟩
      exact hX
    · intro hi
      obtain ⟨X, hXA, rfl⟩ := hρ.surjOn (by simpa using hS hi)
      exact ⟨X, ⟨Finset.mem_coe.1 hXA, hi⟩, rfl⟩
  · intro R hR
    simp only [Finset.mem_coe, Finset.mem_powerset] at hR
    rw [h₁ R hR, ← himage_sdiff R hR, Finset.prod_image (hinj.mono (Finset.coe_subset.2 hR)),
      Finset.prod_image (hinj.mono (Finset.coe_subset.2 Finset.sdiff_subset))]
    congr 2
    · exact Finset.prod_congr rfl fun X hX => by rw [h₂ X (hR hX)]
    · exact Finset.prod_congr rfl fun X hX => by rw [h₂ X (Finset.mem_sdiff.1 hX).1]

end Levels

end Verity
