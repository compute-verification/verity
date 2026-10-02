import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Order.Interval.Finset.Defs
import Mathlib.Data.Int.Interval

/-!
# Lifting MinRank: grid density (L3) and slices (L2)

Two self-contained lemmas of `Lift_MinRank` (`internal/pouw/new-crypto/lifting/README.md`).

* **L3 (grid density).** Points with every coordinate in a finite set `G` whose pairwise differences lie in a
  subspace `V` number at most `|G| ^ finrank V`. On the noise grid `{−31, …, 32}^ι` a set inside an affine subspace
  of codimension `r` has at most `64 ^ (|ι| − r)` points.
* **L2 (slices), in the form used.** If every functional of a subspace `W` of `ι → ℚ` (acting by the dot product)
  is constant on a set of grid points, that set has at most `64 ^ (|ι| − finrank W)` points. For a fixed `F₁`, `W`
  is the image `Ψ_f(C)`, so the slice `Z_f` has at most `64 ^ (m·k − rank Ψ_f)` points.
-/

namespace Pouw.Dimension.Lifting

open Finset Module

section Density

variable {ι K : Type*} [Fintype ι] [Field K]

/-- **L3.** Points with coordinates in `G` and pairwise differences in `V`, with `finrank V ≤ n`, number at most
`|G| ^ n`. -/
theorem card_le_pow_of_sub_mem (G : Finset K) (hG : G.Nonempty) :
    ∀ (n : ℕ) (V : Submodule K (ι → K)), finrank K V ≤ n →
      ∀ s : Finset (ι → K), (∀ x ∈ s, ∀ i, x i ∈ G) → (∀ x ∈ s, ∀ y ∈ s, x - y ∈ V) →
        s.card ≤ G.card ^ n := by
  classical
  intro n
  induction n with
  | zero =>
    intro V hV s _ hsub
    have hV0 : V = ⊥ := Submodule.finrank_eq_zero.mp (Nat.le_zero.mp hV)
    rw [pow_zero]
    refine Finset.card_le_one.mpr fun x hx y hy => ?_
    have h := hsub x hx y hy
    rwa [hV0, Submodule.mem_bot, sub_eq_zero] at h
  | succ n ih =>
    intro V hV s hsG hsub
    by_cases hV0 : V = ⊥
    · have h1 : s.card ≤ 1 := Finset.card_le_one.mpr fun x hx y hy => by
        have h := hsub x hx y hy
        rwa [hV0, Submodule.mem_bot, sub_eq_zero] at h
      exact h1.trans (Nat.one_le_pow _ _ (Finset.card_pos.mpr hG))
    obtain ⟨v, hvV, hv0⟩ := Submodule.exists_mem_ne_zero_of_ne_bot hV0
    obtain ⟨i, hi⟩ : ∃ i, v i ≠ 0 := by
      by_contra h
      exact hv0 (funext fun i => not_not.mp fun hne => h ⟨i, hne⟩)
    let V' : Submodule K (ι → K) := V ⊓ LinearMap.ker (LinearMap.proj i)
    have hlt : V' < V := by
      refine lt_of_le_of_ne inf_le_left fun h => hi ?_
      have hv' : v ∈ V' := h ▸ hvV
      simpa using (Submodule.mem_inf.mp hv').2
    have hV'n : finrank K V' ≤ n := by
      have := Submodule.finrank_lt_finrank_of_lt hlt
      omega
    rw [Finset.card_eq_sum_card_fiberwise (f := fun x => x i) (t := G) (fun x hx => hsG x hx i)]
    calc ∑ a ∈ G, (s.filter fun x => x i = a).card ≤ ∑ _a ∈ G, G.card ^ n := by
          refine Finset.sum_le_sum fun a _ =>
            ih V' hV'n _ (fun x hx => hsG x (Finset.mem_filter.mp hx).1 ) ?_
          intro x hx y hy
          rw [Finset.mem_filter] at hx hy
          refine Submodule.mem_inf.mpr ⟨hsub x hx.1 y hy.1, ?_⟩
          simp [hx.2, hy.2]
      _ = G.card ^ (n + 1) := by rw [Finset.sum_const, smul_eq_mul, pow_succ, mul_comm]

end Density

/-- The noise grid `{−31, …, 32}` as rationals. -/
def grid : Finset ℚ := (Finset.Icc (-31 : ℤ) 32).image (Int.cast : ℤ → ℚ)

theorem grid_card : grid.card = 64 := by
  rw [grid, Finset.card_image_of_injective _ Int.cast_injective, Int.card_Icc]
  rfl

theorem grid_nonempty : grid.Nonempty := Finset.card_pos.mp (by rw [grid_card]; norm_num)

section Slices

variable {ι : Type*} [Fintype ι]

/-- **L3 on the noise grid.** A set of grid points inside a translate of `V` has at most `64 ^ finrank V`
points; for an affine subspace of codimension `r`, `finrank V = |ι| − r`. -/
theorem grid_card_le (V : Submodule ℚ (ι → ℚ)) (s : Finset (ι → ℚ)) (hs : ∀ x ∈ s, ∀ i, x i ∈ grid)
    (hsub : ∀ x ∈ s, ∀ y ∈ s, x - y ∈ V) : s.card ≤ 64 ^ finrank ℚ V := by
  simpa [grid_card] using card_le_pow_of_sub_mem grid grid_nonempty _ V le_rfl s hs hsub

/-- The dimension count behind L2: the common kernel of `r` independent functionals has codimension `r`. -/
theorem finrank_ker_mulVecLin {κ : Type*} [Fintype κ] (A : Matrix κ ι ℚ) :
    finrank ℚ (LinearMap.ker A.mulVecLin) = Fintype.card ι - A.rank := by
  have h := LinearMap.finrank_range_add_finrank_ker A.mulVecLin
  rw [Module.finrank_fintype_fun_eq_card] at h
  rw [Matrix.rank]
  omega

/-- **L2 with L3 (a family of functionals).** If each functional `w c` takes the constant value `h c` on a set of
grid points, the set has at most `64 ^ (|ι| − rank)` points, `rank` being that of the family. -/
theorem slice_card_le {κ : Type*} [Fintype κ] (w : κ → ι → ℚ) (h : κ → ℚ) (s : Finset (ι → ℚ))
    (hs : ∀ x ∈ s, ∀ i, x i ∈ grid) (hZ : ∀ x ∈ s, ∀ c, w c ⬝ᵥ x = h c) :
    s.card ≤ 64 ^ (Fintype.card ι - (Matrix.of w).rank) := by
  rw [← finrank_ker_mulVecLin]
  refine grid_card_le _ s hs fun x hx y hy => ?_
  rw [LinearMap.mem_ker, Matrix.mulVecLin_apply, Matrix.mulVec_sub, sub_eq_zero]
  funext c
  simp only [Matrix.mulVec, Matrix.of_apply]
  exact (hZ x hx c).trans (hZ y hy c).symm

/-- **The counting core of L4.** Grid solutions of `A·x = y` number at most `64 ^ (|ι| − rank A)`. With `A` a
nonzero element of the checkpoint span, `rank A ≥ ρ` (MinRank), so `Pr_{f_j}[A·f_j = y] ≤ 64 ^ (−ρ)`. -/
theorem solutions_card_le {κ : Type*} [Fintype κ] (A : Matrix κ ι ℚ) (y : κ → ℚ) (s : Finset (ι → ℚ))
    (hs : ∀ x ∈ s, ∀ i, x i ∈ grid) (hA : ∀ x ∈ s, A.mulVec x = y) :
    s.card ≤ 64 ^ (Fintype.card ι - A.rank) :=
  slice_card_le (fun c => A c) y s hs fun x hx c => congrFun (hA x hx) c

/-- **L2 with L3 (intrinsic form).** If every functional in a subspace `W` of `ι → ℚ` is constant on a set of grid
points, the set has at most `64 ^ (|ι| − finrank W)` points. For a fixed `F₁` and `W = Ψ_f(C)` this is
`|Z_f| ≤ 64 ^ (m·k − rank Ψ_f)`, so `Pr_E[Z_f] ≤ 64 ^ (−rank Ψ_f)`. -/
theorem slice_card_le_of_subspace (W : Submodule ℚ (ι → ℚ)) (s : Finset (ι → ℚ))
    (hs : ∀ x ∈ s, ∀ i, x i ∈ grid) (hW : ∀ w ∈ W, ∀ x ∈ s, ∀ y ∈ s, w ⬝ᵥ x = w ⬝ᵥ y) :
    s.card ≤ 64 ^ (Fintype.card ι - finrank ℚ W) := by
  rcases s.eq_empty_or_nonempty with rfl | ⟨x₀, hx₀⟩
  · simp
  let b := Module.finBasis ℚ W
  have hspan : Submodule.span ℚ (Set.range fun c => (b c : ι → ℚ)) = W := by
    rw [show (Set.range fun c => (b c : ι → ℚ)) = W.subtype '' Set.range b by
      rw [← Set.range_comp]; rfl]
    rw [Submodule.span_image, b.span_eq, Submodule.map_top, Submodule.range_subtype]
  have hrank : (Matrix.of fun c => (b c : ι → ℚ)).rank = finrank ℚ W := by
    have hrow : Set.range (Matrix.of fun c => (b c : ι → ℚ)).row = Set.range fun c => (b c : ι → ℚ) :=
      rfl
    rw [Matrix.rank_eq_finrank_span_row, hrow, hspan]
  rw [← hrank]
  exact slice_card_le _ (fun c => (b c : ι → ℚ) ⬝ᵥ x₀) s hs
    fun x hx c => hW _ (b c).2 x hx x₀ hx₀

end Slices

end Pouw.Dimension.Lifting
