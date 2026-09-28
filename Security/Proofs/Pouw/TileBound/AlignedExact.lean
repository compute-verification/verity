import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Tactic
import Definitions.Pouw.TileBound.AlignedExactDefs

/-!
# Aligned exactness: from one row to a whole region (the TT_OUT restatements lane)

`no-aligned-exact-region/sm120-unpromoted` says no `a × b` block of a unit's output words is exact on every step of a
window, except with negligible probability over E_A. This file proves the step that handles the dependence through
shared rows, columns and noise lines, as a counting statement over the rows' raw oracle outputs.

* **The model** (the instantiation is in `AlignedExactDefs.lean`). Fix the F lines, the weights, the columns' raw oracle
  outputs `c` (E_B's, which the prover may know), and the activations, for any in-domain choice made before E_A is
  drawn. Each A′ row is then a function of its own raw output `R i : ρ`, and these are independent and uniform. Whether
  word `(i, j)` is exact over the window is `f i j (R i) (c j)`.
* **The hypothesis** (`RowCompat f b θ`): for every row, every `b` columns and every choice of their seeds, at most a `θ`
  share of row seeds make the row exact with all of them. Within-row dependence (a row's codes share its 32 noise
  coefficients) is all inside this one-row statement. It is the open, analytic part: no sampled rate establishes it.
* **The theorems.** `region_count_given_le`: for every assignment of column seeds, at most a `C(m, a)·C(n, b)·θ^a` share
  of row-seed assignments have an all-exact `a × b` block. That is the form a citation uses, since the prover may know
  the columns. `region_count_le` is the same bound averaged over column seeds.
* **What a citation adds.** `f` is one window, one unit and one draw of E_A, so a citation unions over the windows (at
  most k/32 starts for each window width the region family uses), the job's units and the prover's grinding draws. The
  bound bites only for small `θ`: at 1,024 × 512 in an 8,192² unit it is ≤ 2^−128 only if θ < 2^−7.16 ≈ 0.70%. For a
  short, wide region the rows are the shorter side, so `θ` is needed at the wide side's column count: at 432 × 1,728,
  θ < 2^−20 at 1,728 columns.
-/

namespace Pouw.AlignedExact

open Finset Classical

variable {m n : ℕ} {ρ κ : Type*} [Fintype ρ] [Fintype κ]

/-- Row seeds with a condition on the rows of `I` only: the count factors over rows. -/
theorem card_rows_le (I : Finset (Fin m)) (P : Fin m → ρ → Prop) (θ : ℚ) (_hθ : 0 ≤ θ)
    (h : ∀ i ∈ I, ((univ.filter (P i)).card : ℚ) ≤ θ * Fintype.card ρ) :
    ((univ.filter fun R : Fin m → ρ => ∀ i ∈ I, P i (R i)).card : ℚ) ≤
      θ ^ I.card * (Fintype.card ρ : ℚ) ^ m := by
  have hset : (univ.filter fun R : Fin m → ρ => ∀ i ∈ I, P i (R i)) =
      Fintype.piFinset fun i => univ.filter fun r : ρ => i ∈ I → P i r := by
    ext R
    simp only [mem_filter, mem_univ, true_and, Fintype.mem_piFinset]
  rw [hset, Fintype.card_piFinset, Nat.cast_prod]
  have hle : ∀ i ∈ (univ : Finset (Fin m)), ((univ.filter fun r : ρ => i ∈ I → P i r).card : ℚ) ≤
      if i ∈ I then θ * Fintype.card ρ else Fintype.card ρ := by
    intro i _
    by_cases hi : i ∈ I
    · have e : (univ.filter fun r : ρ => i ∈ I → P i r) = univ.filter (P i) := by
        ext r; simp [hi]
      rw [e]
      simpa only [hi, ite_true] using h i hi
    · have e : (univ.filter fun r : ρ => i ∈ I → P i r) = univ := by
        ext r; simp [hi]
      rw [e]
      simp only [hi, ite_false, card_univ, le_refl]
  calc ∏ i, ((univ.filter fun r : ρ => i ∈ I → P i r).card : ℚ)
      ≤ ∏ i, (if i ∈ I then θ * Fintype.card ρ else (Fintype.card ρ : ℚ)) :=
        prod_le_prod₀ (fun _ _ => by positivity) hle
    _ = θ ^ I.card * (Fintype.card ρ : ℚ) ^ m := by
        rw [prod_ite, prod_const, prod_const, filter_mem_eq_inter, univ_inter]
        have hc : (univ.filter fun i : Fin m => i ∉ I).card = m - I.card := by
          rw [filter_not, card_sdiff]; simp
        rw [hc, mul_pow]
        have hI : I.card ≤ m := by simpa using card_le_univ I
        rw [mul_assoc, ← pow_add, Nat.add_sub_cancel' hI]

/-- For one block `I × J`: the seed assignments exact on all of it number at most `θ^a` of all. -/
theorem card_block_le (f : Fin m → Fin n → ρ → κ → Prop) (b : ℕ) (θ : ℚ)
    (hθ : 0 ≤ θ) (hf : RowCompat f b θ) (I : Finset (Fin m)) (J : Finset (Fin n)) (hJ : J.card = b) :
    ((univ.filter fun x : (Fin m → ρ) × (Fin n → κ) => ∀ i ∈ I, ∀ j ∈ J, f i j (x.1 i) (x.2 j)).card : ℚ) ≤
      θ ^ I.card * (Fintype.card ρ : ℚ) ^ m * (Fintype.card κ : ℚ) ^ n := by
  have hsplit : (univ.filter fun x : (Fin m → ρ) × (Fin n → κ) => ∀ i ∈ I, ∀ j ∈ J, f i j (x.1 i) (x.2 j)).card =
      ∑ c : Fin n → κ, (univ.filter fun R : Fin m → ρ => ∀ i ∈ I, ∀ j ∈ J, f i j (R i) (c j)).card := by
    rw [Finset.card_filter, Fintype.sum_prod_type, Finset.sum_comm]
    refine Finset.sum_congr rfl fun c _ => ?_
    rw [Finset.card_filter]
  rw [hsplit, Nat.cast_sum]
  have hrow : ∀ c : Fin n → κ, ((univ.filter fun R : Fin m → ρ => ∀ i ∈ I, ∀ j ∈ J, f i j (R i) (c j)).card : ℚ) ≤
      θ ^ I.card * (Fintype.card ρ : ℚ) ^ m := by
    intro c
    have key := card_rows_le I (fun i r => ∀ j ∈ J, f i j r (c j)) θ hθ (fun i _ => by convert hf i J hJ c)
    convert key using 4
  calc ∑ c : Fin n → κ, ((univ.filter fun R : Fin m → ρ => ∀ i ∈ I, ∀ j ∈ J, f i j (R i) (c j)).card : ℚ)
      ≤ ∑ _c : Fin n → κ, θ ^ I.card * (Fintype.card ρ : ℚ) ^ m := Finset.sum_le_sum fun c _ => hrow c
    _ = θ ^ I.card * (Fintype.card ρ : ℚ) ^ m * (Fintype.card κ : ℚ) ^ n := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fun, Fintype.card_fin, nsmul_eq_mul]
        push_cast
        ring

omit [Fintype κ] in
/-- **The region bound for known columns.** Under `RowCompat f b θ`, for every assignment `c` of column seeds, the
row-seed assignments with an all-exact `a × b` block number at most `C(m, a)·C(n, b)·θ^a` times all row-seed
assignments. -/
theorem region_count_given_le (f : Fin m → Fin n → ρ → κ → Prop) (a b : ℕ) (θ : ℚ)
    (hθ : 0 ≤ θ) (hf : RowCompat f b θ) (c : Fin n → κ) :
    ((univ.filter fun R : Fin m → ρ => HasRegion f a b (R, c)).card : ℚ) ≤
      (m.choose a : ℚ) * (n.choose b) * θ ^ a * Fintype.card (Fin m → ρ) := by
  have hsub : (univ.filter fun R : Fin m → ρ => HasRegion f a b (R, c)) ⊆
      (powersetCard a (univ : Finset (Fin m))).biUnion fun I => (powersetCard b (univ : Finset (Fin n))).biUnion
        fun J => univ.filter fun R : Fin m → ρ => ∀ i ∈ I, ∀ j ∈ J, f i j (R i) (c j) := by
    intro R hR
    obtain ⟨I, J, hI, hJ, h⟩ := (mem_filter.mp hR).2
    simp only [mem_biUnion, mem_powersetCard, mem_filter, mem_univ, true_and, subset_univ]
    exact ⟨I, hI, J, hJ, h⟩
  have hnat : (univ.filter fun R : Fin m → ρ => HasRegion f a b (R, c)).card ≤
      ∑ I ∈ powersetCard a (univ : Finset (Fin m)), ∑ J ∈ powersetCard b (univ : Finset (Fin n)),
        (univ.filter fun R : Fin m → ρ => ∀ i ∈ I, ∀ j ∈ J, f i j (R i) (c j)).card :=
    (card_le_card hsub).trans (card_biUnion_le.trans (sum_le_sum fun _ _ => card_biUnion_le))
  have hblock : ∀ I ∈ powersetCard a (univ : Finset (Fin m)), ∀ J ∈ powersetCard b (univ : Finset (Fin n)),
      ((univ.filter fun R : Fin m → ρ => ∀ i ∈ I, ∀ j ∈ J, f i j (R i) (c j)).card : ℚ) ≤
        θ ^ a * (Fintype.card ρ : ℚ) ^ m := by
    intro I hI J hJ
    rw [mem_powersetCard] at hI hJ
    have key := card_rows_le I (fun i r => ∀ j ∈ J, f i j r (c j)) θ hθ (fun i _ => by convert hf i J hJ.2 c)
    rw [hI.2] at key
    convert key using 4
  calc ((univ.filter fun R : Fin m → ρ => HasRegion f a b (R, c)).card : ℚ)
      ≤ ∑ I ∈ powersetCard a (univ : Finset (Fin m)), ∑ J ∈ powersetCard b (univ : Finset (Fin n)),
          ((univ.filter fun R : Fin m → ρ => ∀ i ∈ I, ∀ j ∈ J, f i j (R i) (c j)).card : ℚ) := by
        exact_mod_cast hnat
    _ ≤ ∑ _I ∈ powersetCard a (univ : Finset (Fin m)), ∑ _J ∈ powersetCard b (univ : Finset (Fin n)),
          θ ^ a * (Fintype.card ρ : ℚ) ^ m :=
        sum_le_sum fun I hI => sum_le_sum fun J hJ => hblock I hI J hJ
    _ = (m.choose a : ℚ) * (n.choose b) * θ ^ a * Fintype.card (Fin m → ρ) := by
        rw [sum_const, sum_const, card_powersetCard, card_powersetCard, card_univ, card_univ, Fintype.card_fin,
          Fintype.card_fin, nsmul_eq_mul, nsmul_eq_mul, Fintype.card_fun, Fintype.card_fin]
        push_cast
        ring

/-- **The region bound, averaged over column seeds.** Under `RowCompat f b θ`, the seed assignments with an all-exact
`a × b` block number at most `C(m, a)·C(n, b)·θ^a` times all assignments. -/
theorem region_count_le (f : Fin m → Fin n → ρ → κ → Prop) (a b : ℕ) (θ : ℚ)
    (hθ : 0 ≤ θ) (hf : RowCompat f b θ) :
    ((univ.filter (HasRegion f a b)).card : ℚ) ≤
      (m.choose a : ℚ) * (n.choose b) * θ ^ a * (Fintype.card (Fin m → ρ) * Fintype.card (Fin n → κ)) := by
  have hsub : univ.filter (HasRegion f a b) ⊆
      (powersetCard a (univ : Finset (Fin m))).biUnion fun I => (powersetCard b (univ : Finset (Fin n))).biUnion
        fun J => univ.filter fun x : (Fin m → ρ) × (Fin n → κ) => ∀ i ∈ I, ∀ j ∈ J, f i j (x.1 i) (x.2 j) := by
    intro x hx
    obtain ⟨I, J, hI, hJ, h⟩ := (mem_filter.mp hx).2
    simp only [mem_biUnion, mem_powersetCard, mem_filter, mem_univ, true_and, subset_univ]
    exact ⟨I, hI, J, hJ, h⟩
  have h1 := card_le_card hsub
  have h2 := card_biUnion_le (s := powersetCard a (univ : Finset (Fin m))) (t := fun I =>
    (powersetCard b (univ : Finset (Fin n))).biUnion fun J =>
      univ.filter fun x : (Fin m → ρ) × (Fin n → κ) => ∀ i ∈ I, ∀ j ∈ J, f i j (x.1 i) (x.2 j))
  have hblock : ∀ I ∈ powersetCard a (univ : Finset (Fin m)), ∀ J ∈ powersetCard b (univ : Finset (Fin n)),
      ((univ.filter fun x : (Fin m → ρ) × (Fin n → κ) => ∀ i ∈ I, ∀ j ∈ J, f i j (x.1 i) (x.2 j)).card : ℚ) ≤
        θ ^ a * (Fintype.card ρ : ℚ) ^ m * (Fintype.card κ : ℚ) ^ n := by
    intro I hI J hJ
    rw [mem_powersetCard] at hI hJ
    have := card_block_le f b θ hθ hf I J hJ.2
    rwa [hI.2] at this
  have hinner : ∀ I ∈ powersetCard a (univ : Finset (Fin m)),
      (((powersetCard b (univ : Finset (Fin n))).biUnion fun J =>
        univ.filter fun x : (Fin m → ρ) × (Fin n → κ) => ∀ i ∈ I, ∀ j ∈ J, f i j (x.1 i) (x.2 j)).card : ℚ) ≤
        (n.choose b : ℚ) * (θ ^ a * (Fintype.card ρ : ℚ) ^ m * (Fintype.card κ : ℚ) ^ n) := by
    intro I hI
    calc (((powersetCard b (univ : Finset (Fin n))).biUnion fun J =>
          univ.filter fun x : (Fin m → ρ) × (Fin n → κ) => ∀ i ∈ I, ∀ j ∈ J, f i j (x.1 i) (x.2 j)).card : ℚ)
        ≤ ∑ J ∈ powersetCard b (univ : Finset (Fin n)),
            ((univ.filter fun x : (Fin m → ρ) × (Fin n → κ) => ∀ i ∈ I, ∀ j ∈ J, f i j (x.1 i) (x.2 j)).card : ℚ) := by
          exact_mod_cast card_biUnion_le
      _ ≤ ∑ _J ∈ powersetCard b (univ : Finset (Fin n)), θ ^ a * (Fintype.card ρ : ℚ) ^ m * (Fintype.card κ : ℚ) ^ n :=
          Finset.sum_le_sum fun J hJ => hblock I hI J hJ
      _ = (n.choose b : ℚ) * (θ ^ a * (Fintype.card ρ : ℚ) ^ m * (Fintype.card κ : ℚ) ^ n) := by
          rw [Finset.sum_const, card_powersetCard, card_univ, Fintype.card_fin, nsmul_eq_mul]
  calc ((univ.filter (HasRegion f a b)).card : ℚ)
      ≤ ∑ I ∈ powersetCard a (univ : Finset (Fin m)), (((powersetCard b (univ : Finset (Fin n))).biUnion fun J =>
          univ.filter fun x : (Fin m → ρ) × (Fin n → κ) => ∀ i ∈ I, ∀ j ∈ J, f i j (x.1 i) (x.2 j)).card : ℚ) := by
        exact_mod_cast h1.trans h2
    _ ≤ ∑ _I ∈ powersetCard a (univ : Finset (Fin m)),
          (n.choose b : ℚ) * (θ ^ a * (Fintype.card ρ : ℚ) ^ m * (Fintype.card κ : ℚ) ^ n) :=
        Finset.sum_le_sum hinner
    _ = (m.choose a : ℚ) * (n.choose b) * θ ^ a * (Fintype.card (Fin m → ρ) * Fintype.card (Fin n → κ)) := by
        rw [Finset.sum_const, card_powersetCard, card_univ, Fintype.card_fin, nsmul_eq_mul, Fintype.card_fun,
          Fintype.card_fun, Fintype.card_fin, Fintype.card_fin]
        push_cast
        ring

end Pouw.AlignedExact
