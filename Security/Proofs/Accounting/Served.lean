import Properties.Accounting.Served
import Proofs.Accounting.Compose.ServiceBudget
import Proofs.Flock.Claims.Sampling.Select.Bounds

/-!
# The served window's draw misses a set no more often than a uniform permutation's subset (`DrawMissesSubset`)

`Properties.Served.DrawMissesSubset` from two bounds that meet at `C(N − |B|, K)/C(N, K)`, `B` the tiles of `U` below
`N`:

* **the code** (`Properties.SampledProofs.SecurityProofs.one_bound`): `drawUnits (.subset K) N` from `L` uniform bytes
  misses `B`'s numbers, and so `U`, with probability at most that ratio;
* **the model** (`miss_ge`): a uniform permutation's `K`-subset misses `B` with probability at least that ratio (it is
  exactly it), by induction on `K` through `Equiv.Perm.decomposeFin`, as `Budget.miss_le` bounds it from above: the
  first image is uniform and misses `B` with probability `1 − |B|/N`, the rest is a uniform permutation of `N − 1`
  tiles that must miss a set of at most `|B|` of them (`card_rest_le`), and `(m + 1)·C(m, K) = C(m + 1, K + 1)·(K + 1)`
  (`Nat.add_one_mul_choose_eq`) turns the product into the ratio.
-/

namespace Properties.Served.SecurityProofs

open Pouw.RelativeSecurity
open Accounting.SecurityProofs.Compose.Budget (forall_mem_subsetDraw rest prob_mem)
open Accounting.SecurityProofs.TS.Stages (prob_prod)

theorem prob_eq_pr {α : Type} [Fintype α] (p : α → Prop) : prob p = Claims.Pr p := by
  classical
  rw [prob_eq_card, Claims.Pr]

theorem card_rest_le {n : ℕ} (B : Finset (Fin (n + 1))) (p : Fin (n + 1)) : (rest B p).card ≤ B.card :=
  Finset.card_le_card_of_injOn (fun z => Equiv.swap 0 p z.succ) (fun z hz => by simpa [rest] using hz)
    (fun a _ b _ h => Fin.succ_injective _ ((Equiv.swap 0 p).injective h))

/-- **The miss bound from below**: a uniform permutation's `K`-subset of `N ≥ K` tiles misses `B` with probability at
least `C(N − |B|, K)/C(N, K)`. -/
theorem miss_ge : ∀ (K N : ℕ) (B : Finset (Fin N)), K ≤ N →
    ((N - B.card).choose K : ℝ) / (N.choose K : ℝ) ≤
      prob (fun π : Equiv.Perm (Fin N) => ∀ v ∈ Pouw.Window.subsetDraw K π, v ∉ B)
  | 0, N, B, _ => by
    classical
    have h1 : prob (fun π : Equiv.Perm (Fin N) => ∀ v ∈ Pouw.Window.subsetDraw 0 π, v ∉ B) = 1 := by
      rw [prob_eq_card, Finset.filter_true_of_mem fun π _ =>
        (forall_mem_subsetDraw (K := 0) π fun v => v ∉ B).mpr fun i hi => absurd hi (Nat.not_lt_zero _),
        Finset.card_univ]
      exact div_self (Nat.cast_ne_zero.2 Fintype.card_ne_zero)
    rw [h1]
    simp
  | K + 1, 0, _, h => absurd h (by omega)
  | K + 1, n + 1, B, h => by
    classical
    have hK : K ≤ n := Nat.le_of_succ_le_succ h
    rw [prob_equiv Equiv.Perm.decomposeFin, prob_prod]
    have hstep : ∀ p : Fin (n + 1),
        ((n - B.card).choose K : ℝ) / (n.choose K : ℝ) * (if p ∈ B then 0 else 1) ≤
          prob fun e : Equiv.Perm (Fin n) =>
            ∀ v ∈ Pouw.Window.subsetDraw (K + 1) (Equiv.Perm.decomposeFin.symm (p, e)), v ∉ B := by
      intro p
      by_cases hp : p ∈ B
      · rw [ite_eq_left hp, mul_zero]
        exact prob_nonneg _
      rw [ite_eq_right hp, mul_one]
      have hle : ((n - B.card).choose K : ℝ) / (n.choose K : ℝ) ≤
          ((n - (rest B p).card).choose K : ℝ) / (n.choose K : ℝ) := by
        refine div_le_div_of_nonneg_right ?_ (Nat.cast_nonneg _)
        exact_mod_cast Nat.choose_le_choose K (by have := card_rest_le B p; omega)
      refine hle.trans ((miss_ge K n (rest B p) hK).trans (prob_mono fun e he => ?_))
      refine (forall_mem_subsetDraw _ _).mpr fun i hi => ?_
      rcases Fin.eq_zero_or_eq_succ i with rfl | ⟨j, rfl⟩
      · rw [Equiv.Perm.decomposeFin_symm_apply_zero]
        exact hp
      · have hj : j.val < K := by simpa using hi
        have hmiss := (forall_mem_subsetDraw _ _).mp he j hj
        rw [Equiv.Perm.decomposeFin_symm_apply_succ]
        simpa [rest] using hmiss
    refine le_trans ?_ (expect_mono hstep)
    rw [expect_mul_left]
    have hnot : expect (fun p : Fin (n + 1) => if p ∈ B then (0 : ℝ) else 1) =
        1 - (B.card : ℝ) / ((n + 1 : ℕ) : ℝ) := by
      have hsum := prob_add_not (fun p : Fin (n + 1) => p ∈ B)
      rw [prob_mem] at hsum
      have hind : prob (fun p : Fin (n + 1) => ¬p ∈ B) =
          expect (fun p : Fin (n + 1) => if p ∈ B then (0 : ℝ) else 1) := by
        rw [prob_eq_expect]
        exact congrArg expect (funext fun p => by by_cases hp : p ∈ B <;> simp [hp])
      linarith
    rw [hnot]
    have hb : B.card ≤ n + 1 := (Finset.card_le_univ B).trans_eq (Fintype.card_fin _)
    have hCn : (0 : ℝ) < (n.choose K : ℝ) := by exact_mod_cast Nat.choose_pos hK
    have hCn1 : (0 : ℝ) < ((n + 1).choose (K + 1) : ℝ) := by exact_mod_cast Nat.choose_pos h
    have hn1 : (0 : ℝ) < ((n + 1 : ℕ) : ℝ) := by exact_mod_cast Nat.succ_pos n
    rcases Nat.lt_or_ge n B.card with hbn | hbn
    · have h0 : n + 1 - B.card = 0 := by omega
      rw [h0, Nat.choose_eq_zero_of_lt (show 0 < K + 1 by omega), Nat.cast_zero, zero_div]
      refine mul_nonneg (div_nonneg (Nat.cast_nonneg _) hCn.le) ?_
      rw [sub_nonneg]
      exact div_le_one_of_le₀ (by exact_mod_cast hb) hn1.le
    · obtain ⟨a, ha⟩ : ∃ a, n = B.card + a := ⟨n - B.card, by omega⟩
      have hm : n + 1 - B.card = a + 1 := by omega
      have hnb : n - B.card = a := by omega
      have hsub : 1 - (B.card : ℝ) / ((n + 1 : ℕ) : ℝ) = ((a + 1 : ℕ) : ℝ) / ((n + 1 : ℕ) : ℝ) := by
        rw [eq_div_iff hn1.ne', sub_mul, one_mul, div_mul_cancel₀ _ hn1.ne']
        have hcast : ((n + 1 : ℕ) : ℝ) = (B.card : ℝ) + ((a + 1 : ℕ) : ℝ) := by
          exact_mod_cast (show n + 1 = B.card + (a + 1) by omega)
        rw [hcast]
        ring
      have key : (a + 1).choose (K + 1) * (n.choose K * (n + 1)) =
          a.choose K * (a + 1) * (n + 1).choose (K + 1) := by
        calc (a + 1).choose (K + 1) * (n.choose K * (n + 1))
            = (a + 1).choose (K + 1) * ((n + 1) * n.choose K) := by ring
          _ = (a + 1).choose (K + 1) * ((n + 1).choose (K + 1) * (K + 1)) := by
            rw [Nat.add_one_mul_choose_eq n K]
          _ = (a + 1) * a.choose K * (n + 1).choose (K + 1) := by
            rw [Nat.add_one_mul_choose_eq a K]
            ring
          _ = a.choose K * (a + 1) * (n + 1).choose (K + 1) := by ring
      rw [hm, hnb, hsub, div_mul_div_comm, div_le_div_iff₀ hCn1 (mul_pos hCn hn1)]
      exact_mod_cast key.le

/-- The tiles of `Fin N` whose numbers are in `U`. -/
def tilesOf (N : ℕ) (U : Finset ℕ) : Finset (Fin N) := Finset.univ.filter fun t => t.val ∈ U

theorem mem_tilesOf {N : ℕ} {U : Finset ℕ} {t : Fin N} : t ∈ tilesOf N U ↔ t.val ∈ U := by
  simp [tilesOf]

/-- **`DrawMissesSubset`**: the code's subset draw misses `U` with probability at most `C(N − |B|, K)/C(N, K)`, `B`
the tiles of `U` (`one_bound` on `B`'s numbers), which the permutation's subset misses `B` with at least
(`miss_ge`). -/
theorem drawMissesSubset : DrawMissesSubset := by
  intro K N L U hK
  classical
  have hr : (tilesOf N U).map Fin.valEmbedding ⊆ Finset.range N := fun u hu => by
    obtain ⟨t, -, rfl⟩ := Finset.mem_map.1 hu
    exact Finset.mem_range.2 t.isLt
  have hU : ∀ u ∈ (tilesOf N U).map Fin.valEmbedding, u ∈ U := fun u hu => by
    obtain ⟨t, ht, rfl⟩ := Finset.mem_map.1 hu
    exact mem_tilesOf.1 ht
  have hcode := (Properties.SampledProofs.SecurityProofs.one_bound N _ hr L).1 K hK
  rw [Finset.card_map] at hcode
  refine le_trans ?_ (hcode.trans ?_)
  · rw [prob_eq_pr]
    refine Claims.Pr_mono fun src h => ?_
    obtain ⟨⟨units, read⟩, hd, hm⟩ := h
    exact ⟨units, read, hd, fun u hu => hm u (hU u hu)⟩
  · exact (miss_ge K N (tilesOf N U) hK).trans (le_of_eq (congrArg prob (funext fun β =>
      propext (forall₂_congr fun t _ => not_congr mem_tilesOf))))

end Properties.Served.SecurityProofs
