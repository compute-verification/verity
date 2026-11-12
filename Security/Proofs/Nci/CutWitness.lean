module

public import Proofs.Nci.Cut

/-!
# The min-cut statements say something

`Cut_honest`, `CountHE_honest` and each guarantee's `_nontrivial`, at the toy `Nci.Cut.Toy`. The toy's `w`- and
`x`-words are each one unregistered input, which every separator from the unregistered inputs holds
(`card_inter_le_kappa`), so `|S ∩ W| + |S ∩ X| ≤ κ(S)` (`toy_card_words_le`): that is (H) at unit costs for every read
cap, and at `κ(S) ≤ 1` no matmul has both a `w`- and an `x`-word in `S`, so `Π(k) = 0` and `α = 0` for `k ≤ 1`
(`toy_piMax`, `toy_alpha`). Each MAC is its output's one wire, so no set of gates finishes an output it holds no wire of
(`toy_clear`), and (E) holds at every `b`. One block holding both MACs reads the three inputs (`toy_reads_P1`); a block
for each reads two (`toy_reads_P2`), so the toy complies with `CompliesAtMost 1 2 2` (`toy_complies`); every block of a
partition reads an unregistered input, so it fails `CompliesAtMost c 0 n` at every `c` and `n` (`toy_not_complies`).
-/

@[expose] public section

namespace Nci.Cut.SecurityProofs

open Verity Finset

theorem toy_allW : Toy.W.allW = {{0}} := by decide

theorem toy_allX : Toy.W.allX = {{1}, {2}} := by decide

theorem toy_wWords : Toy.W.wWords () = {{0}} := by decide

theorem toy_xWords : Toy.W.xWords () = {{1}, {2}} := by decide

theorem toy_N : Toy.W.N = 2 := by decide

theorem toy_O : Toy.W.O = 2 := by decide

theorem toy_size : Toy.d.D.size Toy.d.D.computed = 2 := by decide

theorem toy_reads_P1 : (Toy.d.unregistered (Toy.P1.gates 0)).card = 3 := by decide

theorem toy_not_reads_P1 {k : ℕ} (hk : k < 3) : ¬ ReadsAtMost Toy.d Toy.P1 k := fun h =>
  absurd (toy_reads_P1.symm.le.trans (h 0)) (by omega)

theorem toy_reads_P2 : ReadsAtMost Toy.d Toy.P2 2 := by
  unfold ReadsAtMost
  decide

/-- The toy's words that `S` holds are unregistered inputs in `S`'s words, so at most `κ(S)` of them. -/
theorem toy_card_words_le (S : Finset (Finset (Fin Toy.C.N))) :
    (S ∩ Toy.W.allW).card + (S ∩ Toy.W.allX).card ≤ kappaWords Toy.d S := by
  have hdisj : Disjoint (S ∩ Toy.W.allW) (S ∩ Toy.W.allX) := by
    refine disjoint_of_subset_left inter_subset_right (disjoint_of_subset_right inter_subset_right ?_)
    rw [toy_allW, toy_allX]
    decide
  have hT : (S ∩ Toy.W.allW) ∪ (S ∩ Toy.W.allX) ⊆
      (unregisteredInputs Toy.d ∩ S.biUnion id).image fun g => ({g} : Finset (Fin Toy.C.N)) := by
    intro s hs
    rw [toy_allW, toy_allX] at hs
    simp only [mem_union, mem_inter, mem_insert, mem_singleton] at hs
    obtain ⟨hsS, rfl⟩ | ⟨hsS, rfl | rfl⟩ := hs
    · exact mem_image.2 ⟨(0 : Fin Toy.C.N), mem_inter.2 ⟨by decide, mem_biUnion.2 ⟨_, hsS, mem_singleton_self _⟩⟩, rfl⟩
    · exact mem_image.2 ⟨(1 : Fin Toy.C.N), mem_inter.2 ⟨by decide, mem_biUnion.2 ⟨_, hsS, mem_singleton_self _⟩⟩, rfl⟩
    · exact mem_image.2 ⟨(2 : Fin Toy.C.N), mem_inter.2 ⟨by decide, mem_biUnion.2 ⟨_, hsS, mem_singleton_self _⟩⟩, rfl⟩
  calc (S ∩ Toy.W.allW).card + (S ∩ Toy.W.allX).card
      = ((S ∩ Toy.W.allW) ∪ (S ∩ Toy.W.allX)).card := (card_union_of_disjoint hdisj).symm
    _ ≤ ((unregisteredInputs Toy.d ∩ S.biUnion id).image fun g => ({g} : Finset (Fin Toy.C.N))).card :=
      card_le_card hT
    _ ≤ (unregisteredInputs Toy.d ∩ S.biUnion id).card := card_image_le
    _ ≤ kappaWords Toy.d S := card_inter_le_kappa (unregisteredInputs Toy.d) (S.biUnion id)

theorem mul_eq_zero_of_add_le_one {a b : ℕ} (h : a + b ≤ 1) : a * b = 0 := by
  rcases Nat.eq_zero_or_pos a with rfl | ha
  · exact zero_mul b
  · rw [show b = 0 by omega, mul_zero]

theorem toy_pi_eq_zero (S : Finset (Finset (Fin Toy.C.N))) (h : kappaWords Toy.d S ≤ 1) : Toy.W.pi S = 0 := by
  have hc := (toy_card_words_le S).trans h
  rw [toy_allW, toy_allX] at hc
  simp only [Work.pi, Fintype.univ_unit, sum_singleton, toy_wWords, toy_xWords]
  exact mul_eq_zero_of_add_le_one hc

theorem toy_piMax {k : ℕ} (hk : k ≤ 1) : piMax Toy.d Toy.W k = 0 := by
  unfold piMax
  exact Nat.eq_zero_of_le_zero (Finset.sup_le fun S hS => (toy_pi_eq_zero S ((mem_filter.1 hS).2.trans hk)).le)

theorem toy_alpha {k : ℕ} (hk : k ≤ 1) : alpha Toy.d Toy.W k = 0 := by
  rw [alpha, toy_piMax hk, Nat.cast_zero, mul_zero, zero_div, Real.sqrt_zero]

theorem toy_hardness (k : ℕ) : Assumptions.Hardness Toy.d Toy.arrays Toy.val ![1, 1] k := by
  intro S _
  have h := toy_card_words_le S
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, image_id, one_mul]
  exact_mod_cast h

/-- Each MAC is its output's one wire, so a set holding the MAC holds a wire of the output. -/
theorem toy_clear (B : Finset (Fin Toy.C.N)) : Toy.W.clear B = ∅ := by
  refine filter_false_of_mem fun o _ h => ?_
  obtain ⟨⟨s, rfl, hsB⟩, hdis⟩ := h
  exact disjoint_left.1 hdis (mem_singleton_self _) (hsB (mem_singleton_self _))

theorem toy_exports (b : ℝ) : Assumptions.Exports Toy.W b := by
  intro B _
  rw [toy_clear, card_empty, Nat.cast_zero, mul_zero]
  exact Nat.cast_nonneg _

theorem toy_values : ValuesInjective Toy.W Toy.val := fun _ => ⟨Set.injOn_id _, Set.injOn_id _⟩

theorem toy_noSharedPair : NoSharedPair Toy.W Toy.val := fun a a' h => absurd (Subsingleton.elim a a') h

theorem toy_arrays : Arrays Toy.W ⊤ Toy.arrays Toy.edge 2 := by
  refine ⟨SimpleGraph.cliqueFree_of_card_lt (by simp), fun _ => (SimpleGraph.top_adj _ _).2 ?_,
    fun a a' _ => Subsingleton.elim a a', fun a => ⟨?_, ?_⟩⟩
  · show (0 : Fin 2) ≠ 1
    decide
  · show Toy.W.wWords a ⊆ Toy.W.allW
    exact subset_biUnion_of_mem Toy.W.wWords (mem_univ a)
  · show Toy.W.xWords a ⊆ Toy.W.allX
    exact subset_biUnion_of_mem Toy.W.xWords (mem_univ a)

theorem toy_costs : CostsAtLeast Toy.edge ![1, 1] 1 := fun _ => by
  show (1 : ℝ) ≤ ![(1 : ℝ), 1] 0 * ![(1 : ℝ), 1] 1
  simp

theorem Cut_honest : Guarantees.Cut_honest :=
  ⟨toy_reads_P2, toy_values, toy_noSharedPair, toy_arrays, toy_costs, toy_hardness 2, toy_exports 1⟩

theorem Bound_nontrivial : Guarantees.Bound_nontrivial := by
  refine ⟨toy_not_reads_P1 (by norm_num), ?_⟩
  rw [toy_alpha (k := 0) (by norm_num), toy_piMax (k := 0) (by norm_num), toy_N, toy_O]
  norm_num

theorem Slowdown_nontrivial : Guarantees.Slowdown_nontrivial := by
  refine ⟨toy_not_reads_P1 (by norm_num), ?_⟩
  rw [toy_alpha (k := 1) le_rfl, toy_piMax (k := 1) le_rfl, toy_N, toy_O, toy_size]
  norm_num

theorem HE_nontrivial : Guarantees.HE_nontrivial := by
  refine ⟨one_pos, toy_values, toy_noSharedPair, toy_hardness 0, toy_exports 1, toy_not_reads_P1 (by norm_num), ?_⟩
  rw [toy_N]
  norm_num [alphaHE]

/-- A block for each MAC: the toy complies with `CompliesAtMost 1 2 2`. -/
theorem toy_complies : CompliesAtMost 1 2 2 Toy.d :=
  ⟨2, le_rfl, Toy.P2, ⟨⟨fun _ => rfl, by decide⟩, by decide, toy_reads_P2⟩⟩

/-- Every block of a partition of the toy reads an unregistered input: the block of `y₀` reads `w`. -/
theorem toy_not_complies (c n : ℕ) : ¬ CompliesAtMost c 0 n Toy.d := by
  rintro ⟨B, -, P, hP⟩
  obtain ⟨u, hu⟩ := Option.ne_none_iff_exists'.1 fun h => absurd ((P.input_iff 3).1 h) (by decide)
  have h0 : P.unit 0 = none := (P.input_iff 0).2 (by decide)
  have hmem : (0 : Fin Toy.C.N) ∈ Toy.d.unregistered (P.gates u) := by
    simp only [RegisteredCircuit.unregistered, Circuit.into, Partition.gates, mem_sdiff, mem_filter, mem_univ,
      true_and, h0, reduceCtorEq, not_false_eq_true]
    exact ⟨⟨3, hu, by decide⟩, by decide⟩
  have h := hP.reads u
  rw [Nat.le_zero, card_eq_zero] at h
  rw [h] at hmem
  exact absurd hmem (notMem_empty _)

/-- The toy's bound at its two blocks, `k = 2`, unit costs and `b = 2`: `α = 1`, so `2 (2 + 1) + √4 = 8`. -/
theorem toy_workHE : workHE 2 2 (1 * 1) 2 Toy.W.O = 8 := by
  have h4 : √(4 : ℝ) = 2 := by
    rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq zero_le_two]
  rw [workHE, alphaHE, toy_O]
  norm_num [h4]

theorem CountHE_honest : Guarantees.CountHE_honest :=
  ⟨toy_complies, toy_values, toy_noSharedPair, toy_hardness 2, toy_exports 1⟩

theorem CountHE_nontrivial : Guarantees.CountHE_nontrivial := by
  refine ⟨one_pos, toy_values, toy_noSharedPair, toy_hardness 0, toy_exports 1, toy_not_complies 1 1, ?_⟩
  rw [toy_N]
  norm_num [workHE, alphaHE]

theorem HEShared_nontrivial : Guarantees.HEShared_nontrivial := by
  refine ⟨one_pos, zero_le_one, toy_arrays, toy_values, toy_costs, toy_hardness 0, toy_exports 1,
    toy_not_reads_P1 (by norm_num), ?_⟩
  rw [toy_N]
  norm_num [alphaHE]

end Nci.Cut.SecurityProofs
