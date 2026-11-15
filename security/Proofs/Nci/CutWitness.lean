module

public import Proofs.Nci.Cut
public import Proofs.Nci.Untied

/-!
# The min-cut statements say something

`Cut_honest`, `CountHE_honest` and each guarantee's `_nontrivial`, at the toy `Nci.Cut.Toy`. The toy's `w`- and
`x`-words are each one unregistered input, which every separator from the unregistered inputs holds
(`card_inter_le_kappa`), so `|S ∩ W| + |S ∩ X| ≤ κ(S)` (`toy_card_words_le`): that is (H) at unit costs for every
budget `k`, and at `κ(S) ≤ 1` no matmul has both a `w`- and an `x`-word in `S`, so `Π(k) = 0` and `α = 0` for `k ≤ 1`
(`toy_piMax`, `toy_alpha`). Each MAC is its output's one wire, so no set of gates finishes an output it holds no wire of
(`toy_clear`), and (E) holds at every `b`. One block holding both MACs reads the three inputs (`toy_reads_P1`); a block
for each reads two (`toy_reads_P2`), so the toy complies with `CompliesAtMost 1 2 2` (`toy_complies`); every block of a
partition reads an unregistered input, so it fails `CompliesAtMost c 0 n` at every `c` and `n` (`toy_not_complies`).

The bounds with untied words (`Untied.bound_untie_he`, `Untied.bound_asInputs_he`) take (H) at the untied circuit, and
two toys multiply by an untied word. Words inside a set of gates that is closed under reads and holds no input gate
outside `θ` have `κ = 0` (`kappaWords_eq_zero_of_closed`): words of registered inputs
(`kappaWords_eq_zero_of_registered`), and words computed from them. (H) then fails at every `k`, at any positive cost
of an array that holds such a word (`not_hardness_of_kappa_zero_w`, `not_hardness_of_kappa_zero_x`). So (H) at `d`
fails for a work that multiplies by a word of registered inputs (`untie_hH_false`), and for a work of the circuit with
gates as inputs that multiplies by a word `d` computes from `θ` alone (`asInputs_hH_false`). On the toy with its `w`
gate registered and then untied, the premises of `Untied.bound_untie_he` hold together, with (H) at the untied
circuit, and (H) at `d` fails at every `k` and all positive costs (`toyR_premises`). On `CZ`, with a gate computed
from `θ` as an input and a work that multiplies by it, the premises of `Untied.bound_asInputs_he` hold together, with
(H) at the circuit with that gate as an input, and (H) at `d` fails (`toyZ_premises`). Each bound is applied at its
toy (`toyR_bound`, `toyZ_bound`), so a premise added to a bound breaks its witness. Both bounds are slack there, and
(E) holds because no output is ever in the clear: the toys show that the premises can hold together, not that a bound
is tight.
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

/-! ### (H) at the untied circuit

`Untied.bound_untie_he` and `Untied.bound_asInputs_he` take (H) at the untied circuit. Two toys multiply by an untied
word: there every premise of the bound holds, with (H) at the untied circuit, and (H) at `d` fails at every `k` and all
positive costs. Each bound is then applied at its toy. -/

/-- (H) holds at costs 1 and every `k` where each word of the two arrays is one input gate outside `θ` and no word is
in both: every separator from those gates holds each of them. -/
theorem hardness_of_input_words {V : Type} [DecidableEq V] (d : RegisteredCircuit)
    (Aw Ax : Finset (Finset (Fin d.D.N))) (val : Finset (Fin d.D.N) → V) (hdisj : Disjoint Aw Ax)
    (hw : ∀ E ∈ Aw ∪ Ax, ∃ g ∈ unregisteredInputs d, E = {g}) (k : ℕ) :
    Assumptions.Hardness d ![Aw, Ax] val ![1, 1] k := by
  intro S _
  have hd : Disjoint (S ∩ Aw) (S ∩ Ax) :=
    disjoint_of_subset_left inter_subset_right (disjoint_of_subset_right inter_subset_right hdisj)
  have hT : (S ∩ Aw) ∪ (S ∩ Ax) ⊆
      (unregisteredInputs d ∩ S.biUnion id).image fun g => ({g} : Finset (Fin d.D.N)) := by
    intro s hs
    have hsS : s ∈ S := by
      rcases mem_union.1 hs with h | h
      · exact (mem_inter.1 h).1
      · exact (mem_inter.1 h).1
    have hsA : s ∈ Aw ∪ Ax := by
      rcases mem_union.1 hs with h | h
      · exact mem_union_left _ (mem_inter.1 h).2
      · exact mem_union_right _ (mem_inter.1 h).2
    obtain ⟨g, hg, rfl⟩ := hw s hsA
    exact mem_image.2 ⟨g, mem_inter.2 ⟨hg, mem_biUnion.2 ⟨_, hsS, mem_singleton_self _⟩⟩, rfl⟩
  have hcard : (S ∩ Aw).card + (S ∩ Ax).card ≤ kappaWords d S :=
    calc (S ∩ Aw).card + (S ∩ Ax).card
        = ((S ∩ Aw) ∪ (S ∩ Ax)).card := (card_union_of_disjoint hd).symm
      _ ≤ ((unregisteredInputs d ∩ S.biUnion id).image fun g => ({g} : Finset (Fin d.D.N))).card :=
        card_le_card hT
      _ ≤ (unregisteredInputs d ∩ S.biUnion id).card := card_image_le
      _ ≤ kappaWords d S := card_inter_le_kappa (unregisteredInputs d) (S.biUnion id)
  have h1 : ((S ∩ Aw).image val).card ≤ (S ∩ Aw).card := card_image_le
  have h2 : ((S ∩ Ax).image val).card ≤ (S ∩ Ax).card := card_image_le
  have h3 : ((S ∩ Aw).image val).card + ((S ∩ Ax).image val).card ≤ kappaWords d S :=
    (Nat.add_le_add h1 h2).trans hcard
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, one_mul]
  exact_mod_cast h3

/-- A path that ends in a set of gates closed under reads starts in it. -/
theorem mem_of_feedsWithin_closed {C : Circuit} {R A : Finset (Fin C.N)} (hA : ∀ h ∈ A, C.reads h ⊆ A)
    {g t : Fin C.N} (hp : Nci.Covering.FeedsWithin C R g t) : t ∈ A → g ∈ A := by
  induction hp with
  | refl _ => exact fun h => h
  | step _ hgh _ ih => exact fun ht => hA _ (ih ht) hgh

/-- **Words that are functions of `θ` alone have `κ = 0`.** Words inside a set `A` of gates that is closed under reads
and holds no input gate outside `θ` are reached from no such gate, so the empty set separates them. -/
theorem kappaWords_eq_zero_of_closed (d : RegisteredCircuit) (S : Finset (Finset (Fin d.D.N)))
    (A : Finset (Fin d.D.N)) (hA : ∀ h ∈ A, d.D.reads h ⊆ A) (hU : ∀ g ∈ A, g ∉ unregisteredInputs d)
    (hS : S.biUnion id ⊆ A) : kappaWords d S = 0 := by
  have hsep : Separates d.D ∅ (unregisteredInputs d) (S.biUnion id) := fun x hx y hy hp =>
    hU x (mem_of_feedsWithin_closed hA hp (hS hy)) hx
  have h : kappa d.D (unregisteredInputs d) (S.biUnion id) ≤ (∅ : Finset (Fin d.D.N)).card := kappa_le hsep
  rw [card_empty] at h
  exact Nat.le_zero.1 h

/-- Words of registered inputs have `κ = 0`. -/
theorem kappaWords_eq_zero_of_registered (d : RegisteredCircuit) (S : Finset (Finset (Fin d.D.N)))
    (hS : ∀ g ∈ S.biUnion id, g ∈ d.θ) : kappaWords d S = 0 :=
  kappaWords_eq_zero_of_closed d S (S.biUnion id)
    (fun h hh => by
      rw [d.D.input_reads h (d.θ_input h (hS h hh))]
      exact empty_subset _)
    (fun g hg hu => (mem_filter.1 hu).2.2 (hS g hg)) (Subset.refl _)

/-- (H) fails at every `k`, at a positive cost of a `w`-value, where a `w`-word has `κ = 0`. -/
theorem not_hardness_of_kappa_zero_w {V : Type} [DecidableEq V] {d : RegisteredCircuit}
    {Aw Ax : Finset (Finset (Fin d.D.N))} {val : Finset (Fin d.D.N) → V} {βw βx : ℝ} (hw : 0 < βw) (hx : 0 ≤ βx)
    {E : Finset (Fin d.D.N)} (hE : E ∈ Aw) (hκ : kappaWords d {E} = 0) (k : ℕ) :
    ¬ Assumptions.Hardness d ![Aw, Ax] val ![βw, βx] k := by
  intro hH
  have h := hH {E} (by rw [hκ]; exact Nat.zero_le k)
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, singleton_inter_of_mem hE, image_singleton,
    card_singleton, Nat.cast_one, mul_one] at h
  rw [hκ, Nat.cast_zero] at h
  have h0 : ∀ t : ℝ, 0 ≤ t → βw + βx * t ≤ 0 → False := fun t ht h' => by linarith [mul_nonneg hx ht]
  exact h0 _ (Nat.cast_nonneg _) h

/-- (H) fails at every `k`, at a positive cost of an `x`-value, where an `x`-word has `κ = 0`. -/
theorem not_hardness_of_kappa_zero_x {V : Type} [DecidableEq V] {d : RegisteredCircuit}
    {Aw Ax : Finset (Finset (Fin d.D.N))} {val : Finset (Fin d.D.N) → V} {βw βx : ℝ} (hw : 0 ≤ βw) (hx : 0 < βx)
    {E : Finset (Fin d.D.N)} (hE : E ∈ Ax) (hκ : kappaWords d {E} = 0) (k : ℕ) :
    ¬ Assumptions.Hardness d ![Aw, Ax] val ![βw, βx] k := by
  intro hH
  have h := hH {E} (by rw [hκ]; exact Nat.zero_le k)
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, singleton_inter_of_mem hE, image_singleton,
    card_singleton, Nat.cast_one, mul_one] at h
  rw [hκ, Nat.cast_zero] at h
  have h0 : ∀ t : ℝ, 0 ≤ t → βw * t + βx ≤ 0 → False := fun t ht h' => by linarith [mul_nonneg hw ht]
  exact h0 _ (Nat.cast_nonneg _) h

/-- **(H) at `d` fails for a work that multiplies by a word of registered inputs**, at every `k` and all positive
costs. -/
theorem untie_hH_false {ι V : Type} [Fintype ι] [DecidableEq V] (d : RegisteredCircuit) (W : Work d.D ι)
    (val : Finset (Fin d.D.N) → V) {βw βx : ℝ} (hw : 0 < βw) (hx : 0 < βx) {E : Finset (Fin d.D.N)}
    (hE : E ∈ W.allW ∨ E ∈ W.allX) (hθ : ∀ g ∈ E, g ∈ d.θ) (k : ℕ) :
    ¬ Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] k := by
  have hκ : kappaWords d {E} = 0 :=
    kappaWords_eq_zero_of_registered d {E} (by rw [singleton_biUnion]; exact hθ)
  rcases hE with hE | hE
  · exact not_hardness_of_kappa_zero_w hw hx.le hE hκ k
  · exact not_hardness_of_kappa_zero_x hw.le hx hE hκ k

/-- **(H) at `d` fails for a work of the circuit with `T` as inputs that multiplies by a word `d` computes from `θ`
alone**, at every `k` and all positive costs. -/
theorem asInputs_hH_false {ι V : Type} [Fintype ι] [DecidableEq V] (d : RegisteredCircuit) (T : Finset (Fin d.D.N))
    (W : Work (d.D.asInputs T) ι) (val : Finset (Fin d.D.N) → V) {βw βx : ℝ} (hw : 0 < βw) (hx : 0 < βx)
    {E : Finset (Fin d.D.N)} (hE : E ∈ W.allW ∨ E ∈ W.allX) (hκ : kappaWords d {E} = 0) (k : ℕ) :
    ¬ Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] k := by
  rcases hE with hE | hE
  · exact not_hardness_of_kappa_zero_w hw hx.le hE hκ k
  · exact not_hardness_of_kappa_zero_x hw.le hx hE hκ k

/-- The toy with its `w` gate, gate 0, registered. -/
abbrev dR : RegisteredCircuit := .ofCircuit Toy.C {0} (by decide)

/-- The untied word: the `w` gate. -/
abbrev wordR : Finset (Fin Toy.C.N) := {0}

theorem toyR_word : wordR ∈ Toy.W.allW := by
  rw [toy_allW]
  exact mem_singleton_self _

/-- Untied at its `w` gate, the toy has the input gates outside `θ` of the toy with nothing registered. -/
theorem toyR_unreg : unregisteredInputs (dR.untie wordR) = unregisteredInputs Toy.d := by decide

theorem toyR_kappa (S : Finset (Finset (Fin Toy.C.N))) : kappaWords (dR.untie wordR) S = kappaWords Toy.d S :=
  congrArg (fun X : Finset (Fin Toy.C.N) => kappa Toy.C X (S.biUnion id)) toyR_unreg

/-- (H) holds at the untied circuit, at costs 1 and every `k`. -/
theorem toyR_hardness_untied (k : ℕ) :
    Assumptions.Hardness (dR.untie wordR) Toy.arrays Toy.val ![1, 1] k := by
  intro S hS
  have hκ : kappaWords (dR.untie wordR) S = kappaWords Toy.d S := toyR_kappa S
  exact (toy_hardness k S ((le_of_eq hκ.symm).trans hS)).trans (Nat.cast_le.2 (le_of_eq hκ.symm))

/-- One block reads 2 gates outside `θ`, the two `x` gates. -/
theorem toyR_reads : ReadsAtMost dR Toy.P1 2 := by
  unfold ReadsAtMost
  decide

/-- **The premises of `Untied.bound_untie_he` hold together on a work that multiplies by the untied word.** The toy
with its `w` gate registered, untied at that gate (`𝒲 = {wordR}`, one word of one gate), one block. The block read 2
gates outside `θ` and the untied word has one reader. At the untied circuit, `dR.untie (𝒲.biUnion id)`, (H) holds at
costs 1 and every `k`; (E) holds and the values are distinct. (H) at `d` fails at every `k` and all positive costs:
the work multiplies by the untied word, whose `κ` at `d` is 0. -/
theorem toyR_premises :
    ReadsAtMost dR Toy.P1 2 ∧ Untied.OneReader Toy.P1 ({wordR} : Finset (Finset (Fin Toy.C.N))) ∧
      (∀ k, Assumptions.Hardness (dR.untie (({wordR} : Finset (Finset (Fin Toy.C.N))).biUnion id)) Toy.arrays Toy.val
        ![1, 1] k) ∧
      Assumptions.Exports Toy.W 1 ∧ ValuesInjective Toy.W Toy.val ∧ NoSharedPair Toy.W Toy.val ∧
      ∀ βw βx : ℝ, 0 < βw → 0 < βx → ∀ k, ¬ Assumptions.Hardness dR Toy.arrays Toy.val ![βw, βx] k :=
  ⟨toyR_reads, fun _ _ u v _ _ => Subsingleton.elim u v,
    fun k => by rw [singleton_biUnion]; exact toyR_hardness_untied k, toy_exports 1, toy_values, toy_noSharedPair,
    fun _ _ hw hx k => untie_hH_false dR Toy.W Toy.val hw hx (Or.inl toyR_word) (by decide) k⟩

/-- **`Untied.bound_untie_he` at the toy with its `w` gate untied.** The bound's conclusion at the toy's arguments, with
every block in `G`, costs 1 and `e = 1`, from the conjuncts of `toyR_premises`: a premise added to the bound breaks this
proof. It shows that the premises hold together on a work that multiplies by the untied word, and no more. The bound is
slack there, 2 MACs (`toy_N`) against `m k + b |𝒲| = 3` before the square roots, and (E) holds only because no output
is ever in the clear (`toy_clear`): the toy does not show the bound tight. -/
theorem toyR_bound :
    (Toy.W.N : ℝ) ≤ (∑ u ∈ (univ : Finset (Fin 1))ᶜ, ∑ a, (inBlock Toy.P1 Toy.W u a).card : ℕ) +
        (1 * 2 + 1 * ({wordR} : Finset (Finset (Fin Toy.C.N))).card : ℕ) +
      1 / 2 * √((∑ u : Fin 1,
            (2 + 1 * (Untied.wordsRead dR.D ({wordR} : Finset (Finset (Fin Toy.C.N))) (Toy.P1.gates u)).card) ^ 2 : ℕ) *
          outOn Toy.P1 Toy.W univ / (1 * 1)) +
      1 / 2 * √((∑ u : Fin 1,
            (2 + 1 * (Untied.wordsRead dR.D ({wordR} : Finset (Finset (Fin Toy.C.N))) (Toy.P1.gates u)).card) ^ 2 : ℕ) *
          (1 * 2 + 1 * ({wordR} : Finset (Finset (Fin Toy.C.N))).card : ℕ) / (1 * 1 * 1)) := by
  obtain ⟨hk, hone, hH, hE, hval, hshare, -⟩ := toyR_premises
  exact Untied.bound_untie_he dR Toy.P1 Toy.W Toy.val (b := 1) ({wordR} : Finset (Finset (Fin Toy.C.N))) (by decide) hk
    hone univ one_pos one_pos one_pos hval hshare (fun _ _ => hH _) hE

/-- The gates each gate of `CZ` reads. -/
def readsZ : Fin 4 → Finset (Fin 4) := ![∅, ∅, {0}, {1, 2}]

/-- Four one-bit gates: a registered input (gate 0), an input `x` outside `θ` (gate 1), gate 2 computed from the
registered input alone, and the MAC (gate 3), which reads gates 1 and 2. -/
abbrev CZ : Circuit where
  N := 4
  w := fun _ => 1
  isInput := fun g => decide (g.val < 2)
  reads := readsZ
  reads_lt := by decide
  input_reads := by decide
  op := fun _ _ => 0
  op_congr := fun _ _ _ _ => rfl
  outputs := {3}

/-- `CZ` with gate 0 registered. -/
abbrev dZ : RegisteredCircuit := .ofCircuit CZ {0} (by decide)

/-- The gate computed from `θ`. -/
abbrev TZ : Finset (Fin CZ.N) := {2}

/-- The values: each word carries its own. -/
abbrev valZ : Finset (Fin CZ.N) → Finset (Fin CZ.N) := id

/-- With gate 2 as an input, one matmul of one MAC: `w = {2}`, `x = {1}`, the MAC and its output gate 3. -/
def MZ : Matmul (CZ.asInputs TZ) where
  ni := 1
  nj := 1
  nl := 1
  w := fun _ => ({2} : Finset (Fin 4))
  x := fun _ => ({1} : Finset (Fin 4))
  y := fun _ => ({3} : Finset (Fin 4))
  w_inj := fun a b _ => Subsingleton.elim a b
  x_inj := fun a b _ => Subsingleton.elim a b
  mac := fun _ => ({3} : Finset (Fin 4))
  mac_computed := by decide
  mac_connected := fun _ => ⟨singleton_nonempty _, fun _ hT hne hT' =>
    absurd ((subset_singleton_iff.1 hT).resolve_left hne.ne_empty) hT'⟩
  reads_w := by decide
  reads_x := by decide
  feeds_y := fun _ => ⟨_, mem_singleton_self _, _, mem_singleton_self _, .refl _⟩

/-- The charged work with gate 2 as an input: `MZ` alone. -/
def WZ : Work (CZ.asInputs TZ) Unit where
  M := fun _ => MZ
  mac_disjoint := by decide

theorem toyZ_allW : WZ.allW = {TZ} := by decide

theorem toyZ_word : TZ ∈ WZ.allW := by
  rw [toyZ_allW]
  exact mem_singleton_self _

/-- At `d` the word is a function of `θ` alone: gates 0 and 2 are closed under reads and hold no input gate outside
`θ`. -/
theorem toyZ_kappa : kappaWords dZ ({TZ} : Finset (Finset (Fin CZ.N))) = 0 :=
  kappaWords_eq_zero_of_closed dZ {TZ} ({0, 2} : Finset (Fin CZ.N)) (by decide) (by decide) (by decide)

/-- With the gate as an input, (H) holds at costs 1 and every `k`: both words are input gates outside `θ` there. -/
theorem toyZ_hardness_asInputs (k : ℕ) :
    Assumptions.Hardness (dZ.asInputs TZ) ![WZ.allW, WZ.allX] valZ ![1, 1] k :=
  hardness_of_input_words (dZ.asInputs TZ) WZ.allW WZ.allX valZ (by decide) (by decide) k

/-- One block holding both computed gates of `CZ`. -/
def PZ : Partition CZ 1 where
  unit := fun g => if g.val < 2 then none else some 0
  input_iff := by decide

/-- The block reads one gate outside `θ`, the `x` gate: the other gate it reads is the registered input. -/
theorem toyZ_reads : ReadsAtMost dZ PZ 1 := by
  unfold ReadsAtMost
  decide

/-- The MAC is its output's one wire, so a set holding the MAC holds a wire of the output. -/
theorem toyZ_clear (B : Finset (Fin (CZ.asInputs TZ).N)) : WZ.clear B = ∅ := by
  refine filter_false_of_mem fun o _ h => ?_
  obtain ⟨⟨s, rfl, hsB⟩, hdis⟩ := h
  exact disjoint_left.1 hdis (mem_singleton_self _) (hsB (mem_singleton_self _))

theorem toyZ_exports (b : ℝ) : Assumptions.Exports WZ b := by
  intro B _
  rw [toyZ_clear, card_empty, Nat.cast_zero, mul_zero]
  exact Nat.cast_nonneg _

theorem toyZ_values : ValuesInjective WZ valZ := fun _ => ⟨Set.injOn_id _, Set.injOn_id _⟩

theorem toyZ_noSharedPair : NoSharedPair WZ valZ := fun a a' h => absurd (Subsingleton.elim a a') h

/-- **Every premise of `Untied.bound_asInputs_he` holds together on a work that multiplies by a word of `T`.** `CZ`
computes gate 2 from `θ` alone; `T` is that gate, and the work, of the circuit with it as an input, multiplies by it.
One block holds both computed gates and reads 1 gate outside `θ` (`ReadsAtMost` at `k = 1`). At the circuit with the
gate as an input, (H) holds at costs 1 and every `k`, so at the block's budget, `k` and the gate of `T` it held; (E)
holds at `b = 1`; each word carries its own value, and the one matmul shares no pair with another. (H) at `d` fails at
every `k` and all positive costs: the word's `κ` at `d` is 0. -/
theorem toyZ_premises :
    (∀ g ∈ TZ, CZ.isInput g = false) ∧ ReadsAtMost dZ PZ 1 ∧
      (∀ k, Assumptions.Hardness (dZ.asInputs TZ) ![WZ.allW, WZ.allX] valZ ![1, 1] k) ∧
      Assumptions.Exports WZ 1 ∧ ValuesInjective WZ valZ ∧ NoSharedPair WZ valZ ∧
      ∀ βw βx : ℝ, 0 < βw → 0 < βx → ∀ k, ¬ Assumptions.Hardness dZ ![WZ.allW, WZ.allX] valZ ![βw, βx] k :=
  ⟨by decide, toyZ_reads, toyZ_hardness_asInputs, toyZ_exports 1, toyZ_values, toyZ_noSharedPair,
    fun _ _ hw hx k => asInputs_hH_false dZ TZ WZ valZ hw hx (Or.inl toyZ_word) toyZ_kappa k⟩

/-- **`Untied.bound_asInputs_he` at `CZ`.** The bound's conclusion at the toy's arguments, with every block in `G`,
costs 1 and `e = 1`, from the conjuncts of `toyZ_premises`: a premise added to the bound breaks this proof. It shows
that the premises hold together on a work that multiplies by a word of `T`, and no more. The bound is slack there, 1 MAC
against `m k + |T| = 2` before the square roots, and (E) holds only because no output is ever in the clear
(`toyZ_clear`): the toy does not show the bound tight. -/
theorem toyZ_bound :
    (WZ.N : ℝ) ≤ (∑ u ∈ (univ : Finset (Fin 1))ᶜ, ∑ a, (inBlock (PZ.asInputs TZ) WZ u a).card : ℕ) +
        (1 * 1 + TZ.card : ℕ) +
      1 / 2 * √((∑ u : Fin 1, (1 + (TZ ∩ PZ.gates u).card) ^ 2 : ℕ) * outOn (PZ.asInputs TZ) WZ univ / (1 * 1)) +
      1 / 2 * √((∑ u : Fin 1, (1 + (TZ ∩ PZ.gates u).card) ^ 2 : ℕ) * (1 * 1 + TZ.card : ℕ) / (1 * 1 * 1)) := by
  obtain ⟨hT, hk, hH, hE, hval, hshare, -⟩ := toyZ_premises
  exact Untied.bound_asInputs_he dZ PZ TZ hT WZ valZ hk univ one_pos one_pos one_pos hval hshare (fun _ _ => hH _) hE

end Nci.Cut.SecurityProofs
