import Proofs.Nci.CoveringPriced
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Data.Fin.VecNotation
import Mathlib.Data.Fintype.Powerset
import Mathlib.Data.Fintype.Prod
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.NormNum

/-!
# The priced covering bound is not vacuous, and beats every uniform weight cost

One circuit where every hypothesis of `Nci.Guarantees.CoveringBoundPriced` holds with two price classes, `𝓜*` is one
chained block, and the priced slowdown bound is strictly above `CoveringBoundTwoCosts`'s at every uniform weight cost
whose hypotheses hold (`coveringBoundPriced_nonvacuous`), at `D = 5`, `k = 2`, `s = 4` and `c_a = 1`.

It is `CoveringTwoCostsWitness`'s circuit with one more weight gate, 15 gates. Six weights sit on four weight gates
`u₀ … u₃` (gates 0–3): weights 0 and 1 on `u₀`, 2 and 3 on `u₁`, 4 alone on `u₂` and 5 alone on `u₃`. The token input `x`
(gate 4) is registered (`θ = {x}`), so a block pays only for weight gates. The products `pᵢ = u · x` of weights 0–4
(gates 5–9) feed the activation `a = (((p₀ + p₁) + p₂) + p₃) + p₄` (gates 10–13), and `p₅ = u₃ · a` (gate 14) is chained
through `a` by the five weights 0–4. A weight that shares its gate is priced `1/2` and one alone `1`: a block that
completes a product holds its weight gate as an input, so it pays at least the prices of the weights it completes, at
every `s`. Any five weights cost at least `5/2 > k`. Ten blocks of three gates, each with at most two inputs outside `θ`,
cover it. `𝓜*` is `{p₅}` (`pc_chainedBlocks`): the other products read only input gates, so no activation feeds them.

A uniform weight cost can be at most the cheap class's `1/2` (`pc_uniform_le`): the four-gate block `{u₀, x, p₀, p₁}`
completes two weights from one input outside `θ`. So `CoveringBoundTwoCosts`'s slowdown bound is at most
`(1/15) · 4 / ((2 / (1/2)) · 2 + 2) = 2/75`, while the priced one, at the price ceiling `t = 1`, is
`(1/15) · 4 / (2 · 2 + 1 · 2) = 2/45`. The covering's slowdown is `4 · 10 / 15 = 8/3`.
-/

namespace Nci.SecurityProofs.CoveringPriced

open Verity Finset Nci.Covering

/-- The gates each gate reads. -/
def pcReads : Fin 15 → Finset (Fin 15) :=
  ![∅, ∅, ∅, ∅, ∅, {0, 4}, {0, 4}, {1, 4}, {1, 4}, {2, 4}, {5, 6}, {10, 7}, {11, 8}, {12, 9}, {3, 13}]

/-- The witness circuit, one bit a gate. -/
abbrev pcCircuit : Circuit where
  N := 15
  w := fun _ => 1
  isInput := fun g => decide (g.val < 5)
  reads := pcReads
  reads_lt := by decide
  input_reads := by decide
  op := fun _ _ => 0
  op_congr := fun _ _ _ _ => rfl
  outputs := {14}

/-- The witness declaration: the token input `x` registered. -/
abbrev pcDecl : Declaration := ⟨pcCircuit, {4}, by decide⟩

/-- The product gate of weight `i`. -/
def pcProd : Fin 6 → Fin 15 := ![5, 6, 7, 8, 9, 14]

/-- The weight gate of weight `i`: weights 0–3 two a gate, 4 and 5 one a gate. -/
def pcWeight : Fin 6 → Fin 15 := ![0, 0, 1, 1, 2, 3]

/-- Twice the price of weight `i`. -/
def pcPrice2 : Fin 6 → ℕ := ![1, 1, 1, 1, 2, 2]

/-- The price of weight `i`: `1/2` for a weight that shares its gate, `1` for one alone. -/
noncomputable def pcPrice (i : Fin 6) : ℝ := (pcPrice2 i : ℝ) / 2

/-- One multiplication block per weight, at one token, whose computation is every computed gate. -/
abbrev pcBlocks : MulBlocks pcDecl where
  nw := 6
  nt := 1
  P := univ
  m := fun p => {pcProd p.1}
  o := fun p => pcProd p.1
  T := fun _ => {5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
  m_disjoint := by decide
  m_computed := by decide
  o_mem := by decide
  reach := fun p _ g hg => by
    rw [mem_singleton] at hg
    subst hg
    exact .refl (mem_singleton_self _)
  T_disjoint := fun j j' h => absurd (Subsingleton.elim j j') h
  T_θ := fun _ => by decide

/-- A subcircuit of the witness, its conditions decided. -/
def pcSub (gs is : Finset (Fin 15))
    (h : is ⊆ gs ∧ (∀ g ∈ gs, g ∉ is → pcCircuit.reads g ⊆ gs) ∧ ∀ g ∈ gs, g ∈ sources pcCircuit → g ∈ is) :
    Subcircuit pcDecl.D :=
  ⟨gs, is, h.1, h.2.1, h.2.2⟩

/-- Ten blocks of three gates: the five products, the four additions, and the last product. -/
def pcCover : Fin 10 → Subcircuit pcDecl.D :=
  ![pcSub {0, 4, 5} {0, 4} (by decide), pcSub {0, 4, 6} {0, 4} (by decide), pcSub {1, 4, 7} {1, 4} (by decide),
    pcSub {1, 4, 8} {1, 4} (by decide), pcSub {2, 4, 9} {2, 4} (by decide), pcSub {5, 6, 10} {5, 6} (by decide),
    pcSub {10, 7, 11} {10, 7} (by decide), pcSub {11, 8, 12} {11, 8} (by decide),
    pcSub {12, 9, 13} {12, 9} (by decide), pcSub {3, 13, 14} {3, 13} (by decide)]

theorem pc_covers : Covers pcDecl 4 2 pcCover := by
  refine ⟨?_, ?_⟩
  · unfold InferenceBlock
    decide
  · unfold Subcircuit.inner
    decide

theorem pcPrice_ge (i : Fin 6) : (1 / 2 : ℝ) ≤ pcPrice i := by
  have h : 1 ≤ pcPrice2 i := by revert i; decide
  have h' : (1 : ℝ) ≤ pcPrice2 i := by exact_mod_cast h
  unfold pcPrice
  linarith

theorem pcPrice_le (i : Fin 6) : pcPrice i ≤ 1 := by
  have h : pcPrice2 i ≤ 2 := by revert i; decide
  have h' : (pcPrice2 i : ℝ) ≤ 2 := by exact_mod_cast h
  unfold pcPrice
  linarith

theorem pcPrice_zero : pcPrice 0 = 1 / 2 := by norm_num [pcPrice, pcPrice2]

theorem pcPrice_five : pcPrice 5 = 1 := by norm_num [pcPrice, pcPrice2]

/-- Parameter-hard at the prices, at every `s`: the weights a block completes have their gates among its inputs outside
`θ`, and the prices of the weights on one gate add up to at most `1`. -/
theorem pc_parameterHard (s : ℕ) : Assumptions.ParameterHardPricedAt pcDecl pcBlocks s pcPrice := by
  classical
  intro S _
  have hw : ∀ i : Fin 6, pcWeight i ∈ pcCircuit.reads (pcProd i) ∧ pcWeight i ∈ sources pcCircuit ∧
      pcWeight i ∉ pcDecl.θ := by decide
  have himg : (pcBlocks.weightsCompleted S).image pcWeight ⊆ S.ins \ pcDecl.θ := by
    intro g hg
    obtain ⟨i, hi, rfl⟩ := mem_image.1 hg
    obtain ⟨p, hp, rfl⟩ := mem_image.1 hi
    have hm := (mem_filter.1 hp).2 (mem_singleton_self _)
    obtain ⟨hgs, hgi⟩ := mem_sdiff.1 hm
    exact mem_sdiff.2 ⟨S.sources_ins _ (S.closed _ hgs hgi (hw p.1).1) (hw p.1).2.1, (hw p.1).2.2⟩
  have hkey : ∀ W : Finset (Fin 6), ∑ i ∈ W, pcPrice2 i ≤ 2 * (W.image pcWeight).card := by decide
  have hN : ∑ i ∈ pcBlocks.weightsCompleted S, pcPrice2 i ≤ 2 * (S.ins \ pcDecl.θ).card :=
    (hkey _).trans (Nat.mul_le_mul_left 2 (card_le_card himg))
  have hR : ((∑ i ∈ pcBlocks.weightsCompleted S, pcPrice2 i : ℕ) : ℝ) ≤ 2 * (S.ins \ pcDecl.θ).card := by
    exact_mod_cast hN
  have hsum : ∑ i ∈ pcBlocks.weightsCompleted S, pcPrice i =
      ((∑ i ∈ pcBlocks.weightsCompleted S, pcPrice2 i : ℕ) : ℝ) / 2 := by
    rw [Nat.cast_sum, sum_div]
    rfl
  rw [hsum]
  linarith

/-- Any five weights cost more than `k = 2`: each costs at least `1/2`. -/
theorem pc_depth (W : Finset (Fin 6)) (hW : 5 ≤ W.card) : ((2 : ℕ) : ℝ) < ∑ i ∈ W, pcPrice i := by
  have h5 : (5 : ℝ) ≤ W.card := by exact_mod_cast hW
  have hle : ∑ _i ∈ W, (1 / 2 : ℝ) ≤ ∑ i ∈ W, pcPrice i := sum_le_sum fun i _ => pcPrice_ge i
  rw [sum_const, nsmul_eq_mul] at hle
  push_cast
  linarith

theorem pc_path (i : Fin 6) (hi : i ∈ ({0, 1, 2, 3, 4} : Finset (Fin 6))) :
    FeedsWithin pcDecl.D (pcBlocks.T 0) (pcBlocks.o (i, 0)) 13 := by
  simp only [mem_insert, mem_singleton] at hi
  rcases hi with rfl | rfl | rfl | rfl | rfl
  · exact .step (h := 10) (by decide) (by decide) (.step (h := 11) (by decide) (by decide)
      (.step (h := 12) (by decide) (by decide) (.step (h := 13) (by decide) (by decide) (.refl (by decide)))))
  · exact .step (h := 10) (by decide) (by decide) (.step (h := 11) (by decide) (by decide)
      (.step (h := 12) (by decide) (by decide) (.step (h := 13) (by decide) (by decide) (.refl (by decide)))))
  · exact .step (h := 11) (by decide) (by decide)
      (.step (h := 12) (by decide) (by decide) (.step (h := 13) (by decide) (by decide) (.refl (by decide))))
  · exact .step (h := 12) (by decide) (by decide) (.step (h := 13) (by decide) (by decide) (.refl (by decide)))
  · exact .step (h := 13) (by decide) (by decide) (.refl (by decide))

/-- `p₅` is chained at `(D, c_a) = (5, 1)`, through `a` by the weights 0–4. -/
theorem pc_chained : Assumptions.ActivationChainedAt pcDecl pcBlocks 5 1 (5, 0) := by
  refine ⟨mem_univ _, {13}, {0, 1, 2, 3, 4}, by decide, by decide, by decide,
    fun i hi => ⟨mem_univ _, by revert i; decide, 13, mem_singleton_self _, pc_path i hi⟩, ?_⟩
  intro Q hQ X _ hcut
  rcases X.eq_empty_or_nonempty with rfl | hX
  · have hWQ : 0 < (({0, 1, 2, 3, 4} : Finset (Fin 6)) \ Q).card := by
      have := le_card_sdiff Q ({0, 1, 2, 3, 4} : Finset (Fin 6))
      have h5 : ({0, 1, 2, 3, 4} : Finset (Fin 6)).card = 5 := by decide
      omega
    obtain ⟨i, hi⟩ := card_pos.1 hWQ
    rcases hcut i hi with ⟨x, hx⟩ | hno
    · simp at hx
    · exact absurd (by simpa using pc_path i (mem_sdiff.1 hi).1) (hno 13 (mem_singleton_self _))
  · exact_mod_cast card_pos.2 hX

/-- The products of weights 0–4 are chained at no depth `D > 0`: they read only input gates, so no gate of the token's
computation feeds them. -/
theorem pc_not_chained (i : Fin 6) (hi : i ≠ 5) (D : ℕ) (hD : 0 < D) (ca : ℝ) :
    ¬ Assumptions.ActivationChainedAt pcDecl pcBlocks D ca (i, 0) := by
  rintro ⟨-, A, W, hAT, hAfeed, hDW, hW, -⟩
  obtain ⟨i', hi'⟩ := card_pos.1 (lt_of_lt_of_le hD hDW)
  obtain ⟨-, -, a, ha, -⟩ := hW i' hi'
  obtain ⟨g, hg, hag⟩ := hAfeed a ha
  rw [mem_singleton] at hg
  subst hg
  have key : ∀ i : Fin 6, i ≠ 5 → ∀ a : Fin 15, a ∈ pcReads (pcProd i) →
      a ∉ ({5, 6, 7, 8, 9, 10, 11, 12, 13, 14} : Finset (Fin 15)) := by decide
  exact key i hi a hag (hAT ha)

/-- `𝓜* = {p₅}` at `(D, c_a) = (5, 1)`. -/
theorem pc_chainedBlocks : Assumptions.chainedBlocksAt pcDecl pcBlocks 5 1 = {(5, 0)} := by
  classical
  ext ⟨i, j⟩
  obtain rfl : j = 0 := Subsingleton.elim _ _
  simp only [Assumptions.chainedBlocksAt, mem_filter, mem_singleton, mem_univ, true_and]
  constructor
  · intro h
    by_contra hne
    exact pc_not_chained i (fun h5 => hne (by rw [h5])) 5 (by norm_num) 1 h
  · intro h
    rw [h]
    exact pc_chained

/-- Four gates, `{u₀, x, p₀, p₁}`, that complete weights 0 and 1 from one input outside `θ`. -/
def pcPair : Subcircuit pcDecl.D := pcSub {0, 4, 5, 6} {0, 4} (by decide)

/-- **A uniform weight cost is at most the cheap class's price**: parameter-hard at `c_w` with `s = 4` forces
`c_w ≤ 1/2`, by `pcPair`. -/
theorem pc_uniform_le (cw : ℝ) (h : Assumptions.ParameterHardAt pcDecl pcBlocks 4 cw) : cw ≤ 1 / 2 := by
  classical
  have hW : ({0, 1} : Finset (Fin 6)) ⊆ pcBlocks.weightsCompleted pcPair := by
    intro i hi
    refine mem_image.2 ⟨(i, 0), mem_filter.2 ⟨mem_univ _, ?_⟩, rfl⟩
    unfold MulBlocks.Completes Subcircuit.inner
    revert i
    decide
  have hcard : (2 : ℝ) ≤ (pcBlocks.weightsCompleted pcPair).card := by
    have h2 : ({0, 1} : Finset (Fin 6)).card = 2 := by decide
    exact_mod_cast h2 ▸ card_le_card hW
  have hins : (pcPair.ins \ pcDecl.θ).card = 1 := by decide
  have hh := h pcPair (by decide)
  rw [hins] at hh
  by_contra hc
  push_cast at hh
  nlinarith [not_le.1 hc]

/-- At the cheap class's price `1/2` the two-cost bound's hypotheses hold: the prices are at least `1/2`. -/
theorem pc_parameterHardAt_half (s : ℕ) : Assumptions.ParameterHardAt pcDecl pcBlocks s (1 / 2) := by
  intro S hS
  have hle : ∑ _i ∈ pcBlocks.weightsCompleted S, (1 / 2 : ℝ) ≤ ∑ i ∈ pcBlocks.weightsCompleted S, pcPrice i :=
    sum_le_sum fun i _ => pcPrice_ge i
  rw [sum_const, nsmul_eq_mul] at hle
  linarith [pc_parameterHard s S hS]

/-- **The priced covering bound is not vacuous, and beats every uniform weight cost**: one declaration, its
multiplication blocks and a covering satisfy every hypothesis of `CoveringBoundPriced` with two prices, `1/2` and `1`,
and a chained multiplication block; the two-cost bound's hypotheses hold at some uniform `c_w`, and at every such `c_w`
its slowdown bound is strictly below the priced one at the price ceiling `t = 1`. -/
theorem coveringBoundPriced_nonvacuous :
    ∃ (d : Declaration) (B : MulBlocks d) (s k D : ℕ) (p : Fin B.nw → ℝ) (ca t : ℝ) (n : ℕ)
      (I : Fin n → Subcircuit d.D),
      (∃ i j, p i < p j) ∧ (∀ i, 0 ≤ p i) ∧ (∀ W : Finset (Fin B.nw), D ≤ W.card → (k : ℝ) < ∑ i ∈ W, p i) ∧
      0 < ca ∧ Assumptions.ParameterHardPricedAt d B s p ∧ Covers d s k I ∧
      (Assumptions.chainedBlocksAt d B D ca).Nonempty ∧ (∀ i, p i ≤ t) ∧
      (∃ cw : ℝ, (k : ℝ) < cw * D ∧ Assumptions.ParameterHardAt d B s cw) ∧
      ∀ cw : ℝ, (k : ℝ) < cw * D → Assumptions.ParameterHardAt d B s cw →
        ((Assumptions.chainedBlocksAt d B D ca).card : ℝ) / d.D.N * s / (k / cw * (k / ca) + k) <
          (∑ b ∈ Assumptions.chainedBlocksAt d B D ca, p b.1) / d.D.N * s / (k * (k / ca) + t * k) := by
  refine ⟨pcDecl, pcBlocks, 4, 2, 5, pcPrice, 1, 1, 10, pcCover,
    ⟨0, 5, by rw [pcPrice_zero, pcPrice_five]; norm_num⟩,
    fun i => le_trans (by norm_num) (pcPrice_ge i), pc_depth, one_pos, pc_parameterHard 4, pc_covers,
    ⟨(5, 0), by rw [pc_chainedBlocks]; exact mem_singleton_self _⟩, pcPrice_le,
    ⟨1 / 2, by norm_num, pc_parameterHardAt_half 4⟩, fun cw hkD hhard => ?_⟩
  have hcw0 : 0 < cw := by
    push_cast at hkD
    linarith
  have hcw : cw ≤ 1 / 2 := pc_uniform_le cw hhard
  have h8 : 8 ≤ 2 / cw * 2 := by
    rw [div_mul_eq_mul_div, le_div_iff₀ hcw0]
    linarith
  rw [pc_chainedBlocks, card_singleton, sum_singleton, pcPrice_five]
  push_cast
  rw [div_lt_div_iff₀ (by linarith) (by norm_num)]
  nlinarith

end Nci.SecurityProofs.CoveringPriced
