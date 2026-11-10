module

public import Proofs.Nci.CutWitness
public import Mathlib.Tactic.FinCases

/-!
# Instances against the count over a set of blocks

A scratch file of instances, never merged: what the count over a set `G` of blocks and the bound with a cost for each
block say on small circuits, and what they leave out.

* **(H) in the cap and in the costs.** (H) at a cap is (H) at every smaller cap (`hardness_mono_cap`) and at every
  smaller cost (`hardness_mono_cost`). So (H) at each block of a nonempty `G` at its own cap, with one cost, is (H) at
  the largest cap (`hardness_family_iff`).
* **(H) holds a cost to a word's gates.** `κ(X → Y) ≤ |Y|` (`kappa_le_card_right`), so a cost of (H) is at most the
  number of gates of any word under the cap (`costX_le_card`, `costW_le_card`). On the toy (H) at `β_x = 2` and cap 1
  is false (`toy_not_hardness_two`), at cap 0 it holds at every cost (`toy_hardness_zero`), and with a different cost
  for each of two blocks the premises of the bound hold together (`toy_rows_premises`).
* **The count at each block's own reads**, with the MACs a block holds whole written as a filter and `O` for the
  leaving term, is the count over `G` at those caps (`counts_at_own_reads`).
* **No block in `G`.** The count is then `N ≤ T +` the MACs some block holds whole, with no premise on values or
  outputs (`N_le_whole_add`).
* **Pairs and distinct outputs.** The leaving term counts pairs of a block of `G` and an output with a wire in it.
  The outputs with a wire in a block of `G` are at most the pairs (`values_le_outOn`), and as many where no output has
  a wire in two blocks of `G` (`outOn_eq_values`). On `CT` one output has a wire in each of two blocks: two pairs,
  two wires, one output (`toyT_outOn`, `toyT_wiresOn`, `toyT_values`), and the count's step from the MACs the blocks
  hold to the outputs fails with the distinct outputs in place of the pairs (`toyT_not_values`). With the two gates in
  one block there is one pair and two wires (`toyT_outOn_one`, `toyT_wiresOn_one`).
-/

@[expose] public section

namespace Nci.Cut.Attack

open Verity Finset Nci.Cut.SecurityProofs

/-! ### (H) in the cap and in the costs -/

section Hardness

variable {ρ V : Type} [Fintype ρ] [DecidableEq V] {d : RegisteredCircuit} {A : ρ → Finset (Finset (Fin d.D.N))}
  {val : Finset (Fin d.D.N) → V}

/-- (H) at a cap is (H) at every smaller cap. -/
theorem hardness_mono_cap {β : ρ → ℝ} {k k' : ℕ} (h : Assumptions.Hardness d A val β k) (hk : k' ≤ k) :
    Assumptions.Hardness d A val β k' :=
  fun S hS => h S (hS.trans hk)

/-- (H) at costs `β` is (H) at every smaller cost. -/
theorem hardness_mono_cost {β β' : ρ → ℝ} {k : ℕ} (h : Assumptions.Hardness d A val β k) (hβ : ∀ r, β' r ≤ β r) :
    Assumptions.Hardness d A val β' k := by
  intro S hS
  calc ∑ r, β' r * (((S ∩ A r).image val).card : ℝ)
      ≤ ∑ r, β r * (((S ∩ A r).image val).card : ℝ) :=
        sum_le_sum fun r _ => mul_le_mul_of_nonneg_right (hβ r) (Nat.cast_nonneg _)
    _ ≤ kappaWords d S := h S hS

/-- (H) at each block of a nonempty `G` at its own cap, with one cost, is (H) at the largest cap. -/
theorem hardness_family_iff {m : ℕ} {β : ρ → ℝ} {G : Finset (Fin m)} (hG : G.Nonempty) (κ : Fin m → ℕ) :
    (∀ u ∈ G, Assumptions.Hardness d A val β (κ u)) ↔ Assumptions.Hardness d A val β (G.sup κ) := by
  constructor
  · intro h
    obtain ⟨u, hu, hmax⟩ := G.exists_max_image κ hG
    have hsup : G.sup κ = κ u := le_antisymm (Finset.sup_le hmax) (Finset.le_sup hu)
    rw [hsup]
    exact h u hu
  · intro h u hu
    exact hardness_mono_cap h (Finset.le_sup hu)

end Hardness

/-! ### (H) holds a cost to a word's gates -/

/-- `κ(X → Y) ≤ |Y|`: `Y` meets every path into it. -/
theorem kappa_le_card_right {C : Circuit} (X Y : Finset (Fin C.N)) : kappa C X Y ≤ Y.card :=
  kappa_le fun _ _ _ hy hp => (mem_sdiff.1 (Nci.SecurityProofs.Covering.FeedsWithin.mem_right hp)).2 hy

/-- One word's `κ` is at most its gates. -/
theorem kappaWords_singleton_le (d : RegisteredCircuit) (X0 : Finset (Fin d.D.N)) :
    kappaWords d {X0} ≤ X0.card := by
  unfold kappaWords
  rw [singleton_biUnion]
  exact kappa_le_card_right _ _

/-- (H) holds the cost of an `x`-value to the gates of every `x`-word under the cap: `β_x ≤ |x|`. -/
theorem costX_le_card {ι V : Type} [Fintype ι] [DecidableEq V] {d : RegisteredCircuit} {W : Work d.D ι}
    {val : Finset (Fin d.D.N) → V} {k : ℕ} {βw βx : ℝ} (hw : 0 ≤ βw)
    (hH : Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] k) {X0 : Finset (Fin d.D.N)} (hX : X0 ∈ W.allX)
    (hk : X0.card ≤ k) : βx ≤ X0.card := by
  have hκ : kappaWords d {X0} ≤ X0.card := kappaWords_singleton_le d X0
  have hκ' : (kappaWords d {X0} : ℝ) ≤ X0.card := by exact_mod_cast hκ
  have h := hH {X0} (hκ.trans hk)
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, singleton_inter_of_mem hX, image_singleton,
    card_singleton, Nat.cast_one, mul_one] at h
  have h0 : ∀ t : ℝ, 0 ≤ t → βw * t + βx ≤ kappaWords d {X0} → βx ≤ kappaWords d {X0} :=
    fun t ht h' => by linarith [mul_nonneg hw ht]
  exact (h0 _ (Nat.cast_nonneg _) h).trans hκ'

/-- (H) holds the cost of a `w`-value to the gates of every `w`-word under the cap: `β_w ≤ |w|`. -/
theorem costW_le_card {ι V : Type} [Fintype ι] [DecidableEq V] {d : RegisteredCircuit} {W : Work d.D ι}
    {val : Finset (Fin d.D.N) → V} {k : ℕ} {βw βx : ℝ} (hx : 0 ≤ βx)
    (hH : Assumptions.Hardness d ![W.allW, W.allX] val ![βw, βx] k) {W0 : Finset (Fin d.D.N)} (hW : W0 ∈ W.allW)
    (hk : W0.card ≤ k) : βw ≤ W0.card := by
  have hκ : kappaWords d {W0} ≤ W0.card := kappaWords_singleton_le d W0
  have hκ' : (kappaWords d {W0} : ℝ) ≤ W0.card := by exact_mod_cast hκ
  have h := hH {W0} (hκ.trans hk)
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, singleton_inter_of_mem hW, image_singleton,
    card_singleton, Nat.cast_one, mul_one] at h
  have h0 : ∀ t : ℝ, 0 ≤ t → βw + βx * t ≤ kappaWords d {W0} → βw ≤ kappaWords d {W0} :=
    fun t ht h' => by linarith [mul_nonneg hx ht]
  exact (h0 _ (Nat.cast_nonneg _) h).trans hκ'

/-- On the toy (H) at `β_x = 2` and cap 1 is false: an `x`-word is one gate. -/
theorem toy_not_hardness_two : ¬ Assumptions.Hardness Toy.d Toy.arrays Toy.val ![1, 2] 1 := by
  intro h
  have hX : ({1} : Finset (Fin Toy.C.N)) ∈ Toy.W.allX := by
    rw [toy_allX]
    exact mem_insert_self _ _
  have h2 : (2 : ℝ) ≤ (({1} : Finset (Fin Toy.C.N)).card : ℝ) :=
    costX_le_card (d := Toy.d) (W := Toy.W) (by norm_num) h hX (card_singleton _).le
  rw [card_singleton] at h2
  norm_num at h2

/-- On the toy (H) holds at every cost of at most one gate a value. -/
theorem toy_hardness_le {β : Fin 2 → ℝ} (hβ : ∀ r, β r ≤ 1) (k : ℕ) :
    Assumptions.Hardness Toy.d Toy.arrays Toy.val β k := by
  refine hardness_mono_cost (toy_hardness k) (Fin.forall_fin_two.2 ⟨?_, ?_⟩)
  · simpa using hβ 0
  · simpa using hβ 1

/-- On the toy (H) at cap 0 holds at every cost: no set of words with `κ = 0` holds a `w`- or an `x`-word. -/
theorem toy_hardness_zero (β : Fin 2 → ℝ) : Assumptions.Hardness Toy.d Toy.arrays Toy.val β 0 := by
  intro S hS
  have h := (toy_card_words_le S).trans hS
  have hW : S ∩ Toy.W.allW = ∅ := card_eq_zero.1 (by omega)
  have hX : S ∩ Toy.W.allX = ∅ := card_eq_zero.1 (by omega)
  rw [Fin.sum_univ_two]
  show β 0 * (((S ∩ Toy.W.allW).image Toy.val).card : ℝ) + β 1 * (((S ∩ Toy.W.allX).image Toy.val).card : ℝ) ≤
    kappaWords Toy.d S
  rw [hW, hX, image_empty, card_empty, Nat.cast_zero, mul_zero, mul_zero, add_zero]
  exact Nat.cast_nonneg _

/-- **The premises of the bound with a cost for each block hold together** on the toy, the two blocks at different
costs: positive costs, (H) at each block's cost and cap, (E), the values, and the cap on each block's reads. -/
theorem toy_rows_premises :
    ∃ βx : Fin 2 → ℝ, βx 0 ≠ βx 1 ∧ (∀ u, 0 < βx u) ∧
      (∀ u, Assumptions.Hardness Toy.d Toy.arrays Toy.val ![1, βx u] 2) ∧ Assumptions.Exports Toy.W 1 ∧
      ValuesInjective Toy.W Toy.val ∧ NoSharedPair Toy.W Toy.val ∧ ReadsAtMost Toy.d Toy.P2 2 := by
  refine ⟨![1, 1 / 2], by norm_num, fun u => ?_,
    fun u => toy_hardness_le (β := ![1, (![1, 1 / 2] : Fin 2 → ℝ) u]) (Fin.forall_fin_two.2 ⟨?_, ?_⟩) 2,
    toy_exports 1, toy_values, toy_noSharedPair, toy_reads_P2⟩
  · fin_cases u <;> norm_num
  · norm_num
  · fin_cases u <;> norm_num

/-! ### The count at each block's own reads, and with no block in `G` -/

/-- The count over `G` with each block of `G` at its own reads, the MACs a block holds whole as a filter, and `O` for
the leaving term. -/
theorem counts_at_own_reads {ι : Type} [Fintype ι] [DecidableEq ι] {d : RegisteredCircuit} {m T : ℕ}
    (W : Work d.D ι) (P : Partition d.D m) (G : Finset (Fin m))
    (hT : ∑ u, (d.unregistered (P.gates u)).card ≤ T) :
    ∃ V F E : ℕ, W.N ≤ T + V + ∑ u ∈ Gᶜ, (univ.filter fun s : W.Mac => W.mac s ⊆ P.gates u).card ∧
      V ^ 2 ≤ (∑ u ∈ G, piMax d W (d.unregistered (P.gates u)).card) * F ∧ F ≤ W.O + E ∧
      W.bY * (E : ℝ) ≤ (T : ℝ) ∧ (E = 0 ∨ 0 < W.bY) := by
  obtain ⟨V, F, E, hN, hV, hF, hbE, hb⟩ :=
    counts_on W P G (fun u => (d.unregistered (P.gates u)).card) (fun _ _ => le_rfl) hT
  exact ⟨V, F, E,
    hN.trans_eq (congrArg (T + V + ·) (sum_congr rfl fun u _ => (card_whole_eq P W u).symm)), hV,
    hF.trans (Nat.add_le_add_right (outOn_le P W G) E), hbE, hb⟩

/-- With no block in `G` the count is the MACs some block holds whole and the reads, on no premise. -/
theorem N_le_whole_add {ι : Type} [Fintype ι] [DecidableEq ι] {d : RegisteredCircuit} {m T : ℕ} (W : Work d.D ι)
    (P : Partition d.D m) (hT : ∑ u, (d.unregistered (P.gates u)).card ≤ T) :
    W.N ≤ T + ∑ u, ∑ a, (inBlock P W u a).card := by
  obtain ⟨V, _, _, hN, hV, -, -, -⟩ :=
    counts_on W P ∅ (fun _ => 0) (fun _ hu => absurd hu (notMem_empty _)) hT
  rw [sum_empty, zero_mul] at hV
  have hV0 : V = 0 := by
    by_contra h
    exact absurd hV (not_le.2 (pow_pos (Nat.pos_of_ne_zero h) 2))
  rw [hV0, add_zero, compl_empty] at hN
  exact hN

/-! ### Pairs and distinct outputs -/

/-- The outputs with a wire in a block of `G` are at most the pairs of a block of `G` and an output with a wire in
it. -/
theorem values_le_outOn {ι : Type} [Fintype ι] [DecidableEq ι] {C : Circuit} {m : ℕ} (P : Partition C m)
    (W : Work C ι) (G : Finset (Fin m)) : (G.biUnion (hit P W)).card ≤ outOn P W G :=
  card_biUnion_le

/-- Where no output has a wire in two blocks of `G`, the pairs are the outputs with a wire in a block of `G`. -/
theorem outOn_eq_values {ι : Type} [Fintype ι] [DecidableEq ι] {C : Circuit} {m : ℕ} (P : Partition C m)
    (W : Work C ι) (G : Finset (Fin m))
    (h : ∀ u ∈ G, ∀ v ∈ G, u ≠ v → Disjoint (hit P W u) (hit P W v)) :
    outOn P W G = (G.biUnion (hit P W)).card :=
  (card_biUnion h).symm

/-- The gates each gate of the toy reads. -/
def readsT : Fin 6 → Finset (Fin 6) := ![∅, ∅, ∅, ∅, {0, 1}, {2, 3}]

/-- Six one-bit gates: unregistered inputs `w₀`, `x₀`, `w₁`, `x₁` (gates 0 to 3), and two computed gates,
`w₀ x₀` (gate 4) and `w₁ x₁` (gate 5). -/
abbrev CT : Circuit where
  N := 6
  w := fun _ => 1
  isInput := fun g => decide (g.val < 4)
  reads := readsT
  reads_lt := by decide
  input_reads := by decide
  op := fun _ _ => 0
  op_congr := fun _ _ _ _ => rfl
  outputs := {4, 5}

/-- The toy, nothing registered. -/
abbrev dT : RegisteredCircuit := .ofCircuit CT ∅ (by decide)

/-- `w_0l`'s gate, by `l`. -/
def wGate : Fin 2 → Fin 6 := ![0, 2]

/-- `x_l0`'s gate, by `l`. -/
def xGate : Fin 2 → Fin 6 := ![1, 3]

/-- The MAC `g_00l`'s gate, by `l`. -/
def macGate : Fin 2 → Fin 6 := ![4, 5]

theorem macGate_mem : ∀ l : Fin 2, macGate l ∈ ({4, 5} : Finset (Fin 6)) := by decide

/-- One matmul with one output `y_00 = w_00 x_00 + w_01 x_10`: two MACs, one gate each, and the output's wires are
the two MAC gates. -/
def MT : Matmul CT where
  ni := 1
  nj := 1
  nl := 2
  w := fun p => {wGate p.2}
  x := fun p => {xGate p.1}
  y := fun _ => {4, 5}
  w_inj := by decide
  x_inj := by decide
  mac := fun t => {macGate t.2.2}
  mac_computed := by decide
  mac_connected := fun _ => ⟨singleton_nonempty _, fun _ hT hne hT' =>
    absurd ((subset_singleton_iff.1 hT).resolve_left hne.ne_empty) hT'⟩
  reads_w := by decide
  reads_x := by decide
  feeds_y := fun t => ⟨macGate t.2.2, mem_singleton_self _, macGate t.2.2, macGate_mem t.2.2, .refl _⟩

/-- The charged work: `MT` alone. -/
def WT : Work CT Unit where
  M := fun _ => MT
  mac_disjoint := by decide

/-- A block for each MAC. -/
def PT : Partition CT 2 where
  unit := fun g => if g.val < 4 then none else if g.val = 4 then some 0 else some 1
  input_iff := by decide

/-- One block holding both MACs. -/
def PT1 : Partition CT 1 where
  unit := fun g => if g.val < 4 then none else some 0
  input_iff := by decide

theorem toyT_N : WT.N = 2 := by decide

theorem toyT_O : WT.O = 2 := by decide

/-- The work has one output. -/
theorem toyT_out : Fintype.card WT.Out = 1 := by decide

/-- Two pairs of a block and an output with a wire in it: the one output has a wire in each block. -/
theorem toyT_outOn : outOn PT WT univ = 2 := by decide

/-- Two output wires in the two blocks. -/
theorem toyT_wiresOn : wiresOn PT WT univ = 2 := by decide

/-- One output has a wire in a block. -/
theorem toyT_values : ((univ : Finset (Fin 2)).biUnion (hit PT WT)).card = 1 := by decide

/-- With both MACs in one block: one pair. -/
theorem toyT_outOn_one : outOn PT1 WT univ = 1 := by decide

/-- With both MACs in one block: two output wires. -/
theorem toyT_wiresOn_one : wiresOn PT1 WT univ = 2 := by decide

/-- Each block reads two gates outside `θ`. -/
theorem toyT_reads : ∀ u, (dT.unregistered (PT.gates u)).card = 2 := by decide

/-- The two blocks hold two MACs whole between them. -/
theorem toyT_whole : ∑ u : Fin 2, ∑ a, (inBlock PT WT u a).card = 2 := by decide

theorem toyT_allW : WT.allW = {{0}, {2}} := by decide

theorem toyT_allX : WT.allX = {{1}, {3}} := by decide

theorem toyT_wWords : WT.wWords () = {{0}, {2}} := by decide

theorem toyT_xWords : WT.xWords () = {{1}, {3}} := by decide

/-- Each MAC is a wire of its output, so no set of gates finishes an output it holds no wire of. -/
theorem toyT_clear (B : Finset (Fin CT.N)) : WT.clear B = ∅ := by
  refine filter_false_of_mem fun o _ h => ?_
  obtain ⟨⟨s, rfl, hsB⟩, hdis⟩ := h
  have hmem : ∀ s : WT.Mac, ∃ g, g ∈ WT.mac s ∧ g ∈ WT.y (WT.out s) := by decide
  obtain ⟨g, hg, hgy⟩ := hmem s
  exact disjoint_left.1 hdis hgy (hsB hg)

/-- The toy's words that `S` holds are unregistered inputs in `S`'s words, so at most `κ(S)` of them. -/
theorem toyT_card_words_le (S : Finset (Finset (Fin CT.N))) :
    (S ∩ WT.allW).card + (S ∩ WT.allX).card ≤ kappaWords dT S := by
  have hdisj : Disjoint (S ∩ WT.allW) (S ∩ WT.allX) := by
    refine disjoint_of_subset_left inter_subset_right (disjoint_of_subset_right inter_subset_right ?_)
    rw [toyT_allW, toyT_allX]
    decide
  have hT : (S ∩ WT.allW) ∪ (S ∩ WT.allX) ⊆
      (unregisteredInputs dT ∩ S.biUnion id).image fun g => ({g} : Finset (Fin CT.N)) := by
    intro s hs
    rw [toyT_allW, toyT_allX] at hs
    simp only [mem_union, mem_inter, mem_insert, mem_singleton] at hs
    obtain ⟨hsS, rfl | rfl⟩ | ⟨hsS, rfl | rfl⟩ := hs
    · exact mem_image.2 ⟨(0 : Fin CT.N), mem_inter.2 ⟨by decide, mem_biUnion.2 ⟨_, hsS, mem_singleton_self _⟩⟩, rfl⟩
    · exact mem_image.2 ⟨(2 : Fin CT.N), mem_inter.2 ⟨by decide, mem_biUnion.2 ⟨_, hsS, mem_singleton_self _⟩⟩, rfl⟩
    · exact mem_image.2 ⟨(1 : Fin CT.N), mem_inter.2 ⟨by decide, mem_biUnion.2 ⟨_, hsS, mem_singleton_self _⟩⟩, rfl⟩
    · exact mem_image.2 ⟨(3 : Fin CT.N), mem_inter.2 ⟨by decide, mem_biUnion.2 ⟨_, hsS, mem_singleton_self _⟩⟩, rfl⟩
  calc (S ∩ WT.allW).card + (S ∩ WT.allX).card
      = ((S ∩ WT.allW) ∪ (S ∩ WT.allX)).card := (card_union_of_disjoint hdisj).symm
    _ ≤ ((unregisteredInputs dT ∩ S.biUnion id).image fun g => ({g} : Finset (Fin CT.N))).card :=
      card_le_card hT
    _ ≤ (unregisteredInputs dT ∩ S.biUnion id).card := card_image_le
    _ ≤ kappaWords dT S := card_inter_le_kappa (unregisteredInputs dT) (S.biUnion id)

theorem mul_le_one_of_add_le_two {a b : ℕ} (h : a + b ≤ 2) : a * b ≤ 1 := by
  rcases Nat.lt_or_ge a 2 with ha | ha
  · rcases Nat.eq_zero_or_pos a with rfl | ha0
    · simp
    · have ha1 : a = 1 := by omega
      subst ha1
      omega
  · have hb : b = 0 := by omega
    subst hb
    simp

/-- A set of words with `κ ≤ 2` holds at most two of the toy's words, so at most one pair of a `w`- and an
`x`-word. -/
theorem toyT_pi_le (S : Finset (Finset (Fin CT.N))) (h : kappaWords dT S ≤ 2) : WT.pi S ≤ 1 := by
  have hc := (toyT_card_words_le S).trans h
  rw [toyT_allW, toyT_allX] at hc
  simp only [Work.pi, Fintype.univ_unit, sum_singleton, toyT_wWords, toyT_xWords]
  exact mul_le_one_of_add_le_two hc

/-- `Π(2) ≤ 1` on the toy. -/
theorem toyT_piMax : piMax dT WT 2 ≤ 1 := by
  unfold piMax
  exact Finset.sup_le fun S hS => toyT_pi_le S (mem_filter.1 hS).2

/-- **The count's step fails with distinct outputs in place of pairs.** The two blocks hold two MACs whole, each
block's `Π(2)` is at most 1, one output has a wire in a block, and no block finishes an output it holds no wire of:
`2² ≤ (Π(2) + Π(2)) (1 + 0)` is false. -/
theorem toyT_not_values :
    ¬ ((∑ u : Fin 2, ∑ a, (inBlock PT WT u a).card) ^ 2 ≤
      (∑ _u : Fin 2, piMax dT WT 2) *
        (((univ : Finset (Fin 2)).biUnion (hit PT WT)).card + ∑ u : Fin 2, (WT.clear (PT.gates u)).card)) := by
  intro h
  have hE : ∑ u : Fin 2, (WT.clear (PT.gates u)).card = 0 :=
    sum_eq_zero fun u _ => by rw [toyT_clear, card_empty]
  have h1 := toyT_piMax
  rw [toyT_whole, toyT_values, hE, Fin.sum_univ_two] at h
  omega

end Nci.Cut.Attack
