import Proofs.Nci.Covering

/-!
# The covering bound's hypotheses hold together

An eleven-gate Boolean circuit where `CoveringBound`'s hypotheses all hold with a chained multiplication block to count,
so its premises are not contradictory. Gate 0 is a registered token embedding `x` (the baseline `θ = {0}`), gates 1–4
are the updated weights `w₀, …, w₃`, and the one token's computation `T` is the computed gates 5–10:

* `m₀, m₁, m₂` are gates 5, 6, 7, each `wᵢ · x`;
* gate 8 combines `m₀` and `m₁`, and gate 9 (the activation `A = {9}`) combines gate 8 and `m₂`;
* `m₃` is gate 10, `w₃` times the activation.

Each weight is one bit read by its multiplication, so the circuit is parameter-hard at `r = b = 1` for any `s`
(`parameterHard_of_inputs`). At `D = 3`, `m₃` is chained: the output gates of `m₀, m₁, m₂` all reach the activation inside
`T`, and since fewer than 3 weights leave one of them, no empty set cuts (`c = 1`). Six blocks of at most 3 gates,
each reading at most 2 gates outside `θ`, cover the circuit at `(s, k) = (3, 2)`, with `k = 2 < cD = 3`.
`CoveringBound` then gives `|𝓜*| ≤ |𝓘| · ((k/c)² + k) = 6 · 6`.
-/

namespace Nci.SecurityProofs.CoveringWitness

open Verity Finset Nci.Cover

/-- Parameter-hardness at any `s` and at `c = r·b ≤ 1`, when each weight `i` of `θ'` has a gate `wt i` of its own, an
input of `C` outside `θ` that a gate of each of its multiplication blocks reads: a subcircuit that completes one reads
`wt i`. -/
theorem parameterHard_of_inputs {d : Declaration} {ι τ : Type} [DecidableEq ι] (M : Mults d ι τ) (wt : ι → Fin d.D.N)
    (hinj : Set.InjOn wt M.θ') (hinput : ∀ i ∈ M.θ', d.D.isInput (wt i) = true) (hθ : ∀ i ∈ M.θ', wt i ∉ d.θ)
    (hread : ∀ p ∈ M.P, ∃ g ∈ M.m p.1 p.2, wt p.1 ∈ d.D.reads g) {s r b : ℝ} (hc : r * b ≤ 1) :
    Assumptions.ParameterHard d M s r b := by
  classical
  intro S _
  have hcard : (M.completed S).card ≤ (S.ins \ d.θ).card := by
    refine card_le_card_of_injOn wt (fun i hi => ?_) (hinj.mono fun i hi => ?_)
    · obtain ⟨p, hp, rfl⟩ := mem_image.1 hi
      obtain ⟨hpP, hcomp⟩ := mem_filter.1 hp
      have hθ' := (mem_product.1 (M.P_subset hpP)).1
      obtain ⟨g, hg, hwg⟩ := hread p hpP
      have hgS := mem_sdiff.1 (hcomp hg)
      exact mem_sdiff.2 ⟨S.input_mem _ (S.reads_subset g hgS.1 hgS.2 hwg) (hinput _ hθ'), hθ _ hθ'⟩
    · obtain ⟨p, hp, rfl⟩ := mem_image.1 (mem_coe.1 hi)
      exact (mem_product.1 (M.P_subset (mem_filter.1 hp).1)).1
  calc r * b * (M.completed S).card ≤ (M.completed S).card :=
        mul_le_of_le_one_left (Nat.cast_nonneg _) hc
    _ ≤ (S.ins \ d.θ).card := Nat.cast_le.2 hcard

/-- The circuit: the embedding 0, the weights 1–4, the products 5, 6, 7 and 10, and the activation's gates 8 and 9. -/
abbrev circuit : Circuit where
  N := 11
  w _ := 1
  isInput g := decide (g.val < 5)
  reads g := match g.val with
    | 5 => {1, 0}
    | 6 => {2, 0}
    | 7 => {3, 0}
    | 8 => {5, 6}
    | 9 => {8, 7}
    | 10 => {4, 9}
    | _ => ∅
  reads_lt := by decide +kernel
  input_reads := by decide +kernel
  op _ _ := 0
  op_congr _ _ _ _ := rfl
  outputs := {10}

/-- The registered parameters: the embedding. -/
abbrev decl : Declaration where
  D := circuit
  θ := {0}
  θ_input := by decide +kernel

/-- The output gate of weight `i`'s multiplication with the token. -/
abbrev out : Fin 4 → Fin 11 := ![5, 6, 7, 10]

/-- The four multiplications, one gate each, of the one token, whose computation is the computed gates. -/
abbrev mults : Mults decl (Fin 4) Unit where
  θ' := univ
  J := univ
  P := univ
  P_subset := by decide +kernel
  m i _ := {out i}
  m_computed := by decide +kernel
  o i _ := out i
  o_mem p _ := mem_singleton_self _
  reachesIn_o p _ g hg := by
    rw [mem_singleton.1 hg]
    exact ReachesIn.refl (mem_singleton_self _)
  m_disjoint := by decide +kernel
  T _ := {5, 6, 7, 8, 9, 10}
  T_disjoint_θ := by decide +kernel
  T_disjoint := by decide +kernel

/-- Weight `i`'s gate. -/
abbrev weight : Fin 4 → Fin 11 := ![1, 2, 3, 4]

theorem parameterHard (s : ℝ) : Assumptions.ParameterHard decl mults s 1 1 := by
  have hinj : ∀ i j, weight i = weight j → i = j := by decide +kernel
  exact parameterHard_of_inputs mults weight (fun i _ j _ h => hinj i j h)
    (by decide +kernel) (by decide +kernel) (by decide +kernel) (by norm_num)

/-- The output gates of `m₀, m₁, m₂` reach the activation, gate 9, inside the token's computation. -/
theorem reaches (i' : Fin 4) (hi' : i' ∈ ({0, 1, 2} : Finset (Fin 4))) :
    ∃ a ∈ ({9} : Finset (Fin decl.D.N)), ReachesIn decl.D (mults.T ()) (mults.o i' ()) a := by
  refine ⟨9, mem_singleton_self _, ?_⟩
  simp only [mem_insert, mem_singleton] at hi'
  rcases hi' with rfl | rfl | rfl
  · exact .step (h := 8) (by decide) (by decide) (.step (by decide) (by decide) (.refl (by decide)))
  · exact .step (h := 8) (by decide) (by decide) (.step (by decide) (by decide) (.refl (by decide)))
  · exact .step (h := 9) (by decide) (by decide) (.refl (by decide))

/-- `m₃` is chained at `D = 3`, `r = b = 1`. -/
theorem chained : Assumptions.ActivationChained decl mults 3 1 1 3 () := by
  refine ⟨{9}, by decide, by decide, {0, 1, 2}, by norm_num, fun i' hi' => ⟨mem_univ _, by
    simp only [mem_insert, mem_singleton] at hi'
    rcases hi' with rfl | rfl | rfl <;> decide, reaches i' hi'⟩, ?_⟩
  intro Q _ hQ X _ hcut
  by_contra hX
  have hX0 : X = ∅ := card_eq_zero.1 (by
    have : (X.card : ℝ) < 1 := by linarith
    exact_mod_cast Nat.lt_one_iff.1 (by exact_mod_cast this))
  have hQ2 : Q.card ≤ 2 := by
    have : (Q.card : ℝ) < 3 := hQ
    exact Nat.lt_succ_iff.1 (by exact_mod_cast this)
  have hWQ : 0 < (({0, 1, 2} : Finset (Fin 4)) \ Q).card := by
    have h := le_card_sdiff Q ({0, 1, 2} : Finset (Fin 4))
    have h3 : ({0, 1, 2} : Finset (Fin 4)).card = 3 := by decide
    omega
  obtain ⟨i', hi'⟩ := card_pos.1 hWQ
  rcases hcut i' hi' with hmeet | hno
  · rw [hX0, empty_inter] at hmeet
    exact not_nonempty_empty hmeet
  · rw [hX0, sdiff_empty] at hno
    exact hno (reaches i' (mem_sdiff.1 hi').1)

open Classical in
/-- The chained multiplication blocks are nonempty: they hold `m₃`. -/
theorem chained_nonempty :
    (mults.P.filter fun p => Assumptions.ActivationChained decl mults 3 1 1 p.1 p.2).Nonempty :=
  ⟨(3, ()), mem_filter.2 ⟨mem_univ _, chained⟩⟩

/-- A block of `gates`, reading `ins`. -/
abbrev block (gates ins : Finset (Fin 11)) (h₁ : ins ⊆ gates)
    (h₂ : ∀ g ∈ gates, g ∉ ins → circuit.reads g ⊆ gates) (h₃ : ∀ g ∈ gates, circuit.isInput g = true → g ∈ ins) :
    Subcircuit decl.D :=
  ⟨gates, ins, h₁, h₂, h₃⟩

/-- Six blocks: each product from its weight and the embedding, gate 8 from `m₀, m₁`, gate 9 from gate 8 and `m₂`,
and `m₃` from its weight and gate 9. -/
abbrev blocks : Fin 6 → Subcircuit decl.D := ![
  block {0, 1, 5} {0, 1} (by decide) (by decide) (by decide),
  block {0, 2, 6} {0, 2} (by decide) (by decide) (by decide),
  block {0, 3, 7} {0, 3} (by decide) (by decide) (by decide),
  block {5, 6, 8} {5, 6} (by decide) (by decide) (by decide),
  block {8, 7, 9} {8, 7} (by decide) (by decide) (by decide),
  block {9, 4, 10} {9, 4} (by decide) (by decide) (by decide)]

theorem blocks_injective : Function.Injective blocks := by
  have h : ∀ i j, (blocks i).gates = (blocks j).gates → i = j := by decide +kernel
  exact fun i j hij => h i j (congrArg Subcircuit.gates hij)

/-- The covering: the six blocks. -/
abbrev cover : Finset (Subcircuit decl.D) := univ.map ⟨blocks, blocks_injective⟩

theorem covers : Covers decl 3 2 cover := by
  have hsize : ∀ i, (blocks i).gates.card ≤ 3 ∧ ((blocks i).ins \ decl.θ).card ≤ 2 := by decide +kernel
  have hcomp : ∀ g ∈ decl.D.computed, ∃ i, g ∈ (blocks i).computes := by decide +kernel
  refine ⟨fun I hI => ?_, fun g hg => ?_⟩
  · obtain ⟨i, _, rfl⟩ := mem_map.1 hI
    exact ⟨by exact_mod_cast (hsize i).1, by exact_mod_cast (hsize i).2⟩
  · obtain ⟨i, hi⟩ := hcomp g hg
    exact ⟨blocks i, mem_map_of_mem _ (mem_univ i), hi⟩

open Classical in
/-- **The covering bound's hypotheses are consistent**: at `b = 1`, `r = 1`, `(s, k) = (3, 2)` and `D = 3`, with
`k < r·b·D`, the circuit is parameter-hard, the six blocks cover it, and a multiplication block is chained. -/
theorem nonvacuous : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ∧ (2 : ℝ) < 1 * 1 * 3 ∧
    Assumptions.ParameterHard decl mults 3 1 1 ∧ Covers decl 3 2 cover ∧
    (mults.P.filter fun p => Assumptions.ActivationChained decl mults 3 1 1 p.1 p.2).Nonempty :=
  ⟨one_pos, one_pos, le_rfl, by norm_num, parameterHard 3, covers, chained_nonempty⟩

open Classical in
/-- `CoveringBound` at the witness. -/
theorem bound :
    ((mults.P.filter fun p => Assumptions.ActivationChained decl mults 3 1 1 p.1 p.2).card : ℝ) ≤
      cover.card * ((2 / (1 * 1)) ^ 2 + 2) :=
  (CoveringBound decl mults 1 1 3 2 3 one_pos one_pos le_rfl (by norm_num) (parameterHard 3) cover covers).1

end Nci.SecurityProofs.CoveringWitness
