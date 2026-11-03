import Proofs.Nci.Formalism.Slowdown
import Definitions.Nci.Guarantees.Formalism.Witnesses
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.NormNum

/-!
# The training formalism v2: honest and nontrivial lemmas

The proofs of `Security/Definitions/Nci/Guarantees/Formalism/Witnesses.lean`, at the instances of
`Security/Definitions/Nci/Formalism/Instances.lean`. Their gates, reads and token blocks are small finite sets, which
`decide` computes. At `Chain`, the inference blocks at `(c, k) = (1, 1)` are the empty block and the blocks of one gate
(`ib_cases`), so A1 is five counts and every block holds at most one gate (`entryWork_le`); at twice the state, A1's
budget admits only entries where the work reads none of the token's state, and no block entering there holds a gate of
`{3, 4}`. Each bound is compared with its price in `ℝ`. The `example`s at the end apply each guarantee to its lemmas, so
the lemmas are the premises as the guarantee states them.
-/

namespace Nci.Formalism

open Finset
open Verity (Circuit Declaration Partition)
open Nci.Formalism.Assumptions
open scoped NNReal ENNReal

/-- A2 holds at `p = 1` over one method. -/
theorem floor_one {r : Run} {dec : Verity.Decoder (List Bool)} {c : ℕ} {ℓ : List Bool → ℝ} {ℓstar : ℝ} {D : ℕ∞} :
    EfficiencyFloor {r} dec c ℓ ℓstar D 1 := by
  intro r' hr' _ _ hreach
  rw [ENNReal.ofReal_one, one_mul]
  unfold honestCost
  exact iInf_le_of_le r' (iInf_le_of_le hr' (iInf_le_of_le hreach le_rfl))

theorem coe_div_coe {x y : ℝ≥0} (hy : y ≠ 0) : (x : ℝ≥0∞) / (y : ℝ≥0∞) = ((x / y : ℝ≥0) : ℝ≥0∞) :=
  (ENNReal.coe_div hy).symm

theorem natCast_div_natCast {a b : ℕ} (hb : b ≠ 0) :
    (a : ℝ≥0∞) / (b : ℝ≥0∞) = (((a : ℝ≥0) / (b : ℝ≥0) : ℝ≥0) : ℝ≥0∞) := by
  rw [← coe_div_coe (by exact_mod_cast hb), ENNReal.coe_natCast, ENNReal.coe_natCast]

/-- `W_T(x, ν) ≤ n` when every inference block whose work on `T` begins at `x` holds at most `n` gates of `X` in `T`. -/
theorem entryWork_le_of {d : Declaration} {L : Type} (A : Annotation d L) {c k : ℕ} {E : Type}
    {e : Finset (Fin d.D.N) → L → E} {u : Finset (Fin d.D.N) → ℝ≥0} {X : Finset (Fin d.D.N)} {T : L} {x : E}
    {ν : ℝ≥0} {n : ℕ} (h : ∀ I, InferenceBlock d c k I → e I T = x → (I ∩ A.tokens.holds T ∩ X).card ≤ n) :
    A.entryWork c k e u X T x ν ≤ n := by
  classical
  unfold Annotation.entryWork
  refine Finset.sup_le fun I hI => ?_
  rw [mem_filter] at hI
  exact h I hI.2.1 hI.2.2.1

/-- A block of at most `c` gates holds at most `c` gates of any token's work, so `W_T(x, ν) ≤ c`. -/
theorem entryWork_le {d : Declaration} {L : Type} (A : Annotation d L) {c k : ℕ} {E : Type}
    (e : Finset (Fin d.D.N) → L → E) (u : Finset (Fin d.D.N) → ℝ≥0) (X : Finset (Fin d.D.N)) (T : L) (x : E)
    (ν : ℝ≥0) : A.entryWork c k e u X T x ν ≤ c :=
  entryWork_le_of A fun _ hI _ => (card_le_card (inter_subset_left.trans inter_subset_left)).trans hI.2.1

namespace SecurityProofs

namespace Chain

open Nci.Formalism.Chain

theorem computed_card : d.D.computed.card = 4 := by decide

theorem valid : A.Valid := ⟨by decide, by decide, fun T T' h => absurd (Subsingleton.elim T T') h⟩

/-- The inference blocks at `(c, k) = (1, 1)`: the empty block and the blocks of one gate. -/
theorem ib_cases {I : Finset (Fin d.D.N)} (hI : InferenceBlock d 1 1 I) :
    I = ∅ ∨ I = {1} ∨ I = {2} ∨ I = {3} ∨ I = {4} := by
  rcases I.eq_empty_or_nonempty with h | ⟨g, hg⟩
  · exact Or.inl h
  obtain rfl : I = {g} := eq_singleton_iff_unique_mem.2 ⟨hg, fun x hx => card_le_one.1 hI.2.1 x hx g hg⟩
  rcases (by decide : ∀ g : Fin d.D.N, g ∈ d.D.computed → g = 1 ∨ g = 2 ∨ g = 3 ∨ g = 4) g (hI.1 hg) with
    rfl | rfl | rfl | rfl
  · exact Or.inr (Or.inl rfl)
  · exact Or.inr (Or.inr (Or.inl rfl))
  · exact Or.inr (Or.inr (Or.inr (Or.inl rfl)))
  · exact Or.inr (Or.inr (Or.inr (Or.inr rfl)))

theorem ib_three : InferenceBlock d 1 1 {3} := by decide

theorem le_of_card {a b c : ℕ} (h : a + b ≤ c) : (a : ℝ≥0) + (b : ℝ≥0) ≤ (c : ℝ≥0) := by exact_mod_cast h

theorem not_le_of_card {a b c : ℕ} (h : ¬ a + 2 * b ≤ c) : ¬ (a : ℝ≥0) + 2 * (b : ℝ≥0) ≤ (c : ℝ≥0) := by
  exact_mod_cast h

/-- **A1 holds** at `Chain`: gate `2`'s block reads the new weight, one update bit, and gates `3` and `4`'s each read
one value of the token's state. -/
theorem cutCost : CutCost d 1 1 entry state update := by
  intro I hI
  rw [Fin.sum_univ_one]
  rcases ib_cases hI with rfl | rfl | rfl | rfl | rfl <;> exact le_of_card (by decide)

/-- **A1 fails at twice the state**: the block of gate `3` reads one value of the token's state and no update bit. -/
theorem not_cutCost_two : ¬ CutCost d 1 1 entry (fun T x => 2 * state T x) update := by
  intro h
  have h3 := h {3} ib_three
  rw [Fin.sum_univ_one] at h3
  exact not_le_of_card (by decide) h3

/-- Every block holds at most one gate, so the cap `1` bounds the token work at any budget. -/
theorem tokenCapAt_one : A.TokenCapAt 1 1 entry state update {3, 4} 1 := by
  refine ⟨by decide, fun ν f _ => ?_⟩
  rw [Fin.sum_univ_one]
  exact_mod_cast entryWork_le A entry update {3, 4} 0 (f 0) ν

/-- At twice the state, A1's budget admits only entries where the work reads none of the token's state, and no block
entering there holds a gate of `{3, 4}`: gates `3` and `4`'s blocks each read one value of it. -/
theorem tokenCapAt_zero : A.TokenCapAt 1 1 entry (fun T x => 2 * state T x) update {3, 4} 0 := by
  refine ⟨by decide, fun ν f hbud => ?_⟩
  rw [Fin.sum_univ_one] at hbud ⊢
  have h2 : 2 * ((d.D.into (f 0) ∩ A.tokens.holds 0).card : ℝ≥0) ≤ ((1 : ℕ) : ℝ≥0) := le_add_self.trans hbud
  have hn : (d.D.into (f 0) ∩ A.tokens.holds 0).card = 0 := by
    have : 2 * (d.D.into (f 0) ∩ A.tokens.holds 0).card ≤ 1 := by exact_mod_cast h2
    omega
  suffices h : A.entryWork 1 1 entry update {3, 4} 0 (f 0) ν = 0 by
    rw [h, Nat.cast_zero]
  refine Nat.eq_zero_of_le_zero (entryWork_le_of A fun I hIB hentry => ?_)
  rw [← hentry] at hn
  rcases ib_cases hIB with rfl | rfl | rfl | rfl | rfl
  · decide
  · decide
  · decide
  · exact absurd hn (by decide)
  · exact absurd hn (by decide)

theorem gate_three_four : GateSet d.D {3, 4} := by
  unfold GateSet
  decide

theorem gate_one : GateSet d.D {1} := by
  unfold GateSet
  decide

/-- Every block holds at most one gate of the token's gates `{3, 4}`. -/
theorem caps_token : Caps d 1 1 (fun _ : Unit => {3, 4}) (fun _ => 1) := fun _ =>
  ⟨gate_three_four, fun _ hI => by
    dsimp only
    exact_mod_cast (card_le_card inter_subset_left).trans hI.2.1⟩

/-- Every block holds at most the one gate of the new weight. -/
theorem caps_weight : Caps d 1 1 (fun _ : Unit => {1}) (fun _ => 1) := fun _ =>
  ⟨gate_one, fun _ hI => by
    dsimp only
    exact_mod_cast (card_le_card inter_subset_left).trans hI.2.1⟩

theorem blocks_isCK : blocks.IsCK d 1 1 := ⟨⟨fun _ => rfl, by decide⟩, by decide, by decide⟩

theorem not_whole_isCK : ¬ whole.IsCK d 1 1 := fun h => absurd (h.card 0) (by decide)

theorem price_eq (B c : ℕ) : partitionSlowdown d B c = ((((B * c : ℕ) : ℝ≥0) / 4 : ℝ≥0) : ℝ≥0∞) := by
  rw [partitionSlowdown, computed_card, natCast_div_natCast four_ne_zero, Nat.cast_ofNat]

theorem kindBound_one (S : Finset (Fin d.D.N)) {cap : ℝ≥0} (hcap : cap ≠ 0) :
    kindBound d 1 S cap = ((((S.card : ℝ≥0) / 4) / cap : ℝ≥0) : ℝ≥0∞) := by
  rw [kindBound, computed_card, Nat.cast_one, div_one, natCast_div_natCast four_ne_zero, coe_div_coe hcap,
    Nat.cast_ofNat]

/-- A kind with gates and the cap `0` has the bound `⊤`. -/
theorem kindBound_zero {S : Finset (Fin d.D.N)} (hS : S.Nonempty) : kindBound d 1 S 0 = ⊤ := by
  rw [kindBound, ENNReal.coe_zero, ENNReal.zero_div, ENNReal.div_zero]
  exact (ENNReal.div_pos_iff.2 ⟨by exact_mod_cast hS.card_pos.ne', ENNReal.natCast_ne_top _⟩).ne'

theorem honestCost_eq : honestCost methods (fun _ => 0) 0 = ((4 : ℕ) : ℝ≥0∞) := by
  have hreach : r.Reaches (fun _ => 0) 0 := le_rfl
  unfold honestCost
  refine le_antisymm (iInf_le_of_le r (iInf_le_of_le (Set.mem_singleton r) (iInf_le_of_le hreach ?_))) ?_
  · rw [show r.d.D.computed.card = 4 from computed_card]
  · refine le_iInf fun r' => le_iInf fun hr' => le_iInf fun _ => ?_
    obtain rfl := Set.mem_singleton_iff.1 hr'
    rw [show r.d.D.computed.card = 4 from computed_card]

/-- The bound of the kind `{3, 4}` with the cap `1`, `1/2`, is above `whole`'s price `1/4`. -/
theorem not_counting : ¬ (⨆ _K : Unit, kindBound d 1 {3, 4} 1) ≤ partitionSlowdown d 1 1 := by
  rw [iSup_const, kindBound_one _ one_ne_zero, (by decide : ({3, 4} : Finset (Fin d.D.N)).card = 2), price_eq,
    ENNReal.coe_le_coe, ← NNReal.coe_le_coe]
  push_cast
  norm_num

/-- With the token kind's cap `0`, the bound is `⊤`, above `blocks`' price `1`. -/
theorem not_slowdown :
    ¬ max (kindBound d 1 {3, 4} 0) (⨆ _K : Unit, kindBound d 1 {1} 1) ≤ partitionSlowdown d 4 1 := by
  rw [kindBound_zero ⟨3, by decide⟩, max_eq_left le_top, top_le_iff, price_eq]
  exact ENNReal.coe_ne_top

end Chain

namespace Honest

open Nci.Formalism.Honest

theorem boolean : d.D.Boolean := ⟨fun _ => rfl, by decide⟩

/-- Each block reads beyond `θ` the new weight, a description of one value, and one token's one data bit. -/
theorem computeBlock (u : Fin 2) : ComputeBlock d (P.gates u) 1 1 1 :=
  ⟨{2}, fun _ => {1}, by decide, fun _ => (card_singleton _).le, by revert u; decide⟩

theorem sessionBlocks : SessionBlocks d P 1 0 1 1 1 := fun u =>
  ⟨by revert u; decide, by revert u; decide, computeBlock u⟩

theorem trainingBlocks : TrainingBlocks d P 1 0 1 1 1 1 := fun u =>
  ⟨(sessionBlocks u).1, (sessionBlocks u).2.1, Or.inl (computeBlock u)⟩

/-- No partition into inference blocks at `(c, k) = (1, 1)`: gate `3`'s block is gate `3`, which reads two values. -/
theorem not_isCK {B : ℕ} (P' : Partition d.D B) : ¬ P'.IsCK d 1 1 := by
  intro h
  have hne : P'.unit 3 ≠ none := fun h0 => by
    rw [P'.input_iff] at h0
    exact absurd h0 (by decide)
  obtain ⟨u, hu⟩ := Option.ne_none_iff_exists'.1 hne
  have h3 : (3 : Fin d.D.N) ∈ P'.gates u := P'.mem_gates.2 hu
  have hI : P'.gates u = {3} :=
    eq_singleton_iff_unique_mem.2 ⟨h3, fun x hx => card_le_one.1 (h.card u) x hx 3 h3⟩
  have hr := h.reads u
  rw [hI] at hr
  exact absurd hr (by decide)

theorem slowdown_eq_top : slowdown d 1 1 = ⊤ :=
  iInf_eq_top.2 fun _ => iInf_eq_top.2 fun P' => iInf_eq_top.2 fun h => absurd h (not_isCK P')

theorem not_bound : ¬ slowdown d 1 1 ≤ ((1 : ℕ) : ℝ≥0∞) / ((1 - 0 : ℕ) : ℝ≥0∞) := by
  rw [slowdown_eq_top, top_le_iff]
  simp

end Honest

theorem Counting_honest : Guarantees.Counting_honest := ⟨Chain.caps_token, Chain.blocks_isCK⟩

theorem Counting_nontrivial : Guarantees.Counting_nontrivial :=
  ⟨Chain.caps_token, Chain.not_whole_isCK, Chain.not_counting⟩

theorem Slowdown_honest : Guarantees.Slowdown_honest :=
  ⟨Chain.valid, Chain.cutCost, Chain.tokenCapAt_one, Chain.caps_weight, Chain.blocks_isCK⟩

theorem Slowdown_nontrivial : Guarantees.Slowdown_nontrivial :=
  ⟨Chain.valid, Chain.not_cutCost_two, Chain.tokenCapAt_zero, Chain.caps_weight, Chain.blocks_isCK,
    Chain.not_slowdown⟩

theorem Security_honest : Guarantees.Security_honest :=
  ⟨floor_one, Set.mem_singleton _, Chain.valid, le_top, le_rfl, Chain.cutCost, Chain.tokenCapAt_one,
    Chain.caps_weight, Chain.blocks_isCK, le_rfl⟩

theorem Security_nontrivial : Guarantees.Security_nontrivial := by
  refine ⟨floor_one, Set.mem_singleton _, Chain.valid, le_top, le_rfl, Chain.not_cutCost_two, Chain.tokenCapAt_zero,
    Chain.caps_weight, Chain.blocks_isCK, le_rfl, ?_⟩
  rw [show kindBound Chain.r.d 1 {3, 4} 0 = ⊤ from Chain.kindBound_zero ⟨3, by decide⟩, max_eq_left le_top,
    ENNReal.ofReal_one, mul_one, Chain.honestCost_eq, ENNReal.top_mul (by norm_num), top_le_iff]
  exact ENNReal.natCast_ne_top _

theorem FitServing_honest : Guarantees.FitServing_honest := ⟨Honest.boolean, Honest.sessionBlocks, le_rfl⟩

theorem FitServing_nontrivial : Guarantees.FitServing_nontrivial :=
  ⟨Honest.boolean, Honest.sessionBlocks, by unfold ServingFits; decide, Honest.not_bound⟩

theorem FitTraining_honest : Guarantees.FitTraining_honest :=
  ⟨Honest.boolean, Honest.trainingBlocks, le_rfl, le_rfl, le_rfl⟩

theorem FitTraining_nontrivial : Guarantees.FitTraining_nontrivial :=
  ⟨Honest.boolean, Honest.trainingBlocks, fun h => absurd h.1 (by decide), Honest.not_bound⟩

/-! Each lemma is its guarantee's premises as the guarantee states them: the guarantee applies at the honest instance,
and at the nontrivial one the dropped premise would give the bound its last conjunct negates. -/

example : (⨆ _K : Unit, kindBound Chain.d 1 {3, 4} 1) ≤ partitionSlowdown Chain.d 4 1 :=
  Counting Chain.d 1 1 Unit (fun _ => {3, 4}) (fun _ => 1) 4 Chain.blocks Counting_honest.1 Counting_honest.2

example (h : Chain.whole.IsCK Chain.d 1 1) : False :=
  Counting_nontrivial.2.2
    (Counting Chain.d 1 1 Unit (fun _ => {3, 4}) (fun _ => 1) 1 Chain.whole Counting_nontrivial.1 h)

example : max (kindBound Chain.d 1 {3, 4} 1) (⨆ _K : Unit, kindBound Chain.d 1 {1} 1) ≤
    partitionSlowdown Chain.d 4 1 := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := Slowdown_honest
  exact Slowdown Chain.d (Fin 1) Chain.A 1 1 (Finset (Fin Chain.d.D.N)) Chain.entry Chain.state Chain.update {3, 4} 1
    Unit (fun _ => {1}) (fun _ => 1) 4 Chain.blocks h1 h2 h3 h4 h5

example (h2 : CutCost Chain.d 1 1 Chain.entry (fun T x => 2 * Chain.state T x) Chain.update) : False := by
  obtain ⟨h1, -, h3, h4, h5, hlt⟩ := Slowdown_nontrivial
  exact hlt (Slowdown Chain.d (Fin 1) Chain.A 1 1 (Finset (Fin Chain.d.D.N)) Chain.entry
    (fun T x => 2 * Chain.state T x) Chain.update {3, 4} 0 Unit (fun _ => {1}) (fun _ => 1) 4 Chain.blocks h1 h2 h3 h4
    h5)

example : max (kindBound Chain.r.d 1 {3, 4} 1) (⨆ _K : Unit, kindBound Chain.r.d 1 {1} 1) * ENNReal.ofReal 1 *
    honestCost Chain.methods (fun _ => 0) 0 ≤ ((1 * 4 * 1 : ℕ) : ℝ≥0∞) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10⟩ := Security_honest
  exact Security Chain.methods silent (fun _ => 0) 0 ⊤ 1 Chain.r 1 1 4 1 (Finset (Fin Chain.d.D.N)) Chain.entry
    Chain.state Chain.update {3, 4} 1 Unit (fun _ => {1}) (fun _ => 1) 4 Chain.blocks h1 h2 h3 h4 h5 h6 h7 h8 h9 h10

example (h6 : CutCost Chain.r.d 1 1 Chain.entry (fun T x => 2 * Chain.state T x) Chain.update) : False := by
  obtain ⟨h1, h2, h3, h4, h5, -, h7, h8, h9, h10, hlt⟩ := Security_nontrivial
  exact hlt (Security Chain.methods silent (fun _ => 0) 0 ⊤ 1 Chain.r 1 1 4 1 (Finset (Fin Chain.d.D.N)) Chain.entry
    (fun T x => 2 * Chain.state T x) Chain.update {3, 4} 0 Unit (fun _ => {1}) (fun _ => 1) 4 Chain.blocks h1 h2 h3 h4
    h5 h6 h7 h8 h9 h10)

example : slowdown Honest.d 1 2 ≤ ((1 : ℕ) : ℝ≥0∞) / ((1 - 0 : ℕ) : ℝ≥0∞) := by
  obtain ⟨h1, h2, h3⟩ := FitServing_honest
  exact FitServing Honest.d 2 Honest.P 1 2 0 1 1 1 h1 h2 h3

example (h3 : ServingFits 1 1 1 1) : False := by
  obtain ⟨h1, h2, -, hlt⟩ := FitServing_nontrivial
  exact hlt (FitServing Honest.d 2 Honest.P 1 1 0 1 1 1 h1 h2 h3)

example : slowdown Honest.d 1 2 ≤ ((1 : ℕ) : ℝ≥0∞) / ((1 - 0 : ℕ) : ℝ≥0∞) := by
  obtain ⟨h1, h2, h3⟩ := FitTraining_honest
  exact FitTraining Honest.d 2 Honest.P 1 2 0 1 1 1 1 h1 h2 h3

example (h3 : TrainingFits 1 1 1 1 1) : False := by
  obtain ⟨h1, h2, -, hlt⟩ := FitTraining_nontrivial
  exact hlt (FitTraining Honest.d 2 Honest.P 1 1 0 1 1 1 1 h1 h2 h3)

end SecurityProofs

end Nci.Formalism
