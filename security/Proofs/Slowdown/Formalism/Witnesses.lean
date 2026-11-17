module

public import Proofs.Slowdown.Formalism.Slowdown
public import Definitions.Slowdown.Guarantees.Formalism.Witnesses
public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Tactic.NormNum

/-! The training formalism v2's honest and nontrivial lemmas, at the instances of
`Definitions/Slowdown/Formalism/Instances.lean`. -/

@[expose] public section

namespace Slowdown.Formalism

open Finset
open Verity (Circuit RegisteredCircuit Partition)
open Slowdown.Formalism.Assumptions
open scoped NNReal ENNReal

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

theorem entryWork_le_of {d : RegisteredCircuit} {L : Type} [Fintype L] (A : Annotation d L) {c k : ℕ} {E : Type}
    {e : Finset (Fin d.D.N) → L → E} {𝒲 : Finset (Finset (Fin d.D.N))} {X : Finset (Fin d.D.N)} {T : L} {x : E}
    {ν n : ℕ}
    (h : ∀ I, InferenceBlock d c k I → e I T = x → A.blockShared e 𝒲 I ≤ ν → (I ∩ A.tokens.holds T ∩ X).card ≤ n) :
    A.entryWork c k e 𝒲 X T x ν ≤ n := by
  classical
  unfold Annotation.entryWork
  refine Finset.sup_le fun I hI => ?_
  rw [mem_filter] at hI
  exact h I hI.2.1 hI.2.2.1 hI.2.2.2

theorem entryWork_le {d : RegisteredCircuit} {L : Type} [Fintype L] (A : Annotation d L) {c k : ℕ} {E : Type}
    (e : Finset (Fin d.D.N) → L → E) (𝒲 : Finset (Finset (Fin d.D.N))) (X : Finset (Fin d.D.N)) (T : L) (x : E)
    (ν : ℕ) : A.entryWork c k e 𝒲 X T x ν ≤ c :=
  entryWork_le_of A fun _ hI _ _ => (card_le_card (inter_subset_left.trans inter_subset_left)).trans hI.2.1

theorem enters_singleton {d : RegisteredCircuit} {L : Type} (A : Annotation d L) {E : Type}
    {e : Finset (Fin d.D.N) → L → E} {W : Finset (Fin d.D.N)} {T : L} {x : E} (hW : A.Enters e {W} T x W) :
    {W' | A.Enters e {W} T x W'} = {W} :=
  Set.ext fun _ => ⟨fun h => Finset.mem_singleton.1 h.1, fun h => (Set.mem_singleton_iff.1 h) ▸ hW⟩

theorem stateAt_of_eq {d : RegisteredCircuit} {L : Type} (A : Annotation d L) {E : Type}
    {e : Finset (Fin d.D.N) → L → E} {𝒲 : Finset (Finset (Fin d.D.N))} {W : Finset (Fin d.D.N)} {T : L} {x : E}
    (h : {W' | A.Enters e 𝒲 T x W'} = {W}) : A.stateAt e 𝒲 T x = A.stateReads W T := by
  rw [Annotation.stateAt, h, Set.image_singleton, csInf_singleton]

theorem readsAt_of_eq {d : RegisteredCircuit} {L : Type} (A : Annotation d L) {E : Type}
    {e : Finset (Fin d.D.N) → L → E} {𝒲 : Finset (Finset (Fin d.D.N))} {W : Finset (Fin d.D.N)} {T : L} {x : E}
    (h : {W' | A.Enters e 𝒲 T x W'} = {W}) : A.readsAt e 𝒲 T x = d.unregistered W := by
  have hW : A.Enters e 𝒲 T x W := (Set.ext_iff.1 h W).2 (Set.mem_singleton W)
  ext g
  simp only [Annotation.readsAt, mem_filter, mem_univ, true_and]
  refine ⟨fun hg => hg.2 W hW, fun hg => ⟨⟨W, hW⟩, fun W' hW' => ?_⟩⟩
  obtain rfl : W' = W := Set.mem_singleton_iff.1 ((Set.ext_iff.1 h W').1 hW')
  exact hg

theorem stateAt_of_empty {d : RegisteredCircuit} {L : Type} (A : Annotation d L) {E : Type}
    {e : Finset (Fin d.D.N) → L → E} {𝒲 : Finset (Finset (Fin d.D.N))} {T : L} {x : E}
    (h : {W' | A.Enters e 𝒲 T x W'} = ∅) : A.stateAt e 𝒲 T x = 0 := by
  rw [Annotation.stateAt, h, Set.image_empty, Nat.sInf_empty]

theorem readsAt_of_empty {d : RegisteredCircuit} {L : Type} (A : Annotation d L) {E : Type}
    {e : Finset (Fin d.D.N) → L → E} {𝒲 : Finset (Finset (Fin d.D.N))} {T : L} {x : E}
    (h : {W' | A.Enters e 𝒲 T x W'} = ∅) : A.readsAt e 𝒲 T x = ∅ := by
  refine Finset.eq_empty_of_forall_notMem fun g hg => ?_
  simp only [Annotation.readsAt, mem_filter, mem_univ, true_and] at hg
  obtain ⟨⟨W, hW⟩, -⟩ := hg
  exact Set.eq_empty_iff_forall_notMem.1 h W hW

namespace SecurityProofs

namespace Chain

open Slowdown.Formalism.Chain

theorem computed_card : d.D.computed.card = 4 := by decide

theorem valid : A.Valid := ⟨by decide, by decide, fun T T' h => absurd (Subsingleton.elim T T') h⟩

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

theorem enters_witnesses (T : Fin 1) (x : Unit) : A.Enters entry witnesses T x {2, 3, 4} := by
  obtain rfl : T = 0 := Subsingleton.elim T 0
  exact ⟨mem_singleton_self _, by unfold GateSet; decide, by decide, fun T' h => absurd (Subsingleton.elim T' 0) h,
    rfl⟩

theorem enters_skipping (T : Fin 1) (x : Unit) : A.Enters entry skipping T x {2, 4} := by
  obtain rfl : T = 0 := Subsingleton.elim T 0
  exact ⟨mem_singleton_self _, by unfold GateSet; decide, by decide, fun T' h => absurd (Subsingleton.elim T' 0) h,
    rfl⟩

theorem stateAt_witnesses (T : Fin 1) (x : Unit) : A.stateAt entry witnesses T x = 0 := by
  rw [stateAt_of_eq A (enters_singleton A (enters_witnesses T x))]
  obtain rfl : T = 0 := Subsingleton.elim T 0
  decide

theorem readsAt_witnesses (T : Fin 1) (x : Unit) : A.readsAt entry witnesses T x = {1} := by
  rw [readsAt_of_eq A (enters_singleton A (enters_witnesses T x))]
  decide

theorem stateAt_skipping (T : Fin 1) (x : Unit) : A.stateAt entry skipping T x = 1 := by
  rw [stateAt_of_eq A (enters_singleton A (enters_skipping T x))]
  obtain rfl : T = 0 := Subsingleton.elim T 0
  decide

theorem readsAt_skipping (T : Fin 1) (x : Unit) : A.readsAt entry skipping T x = {1, 3} := by
  rw [readsAt_of_eq A (enters_singleton A (enters_skipping T x))]
  decide

theorem blockShared_eq {𝒲 : Finset (Finset (Fin d.D.N))} {R : Finset (Fin d.D.N)}
    (hR : ∀ T x, A.readsAt entry 𝒲 T x = R) (I : Finset (Fin d.D.N)) :
    A.blockShared entry 𝒲 I =
      ((((A.worksOn I).biUnion fun _ => R) \ I) \ (A.worksOn I).biUnion fun T => R ∩ A.tokens.holds T).card := by
  simp only [Annotation.blockShared, hR]

theorem cutCost : CutCost A 1 1 entry witnesses := by
  intro I hI
  rw [blockShared_eq readsAt_witnesses]
  simp only [stateAt_witnesses, sum_const_zero, add_zero]
  rcases ib_cases hI with rfl | rfl | rfl | rfl | rfl <;> decide

theorem not_cutCost_skipping : ¬ CutCost A 1 1 entry skipping := by
  intro h
  have h3 := h {3} ib_three
  rw [blockShared_eq readsAt_skipping] at h3
  simp only [stateAt_skipping] at h3
  exact absurd h3 (by decide)

theorem tokenWork_one : TokenWork A 1 1 entry witnesses {3, 4} 1 := by
  refine ⟨by decide, fun S f ν _ => ?_⟩
  calc ∑ T ∈ S, (A.entryWork 1 1 entry witnesses {3, 4} T (f T) ν : ℝ≥0)
      ≤ ∑ T, (A.entryWork 1 1 entry witnesses {3, 4} T (f T) ν : ℝ≥0) := sum_le_sum_of_subset (subset_univ S)
    _ ≤ 1 := by
        rw [Fin.sum_univ_one]
        exact_mod_cast entryWork_le A entry witnesses {3, 4} 0 (f 0) ν

theorem tokenWork_zero : TokenWork A 1 1 entry skipping {3, 4} 0 := by
  refine ⟨by decide, fun S f ν hbud => le_of_eq (sum_eq_zero fun T hT => ?_)⟩
  simp only [stateAt_skipping, sum_const, smul_eq_mul, mul_one] at hbud
  have hν : ν = 0 := by
    have := card_pos.2 ⟨T, hT⟩
    omega
  subst hν
  suffices h : A.entryWork 1 1 entry skipping {3, 4} T (f T) 0 = 0 by rw [h, Nat.cast_zero]
  refine Nat.eq_zero_of_le_zero (entryWork_le_of A fun I hIB _ hu => ?_)
  rw [blockShared_eq readsAt_skipping] at hu
  obtain rfl : T = 0 := Subsingleton.elim T 0
  rcases ib_cases hIB with rfl | rfl | rfl | rfl | rfl
  · decide
  · decide
  · exact absurd hu (by decide)
  · exact absurd hu (by decide)
  · exact absurd hu (by decide)

theorem gate_three_four : GateSet d.D {3, 4} := by
  unfold GateSet
  decide

theorem gate_one : GateSet d.D {1} := by
  unfold GateSet
  decide

theorem caps_token : Caps d 1 1 (fun _ : Unit => {3, 4}) (fun _ => 1) := fun _ =>
  ⟨gate_three_four, fun _ hI => by
    dsimp only
    exact_mod_cast (card_le_card inter_subset_left).trans hI.2.1⟩

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

theorem partitionBound_one : PartitionBound d 1 1 1 := by
  intro m P hP
  rw [partitionSlowdown, ENNReal.le_div_iff_mul_le (Or.inl (by rw [computed_card]; norm_num))
    (Or.inl (ENNReal.natCast_ne_top _)), one_mul]
  exact_mod_cast card_computed_le P hP.card

theorem not_kind_whole : ¬ (⨆ _K : Unit, kindBound d 1 {3, 4} 1) ≤ partitionSlowdown d 1 1 := by
  rw [iSup_const, kindBound_one _ one_ne_zero, (by decide : ({3, 4} : Finset (Fin d.D.N)).card = 2), price_eq,
    ENNReal.coe_le_coe, ← NNReal.coe_le_coe]
  push_cast
  norm_num

theorem not_counting :
    ¬ max (⨆ _K : Unit, kindBound d 1 {3, 4} 1) (⨆ _J : Unit, (1 : ℝ≥0∞)) ≤ partitionSlowdown d 1 1 :=
  fun h => not_kind_whole ((le_max_left _ _).trans h)

theorem not_slowdown :
    ¬ max (kindBound d 1 {3, 4} 0) (max (⨆ _K : Unit, kindBound d 1 {1} 1) (⨆ _J : Unit, (1 : ℝ≥0∞))) ≤
      partitionSlowdown d 4 1 := by
  rw [kindBound_zero ⟨3, by decide⟩, max_eq_left le_top, top_le_iff, price_eq]
  exact ENNReal.coe_ne_top

end Chain

namespace Honest

open Slowdown.Formalism.Honest

theorem boolean : d.D.Boolean := ⟨fun _ => rfl, by decide⟩

theorem computeBlock (u : Fin 2) : ComputeBlock d (P.gates u) 1 1 1 :=
  ⟨{2}, fun _ => {1}, by decide, fun _ => (card_singleton _).le, by revert u; decide⟩

theorem sessionBlocks : SessionBlocks d P 1 0 1 1 1 := fun u =>
  ⟨by revert u; decide, by revert u; decide, computeBlock u⟩

theorem trainingBlocks : TrainingBlocks d P 1 0 1 1 1 1 1 := fun u =>
  ⟨(sessionBlocks u).1, (sessionBlocks u).2.1, Or.inl (computeBlock u)⟩

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

theorem Counting_honest : Guarantees.Counting_honest :=
  ⟨Chain.caps_token, Chain.partitionBound_one, Chain.blocks_isCK⟩

theorem Counting_nontrivial : Guarantees.Counting_nontrivial :=
  ⟨Chain.caps_token, Chain.partitionBound_one, Chain.not_whole_isCK, Chain.not_counting⟩

theorem TokenCap_honest : Guarantees.TokenCap_honest := ⟨Chain.cutCost, Chain.tokenWork_one, Chain.ib_three⟩

theorem TokenCap_nontrivial : Guarantees.TokenCap_nontrivial := by
  refine ⟨Chain.not_cutCost_skipping, Chain.tokenWork_zero, Chain.ib_three, ?_⟩
  rw [(by decide : (({3} : Finset (Fin Chain.d.D.N)) ∩ {3, 4}).card = 1), Nat.cast_one]
  exact not_le.2 one_pos

theorem Slowdown_honest : Guarantees.Slowdown_honest :=
  ⟨Chain.valid, Chain.cutCost, Chain.tokenWork_one, Chain.caps_weight, Chain.partitionBound_one, Chain.blocks_isCK⟩

theorem Slowdown_nontrivial : Guarantees.Slowdown_nontrivial :=
  ⟨Chain.valid, Chain.not_cutCost_skipping, Chain.tokenWork_zero, Chain.caps_weight, Chain.partitionBound_one,
    Chain.blocks_isCK, Chain.not_slowdown⟩

theorem Security_honest : Guarantees.Security_honest :=
  ⟨floor_one, Set.mem_singleton _, Chain.valid, le_top, le_rfl, Chain.cutCost, Chain.tokenWork_one,
    Chain.caps_weight, Chain.partitionBound_one, Chain.blocks_isCK, le_rfl⟩

theorem Security_nontrivial : Guarantees.Security_nontrivial := by
  refine ⟨floor_one, Set.mem_singleton _, Chain.valid, le_top, le_rfl, Chain.not_cutCost_skipping,
    Chain.tokenWork_zero, Chain.caps_weight, Chain.partitionBound_one, Chain.blocks_isCK, le_rfl, ?_⟩
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

example : max (⨆ _K : Unit, kindBound Chain.d 1 {3, 4} 1) (⨆ _J : Unit, (1 : ℝ≥0∞)) ≤
    partitionSlowdown Chain.d 4 1 := by
  obtain ⟨h1, h2, h3⟩ := Counting_honest
  exact Counting Chain.d 1 1 Unit (fun _ => {3, 4}) (fun _ => 1) Unit (fun _ => 1) 4 Chain.blocks h1 (fun _ => h2) h3

example (h3 : Chain.whole.IsCK Chain.d 1 1) : False := by
  obtain ⟨h1, h2, -, hlt⟩ := Counting_nontrivial
  exact hlt (Counting Chain.d 1 1 Unit (fun _ => {3, 4}) (fun _ => 1) Unit (fun _ => 1) 1 Chain.whole h1 (fun _ => h2)
    h3)

example : ((({3} : Finset (Fin Chain.d.D.N)) ∩ {3, 4}).card : ℝ≥0) ≤ 1 := by
  obtain ⟨h1, h2, h3⟩ := TokenCap_honest
  exact TokenCap Chain.d (Fin 1) Chain.A 1 1 Unit Chain.entry Chain.witnesses {3, 4} 1 h1 h2 {3} h3

example (h1 : CutCost Chain.A 1 1 Chain.entry Chain.skipping) : False := by
  obtain ⟨-, h2, h3, hlt⟩ := TokenCap_nontrivial
  exact hlt (TokenCap Chain.d (Fin 1) Chain.A 1 1 Unit Chain.entry Chain.skipping {3, 4} 0 h1 h2 {3} h3)

example : max (kindBound Chain.d 1 {3, 4} 1) (max (⨆ _K : Unit, kindBound Chain.d 1 {1} 1) (⨆ _J : Unit, (1 : ℝ≥0∞))) ≤
    partitionSlowdown Chain.d 4 1 := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := Slowdown_honest
  exact Slowdown Chain.d (Fin 1) Chain.A 1 1 Unit Chain.entry Chain.witnesses {3, 4} 1 Unit (fun _ => {1})
    (fun _ => 1) Unit (fun _ => 1) 4 Chain.blocks h1 h2 h3 h4 (fun _ => h5) h6

example (h2 : CutCost Chain.A 1 1 Chain.entry Chain.skipping) : False := by
  obtain ⟨h1, -, h3, h4, h5, h6, hlt⟩ := Slowdown_nontrivial
  exact hlt (Slowdown Chain.d (Fin 1) Chain.A 1 1 Unit Chain.entry Chain.skipping {3, 4} 0 Unit (fun _ => {1})
    (fun _ => 1) Unit (fun _ => 1) 4 Chain.blocks h1 h2 h3 h4 (fun _ => h5) h6)

example : max (kindBound Chain.r.d 1 {3, 4} 1)
    (max (⨆ _K : Unit, kindBound Chain.r.d 1 {1} 1) (⨆ _J : Unit, (1 : ℝ≥0∞))) * ENNReal.ofReal 1 *
      honestCost Chain.methods (fun _ => 0) 0 ≤ ((1 * 4 * 1 : ℕ) : ℝ≥0∞) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := Security_honest
  exact Security Chain.methods silent (fun _ => 0) 0 ⊤ 1 Chain.r 1 1 4 1 Unit Chain.entry Chain.witnesses {3, 4} 1
    Unit (fun _ => {1}) (fun _ => 1) Unit (fun _ => 1) 4 Chain.blocks h1 h2 h3 h4 h5 h6 h7 h8 (fun _ => h9) h10 h11

example (h6 : CutCost Chain.r.A 1 1 Chain.entry Chain.skipping) : False := by
  obtain ⟨h1, h2, h3, h4, h5, -, h7, h8, h9, h10, h11, hlt⟩ := Security_nontrivial
  exact hlt (Security Chain.methods silent (fun _ => 0) 0 ⊤ 1 Chain.r 1 1 4 1 Unit Chain.entry Chain.skipping {3, 4} 0
    Unit (fun _ => {1}) (fun _ => 1) Unit (fun _ => 1) 4 Chain.blocks h1 h2 h3 h4 h5 h6 h7 h8 (fun _ => h9) h10 h11)

example : slowdown Honest.d 1 2 ≤ ((1 : ℕ) : ℝ≥0∞) / ((1 - 0 : ℕ) : ℝ≥0∞) := by
  obtain ⟨h1, h2, h3⟩ := FitServing_honest
  exact FitServing Honest.d 2 Honest.P 1 2 0 1 1 1 h1 h2 h3

example (h3 : ServingFits 1 1 1 1) : False := by
  obtain ⟨h1, h2, -, hlt⟩ := FitServing_nontrivial
  exact hlt (FitServing Honest.d 2 Honest.P 1 1 0 1 1 1 h1 h2 h3)

example : slowdown Honest.d 1 2 ≤ ((1 : ℕ) : ℝ≥0∞) / ((1 - 0 : ℕ) : ℝ≥0∞) := by
  obtain ⟨h1, h2, h3⟩ := FitTraining_honest
  exact FitTraining Honest.d 2 Honest.P 1 2 0 1 1 1 1 1 h1 h2 h3

example (h3 : TrainingFits 1 1 1 1 1 1) : False := by
  obtain ⟨h1, h2, -, hlt⟩ := FitTraining_nontrivial
  exact hlt (FitTraining Honest.d 2 Honest.P 1 1 0 1 1 1 1 1 h1 h2 h3)

namespace Pair

abbrev C : Circuit where
  N := 5
  w _ := 1
  isInput := ![true, false, false, false, false]
  reads := ![∅, {0}, {1}, {1, 2}, {2, 3}]
  reads_lt := by decide
  input_reads := by decide
  op g v := if g = 1 then v 0 else if g = 2 then v 1 else if g = 3 then v 1 &&& v 2 else if g = 4 then v 2 &&& v 3
    else 0
  op_congr g v v' hv := by
    fin_cases g
    · rfl
    · simpa using hv 0 (by decide)
    · simpa using hv 1 (by decide)
    · simp [hv 1 (by decide), hv 2 (by decide)]
    · simp [hv 2 (by decide), hv 3 (by decide)]
  outputs := {4}

abbrev d : RegisteredCircuit := .ofCircuit C {0} (by decide)

abbrev A : Annotation d (Fin 3) := ⟨{1}, ⟨![{2}, {3}, {4}]⟩⟩

abbrev entry (_ : Finset (Fin d.D.N)) (_ : Fin 3) : Unit := ()

abbrev witnesses : Finset (Finset (Fin d.D.N)) := {{3}, {4}}

theorem valid : A.Valid :=
  ⟨by decide, by decide, fun T T' h => by fin_cases T <;> fin_cases T' <;> first | exact absurd rfl h | decide⟩

theorem enters_zero (x : Unit) : {W | A.Enters entry witnesses 0 x W} = ∅ := by
  refine Set.eq_empty_of_forall_notMem fun W hW => ?_
  obtain ⟨hW, -, hne, -, -⟩ := hW
  rcases mem_insert.1 hW with rfl | hW
  · exact absurd hne (by decide)
  · rw [mem_singleton.1 hW] at hne
    exact absurd hne (by decide)

theorem enters_one (x : Unit) : {W | A.Enters entry witnesses 1 x W} = {{3}} := by
  ext W
  simp only [Set.mem_ofPred_eq, Set.mem_singleton_iff]
  constructor
  · rintro ⟨hW, -, hne, -, -⟩
    rcases mem_insert.1 hW with rfl | hW
    · rfl
    · rw [mem_singleton.1 hW] at hne
      exact absurd hne (by decide)
  · rintro rfl
    exact ⟨by decide, by unfold GateSet; decide, by decide, by decide, Subsingleton.elim _ _⟩

theorem enters_two (x : Unit) : {W | A.Enters entry witnesses 2 x W} = {{4}} := by
  ext W
  simp only [Set.mem_ofPred_eq, Set.mem_singleton_iff]
  constructor
  · rintro ⟨hW, -, hne, -, -⟩
    rcases mem_insert.1 hW with rfl | hW
    · exact absurd hne (by decide)
    · exact mem_singleton.1 hW
  · rintro rfl
    exact ⟨by decide, by unfold GateSet; decide, by decide, by decide, Subsingleton.elim _ _⟩

theorem stateAt_eq (T : Fin 3) (x : Unit) : A.stateAt entry witnesses T x = 0 := by
  fin_cases T
  · exact stateAt_of_empty A (enters_zero x)
  · exact (stateAt_of_eq A (enters_one x)).trans (by decide)
  · exact (stateAt_of_eq A (enters_two x)).trans (by decide)

theorem readsAt_eq (T : Fin 3) (x : Unit) : A.readsAt entry witnesses T x = ![∅, {1, 2}, {2, 3}] T := by
  fin_cases T
  · exact readsAt_of_empty A (enters_zero x)
  · exact (readsAt_of_eq A (enters_one x)).trans (by decide)
  · exact (readsAt_of_eq A (enters_two x)).trans (by decide)

theorem cutCost : CutCost A 2 2 entry witnesses := by
  intro I hI
  simp only [Annotation.blockShared, readsAt_eq, stateAt_eq, sum_const_zero, add_zero]
  revert I
  decide

theorem charge : InferenceBlock d 2 2 {3, 4} ∧
    A.blockShared entry witnesses {3, 4} + ∑ T ∈ A.worksOn {3, 4}, A.stateAt entry witnesses T (entry {3, 4} T) = 2 ∧
    (d.unregistered {3, 4}).card = 2 := by
  simp only [Annotation.blockShared, readsAt_eq, stateAt_eq, sum_const_zero, add_zero]
  decide

end Pair

end SecurityProofs

end Slowdown.Formalism
