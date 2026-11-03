import Proofs.Nci.Formalism.Slowdown
import Definitions.Nci.Guarantees.Formalism.Witnesses

/-!
# The training formalism v2: honest and nontrivial lemmas

The proofs of `Security/Definitions/Nci/Guarantees/Formalism/Witnesses.lean`, at the instances of
`Security/Definitions/Nci/Formalism/Instances.lean`. Their gates, reads and token blocks are small finite sets, which
`decide` computes. At `Chain`, `A` is the token's three gates, since a gate feeds only later gates (`le_of_feeds`); the
inference blocks at `(c, k) = (1, 1)` are the empty block and the blocks of one gate (`ib_cases`), of which only gate
`2`'s computes the token from its inputs, so `R` is gate `2`, and every count and share follows. Each bound is compared
with its price in `ℝ`. The `example`s at the end apply each guarantee to its lemmas, so the lemmas are the premises as
the guarantee states them.
-/

namespace Nci.Formalism

open Finset
open Verity (Circuit Declaration Partition)
open Nci.Formalism.Assumptions
open scoped NNReal ENNReal

/-- A gate feeds only itself and later gates. -/
theorem le_of_feeds {C : Circuit} {h g : Fin C.N} (hf : C.Feeds h g) : h ≤ g := by
  induction hf with
  | refl => exact le_rfl
  | step hr _ ih => exact (C.reads_lt _ _ hr).le.trans ih

theorem mem_affected {d : Declaration} {L : Type} (A : Annotation d L) {g : Fin d.D.N} :
    g ∈ A.affected ↔ g ∈ d.D.computed ∧ ∃ v ∈ A.θ', DependsOn d.D g v := by
  classical
  rw [Annotation.affected, mem_filter]

theorem mem_dependencies {d : Declaration} {L : Type} (A : Annotation d L) {T : L} {g h : Fin d.D.N} :
    h ∈ A.dependencies T g ↔ h ∈ A.tokenAffected T ∧ FeedsWithin d.D (A.tokens.holds T) h g := by
  classical
  rw [Annotation.dependencies, mem_filter]

/-- A token's dependency profile at `x` is at most `|S| / |T ∩ A|` when `S` holds each of its affected gates whose
dependency share is at most `x`. -/
theorem dependencyProfile_le {d : Declaration} {L : Type} (A : Annotation d L) (T : L) (x : ℝ≥0)
    {S : Finset (Fin d.D.N)} (h : ∀ g ∈ A.tokenAffected T, A.dependencyShare T g ≤ x → g ∈ S) :
    A.dependencyProfile T x ≤ (S.card : ℝ≥0) / ((A.tokenAffected T).card : ℝ≥0) := by
  classical
  unfold Annotation.dependencyProfile
  refine div_le_div_of_nonneg_right (Nat.cast_le.2 (card_le_card fun g hg => ?_)) zero_le
  rw [mem_filter] at hg
  exact h g hg.1 hg.2

/-- A block that computes from its inputs every token whose affected gates it holds cuts none. -/
theorem cuts_eq_zero {d : Declaration} {L : Type} [Fintype L] (A : Annotation d L) {I : Finset (Fin d.D.N)}
    (h : ∀ T, (I ∩ A.tokenAffected T).Nonempty → A.FromInputs I T) : A.cuts I = 0 := by
  classical
  rw [Annotation.cuts, card_eq_zero, filter_eq_empty_iff]
  exact fun T _ hT => hT.2 (h T hT.1)

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

theorem third_le_one : (1 / 3 : ℝ≥0) ≤ 1 := by
  rw [← NNReal.coe_le_coe]
  push_cast
  norm_num

theorem one_sub_third : (1 : ℝ≥0) - 1 / 3 = 2 / 3 := by
  rw [← NNReal.coe_inj, NNReal.coe_sub third_le_one]
  push_cast
  norm_num

namespace SecurityProofs

namespace Chain

open Nci.Formalism.Chain

theorem computed_card : d.D.computed.card = 4 := by decide

theorem valid : A.Valid := ⟨by decide, by decide, fun T T' h => absurd (Subsingleton.elim T T') h⟩

theorem gateSet : GateSet d.D {3, 4} := by
  unfold GateSet
  decide

theorem affected_eq : A.affected = {2, 3, 4} := by
  ext g
  rw [mem_affected]
  constructor
  · rintro ⟨-, v, hv, h, hvh, hf⟩
    obtain rfl : v = 1 := mem_singleton.1 hv
    obtain rfl : h = 2 := (by decide : ∀ h : Fin d.D.N, (1 : Fin d.D.N) ∈ d.D.reads h → h = 2) h hvh
    exact (by decide : ∀ g : Fin d.D.N, 2 ≤ g → g ∈ ({2, 3, 4} : Finset (Fin d.D.N))) g (le_of_feeds hf)
  · intro hg
    refine ⟨(by decide : ({2, 3, 4} : Finset (Fin d.D.N)) ⊆ d.D.computed) hg, 1, mem_singleton_self 1, 2, by decide,
      ?_⟩
    rcases mem_insert.1 hg with rfl | hg
    · exact .refl 2
    rcases mem_insert.1 hg with rfl | hg
    · exact .step (by decide) (.refl 3)
    obtain rfl := mem_singleton.1 hg
    exact .step (h := 3) (by decide) (.step (by decide) (.refl 4))

theorem tokenAffected_eq (T : Fin 1) : A.tokenAffected T = {2, 3, 4} := by
  rw [Annotation.tokenAffected, affected_eq]
  exact inter_self _

theorem card_tokenAffected (T : Fin 1) : (A.tokenAffected T).card = 3 := by
  rw [tokenAffected_eq]
  decide

theorem identical : A.Identical 3 := ⟨by norm_num, card_tokenAffected⟩

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

theorem ib_two : InferenceBlock d 1 1 {2} := by decide

theorem ib_three : InferenceBlock d 1 1 {3} := by decide

theorem fromInputs_two (T : Fin 1) : A.FromInputs {2} T := by
  revert T
  unfold Annotation.FromInputs
  decide

theorem not_fromInputs_three (T : Fin 1) : ¬ A.FromInputs {3} T := fun h =>
  h 2 (by decide) (show (2 : Fin d.D.N) ∈ ({2, 3, 4} : Finset (Fin d.D.N)) by decide)

theorem not_fromInputs_four (T : Fin 1) : ¬ A.FromInputs {4} T := fun h =>
  h 3 (by decide) (show (3 : Fin d.D.N) ∈ ({2, 3, 4} : Finset (Fin d.D.N)) by decide)

/-- `R` is gate `2`: its block computes the token from its inputs, and gates `3` and `4`'s read the token. -/
theorem reachable_eq : A.reachable 1 1 = {2} := by
  ext g
  rw [mem_reachable, mem_singleton]
  constructor
  · rintro ⟨hgA, T, -, I, hI, hgI, hfrom⟩
    rcases ib_cases hI with rfl | rfl | rfl | rfl | rfl
    · exact absurd hgI (notMem_empty g)
    · obtain rfl := mem_singleton.1 hgI
      rw [affected_eq] at hgA
      exact absurd hgA (by decide)
    · exact mem_singleton.1 hgI
    · exact absurd hfrom (not_fromInputs_three T)
    · exact absurd hfrom (not_fromInputs_four T)
  · rintro rfl
    exact ⟨by rw [affected_eq]; decide, 0, by decide, {2}, ib_two, mem_singleton_self 2, fromInputs_two 0⟩

theorem reachableShare_eq (T : Fin 1) : A.reachableShare 1 1 T = 1 / 3 := by
  rw [Annotation.reachableShare, tokenAffected_eq, reachable_eq,
    (by decide : (({2, 3, 4} : Finset (Fin d.D.N)) ∩ {2}).card = 1),
    (by decide : ({2, 3, 4} : Finset (Fin d.D.N)).card = 3)]
  norm_num

theorem reachableAtMost : A.ReachableAtMost 1 1 (1 / 3) := fun T => (reachableShare_eq T).le

theorem unreachable_eq : A.affected \ A.reachable 1 1 = {3, 4} := by
  rw [affected_eq, reachable_eq]
  decide

theorem unreachableWork_eq (I : Finset (Fin d.D.N)) :
    A.unreachableWork 1 1 I = ((I ∩ {3, 4}).card : ℝ≥0) / 3 := by
  rw [Annotation.unreachableWork, Fintype.sum_unique, tokenAffected_eq, reachable_eq,
    (by decide : ({2, 3, 4} : Finset (Fin d.D.N)) \ {2} = {3, 4}),
    (by decide : ({2, 3, 4} : Finset (Fin d.D.N)).card = 3)]
  norm_num

theorem unreachableCapacity_eq : A.unreachableCapacity 1 1 = 1 / 3 := by
  refine le_antisymm (Finset.sup_le fun I hI => ?_) ?_
  · rw [unreachableWork_eq]
    have h : (I ∩ {3, 4}).card ≤ 1 := (card_le_card inter_subset_left).trans (mem_filter.1 hI).2.2.1
    exact div_le_div_of_nonneg_right (by exact_mod_cast h) zero_le
  · have h := unreachableWork_le_capacity A ib_three
    rwa [unreachableWork_eq, (by decide : (({3} : Finset (Fin d.D.N)) ∩ {3, 4}).card = 1), Nat.cast_one] at h

theorem maxInBlock_eq : maxInBlock d 1 1 {3, 4} = 1 := by
  refine le_antisymm (Finset.sup_le fun I hI => (card_le_card inter_subset_left).trans (mem_filter.1 hI).2.2.1) ?_
  calc 1 = (({3} : Finset (Fin d.D.N)) ∩ {3, 4}).card := by decide
    _ ≤ maxInBlock d 1 1 {3, 4} := card_le_maxInBlock ib_three

theorem tokenShare_eq : A.tokenShare = 3 / 4 := by
  rw [Annotation.tokenShare, (by decide : A.tokenGates.card = 3), computed_card]
  norm_num

theorem tokensPerBlock_eq : A.tokensPerBlock 1 = 1 / 3 := by
  rw [Annotation.tokensPerBlock, (by decide : A.work = 3)]
  norm_num

theorem maxAffected_eq : A.maxAffected = 3 :=
  le_antisymm (Finset.sup_le fun T _ => (card_tokenAffected T).le)
    ((card_tokenAffected 0).symm.le.trans (le_sup (f := fun T => (A.tokenAffected T).card) (mem_univ 0)))

theorem affected_inter_card : (A.affected ∩ A.tokenGates).card = 3 := by
  rw [affected_eq]
  decide

theorem share_eq (T : Fin 1) (I : Finset (Fin d.D.N)) : A.share T I = ((I ∩ {2, 3, 4}).card : ℝ≥0) / 3 := by
  rw [Annotation.share, tokenAffected_eq, (by decide : ({2, 3, 4} : Finset (Fin d.D.N)).card = 3)]
  norm_num

theorem maxShare_eq (I : Finset (Fin d.D.N)) : A.maxShare I = ((I ∩ {2, 3, 4}).card : ℝ≥0) / 3 := by
  rw [← share_eq 0 I]
  exact le_antisymm (Finset.sup_le fun T _ => by rw [Subsingleton.elim T 0]) (share_le_maxShare A 0 I)

theorem inter_le_computedReads {I : Finset (Fin d.D.N)} (hI : InferenceBlock d 1 1 I) :
    (I ∩ {2, 3, 4}).card ≤ (computedReads d I).card := by
  rcases ib_cases hI with rfl | rfl | rfl | rfl | rfl <;> decide

theorem inter_le_unregistered {I : Finset (Fin d.D.N)} (hI : InferenceBlock d 1 1 I) :
    (I ∩ {2, 3, 4}).card ≤ (d.unregistered I).card := by
  rcases ib_cases hI with rfl | rfl | rfl | rfl | rfl <;> decide

theorem cuts_le_one (I : Finset (Fin d.D.N)) : A.cuts I ≤ 1 :=
  (card_le_univ _).trans_eq (Fintype.card_fin 1)

/-- A block cuts the token only when it reads a computed value. -/
theorem cuts_le_computedReads {I : Finset (Fin d.D.N)} (hI : InferenceBlock d 1 1 I) :
    A.cuts I ≤ (computedReads d I).card := by
  rcases ib_cases hI with rfl | rfl | rfl | rfl | rfl
  · rw [cuts_eq_zero A fun T h => absurd h (by simp)]
    exact Nat.zero_le _
  · rw [cuts_eq_zero A fun T h => absurd h (by rw [tokenAffected_eq]; decide)]
    exact Nat.zero_le _
  · rw [cuts_eq_zero A fun T _ => fromInputs_two T]
    exact Nat.zero_le _
  · exact (cuts_le_one _).trans (by decide)
  · exact (cuts_le_one _).trans (by decide)

theorem cuts_three : A.cuts {3} = 1 := by
  classical
  refine le_antisymm (cuts_le_one _) ?_
  rw [Annotation.cuts]
  refine one_le_card.2 ⟨0, ?_⟩
  rw [mem_filter, tokenAffected_eq]
  exact ⟨mem_univ 0, by decide, not_fromInputs_three 0⟩

/-- **A1 fails** at the block of gate `3` when `u(1/3) + a > 1`: it does a third of the token's work, cuts it, and reads
one computed value. -/
theorem not_cutCost {u : ℝ≥0 → ℝ≥0} {a : ℝ≥0} (h : 1 < u (1 / 3) + a) : ¬ CutCost d A 1 1 u a := by
  intro hc
  have h3 := hc {3} ib_three
  rw [maxShare_eq, cuts_three, (by decide : (({3} : Finset (Fin d.D.N)) ∩ {2, 3, 4}).card = 1),
    (by decide : (computedReads d {3}).card = 1), Nat.cast_one, mul_one] at h3
  exact absurd h3 (not_le.2 h)

theorem cutCost_zero : CutCost d A 1 1 (fun _ => 0) 1 := by
  intro I hI
  rw [zero_add, one_mul]
  exact_mod_cast cuts_le_computedReads hI

theorem cutCost_three : CutCost d A 1 1 (fun x => x * 3) 0 := by
  intro I hI
  beta_reduce
  rw [zero_mul, add_zero, maxShare_eq, div_mul_cancel₀ _ three_ne_zero]
  exact_mod_cast inter_le_computedReads hI

theorem inputCost_three : InputCost d A 1 1 (fun x => x * 3) := by
  intro I hI T _
  beta_reduce
  rw [share_eq, div_mul_cancel₀ _ three_ne_zero]
  exact_mod_cast inter_le_unregistered hI

theorem peak_zero : PeakBound (fun _ => 0) 1 1 := fun x hx => by simpa using hx

theorem reach_three : ReachLimit (fun x => x * 3) 1 (1 / 3) := by
  have h := reach_even (by norm_num : (0 : ℝ≥0) < 3) 1
  rwa [Nat.cast_one] at h

theorem frozen_three : FrozenShare (fun x => x * 3) 1 (1 / 3) := fun x hx _ => by
  rw [Nat.cast_one]
  exact (div_lt_iff₀ (by norm_num)).1 hx

theorem deps_three (T : Fin 1) : 2 ≤ (A.dependencies T 3).card := by
  have f23 : FeedsWithin d.D {2, 3, 4} 2 3 := .step (by decide) (by decide) (.refl (by decide))
  have f33 : FeedsWithin d.D {2, 3, 4} 3 3 := .refl (by decide)
  calc 2 = ({2, 3} : Finset (Fin d.D.N)).card := by decide
    _ ≤ (A.dependencies T 3).card := card_le_card fun h hh => by
        rw [mem_dependencies, tokenAffected_eq]
        rcases mem_insert.1 hh with rfl | hh
        · exact ⟨by decide, f23⟩
        · obtain rfl := mem_singleton.1 hh
          exact ⟨by decide, f33⟩

theorem deps_four (T : Fin 1) : 2 ≤ (A.dependencies T 4).card := by
  have f34 : FeedsWithin d.D {2, 3, 4} 3 4 := .step (by decide) (by decide) (.refl (by decide))
  have f44 : FeedsWithin d.D {2, 3, 4} 4 4 := .refl (by decide)
  calc 2 = ({3, 4} : Finset (Fin d.D.N)).card := by decide
    _ ≤ (A.dependencies T 4).card := card_le_card fun h hh => by
        rw [mem_dependencies, tokenAffected_eq]
        rcases mem_insert.1 hh with rfl | hh
        · exact ⟨by decide, f34⟩
        · obtain rfl := mem_singleton.1 hh
          exact ⟨by decide, f44⟩

theorem third_lt_dependencyShare {T : Fin 1} {g : Fin d.D.N} (h : 2 ≤ (A.dependencies T g).card) :
    1 / 3 < A.dependencyShare T g := by
  rw [Annotation.dependencyShare, card_tokenAffected, ← NNReal.coe_lt_coe]
  push_cast
  have h2 : (2 : ℝ) ≤ (A.dependencies T g).card := by exact_mod_cast h
  linarith

/-- The token's dependency profile at `1/3` is `1/3`: gate `2` alone depends on a third of its work or less. -/
theorem profile_third : A.ProfileAtMost (1 / 3) (1 / 3) := by
  intro T
  refine (dependencyProfile_le A T (1 / 3) (S := {2}) fun g hg hdep => ?_).trans ?_
  · rw [tokenAffected_eq] at hg
    rcases mem_insert.1 hg with rfl | hg
    · exact mem_singleton_self 2
    rcases mem_insert.1 hg with rfl | hg
    · exact absurd hdep (not_le.2 (third_lt_dependencyShare (deps_three T)))
    · obtain rfl := mem_singleton.1 hg
      exact absurd hdep (not_le.2 (third_lt_dependencyShare (deps_four T)))
  · rw [card_singleton, card_tokenAffected]
    norm_num

theorem blocks_isCK : blocks.IsCK d 1 1 := ⟨⟨fun _ => rfl, by decide⟩, by decide, by decide⟩

theorem not_whole_isCK : ¬ whole.IsCK d 1 1 := fun h => absurd (h.card 0) (by decide)

theorem price_eq (B c : ℕ) : partitionSlowdown d B c = ((((B * c : ℕ) : ℝ≥0) / 4 : ℝ≥0) : ℝ≥0∞) := by
  rw [partitionSlowdown, computed_card, natCast_div_natCast four_ne_zero, Nat.cast_ofNat]

theorem honestCost_eq : honestCost methods (fun _ => 0) 0 = ((4 : ℕ) : ℝ≥0∞) := by
  have hreach : r.Reaches (fun _ => 0) 0 := le_rfl
  unfold honestCost
  refine le_antisymm (iInf_le_of_le r (iInf_le_of_le (Set.mem_singleton r) (iInf_le_of_le hreach ?_))) ?_
  · rw [show r.d.D.computed.card = 4 from computed_card]
  · refine le_iInf fun r' => le_iInf fun hr' => le_iInf fun _ => ?_
    obtain rfl := Set.mem_singleton_iff.1 hr'
    rw [show r.d.D.computed.card = 4 from computed_card]

/-- The counting bound at `S = {3, 4}`, `1/2`, is above `whole`'s price `1/4`. -/
theorem not_counting : ¬ (((({3, 4} : Finset (Fin d.D.N)).card : ℝ≥0∞) / (d.D.computed.card : ℝ≥0∞)) /
    ((maxInBlock d 1 1 {3, 4} : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ≤ partitionSlowdown d 1 1) := by
  rw [(by decide : ({3, 4} : Finset (Fin d.D.N)).card = 2), computed_card, maxInBlock_eq,
    natCast_div_natCast four_ne_zero, natCast_div_natCast one_ne_zero,
    coe_div_coe (y := ((1 : ℕ) : ℝ≥0) / ((1 : ℕ) : ℝ≥0)) (by simp), price_eq, ENNReal.coe_le_coe,
    ← NNReal.coe_le_coe]
  push_cast
  norm_num

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

theorem Counting_honest : Guarantees.Counting_honest := ⟨Chain.gateSet, Chain.blocks_isCK⟩

theorem Counting_nontrivial : Guarantees.Counting_nontrivial :=
  ⟨Chain.gateSet, Chain.not_whole_isCK, Chain.not_counting⟩

theorem Slowdown_honest : Guarantees.Slowdown_honest := Chain.blocks_isCK

theorem Slowdown_nontrivial : Guarantees.Slowdown_nontrivial :=
  ⟨Chain.not_whole_isCK, by rw [Chain.unreachable_eq]; exact Chain.not_counting⟩

theorem TokenForm_honest : Guarantees.TokenForm_honest :=
  ⟨Chain.identical, Chain.reachableAtMost, Chain.blocks_isCK⟩

theorem TokenForm_nontrivial : Guarantees.TokenForm_nontrivial := by
  refine ⟨Chain.identical, Chain.reachableAtMost, Chain.not_whole_isCK, ?_⟩
  rw [Chain.tokenShare_eq, Chain.tokensPerBlock_eq, one_sub_third, Chain.unreachableCapacity_eq,
    coe_div_coe (y := 1 / 3) (by positivity), Chain.price_eq, ENNReal.coe_le_coe, ← NNReal.coe_le_coe]
  push_cast
  norm_num

theorem TokenFormVarying_honest : Guarantees.TokenFormVarying_honest := ⟨Chain.reachableAtMost, Chain.blocks_isCK⟩

theorem TokenFormVarying_nontrivial : Guarantees.TokenFormVarying_nontrivial := by
  refine ⟨Chain.reachableAtMost, Chain.not_whole_isCK, ?_⟩
  rw [one_sub_third, Chain.affected_inter_card, Chain.computed_card, Chain.maxAffected_eq,
    Chain.unreachableCapacity_eq, coe_div_coe (y := ((4 : ℕ) : ℝ≥0) * ((3 : ℕ) : ℝ≥0) * (1 / 3)) (by positivity),
    Chain.price_eq, ENNReal.coe_le_coe, ← NNReal.coe_le_coe]
  push_cast
  norm_num

theorem SlowdownClaim_honest : Guarantees.SlowdownClaim_honest :=
  ⟨Chain.valid, Chain.identical, Chain.cutCost_zero, Chain.peak_zero, Chain.inputCost_three, Chain.reach_three,
    Chain.profile_third, Chain.blocks_isCK⟩

theorem SlowdownClaim_nontrivial : Guarantees.SlowdownClaim_nontrivial := by
  refine ⟨Chain.valid, Chain.identical, Chain.not_cutCost (by norm_num), Chain.peak_zero, Chain.inputCost_three,
    Chain.reach_three, Chain.profile_third, Chain.blocks_isCK, ?_⟩
  rw [Chain.tokenShare_eq, Chain.tokensPerBlock_eq, one_sub_third, ENNReal.coe_one, div_one, Chain.price_eq,
    ENNReal.coe_le_coe, ← NNReal.coe_le_coe]
  push_cast
  norm_num

theorem SlowdownEven_honest : Guarantees.SlowdownEven_honest :=
  ⟨Chain.valid, Chain.identical, Chain.cutCost_three, Chain.inputCost_three, by norm_num,
    by rw [Nat.cast_one]; exact Chain.profile_third, Chain.blocks_isCK⟩

theorem SlowdownEven_nontrivial : Guarantees.SlowdownEven_nontrivial := by
  refine ⟨Chain.valid, Chain.identical, Chain.not_cutCost ?_, Chain.inputCost_three, by norm_num,
    by rw [Nat.cast_one]; exact Chain.profile_third, Chain.blocks_isCK, ?_⟩
  · rw [← NNReal.coe_lt_coe]
    push_cast
    norm_num
  rw [Chain.tokenShare_eq, Chain.tokensPerBlock_eq, one_sub_third, Nat.cast_one, one_pow, ENNReal.coe_one, div_one,
    Chain.price_eq, ENNReal.coe_le_coe, ← NNReal.coe_le_coe]
  push_cast
  norm_num

theorem SlowdownFrozen_honest : Guarantees.SlowdownFrozen_honest :=
  ⟨Chain.valid, Chain.identical, Chain.cutCost_three, Chain.frozen_three, Chain.inputCost_three, Chain.reach_three,
    Chain.profile_third, Chain.blocks_isCK⟩

theorem SlowdownFrozen_nontrivial : Guarantees.SlowdownFrozen_nontrivial := by
  refine ⟨Chain.valid, Chain.identical, Chain.not_cutCost ?_, Chain.frozen_three, Chain.inputCost_three,
    Chain.reach_three, Chain.profile_third, Chain.blocks_isCK, ?_⟩
  · rw [← NNReal.coe_lt_coe]
    push_cast
    norm_num
  rw [Chain.tokenShare_eq, Chain.tokensPerBlock_eq, one_sub_third, Nat.cast_one, mul_one,
    coe_div_coe (y := 1 / 3) (by positivity), Chain.price_eq, ENNReal.coe_le_coe, ← NNReal.coe_le_coe]
  push_cast
  norm_num

theorem Security_honest : Guarantees.Security_honest :=
  ⟨floor_one, Set.mem_singleton _, Chain.valid, le_top, le_rfl, Chain.identical, Chain.cutCost_zero, Chain.peak_zero,
    Chain.inputCost_three, Chain.reach_three, Chain.profile_third, Chain.blocks_isCK, le_rfl⟩

theorem Security_nontrivial : Guarantees.Security_nontrivial := by
  refine ⟨floor_one, Set.mem_singleton _, Chain.valid, le_top, le_rfl, Chain.identical,
    Chain.not_cutCost (by norm_num), Chain.peak_zero, Chain.inputCost_three, Chain.reach_three, Chain.profile_third,
    Chain.blocks_isCK, le_rfl, ?_⟩
  rw [show Chain.r.A.tokenShare = 3 / 4 from Chain.tokenShare_eq,
    show Chain.r.A.tokensPerBlock 1 = 1 / 3 from Chain.tokensPerBlock_eq, one_sub_third, ENNReal.coe_one, div_one,
    ENNReal.ofReal_one, mul_one, Chain.honestCost_eq, ← ENNReal.coe_natCast 4, ← ENNReal.coe_mul, ENNReal.coe_le_coe,
    ← NNReal.coe_le_coe]
  push_cast
  norm_num

theorem FitServing_honest : Guarantees.FitServing_honest := ⟨Honest.boolean, Honest.sessionBlocks, le_rfl⟩

theorem FitServing_nontrivial : Guarantees.FitServing_nontrivial :=
  ⟨Honest.boolean, Honest.sessionBlocks, by unfold ServingFits; decide, Honest.not_bound⟩

theorem FitTraining_honest : Guarantees.FitTraining_honest :=
  ⟨Honest.boolean, Honest.trainingBlocks, le_rfl, le_rfl, le_rfl⟩

theorem FitTraining_nontrivial : Guarantees.FitTraining_nontrivial :=
  ⟨Honest.boolean, Honest.trainingBlocks, fun h => absurd h.1 (by decide), Honest.not_bound⟩

/-! Each lemma is its guarantee's premises as the guarantee states them: the guarantee applies at the honest instance,
and at the nontrivial one the dropped premise would give the bound its last conjunct negates. -/

example : ((({3, 4} : Finset (Fin Chain.d.D.N)).card : ℝ≥0∞) / (Chain.d.D.computed.card : ℝ≥0∞)) /
    ((maxInBlock Chain.d 1 1 {3, 4} : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ≤ partitionSlowdown Chain.d 4 1 :=
  Counting Chain.d 1 1 {3, 4} 4 Chain.blocks Counting_honest.1 Counting_honest.2

example (h : Chain.whole.IsCK Chain.d 1 1) : False :=
  Counting_nontrivial.2.2 (Counting Chain.d 1 1 {3, 4} 1 Chain.whole Counting_nontrivial.1 h)

example : (((Chain.A.affected \ Chain.A.reachable 1 1).card : ℝ≥0∞) / (Chain.d.D.computed.card : ℝ≥0∞)) /
    ((maxInBlock Chain.d 1 1 (Chain.A.affected \ Chain.A.reachable 1 1) : ℝ≥0∞) / ((1 : ℕ) : ℝ≥0∞)) ≤
      partitionSlowdown Chain.d 4 1 :=
  Slowdown Chain.d (Fin 1) Chain.A 1 1 4 Chain.blocks Slowdown_honest

example (h : Chain.whole.IsCK Chain.d 1 1) : False :=
  Slowdown_nontrivial.2 (Slowdown Chain.d (Fin 1) Chain.A 1 1 1 Chain.whole h)

example : ((Chain.A.tokenShare * (1 - 1 / 3) * Chain.A.tokensPerBlock 1 : ℝ≥0) : ℝ≥0∞) /
    (Chain.A.unreachableCapacity 1 1 : ℝ≥0∞) ≤ partitionSlowdown Chain.d 4 1 := by
  obtain ⟨h1, h2, h3⟩ := TokenForm_honest
  exact TokenForm Chain.d (Fin 1) Chain.A 1 1 3 (1 / 3) 4 Chain.blocks h1 h2 h3

example (h : Chain.whole.IsCK Chain.d 1 1) : False := by
  obtain ⟨h1, h2, -, hlt⟩ := TokenForm_nontrivial
  exact hlt (TokenForm Chain.d (Fin 1) Chain.A 1 1 3 (1 / 3) 1 Chain.whole h1 h2 h)

example : (((1 - 1 / 3) * (Chain.A.affected ∩ Chain.A.tokenGates).card * ((1 : ℕ) : ℝ≥0) : ℝ≥0) : ℝ≥0∞) /
    ((Chain.d.D.computed.card * Chain.A.maxAffected * Chain.A.unreachableCapacity 1 1 : ℝ≥0) : ℝ≥0∞) ≤
      partitionSlowdown Chain.d 4 1 := by
  obtain ⟨h1, h2⟩ := TokenFormVarying_honest
  exact TokenFormVarying Chain.d (Fin 1) Chain.A 1 1 (1 / 3) 4 Chain.blocks h1 h2

example (h : Chain.whole.IsCK Chain.d 1 1) : False := by
  obtain ⟨h1, -, hlt⟩ := TokenFormVarying_nontrivial
  exact hlt (TokenFormVarying Chain.d (Fin 1) Chain.A 1 1 (1 / 3) 1 Chain.whole h1 h)

example : ((Chain.A.tokenShare * (1 - 1 / 3) * Chain.A.tokensPerBlock 1 * 1 : ℝ≥0) : ℝ≥0∞) / ((1 : ℝ≥0) : ℝ≥0∞) ≤
    partitionSlowdown Chain.d 4 1 := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := SlowdownClaim_honest
  exact SlowdownClaim Chain.d (Fin 1) Chain.A 1 1 3 (fun _ => 0) (fun x => x * 3) 1 1 (1 / 3) (1 / 3) 4 Chain.blocks
    h1 h2 h3 h4 h5 h6 h7 h8

example (h3 : CutCost Chain.d Chain.A 1 1 (fun _ => 0) 12) : False := by
  obtain ⟨h1, h2, -, h4, h5, h6, h7, h8, hlt⟩ := SlowdownClaim_nontrivial
  exact hlt (SlowdownClaim Chain.d (Fin 1) Chain.A 1 1 3 (fun _ => 0) (fun x => x * 3) 12 1 (1 / 3) (1 / 3) 4
    Chain.blocks h1 h2 h3 h4 h5 h6 h7 h8)

example : ((4 * Chain.A.tokenShare * (1 - 1 / 3) * Chain.A.tokensPerBlock 1 * 0 * 3 : ℝ≥0) : ℝ≥0∞) /
    ((((1 : ℕ) : ℝ≥0) ^ 2 : ℝ≥0) : ℝ≥0∞) ≤ partitionSlowdown Chain.d 4 1 := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := SlowdownEven_honest
  exact SlowdownEven Chain.d (Fin 1) Chain.A 1 1 3 3 0 (1 / 3) 4 Chain.blocks h1 h2 h3 h4 h5 h6 h7

example (h3 : CutCost Chain.d Chain.A 1 1 (fun x => x * 3) 1) : False := by
  obtain ⟨h1, h2, -, h4, h5, h6, h7, hlt⟩ := SlowdownEven_nontrivial
  exact hlt (SlowdownEven Chain.d (Fin 1) Chain.A 1 1 3 3 1 (1 / 3) 4 Chain.blocks h1 h2 h3 h4 h5 h6 h7)

example : ((Chain.A.tokenShare * (1 - 1 / 3) * Chain.A.tokensPerBlock 1 * 0 : ℝ≥0) : ℝ≥0∞) /
    ((1 / 3 * ((1 : ℕ) : ℝ≥0) : ℝ≥0) : ℝ≥0∞) ≤ partitionSlowdown Chain.d 4 1 := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := SlowdownFrozen_honest
  exact SlowdownFrozen Chain.d (Fin 1) Chain.A 1 1 3 (fun x => x * 3) (fun x => x * 3) 0 (1 / 3) (1 / 3) (1 / 3) 4
    Chain.blocks h1 h2 h3 h4 h5 h6 h7 h8

example (h3 : CutCost Chain.d Chain.A 1 1 (fun x => x * 3) 3) : False := by
  obtain ⟨h1, h2, -, h4, h5, h6, h7, h8, hlt⟩ := SlowdownFrozen_nontrivial
  exact hlt (SlowdownFrozen Chain.d (Fin 1) Chain.A 1 1 3 (fun x => x * 3) (fun x => x * 3) 3 (1 / 3) (1 / 3) (1 / 3)
    4 Chain.blocks h1 h2 h3 h4 h5 h6 h7 h8)

example : ((Chain.r.A.tokenShare * (1 - 1 / 3) * Chain.r.A.tokensPerBlock 1 * 1 : ℝ≥0) : ℝ≥0∞) /
    ((1 : ℝ≥0) : ℝ≥0∞) * ENNReal.ofReal 1 * honestCost Chain.methods (fun _ => 0) 0 ≤ ((1 * 4 * 1 : ℕ) : ℝ≥0∞) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := Security_honest
  exact Security Chain.methods silent (fun _ => 0) 0 ⊤ 1 Chain.r 1 1 4 1 3 (fun _ => 0) (fun x => x * 3) 1 1 (1 / 3)
    (1 / 3) 4 Chain.blocks h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13

example (h7 : CutCost Chain.r.d Chain.r.A 1 1 (fun _ => 0) 12) : False := by
  obtain ⟨h1, h2, h3, h4, h5, h6, -, h8, h9, h10, h11, h12, h13, hlt⟩ := Security_nontrivial
  exact hlt (Security Chain.methods silent (fun _ => 0) 0 ⊤ 1 Chain.r 1 1 4 1 3 (fun _ => 0) (fun x => x * 3) 12 1
    (1 / 3) (1 / 3) 4 Chain.blocks h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13)

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
