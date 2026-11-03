import Proofs.Nci.Formalism.Slowdown
import Definitions.Nci.Guarantees.Formalism.Witnesses

/-!
# The training formalism v1: honest and nontrivial lemmas

The proofs of `Security/Definitions/Nci/Guarantees/Formalism/Witnesses.lean`, at the instances of
`Security/Definitions/Nci/Formalism/Instances.lean`. Their gates, reads and token blocks are small finite sets, which
`decide` computes, and `Dep` follows from a gate feeding only later gates (`le_of_feeds`). The `example`s at the end
apply each guarantee to its lemmas, so the lemmas are the premises as the guarantee states them.
-/

namespace Nci.Formalism

open Finset
open Verity (Circuit Declaration)
open Nci.Formalism.Assumptions
open scoped NNReal ENNReal

/-- A gate feeds only itself and later gates. -/
theorem le_of_feeds {C : Circuit} {h g : Fin C.N} (hf : C.Feeds h g) : h ≤ g := by
  induction hf with
  | refl => exact le_rfl
  | step hr _ ih => exact (C.reads_lt _ _ hr).le.trans ih

/-- Tokens whose dependent work is one gate each are linked. -/
theorem linked_of_tokenDep {d : Declaration} {L : Type} (A : Annotation d L) (h : ∀ T, ∃ g, A.tokenDep T = {g}) :
    A.TokensLinked 0 := by
  intro T
  obtain ⟨g, hg⟩ := h T
  refine ⟨∅, empty_subset _, by simp, fun g₁ h₁ g₂ h₂ => ?_⟩
  rw [hg, sdiff_empty, mem_singleton] at h₁ h₂
  have hT : g ∈ A.tokens.holds T := by
    have hd : g ∈ A.tokenDep T := by
      rw [hg]
      exact mem_singleton_self g
    exact mem_of_mem_inter_left hd
  subst h₁ h₂
  exact ⟨_, .refl hT, .refl hT⟩

theorem maxShare_eq_zero {d : Declaration} {L : Type} [Fintype L] (A : Annotation d L) {I : Finset (Fin d.D.N)}
    (h : ∀ T, I ∩ A.tokenDep T = ∅) : A.maxShare I = 0 :=
  le_antisymm (Finset.sup_le fun T _ => by simp [Annotation.share, h T]) zero_le

/-- A2 holds at `p = 1` over one method. -/
theorem floor_one {r : Run} {dec : Verity.Decoder (List Bool)} {c : ℕ} {ℓ : List Bool → ℝ} {ℓstar : ℝ} {D : ℕ∞} :
    EfficiencyFloor {r} dec c ℓ ℓstar D 1 := by
  intro r' hr' _ _ hreach
  rw [ENNReal.ofReal_one, one_mul]
  unfold honestCost
  exact iInf_le_of_le r' (iInf_le_of_le hr' (iInf_le_of_le hreach le_rfl))

theorem peak_two : PeakBound (fun x => (x : ℝ)) 2 1 := by
  intro x _
  show (x : ℝ) * ((2 : ℕ) - (x : ℝ)) ≤ 1
  push_cast
  nlinarith [sq_nonneg ((x : ℝ) - 1)]

theorem peak_zero : PeakBound (fun x => (x : ℝ)) 0 1 := by
  intro x _
  show (x : ℝ) * ((0 : ℕ) - (x : ℝ)) ≤ 1
  push_cast
  nlinarith [sq_nonneg (x : ℝ)]

theorem bottom_two : BottomShare (fun x => (x : ℝ)) 2 1 := fun _ hx _ => by exact_mod_cast hx

theorem bottom_zero : BottomShare (fun x => (x : ℝ)) 0 0 := fun _ _ hx => by simpa using hx

theorem frozen_id : FrozenTail (fun x => (x : ℝ)) 2 1 :=
  ⟨fun x => NNReal.coe_nonneg x, fun _ hx hx1 => absurd hx1 (not_le.2 (by exact_mod_cast hx))⟩

theorem frozen_const : FrozenTail (fun _ => 2) 1 (1 / 2) := ⟨fun _ => by norm_num, fun _ _ _ => by norm_num⟩

namespace SecurityProofs

namespace Honest

open Nci.Formalism.Honest

theorem valid : A.Valid := ⟨by decide, by decide, fun T T' h => absurd (Subsingleton.elim T T') h⟩

theorem dep_eq : A.dep = {3} := by
  ext g
  simp only [Annotation.dep, mem_filter, mem_singleton]
  constructor
  · rintro ⟨-, v, rfl, h, hvh, hf⟩
    obtain rfl : h = 3 := (by decide : ∀ h, (2 : Fin d.D.N) ∈ d.D.reads h → h = 3) h hvh
    exact (by decide : ∀ g : Fin d.D.N, 3 ≤ g → g = 3) g (le_of_feeds hf)
  · rintro rfl
    refine ⟨by decide, 2, by decide, ?_⟩
    unfold DependsOn
    exact ⟨3, by decide, .refl 3⟩

theorem tokenDep_eq (T : Fin 1) : A.tokenDep T = {3} := by
  rw [Annotation.tokenDep, dep_eq]
  exact inter_self _

theorem linked : A.TokensLinked 0 := linked_of_tokenDep A fun T => ⟨3, tokenDep_eq T⟩

theorem isCK : IsCKPartition d P 1 2 := ⟨by decide, by decide⟩

/-- A1 at `c = 1`, `a = 0` and any update profile at most the share: of the blocks of one gate, only the token's does
any of its dependent work, and it reads `θ'`, one computed value. -/
theorem cutCost {u : ℝ≥0 → ℝ} (hu : ∀ x, u x ≤ x) : CutCost d A 1 u 0 := by
  intro I _ hc
  rw [zero_mul, add_zero]
  refine (hu _).trans ?_
  by_cases h3 : (3 : Fin d.D.N) ∈ I
  · obtain rfl : I = {3} := eq_singleton_iff_unique_mem.2 ⟨h3, fun x hx => card_le_one.1 hc x hx 3 h3⟩
    have h1 : (computedReads d {3}).card = 1 := by decide
    rw [h1, Nat.cast_one]
    exact_mod_cast maxShare_le_one A {3}
  · rw [maxShare_eq_zero A fun T => by
      rw [tokenDep_eq]
      exact disjoint_iff_inter_eq_empty.1 (disjoint_singleton_right.2 h3), NNReal.coe_zero]
    exact Nat.cast_nonneg _

end Honest

namespace Counter

open Nci.Formalism.Counter

theorem valid : A.Valid := ⟨by decide, by decide, fun T T' h => absurd (Subsingleton.elim T T') h⟩

theorem dep_eq : A.dep = {2} := by
  ext g
  simp only [Annotation.dep, mem_filter, mem_singleton]
  constructor
  · rintro ⟨-, v, rfl, h, hvh, hf⟩
    obtain rfl : h = 2 := (by decide : ∀ h, (1 : Fin d.D.N) ∈ d.D.reads h → h = 2) h hvh
    exact (by decide : ∀ g : Fin d.D.N, 2 ≤ g → g = 2) g (le_of_feeds hf)
  · rintro rfl
    refine ⟨by decide, 1, by decide, ?_⟩
    unfold DependsOn
    exact ⟨2, by decide, .refl 2⟩

theorem tokenDep_eq (T : Fin 1) : A.tokenDep T = {2} := by
  rw [Annotation.tokenDep, dep_eq]
  exact inter_self _

theorem linked : A.TokensLinked 0 := linked_of_tokenDep A fun T => ⟨2, tokenDep_eq T⟩

theorem isCK (k : ℕ) : IsCKPartition d P 2 k :=
  ⟨by decide, fun u => ((by decide : ∀ u, (d.unregistered (P.gates u)).card = 0) u).trans_le (Nat.zero_le k)⟩

theorem computed_card : d.D.computed.card = 2 := by decide

/-- The one block of both gates: the slowdown at `c = 2` is at most `1`, for every `k`. -/
theorem slowdown_le (k : ℕ) : slowdown d 2 k ≤ 1 := by
  unfold slowdown
  refine iInf_le_of_le 1 (iInf_le_of_le P (iInf_le_of_le (isCK k) ?_))
  rw [computed_card, one_mul, ENNReal.div_self (by norm_num) (ENNReal.natCast_ne_top 2)]

theorem not_bound {x : ℝ} (hx : 1 < x) (k : ℕ) : ¬ ENNReal.ofReal x ≤ slowdown d 2 k :=
  not_le.2 ((slowdown_le k).trans_lt (ENNReal.one_lt_ofReal.2 hx))

theorem share_eq : A.share 0 {1, 2} = 1 := by
  rw [Annotation.share, tokenDep_eq]
  have h : (({1, 2} : Finset (Fin d.D.N)) ∩ {2}).card = 1 := by decide
  rw [h, card_singleton]
  simp

theorem maxShare_eq : A.maxShare {1, 2} = 1 :=
  le_antisymm (maxShare_le_one A _) (share_eq.symm.trans_le (share_le_maxShare A 0 _))

/-- A1 fails at the block of both gates: it does all of the token's dependent work and reads no computed value. -/
theorem not_cutCost {u : ℝ≥0 → ℝ} {a : ℝ} (ha : 0 ≤ a) (hu : 0 < u 1) : ¬ CutCost d A 2 u a := by
  intro h
  have h1 := h {1, 2} (by decide) (by decide)
  have h0 : (computedReads d {1, 2}).card = 0 := by decide
  rw [maxShare_eq, h0, Nat.cast_zero] at h1
  have hm : 0 ≤ a * (A.midEntries {1, 2} : ℝ) := mul_nonneg ha (Nat.cast_nonneg _)
  linarith

theorem depShare_eq : A.depShare = 1 / 2 := by
  have h : (A.dep ∩ A.tokenGates).card = 1 := by
    rw [dep_eq]
    decide
  rw [Annotation.depShare, h, computed_card]
  norm_num

theorem tokensPerBlock_eq : A.tokensPerBlock 2 = 2 := by
  have h : A.work = 1 := by decide
  rw [Annotation.tokensPerBlock, h]
  norm_num

theorem r_card : r.d.D.computed.card = 2 := computed_card

theorem honestCost_eq : honestCost methods (fun _ => 0) 0 = ((2 : ℕ) : ℝ≥0∞) := by
  have hreach : r.Reaches (fun _ => 0) 0 := le_rfl
  unfold honestCost
  refine le_antisymm (iInf_le_of_le r (iInf_le_of_le (Set.mem_singleton r) (iInf_le_of_le hreach ?_))) ?_
  · rw [r_card]
  · refine le_iInf fun r' => le_iInf fun hr' => le_iInf fun _ => ?_
    obtain rfl := Set.mem_singleton_iff.1 hr'
    rw [r_card]

end Counter

theorem Slowdown_honest : Guarantees.Slowdown_honest :=
  ⟨Honest.valid, Honest.cutCost fun _ => le_rfl, Honest.linked, peak_two, bottom_two, le_rfl⟩

theorem Slowdown_nontrivial : Guarantees.Slowdown_nontrivial :=
  ⟨Counter.valid, Counter.not_cutCost zero_le_two (by norm_num), Counter.linked, peak_zero, bottom_zero, zero_le_two,
    Counter.not_bound (by rw [Counter.depShare_eq, Counter.tokensPerBlock_eq]; norm_num) 0⟩

theorem SlowdownEven_honest : Guarantees.SlowdownEven_honest :=
  ⟨Honest.valid, Honest.cutCost fun x => (mul_one (x : ℝ)).le, Honest.linked, Nat.one_pos, Nat.cast_one.le, le_rfl⟩

theorem SlowdownEven_nontrivial : Guarantees.SlowdownEven_nontrivial :=
  ⟨Counter.valid, Counter.not_cutCost zero_le_one (by norm_num), Counter.linked, Nat.one_pos, by norm_num,
    zero_le_one, Counter.not_bound (by rw [Counter.depShare_eq, Counter.tokensPerBlock_eq]; norm_num) 1⟩

theorem SlowdownFrozenTail_honest : Guarantees.SlowdownFrozenTail_honest :=
  ⟨Honest.valid, Honest.cutCost fun _ => le_rfl, Honest.linked, frozen_id, zero_le_one, le_rfl⟩

theorem SlowdownFrozenTail_nontrivial : Guarantees.SlowdownFrozenTail_nontrivial :=
  ⟨Counter.valid, Counter.not_cutCost zero_le_two (by norm_num), Counter.linked, frozen_const, by norm_num,
    zero_le_two, Counter.not_bound (by rw [Counter.depShare_eq, Counter.tokensPerBlock_eq]; norm_num) 1⟩

theorem Security_honest : Guarantees.Security_honest :=
  ⟨floor_one, Set.mem_singleton _, Honest.valid, le_top, le_rfl, Honest.cutCost fun _ => le_rfl, Honest.linked,
    peak_two, bottom_two, le_rfl, zero_le_one, Honest.isCK, le_rfl⟩

theorem Security_nontrivial : Guarantees.Security_nontrivial := by
  refine ⟨floor_one, Set.mem_singleton _, Counter.valid, le_top, le_rfl,
    Counter.not_cutCost zero_le_two (by norm_num), Counter.linked, peak_zero, bottom_zero, zero_le_two, zero_le_one,
    Counter.isCK 0, le_rfl, ?_⟩
  have hx : Counter.r.A.depShare * (1 - 0 - 0) * Counter.r.A.tokensPerBlock 2 * 2 / 1 * 1 = ((2 : ℕ) : ℝ) := by
    rw [show Counter.r.A.depShare = 1 / 2 from Counter.depShare_eq,
      show Counter.r.A.tokensPerBlock 2 = 2 from Counter.tokensPerBlock_eq]
    norm_num
  rw [hx, ENNReal.ofReal_natCast, Counter.honestCost_eq, ← Nat.cast_mul]
  exact not_le.2 (Nat.cast_lt.2 (by decide))

/-! Each lemma is its guarantee's premises as the guarantee states them: the guarantee applies at the honest instance,
and at the counter-example A1 would give the bound the lemma's last conjunct negates. -/

example : ENNReal.ofReal (Honest.A.depShare * (1 - 1 - 0) * Honest.A.tokensPerBlock 1 * 0 / 1) ≤
    slowdown Honest.d 1 2 := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := Slowdown_honest
  exact Slowdown Honest.d (Fin 1) Honest.A 1 2 (fun x => (x : ℝ)) 0 1 0 1 h1 h2 h3 h4 h5 h6

example (h2 : CutCost Counter.d Counter.A 2 (fun x => (x : ℝ)) 2) : False := by
  obtain ⟨h1, -, h3, h4, h5, h6, hlt⟩ := Slowdown_nontrivial
  exact hlt (Slowdown Counter.d (Fin 1) Counter.A 2 0 (fun x => (x : ℝ)) 2 0 0 1 h1 h2 h3 h4 h5 h6)

example : ENNReal.ofReal (4 * Honest.A.depShare * Honest.A.tokensPerBlock 1 * 0 * (1 - ((1 : ℕ) : ℝ) - 0 * 1) /
    ((1 : ℕ) : ℝ) ^ 2) ≤ slowdown Honest.d 1 1 := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := SlowdownEven_honest
  exact SlowdownEven Honest.d (Fin 1) Honest.A 1 1 1 0 0 h1 h2 h3 h4 h5 h6

example (h2 : CutCost Counter.d Counter.A 2 (fun x => (x : ℝ) * 2) 1) : False := by
  obtain ⟨h1, -, h3, h4, h5, h6, hlt⟩ := SlowdownEven_nontrivial
  exact hlt (SlowdownEven Counter.d (Fin 1) Counter.A 2 1 2 1 0 h1 h2 h3 h4 h5 h6)

example : ENNReal.ofReal (Honest.A.depShare * (1 - 1 - 0) * Honest.A.tokensPerBlock 1 * 0 / (1 * ((2 : ℕ) : ℝ))) ≤
    slowdown Honest.d 1 2 := by
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := SlowdownFrozenTail_honest
  exact SlowdownFrozenTail Honest.d (Fin 1) Honest.A 1 2 (fun x => (x : ℝ)) 0 1 0 h1 h2 h3 h4 h5 h6

example (h2 : CutCost Counter.d Counter.A 2 (fun _ => 2) 2) : False := by
  obtain ⟨h1, -, h3, h4, h5, h6, hlt⟩ := SlowdownFrozenTail_nontrivial
  exact hlt (SlowdownFrozenTail Counter.d (Fin 1) Counter.A 2 1 (fun _ => 2) 2 (1 / 2) 0 h1 h2 h3 h4 h5 h6)

example : ENNReal.ofReal (Honest.r.A.depShare * (1 - 1 - 0) * Honest.r.A.tokensPerBlock 1 * 0 / 1 * 1) *
    honestCost Honest.methods (fun _ => 0) 0 ≤ ((1 * 2 * 1 : ℕ) : ℝ≥0∞) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := Security_honest
  exact Security Honest.methods silent (fun _ => 0) 0 ⊤ 1 Honest.r 1 2 2 1 2 Honest.P (fun x => (x : ℝ)) 0 1 0 1
    h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13

example (h6 : CutCost Counter.r.d Counter.r.A 2 (fun x => (x : ℝ)) 2) : False := by
  obtain ⟨h1, h2, h3, h4, h5, -, h7, h8, h9, h10, h11, h12, h13, hlt⟩ := Security_nontrivial
  exact hlt (Security Counter.methods silent (fun _ => 0) 0 ⊤ 1 Counter.r 2 0 1 1 1 Counter.P (fun x => (x : ℝ)) 2 0 0 1
    h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13)

end SecurityProofs

end Nci.Formalism
