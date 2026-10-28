import Proofs.Pouw.Milestone
import Proofs.Pouw.QTree

/-!
# Satisfiability witnesses and non-vacuity

* `ttWitness`: in `outputPriced`, TT(0.5%) holds with error 0 and one program runs `H_ref` at `≤ W_ref`, for `D1.m1`.
* `honestComplete`: the honest reference is correct on every unit and does not win at `T = W`.
* `gameCanBeLost`: the honest program at budget 0 (free model) or `W_mm` (witness model) wins with probability 1.
-/

namespace Pouw.Proofs

open Finset QTree

variable {Q R S : Type}

/-! ## The honest reference as an oracle program -/

section Honest

variable (P : D1.Params Q R S)

/-- Unit `u`'s honest commitment on activations `A`, from the answers `a` to `qAct` and `b` to `qWt` (the formulas
of `D1.Params.refChecked` and `refOut` with the oracle's answers as inputs). -/
def commitAns (U : Workload) (u : ℕ) (A : Mat) (a b : List R) : Commit :=
  let sh := U.layout.shape u
  let E := P.expAct sh a b
  let F := P.expWt sh b
  let act' : Mat := fun i l => wrap32 (A i l + mmW P.r E.1 E.2 i l)
  let wt' : Mat := fun l j => wrap32 (U.weight (U.layout.wid u) l j + mmW P.r F.1 F.2 l j)
  ⟨A, fun t => mmW (D1.blockEnd P.d sh.k t) act' wt',
    fun i j => wrap32 (wrap32 (mmW sh.k act' wt' i j - mmW P.r (mmW sh.k A F.1) F.2 i j)
      - mmW P.r E.1 (mmW sh.k E.2 wt') i j)⟩

theorem commitAns_eq (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) :
    commitAns P U u A ((P.qAct s u A (U.layout.wid u)).map H) ((P.qWt s u (U.layout.wid u)).map H)
      = ⟨A, P.toProtocol.refChecked H s U u A, P.toProtocol.refOut H s U u A⟩ := rfl

/-- Unit `u`'s honest program: query its noise, then commit. -/
def unitProg (U : Workload) (act : ℕ → Mat) (s : S) (u : ℕ) : QTree Q R Commit :=
  (queryAll (P.qAct s u (act u) (U.layout.wid u))).bind fun a =>
    (queryAll (P.qWt s u (U.layout.wid u))).bind fun b => done (commitAns P U u (act u) a b)

/-- The honest reference: every unit's honest program, in order. -/
def honestProg (U : Workload) (act : ℕ → Mat) (s : S) : QTree Q R CommitLog :=
  upTo (unitProg P U act s) U.layout.N

theorem eval_honestProg (H : Q → R) (U : Workload) (act : ℕ → Mat) (s : S) :
    (honestProg P U act s).eval H = honestTranscript P.toProtocol H s U act := by
  unfold honestProg honestTranscript
  rw [eval_upTo]
  funext u
  simp only [unitProg, eval_bind, eval_queryAll, eval, commitAns_eq]

theorem calls_honestProg (H : Q → R) (U : Workload) (act : ℕ → Mat) (s : S) {c : ℕ} (hc : P.QueryBound c) :
    (honestProg P U act s).calls H ≤ U.layout.N * c := by
  refine calls_upTo_le H _ _ c fun u _ => ?_
  simp only [unitProg, calls_bind, calls_queryAll, eval_queryAll, calls]
  have := hc s u (act u) (U.layout.wid u)
  omega

end Honest

/-! ## The witness models -/

theorem outputPriced_queries : (outputPriced Q R S).Queries :=
  fun prog U act σ s => ⟨prog U act σ s, fun _ => ⟨rfl, rfl⟩⟩

theorem freeModel_queries : (freeModel Q R S).Queries :=
  fun prog U act σ s => ⟨prog U act σ s, fun _ => ⟨rfl, rfl⟩⟩

theorem outputPriced_cost_le (prog : (outputPriced Q R S).Prog) (H : Q → R) (U : Workload) (act : ℕ → Mat)
    (σ : State) (s : S) :
    (outputPriced Q R S).cost prog H U act σ s ≤ ∑ u ∈ range U.layout.N, ((U.layout.shape u).Wmm : ℝ) := by
  simp only [outputPriced]
  exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) fun _ _ _ => by positivity

theorem checked_le_outputPriced_cost (P : Protocol Q R S) (prog : (outputPriced Q R S).Prog) (H : Q → R)
    (U : Workload) (act : ℕ → Mat) (σ : State) (s : S) :
    ∑ u ∈ checkedSet P U H s ((outputPriced Q R S).run prog H U act σ s), ((U.layout.shape u).Wmm : ℝ)
      ≤ (outputPriced Q R S).cost prog H U act σ s := by
  simp only [outputPriced]
  refine Finset.sum_le_sum_of_subset_of_nonneg (fun u hu => ?_) fun _ _ _ => by positivity
  simp only [checkedSet, Finset.mem_filter] at hu ⊢
  obtain ⟨hu, c, hc, -⟩ := hu
  exact ⟨hu, by rw [hc]; rfl⟩

/-- In the witness model TT(0.5%) holds with error 0 for every protocol and domain: the checked units' work is at
most the cost, which is at most the budget. -/
theorem tt_outputPriced [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S)
    (D : UnitLayout → Prop) : Assumptions.TT (outputPriced Q R S) P D Params.γ0 (fun _ _ => 0) := by
  refine ⟨outputPriced_queries, fun U _ T hT q _ A hA => le_of_eq (pr_eq_zero fun x hx => ?_)⟩
  have h1 := checked_le_outputPriced_cost P A.online x.1 U A.act (A.pre.eval x.1) x.2
  have h2 := (hA x.1 x.2).1
  have h3 : T ≤ T / (1 - Params.γ0) := by
    rw [le_div_iff₀ (by unfold Params.γ0; norm_num)]
    unfold Params.γ0; nlinarith
  exact absurd hx (not_lt.2 (h1.trans (h2.trans h3)))

/-- **Satisfiability witness for TT and `AdmitsRef`** in `outputPriced`, for `D1.m1` with `W_ref ≥ W_mm`. -/
theorem ttWitness : Pinned.TTWitness := by
  intro Wref hW
  refine ⟨tt_outputPriced _ _, fun U act _ s => honestProg (D1.m1 Wref) U act s, fun U hU act _ H s =>
    ⟨eval_honestProg _ H U act s, ?_, calls_honestProg _ H U act s (M1.queryBound Wref)⟩⟩
  refine (outputPriced_cost_le _ H U act [] s).trans (Finset.sum_le_sum fun u hu => ?_)
  exact_mod_cast hW U hU u (Finset.mem_range.1 hu)

/-! ## Honest completeness and a game that can be lost -/

theorem honest_correctSet (P : Protocol Q R S) (D : UnitLayout → Prop) (U : Workload) (H : Q → R) (s : S)
    (act : ℕ → Mat) (hP : P.Complete D) (hU : U.InDomain D) (hact : ∀ u < U.layout.N, ActOK (U.layout.shape u) (act u)) :
    correctSet P U H s (honestTranscript P H s U act) = range U.layout.N := by
  classical
  ext u
  unfold correctSet
  rw [Finset.mem_filter, Finset.mem_range, and_iff_left_iff_imp]
  intro hu
  unfold honestTranscript
  have hc := hP U hU u hu H s (act u) (hact u hu)
  exact ⟨⟨act u, P.refChecked H s U u (act u), P.refOut H s U u (act u)⟩, by simp [hu], ⟨hact u hu, fun x hx => hc.1 x hx⟩, fun i hi j hj => hc.2 i hi j hj⟩

/-- **The honest reference is complete and does not win.** -/
theorem honestComplete : Pinned.HonestComplete := by
  intro Q R S P D U H s act hP hU hact
  have hK := honest_correctSet P D U H s act hP hU hact
  refine ⟨hK, fun γ hγ0 hγ1 hwin => ?_⟩
  unfold Wins at hwin
  rw [hK] at hwin
  have hW : 0 ≤ totalWref P U.layout := Finset.sum_nonneg fun _ _ => by positivity
  have : totalWref P U.layout ≤ totalWref P U.layout / (1 - γ) := by
    rw [le_div_iff₀ (by linarith)]; nlinarith
  exact absurd hwin (not_lt.2 this)

theorem zero_actOK (sh : Shape) : ActOK sh (0 : Mat) := fun _ _ _ _ => by unfold InInt7; norm_num

theorem sixteen_Wref {Q R : Type} (P : D1.Params Q R Salt) {Ω : ℝ}
    (hW : ∀ U : Workload, U.InDomain D1.m1Domain → ∀ u < U.layout.N,
      Ω * ((U.layout.shape u).Wmm : ℝ) ≤ (P.Wref U.layout u : ℝ)) :
    16 * 2 ^ 44 * Ω ≤ ∑ u ∈ range 16, (P.toProtocol.Wref M1.sixteen.layout u : ℝ) := by
  have h : ∀ u ∈ range 16, Ω * 2 ^ 44 ≤ (P.toProtocol.Wref M1.sixteen.layout u : ℝ) := fun u hu => by
    have := hW M1.sixteen M1.sixteen_inDomain u (Finset.mem_range.1 hu)
    have e : ((M1.sixteen.layout.shape u).Wmm : ℝ) = 2 ^ 44 := by
      show ((milestoneShape.Wmm : ℕ) : ℝ) = 2 ^ 44
      unfold milestoneShape Shape.Wmm
      norm_num
    rw [e] at this
    exact this
  have := Finset.sum_le_sum h
  simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul] at this
  push_cast at this
  linarith

theorem honest_wins_sixteen {Q R : Type} (P : D1.Params Q R Salt) (hN : P.NoiseRange) {Ω : ℝ}
    (hW : ∀ U : Workload, U.InDomain D1.m1Domain → ∀ u < U.layout.N,
      Ω * ((U.layout.shape u).Wmm : ℝ) ≤ (P.Wref U.layout u : ℝ))
    (H : Q → R) (s : Salt) {γ T : ℝ} (hT : T / (1 - γ) < 16 * 2 ^ 44 * Ω) :
    Wins P.toProtocol M1.sixteen γ T (honestTranscript P.toProtocol H s M1.sixteen fun _ => 0) H s := by
  unfold Wins
  rw [honest_correctSet P.toProtocol _ M1.sixteen H s _ (d1_complete_m1Domain P hN) M1.sixteen_inDomain
    fun _ _ => zero_actOK _]
  exact lt_of_lt_of_le hT (sixteen_Wref P hW)

theorem salt_nonempty : Nonempty Salt := ⟨⟨0, Nat.two_pow_pos 128⟩⟩

/-- The honest reference as an adversary of the free model: no preprocessing, activation input 0. -/
def freeHonest {Q R : Type} (P : D1.Params Q R Salt) : Adversary (freeModel Q R Salt) :=
  ⟨done [], fun U act _ s => honestProg P U act s, fun _ => 0⟩

/-- The honest reference as an adversary of the witness model. -/
noncomputable def pricedHonest {Q R : Type} (P : D1.Params Q R Salt) : Adversary (outputPriced Q R Salt) :=
  ⟨done [], fun U act _ s => honestProg P U act s, fun _ => 0⟩

theorem freeHonest_bounded {Q R : Type} (P : D1.Params Q R Salt) {c : ℕ} (hc : P.QueryBound c) (U : Workload) :
    (freeHonest P).Bounded U 0 (U.layout.N * c) := fun H s =>
  ⟨le_rfl, (Nat.zero_add _).le.trans (calls_honestProg P H U _ s hc)⟩

theorem pricedHonest_bounded {Q R : Type} (P : D1.Params Q R Salt) {c : ℕ} (hc : P.QueryBound c) (U : Workload) :
    (pricedHonest P).Bounded U (∑ u ∈ range U.layout.N, ((U.layout.shape u).Wmm : ℝ)) (U.layout.N * c) :=
  fun H s => ⟨outputPriced_cost_le _ H U _ _ s, (Nat.zero_add _).le.trans (calls_honestProg P H U _ s hc)⟩

theorem freeHonest_transcript {Q R : Type} (P : D1.Params Q R Salt) (U : Workload) (H : Q → R) (s : Salt) :
    (freeHonest P).transcript U H s = honestTranscript P.toProtocol H s U fun _ => 0 :=
  eval_honestProg P H U _ s

theorem pricedHonest_transcript {Q R : Type} (P : D1.Params Q R Salt) (U : Workload) (H : Q → R) (s : Salt) :
    (pricedHonest P).transcript U H s = honestTranscript P.toProtocol H s U fun _ => 0 :=
  eval_honestProg P H U _ s

theorem sum_Wmm_sixteen : ∑ u ∈ range M1.sixteen.layout.N, ((M1.sixteen.layout.shape u).Wmm : ℝ) = 16 * 2 ^ 44 := by
  show ∑ u ∈ range 16, ((milestoneShape.Wmm : ℕ) : ℝ) = 16 * 2 ^ 44
  rw [Finset.sum_const, Finset.card_range]
  unfold milestoneShape Shape.Wmm
  norm_num

theorem lost_free {Q R : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Nonempty R] (P : D1.Params Q R Salt)
    (c : ℕ) (Ω : ℝ) (hN : P.NoiseRange) (hc : P.QueryBound c) (hc64 : 16 * c ≤ 2 ^ 64) (hΩ : 1 ≤ Ω)
    (hW : ∀ U : Workload, U.InDomain D1.m1Domain → ∀ u < U.layout.N,
      Ω * ((U.layout.shape u).Wmm : ℝ) ≤ (P.Wref U.layout u : ℝ))
    (γ : ℝ) (_hγ : γ < 1) (η : ℕ → ℕ → ℝ) (hG : Gγ (freeModel Q R Salt) P.toProtocol D1.m1Domain γ η) :
    1 ≤ η (16 * c) 16 := by
  have := salt_nonempty
  have h := hG M1.sixteen M1.sixteen_inDomain 0 le_rfl (M1.sixteen.layout.N * c) hc64 (freeHonest P)
    (freeHonest_bounded P hc M1.sixteen)
  refine le_of_eq_of_le (pr_eq_one fun x => ?_).symm h
  rw [freeHonest_transcript]
  exact honest_wins_sixteen P hN hW x.1 x.2 (by rw [zero_div]; positivity)

theorem lost_priced {Q R : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Nonempty R] (P : D1.Params Q R Salt)
    (c : ℕ) (Ω : ℝ) (hN : P.NoiseRange) (hc : P.QueryBound c) (hc64 : 16 * c ≤ 2 ^ 64) (hΩ : 1 ≤ Ω)
    (hW : ∀ U : Workload, U.InDomain D1.m1Domain → ∀ u < U.layout.N,
      Ω * ((U.layout.shape u).Wmm : ℝ) ≤ (P.Wref U.layout u : ℝ))
    (γ : ℝ) (hγ : γ < 1 - 1 / Ω) (η : ℕ → ℕ → ℝ)
    (hG : Gγ (outputPriced Q R Salt) P.toProtocol D1.m1Domain γ η) : 1 ≤ η (16 * c) 16 := by
  have := salt_nonempty
  have hB := pricedHonest_bounded P hc M1.sixteen
  rw [sum_Wmm_sixteen] at hB
  have h := hG M1.sixteen M1.sixteen_inDomain (16 * 2 ^ 44) (by positivity) (M1.sixteen.layout.N * c) hc64
    (pricedHonest P) hB
  refine le_of_eq_of_le (pr_eq_one fun x => ?_).symm h
  rw [pricedHonest_transcript]
  refine honest_wins_sixteen P hN hW x.1 x.2 ?_
  have hΩ0 : 0 < Ω := by linarith
  have h1 : 1 / Ω < 1 - γ := by linarith
  have h1' : (0 : ℝ) < 1 - γ := lt_of_le_of_lt (by positivity) h1
  rw [div_lt_iff₀ h1']
  rw [div_lt_iff₀ hΩ0] at h1
  nlinarith

/-- **The game can be lost.** -/
theorem gameCanBeLost : Pinned.GameCanBeLost := by
  intro Q R _ _ _ _ P c Ω hN hc hc64 hΩ hW
  exact ⟨fun γ hγ η hG => lost_free P c Ω hN hc hc64 hΩ hW γ hγ η hG,
    fun γ hγ η hG => lost_priced P c Ω hN hc hc64 hΩ hW γ hγ η hG⟩

end Pouw.Proofs
