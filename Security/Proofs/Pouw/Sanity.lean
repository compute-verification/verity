import Proofs.Pouw.Witness
import Proofs.Pouw.Copy
import Mathlib.Data.Set.Finite.List
import Mathlib.Data.Fintype.Pigeonhole

/-!
# Sanity: degenerate instances and side conditions

Each degenerate instance named in the task has a lemma here saying what happens there, and which side condition of
the pinned statements excludes it or makes it harmless.

| Instance | Lemma |
|---|---|
| empty workload | `wins_empty`, `tt_event_empty`: nothing can be won or certified |
| unbounded `N` | `index_collision`, `identical_checked`: two units get identical noise (why `InDomain` bounds `N`) |
| a unit alone in its group | `single_unit_not_inDomain`: outside the milestone domain; `lone_unit_over_budget`: its `m1Wref` exceeds `(199/198)·W_mm` |
| full groups of 16 on one weight | `fullGroups_inDomain`: inside the milestone domain |
| shared work in `W_ref` | `rowShare_covers`: the rounded-up row shares cover the group's and the weight's work |
| `m`, `k` or `n = 0` | `default_pos` (excluded from the default domain), `Wmm_eq_zero`, `Wref_eq_zero_of_Wmm` |
| `r = 0` | `noisedAct_rank_zero`, `tt_false_rank_zero` |
| `d = 0`, `d ∤ k` | `checkIdx_depth_zero`, `tt_false_depth_zero`; `blockEnd_full`, `blockEnd_last`, `Y_last` |
| zero noise | `tt_false_zero_noise` |
| `T = 0` | `wins_T_zero_iff` |
| `q = 0` | `transcript_indep_of_q_zero` |
| `Ω_max < 1/(1 − γ)` | `certified_le`, `omega_lt_inv_one_sub`, `reductionGamma_lt_of_omega_lt_one` |
| `γ₀ ≥ 1` | `tt_false_of_gamma0_ge_one`, and `tt_admitsRef_bound` for `γ₀ < 1` |
| randomized adversaries | `randomized_of_deterministic` |
| R_copy with zero noise | `copy_zero_noise`: Nz fails and R_copy's bound fails |
| (d′)'s hypotheses together | `milestone_hypotheses_consistent` (at `W_ref = m1Wref`, `Ω_max = Ω_M1`) |
-/

namespace Pouw.Sanity

open Finset Pouw.Proofs

variable {Q R S : Type}

theorem pr_of_isEmpty {α : Type} [Fintype α] [IsEmpty α] (P : α → Prop) : pr P = 0 :=
  pr_eq_zero fun x => isEmptyElim x

/-! ## Empty workload -/

theorem wins_empty (P : Protocol Q R S) (U : Workload) (hN : U.layout.N = 0) {γ T : ℝ} (hγ : γ < 1) (hT : 0 ≤ T)
    (τ : CommitLog) (H : Q → R) (s : S) : ¬ Wins P U γ T τ H s := by
  unfold Wins
  have : correctSet P U H s τ = ∅ := by
    have h := correctSet_subset_range P U H s τ
    rw [hN, Finset.range_zero] at h
    exact Finset.subset_empty.1 h
  rw [this, Finset.sum_empty, not_lt]
  exact div_nonneg hT (by linarith)

theorem tt_event_empty (P : Protocol Q R S) (U : Workload) (hN : U.layout.N = 0) {γ₀ T : ℝ} (hγ₀ : γ₀ < 1)
    (hT : 0 ≤ T) (τ : CommitLog) (H : Q → R) (s : S) :
    ¬ T / (1 - γ₀) < ∑ u ∈ checkedSet P U H s τ, ((U.layout.shape u).Wmm : ℝ) := by
  have : checkedSet P U H s τ = ∅ := by
    refine Finset.subset_empty.1 fun u hu => ?_
    simp only [checkedSet, Finset.mem_filter, hN, Finset.range_zero] at hu
    exact absurd hu.1 (Finset.notMem_empty u)
  rw [this, Finset.sum_empty, not_lt]
  exact div_nonneg hT (by linarith)

/-! ## Zero dimensions -/

theorem Wmm_eq_zero (sh : Shape) (h : sh.m = 0 ∨ sh.k = 0 ∨ sh.n = 0) : sh.Wmm = 0 := by
  unfold Shape.Wmm; rcases h with h | h | h <;> simp [h]

theorem default_pos (sh : Shape) (h : sh.InDefaultDomain) : 0 < sh.m ∧ 0 < sh.k ∧ 0 < sh.n := by
  obtain ⟨hm, -, hk, -, -, hn, -⟩ := h
  exact ⟨hm, by omega, by omega⟩

/-- Under the reduction's `Ω_max` bound a zero-work unit has `W_ref = 0`: it certifies nothing. -/
theorem Wref_eq_zero_of_Wmm (P : Protocol Q R S) (L : UnitLayout) (u : ℕ) (Ωmax : ℝ)
    (hW : (P.Wref L u : ℝ) ≤ Ωmax * ((L.shape u).Wmm : ℝ)) (h0 : (L.shape u).Wmm = 0) : P.Wref L u = 0 := by
  rw [h0] at hW
  exact_mod_cast le_antisymm (by simpa using hW) (Nat.cast_nonneg _)

/-! ## Rank, depth and the last block -/

theorem noisedAct_rank_zero (P : D1.Params Q R S) (hr : P.r = 0) (H : Q → R) (s : S) (U : Workload) (u : ℕ)
    (A : Mat) : P.noisedAct H s U u A = A := by
  funext i l
  simp [D1.Params.noisedAct, hr, mm]

theorem checkIdx_depth_zero (P : D1.Params Q R S) (hd : P.d = 0) (sh : Shape) : P.checkIdx sh = ∅ := by
  simp [D1.Params.checkIdx, D1.nBlocks, hd]

/-- Every block but the last is a full step of `d`. -/
theorem blockEnd_full {d k t : ℕ} (hd : 0 < d) (ht : t + 1 < D1.nBlocks d k) : D1.blockEnd d k t = (t + 1) * d := by
  unfold D1.blockEnd
  unfold D1.nBlocks at ht
  have h := (Nat.le_div_iff_mul_le hd).1 (show t + 2 ≤ (k + d - 1) / d by omega)
  have : (t + 1) * d ≤ k := by
    have : (t + 2) * d = (t + 1) * d + d := by ring
    omega
  exact min_eq_left this

/-- The last block ends at `k`, whether or not `d ∣ k`. -/
theorem blockEnd_last {d k : ℕ} (hd : 0 < d) (hk : 0 < k) : D1.blockEnd d k (D1.nBlocks d k - 1) = k := by
  unfold D1.blockEnd D1.nBlocks
  have h1 := Nat.div_add_mod (k + d - 1) d
  have h2 := Nat.mod_lt (k + d - 1) hd
  have hq : 1 ≤ (k + d - 1) / d := (Nat.le_div_iff_mul_le hd).2 (by omega)
  have : k ≤ ((k + d - 1) / d - 1 + 1) * d := by
    rw [Nat.sub_add_cancel hq, mul_comm]
    generalize d * ((k + d - 1) / d) = M at h1 ⊢
    omega
  exact min_eq_right this

/-- The last checked block is the whole noised product. -/
theorem Y_last (P : D1.Params Q R S) (hd : 0 < P.d) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat)
    (hk : 0 < (U.layout.shape u).k) :
    P.Y H s U u A (D1.nBlocks P.d (U.layout.shape u).k - 1)
      = mm (U.layout.shape u).k (P.noisedAct H s U u A) (P.noisedWt H s U u) := by
  unfold D1.Params.Y
  rw [blockEnd_last hd hk]

/-! ## TT fails in degenerate protocols -/

/-- A cost model in which every constant transcript is a free program with no oracle calls. -/
def FreeConstants (CM : CostModel Q R S) : Prop :=
  ∀ τ : CommitLog, ∃ p : CM.Prog, ∀ (H : Q → R) (U : Workload) (act : ℕ → Mat) (σ : State) (s : S),
    CM.run p H U act σ s = τ ∧ CM.cost p H U act σ s ≤ 0 ∧ CM.calls p H U act σ s = 0

theorem freeModel_freeConstants : FreeConstants (freeModel Q R S) :=
  fun τ => ⟨fun _ _ _ _ => QTree.done τ, fun _ _ _ _ _ => ⟨rfl, le_rfl, rfl⟩⟩

/-- If a constant transcript has every unit's checked values correct for every `H` and `s`, TT fails at `q = 0`
in every model with free constants, on every workload with positive work. -/
theorem tt_false_of_const [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] [Nonempty R] [Nonempty S]
    (CM : CostModel Q R S) (hCM : FreeConstants CM) (P : Protocol Q R S) (D : UnitLayout → Prop) (U : Workload)
    (hU : U.InDomain D) (hpos : 0 < ∑ u ∈ range U.layout.N, ((U.layout.shape u).Wmm : ℝ)) (τ : CommitLog)
    (hτ : ∀ H s, checkedSet P U H s τ = range U.layout.N) (γ₀ : ℝ) (ε : ℕ → ℕ → ℝ)
    (hTT : Assumptions.TT CM P D γ₀ ε) : 1 ≤ ε 0 U.layout.N := by
  obtain ⟨p, hp⟩ := hCM τ
  let A : Adversary CM := ⟨QTree.done [], p, fun _ => 0⟩
  have hA : A.Bounded U 0 0 := fun H s =>
    ⟨(hp H U _ [] s).2.1, by simp [A, QTree.calls, QTree.eval, (hp H U _ [] s).2.2]⟩
  have h := hTT.2 U hU 0 le_rfl 0 (by norm_num) A hA
  refine le_of_eq_of_le (pr_eq_one fun x => ?_).symm h
  show 0 / (1 - γ₀) < ∑ u ∈ checkedSet P U x.1 x.2 (CM.run p x.1 U (fun _ => 0) [] x.2), _
  rw [(hp x.1 U _ [] x.2).1, hτ, zero_div]
  exact hpos

/-- The all-zero commitment on every unit. -/
def zeroTranscript (U : Workload) : CommitLog := fun u => if u < U.layout.N then some ⟨0, 0, 0⟩ else none

theorem checkedSet_zeroTranscript (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S)
    (hY : ∀ u < U.layout.N, ∀ x ∈ P.checkIdx (U.layout.shape u), P.Y H s U u 0 x.1 x.2.1 x.2.2 = 0) :
    checkedSet P U H s (zeroTranscript U) = range U.layout.N := by
  classical
  ext u
  unfold checkedSet
  rw [Finset.mem_filter, Finset.mem_range, and_iff_left_iff_imp]
  intro hu
  refine ⟨⟨0, 0, 0⟩, by simp [zeroTranscript, hu], zero_actOK _, fun x hx => ?_⟩
  rw [hY u hu x hx]; rfl

/-- **Zero noise**: with zero activation noise, the zero-input transcript is correct for free, so TT fails. -/
theorem tt_false_zero_noise [Fintype Q] [DecidableEq Q] [Fintype R] [Nonempty R] (CM : CostModel Q R Salt)
    (hCM : FreeConstants CM) (P : D1.Params Q R Salt)
    (hE : ∀ sh a b, mm P.r (P.expAct sh a b).1 (P.expAct sh a b).2 = 0) (D : UnitLayout → Prop) (U : Workload)
    (hU : U.InDomain D) (hpos : 0 < ∑ u ∈ range U.layout.N, ((U.layout.shape u).Wmm : ℝ)) (γ₀ : ℝ)
    (ε : ℕ → ℕ → ℝ) (hTT : Assumptions.TT CM P.toProtocol D γ₀ ε) : 1 ≤ ε 0 U.layout.N := by
  have := salt_nonempty
  refine tt_false_of_const CM hCM P.toProtocol D U hU hpos (zeroTranscript U) (fun H s => ?_) γ₀ ε hTT
  refine checkedSet_zeroTranscript _ U H s fun u _ x _ => ?_
  show mm _ (P.noisedAct H s U u 0) (P.noisedWt H s U u) x.2.1 x.2.2 = 0
  have h0 : P.noisedAct H s U u 0 = 0 := by
    unfold D1.Params.noisedAct D1.Params.actF; rw [hE]; simp
  rw [h0]; simp [mm]

/-- **`r = 0`**: the noise vanishes and TT fails. -/
theorem tt_false_rank_zero [Fintype Q] [DecidableEq Q] [Fintype R] [Nonempty R] (CM : CostModel Q R Salt)
    (hCM : FreeConstants CM) (P : D1.Params Q R Salt) (hr : P.r = 0) (D : UnitLayout → Prop) (U : Workload)
    (hU : U.InDomain D) (hpos : 0 < ∑ u ∈ range U.layout.N, ((U.layout.shape u).Wmm : ℝ)) (γ₀ : ℝ)
    (ε : ℕ → ℕ → ℝ) (hTT : Assumptions.TT CM P.toProtocol D γ₀ ε) : 1 ≤ ε 0 U.layout.N :=
  tt_false_zero_noise CM hCM P (fun sh a b => by funext i l; simp [mm, hr]) D U hU hpos γ₀ ε hTT

/-- **`d = 0`**: nothing is compared and TT fails. -/
theorem tt_false_depth_zero [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] [Nonempty R] [Nonempty S]
    (CM : CostModel Q R S) (hCM : FreeConstants CM) (P : D1.Params Q R S) (hd : P.d = 0) (D : UnitLayout → Prop)
    (U : Workload) (hU : U.InDomain D) (hpos : 0 < ∑ u ∈ range U.layout.N, ((U.layout.shape u).Wmm : ℝ)) (γ₀ : ℝ)
    (ε : ℕ → ℕ → ℝ) (hTT : Assumptions.TT CM P.toProtocol D γ₀ ε) : 1 ≤ ε 0 U.layout.N := by
  refine tt_false_of_const CM hCM P.toProtocol D U hU hpos (zeroTranscript U) (fun H s => ?_) γ₀ ε hTT
  refine checkedSet_zeroTranscript _ U H s fun u _ x hx => ?_
  have : x ∈ P.checkIdx (U.layout.shape u) := hx
  rw [checkIdx_depth_zero P hd] at this
  exact absurd this (Finset.notMem_empty x)

/-! ## `T = 0` and `q = 0` -/

theorem wins_T_zero_iff (P : Protocol Q R S) (U : Workload) (γ : ℝ) (τ : CommitLog) (H : Q → R) (s : S) :
    Wins P U γ 0 τ H s ↔ 0 < ∑ u ∈ correctSet P U H s τ, (P.Wref U.layout u : ℝ) := by
  unfold Wins; rw [zero_div]

/-- At `q = 0` an adversary of the witness model commits the same transcript against every oracle. -/
theorem transcript_indep_of_q_zero (A : Adversary (outputPriced Q R S)) (U : Workload) (T : ℝ)
    (hA : A.Bounded U T 0) (H H' : Q → R) (s : S) : A.transcript U H s = A.transcript U H' s := by
  have h := (hA H s).2
  have h1 : A.pre.calls H = 0 := by omega
  have h2 : (A.online U A.act (A.pre.eval H) s).calls H = 0 := by
    have : (outputPriced Q R S).calls A.online H U A.act (A.pre.eval H) s = 0 := by omega
    exact this
  show (A.online U A.act (A.pre.eval H) s).eval H = (A.online U A.act (A.pre.eval H') s).eval H'
  rw [QTree.eval_of_calls_eq_zero h1 H', QTree.eval_of_calls_eq_zero h2 H']

/-! ## The certified work and `Ω_max` -/

/-- What the reduction certifies per unit is `(1 − γ)·W_ref ≤ (1 − γ₀)·W_mm < W_mm`. -/
theorem certified_le {γ₀ Ωmax Wref Wmm : ℝ} (hΩ : 0 < Ωmax) (hγ₀ : γ₀ < 1) (hW : Wref ≤ Ωmax * Wmm) :
    (1 - reductionGamma γ₀ Ωmax) * Wref ≤ (1 - γ₀) * Wmm := by
  rw [one_sub_reductionGamma]
  have h : 0 ≤ (1 - γ₀) / Ωmax := div_nonneg (by linarith) hΩ.le
  calc (1 - γ₀) / Ωmax * Wref ≤ (1 - γ₀) / Ωmax * (Ωmax * Wmm) := mul_le_mul_of_nonneg_left hW h
    _ = (1 - γ₀) * Wmm := by field_simp

/-- For `γ₀ > 0`, the reduction's `γ` always has `Ω_max < 1/(1 − γ)`: the game certifies less than `m·k·n`. -/
theorem omega_lt_inv_one_sub {γ₀ Ωmax : ℝ} (hΩ : 0 < Ωmax) (h0 : 0 < γ₀) (h1 : γ₀ < 1) :
    Ωmax < 1 / (1 - reductionGamma γ₀ Ωmax) := by
  rw [one_sub_reductionGamma, one_div_div, lt_div_iff₀ (by linarith)]
  nlinarith

/-- An `H_ref` cheaper than the GEMM (`Ω_max < 1`) gives `γ < γ₀`. -/
theorem reductionGamma_lt_of_omega_lt_one {γ₀ Ωmax : ℝ} (hΩ : 0 < Ωmax) (hΩ1 : Ωmax < 1) (h1 : γ₀ < 1) :
    reductionGamma γ₀ Ωmax < γ₀ := by
  unfold reductionGamma
  have : 1 - γ₀ < (1 - γ₀) / Ωmax := by rw [lt_div_iff₀ hΩ]; nlinarith
  linarith

/-! ## TT against the honest reference -/

theorem honest_checkedSet (P : Protocol Q R S) (D : UnitLayout → Prop) (U : Workload) (H : Q → R) (s : S)
    (act : ℕ → Mat) (hP : P.Complete D) (hU : U.InDomain D) (hact : ∀ u < U.layout.N, ActOK (U.layout.shape u) (act u)) :
    checkedSet P U H s (honestTranscript P H s U act) = range U.layout.N := by
  refine le_antisymm (fun u hu => ?_) ?_
  · simp only [checkedSet, Finset.mem_filter] at hu; exact hu.1
  · rw [← honest_correctSet P D U H s act hP hU hact]
    exact correctSet_subset_checkedSet P U H s _

/-- The honest reference is an adversary of TT's game: if TT holds with `ε(qref) < 1`, the model's `H_ref` costs at
least `(1 − γ₀)·∑ W_mm`. -/
theorem tt_admitsRef_bound [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] [Nonempty R] [Nonempty S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : UnitLayout → Prop) (qref : Workload → ℕ) (hP : P.Complete D)
    (hAdm : AdmitsRef CM P D qref) (U : Workload) (hU : U.InDomain D) (hq : qref U ≤ 2 ^ 64) (γ₀ : ℝ)
    (ε : ℕ → ℕ → ℝ) (hTT : Assumptions.TT CM P D γ₀ ε) (hε : ε (qref U) U.layout.N < 1) :
    ∑ u ∈ range U.layout.N, ((U.layout.shape u).Wmm : ℝ) ≤ totalWref P U.layout / (1 - γ₀) := by
  by_contra hlt
  push Not at hlt
  obtain ⟨prog, hprog⟩ := hAdm
  have hp := hprog U hU (fun _ => 0) fun _ _ => zero_actOK _
  let A : Adversary CM := ⟨QTree.done [], prog, fun _ => 0⟩
  have hA : A.Bounded U (totalWref P U.layout) (qref U) := fun H s =>
    ⟨(hp H s).2.1, by simpa [A, QTree.calls, QTree.eval] using (hp H s).2.2⟩
  have hT : 0 ≤ totalWref P U.layout := Finset.sum_nonneg fun _ _ => by positivity
  have h := hTT.2 U hU _ hT (qref U) hq A hA
  have h1 : (1 : ℝ) ≤ ε (qref U) U.layout.N := by
    refine le_of_eq_of_le (pr_eq_one fun x => ?_).symm h
    show totalWref P U.layout / (1 - γ₀) < ∑ u ∈ checkedSet P U x.1 x.2 (CM.run prog x.1 U (fun _ => 0) [] x.2), _
    rw [(hp x.1 x.2).1, honest_checkedSet P D U x.1 x.2 _ hP hU fun _ _ => zero_actOK _]
    exact hlt
  linarith

/-- **`γ₀ ≥ 1`**: TT is unsatisfiable (error below 1) whenever `H_ref` runs in the model on a workload with work. -/
theorem tt_false_of_gamma0_ge_one [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] [Nonempty R] [Nonempty S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : UnitLayout → Prop) (qref : Workload → ℕ) (hP : P.Complete D)
    (hAdm : AdmitsRef CM P D qref) (U : Workload) (hU : U.InDomain D) (hq : qref U ≤ 2 ^ 64)
    (hpos : 0 < ∑ u ∈ range U.layout.N, ((U.layout.shape u).Wmm : ℝ)) (γ₀ : ℝ) (hγ₀ : 1 ≤ γ₀) (ε : ℕ → ℕ → ℝ)
    (hTT : Assumptions.TT CM P D γ₀ ε) : 1 ≤ ε (qref U) U.layout.N := by
  by_contra hε
  push Not at hε
  have h := tt_admitsRef_bound CM P D qref hP hAdm U hU hq γ₀ ε hTT hε
  have hT : 0 ≤ totalWref P U.layout := Finset.sum_nonneg fun _ _ => by positivity
  have : totalWref P U.layout / (1 - γ₀) ≤ 0 := div_nonpos_of_nonneg_of_nonpos hT (by linarith)
  linarith

/-- **`γ ≥ 1`**: every budget is beaten by one correct unit of positive work, so `G_γ` cannot hold with `η < 1`
against the honest reference (the pinned statements assume `γ < 1` or `γ₀ < 1`). -/
theorem wins_of_gamma_ge_one (P : Protocol Q R S) (U : Workload) {γ T : ℝ} (hγ : 1 ≤ γ) (hT : 0 ≤ T)
    (τ : CommitLog) (H : Q → R) (s : S) (hpos : 0 < ∑ u ∈ correctSet P U H s τ, (P.Wref U.layout u : ℝ)) :
    Wins P U γ T τ H s :=
  lt_of_le_of_lt (div_nonpos_of_nonneg_of_nonpos hT (by linarith)) hpos

/-! ## Unbounded `N` -/

/-- With a finite query type and `c` queries per unit, some two indices get the same queries. -/
theorem index_collision [Finite Q] (P : D1.Params Q R S) (c : ℕ) (hc : P.QueryBound c) (s : S) (A : Mat)
    (w : ℕ) : ∃ u u', u ≠ u' ∧ P.qAct s u A w = P.qAct s u' A w ∧ P.qWt s u w = P.qWt s u' w := by
  let T := {l : List Q // l.length ≤ c}
  have : Finite T := (List.finite_length_le Q c).to_subtype
  let f : ℕ → T × T := fun u =>
    (⟨P.qAct s u A w, by have := hc s u A w; omega⟩, ⟨P.qWt s u w, by have := hc s u A w; omega⟩)
  obtain ⟨u, u', hne, heq⟩ := Finite.exists_ne_map_eq_of_infinite f
  refine ⟨u, u', hne, ?_, ?_⟩
  · have := congrArg (fun p => p.1.1) heq; simpa [f] using this
  · have := congrArg (fun p => p.2.1) heq; simpa [f] using this

/-- **Why `InDomain` bounds `N`**: two such indices of equal shape and weight have identical checked values for every
oracle, so a workload holding both lets one computation be committed twice. -/
theorem identical_checked [Finite Q] (P : D1.Params Q R S) (c : ℕ) (hc : P.QueryBound c) (s : S) (A : Mat) :
    ∃ u u', u ≠ u' ∧ ∀ (U : Workload) (H : Q → R), U.layout.shape u = U.layout.shape u' →
      U.layout.wid u = 0 → U.layout.wid u' = 0 → P.Y H s U u A = P.Y H s U u' A := by
  obtain ⟨u, u', hne, h1, h2⟩ := index_collision P c hc s A 0
  refine ⟨u, u', hne, fun U H hs hw hw' => ?_⟩
  unfold D1.Params.Y D1.Params.noisedAct D1.Params.noisedWt D1.Params.actF D1.Params.wtF
  rw [hs, hw, hw', h1, h2]

/-! ## The milestone domain and (d′)'s hypotheses -/

/-- A milestone-shape unit alone in its group on its weight (`2^12` rows) is outside the milestone domain, so its
group's and its weight's shared work are never charged to a single unit. -/
theorem single_unit_not_inDomain :
    ¬ (⟨⟨1, fun _ => milestoneShape, fun _ => 0⟩, fun _ => 0⟩ : Workload).InDomain D1.m1Domain := by
  intro h
  have := (h.2.1 0 (by norm_num)).2.2
  revert this
  show ¬ 2 ^ 16 ≤ ∑ u ∈ (range 1).filter (fun _ => (0 : ℕ) = 0), milestoneShape.m
  rw [Finset.filter_true_of_mem fun _ _ => rfl, Finset.sum_range_one]
  unfold milestoneShape
  norm_num

/-- **Full groups are in the milestone domain**: a layout of milestone shapes and 64-bit weight ids in which every
unit's group `16·⌊u/16⌋ … 16·⌊u/16⌋ + 15` lies below `N` and on the unit's weight satisfies `m1Domain`'s rows
condition (`16·2^12 = 2^16`). -/
theorem fullGroups_inDomain (L : UnitLayout) (hsh : ∀ u < L.N, D1.M1Shape (L.shape u) ∧ L.wid u < 2 ^ 64)
    (hfull : ∀ u < L.N, ∀ t < 16, 16 * (u / 16) + t < L.N ∧ L.wid (16 * (u / 16) + t) = L.wid u) :
    D1.m1Domain L := by
  intro u hu
  refine ⟨(hsh u hu).1, (hsh u hu).2, ?_⟩
  have hsub : (range 16).image (fun t => 16 * (u / 16) + t) ⊆
      (range L.N).filter (fun u' => u' / 16 = u / 16 ∧ L.wid u' = L.wid u) := by
    intro x hx
    obtain ⟨t, ht, rfl⟩ := Finset.mem_image.1 hx
    have ht' := Finset.mem_range.1 ht
    obtain ⟨hlt, hw⟩ := hfull u hu t ht'
    exact Finset.mem_filter.2 ⟨Finset.mem_range.2 hlt, by omega, hw⟩
  have h1 : 2 ^ 16 ≤ ∑ t ∈ range 16, (L.shape (16 * (u / 16) + t)).m := by
    calc 2 ^ 16 = ∑ _t ∈ range 16, 2 ^ 12 := by simp
      _ ≤ ∑ t ∈ range 16, (L.shape (16 * (u / 16) + t)).m :=
          Finset.sum_le_sum fun t ht => (hsh _ (hfull u hu t (Finset.mem_range.1 ht)).1).1.1
  have h2 : ∑ t ∈ range 16, (L.shape (16 * (u / 16) + t)).m =
      ∑ u' ∈ (range 16).image (fun t => 16 * (u / 16) + t), (L.shape u').m := by
    rw [Finset.sum_image fun _ _ _ _ h => by simpa using h]
  have h3 : ∑ u' ∈ (range 16).image (fun t => 16 * (u / 16) + t), (L.shape u').m ≤ D1.m1GroupRows L u := by
    unfold D1.m1GroupRows
    exact Finset.sum_le_sum_of_subset hsub
  omega

/-- **(d′)'s hypotheses hold together at the explicit schedule**: at `W_ref = m1Wref` and `Ω_max = Ω_M1`, the
output-priced model runs `H_ref` and satisfies TT(0.5%) with error 0 for `D1.m1`, so (d′) is not vacuous there. -/
theorem milestone_hypotheses_consistent :
    ∃ qref : Workload → ℕ, 0 < D1.m1Omega ∧ D1.m1Omega ≤ 199 / 198 ∧
      (∀ U : Workload, U.InDomain D1.m1Domain → ∀ u < U.layout.N,
        (D1.m1Wref U.layout u : ℝ) ≤ D1.m1Omega * ((U.layout.shape u).Wmm : ℝ)) ∧
      AdmitsRef (outputPriced D1.M1Q D1.M1R Salt) (D1.m1 D1.m1Wref).toProtocol D1.m1Domain qref ∧
      Assumptions.TT (outputPriced D1.M1Q D1.M1R Salt) (D1.m1 D1.m1Wref).toProtocol D1.m1Domain Params.γ0
        (fun _ _ => 0) := by
  obtain ⟨hW, hΩ, -⟩ := milestoneWref
  obtain ⟨hTT, hAdm⟩ := ttWitness D1.m1Wref fun U hU u hu => (hW U hU u hu).1
  exact ⟨_, by unfold D1.m1Omega; norm_num, hΩ, fun U hU u hu => (hW U hU u hu).2, hAdm, hTT⟩

/-- **The shares cover the shared work**: over units whose rows sum to `rows > 0`, the rounded-up shares
`⌈work·m_u/rows⌉` add up to at least `work`. `m1Wref` charges each (epoch, weight, group) product over the group's
units on the weight, and each (epoch, weight) product over the weight's units, this way. -/
theorem rowShare_covers (T : Finset ℕ) (mu : ℕ → ℕ) (work : ℕ) (h : 0 < ∑ u ∈ T, mu u) :
    work ≤ ∑ u ∈ T, D1.rowShare work (mu u) (∑ u ∈ T, mu u) := by
  set rows := ∑ u ∈ T, mu u with hrows
  have key : ∀ u, work * mu u ≤ rows * D1.rowShare work (mu u) rows := by
    intro u
    unfold D1.rowShare
    have h1 := Nat.div_add_mod (work * mu u + rows - 1) rows
    have h2 := Nat.mod_lt (work * mu u + rows - 1) h
    generalize (work * mu u + rows - 1) / rows = q at *
    generalize (work * mu u + rows - 1) % rows = r at *
    generalize work * mu u = a at *
    generalize rows * q = b at *
    omega
  have h3 : rows * work ≤ rows * ∑ u ∈ T, D1.rowShare work (mu u) rows := by
    rw [Finset.mul_sum]
    calc rows * work = ∑ u ∈ T, work * mu u := by rw [← Finset.mul_sum, hrows]; ring
      _ ≤ _ := Finset.sum_le_sum fun u _ => key u
  exact Nat.le_of_mul_le_mul_left h3 h

/-- **Why the domain asks for `2^16` rows per group**: a milestone-shape unit alone on its weight carries its group's
and its weight's whole shared work, and `m1Wref` is then `2^44 + 253·2^30 + 9792·2^12`, above `(199/198)·W_mm`, the
budget of a 1% gap. -/
theorem lone_unit_over_budget :
    (199 / 198 : ℝ) * (milestoneShape.Wmm : ℝ) < (D1.m1Wref ⟨1, fun _ => milestoneShape, fun _ => 0⟩ 0 : ℝ) := by
  have hg : D1.m1GroupRows ⟨1, fun _ => milestoneShape, fun _ => 0⟩ 0 = 2 ^ 12 := by
    unfold D1.m1GroupRows
    rw [Finset.filter_true_of_mem fun u hu => ⟨by have : u < 1 := Finset.mem_range.1 hu; omega, rfl⟩,
      Finset.sum_range_one]
    rfl
  have hr : UnitLayout.rowsOf ⟨1, fun _ => milestoneShape, fun _ => 0⟩ 0 = 2 ^ 12 := by
    unfold UnitLayout.rowsOf
    rw [Finset.filter_true_of_mem fun u _ => rfl, Finset.sum_range_one]
    rfl
  have hW : D1.m1Wref ⟨1, fun _ => milestoneShape, fun _ => 0⟩ 0 = 2 ^ 44 + 253 * 2 ^ 30 + 9792 * 2 ^ 12 := by
    dsimp only [D1.m1Wref]
    rw [hg, hr]
    unfold D1.rowShare milestoneShape
    norm_num
  rw [hW]
  unfold Shape.Wmm milestoneShape
  norm_num

/-! ## Randomized adversaries -/

/-- **Deterministic adversaries suffice**: an adversary using coins `C` independent of `(H, s)`, within `(T, q)` for
every coin value, wins with probability at most `η`. -/
theorem randomized_of_deterministic [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (D : UnitLayout → Prop) (γ : ℝ) (η : ℕ → ℕ → ℝ) (hG : Gγ CM P D γ η) (U : Workload)
    (hU : U.InDomain D) (T : ℝ) (hT : 0 ≤ T) (q : ℕ) (hq : q ≤ 2 ^ 64) (C : Type) [Fintype C] [Nonempty C]
    (A : C → Adversary CM) (hA : ∀ c, (A c).Bounded U T q) :
    pr (fun x : (Q → R) × S × C => Wins P U γ T ((A x.2.2).transcript U x.1 x.2.1) x.1 x.2.1) ≤ η q U.layout.N := by
  let e : ((Q → R) × S × C) ≃ C × ((Q → R) × S) :=
    { toFun := fun x => (x.2.2, (x.1, x.2.1)), invFun := fun y => (y.2.1, y.2.2, y.1),
      left_inv := fun _ => rfl, right_inv := fun _ => rfl }
  let W := fun y : C × ((Q → R) × S) => Wins P U γ T ((A y.1).transcript U y.2.1 y.2.2) y.2.1 y.2.2
  have hη : 0 ≤ η q U.layout.N := by
    obtain ⟨c⟩ := ‹Nonempty C›
    exact (pr_nonneg _).trans (hG U hU T hT q hq (A c) (hA c))
  calc _ = pr W := pr_equiv e W
    _ ≤ _ := pr_prod_le W _ hη fun c => hG U hU T hT q hq (A c) (hA c)

/-! ## R_copy without noise -/

/-- **R_copy's bound needs the noise**: with constant values Nz fails, and the one-class adversary is correct on a
two-position unit at cost 16, so for `0 < γ₀ < 1/2` the event has probability 1, above the bound `2^(−2γ₀)`. -/
theorem copy_zero_noise (γ₀ : ℝ) (h0 : 0 < γ₀) (h1 : γ₀ < 1 / 2) :
    ¬ Copy.Nz (fun _ => 2) (fun (_ : Unit) (_ _ : ℕ) => (0 : ℕ)) ∧
      ∑ _u ∈ range 1, (2 : ℝ) ^ (-(γ₀ * ((2 : ℕ) : ℝ))) <
        pr (fun ω : Unit => (Copy.cost (fun _ => 2) 1 ⟨fun _ _ => 0⟩ : ℝ) / (1 - γ₀) <
          16 * ∑ _u ∈ Copy.correctSet (fun _ => 2) (fun (_ : Unit) (_ _ : ℕ) => (0 : ℕ)) 1 ⟨fun _ _ => 0⟩ ω,
            ((2 : ℕ) : ℝ)) := by
  refine ⟨fun h => ?_, ?_⟩
  · have := h 0 {0} (fun x hx => by rw [Finset.mem_singleton.1 hx]; simp) (fun _ _ => 0) fun _ _ _ _ _ => rfl
    rw [pr_eq_one fun _ => fun _ _ => rfl] at this
    norm_num at this
  · have hcls : (Copy.classes (fun _ => 2) ⟨fun _ _ => 0⟩ 0).card = 1 := by
      dsimp only [Copy.classes]
      rw [Finset.image_const ⟨0, by simp⟩, Finset.card_singleton]
    have hcost : Copy.cost (fun _ => 2) 1 ⟨fun _ _ => 0⟩ = 16 := by
      unfold Copy.cost
      rw [Finset.sum_range_one, hcls]
    have hK : ∀ ω : Unit,
        Copy.correctSet (fun _ => 2) (fun (_ : Unit) (_ _ : ℕ) => (0 : ℕ)) 1 ⟨fun _ _ => 0⟩ ω = {0} := fun ω => by
      ext u
      simp only [Copy.correctSet, Finset.mem_filter, Finset.mem_range, Finset.mem_singleton]
      exact ⟨fun h => by omega, fun h => ⟨by omega, fun _ _ _ _ _ => rfl⟩⟩
    rw [pr_eq_one fun ω => ?_]
    · rw [Finset.sum_range_one]
      exact Real.rpow_lt_one_of_one_lt_of_neg (by norm_num) (by push_cast; linarith)
    · rw [hcost, hK ω, Finset.sum_singleton, div_lt_iff₀ (by linarith)]
      push_cast
      nlinarith

end Pouw.Sanity
