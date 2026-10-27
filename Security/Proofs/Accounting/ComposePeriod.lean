import Definitions.Accounting.Guarantees.ComposePeriod
import Proofs.Accounting.TS.Rounds
import Proofs.Accounting.Compose.CircuitWork
import Proofs.Accounting.ComposePeriod.Covers

/-!
# PoUW's work over a monitoring period: proofs

`Accounting.Guarantees.ComposePeriod.PeriodBudget` is proved here: a window's `serviceGame` is one round
`.round d pass (fun _ => .done)`, `periodGame` puts the next window's game in place of its end (`consGame`), and
`RoundGame.BudgetAt` of `.round d pass next` is the round's two conditions and `BudgetAt` of every `next ω`, so the
period's budget is its windows', by induction on the period. The window's live tiles only pick its weights.

`Accounting.Guarantees.ComposePeriod.PeriodTailMax` is proved here too, by induction on the game for every threshold
`D` at once (`prob_prefix_le`): a prefix reaching `D > 0` passes its first round and has the rest reach `D − d`, so,
conditioning on the first round's coins as `expect_weight_le` does, its probability is at most
`E[1{pass}·e^{c·d}]·e^{−c·D} ≤ e^{−c·D}`. At `c = 1/Λ` the round bound is `BudgetAt`'s (`expBound_of_budgetAt`).

`CircuitWorkPeriod` is proved here from the period layer's covers (`PeriodCoversStmt`, which
`circuitWorkPeriod_of_covers` reads as a hypothesis and `ComposePeriod/Covers.lean` proves as `periodCovers`), and
`CircuitCreditPeriod` from its assembly at `a, b > 0`. It is `CircuitWork`'s proof at `n` rounds, on the accepted
prefix's live tiles:
* per window, `ServiceBudget` at the detected live weights (`liveShare`: `0` on a tile of no live words, else
  `liveCone`), which are at most `shareCone`, so `CircuitWork`'s `ρ`, stage-share and luck premises hold of them, with
  its `LuckAt` from `PerLiveTile` (`luckAt_of_perLiveTile`: luck `1` on a dead tile, of weight `0`); then
  `PeriodBudget` (`budget_period`);
* `PeriodTailMax` once at `t = ln(1/δ_s)`, read at the prefix `j = prefixLen` (the credited windows are the first
  `prefixLen`, `credited_iff`); `DrawnRightOff` at `n` rounds (its event `n·(δ_bind + δ_ks)`), off which every drawn
  tile of a credited window is right at its committed values, so `PrefixOf` passes the play's first `j` rounds, and the
  budget game's, whose checks are the play's (`acceptsUpTo_periodGame`);
* per credited window, the covers, `UnitFixed`'s sums (a good tile is live in a credited window, `periodMode`'s
  not-whole tiles are its live tiles not skipped whole) and `liveCone = liveShare` plus the cones of tiles of no live
  words, at most `zeroCone` a window (`zeroS_le`: a dead tile with words reaches no live words); summed over the
  prefix (`deficiencyUpTo_periodGame`), the shares in the cheaper of the two (`liveWork_of`).
-/

namespace Accounting.SecurityProofs.ComposePeriod

open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-- A window's round then the rest has its budget when the round has and every continuation has. -/
theorem budgetAt_consGame {Ω : Type} [Fintype Ω] {n : ℕ} {Λ : ℝ} (g : RoundGame Ω 1) (next : Ω → RoundGame Ω n)
    (hg : RoundGame.BudgetAt Λ g) (hn : ∀ ω, RoundGame.BudgetAt Λ (next ω)) :
    RoundGame.BudgetAt Λ (consGame g next) := by
  cases g with
  | round d pass rest =>
    simp only [consGame, RoundGame.BudgetAt] at hg ⊢
    exact ⟨hg.1, hg.2.1, hn⟩

/-- `PeriodBudget`: by induction on the period, `budgetAt_consGame` at each window. -/
theorem PeriodBudget : Accounting.Guarantees.ComposePeriod.PeriodBudget := by
  intro C Ξ _ _ N M n stg w a b K Λ Wp h
  induction Wp with
  | done => simp only [periodGame, RoundGame.BudgetAt]
  | window live W next ih =>
    simp only [Period.EveryWindow] at h
    simp only [periodGame]
    exact budgetAt_consGame _ _ h.1 fun ω => ih ω (h.2 ω)

/-- **The exponential round bound at every prefix**: for `c ≥ 0`, some prefix of the play passes with share at least
`D` with probability at most `e^{−c·D}`. -/
theorem prob_prefix_le {Ω : Type} [Fintype Ω] {c : ℝ} (hc : 0 ≤ c) :
    ∀ {n : ℕ} (G : RoundGame Ω n), G.ExpBound c → ∀ D : ℝ,
      prob (fun x => ∃ j, RoundGame.acceptsUpTo G j x ∧ D ≤ RoundGame.deficiencyUpTo G j x) ≤ Real.exp (-(c * D))
  | _, .done, _, D => by
    rcases le_or_gt D 0 with hD | hD
    · exact (prob_le_one _).trans (Real.one_le_exp (neg_nonneg.mpr (mul_nonpos_of_nonneg_of_nonpos hc hD)))
    refine (prob_mono (q := fun _ => False) fun x hx => ?_).trans
      (Accounting.SecurityProofs.TS.Rounds.prob_false.trans_le (Real.exp_pos _).le)
    obtain ⟨j, -, hj⟩ := hx
    simp only [RoundGame.deficiencyUpTo] at hj
    linarith
  | _, .round d pass next, h, D => by
    classical
    obtain ⟨h1, h2⟩ := h
    rcases le_or_gt D 0 with hD | hD
    · exact (prob_le_one _).trans (Real.one_le_exp (neg_nonneg.mpr (mul_nonpos_of_nonneg_of_nonpos hc hD)))
    rw [prob_eq_expect, expect_prod]
    calc _ ≤ expect (fun a => Real.exp (-(c * D)) * if pass a then Real.exp (c * d a) else 0) :=
          expect_mono fun a => ?_
      _ = Real.exp (-(c * D)) * expect (fun a => if pass a then Real.exp (c * d a) else 0) :=
          expect_mul_left _ _
      _ ≤ Real.exp (-(c * D)) * 1 := mul_le_mul_of_nonneg_left h1 (Real.exp_pos _).le
      _ = Real.exp (-(c * D)) := mul_one _
    by_cases hpa : pass a
    · rw [ite_eq_left hpa]
      refine (Accounting.SecurityProofs.TS.Rounds.expect_ind_le_prob (Q := fun y => ∃ j,
        RoundGame.acceptsUpTo (next a) j y ∧ D - d a ≤ RoundGame.deficiencyUpTo (next a) j y)
        fun y hy => ?_).trans ?_
      · obtain ⟨j, hj1, hj2⟩ := hy
        cases j with
        | zero =>
          simp only [RoundGame.deficiencyUpTo] at hj2
          linarith
        | succ j =>
          simp only [RoundGame.acceptsUpTo, RoundGame.deficiencyUpTo] at hj1 hj2
          exact ⟨j, hj1.2, by linarith⟩
      · calc _ ≤ Real.exp (-(c * (D - d a))) := prob_prefix_le hc (next a) (h2 a) (D - d a)
          _ = Real.exp (-(c * D)) * Real.exp (c * d a) := by rw [← Real.exp_add]; ring_nf
    · rw [ite_eq_right hpa, mul_zero]
      refine (Accounting.SecurityProofs.TS.Rounds.expect_ind_le_prob (Q := fun _ => False) fun y hy => ?_).trans
        Accounting.SecurityProofs.TS.Rounds.prob_false.le
      obtain ⟨j, hj1, hj2⟩ := hy
      cases j with
      | zero =>
        simp only [RoundGame.deficiencyUpTo] at hj2
        linarith
      | succ j =>
        simp only [RoundGame.acceptsUpTo] at hj1
        exact hpa hj1.1

/-- `PeriodTailMax`: `prob_prefix_le` at `c = 1/Λ`, whose round bound `BudgetAt` gives. -/
theorem PeriodTailMax : Accounting.Guarantees.ComposePeriod.PeriodTailMax := by
  intro Ω _ n g Λ hΛ hB t
  have h := prob_prefix_le (c := 1 / Λ) (by positivity) g
    (Accounting.SecurityProofs.TS.Rounds.expBound_of_budgetAt g hB) (Λ * t)
  rwa [one_div, inv_mul_cancel_left₀ hΛ.ne'] at h

/-! ## The live weights -/

/-- **A tile's detected live weight**: its `liveCone` when it carries live words, and `0` for a tile of none, whose
skip no draw detects. The period's budget is taken at these weights. -/
noncomputable def liveShare {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (live : Fin Np → Bool)
    (t : Fin Np) : ℝ :=
  if liveWords Sc Cd Np live t = 0 then 0 else liveCone Sc Cd Np live t

section Weights

variable {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (live : Fin Np → Bool)

theorem liveWords_le_tileWords (t : Fin Np) : liveWords Sc Cd Np live t ≤ tileWords Sc Cd Np t := by
  unfold liveWords
  split_ifs
  · exact le_rfl
  · exact Nat.zero_le _

theorem liveWords_of_live {t : Fin Np} (h : live t = true) : liveWords Sc Cd Np live t = tileWords Sc Cd Np t := by
  unfold liveWords
  rw [ite_eq_left h]

theorem liveWords_of_dead {t : Fin Np} (h : ¬live t = true) : liveWords Sc Cd Np live t = 0 := by
  unfold liveWords
  rw [ite_eq_right h]

theorem liveCone_nonneg (t : Fin Np) : 0 ≤ liveCone Sc Cd Np live t :=
  Finset.sum_nonneg fun t' _ => by
    split_ifs
    · exact div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
    · exact le_rfl

theorem liveCone_le_tileCone (t : Fin Np) : liveCone Sc Cd Np live t ≤ tileCone Sc Cd Np t :=
  Finset.sum_le_sum fun t' _ => by
    split_ifs
    · exact div_le_div_of_nonneg_right (Nat.cast_le.mpr (liveWords_le_tileWords Sc Cd Np live t'))
        (Nat.cast_nonneg _)
    · exact le_rfl

theorem liveShare_nonneg (t : Fin Np) : 0 ≤ liveShare Sc Cd Np live t := by
  unfold liveShare
  split_ifs
  · exact le_rfl
  · exact liveCone_nonneg Sc Cd Np live t

theorem liveShare_of_dead {t : Fin Np} (h : ¬live t = true) : liveShare Sc Cd Np live t = 0 := by
  unfold liveShare
  rw [ite_eq_left (liveWords_of_dead Sc Cd Np live h)]

theorem liveShare_le_shareCone (t : Fin Np) : liveShare Sc Cd Np live t ≤ shareCone Sc Cd Np t := by
  unfold liveShare
  by_cases h0 : liveWords Sc Cd Np live t = 0
  · rw [ite_eq_left h0]
    exact Accounting.SecurityProofs.Compose.WorkProof.shareCone_nonneg Sc Cd Np t
  · rw [ite_eq_right h0]
    have hl : live t = true := by
      by_contra hl
      exact h0 (liveWords_of_dead Sc Cd Np live hl)
    have hw : ¬tileWords Sc Cd Np t = 0 := by
      rwa [liveWords_of_live Sc Cd Np live hl] at h0
    unfold shareCone
    rw [ite_eq_right hw]
    exact liveCone_le_tileCone Sc Cd Np live t

/-- **A tile's live words are at most `N_w` times its detected live weight**: its cone holds the tile itself. -/
theorem liveWords_le_liveShare (t : Fin Np) :
    (liveWords Sc Cd Np live t : ℝ) ≤ liveShare Sc Cd Np live t * (wordsOf Sc Cd : ℝ) := by
  classical
  unfold liveShare
  by_cases h0 : liveWords Sc Cd Np live t = 0
  · rw [ite_eq_left h0, h0]
    simp
  · rw [ite_eq_right h0]
    have hpos : 0 < wordsOf Sc Cd := Nat.lt_of_lt_of_le (Nat.pos_of_ne_zero h0)
      ((liveWords_le_tileWords Sc Cd Np live t).trans
        (Accounting.SecurityProofs.Compose.CoversProof.tileWords_le_wordsOf Sc Cd Np t))
    have hnn : ∀ t' ∈ (Finset.univ : Finset (Fin Np)), 0 ≤ (if Cd.Reaches t.val t'.val then
        (liveWords Sc Cd Np live t' : ℝ) / (wordsOf Sc Cd : ℝ) else 0) := fun t' _ => by
      split_ifs
      · exact div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
      · exact le_rfl
    have hself : (if Cd.Reaches t.val t.val then (liveWords Sc Cd Np live t : ℝ) / (wordsOf Sc Cd : ℝ) else 0) =
        (liveWords Sc Cd Np live t : ℝ) / (wordsOf Sc Cd : ℝ) := ite_eq_left Relation.ReflTransGen.refl
    have hle : (liveWords Sc Cd Np live t : ℝ) / (wordsOf Sc Cd : ℝ) ≤ liveCone Sc Cd Np live t :=
      calc (liveWords Sc Cd Np live t : ℝ) / (wordsOf Sc Cd : ℝ)
          = (if Cd.Reaches t.val t.val then (liveWords Sc Cd Np live t : ℝ) / (wordsOf Sc Cd : ℝ) else 0) :=
            hself.symm
        _ ≤ liveCone Sc Cd Np live t := Finset.single_le_sum hnn (Finset.mem_univ t)
    calc (liveWords Sc Cd Np live t : ℝ)
        = (liveWords Sc Cd Np live t : ℝ) / (wordsOf Sc Cd : ℝ) * (wordsOf Sc Cd : ℝ) :=
          (div_mul_cancel₀ _ (Nat.cast_pos.mpr hpos).ne').symm
      _ ≤ liveCone Sc Cd Np live t * (wordsOf Sc Cd : ℝ) := mul_le_mul_of_nonneg_right hle (Nat.cast_nonneg _)

end Weights

theorem stageShare_mono {N : ℕ} (stg : Fin N → ℕ) {w w' : Fin N → ℝ} (h : ∀ v, w v ≤ w' v) (j : ℕ) :
    stageShare stg w j ≤ stageShare stg w' j :=
  Finset.sum_le_sum fun v _ => by
    split_ifs
    · exact h v
    · exact le_rfl

/-! ## The period's budget -/

/-- **The live claims give the stages' luck**: `tileLuck` folded and `tileLuck0` whole on a live tile, as
`luckAt_of_perTile` gives them, and `1` on a dead one. -/
theorem luckAt_of_perLiveTile {ι K C : Type} [Fintype C] (Sc : Scheme K) (Cd : Declared ι) {Np : ℕ}
    (hLP : Sc.LuckProved) (live : Fin Np → Bool) : ∀ {M : ℕ} (W : SplitStages C (Fin Np) M),
    PerLiveTile Sc Cd live W (fun k G _ => OperandHMu (Sc.mu k) G) →
    PerLiveTile Sc Cd live W (fun k G _ => ReceiptLuckAt (Sc.rB k) G) →
    PerLiveTile Sc Cd live W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0) →
    W.LuckAt (fun t => if live t then tileLuck Sc Cd Np t else 1)
      (fun t => if live t then tileLuck0 Sc Cd Np t else 1) (tileStage Cd Np)
  | _, .done, _, _, _ => trivial
  | _, .stage mode right same next, h1, h2, h3 => by
    obtain ⟨h1, h1n⟩ := h1
    obtain ⟨h2, h2n⟩ := h2
    obtain ⟨h3, h3n⟩ := h3
    refine ⟨fun t ht => ?_, fun c => luckAt_of_perLiveTile Sc Cd hLP live (next c) (h1n c) (h2n c) (h3n c)⟩
    have hone : ∀ s : Mode, prob (fun c => mode c t = s ∧ same c t) ≤ 1 * prob (fun c => mode c t = s) :=
      fun s => by
        rw [one_mul]
        exact prob_mono fun c hc => hc.1
    show prob (fun c => mode c t = .folded ∧ same c t) ≤
        (if live t then tileLuck Sc Cd Np t else 1) * prob (fun c => mode c t = .folded) ∧
      prob (fun c => mode c t = .whole ∧ same c t) ≤
        (if live t then tileLuck0 Sc Cd Np t else 1) * prob (fun c => mode c t = .whole)
    by_cases hl : live t = true
    · rw [ite_eq_left hl, ite_eq_left hl]
      rcases hk : tileKind Cd Np t with _ | k
      · have e1 : tileLuck Sc Cd Np t = 1 := by simp [tileLuck, hk, Option.elim]
        have e0 : tileLuck0 Sc Cd Np t = 1 := by simp [tileLuck0, hk, Option.elim]
        rw [e1, e0]
        exact ⟨hone _, hone _⟩
      · cases hc : Sc.credited k with
        | false =>
          have e1 : tileLuck Sc Cd Np t = 1 := by simp [tileLuck, hk, hc, Option.elim]
          have e0 : tileLuck0 Sc Cd Np t = 1 := by simp [tileLuck0, hk, hc, Option.elim]
          rw [e1, e0]
          exact ⟨hone _, hone _⟩
        | true =>
          have e1 : tileLuck Sc Cd Np t = Sc.foldLuck k := by simp [tileLuck, hk, hc, Option.elim]
          have e0 : tileLuck0 Sc Cd Np t = Sc.guess k := by simp [tileLuck0, hk, hc, Option.elim]
          rw [e1, e0]
          have hPin : (tileGame mode right same .folded t).Pinned := ⟨fun _ h => h, fun _ => trivial⟩
          have hL := Accounting.SecurityProofs.TS.LuckOfReceipt (tileGame mode right same .folded t)
            (Sc.mu k) (Sc.rB k) hPin
            (Accounting.SecurityProofs.TS.Headline.operandLuck_of_pointMass (Sc.mu k) _ (h1 t hl ht k hk hc))
            (h2 t hl ht k hk hc) (hLP k).2.2.1
          exact ⟨Accounting.SecurityProofs.Compose.WorkProof.tileLuck_of mode right same .folded t
              (Sc.foldLuck k) hL,
            Accounting.SecurityProofs.Compose.WorkProof.tileLuck_of mode right same .whole t (Sc.guess k)
              (h3 t hl ht k hk hc)⟩
    · rw [ite_eq_right hl, ite_eq_right hl]
      exact ⟨hone _, hone _⟩

/-- **One window's budget at its live weights**: `ServiceBudget` at `liveShare`, whose premises `CircuitWork`'s at
`shareCone` give. -/
theorem budget_window {ι K C Ξ : Type} [Fintype C] [Fintype Ξ] (Sc : Scheme K) (Cd : Declared ι) {Np Kd : ℕ}
    (hLP : Sc.LuckProved) (live : Fin Np → Bool) (W : SplitStages C (Fin Np) Cd.rounds) {ρ p p0 a b wmax Λ : ℝ}
    (hρ : 0 < ρ) (hρN : ∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1)
    (hwmax : ∀ j, stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j ≤ wmax)
    (hlk : ∀ t : Fin Np, 0 < shareCone Sc Cd Np t → tileLuck Sc Cd Np t ≤ p ∧ tileLuck0 Sc Cd Np t ≤ p0)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hSR : SplitRate Kd ρ p p0 a b wmax Λ)
    (hMu : PerLiveTile Sc Cd live W (fun k G _ => OperandHMu (Sc.mu k) G))
    (hRB : PerLiveTile Sc Cd live W (fun k G _ => ReceiptLuckAt (Sc.rB k) G))
    (hGs : PerLiveTile Sc Cd live W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0)) :
    RoundGame.BudgetAt Λ (serviceGame (Ξ := Ξ) (tileStage Cd Np) W (liveShare Sc Cd Np live) a b Kd) := by
  have hρN' : ∀ t : Fin Np, ρ * Np * liveShare Sc Cd Np live t ≤ 1 := fun t =>
    (mul_le_mul_of_nonneg_left (liveShare_le_shareCone Sc Cd Np live t)
      (mul_nonneg hρ.le (Nat.cast_nonneg _))).trans (hρN t)
  have hwmax' : ∀ j, stageShare (tileStage Cd Np) (liveShare Sc Cd Np live) j ≤ wmax := fun j =>
    (stageShare_mono _ (liveShare_le_shareCone Sc Cd Np live) j).trans (hwmax j)
  have hlk' : ∀ t : Fin Np, 0 < liveShare Sc Cd Np live t →
      (if live t then tileLuck Sc Cd Np t else 1) ≤ p ∧ (if live t then tileLuck0 Sc Cd Np t else 1) ≤ p0 := by
    intro t ht
    by_cases hl : live t = true
    · rw [ite_eq_left hl, ite_eq_left hl]
      exact hlk t (lt_of_lt_of_le ht (liveShare_le_shareCone Sc Cd Np live t))
    · rw [liveShare_of_dead Sc Cd Np live hl] at ht
      exact absurd ht (lt_irrefl 0)
  exact Accounting.SecurityProofs.Compose.Budget.serviceBudget (tileStage Cd Np) W (liveShare Sc Cd Np live)
    (fun t => if live t then tileLuck Sc Cd Np t else 1) (fun t => if live t then tileLuck0 Sc Cd Np t else 1) Kd
    ρ p p0 a b wmax Λ (liveShare_nonneg Sc Cd Np live) hρ hρN' hwmax' hlk'
    (luckAt_of_perLiveTile Sc Cd hLP live W hMu hRB hGs) ha hb hSR

/-- **The period's budget at its windows' live weights**: `budget_window` in every window, then `PeriodBudget`'s
induction. -/
theorem budget_period {ι K C Ξ : Type} [Fintype C] [Fintype Ξ] (Sc : Scheme K) (Cd : Declared ι) {n Np Kd : ℕ}
    (hLP : Sc.LuckProved) {ρ p p0 a b wmax Λ : ℝ}
    (hρ : 0 < ρ) (hρN : ∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1)
    (hwmax : ∀ j, stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j ≤ wmax)
    (hlk : ∀ t : Fin Np, 0 < shareCone Sc Cd Np t → tileLuck Sc Cd Np t ≤ p ∧ tileLuck0 Sc Cd Np t ≤ p0)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hSR : SplitRate Kd ρ p p0 a b wmax Λ)
    (Wp : Period C (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) (Fin Np) Cd.rounds n)
    (hMu : Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k G _ => OperandHMu (Sc.mu k) G)))
    (hRB : Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k G _ => ReceiptLuckAt (Sc.rB k) G)))
    (hGs : Wp.EveryWindow (fun live W => PerLiveTile Sc Cd live W (fun k _ G0 => GuessLuckAt (Sc.guess k) G0))) :
    RoundGame.BudgetAt Λ (periodGame (Ξ := Ξ) (tileStage Cd Np) (liveShare Sc Cd Np) a b Kd Wp) := by
  induction Wp with
  | done => simp only [periodGame, RoundGame.BudgetAt]
  | window live W next ih =>
    simp only [Period.EveryWindow] at hMu hRB hGs
    simp only [periodGame]
    exact budgetAt_consGame _ _
      (budget_window Sc Cd hLP live W hρ hρN hwmax hlk ha hb hSR hMu.1 hRB.1 hGs.1)
      fun ω => ih ω (hMu.2 ω) (hRB.2 ω) (hGs.2 ω)

/-! ## The period's game, window by window -/

/-- `P` of every window gives `P` of the window played at each window along every path. -/
theorem everyWindow_at {C Ω V : Type} {M : ℕ} {P : (V → Bool) → SplitStages C V M → Prop} :
    ∀ {n : ℕ} (Wp : Period C Ω V M n), Wp.EveryWindow P → ∀ x w, P (Wp.liveAt x w) (Wp.windowAt x w)
  | _, .done, _, _, w => w.elim0
  | _, .window _ _ next, hP, x, w => by
    refine Fin.cases ?_ (fun w' => ?_) w
    · exact hP.1
    · exact everyWindow_at (next x.1) (hP.2 x.1) x.2 w'

/-- **Window `w`'s modes** along the period's coins: the run of the window played there, at its rounds' coins. -/
def windowMode {C B : Type} {Np M n : ℕ} (stg : Fin Np → ℕ) (Wp : Period C (Paths C M × B) (Fin Np) M n)
    (x : Paths (Paths C M × B) n) (w : Fin n) (u : Fin Np) : Mode :=
  (Wp.windowAt x w).modeAt stg (coinAt x w).1 u

/-- **The period game's checks don't read its weights or costs**: its first `j` rounds pass at one set of weights
and costs when they pass at another. -/
theorem acceptsUpTo_periodGame {C Ξ : Type} {N M : ℕ} (stg : Fin N → ℕ) (wt wt' : (Fin N → Bool) → Fin N → ℝ)
    (a b a' b' : ℝ) (K : ℕ) : ∀ {n : ℕ} (Wp : Period C (Paths C M × (Equiv.Perm (Fin N) × Ξ)) (Fin N) M n) (j : ℕ)
    (x : Paths (Paths C M × (Equiv.Perm (Fin N) × Ξ)) n),
    RoundGame.acceptsUpTo (periodGame stg wt a b K Wp) j x → RoundGame.acceptsUpTo (periodGame stg wt' a' b' K Wp) j x
  | _, .done, _, _, _ => trivial
  | _, .window _ _ _, 0, _, _ => trivial
  | _, .window _ _ next, j + 1, x, h =>
    ⟨h.1, acceptsUpTo_periodGame stg wt wt' a b a' b' K (next x.1) j x.2 h.2⟩

open Classical in
/-- **The period game's first `j` rounds' shares**: window `w`'s round's, `a` per unit of folded and `b` per unit of
whole work of its run at its live weights, summed over the windows before `j`. -/
theorem deficiencyUpTo_periodGame {C Ξ : Type} {N M : ℕ} (stg : Fin N → ℕ) (wt : (Fin N → Bool) → Fin N → ℝ)
    (a b : ℝ) (K : ℕ) : ∀ {n : ℕ} (Wp : Period C (Paths C M × (Equiv.Perm (Fin N) × Ξ)) (Fin N) M n) (j : ℕ)
    (x : Paths (Paths C M × (Equiv.Perm (Fin N) × Ξ)) n),
    RoundGame.deficiencyUpTo (periodGame stg wt a b K Wp) j x =
      ∑ w : Fin n, if w.val < j then
        a * (∑ u, if windowMode stg Wp x w u = .folded then wt (Wp.liveAt x w) u else 0) +
          b * (∑ u, if windowMode stg Wp x w u = .whole then wt (Wp.liveAt x w) u else 0)
      else 0
  | _, .done, _, _ => by
    rw [Fin.sum_univ_zero]
    rfl
  | _, .window _ _ _, 0, _ => (Finset.sum_eq_zero fun _ _ => ite_eq_right (Nat.not_lt_zero _)).symm
  | _, .window live W next, j + 1, x => by
    rw [Fin.sum_univ_succ, Fin.val_zero, ite_eq_left (by omega : 0 < j + 1)]
    show (a * JointRun.share (W.joint stg) (wt live) x.1.1 .folded +
        b * JointRun.share (W.joint stg) (wt live) x.1.1 .whole) +
      RoundGame.deficiencyUpTo (periodGame stg wt a b K (next x.1)) j x.2 = _
    rw [deficiencyUpTo_periodGame stg wt a b K (next x.1) j x.2]
    refine congrArg₂ (· + ·) rfl (Finset.sum_congr rfl fun w' _ => ?_)
    by_cases hw : w'.val < j
    · rw [ite_eq_left hw, ite_eq_left (show (w'.succ : ℕ) < j + 1 by rw [Fin.val_succ]; omega)]
      rfl
    · rw [ite_eq_right hw, ite_eq_right (show ¬(w'.succ : ℕ) < j + 1 by rw [Fin.val_succ]; omega)]

/-! ## The accepted prefix -/

theorem credited_mono {C B V : Type} {m s Np n : ℕ} (pr : PeriodRun C B m s Np n V) {x : Paths (Paths C m × B) n}
    {w w' : Fin n} (h : pr.Credited x w) (hw : w' ≤ w) : pr.Credited x w' :=
  fun w'' h'' i => h w'' (h''.trans hw) i

/-- **The credited windows are the first `prefixLen`**: being credited is closed downward. -/
theorem credited_iff {C B V : Type} {m s Np n : ℕ} (pr : PeriodRun C B m s Np n V) (x : Paths (Paths C m × B) n)
    (w : Fin n) : pr.Credited x w ↔ w.val < pr.prefixLen x := by
  classical
  unfold PeriodRun.prefixLen
  constructor
  · intro h
    have hsub : Finset.Iic w ⊆ Finset.univ.filter fun w : Fin n => pr.Credited x w := fun w' hw' =>
      Finset.mem_filter.mpr ⟨Finset.mem_univ _, credited_mono pr h (Finset.mem_Iic.mp hw')⟩
    have hc := Finset.card_le_card hsub
    rw [Fin.card_Iic] at hc
    omega
  · intro h
    by_contra hc
    have hsub : (Finset.univ.filter fun w : Fin n => pr.Credited x w) ⊆ Finset.Iio w := fun w' hw' =>
      Finset.mem_Iio.mpr (not_le.mp fun hle => hc (credited_mono pr (Finset.mem_filter.mp hw').2 hle))
    have hc' := Finset.card_le_card hsub
    rw [Fin.card_Iio] at hc'
    omega

/-! ## One window's words -/

section Window

variable {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) {Np : ℕ} (live : Fin Np → Bool) (md : Fin Np → Mode)

/-- A window's folded share at its live weights. -/
noncomputable def foldS : ℝ := ∑ u, if md u = .folded then liveShare Sc Cd Np live u else 0

/-- A window's whole share at its live weights. -/
noncomputable def wholeS : ℝ := ∑ u, if md u = .whole then liveShare Sc Cd Np live u else 0

/-- The cones of a window's skipped tiles of no live words. -/
noncomputable def zeroS : ℝ :=
  ∑ u, if md u = .paid then 0 else (if liveWords Sc Cd Np live u = 0 then liveCone Sc Cd Np live u else 0)

theorem foldS_nonneg : 0 ≤ foldS Sc Cd live md :=
  Finset.sum_nonneg fun u _ => by
    split_ifs
    · exact liveShare_nonneg Sc Cd Np live u
    · exact le_rfl

theorem wholeS_nonneg : 0 ≤ wholeS Sc Cd live md :=
  Finset.sum_nonneg fun u _ => by
    split_ifs
    · exact liveShare_nonneg Sc Cd Np live u
    · exact le_rfl

theorem zeroS_nonneg : 0 ≤ zeroS Sc Cd live md :=
  Finset.sum_nonneg fun u _ => by
    split_ifs
    · exact le_rfl
    · exact liveCone_nonneg Sc Cd Np live u
    · exact le_rfl

/-- **The unpaid cones are the shares and the cones of tiles of no live words.** -/
theorem unpaid_split : (∑ u, if md u = .paid then 0 else liveCone Sc Cd Np live u) =
    foldS Sc Cd live md + wholeS Sc Cd live md + zeroS Sc Cd live md := by
  unfold foldS wholeS zeroS
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun u _ => ?_
  unfold liveShare
  by_cases h0 : liveWords Sc Cd Np live u = 0 <;> cases md u <;> simp [h0]

/-- **The cones of tiles of no live words are at most `D₀`**: such a tile has no words, or it is dead and its cone
holds no live words (`DeadConeDead`). -/
theorem zeroS_le (hDD : DeadConeDead Sc Cd Np live) : zeroS Sc Cd live md ≤ zeroCone Sc Cd Np := by
  unfold zeroS zeroCone
  refine Finset.sum_le_sum fun u _ => ?_
  have hz : 0 ≤ (if tileWords Sc Cd Np u = 0 then tileCone Sc Cd Np u else 0) := by
    split_ifs
    · exact Accounting.SecurityProofs.Compose.WorkProof.tileCone_nonneg Sc Cd Np u
    · exact le_rfl
  by_cases hp : md u = .paid
  · rw [ite_eq_left hp]
    exact hz
  rw [ite_eq_right hp]
  by_cases h0 : liveWords Sc Cd Np live u = 0
  · rw [ite_eq_left h0]
    by_cases hw : tileWords Sc Cd Np u = 0
    · rw [ite_eq_left hw]
      exact liveCone_le_tileCone Sc Cd Np live u
    rw [ite_eq_right hw]
    have hd : live u = false := by
      cases hlu : live u
      · rfl
      · exact absurd (by rwa [liveWords_of_live Sc Cd Np live hlu] at h0) hw
    refine le_of_eq (Finset.sum_eq_zero fun t' _ => ?_)
    split_ifs with hR
    · rw [hDD u t' hd hR, Nat.cast_zero, zero_div]
    · rfl
  · rw [ite_eq_right h0]
    exact hz

/-- **A credited window's fixed words**: its live words are its good tiles' words and the ones that aren't good, which
the covers bound. -/
theorem window_fixed (pm : Fin Np → Mode) (good : Fin Np → Prop) [DecidablePred good]
    (hgood : ∀ t, good t → pm t = .paid) (hpm : ∀ t, pm t = if live t then md t else .whole)
    (hcov : (∑ t, if good t then (0 : ℝ) else (liveWords Sc Cd Np live t : ℝ)) ≤
      (∑ u, if md u = .paid then 0 else liveCone Sc Cd Np live u) * (wordsOf Sc Cd : ℝ)) :
    (∑ t, (liveWords Sc Cd Np live t : ℝ)) ≤ (∑ t, if good t then (tileWords Sc Cd Np t : ℝ) else 0) +
      (∑ u, if md u = .paid then 0 else liveCone Sc Cd Np live u) * (wordsOf Sc Cd : ℝ) := by
  have hsplit : (∑ t, if good t then (tileWords Sc Cd Np t : ℝ) else 0) +
      (∑ t, if good t then (0 : ℝ) else (liveWords Sc Cd Np live t : ℝ)) =
        ∑ t, (liveWords Sc Cd Np live t : ℝ) := by
    rw [← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun t _ => ?_
    by_cases hg : good t
    · have hl : live t = true := by
        by_contra hl
        have h := hgood t hg
        rw [hpm t, ite_eq_right hl] at h
        cases h
      rw [ite_eq_left hg, ite_eq_left hg, add_zero, liveWords_of_live Sc Cd Np live hl]
    · rw [ite_eq_right hg, ite_eq_right hg, zero_add]
  linarith

/-- **A credited window's folded words**: its live words skipped whole are at most `N_w` times its whole share, and
the rest are `periodMode`'s not-whole tiles' words. -/
theorem window_fold (pm : Fin Np → Mode) (hpm : ∀ t, pm t = if live t then md t else .whole) :
    (∑ t, (liveWords Sc Cd Np live t : ℝ)) ≤ (∑ t, if pm t = .whole then (0 : ℝ) else (tileWords Sc Cd Np t : ℝ)) +
      wholeS Sc Cd live md * (wordsOf Sc Cd : ℝ) := by
  unfold wholeS
  rw [Finset.sum_mul, ← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun t _ => ?_
  have hW := liveWords_le_liveShare Sc Cd Np live t
  by_cases hl : live t = true
  · have hp : pm t = md t := by rw [hpm t, ite_eq_left hl]
    rw [hp]
    by_cases hw : md t = .whole
    · rw [ite_eq_left hw, ite_eq_left hw, zero_add]
      exact hW
    · rw [ite_eq_right hw, ite_eq_right hw, zero_mul, add_zero]
      exact (congrArg (fun k : ℕ => (k : ℝ)) (liveWords_of_live Sc Cd Np live hl)).le
  · have hp : pm t = .whole := by rw [hpm t, ite_eq_right hl]
    rw [hp, ite_eq_left (rfl : Mode.whole = Mode.whole), zero_add, liveWords_of_dead Sc Cd Np live hl,
      Nat.cast_zero]
    split_ifs
    · exact mul_nonneg (liveShare_nonneg Sc Cd Np live t) (Nat.cast_nonneg _)
    · rw [zero_mul]

end Window

/-- **The shares in the cheaper of the two**: `d_F = S_F + Z`, `d₀ = S_0` for `a ≤ b`, else `d_F = S_F`,
`d₀ = S_0 + Z`, the cones `Z` of tiles of no live words at most `j·D₀` over a prefix of `j` windows. -/
theorem liveWork_of {Nlive Nw a b Λl D0 fixed fd Gt NWt SF S0 Z j : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hNw : 0 ≤ Nw)
    (hSF : 0 ≤ SF) (hS0 : 0 ≤ S0) (hZ : 0 ≤ Z) (hZD : Z ≤ j * D0)
    (hfixL : Nlive ≤ Gt + SF * Nw + S0 * Nw + Z * Nw) (hfoldL : Nlive ≤ NWt + S0 * Nw)
    (hG : Gt ≤ fixed) (hNW : NWt ≤ fd) (hlt : a * SF + b * S0 < Λl) :
    LiveWorkAt Nlive Nw a b (Λl + j * (min a b * D0)) fixed fd := by
  have hZa := mul_le_mul_of_nonneg_left hZD ha
  have hZb := mul_le_mul_of_nonneg_left hZD hb
  have hZN := mul_nonneg hZ hNw
  rcases le_total a b with hab | hab
  · refine ⟨SF + Z, S0, by linarith, hS0, ?_, by linarith, ?_⟩
    · have e : (SF + Z + S0) * Nw = SF * Nw + S0 * Nw + Z * Nw := by ring
      linarith
    · rw [min_eq_left hab]
      have e : j * (a * D0) = a * (j * D0) := by ring
      linarith
  · refine ⟨SF, S0 + Z, hSF, by linarith, ?_, ?_, ?_⟩
    · have e : (SF + (S0 + Z)) * Nw = SF * Nw + S0 * Nw + Z * Nw := by ring
      linarith
    · have e : (S0 + Z) * Nw = S0 * Nw + Z * Nw := by ring
      linarith
    · rw [min_eq_right hab]
      have e : j * (b * D0) = b * (j * D0) := by ring
      linarith

/-! ## The period's work -/

open Classical in
/-- **One accepted run's work over the period**, off both events: per credited window, the covers, `UnitFixed`'s
sums and the zero-word cones, summed over the accepted prefix, whose budget game's shares are below `Λl`. -/
theorem work_step {ι K C Ξ V : Type} (Sc : Scheme K) (Cd : Declared ι) {n Np Kd : ℕ} (hn : 0 < n)
    (hr : 0 < Cd.rounds) (hK : Kd ≤ Np)
    (Wp : Period C (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) (Fin Np) Cd.rounds n)
    (pr : PeriodRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np n V)
    (x : Paths (Paths C Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) n) {a b Λl fixed fd : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hDD : ∀ w, DeadConeDead Sc Cd Np (Wp.liveAt x w))
    (hcov : ∀ w, pr.Credited x w →
      (∑ t : Fin Np, if (Cd.periodLayer hr hK pr).Good (periodMode Cd Wp pr) x w t then (0 : ℝ)
        else (liveWords Sc Cd Np (Wp.liveAt x w) t : ℝ)) ≤
        (∑ u : Fin Np, if windowMode (tileStage Cd Np) Wp x w u = .paid then 0
          else liveCone Sc Cd Np (Wp.liveAt x w) u) * (wordsOf Sc Cd : ℝ))
    (hfix : (∑ w : Fin n, ∑ t : Fin Np, if (Cd.periodLayer hr hK pr).Good (periodMode Cd Wp pr) x w t
      then (tileWords Sc Cd Np t : ℝ) else 0) / n ≤ fixed)
    (hfold : (∑ w : Fin n, ∑ t : Fin Np, if periodMode Cd Wp pr x (w, t) = .whole then (0 : ℝ)
      else (tileWords Sc Cd Np t : ℝ)) / n ≤ fd)
    (hlt : RoundGame.deficiencyUpTo (periodGame (Ξ := Ξ) (tileStage Cd Np) (liveShare Sc Cd Np) a b Kd Wp)
      (pr.prefixLen x) x < Λl) :
    LiveWorkAt (creditedLive Sc Cd Wp pr x) (wordsOf Sc Cd : ℝ) a b
      (Λl + (pr.prefixLen x : ℝ) * (min a b * zeroCone Sc Cd Np)) (n * fixed) (n * fd) := by
  have hn' : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hgood : ∀ w t, (Cd.periodLayer hr hK pr).Good (periodMode Cd Wp pr) x w t →
      periodMode Cd Wp pr x (w, t) = .paid := fun w t hg => hg (w, t) Relation.ReflTransGen.refl
  have hpm : ∀ w, pr.Credited x w → ∀ t, periodMode Cd Wp pr x (w, t) =
      if Wp.liveAt x w t then windowMode (tileStage Cd Np) Wp x w t else .whole := by
    intro w hc t
    show (if pr.Credited x w ∧ Wp.liveAt x w t = true then
      ((Wp.windowAt x w).joint (tileStage Cd Np)).mode (coinAt x w).1 t else .whole) = _
    by_cases hl : Wp.liveAt x w t = true
    · rw [ite_eq_left (And.intro hc hl), ite_eq_left hl]
      rfl
    · rw [ite_eq_right (fun h : pr.Credited x w ∧ Wp.liveAt x w t = true => hl h.2), ite_eq_right hl]
  have hdead : ∀ w, ¬pr.Credited x w → ∀ t, periodMode Cd Wp pr x (w, t) = .whole := by
    intro w hc t
    show (if pr.Credited x w ∧ Wp.liveAt x w t = true then
      ((Wp.windowAt x w).joint (tileStage Cd Np)).mode (coinAt x w).1 t else .whole) = _
    rw [ite_eq_right (fun h : pr.Credited x w ∧ Wp.liveAt x w t = true => hc h.1)]
  have hGnn : ∀ w, 0 ≤ ∑ t : Fin Np, if (Cd.periodLayer hr hK pr).Good (periodMode Cd Wp pr) x w t
      then (tileWords Sc Cd Np t : ℝ) else 0 := fun w =>
    Finset.sum_nonneg fun t _ => by split_ifs <;> positivity
  have hNWnn : ∀ w, 0 ≤ ∑ t : Fin Np, if periodMode Cd Wp pr x (w, t) = .whole then (0 : ℝ)
      else (tileWords Sc Cd Np t : ℝ) := fun w =>
    Finset.sum_nonneg fun t _ => by split_ifs <;> positivity
  have hA : ∀ w : Fin n, (if pr.Credited x w then ∑ t : Fin Np, (liveWords Sc Cd Np (Wp.liveAt x w) t : ℝ) else 0) ≤
      (∑ t : Fin Np, if (Cd.periodLayer hr hK pr).Good (periodMode Cd Wp pr) x w t
        then (tileWords Sc Cd Np t : ℝ) else 0) +
      (if pr.Credited x w then foldS Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w) else 0) *
        (wordsOf Sc Cd : ℝ) +
      (if pr.Credited x w then wholeS Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w) else 0) *
        (wordsOf Sc Cd : ℝ) +
      (if pr.Credited x w then zeroS Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w) else 0) *
        (wordsOf Sc Cd : ℝ) := by
    intro w
    by_cases hc : pr.Credited x w
    · simp only [hc, ↓reduceIte]
      have h := window_fixed Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w)
        (fun t => periodMode Cd Wp pr x (w, t)) (fun t => (Cd.periodLayer hr hK pr).Good (periodMode Cd Wp pr) x w t)
        (hgood w) (hpm w hc) (hcov w hc)
      rw [unpaid_split Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w)] at h
      have e : (foldS Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w) +
          wholeS Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w) +
          zeroS Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w)) * (wordsOf Sc Cd : ℝ) =
        foldS Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w) * (wordsOf Sc Cd : ℝ) +
          wholeS Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w) * (wordsOf Sc Cd : ℝ) +
          zeroS Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w) * (wordsOf Sc Cd : ℝ) := by ring
      linarith
    · simp only [hc, ↓reduceIte, zero_mul, add_zero]
      exact hGnn w
  have hB : ∀ w : Fin n, (if pr.Credited x w then ∑ t : Fin Np, (liveWords Sc Cd Np (Wp.liveAt x w) t : ℝ) else 0) ≤
      (∑ t : Fin Np, if periodMode Cd Wp pr x (w, t) = .whole then (0 : ℝ) else (tileWords Sc Cd Np t : ℝ)) +
      (if pr.Credited x w then wholeS Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w) else 0) *
        (wordsOf Sc Cd : ℝ) := by
    intro w
    by_cases hc : pr.Credited x w
    · simp only [hc, ↓reduceIte]
      exact window_fold Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w)
        (fun t => periodMode Cd Wp pr x (w, t)) (hpm w hc)
    · simp only [hc, ↓reduceIte, zero_mul, add_zero]
      exact hNWnn w
  have hfixL : creditedLive Sc Cd Wp pr x ≤
      (∑ w : Fin n, ∑ t : Fin Np, if (Cd.periodLayer hr hK pr).Good (periodMode Cd Wp pr) x w t
        then (tileWords Sc Cd Np t : ℝ) else 0) +
      (∑ w : Fin n, if pr.Credited x w then foldS Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w)
        else 0) * (wordsOf Sc Cd : ℝ) +
      (∑ w : Fin n, if pr.Credited x w then wholeS Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w)
        else 0) * (wordsOf Sc Cd : ℝ) +
      (∑ w : Fin n, if pr.Credited x w then zeroS Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w)
        else 0) * (wordsOf Sc Cd : ℝ) := by
    unfold creditedLive
    simp only [Finset.sum_mul, ← Finset.sum_add_distrib]
    exact Finset.sum_le_sum fun w _ => hA w
  have hfoldL : creditedLive Sc Cd Wp pr x ≤
      (∑ w : Fin n, ∑ t : Fin Np, if periodMode Cd Wp pr x (w, t) = .whole then (0 : ℝ)
        else (tileWords Sc Cd Np t : ℝ)) +
      (∑ w : Fin n, if pr.Credited x w then wholeS Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w)
        else 0) * (wordsOf Sc Cd : ℝ) := by
    unfold creditedLive
    simp only [Finset.sum_mul, ← Finset.sum_add_distrib]
    exact Finset.sum_le_sum fun w _ => hB w
  have hdef : RoundGame.deficiencyUpTo (periodGame (Ξ := Ξ) (tileStage Cd Np) (liveShare Sc Cd Np) a b Kd Wp)
      (pr.prefixLen x) x =
      a * (∑ w : Fin n, if pr.Credited x w then foldS Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w)
        else 0) +
      b * (∑ w : Fin n, if pr.Credited x w then wholeS Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w)
        else 0) := by
    rw [deficiencyUpTo_periodGame, Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun w _ => ?_
    by_cases hc : pr.Credited x w
    · rw [ite_eq_left ((credited_iff pr x w).mp hc), ite_eq_left hc, ite_eq_left hc]
      rfl
    · rw [ite_eq_right (fun h => hc ((credited_iff pr x w).mpr h)), ite_eq_right hc, ite_eq_right hc, mul_zero,
        mul_zero, add_zero]
  have hZD : (∑ w : Fin n, if pr.Credited x w then zeroS Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w)
      else 0) ≤ (pr.prefixLen x : ℝ) * zeroCone Sc Cd Np := by
    calc _ ≤ ∑ w : Fin n, if pr.Credited x w then zeroCone Sc Cd Np else 0 :=
          Finset.sum_le_sum fun w _ => by
            split_ifs
            · exact zeroS_le Sc Cd _ _ (hDD w)
            · exact le_rfl
      _ = (pr.prefixLen x : ℝ) * zeroCone Sc Cd Np := by
          rw [← Finset.sum_filter, Finset.sum_const, nsmul_eq_mul]
          rfl
  have hnnS : ∀ f : Fin n → ℝ, (∀ w, 0 ≤ f w) → 0 ≤ ∑ w : Fin n, if pr.Credited x w then f w else 0 :=
    fun f hf => Finset.sum_nonneg fun w _ => by
      split_ifs
      · exact hf w
      · exact le_rfl
  have hSF : 0 ≤ ∑ w : Fin n, if pr.Credited x w then
      foldS Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w) else 0 :=
    hnnS _ fun w => foldS_nonneg Sc Cd _ _
  have hS0 : 0 ≤ ∑ w : Fin n, if pr.Credited x w then
      wholeS Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w) else 0 :=
    hnnS _ fun w => wholeS_nonneg Sc Cd _ _
  have hZ : 0 ≤ ∑ w : Fin n, if pr.Credited x w then
      zeroS Sc Cd (Wp.liveAt x w) (windowMode (tileStage Cd Np) Wp x w) else 0 :=
    hnnS _ fun w => zeroS_nonneg Sc Cd _ _
  have hG := ((div_le_iff₀ hn').mp hfix).trans_eq (mul_comm _ _)
  have hNW := ((div_le_iff₀ hn').mp hfold).trans_eq (mul_comm _ _)
  rw [hdef] at hlt
  exact liveWork_of ha hb (Nat.cast_nonneg _) hSF hS0 hZ hZD hfixL hfoldL hG hNW hlt

/-! ## `CircuitWorkPeriod` -/

/-- **`CircuitWorkPeriod` from the period layer's covers**: the assembly, reading `PeriodCoversStmt` as a hypothesis
(`periodCovers` proves it). Off `DrawnRightOff`'s event and the period budget's tail at its accepted prefix,
`work_step`. -/
theorem circuitWorkPeriod_of_covers (hC : PeriodCoversStmt) :
    Accounting.Guarantees.ComposePeriod.CircuitWorkPeriod := by
  intro ι K T L U C Ξ V _ _ _ _ _ _ _ Sc Cd n Np Kd hn _ hr hK Wp Pl pr folded ρ p p0 a b wmax Λ δs δbind δks hS
    hρ hρN hwmax hlk ha hb hΛ hSR hG hDD hPre hMoved hJS hPins hMu hRB hGs hUF
  obtain ⟨hwf, -, hδs, hLP, hlc⟩ := hS
  obtain ⟨hWR, hCov⟩ := hC Sc Cd hr hK Wp pr hwf hlc
  obtain ⟨E, hE, hEoff⟩ := Accounting.SecurityProofs.TS.DrawnRightOff C (Equiv.Perm (Fin Np) × Ξ) n Cd.rounds Kd
    (Cd.periodLayer hr hK pr) δbind δks hMoved hJS hPins
  have hBud : RoundGame.BudgetAt Λ (periodGame (Ξ := Ξ) (tileStage Cd Np) (liveShare Sc Cd Np) a b Kd Wp) :=
    budget_period Sc Cd hLP hρ hρN hwmax hlk ha hb hSR Wp hMu hRB hGs
  have hTail := PeriodTailMax (periodGame (Ξ := Ξ) (tileStage Cd Np) (liveShare Sc Cd Np) a b Kd Wp) Λ hΛ hBud
    (Real.log (1 / δs))
  have hexp : Real.exp (-Real.log (1 / δs)) = δs := by
    rw [Real.exp_neg, Real.exp_log (one_div_pos.mpr hδs), one_div, inv_inv]
  rw [hexp] at hTail
  have hFix := hUF hWR
  have hDDx : ∀ x w, DeadConeDead Sc Cd Np (Wp.liveAt x w) := fun x w => everyWindow_at Wp hDD x w
  have hsub : ∀ x, (Pl.accepts x ∧
      ¬ LiveWorkAt (creditedLive Sc Cd Wp pr x) (wordsOf Sc Cd : ℝ) a b
        (rOfPrefix Sc Cd Np (pr.prefixLen x) a b Λ δs) (n * (Pl.run x).fixed) (n * folded x)) →
      E x ∨ ∃ j : ℕ, RoundGame.acceptsUpTo (periodGame (Ξ := Ξ) (tileStage Cd Np) (liveShare Sc Cd Np) a b Kd Wp) j x ∧
        Λ * Real.log (1 / δs) ≤
          RoundGame.deficiencyUpTo (periodGame (Ξ := Ξ) (tileStage Cd Np) (liveShare Sc Cd Np) a b Kd Wp) j x := by
    rintro x ⟨hacc, hnw⟩
    by_cases hEx : E x
    · exact Or.inl hEx
    refine Or.inr ⟨pr.prefixLen x, ?_, not_lt.mp fun hlt => hnw ?_⟩
    · have h1 := hPre x (pr.prefixLen x) fun w hw i =>
        hEoff x hEx w i (((credited_iff pr x w).mpr hw) w le_rfl i)
      rw [hG] at h1
      exact acceptsUpTo_periodGame _ _ _ _ _ _ _ _ Wp _ x h1
    · exact work_step Sc Cd hn hr hK Wp pr x ha hb (hDDx x) (fun w hc => hCov x w hc (hDDx x w))
        (hFix x hacc).1 (hFix x hacc).2 hlt
  refine (prob_mono hsub).trans ((prob_or_le _ _).trans ?_)
  linarith

/-! ## `CircuitCreditPeriod` -/

/-- **The credit is below the work**: under `LiveWorkAt` at `r(j)` with `a, b > 0`, the words not fixed right,
`(d_F + d₀)·N_w`, are at most `r(j)·N_w/min(a, b) = ε(j)·N_live`, so `(1 − ε(j))₊·N_live` is at most the words fixed
right when those aren't negative. -/
theorem creditPrefix_le {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np j : ℕ) {Nlive a b Λ δs F fd : ℝ}
    (ha : 0 < a) (hb : 0 < b) (hN : 0 ≤ Nlive) (hF : 0 ≤ F)
    (h : LiveWorkAt Nlive (wordsOf Sc Cd : ℝ) a b (rOfPrefix Sc Cd Np j a b Λ δs) F fd) :
    creditPrefix Sc Cd Np j Nlive a b Λ δs ≤ F := by
  obtain ⟨dF, d0, hdF, hd0, hfix, -, hlt⟩ := h
  have hm : 0 < min a b := lt_min ha hb
  have hNw : (0 : ℝ) ≤ wordsOf Sc Cd := Nat.cast_nonneg _
  have hsum : min a b * (dF + d0) < rOfPrefix Sc Cd Np j a b Λ δs := by
    nlinarith [mul_le_mul_of_nonneg_right (min_le_left a b) hdF, mul_le_mul_of_nonneg_right (min_le_right a b) hd0]
  have hdef : (dF + d0) * (wordsOf Sc Cd : ℝ) ≤ rOfPrefix Sc Cd Np j a b Λ δs * (wordsOf Sc Cd : ℝ) / min a b := by
    rw [le_div_iff₀ hm]
    nlinarith [mul_le_mul_of_nonneg_right hsum.le hNw]
  unfold creditPrefix
  rcases hN.lt_or_eq with hpos | hzero
  · have heps : epsPrefix Sc Cd Np j Nlive a b Λ δs * Nlive =
        rOfPrefix Sc Cd Np j a b Λ δs * (wordsOf Sc Cd : ℝ) / min a b := by
      unfold epsPrefix
      rw [div_mul_eq_mul_div, mul_div_mul_right _ _ hpos.ne']
    rcases le_total (1 - epsPrefix Sc Cd Np j Nlive a b Λ δs) 0 with hle | hle
    · rw [max_eq_right hle, zero_mul]
      exact hF
    · rw [max_eq_left hle, sub_mul, one_mul, heps]
      linarith
  · rw [← hzero, mul_zero]
    exact hF

/-- **`CircuitCreditPeriod` from the period layer's covers**: `circuitWorkPeriod_of_covers` at `a, b > 0`; on an
accepted run `UnitFixed`'s sums keep `fixed` nonnegative, so the credit is below the work (`creditPrefix_le`). -/
theorem circuitCreditPeriod_of_covers (hC : PeriodCoversStmt) :
    Accounting.Guarantees.ComposePeriod.CircuitCreditPeriod := by
  intro ι K T L U C Ξ V _ _ _ _ _ _ _ Sc Cd n Np Kd hn hN hr hK Wp Pl pr folded ρ p p0 a b wmax Λ δs δbind δks hS
    hρ hρN hwmax hlk ha hb hΛ hSR hG hDD hPre hMoved hJS hPins hMu hRB hGs hUF
  have hW := circuitWorkPeriod_of_covers hC Sc Cd hn hN hr hK Wp Pl pr folded ρ p p0 a b wmax Λ δs δbind δks hS
    hρ hρN hwmax hlk ha.le hb.le hΛ hSR hG hDD hPre hMoved hJS hPins hMu hRB hGs hUF
  obtain ⟨hwf, -, -, -, hlc⟩ := hS
  have hFix := hUF (hC Sc Cd hr hK Wp pr hwf hlc).1
  have hfix0 : ∀ x, Pl.accepts x → 0 ≤ (Pl.run x).fixed := by
    intro x hx
    refine le_trans ?_ (hFix x hx).1
    exact div_nonneg (Finset.sum_nonneg fun w _ => Finset.sum_nonneg fun t _ => by
      split_ifs
      exacts [Nat.cast_nonneg _, le_rfl]) (Nat.cast_nonneg _)
  have hsub : ∀ x, (Pl.accepts x ∧ n * (Pl.run x).fixed <
      creditPrefix Sc Cd Np (pr.prefixLen x) (creditedLive Sc Cd Wp pr x) a b Λ δs) →
      Pl.accepts x ∧ ¬ LiveWorkAt (creditedLive Sc Cd Wp pr x) (wordsOf Sc Cd : ℝ) a b
        (rOfPrefix Sc Cd Np (pr.prefixLen x) a b Λ δs) (n * (Pl.run x).fixed) (n * folded x) := by
    rintro x ⟨hacc, hlt⟩
    refine ⟨hacc, fun hw => (not_lt.mpr ?_) hlt⟩
    have hNl : 0 ≤ creditedLive Sc Cd Wp pr x := by
      unfold creditedLive
      exact Finset.sum_nonneg fun w _ => by
        split_ifs
        exacts [Finset.sum_nonneg fun t _ => Nat.cast_nonneg _, le_rfl]
    exact creditPrefix_le Sc Cd Np _ ha hb hNl (mul_nonneg (Nat.cast_nonneg _) (hfix0 x hacc)) hw
  exact (prob_mono hsub).trans hW

/-- `CircuitWorkPeriod`: the assembly at the period layer's covers (`periodCovers`). -/
theorem CircuitWorkPeriod : Accounting.Guarantees.ComposePeriod.CircuitWorkPeriod :=
  circuitWorkPeriod_of_covers periodCovers

/-- `CircuitCreditPeriod`: the assembly at the period layer's covers (`periodCovers`). -/
theorem CircuitCreditPeriod : Accounting.Guarantees.ComposePeriod.CircuitCreditPeriod :=
  circuitCreditPeriod_of_covers periodCovers

end Accounting.SecurityProofs.ComposePeriod
