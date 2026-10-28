import Properties.Pouw.ServedValues
import Proofs.Pouw.PeriodJ10R

/-!
# Proof of `RightValues`

`H1J10RCheckedCreditPeriod` at the play whose fixed words are the right blocks' words (`rightWords`) and whose folded
words are the blocks not skipped whole. There `UnitFixed` holds with equality, so H1 bounds the right blocks' words
below `(1 − ε(j))₊·N_live` with no premise on the play's anatomy. The step to size is deterministic: a right block is
live in a credited window (`good_live`, through `credited_iff`), so the size the right blocks miss of `S` is at most
`σ` times the words they miss of `N_live` (`creditedSize_sub_le`), and `(1 − ε(j))₊·N_live` right words give
`(1 − ε_skip)₊·S` right size (`creditFloor_le`).
-/

namespace Properties.Pouw.ServedValues.SecurityProofs

open Properties.Pouw.PeriodJ10R Properties.Pouw.PeriodJ10R.SecurityProofs Properties.Pouw.ServedValues
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity

/-- **The blocks' sizes sum to at most the credit** of the registered declaration under its pinned lowering: the
blocks are disjoint, so their AND gates outside the ledger are among `lower D`'s AND gates. -/
theorem sum_size_le_creditSize {ι : Type} {Cd : Declared ι} {Np : ℕ} {lower : Verity.RegisteredCircuit → Verity.Circuit}
    {D : Verity.RegisteredCircuit} (M : BlockMap Cd Np lower D) (ledger : Finset (Fin (lower D).N)) :
    ∑ t, M.size ledger t ≤ Verity.creditSize lower D := by
  classical
  unfold BlockMap.size Verity.creditSize Verity.Circuit.andSize
  rw [← Finset.card_biUnion]
  · exact Finset.card_le_card (Finset.biUnion_subset.mpr fun t _ =>
      Finset.filter_subset_filter _ (Finset.subset_univ _))
  · intro t _ t' _ h
    exact Finset.disjoint_filter_filter ((M.disjoint t t' h).mono Finset.sdiff_subset Finset.sdiff_subset)

open Classical in
/-- **The right blocks' words**: the credited words of every block `good` names in each window. -/
noncomputable def rightWords {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) {Np n : ℕ}
    (good : Fin n → Fin Np → Prop) : ℝ :=
  ∑ w : Fin n, ∑ t : Fin Np, if good w t then (tileWords Sc Cd Np t : ℝ) else 0

theorem sizePerWord_nonneg {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (sz : Fin Np → ℕ) :
    0 ≤ sizePerWord Sc Cd Np sz :=
  Real.iSup_nonneg fun t => by
    split_ifs
    exacts [le_rfl, div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)]

/-- **A block of credited words carries at most `σ` a word.** -/
theorem size_le_sizePerWord {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) {Np : ℕ} (sz : Fin Np → ℕ) (t : Fin Np)
    (h : tileWords Sc Cd Np t ≠ 0) : (sz t : ℝ) ≤ sizePerWord Sc Cd Np sz * tileWords Sc Cd Np t := by
  have hle : (if tileWords Sc Cd Np t = 0 then 0 else (sz t : ℝ) / (tileWords Sc Cd Np t : ℝ)) ≤
      sizePerWord Sc Cd Np sz :=
    le_ciSup (f := fun t => if tileWords Sc Cd Np t = 0 then (0 : ℝ) else (sz t : ℝ) / (tileWords Sc Cd Np t : ℝ))
      (Set.finite_range _).bddAbove t
  rw [if_neg h] at hle
  rwa [div_le_iff₀ (Nat.cast_pos.mpr (Nat.pos_of_ne_zero h))] at hle

theorem creditLive_nonneg {ι K C B : Type} (Sc : Scheme K) (Cd : Declared ι) {Np n : ℕ}
    (Wp : Period C (Paths C Cd.rounds × B) (Fin Np) Cd.rounds n) (ws : CreditWindows (Paths C Cd.rounds × B) n)
    (x : Paths (Paths C Cd.rounds × B) n) : 0 ≤ creditLive Sc Cd Wp ws x := by
  unfold creditLive
  refine Finset.sum_nonneg fun w _ => ?_
  split_ifs
  exacts [Finset.sum_nonneg fun t _ => Nat.cast_nonneg _, le_rfl]

theorem creditedSize_nonneg {ι K C B : Type} (Sc : Scheme K) (Cd : Declared ι) {Np n : ℕ}
    (Wp : Period C (Paths C Cd.rounds × B) (Fin Np) Cd.rounds n) (ws : CreditWindows (Paths C Cd.rounds × B) n)
    (sz : Fin Np → ℕ) (x : Paths (Paths C Cd.rounds × B) n) : 0 ≤ creditedSize Sc Cd Wp ws sz x := by
  unfold creditedSize
  refine Finset.sum_nonneg fun w _ => ?_
  split_ifs
  exacts [Finset.sum_nonneg fun t _ => Nat.cast_nonneg _, le_rfl]

theorem rightSize_nonneg {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) {Np n : ℕ} (sz : Fin Np → ℕ)
    (good : Fin n → Fin Np → Prop) : 0 ≤ rightSize Sc Cd sz good := by
  classical
  unfold rightSize
  refine Finset.sum_nonneg fun w _ => Finset.sum_nonneg fun t _ => ?_
  split_ifs
  exacts [Nat.cast_nonneg _, le_rfl]

/-- **The size the right blocks miss is at most `σ` times the words they miss**, when every right block is live in a
credited window. -/
theorem creditedSize_sub_le {ι K C B : Type} (Sc : Scheme K) (Cd : Declared ι) {Np n : ℕ}
    (Wp : Period C (Paths C Cd.rounds × B) (Fin Np) Cd.rounds n) (ws : CreditWindows (Paths C Cd.rounds × B) n)
    (sz : Fin Np → ℕ) (x : Paths (Paths C Cd.rounds × B) n) (good : Fin n → Fin Np → Prop)
    (hgood : ∀ w t, good w t → ws.AcceptedUpTo x w ∧ Wp.liveAt x w t = true) :
    creditedSize Sc Cd Wp ws sz x - rightSize Sc Cd sz good ≤
      sizePerWord Sc Cd Np sz * (creditLive Sc Cd Wp ws x - rightWords Sc Cd good) := by
  classical
  have hσ := sizePerWord_nonneg Sc Cd Np sz
  unfold creditedSize rightSize creditLive rightWords
  rw [← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib, Finset.mul_sum]
  refine Finset.sum_le_sum fun w _ => ?_
  by_cases hw : ws.AcceptedUpTo x w
  · rw [if_pos hw, if_pos hw, ← Finset.sum_sub_distrib, ← Finset.sum_sub_distrib, Finset.mul_sum]
    refine Finset.sum_le_sum fun t _ => ?_
    by_cases hg : good w t
    · have hl := (hgood w t hg).2
      have hlw : liveWords Sc Cd Np (Wp.liveAt x w) t = tileWords Sc Cd Np t := if_pos hl
      unfold liveSize
      rw [hlw, if_pos hg, sub_self, mul_zero]
      by_cases h0 : tileWords Sc Cd Np t = 0
      · rw [if_pos h0, if_neg (show ¬ (good w t ∧ tileWords Sc Cd Np t ≠ 0) from fun h => h.2 h0)]
        simp
      · rw [if_neg h0, if_pos (show good w t ∧ tileWords Sc Cd Np t ≠ 0 from ⟨hg, h0⟩)]
        simp
    · rw [if_neg (show ¬ (good w t ∧ tileWords Sc Cd Np t ≠ 0) from fun h => hg h.1), if_neg hg, sub_zero,
        sub_zero]
      unfold liveSize
      split_ifs with h0
      · rw [Nat.cast_zero]
        exact mul_nonneg hσ (Nat.cast_nonneg _)
      · unfold liveWords at h0 ⊢
        split_ifs at h0 ⊢ with hl
        · exact size_le_sizePerWord Sc Cd sz t h0
        · exact absurd rfl h0
  · rw [if_neg hw, if_neg hw]
    have h1 : ∀ t, ¬ good w t := fun t h => hw (hgood w t h).1
    simp [h1]

/-- **`(1 − ε)₊·N_live` words give `(1 − ε_skip)₊·S` size**, at `ε_skip = σ·ε·N_live/S`, when the size missed is at
most `σ` times the words missed. -/
theorem creditFloor_le {N S σ ε Gw Gs : ℝ} (hN : 0 ≤ N) (hS : 0 ≤ S) (hσ : 0 ≤ σ) (hGs : 0 ≤ Gs)
    (hsub : S - Gs ≤ σ * (N - Gw)) (hW : max (1 - ε) 0 * N ≤ Gw) :
    max (1 - σ * ε * N / S) 0 * S ≤ Gs := by
  rcases le_or_gt (1 - σ * ε * N / S) 0 with h | h
  · rw [max_eq_right h, zero_mul]
    exact hGs
  · rw [max_eq_left h.le]
    rcases hS.eq_or_lt with h0 | h0
    · rw [← h0, mul_zero]
      exact hGs
    · have hW' : (1 - ε) * N ≤ Gw := (mul_le_mul_of_nonneg_right (le_max_left _ _) hN).trans hW
      have hmiss : σ * (N - Gw) ≤ σ * (ε * N) := mul_le_mul_of_nonneg_left (by linarith) hσ
      have hS' : (1 - σ * ε * N / S) * S = S - σ * (ε * N) := by
        rw [sub_mul, one_mul, div_mul_cancel₀ _ h0.ne', mul_assoc]
      rw [hS']
      linarith

/-- **A good block is live in a credited window**: it is paid, and `periodMode` skips every other block whole. -/
theorem good_live {ι C Ξ V : Type} (Cd : Declared ι) {nB mB n : ℕ} (hr : 0 < Cd.rounds) (σ : Fin nB → Fin mB)
    (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB) (unitOf : Fin nB → ℕ) (hKd : 0 < k sT)
    (ws : CreditWindows (Coin C Ξ Cd.rounds σ sT) n)
    (opened : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → ℕ → ℕ → V)
    (rel right : Paths (Coin C Ξ Cd.rounds σ sT) n → Fin n → Fin (tileCount σ sT) → (ℕ → V) → Prop)
    (Wp : Period C (Coin C Ξ Cd.rounds σ sT) (Fin (tileCount σ sT)) Cd.rounds n)
    (x : Paths (Coin C Ξ Cd.rounds σ sT) n) (w : Fin n) (t : Fin (tileCount σ sT))
    (hg : (tileLayer Cd hr σ k hk sT unitOf ws opened rel right).Good
      (periodMode Cd Wp (tileRun σ k hk sT unitOf ws opened rel right)) x w t) :
    ws.AcceptedUpTo x w ∧ Wp.liveAt x w t = true := by
  have hp := hg (w, t) Relation.ReflTransGen.refl
  unfold periodMode at hp
  split_ifs at hp with hc
  exact ⟨(credited_iff σ k hk sT unitOf hKd ws opened rel right x w).1 hc.1, hc.2⟩

theorem rightValues : RightValues := by
  intro ι K T L U C Ξ V _ _ _ _ _ _ _ nB mB σ k hk sT unitOf Sc Cd n hn hN hr hM hKd D lower M ledger ws opened enc
    place tile Wp Pl ρ δs δbind δks hS hρ hρ1 hρN hwmax hlk hG hDD hPre hMoved hReg hDraw hServ hSalt hLucks
  classical
  let pf := tileLayer Cd hr σ k hk sT unitOf ws opened (pinnedAt (saltOf ws) place tile) (j10rAtCoin hr enc place tile)
  let mode := periodMode Cd Wp (tileRun σ k hk sT unitOf ws opened (pinnedAt (saltOf ws) place tile)
    (j10rAtCoin hr enc place tile))
  let Gw : Paths (Coin C Ξ Cd.rounds σ sT) n → ℝ := fun x =>
    rightWords Sc Cd (Np := tileCount σ sT) fun w t => pf.Good mode x w t
  let Pl' : Play (Coin C Ξ Cd.rounds σ sT) n K T L U :=
    { game := Pl.game, accepts := Pl.accepts, run := fun x => { Pl.run x with fixed := Gw x / n } }
  have hUF : UnitFixed Pl' pf mode (fun _ t => (tileWords Sc Cd (tileCount σ sT) t : ℝ))
      (fun x => (∑ w, ∑ j, if mode x (w, j) = .whole then 0 else (tileWords Sc Cd (tileCount σ sT) j : ℝ)) / n) :=
    fun _ _ _ => ⟨le_rfl, le_rfl⟩
  have hH1 := h1J10RCheckedCreditPeriod σ k hk sT unitOf Sc Cd hn hN hr hM hKd ws opened enc place tile Wp Pl' _
    ρ δs δbind δks hS hρ hρ1 hρN hwmax hlk hG hDD hPre hMoved hReg hDraw hServ hSalt hLucks hUF
  refine le_trans (prob_mono ?_) hH1
  intro x hx
  refine ⟨hx.1, ?_⟩
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  show (n : ℝ) * (Gw x / n) < _
  rw [mul_comm, div_mul_cancel₀ _ hn']
  by_contra hge
  refine absurd hx.2 (not_lt.mpr (creditFloor_le (creditLive_nonneg Sc Cd Wp ws x)
    (creditedSize_nonneg Sc Cd Wp ws (M.size ledger) x) (sizePerWord_nonneg Sc Cd _ (M.size ledger))
    (rightSize_nonneg Sc Cd (M.size ledger) _) ?_ (not_lt.mp hge)))
  exact creditedSize_sub_le Sc Cd Wp ws (M.size ledger) x _ fun w t hg =>
    good_live Cd hr σ k hk sT unitOf hKd ws opened _ _ Wp x w t hg

end Properties.Pouw.ServedValues.SecurityProofs
