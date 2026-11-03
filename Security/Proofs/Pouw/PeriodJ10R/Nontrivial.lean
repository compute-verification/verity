import Proofs.Pouw.PeriodJ10R.Honest

/-!
# `H1J10RCheckedCreditPeriod`'s conclusion fails at a verifier that accepts everything

The nontrivial twin of `HonestJ10R.j10r_meets`, the honest one. The statement reads its verdict, `Credit.window`'s, in
its conclusion only through the accepted prefix (`CreditWindows.prefixLen`) and the prefix's live words (`creditLive`).
`CreditBoundBy` is that conclusion at any verdict `acc` (window `w` credited on path `x`), and at `Credit.window`'s it is
the statement's own (`creditBound_window`).

`j10r_nontrivial`: at the verdict that credits every window, over the honest instance's circuit, scheme, period and
rate (`Λ = j10rRate (k s_T) 1`), the prover that fixes no word (`idle`, accepted on every path) breaks it. The prefix
is all `n` windows and its live words are `N_p·n` (`prefixLenBy_all`, `creditLiveBy_all`), so once
`Λ·ln(1/δ_s) < n` the credit `(1 − Λ·ln(1/δ_s)/n)·N_p·n` is positive (`creditPrefix_pos`), the event holds on every
path, and its probability, `1`, exceeds any `δ_s + n·(δ_bind + δ_ks) < 1`. At `k s_T = 24` and `δ_s = 1%` that is
every period of 39 windows or more.
-/

namespace Properties.Pouw.PeriodJ10R.SecurityProofs.HonestJ10R

open Properties.Pouw.PeriodJ10R
open Accounting.TS Accounting.Compose Pouw.RelativeSecurity
open Accounting.SecurityProofs.ComposePeriod

section By

variable {Ω : Type} {n : ℕ}

/-- **Window `w` and every earlier one are credited** by the verdict `acc` on path `x`. -/
def UpToBy (acc : Paths Ω n → Fin n → Prop) (x : Paths Ω n) (w : Fin n) : Prop :=
  ∀ w' : Fin n, w' ≤ w → acc x w'

open Classical in
/-- **The accepted prefix at verdict `acc`**: `CreditWindows.prefixLen` with `acc` for `Credit.window`'s verdict. -/
noncomputable def prefixLenBy (acc : Paths Ω n → Fin n → Prop) (x : Paths Ω n) : ℕ :=
  (Finset.univ.filter fun w : Fin n => UpToBy acc x w).card

end By

open Classical in
/-- **The accepted prefix's live words at verdict `acc`**: `creditLive` with `acc` for `Credit.window`'s verdict. -/
noncomputable def creditLiveBy {ι K C B : Type} (Sc : Scheme K) (Cd : Declared ι) {Np n : ℕ}
    (Wp : WindowPlay C (Paths C Cd.rounds × B) (Fin Np) Cd.rounds n)
    (acc : Paths (Paths C Cd.rounds × B) n → Fin n → Prop) (x : Paths (Paths C Cd.rounds × B) n) : ℝ :=
  ∑ w : Fin n, if UpToBy acc x w then ∑ t : Fin Np, (liveWords Sc Cd Np (Wp.liveAt x w) t : ℝ) else 0

/-- **`H1J10RCheckedCreditPeriod`'s conclusion at verdict `acc`**: except with probability `δ`, an accepted period's
fixed words are at least the credit of `acc`'s accepted prefix, `(1 − ε(j))₊` of its live words. -/
def CreditBoundBy {ι K T L U C B : Type} [Fintype C] [Fintype B] (Sc : Scheme K) (Cd : Declared ι) {Np n : ℕ}
    (Wp : WindowPlay C (Paths C Cd.rounds × B) (Fin Np) Cd.rounds n)
    (acc : Paths (Paths C Cd.rounds × B) n → Fin n → Prop) (Pl : Play (Paths C Cd.rounds × B) n K T L U)
    (Λ δs δ : ℝ) : Prop :=
  prob (fun x => Pl.accepts x ∧
    n * (Pl.run x).fixed < creditPrefix Sc Cd Np (prefixLenBy acc x) (creditLiveBy Sc Cd Wp acc x) 1 1 Λ δs) ≤ δ

/-- **At `Credit.window`'s verdict it is the statement's conclusion**, so `j10r_nontrivial`, `CreditBoundBy` at the verdict
that credits every window, is the conclusion with that verdict in place of `Credit.window`'s. -/
theorem creditBound_window {ι K T L U C Ξ : Type} [Fintype C] [Fintype Ξ] (Sc : Scheme K) (Cd : Declared ι)
    {nB mB n : ℕ} (σ : Fin nB → Fin mB) (sT : Fin mB) (ws : CreditWindows (Coin C Ξ Cd.rounds σ sT) n)
    (Wp : WindowPlay C (Coin C Ξ Cd.rounds σ sT) (Fin (tileCount σ sT)) Cd.rounds n)
    (Pl : Play (Coin C Ξ Cd.rounds σ sT) n K T L U) (Λ δs δ : ℝ) :
    prob (fun x => Pl.accepts x ∧ n * (Pl.run x).fixed <
      creditPrefix Sc Cd (tileCount σ sT) (ws.prefixLen x) (creditLive Sc Cd Wp ws x) 1 1 Λ δs) ≤ δ ↔
      CreditBoundBy Sc Cd Wp (fun x w => WindowAccepts (ws.reg x w) (ws.ev x w)) Pl Λ δs δ :=
  Iff.rfl

/-! ## The verifier that accepts everything -/

/-- **The prover that does nothing**: the honest play's game, accepted on every path, fixing no word. -/
noncomputable def idle (B : Type) (N Kd n : ℕ) : Play (WCoin N B) n Unit Unit Unit Unit :=
  { play B N Kd n with run := fun _ => { anatomy N with fixed := 0 } }

theorem prefixLenBy_all {Ω : Type} {n : ℕ} (x : Paths Ω n) : prefixLenBy (fun _ _ => True) x = n := by
  classical
  unfold prefixLenBy
  rw [Finset.filter_true_of_mem fun _ _ _ _ => trivial, Finset.card_univ, Fintype.card_fin]

theorem creditLiveBy_all {B : Type} {N n : ℕ} (x : Paths (WCoin N B) n) :
    creditLiveBy Honest.scheme (circuit N) (period (WCoin N B) N n) (fun _ _ => True) x = (N : ℝ) * n := by
  unfold creditLiveBy
  calc _ = ∑ _w : Fin n, (N : ℝ) := Finset.sum_congr rfl fun w _ => by
          rw [if_pos (show UpToBy (fun _ _ => True) x w from fun _ _ => trivial), liveAt_period n x w]
          simp only [liveWords_true]
          exact sum_tileWords _
    _ = N * n := by simp [mul_comm]

theorem zeroCone_eq (N : ℕ) : zeroCone Honest.scheme (circuit N) N = 0 := by
  unfold zeroCone
  exact Finset.sum_eq_zero fun t _ => if_neg (by rw [tileWords_eq]; exact one_ne_zero)

/-- **The whole period's credit is positive once `Λ·ln(1/δ_s) < n`**: `ε(n) = Λ·ln(1/δ_s)/n` at `N_p·n` live words. -/
theorem creditPrefix_pos {N n : ℕ} (hN : 0 < N) (hn : 0 < n) {Λ δs : ℝ} (hΛ : Λ * Real.log (1 / δs) < n) :
    0 < creditPrefix Honest.scheme (circuit N) N n ((N : ℝ) * n) 1 1 Λ δs := by
  have hN' : (0 : ℝ) < N := Nat.cast_pos.mpr hN
  have hn' : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have heps : epsPrefix Honest.scheme (circuit N) N n ((N : ℝ) * n) 1 1 Λ δs = Λ * Real.log (1 / δs) / n := by
    unfold epsPrefix rOfPrefix
    rw [zeroCone_eq, wordsOf_eq, min_self, mul_zero, mul_zero, add_zero, one_mul,
      div_eq_div_iff (mul_pos hN' hn').ne' hn'.ne']
    ring
  have h1 : 0 < 1 - Λ * Real.log (1 / δs) / n := by
    rw [sub_pos, div_lt_one hn']
    exact hΛ
  unfold creditPrefix
  rw [heps, max_eq_left h1.le]
  exact mul_pos h1 (mul_pos hN' hn')

theorem prob_true {Ω : Type} [Fintype Ω] (a : Ω) : prob (fun _ : Ω => True) = 1 := by
  classical
  have : Nonempty Ω := ⟨a⟩
  rw [prob_eq_card, Finset.filter_true_of_mem fun _ _ => trivial, Finset.card_univ]
  exact div_self (Nat.cast_ne_zero.mpr Fintype.card_ne_zero)

/-- **`H1J10RCheckedCreditPeriod`'s conclusion fails at the verifier that accepts everything**: at the honest
instance's circuit, scheme, period and rate over the tile stratum (`N_p = |stratum σ s_T|`, `1 ≤ k s_T`,
`Λ = j10rRate (k s_T) 1`), with every window credited, the prover that fixes no word (`idle`) is accepted on every
path with less than the credit, so the bound fails at every `δ_s + n·(δ_bind + δ_ks) < 1` once
`Λ·ln(1/δ_s) < n`. Its statement is `CreditBoundBy` at that verdict and play, unfolded, so it negates the conclusion's
own `prob … ≤ …`. -/
theorem j10r_nontrivial {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (sT : Fin mB) (hKd : 1 ≤ k sT) (n : ℕ) (hn : 0 < n) {δs δbind δks : ℝ}
    (hΛ : j10rRate (k sT) 1 * Real.log (1 / δs) < n) (hδ : δs + n * (δbind + δks) < 1) :
    ¬ prob (fun x : Paths (Ω1 σ sT) n => (idle Unit (tileCount σ sT) (k sT) n).accepts x ∧
        n * ((idle Unit (tileCount σ sT) (k sT) n).run x).fixed <
          creditPrefix Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) (prefixLenBy (fun _ _ => True) x)
            (creditLiveBy Honest.scheme (circuit (tileCount σ sT)) (period (Ω1 σ sT) (tileCount σ sT) n)
              (fun _ _ => True) x) 1 1 (j10rRate (k sT) 1) δs) ≤ δs + n * (δbind + δks) := by
  intro h
  have hN : 0 < tileCount σ sT := Nat.lt_of_lt_of_le hKd (hk sT)
  have hpos := creditPrefix_pos hN hn hΛ
  have hall := (prob_mono (p := fun _ : Paths (Ω1 σ sT) n => True) fun x _ => by
    refine ⟨trivial, ?_⟩
    rw [prefixLenBy_all, creditLiveBy_all]
    show (n : ℝ) * 0 < _
    rw [mul_zero]
    exact hpos).trans h
  rw [prob_true (Paths.const (Paths.const true _, (Equiv.refl _, ())) n)] at hall
  linarith

end Properties.Pouw.PeriodJ10R.SecurityProofs.HonestJ10R
