import Proofs.Pouw.ServedWork
import Proofs.Pouw.PeriodJ10R.Honest

/-!
# `ServedWork` is honest and nontrivial

The honest and nontrivial lemmas of `Properties.Pouw.Served.ServedWork`, over `Proofs.Pouw.PeriodJ10R.Honest`'s honest J10-R
period, conditional as `j10r_meets` is on `Credit.window`'s verdict on the windows' inputs (`hacc`, `hsalt`, `hdraw`).

* `fm`, `fa c`: a flat machine. A program is any function, and every run uses `c` units of one resource at rate 1 and
  exchanges nothing, so its cost at price 1 is `c` (`fa_cost`). It meets the device premises (`fa_G1`, `fa_G2`).
* `sakServed_fa`: `SAKServed` holds on `fa c` at a price `p` a credited word whenever `c` pays for every word, with no
  degenerate salt.
* The honest inventory: one instance (`Unit`), no advice, and a program whose every tile's word is the window's salt bit
  (`hon`, `work`): it gets every tile right.
* `served_meets`: at `c = N_p` and `p = 1` the honest inventory meets every premise `ServedWork` adds to
  `H1J10RCheckedCreditPeriod`'s, which `j10r_meets` gives; `served_honest` is the conclusion there.
* `served_nontrivial`: the conclusion fails for a verifier that accepts everything. On the free machine (`fa 0`), at
  `n ≥ 17` windows and `δ_s = e^{−1}`, every path is accepted and earns credit `(1 − ε(n))·N_p·n > 0` for no work, so
  the event has probability 1, past the bound `e^{−1}`. The free machine is what `SAKServed` rules out.
-/

namespace Properties.Pouw.Served.SecurityProofs.Honest

open Properties.Pouw.Served Properties.Pouw.Served.SecurityProofs
open Properties.Pouw.PeriodJ10R Properties.Pouw.PeriodJ10R.SecurityProofs
open Properties.Pouw.PeriodJ10R.SecurityProofs.HonestJ10R
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity
open Accounting.SecurityProofs.ComposePeriod

/-! ## A flat machine -/

/-- Programs are functions; the machine's own costs are zero (the accelerator's `use` prices runs). -/
abbrev fm : Machine where
  Alg I O := I → O
  eval f := f
  time _ := 0
  cost _ _ := 0
  size _ := 0
  seq f g := g ∘ f
  pair f g := fun x => (f x, g x)
  fst := Prod.fst
  snd := Prod.snd
  eval_seq _ _ _ := rfl
  eval_pair _ _ _ := rfl
  eval_fst _ := rfl
  eval_snd _ := rfl
  time_nonneg _ := le_rfl
  cost_le_time _ _ := le_rfl
  size_le_time _ := le_rfl
  size_nonneg _ := le_rfl
  time_seq_le _ _ := by norm_num
  le_time_seq _ _ := le_rfl
  time_pair_le _ _ := by norm_num
  le_time_pair_left _ _ := le_rfl
  le_time_pair_right _ _ := le_rfl
  time_fst := rfl
  time_snd := rfl

/-- **The flat accelerator**: one resource at rate 1, every run using `c` units of it over `c` seconds, holding and
exchanging nothing. -/
noncomputable abbrev fa (c : ℝ) : Pouw.Dither.Accel where
  Mc := fm
  Res := Unit
  rate _ := 1
  mem := 0
  use _ _ _ := c
  held _ _ := 0
  wall _ _ := c
  ext _ _ := 0
  calls _ _ := 0
  callUse _ := 0
  P1 := True
  P2 := True

theorem fa_G1 (c : ℝ) : (fa c).G1 := fun _ _ => by
  show (0 : ℝ) + 0 ≤ 0
  norm_num

theorem fa_G2 {c : ℝ} (hc : 0 ≤ c) : (fa c).G2 :=
  ⟨fun _ => one_pos, fun {_ _} _ _ => ⟨le_rfl, fun _ => ⟨hc, by show c ≤ 1 * c; rw [one_mul]⟩⟩⟩

theorem fa_cost (c : ℝ) {I O : Type} (f : (fa c).Mc.Alg I O) (x : I) : (fa c).cost (fun _ => 1) f x = c := by
  show ∑ _r : Unit, (1 : ℝ) * c = c
  simp

theorem rightWords_le {I D W : Type} {Np : ℕ} (hon : I → D → Fin Np → W) (wt : Fin Np → ℝ) (hwt : ∀ t, 0 ≤ wt t)
    (a : I) (d : D) (T : Fin Np → W) : Pouw.Assumptions.rightWords hon wt a d T ≤ ∑ t, wt t := by
  unfold Pouw.Assumptions.rightWords
  exact Finset.sum_le_sum fun t _ => by split_ifs; exacts [le_rfl, hwt t]

/-- **`SAKServed` on the flat machine**: at a price `p ≥ 0` a credited word, with `c` paying for every word less `lam`,
every program meets it with no degenerate salt. -/
theorem sakServed_fa {I D W : Type} [Fintype D] {Np : ℕ} (Reg : Set I) (hon : I → D → Fin Np → W)
    (wt : Fin Np → ℝ) (p c : ℝ) (lam : I → ℝ) (hwt : ∀ t, 0 ≤ wt t) (hp : 0 ≤ p)
    (hc : ∀ a ∈ Reg, p * ∑ t, wt t - lam a ≤ c) :
    Pouw.Assumptions.SAKServed (fa c) (fun _ => 1) Reg hon wt p lam (fun _ => 0) := by
  intro _ _ _ _ a ha B St _ s Mw _
  refine ⟨fun _ => False, prob_false_eq.le, fun d _ => ?_⟩
  rw [fa_cost]
  have := mul_le_mul_of_nonneg_left (rightWords_le hon wt hwt a d ((fa c).Mc.eval Mw (s, d))) hp
  linarith [hc a ha]

/-! ## The honest inventory -/

/-- The honest words: every tile's word is the window's salt bit. -/
def hon {Np : ℕ} : Unit → Bool → Fin Np → Bool := fun _ d _ => d

/-- The honest program: the honest words of its salt, on any advice. -/
def work (c : ℝ) {Ω : Type} {n Np : ℕ} : Paths Ω n → Fin n → (fa c).Mc.Alg (Unit × Bool) (Fin Np → Bool) :=
  fun _ _ sd _ => sd.2

theorem rightWords_hon {Np : ℕ} (wt : Fin Np → ℝ) (d : Bool) :
    Pouw.Assumptions.rightWords hon wt () d (fun _ => d) = ∑ t, wt t := by
  unfold Pouw.Assumptions.rightWords
  exact Finset.sum_congr rfl fun t _ => ite_eq_left rfl

section Meets

variable {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB)
  (unitOf : Fin nB → ℕ) (regOf : Bool → Registered)
  (evOf : Bool → Equiv.Perm (Fin (tileCount σ sT)) × Unit → Answered) (enc : Bool → ByteArray)

/-- **The honest inventory meets every premise `ServedWork` adds**, in its order, on the flat machine at `c = N_p`
(a window's words at price 1) with `p = 1`, no luck allowance, no degenerate salts and no advice (`B = 0`). -/
theorem served_meets (n : ℕ) :
    (fa (tileCount σ sT)).G1 ∧ (fa (tileCount σ sT)).G2 ∧ (fa (tileCount σ sT)).P1 ∧ (fa (tileCount σ sT)).P2 ∧
    Pouw.Assumptions.SAKServed (fa (tileCount σ sT)) (fun _ => 1) Set.univ (hon (Np := tileCount σ sT))
      (fun t => (tileWords Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) t : ℝ)) 1 (fun _ => 0)
      (fun _ => 0) ∧
    (∃ e : Unit → Fin (2 ^ 0), Function.Injective e) ∧
    (∀ (x : Paths (Ω1 σ sT) n) w i, (fa (tileCount σ sT)).ext (work (tileCount σ sT) (Np := tileCount σ sT) x w) i = 0) ∧
    (∀ (_x : Paths (Ω1 σ sT) n) (_w : Fin n), () ∈ (Set.univ : Set Unit)) ∧ (0 : ℝ) ≤ 1 ∧
    FixedBeforeSalt (fun (x : Paths (Ω1 σ sT) n) w => ((), (), work (tileCount σ sT) (Np := tileCount σ sT) x w)) ∧
    FixedAreRight (play Unit (tileCount σ sT) (k sT) n) (fun x w => Pouw.Assumptions.rightWords hon
      (fun t => (tileWords Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) t : ℝ)) ()
      (saltCoin (hr1 _) x w)
      ((fa (tileCount σ sT)).Mc.eval (work (tileCount σ sT) x w) ((), saltCoin (hr1 _) x w))) := by
  have hN : (0 : ℝ) ≤ tileCount σ sT := Nat.cast_nonneg _
  refine ⟨fa_G1 _, fa_G2 hN, trivial, trivial,
    sakServed_fa _ _ _ 1 _ _ (fun _ => Nat.cast_nonneg _) zero_le_one fun _ _ => ?_,
    ⟨fun _ => 0, fun _ _ _ => rfl⟩, fun _ _ _ => rfl, fun _ _ => trivial, zero_le_one,
    fun _ _ _ => rfl, fun x _ => ?_⟩
  · simp [sum_tileWords]
  · have h : ∀ w, Pouw.Assumptions.rightWords hon
        (fun t => (tileWords Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) t : ℝ)) ()
        (saltCoin (hr1 _) x w)
        ((fa (tileCount σ sT)).Mc.eval (work (tileCount σ sT) (Np := tileCount σ sT) x w)
          ((), saltCoin (hr1 _) x w)) = tileCount σ sT := fun w => by
      have e : (fa (tileCount σ sT)).Mc.eval (work (tileCount σ sT) (Np := tileCount σ sT) x w)
          ((), saltCoin (hr1 _) x w) = fun _ => saltCoin (hr1 _) x w := rfl
      rw [e, rightWords_hon]
      exact sum_tileWords _
    show (n : ℝ) * (tileCount σ sT : ℝ) ≤ _
    refine le_of_eq (Eq.symm ((Finset.sum_congr rfl fun w _ => h w).trans ?_))
    simp

/-- **`ServedWork`'s conclusion at the honest J10-R period with the honest inventory**: `j10r_meets` and
`served_meets` give it every premise, so except with probability `δ_s` its `N_p` a window of priced work is at least
`(1 − ε(n))₊` of its `N_p·n` live words. -/
theorem served_honest (hKd : 1 ≤ k sT) (hacc : ∀ c β, WindowAccepts (regOf c) (evOf c β))
    (hsalt : ∀ c, (PouwCheck.hex? (regOf c).salt).getD ByteArray.empty = enc c)
    (hdraw : ∀ c β, ∃ d : Array ℕ, creditDrawn (regOf c) (evOf c β) = .ok d ∧
      ∀ u : ℕ, u ∈ d.toList ↔ ∃ i : Fin (k sT), unitOf (tileAt σ sT (drawOf (hk sT) β i)) = u)
    (n : ℕ) (hn : 0 < n) {δs : ℝ} (hδs : 0 < δs) :
    prob (fun x => (play Unit (tileCount σ sT) (k sT) n).accepts x ∧
      ∑ w, (fa (tileCount σ sT)).cost (fun _ => 1) (work (tileCount σ sT) (Np := tileCount σ sT) x w)
        ((), saltCoin (hr1 _) x w) <
        1 * creditPrefix Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT)
          ((windows σ sT regOf evOf n).prefixLen x)
          (creditLive Honest.scheme (circuit (tileCount σ sT)) (period (Ω1 σ sT) (tileCount σ sT) n)
            (windows σ sT regOf evOf n) x) 1 1 (j10rRate (k sT) 1) δs - ∑ _w : Fin n, (0 : ℝ)) ≤
      δs + n * (0 + 0) + n * 0 := by
  obtain ⟨hN, hM, hS, hρN, hwmax, hlk, hG, hDD, hPre, hMoved, hReg, hDraw, hServ, hSalt, hLucks, hUF, -, -, -⟩ :=
    j10r_meets σ k hk sT unitOf regOf evOf enc hKd hacc hsalt hdraw n hn hδs
  obtain ⟨g1, g2, p1, p2, hSAK, hSt, hext, hIn, hp, hFix, hRight⟩ := served_meets σ k sT n
  exact servedWork σ k hk sT unitOf Honest.scheme (circuit (tileCount σ sT)) hn hN (hr1 _) hM hKd
    (windows σ sT regOf evOf n) opened enc place tile (period (Ω1 σ sT) (tileCount σ sT) n)
    (play Unit (tileCount σ sT) (k sT) n) (fun _ => (tileCount σ sT : ℝ)) 1 δs 0 0
    (fa (tileCount σ sT)) (fun _ => 1) Set.univ hon 1 (fun _ => 0) (fun _ => 0) 0 (fun _ _ => ()) (fun _ _ => ())
    (work (tileCount σ sT)) hS one_pos le_rfl hρN hwmax hlk hG hDD hPre hMoved hReg hDraw hServ hSalt hLucks hUF
    g1 g2 p1 p2 hSAK hSt hext hIn hp hFix hRight

/-! ## Nontrivial -/

theorem zeroCone_circuit (N : ℕ) : zeroCone Honest.scheme (circuit N) N = 0 := by
  unfold zeroCone
  exact Finset.sum_eq_zero fun t _ => ite_eq_right (by rw [tileWords_eq]; exact one_ne_zero)

theorem prob_true {Ω : Type} [Fintype Ω] [Nonempty Ω] {P : Ω → Prop} (h : ∀ x, P x) : prob P = 1 := by
  classical
  unfold prob
  rw [Finset.filter_true_of_mem fun x _ => h x, Finset.card_univ]
  exact div_self (Nat.cast_pos.mpr Fintype.card_pos).ne'

include hk in
/-- **The credit of the honest period is positive** past `n ≥ 17` windows at `δ_s = e^{−1}`: `ε(n) = Λ/n` with
`Λ = j10rRate (k s_T) 1 ≤ 16`. -/
theorem creditPrefix_pos (hKd : 1 ≤ k sT) (hacc : ∀ c β, WindowAccepts (regOf c) (evOf c β)) (n : ℕ) (hn : 17 ≤ n)
    (x : Paths (Ω1 σ sT) n) :
    0 < creditPrefix Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT)
      ((windows σ sT regOf evOf n).prefixLen x)
      (creditLive Honest.scheme (circuit (tileCount σ sT)) (period (Ω1 σ sT) (tileCount σ sT) n)
        (windows σ sT regOf evOf n) x) 1 1 (j10rRate (k sT) 1) (Real.exp (-1)) := by
  have hN : (0 : ℝ) < tileCount σ sT := Nat.cast_pos.mpr (Nat.lt_of_lt_of_le hKd (hk sT))
  have hn17 : (17 : ℝ) ≤ n := by exact_mod_cast hn
  have hn0 : (0 : ℝ) < n := by linarith
  have hK : (1 : ℝ) ≤ k sT := by exact_mod_cast hKd
  have hlog : Real.log (1 / Real.exp (-1)) = 1 := by
    rw [one_div, ← Real.exp_neg, neg_neg, Real.log_exp]
  have hΛ : j10rRate (k sT) 1 ≤ 16 := by
    unfold j10rRate
    rw [div_le_iff₀ (by linarith)]
    linarith
  have heps : epsPrefix Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) n
      ((tileCount σ sT : ℝ) * n) 1 1 (j10rRate (k sT) 1) (Real.exp (-1)) ≤ 16 / 17 := by
    unfold epsPrefix rOfPrefix
    rw [zeroCone_circuit, hlog, wordsOf_eq]
    simp only [min_self, mul_zero, add_zero, mul_one, one_mul]
    rw [div_le_div_iff₀ (mul_pos hN hn0) (by norm_num)]
    have a1 : j10rRate (k sT) 1 * (tileCount σ sT : ℝ) ≤ 16 * (tileCount σ sT : ℝ) :=
      mul_le_mul_of_nonneg_right hΛ hN.le
    have a2 : (tileCount σ sT : ℝ) * 17 ≤ (tileCount σ sT : ℝ) * n := mul_le_mul_of_nonneg_left hn17 hN.le
    linarith
  rw [prefixLen_all σ sT regOf evOf hacc n x, creditLive_all σ sT regOf evOf hacc n x]
  unfold creditPrefix
  exact mul_pos (lt_max_of_lt_left (by linarith)) (mul_pos hN hn0)

include hk in
/-- **Nontrivial: the conclusion fails for a verifier that accepts everything.** On the free machine (`fa 0`: every
program costs nothing), at the honest J10-R period of `n ≥ 17` windows, every path accepted, `δ_s = e^{−1}`, price 1
a word, no luck allowance and no degenerate salts, every path is in the conclusion's event: no work, positive credit
(`creditPrefix_pos`). So its probability is 1, past the bound `e^{−1}`. -/
theorem served_nontrivial (hKd : 1 ≤ k sT) (hacc : ∀ c β, WindowAccepts (regOf c) (evOf c β)) (n : ℕ) (hn : 17 ≤ n) :
    ¬ prob (fun x => (play Unit (tileCount σ sT) (k sT) n).accepts x ∧
      ∑ w, (fa 0).cost (fun _ => 1) (work 0 (Np := tileCount σ sT) x w) ((), saltCoin (hr1 _) x w) <
        1 * creditPrefix Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT)
          ((windows σ sT regOf evOf n).prefixLen x)
          (creditLive Honest.scheme (circuit (tileCount σ sT)) (period (Ω1 σ sT) (tileCount σ sT) n)
            (windows σ sT regOf evOf n) x) 1 1 (j10rRate (k sT) 1) (Real.exp (-1)) - ∑ _w : Fin n, (0 : ℝ)) ≤
      Real.exp (-1) + n * (0 + 0) + n * 0 := by
  have hne : Nonempty (Ω1 σ sT) := ⟨((true, ()), (Equiv.refl _, ()))⟩
  have := @paths_nonempty (Ω1 σ sT) hne n
  intro hle
  rw [prob_true] at hle
  · have := Real.exp_lt_one_iff.mpr (show (-1 : ℝ) < 0 by norm_num)
    simp only [mul_zero, add_zero] at hle
    linarith
  · intro x
    refine ⟨trivial, ?_⟩
    simp only [fa_cost, Finset.sum_const_zero, sub_zero, one_mul]
    exact creditPrefix_pos σ k hk sT regOf evOf hKd hacc n hn x

end Meets

end Properties.Pouw.Served.SecurityProofs.Honest
