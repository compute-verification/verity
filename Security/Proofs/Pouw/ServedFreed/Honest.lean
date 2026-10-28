import Proofs.Pouw.ServedFreed
import Proofs.Pouw.ServedWork.Honest

/-!
# `ServedFreed` is honest and nontrivial

The twins of `Properties.Pouw.Served.ServedFreed`, over `Proofs.Pouw.ServedWork.Honest`'s flat machine and honest
inventory at the honest J10-R period, conditional as `served_honest` is on `Credit.window`'s verdict on the
windows' inputs (`hacc`, `hsalt`, `hdraw`).

* `pd c H`: the registered devices' period on the flat machine: `H` seconds of its one resource, all of it the honest
  use and worth `H`, beside one other job worth 1 a unit at one unit of the resource. `pd_dualOpt`: price 1 is its
  dual price, the honest strategy leaving nothing.
* `outsource_fa`: on the flat machine nothing is exchanged, so every program is its own exchange-free replacement
  (`OutsourceWithinEgress` at `β = 0`). `certified_fa`: a certifier that accepts every period meets
  `CertifiedExchange` at `K = δ = 0`.
* `servedFreed_honest`: at `c = N_p`, `p = 1` and `H = N_p·n`, the honest inventory meets every premise
  `ServedFreed` adds to `ServedWork`'s, so the conclusion holds there.
* `servedFreed_nontrivial`: `ServedFreed` is false without `SAKServed`. On the free machine (`fa 0`) every other
  premise holds and `SAKServed` fails (`not_sakServed_free`), and every accepted path leaves the whole period free for
  the other job: worth `2H` against an honest strategy's `H`, past `H + ε(x) H` since its credit is positive
  (`creditPrefix_pos`) and so `ε(x) < 1`. The event has probability 1, past the bound `e^{−1}`.
-/

namespace Properties.Pouw.Served.SecurityProofs.Honest

open Properties.Pouw.Served Properties.Pouw.Served.SecurityProofs
open Properties.Pouw.PeriodJ10R Properties.Pouw.PeriodJ10R.SecurityProofs
open Properties.Pouw.PeriodJ10R.SecurityProofs.HonestJ10R
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity
open Accounting.SecurityProofs.ComposePeriod

/-! ## The registered devices' period -/

/-- The flat machine's period: `H` seconds of its one resource, all of it the honest use, worth `H`, and one other job
worth 1 a unit at one unit of the resource. -/
def pd (c H : ℝ) : Pouw.Dither.Jobs (fa c) where
  T := H
  J := Unit
  val _ := 1
  needs _ _ := 1
  hon _ := H
  rv := H

/-- Price 1 is the period's dual price at one credit: the honest strategy uses the whole period and leaves the other
job nothing. -/
theorem pd_dualOpt (c H : ℝ) : (pd c H).DualOpt (fun _ => 1) 1 := by
  refine ⟨fun _ => zero_le_one, fun _ _ => zero_le_one, fun _ => ?_, fun _ => 0, ⟨fun _ => le_rfl, fun _ => ?_⟩, ?_⟩
  · show (1 : ℝ) ≤ ∑ _r : Unit, (1 : ℝ) * 1
    simp
  · show 1 * H + ∑ _j : Unit, (0 : ℝ) * 1 ≤ 1 * H
    simp
  · show ∑ _j : Unit, (1 : ℝ) * 0 = ∑ _r : Unit, (1 : ℝ) * (1 * H - 1 * H)
    simp

theorem pd_cap (c H : ℝ) : ∑ r, (fun _ => (1 : ℝ)) r * (1 * (pd c H).hon r) = H := by
  show ∑ _r : Unit, (1 : ℝ) * (1 * H) = H
  simp

/-- **The whole period free is worth more than an honest strategy's plus `ε H`**, for `ε < 1`. -/
theorem not_freed {c H ε C : ℝ} (hH : 0 < H) (hC : C = H) (hε : ε < 1) :
    ¬ (pd c H).FreedWithin 1 (fun _ => H) ε C := by
  rintro ⟨W', z', -, ⟨-, hfit⟩, hv⟩
  have h1 := hfit ()
  change W' * H + ∑ j : Unit, z' j * 1 ≤ 1 * H at h1
  change 1 * H + ∑ _j : Unit, (1 : ℝ) * H ≤ W' * H + ∑ j : Unit, 1 * z' j + ε * C at hv
  simp only [Finset.univ_unique, Finset.sum_singleton, mul_one, one_mul] at h1 hv
  have := mul_lt_mul_of_pos_right hε hH
  rw [hC] at hv
  linarith

theorem eps_lt_one {S F : ℝ} (hS : 0 < S) (hF : 0 < F) : max 0 (S - F + 0 * 0) / S < 1 := by
  rw [div_lt_one hS]
  exact max_lt hS (by linarith)

/-! ## The exchange and its certifier -/

/-- On the flat machine every program is its own exchange-free replacement. -/
theorem outsource_fa (c : ℝ) : Pouw.Assumptions.OutsourceWithinEgress (fa c) (fun _ => 1) 0 := by
  intro _ _ _ _ _ _ f
  exact ⟨f, fun _ => ⟨rfl, rfl, le_add_of_nonneg_right (by simp)⟩⟩

/-- A certifier that accepts every period counts the flat machine's exchange at `K = δ = 0`. -/
theorem certified_fa (c : ℝ) {Ω St C O : Type} [Fintype Ω] {n : ℕ} (work : Ω → Fin n → (fa c).Mc.Alg (St × C) O)
    (i : Ω → Fin n → St × C) :
    CertifiedExchange (fun _ => True) (fun x => ∑ w, (fa c).ext (work x w) (i x w)) 0 0 := by
  unfold CertifiedExchange
  refine (prob_mono fun x h => ?_).trans prob_false_eq.le
  have e : ∑ w, (fa c).ext (work x w) (i x w) = 0 := Finset.sum_eq_zero fun _ _ => rfl
  have h2 : (0 : ℝ) < ∑ w, (fa c).ext (work x w) (i x w) := h.2
  rw [e] at h2
  exact lt_irrefl 0 h2

section Meets

variable {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB)
  (unitOf : Fin nB → ℕ) (regOf : Bool → Registered)
  (evOf : Bool → Equiv.Perm (Fin (tileCount σ sT)) × Unit → Answered) (enc : Bool → ByteArray)

/-- **`ServedFreed`'s conclusion at the honest J10-R period with the honest inventory**: `j10r_meets` and
`served_meets` give it `ServedWork`'s premises, and the flat machine's period (`pd`, at `H = N_p·n`), its
exchange-free programs and a certifier that accepts every period the rest. -/
theorem servedFreed_honest (hKd : 1 ≤ k sT) (hacc : ∀ c β, WindowAccepts (regOf c) (evOf c β))
    (hsalt : ∀ c, (PouwCheck.hex? (regOf c).salt).getD ByteArray.empty = enc c)
    (hdraw : ∀ c β, ∃ d : Array ℕ, creditDrawn (regOf c) (evOf c β) = .ok d ∧
      ∀ u : ℕ, u ∈ d.toList ↔ ∃ i : Fin (k sT), unitOf (tileAt σ sT (drawOf (hk sT) β i)) = u)
    (n : ℕ) (hn : 0 < n) {δs : ℝ} (hδs : 0 < δs) :
    prob (fun x => (play Unit (tileCount σ sT) (k sT) n).accepts x ∧ True ∧
      ∃ z, (pd (tileCount σ sT) (n * tileCount σ sT)).Fits
          (fun r => ∑ w, (fa (tileCount σ sT)).use (work (tileCount σ sT) (Np := tileCount σ sT) x w)
            ((), saltCoin (hr1 _) x w) r) z ∧
        ¬ (pd (tileCount σ sT) (n * tileCount σ sT)).FreedWithin 1 z
          (max 0 (∑ r, (fun _ => (1 : ℝ)) r * (1 * (pd (tileCount σ sT) (n * tileCount σ sT)).hon r) -
              (1 * creditPrefix Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT)
                ((windows σ sT regOf evOf n).prefixLen x)
                (creditLive Honest.scheme (circuit (tileCount σ sT)) (period (Ω1 σ sT) (tileCount σ sT) n)
                  (windows σ sT regOf evOf n) x) 1 1 (j10rRate (k sT) 1) δs - ∑ _w : Fin n, (0 : ℝ)) + 0 * 0) /
            ∑ r, (fun _ => (1 : ℝ)) r * (1 * (pd (tileCount σ sT) (n * tileCount σ sT)).hon r))
          (∑ r, (fun _ => (1 : ℝ)) r * (1 * (pd (tileCount σ sT) (n * tileCount σ sT)).hon r))) ≤
      δs + n * (0 + 0) + n * 0 + 0 := by
  obtain ⟨hN, hM, hS, hρN, hwmax, hlk, hG, hDD, hPre, hMoved, hReg, hDraw, hServ, hSalt, hLucks, hUF, -, -, -⟩ :=
    j10r_meets σ k hk sT unitOf regOf evOf enc hKd hacc hsalt hdraw n hn hδs
  obtain ⟨g1, g2, p1, p2, hSAK, hSt, -, hIn, hp, hFix, hRight⟩ := served_meets σ k sT n
  have hNp : (0 : ℝ) < tileCount σ sT := Nat.cast_pos.mpr (Nat.lt_of_lt_of_le hKd (hk sT))
  have hC : 0 < ∑ r, (fun _ => (1 : ℝ)) r * (1 * (pd (tileCount σ sT) (n * tileCount σ sT)).hon r) :=
    (mul_pos (Nat.cast_pos.mpr hn) hNp).trans_eq (pd_cap _ _).symm
  exact servedFreed σ k hk sT unitOf Honest.scheme (circuit (tileCount σ sT)) hn hN (hr1 _) hM hKd
    (windows σ sT regOf evOf n) opened enc place tile (period (Ω1 σ sT) (tileCount σ sT) n)
    (play Unit (tileCount σ sT) (k sT) n) (fun _ => (tileCount σ sT : ℝ)) 1 δs 0 0
    (fa (tileCount σ sT)) (fun _ => 1) Set.univ hon 1 (fun _ => 0) (fun _ => 0) 0 (fun _ _ => ()) (fun _ _ => ())
    (work (tileCount σ sT)) (pd (tileCount σ sT) (n * tileCount σ sT)) (fun _ => True) 0 0 0
    hS one_pos le_rfl hρN hwmax hlk hG hDD hPre hMoved hReg hDraw hServ hSalt hLucks hUF
    g1 g2 p1 p2 hSAK hSt hIn hp hFix hRight (pd_dualOpt _ _) hC le_rfl
    (outsourceServed_of_perProgram _ (hr1 _) _ _ _ _ _ (outsource_fa _))
    (certified_fa _ (work (tileCount σ sT)) fun x w => ((), saltCoin (hr1 _) x w))

/-- **`ServedFreed` is false without `SAKServed`.** On the free machine (`fa 0`), at the honest J10-R period of
`n ≥ 17` windows, `δ_s = e^{−1}`, price 1 a word, no luck allowance, no degenerate salts, the honest inventory, the
period `pd 0 H` at `H = N_p·n`, `β = K = δ = 0` and a certifier that accepts every period, every other premise of
`ServedFreed` holds, in its order (`j10r_meets`, `served_rest`, `pd_dualOpt`, `outsource_fa`, `certified_fa`), and
`SAKServed` fails (`not_sakServed_free`). The conclusion fails with it: every accepted path's windows use nothing, so
the other job fits the whole period (`z = H`), worth `2H` against an honest strategy's `H`, and `ε(x) < 1` since its
credit is positive (`creditPrefix_pos`). So its event has probability 1, past the bound `e^{−1}`. -/
theorem servedFreed_nontrivial (hKd : 1 ≤ k sT) (hacc : ∀ c β, WindowAccepts (regOf c) (evOf c β))
    (hsalt : ∀ c, (PouwCheck.hex? (regOf c).salt).getD ByteArray.empty = enc c)
    (hdraw : ∀ c β, ∃ d : Array ℕ, creditDrawn (regOf c) (evOf c β) = .ok d ∧
      ∀ u : ℕ, u ∈ d.toList ↔ ∃ i : Fin (k sT), unitOf (tileAt σ sT (drawOf (hk sT) β i)) = u)
    (n : ℕ) (hn : 17 ≤ n) :
    (∀ s, k s ≤ (stratum σ s).card) ∧ 0 < n ∧ (circuit (tileCount σ sT)).units ≤ tileCount σ sT ∧
    0 < (circuit (tileCount σ sT)).rounds ∧ (circuit (tileCount σ sT)).rounds = 1 ∧ 1 ≤ k sT ∧
    Scope Honest.scheme (circuit (tileCount σ sT)) (Real.exp (-1)) ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ∧
    (∀ t : Fin (tileCount σ sT),
      (1 : ℝ) * (tileCount σ sT) * shareCone Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) t ≤ 1) ∧
    (∀ j, stageShare (tileStage (circuit (tileCount σ sT)) (tileCount σ sT))
      (shareCone Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT)) j ≤ 1) ∧
    (∀ t : Fin (tileCount σ sT), 0 < shareCone Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) t →
      tileLuck Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) t ≤ 7 / 8 ∧
        tileLuck0 Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) t ≤ 1 / 2) ∧
    (play Unit (tileCount σ sT) (k sT) n).game =
      periodGame (tileStage (circuit (tileCount σ sT)) (tileCount σ sT))
        (liveCone Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT)) 1 1 (k sT)
        (period (Ω1 σ sT) (tileCount σ sT) n) ∧
    (period (Ω1 σ sT) (tileCount σ sT) n).EveryWindow
      (fun live _ => DeadConeDead Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) live) ∧
    PrefixOf (tileLayer (circuit (tileCount σ sT)) (hr1 _) σ k hk sT unitOf (windows σ sT regOf evOf n) opened
      (pinnedAt (saltOf (windows σ sT regOf evOf n)) place tile) (j10rAtCoin (hr1 _) enc place tile))
      (play Unit (tileCount σ sT) (k sT) n).game ∧
    (∀ w, prob (fun x => (tileLayer (circuit (tileCount σ sT)) (hr1 _) σ k hk sT unitOf (windows σ sT regOf evOf n)
      opened (pinnedAt (saltOf (windows σ sT regOf evOf n)) place tile)
      (j10rAtCoin (hr1 _) enc place tile)).MovedIn x w) ≤ 0) ∧
    RegisteredBeforeDraw (windows σ sT regOf evOf n) ∧ TileDrawIsCoin σ k hk sT unitOf (windows σ sT regOf evOf n) ∧
    TileServiceSound σ sT unitOf (windows σ sT regOf evOf n) opened
      (pinnedAt (saltOf (windows σ sT regOf evOf n)) place tile) 0 ∧
    Properties.Pouw.Period.SaltIsLiveCoin (hr1 _) (saltOf (windows σ sT regOf evOf n)) enc ∧
    SharedNoiseLucks Honest.scheme (circuit (tileCount σ sT)) (period (Ω1 σ sT) (tileCount σ sT) n) ∧
    UnitFixed (play Unit (tileCount σ sT) (k sT) n)
      (tileLayer (circuit (tileCount σ sT)) (hr1 _) σ k hk sT unitOf (windows σ sT regOf evOf n) opened
        (pinnedAt (saltOf (windows σ sT regOf evOf n)) place tile) (j10rAtCoin (hr1 _) enc place tile))
      (periodMode (circuit (tileCount σ sT)) (period (Ω1 σ sT) (tileCount σ sT) n)
        (tileRun σ k hk sT unitOf (windows σ sT regOf evOf n) opened
          (pinnedAt (saltOf (windows σ sT regOf evOf n)) place tile) (j10rAtCoin (hr1 _) enc place tile)))
      (fun _ t => (tileWords Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) t : ℝ))
      (fun _ => (tileCount σ sT : ℝ)) ∧
    (fa 0).G1 ∧ (fa 0).G2 ∧ (fa 0).P1 ∧ (fa 0).P2 ∧
    ¬ Pouw.Assumptions.SAKServed (fa 0) (fun _ => 1) Set.univ (hon (Np := tileCount σ sT))
      (fun t => (tileWords Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) t : ℝ)) 1 (fun _ => 0)
      (fun _ => 0) ∧
    (∃ e : Unit → Fin (2 ^ 0), Function.Injective e) ∧
    (∀ (_x : Paths (Ω1 σ sT) n) (_w : Fin n), () ∈ (Set.univ : Set Unit)) ∧ (0 : ℝ) ≤ 1 ∧
    FixedBeforeSalt (fun (x : Paths (Ω1 σ sT) n) w => ((), (), work 0 (Np := tileCount σ sT) x w)) ∧
    FixedAreRight (play Unit (tileCount σ sT) (k sT) n) (fun x w => Pouw.Assumptions.rightWords hon
      (fun t => (tileWords Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT) t : ℝ)) ()
      (saltCoin (hr1 _) x w) ((fa 0).Mc.eval (work 0 x w) ((), saltCoin (hr1 _) x w))) ∧
    (pd 0 (n * tileCount σ sT)).DualOpt (fun _ => 1) 1 ∧
    0 < ∑ r, (fun _ => (1 : ℝ)) r * (1 * (pd 0 (n * tileCount σ sT)).hon r) ∧ (0 : ℝ) ≤ 0 ∧
    OutsourceWithinEgressServed (Fin (tileCount σ sT) → Bool) (hr1 _) (fa 0) (fun _ => 1) 0
      (fun (_ : Paths (Ω1 σ sT) n) _ => ()) (fun _ _ => ()) ∧
    CertifiedExchange (fun _ => True)
      (fun x : Paths (Ω1 σ sT) n => ∑ w, (fa 0).ext (work 0 (Np := tileCount σ sT) x w) ((), saltCoin (hr1 _) x w))
      0 0 ∧
    ¬ prob (fun x => (play Unit (tileCount σ sT) (k sT) n).accepts x ∧ True ∧
      ∃ z, (pd 0 (n * tileCount σ sT)).Fits
          (fun r => ∑ w, (fa 0).use (work 0 (Np := tileCount σ sT) x w) ((), saltCoin (hr1 _) x w) r) z ∧
        ¬ (pd 0 (n * tileCount σ sT)).FreedWithin 1 z
          (max 0 (∑ r, (fun _ => (1 : ℝ)) r * (1 * (pd 0 (n * tileCount σ sT)).hon r) -
              (1 * creditPrefix Honest.scheme (circuit (tileCount σ sT)) (tileCount σ sT)
                ((windows σ sT regOf evOf n).prefixLen x)
                (creditLive Honest.scheme (circuit (tileCount σ sT)) (period (Ω1 σ sT) (tileCount σ sT) n)
                  (windows σ sT regOf evOf n) x) 1 1 (j10rRate (k sT) 1) (Real.exp (-1)) -
                ∑ _w : Fin n, (0 : ℝ)) + 0 * 0) /
            ∑ r, (fun _ => (1 : ℝ)) r * (1 * (pd 0 (n * tileCount σ sT)).hon r))
          (∑ r, (fun _ => (1 : ℝ)) r * (1 * (pd 0 (n * tileCount σ sT)).hon r))) ≤
      Real.exp (-1) + n * (0 + 0) + n * 0 + 0 := by
  have hn0 : 0 < n := by omega
  obtain ⟨hN, hM, hS, hρN, hwmax, hlk, hG, hDD, hPre, hMoved, hReg, hDraw, hServ, hSalt, hLucks, hUF, -, -, -⟩ :=
    j10r_meets σ k hk sT unitOf regOf evOf enc hKd hacc hsalt hdraw n hn0 (Real.exp_pos (-1))
  obtain ⟨g1, g2, p1, p2, hSt, -, hIn, hp, hFix, hRight⟩ := served_rest σ k sT (le_refl (0 : ℝ)) n
  have hne : Nonempty (Ω1 σ sT) := ⟨((true, ()), (Equiv.refl _, ()))⟩
  have := @paths_nonempty (Ω1 σ sT) hne n
  have hNp : (0 : ℝ) < tileCount σ sT := Nat.cast_pos.mpr (Nat.lt_of_lt_of_le hKd (hk sT))
  have hH : (0 : ℝ) < n * tileCount σ sT := mul_pos (Nat.cast_pos.mpr hn0) hNp
  refine ⟨hk, hn0, hN, hr1 _, hM, hKd, hS, one_pos, le_rfl, hρN, hwmax, hlk, hG, hDD, hPre, hMoved, hReg, hDraw,
    hServ, hSalt, hLucks, hUF, g1, g2, p1, p2, not_sakServed_free σ k hk sT hKd, hSt, hIn, hp, hFix, hRight,
    pd_dualOpt _ _, hH.trans_eq (pd_cap _ _).symm, le_rfl,
    outsourceServed_of_perProgram _ (hr1 _) _ _ _ _ _ (outsource_fa _),
    certified_fa _ (work 0) fun x w => ((), saltCoin (hr1 _) x w), fun hle => ?_⟩
  rw [prob_true] at hle
  · have := Real.exp_lt_one_iff.mpr (show (-1 : ℝ) < 0 by norm_num)
    simp only [mul_zero, add_zero] at hle
    linarith
  · intro x
    refine ⟨trivial, trivial, fun _ => (n * tileCount σ sT : ℝ), ⟨fun _ => hH.le, fun _ => ?_⟩,
      not_freed hH (pd_cap _ _) (eps_lt_one (hH.trans_eq (pd_cap _ _).symm) ?_)⟩
    · show ∑ _w : Fin n, (0 : ℝ) + ∑ _j : Unit, (n * tileCount σ sT : ℝ) * 1 ≤ 1 * (n * tileCount σ sT)
      simp
    · simp only [Finset.sum_const_zero, sub_zero, one_mul]
      exact creditPrefix_pos σ k hk sT regOf evOf hKd hacc n hn x

end Meets

end Properties.Pouw.Served.SecurityProofs.Honest
