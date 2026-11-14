module

public import Proofs.Pouw.Period.Honest
public import Proofs.Pouw.RelativeSecurity
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.RingTheory.WittVector.IsPoly
import Mathlib.Tactic.ENatToNat
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Std.Tactic.BVDecide.Normalize.BitVec
import Std.Tactic.BVDecide.Normalize.Prop

/-!
# `H1CheckedCreditPeriod`'s witnesses

The honest and nontrivial witnesses of `Properties.Pouw.Period.SecurityProofs.h1CheckedCreditPeriod`, both at one
instance: H1's honest live-salt period (`Properties.Pouw.Period.Witnesses`, `Proofs.Pouw.Period.Honest`) of `n = 64`
windows at `δ_s = 1%`, `δ_bind = δ_ks = 0`.

* `h1CheckedCreditPeriod_honest`: every premise of `H1CheckedCreditPeriod` holds there (`HonestH1.h1_meets`), with the
  period's play accepted exactly where the deployed check (`PouwCheck.protocol.check` at `Accept.h1Params`) accepts
  every window (`HonestH1.checkedPlay`), which it does on every path; the accepted prefix is the whole period and its
  live words are `128·64`.
* `h1CheckedCreditPeriod_nontrivial`: on the same windows, with the deployed check's verdict, a play that accepts every
  path and fixes no word (`HonestH1.acceptAll`) breaks the conclusion. At `64` accepted windows
  `ε(64) = (67/8)·ln 100/64 < 1` (`ln 100 < ln 128 = 7·ln 2 ≤ 7`), so the credit is positive (`creditPrefix_pos`), the
  event holds on every path, and its probability, `1`, exceeds `δ_s + n·(δ_bind + δ_ks) = 1/100`.
-/

@[expose] public section

namespace Properties.Pouw.Period.SecurityProofs.HonestH1

open Properties.Pouw.Period Properties.Pouw.Period.SecurityProofs Properties.Pouw.Period.SecurityProofs.Accept
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity
open Accounting.SecurityProofs.ComposeMonitoringPeriod

theorem zeroCone_eq : zeroCone Honest.scheme circuit 144 = 0 := by
  unfold zeroCone
  refine Finset.sum_eq_zero fun t _ => ?_
  by_cases h : tileWords Honest.scheme circuit 144 t = 0
  · rw [if_pos h, tileCone_eq, tileWeight, h, Nat.cast_zero, zero_div]
  · exact if_neg h

/-- **The whole period's credit at `64` accepted windows and `δ_s = 1%` is positive**: `ε(64) = (67/8)·ln 100/64`, and
`ln 100 < ln 128 = 7·ln 2 ≤ 7`. -/
theorem creditPrefix_pos :
    0 < creditPrefix Honest.scheme circuit 144 64 (128 * (64 : ℕ)) 1 1 (67 / 8) (1 / 100) := by
  have h2 : Real.log 2 ≤ 1 := by linarith [Real.log_le_sub_one_of_pos (by norm_num : (0 : ℝ) < 2)]
  have h128 : Real.log 128 = 7 * Real.log 2 := by
    rw [show (128 : ℝ) = 2 ^ 7 by norm_num, Real.log_pow]
    norm_num
  have hlt : Real.log 100 < Real.log 128 := Real.log_lt_log (by norm_num) (by norm_num)
  have hlog : Real.log (1 / (1 / 100)) < 7 := by
    rw [show (1 : ℝ) / (1 / 100) = 100 by norm_num]
    linarith
  have heps : epsPrefix Honest.scheme circuit 144 64 (128 * (64 : ℕ)) 1 1 (67 / 8) (1 / 100) =
      67 / 8 * Real.log (1 / (1 / 100)) / 64 := by
    unfold epsPrefix rOfPrefix
    simp only [zeroCone_eq, wordsOf_eq, min_self, mul_zero, add_zero, one_mul]
    rw [div_eq_div_iff (by norm_num) (by norm_num)]
    push_cast
    ring
  have h1 : 0 < 1 - 67 / 8 * Real.log (1 / (1 / 100)) / 64 := by
    have : 67 / 8 * Real.log (1 / (1 / 100)) / 64 < 1 := by
      rw [div_lt_one (by norm_num)]
      linarith
    linarith
  unfold creditPrefix
  rw [heps, max_eq_left h1.le]
  exact mul_pos h1 (by norm_num)

theorem prob_true {Ω : Type} [Fintype Ω] (a : Ω) : prob (fun _ : Ω => True) = 1 := by
  classical
  have : Nonempty Ω := ⟨a⟩
  rw [prob_eq_card, Finset.filter_true_of_mem fun _ _ => trivial, Finset.card_univ]
  exact div_self (Nat.cast_ne_zero.mpr Fintype.card_ne_zero)

end Properties.Pouw.Period.SecurityProofs.HonestH1

namespace Properties.Pouw.Period.SecurityProofs

open Properties.Pouw.Period Properties.Pouw.Period.SecurityProofs.Accept HonestH1
open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity
open Accounting.SecurityProofs.ComposeMonitoringPeriod

/-- **The honest witness of `h1CheckedCreditPeriod`**: H1's honest live-salt period of `64` windows meets every premise
of `H1CheckedCreditPeriod`, in its order, at its numbers (`N_p = 144`, `K_d = 24`, one round, `ρ = 8/9`, `w_max = 1`,
lucks `7/8` and `½`), `δ_s = 1%` and `δ_bind = δ_ks = 0`, its play accepted where the deployed check accepts every
window (`checkedPlay`). The check accepts every window on every path, so the play accepts on every path, the accepted
prefix is the whole period and its live words are `128·64`. -/
theorem h1CheckedCreditPeriod_honest :
    circuit.units ≤ 144 ∧ circuit.rounds = 1 ∧ Scope Honest.scheme circuit (1 / 100) ∧
    (∀ t : Fin 144, (8 / 9 : ℝ) * (144 : ℕ) * shareCone Honest.scheme circuit 144 t ≤ 1) ∧
    (∀ j, stageShare (tileStage circuit 144) (shareCone Honest.scheme circuit 144) j ≤ 1) ∧
    (∀ t : Fin 144, 0 < shareCone Honest.scheme circuit 144 t →
      tileLuck Honest.scheme circuit 144 t ≤ 7 / 8 ∧ tileLuck0 Honest.scheme circuit 144 t ≤ 1 / 2) ∧
    (checkedPlay 64).game =
      monitoringPeriodGame (tileStage circuit 144) (liveCone Honest.scheme circuit 144) 1 1 24 (period 64) ∧
    (period 64).EveryExecutionPhase (fun live _ => DeadConeDead Honest.scheme circuit 144 live) ∧
    PrefixOf (layer 64) (checkedPlay 64).game ∧
    (∀ w, prob (fun x => (layer 64).MovedIn x w) ≤ 0) ∧
    RegisteredBeforeDraw (windows 64) ∧ SampleIsCoin hK24 prm (windows 64) ∧
    ServiceSound prm (windows 64) opened (stagedAt salt staged) 0 ∧
    SaltIsLiveCoin hr1 (salt (n := 64)) enc ∧ StagedPins staged flat ∧
    (period 64).EveryExecutionPhase
      (fun live W => PerLiveTile Honest.scheme circuit live W (fun k G _ => OperandHMu (Honest.scheme.mu k) G)) ∧
    (period 64).EveryExecutionPhase
      (fun live W => PerLiveTile Honest.scheme circuit live W (fun k G _ => ReceiptLuckAt (Honest.scheme.rB k) G)) ∧
    (period 64).EveryExecutionPhase
      (fun live W => PerLiveTile Honest.scheme circuit live W (fun k _ G0 => GuessLuckAt (Honest.scheme.guess k) G0)) ∧
    UnitFixed (checkedPlay 64) (layer 64) (monitoringPeriodMode circuit (period 64) (run 64))
      (fun _ t => (tileWords Honest.scheme circuit 144 t : ℝ)) (fun _ => 128) ∧
    (∀ x w, Accepts prm ((windows 64).reg x w) ((windows 64).ev x w)) ∧
    (∀ x, (checkedPlay 64).accepts x) ∧
    (∀ x, (windows 64).prefixLen prm x = 64) ∧
    ∀ x, checkedLive Honest.scheme circuit (period 64) prm (windows 64) x = 128 * 64 := by
  obtain ⟨hN, hM, hS, hρN, hwmax, hlk, hG, hDD, hPre, hMoved, hReg, hDraw, hServ, hSalt, hStaged, hMu, hRB, hGs,
    hUF, hAcc, hPrefix, hLive⟩ := h1_meets 64 (by norm_num) (δs := 1 / 100) (by norm_num)
  exact ⟨hN, hM, hS, hρN, hwmax, hlk, hG, hDD, hPre, hMoved, hReg, hDraw, hServ, hSalt, hStaged, hMu, hRB, hGs,
    fun hW x _ => hUF hW x trivial, hAcc, fun x => hAcc x, hPrefix, fun x => by rw [hLive x]; norm_num⟩

/-- **The nontrivial witness of `h1CheckedCreditPeriod`**: its conclusion fails at the verifier that accepts everything.
On H1's honest windows of `64` at the deployed check's verdict, at `δ_s = 1%` and `δ_bind = δ_ks = 0`, the play that
accepts every path and fixes no word (`acceptAll`) has fewer fixed words than the credit on every path, since the
check accepts all `64` windows, their live words are `128·64` and `ε(64) < 1` (`creditPrefix_pos`). So the event has
probability `1`, above `δ_s + n·(δ_bind + δ_ks) = 1/100`. -/
theorem h1CheckedCreditPeriod_nontrivial :
    ¬ prob (fun x : Paths Ω1 64 => (acceptAll 64).accepts x ∧ 64 * ((acceptAll 64).run x).fixed <
        creditPrefix Honest.scheme circuit 144 ((windows 64).prefixLen prm x)
          (checkedLive Honest.scheme circuit (period 64) prm (windows 64) x) 1 1 (67 / 8) (1 / 100)) ≤
      1 / 100 + 64 * (0 + 0) := by
  intro h
  have hall := (prob_mono (p := fun _ : Paths Ω1 64 => True) fun x _ => by
    refine ⟨trivial, ?_⟩
    rw [prefixLen_all, checkedLive_all]
    show (64 : ℝ) * 0 < _
    rw [mul_zero]
    exact creditPrefix_pos).trans h
  rw [prob_true (Paths.const (Paths.const true _, (Equiv.refl _, ())) 64)] at hall
  norm_num at hall

end Properties.Pouw.Period.SecurityProofs
