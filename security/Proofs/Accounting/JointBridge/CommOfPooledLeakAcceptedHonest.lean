module

public import Definitions.Accounting.Guarantees.JointBridge.CommOfPooledLeakAccepted
public import Definitions.Accounting.NetWitness
public import Proofs.NetworkCertifier.Witnesses
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
import Mathlib.Topology.Sheaves.Init
import Proofs.Pouw.Prob
/-!
# The pooled bridge's two inputs hold at an honest run (witness)

`Accounting.SecurityProofs.Bridge.CommOfPooledLeakAccepted_honest`: the two inputs that
`Accounting.Guarantees.Bridge.CommOfPooledLeakAccepted` takes at each execution phase, the miss probability given the
earlier phases' coins and the leak relation at the witness's incorrect set, hold together at one run that realizes the
network certifier's monitoring period (`honestPooled`).

The run has two execution phases of one step each. Its circuit negates its one ingress bit, its operator sends
`false`, and the network certifier's record at each phase is the frame the circuit then puts on the link
(`frameRecord`), so the record meets the leak relation with no incorrect proof block (`leakAt_honest`). The network
audit accepts when both phases' records meet that relation, and its claim is the period's good event at the record.
No proof block is incorrect, so the miss probability is bounded by `1` and the leak relation fails with probability
`0`. The network certifier sees no connection, and its code is run as written.
-/

@[expose] public section

namespace Accounting.SecurityProofs.Bridge

open Claims Claims.Sampling NetTiming Accounting.JointBridge
open Accounting.PouwWitness Accounting.NetWitness
open NetTiming.Witness (notLay obs1 notLay_fits notProg_val)
open Claims.Witness.SamplingGeneric (notProg notStmt notPart)

/-- The network certifier's record of two execution phases of one step each: a frame on the link at each. -/
def frameRecord : Fin 2 → List (Obs 1 1 1) := fun _ => [obs1]

/-- At ingress `false` the circuit that negates its ingress puts a frame of no bits on its link. -/
theorem notProg_row (a : ℕ → Bool) (ℓ σ : Fin 1) :
    rowOf (valOver notProg (notLay.inputOf 1 [[false]] a) (fun _ => none)) (notLay.egress 0 ℓ) σ = some [] := by
  have hin : notLay.inputOf 1 [[false]] a 0 = false := rfl
  have hrow : rowOf (valOver notProg (notLay.inputOf 1 [[false]] a) (fun _ => none)) (notLay.egress 0 ℓ) σ =
      if valOver notProg (notLay.inputOf 1 [[false]] a) (fun _ => none) 1 then
        some (([] : List ℕ).map (valOver notProg (notLay.inputOf 1 [[false]] a) (fun _ => none)))
      else none := rfl
  rw [notProg_val _ hin] at hrow
  exact hrow.trans (by simp)

/-- The circuit's Program accepts the one-step record of that frame, at the operator that sends `false`, with no gate
held. -/
theorem accepted_obs1 (a : ℕ → Bool) :
    Accepted (circuitProgram notProg notLay 1) (fun _ _ => [false]) 1 [obs1] (a, fun _ => none) := by
  intro t ht
  obtain rfl : t = 0 := by omega
  refine ⟨Nat.one_pos, ?_, fun _ => rfl⟩
  funext ℓ σ
  exact (notProg_row a ℓ σ).symm

/-- **Each execution phase's record meets the leak relation with no incorrect proof block**: the layout fits the
circuit, the record has the phase's one step and jitter `0`, and the circuit's Program accepts it at the advice
`false` with no gate held. -/
theorem leakAt_honest (i : Fin 2) :
    LeakAt (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 notStmt notPart (fun _ => false) (fun _ _ => [false]) ∅
      (frameRecord i) := by
  refine ⟨notLay_fits, rfl, fun y hy => ?_, fun _ => false,
    ⟨fun g hg _ => absurd hg (Finset.notMem_empty g), fun _ _ => rfl⟩, fun _ => none,
    fun _ _ h => absurd h.symm (Option.some_ne_none _), accepted_obs1 _⟩
  obtain rfl : y = obs1 := by simpa [frameRecord] using hy
  exact Nat.one_pos

/-- A period run on `nullDevice` over `emptyCircuit`, idle, on the coins of two verification phases: its every audit
accepts when both execution phases' records meet the leak relation with no incorrect proof block, and the network
certifier's claim is the period's good event at that record. -/
def honestNet : PeriodRun nullDevice Unit emptyCircuit (fun _ : Unit => Fin 2 → Bool) where
  units := 0
  parts := ⟨fun g => g.elim0, fun g => g.elim0⟩
  own _ := ()
  acct _ := ()
  win := ⟨0, 1⟩
  schedule _ _ := idleStep
  fits _ := by
    intro t
    simp [Step.Fits, idleStep, nullDevice]
  trace _ g := g.elim0
  commWithin _ _ := PooledGood (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 (fun _ => notStmt) (fun _ => notPart)
    (fun _ _ => false) (fun _ _ _ => [false]) 1 (fun _ => ∅) frameRecord
  accepts _ _ := ∀ i, LeakAt (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 notStmt notPart (fun _ => false)
    (fun _ _ => [false]) ∅ (frameRecord i)

/-- The record format of a network certifier that sees no connection and writes `frameRecord`. -/
def honestFmt : RecordFormat Empty Unit (Fin 2 → List (Obs 1 1 1)) where
  record _ _ _ := frameRecord
  read _ _ c := c.elim
  fits _ := True
  read_record _ _ _ := funext fun c => c.elim

/-- **`honestNet` realizes the network certifier's monitoring period of two sampled audits, pooled**: the audit's
coins are the two verification phases' coins, the network certifier's device runs its code as written, each
verification phase's audit accepts when its execution phase's record meets the leak relation with no incorrect proof
block, and no proof block is incorrect. -/
def honestPooled : RealizesCommOfPooledLeakAccepted honestNet 0 1 1 1 (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 2 1
    where
  sw _ := notStmt
  Pw _ := notPart
  ρ _ _ := false
  op _ _ _ := [false]
  Coin := Bool
  coins := Equiv.refl _
  Conn := Empty
  Cell := Unit
  Srv := Unit
  Dev := Unit
  arr _ c := c.elim
  fmt := honestFmt
  device := ()
  run _ p := recordingCode honestFmt p.1 fun c => c.elim
  trusted _ := rfl
  sched _ := ⟨0, 0, Empty.elim, Empty.elim⟩
  srv _ := ()
  fits _ := trivial
  acc _ i _ _ := LeakAt (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 notStmt notPart (fun _ => false)
    (fun _ _ => [false]) ∅ (frameRecord i)
  acc_prefix _ _ _ _ _ := rfl
  W _ _ _ := ∅
  W_prefix _ _ _ _ _ := rfl
  accepts_imp _ _ h := h
  comm_iff _ _ := Iff.rfl

/-- **`CommOfPooledLeakAccepted`'s two inputs hold together at `honestPooled`**, at detection probability `½`: no
proof block is incorrect, so the miss probability given the earlier phases' coins is bounded by `(1 − ½)^0 = 1`, and
each execution phase's record meets the leak relation at the empty incorrect set, so the leak relation fails with
probability `0`. -/
theorem CommOfPooledLeakAccepted_honest :
    PeriodEscapes Set.univ honestPooled.acc honestPooled.W (1 / 2) (fun _ => 0) ∧
      PhaseLeaks (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 honestPooled.sw honestPooled.Pw honestPooled.ρ
        honestPooled.op Set.univ honestPooled.acc honestPooled.W honestPooled.record (fun _ => 0) := by
  refine ⟨fun ξ _ i ω => ?_, fun ξ _ i => ?_⟩
  · refine (Pouw.pr_le_one _).trans (le_of_eq ?_)
    show (1 : ℝ) = (1 - 1 / 2) ^ (∅ : Finset ℕ).card + 0
    rw [Finset.card_empty, pow_zero, add_zero]
  · exact le_of_eq (Pouw.pr_eq_zero fun ω h => h.2 (leakAt_honest i))

end Accounting.SecurityProofs.Bridge
