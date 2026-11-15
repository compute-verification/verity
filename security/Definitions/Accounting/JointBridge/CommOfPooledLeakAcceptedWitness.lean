module

public import Definitions.Accounting.JointBridge.CommOfPooledLeakAccepted
public import Definitions.Accounting.NetWitness
public import Definitions.NetworkCertifier.Witness
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

/-!
# An honest run of the network certifier's monitoring period, pooled (trusted definitions; used only by the
satisfiability witnesses)

`honestPooled`: a run that realizes the network certifier's monitoring period of two execution phases
(`RealizesCommOfPooledLeakAccepted`), at which the two inputs of
`Accounting.Guarantees.Bridge.CommOfPooledLeakAccepted` hold together.

Each execution phase is one step of the circuit that negates its one ingress bit (`negStmt`, on the layout
`NetTiming.Witness.notLay`). The period's operator sends `false` at every step of every execution phase, whatever the
phases closed before it observed, and the network certifier's record of each phase is the frame that circuit then
puts on the link (`frameRecord`). The period run (`honestNet`) is idle, on the device with no capacity over the
circuit with no gate, as `acceptAllNet` is; its network audit accepts when both phases' records meet the leak relation
with no incorrect proof block, and its claim is the monitoring period's good event at that record. The network
certifier sees no connection and its run is its code's (`honestFmt`), so `trusted-host` holds there by definition.
-/

@[expose] public section

namespace Accounting.NetWitness

open PouwWitness Claims Claims.Sampling NetTiming Accounting.JointBridge
open NetTiming.Witness (notLay)

/-- The registration of the circuit that negates its input: one input gate and its negation, the negation returned,
sampled `all`. -/
def negStmt : AuditStatement := ⟨[], ⟨[.input, .not 0], [1]⟩, ⟨"", 0, []⟩, ⟨[.all]⟩, []⟩

/-- Its partition: one proof block, of both gates. -/
def negPart : Partition := ⟨[[[[0, 1]]]]⟩

/-- One step as the network certifier observes it: a frame on the link, no jitter, every status complete. -/
def frameObs : Obs 1 1 1 := ⟨fun _ _ => some [], 0, fun _ => .complete⟩

/-- The network certifier's record of two execution phases of one step each: at each, a frame on the link. -/
def frameRecord : Fin 2 → List (Obs 1 1 1) := fun _ => [frameObs]

/-- A period run on `nullDevice` over `emptyCircuit`, idle, on the coins of two verification phases: its every audit
accepts when both execution phases' records meet the leak relation with no incorrect proof block, and the network
certifier's claim is the monitoring period's good event at that record, at the period's operator that sends `false`
whatever the earlier phases observed. -/
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
  commWithin _ _ := PooledGood (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 (fun _ => negStmt) (fun _ => negPart)
    (fun _ _ => false) (fun _ _ _ _ => [false]) 1 (fun _ => ∅) frameRecord
  accepts _ _ := ∀ i, LeakAt (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 negStmt negPart (fun _ => false)
    (fun _ _ => [false]) ∅ (frameRecord i)

/-- The record format of a network certifier that sees no connection and writes `frameRecord`. -/
def honestFmt : RecordFormat Empty Unit (Fin 2 → List (Obs 1 1 1)) where
  record _ _ _ := frameRecord
  read _ _ c := c.elim
  fits _ := True
  read_record _ _ _ := funext fun c => c.elim

/-- **`honestNet` realizes the network certifier's monitoring period of two sampled audits, pooled**: the audit's
coins are the two verification phases' coins, the network certifier's device runs its code as written, the period's
operator sends `false` and reads no execution phase, each verification phase's audit accepts when its execution
phase's record meets the leak relation with no incorrect proof block, and no proof block is incorrect. -/
def honestPooled :
    RealizesCommOfPooledLeakAccepted honestNet 0 1 1 1 (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 2 1 where
  sw _ := negStmt
  Pw _ := negPart
  ρ _ _ := false
  op _ _ _ _ := [false]
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
  acc _ i _ _ := LeakAt (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 negStmt negPart (fun _ => false)
    (fun _ _ => [false]) ∅ (frameRecord i)
  acc_prefix _ _ _ _ _ := rfl
  W _ _ _ := ∅
  W_prefix _ _ _ _ _ := rfl
  accepts_imp _ _ h := h
  comm_iff _ _ := Iff.rfl

end Accounting.NetWitness
