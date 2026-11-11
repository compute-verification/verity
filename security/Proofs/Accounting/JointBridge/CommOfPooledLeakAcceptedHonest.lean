module

public import Definitions.Accounting.Guarantees.JointBridge.CommOfPooledLeakAccepted
public import Definitions.Accounting.JointBridge.CommOfPooledLeakAcceptedWitness
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
network certifier's monitoring period (`Accounting.NetWitness.honestPooled`, among the definitions).

The run has two execution phases of one step each. Its circuit negates its one ingress bit, the period's operator
sends `false` whatever the phases closed before it observed, and the network certifier's record at each phase is the
frame the circuit then puts on the link (`frameRecord`), so the record meets the leak relation at that operator with
no incorrect proof block (`leakAt_honest`). The network audit accepts when both phases' records meet that relation,
and its claim is the period's good event at the record. No proof block is incorrect, so the miss probability is
bounded by `1` and the leak relation fails with probability `0`. The network certifier sees no connection, and its
code is run as written.
-/

@[expose] public section

namespace Accounting.SecurityProofs.Bridge

open Claims Claims.Sampling NetTiming Accounting.JointBridge Accounting.NetWitness
open NetTiming.Witness (notLay obs1 notLay_fits notProg_val)
open Claims.Witness.SamplingGeneric (notProg)

/-- `negStmt` is the witnesses' `notStmt`, restated among the definitions: an edit to either side fails this file. -/
theorem negStmt_eq_notStmt : negStmt = Claims.Witness.SamplingGeneric.notStmt := rfl

/-- `negPart` is the witnesses' `notPart`, restated among the definitions: an edit to either side fails this file. -/
theorem negPart_eq_notPart : negPart = Claims.Witness.SamplingGeneric.notPart := rfl

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
`false` with no gate held. The registration `negStmt` and the frame `frameObs` are the witnesses' `notStmt` and
`obs1`, written out among the definitions. -/
theorem leakAt_honest (i : Fin 2) :
    LeakAt (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 negStmt negPart (fun _ => false) (fun _ _ => [false]) ∅
      (frameRecord i) := by
  refine ⟨notLay_fits, rfl, fun y hy => ?_, fun _ => false,
    ⟨fun g hg _ => absurd hg (Finset.notMem_empty g), fun _ _ => rfl⟩, fun _ => none,
    fun _ _ h => absurd h.symm (Option.some_ne_none _), accepted_obs1 _⟩
  obtain rfl : y = frameObs := by simpa [frameRecord] using hy
  exact Nat.one_pos

/-- **`CommOfPooledLeakAccepted`'s two inputs hold together at `honestPooled`**, at detection probability `½`: no
proof block is incorrect, so the miss probability given the earlier phases' coins is bounded by `(1 − ½)^0 = 1`, and
each execution phase's record meets the leak relation at the empty incorrect set and at the phase's operator, which
sends `false` at every record of the earlier phases, so the leak relation fails with probability `0`. -/
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
