module

public import Definitions.Pouw.Dither
public import Definitions.Pouw.DitherInstance
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

/-! The headline's causality premises on the cards, which the verdict can't check because each bounds a signed time by a program's wall time in the model, and the period's inventory premise. -/

@[expose] public section

namespace Pouw.Assumptions

open Pouw.Dither

/-- `pouw-root-after-coin` (proposed; physical causality, the node within the model). **A commitment arrives no
sooner than the work that computed it**: sampling block `b`'s work reads the dither coins the challenger issues at
`t ob b` (its signed issue time, as the verdict reads it from the readings `ob = dev A o x`), so the network
attester, watching the replica's boundary, can't see the block's root arrive (`arrive ob b`, its signed time)
before `t ob b` plus the work's wall time in the model. The challenger's and the certifier's clocks are the
auditor's devices. That the node takes at least the model's wall time is the node's being within the model, not a
property of the clocks. -/
def RootAfterCoin (G : Accel) (P : RoundProto) (nsb : ℕ) {Obs : Type} (dev : Readings G P nsb Obs)
    (t arrive : Obs → Fin nsb → ℝ) : Prop :=
  ∀ (A : Prover G P nsb) (o : P.Orc) (x : Fin nsb → P.Draw × P.Chk) (b : Fin nsb),
    t (dev A o x) b + G.wall (A.work b) (A.workIn o x b) ≤ arrive (dev A o x) b

/-- `pouw-commit-after-work` (proposed; physical causality, the node within the model). **An operand root arrives
no sooner than the commit phase that computed it**: sampling block `b + 1`'s commit phase reads the state after
block `b`'s work, which reads the dither coins the challenger issues at `t ob b`, so the network attester,
watching the replica's boundary, can't see block `b + 1`'s operand root arrive (`opArrive ob (b + 1)`, its signed
time) before `t ob b` plus the work's and the commit phase's wall times in the model. The first block's commit
phase starts no sooner than the period, at time 0: what a prover computes before the period is its start state,
fixed before it. The clocks are the auditor's devices, and the node's taking at least the model's wall time is the
node's being within the model.

Both pieces are wall time: the verdict checks the order of the signed times (`opArrive b ≤ t b`, `arrive b ≤ t b'`),
but not that the commit phase between the previous block's work and the operand root took its wall time, which only
the prover's programs determine. `TimedWorkFits` needs both: the first to keep block 0's commit phase inside the
period, the second to keep each commit phase after the previous block's work. -/
def CommitAfterWork (G : Accel) (P : RoundProto) (nsb : ℕ) {Obs : Type} (dev : Readings G P nsb Obs)
    (t opArrive : Obs → Fin nsb → ℝ) : Prop :=
  ∀ (A : Prover G P nsb) (o : P.Orc) (x : Fin nsb → P.Draw × P.Chk),
    (∀ b : Fin nsb, (b : ℕ) = 0 → G.wall (A.commit b) (A.commitIn o x b) ≤ opArrive (dev A o x) b) ∧
    ∀ b b' : Fin nsb, (b : ℕ) + 1 = b' →
      t (dev A o x) b + G.wall (A.work b) (A.workIn o x b) + G.wall (A.commit b') (A.commitIn o x b') ≤
        opArrive (dev A o x) b'

/-- `pouw-answer-after-work` (proposed; physical causality, the node within the model). **A selected answer
arrives no sooner than its block's dither coins plus its own run**: sampling block `b`'s answer reads the state
after the block's work, which reads the dither coins issued at `t ob b`, so the network attester can't see a
selected block's answer arrive (`ansAt ob b`) before `t ob b` plus the answer's wall time in the model. It isn't
anchored at the check coin: the answer's run may start at the block's root, before the check coin is issued, and a
deadline from the check coin bounds it only through this. An unselected block's answer never runs.

It is the one piece of the answer's timing that is wall time: the verdict checks that the dither coins were issued
inside the period and the answer arrived within `τ_ans` of the check coin, before the period's end (`Timed`), but not
how long the answer's program ran, which only the prover's programs determine. That is all `AnswersFit` reads: the
work's wall time before the answer isn't needed, so it isn't assumed. -/
def AnswerAfterWork (G : Accel) (P : RoundProto) (nsb : ℕ) {Obs : Type} (dev : Readings G P nsb Obs)
    (sel : P.Chk → Prop) (t ansAt : Obs → Fin nsb → ℝ) : Prop :=
  ∀ (A : Prover G P nsb) (o : P.Orc) (x : Fin nsb → P.Draw × P.Chk) (b : Fin nsb), sel (x b).2 →
    t (dev A o x) b + G.wall (A.answer b) (A.answerIn o x b) ≤ ansAt (dev A o x) b

/-- `pouw-period-inventory`, proposed and read by no guarantee: every processor the period's cards reach without crossing a
network attester is a device of the period, and what they compute within `t` seconds some program of `G` computes
within `t` seconds, exchanging nothing. -/
def MonitoringPeriodInventory (G : Accel) (can : ∀ {I O : Type}, (I → O) → ℝ → Prop) : Prop :=
  ∀ {I O : Type} (F : I → O) (t : ℝ), can F t →
    ∃ f : G.Mc.Alg I O, ∀ i, G.Mc.eval f i = F i ∧ G.wall f i ≤ t ∧ G.ext f i = 0

end Pouw.Assumptions
