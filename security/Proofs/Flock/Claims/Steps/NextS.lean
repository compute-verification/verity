module

public import Proofs.Flock.C3.Reps.AtList
public import Proofs.Flock.C3.Reps.RoundLinks
public import Proofs.Flock.C3.VStarList.Accepts
public import Proofs.Flock.Claims.Complete.Json.UnitSample
public import Proofs.Flock.Claims.Complete.Walk
public import Proofs.Flock.Claims.Complete.ZkTables.Scope
public import Proofs.Flock.Claims.KS.PlainKSLin
public import Proofs.Flock.Claims.KS.ZkOuter
public import Proofs.Flock.Claims.Sampling.Bext
public import Proofs.Flock.Claims.Steps.NextP
public import Proofs.Flock.Claims.Steps.NextSha
public import Proofs.Flock.Claims.Steps.NextShaDec
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

/-! # What the plain KS proofs read of `Claims.Recursion.NextS`, without its recursion imports. -/

@[expose] public section

namespace FlockDeploy

open Flock

open Lean (Json)

/-- What `nextS` does at the prover's messages. -/
inductive ActS where
  | rej
  | coin (c : Coin)
  | decide (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (link : List ByteArray × List ByteArray) (c₁ : Coin)
      (mids : List (Claims.Msg × Coin)) (f : FinalP)

theorem outOf_acc_iff (H : Claims.Msg → Claims.Msg) (d : Claims.OCV Unit) :
    ((outOf d).eval H).1 = .acc ↔ (d.run.eval H).1 = .ok () := by
  simp only [outOf, Claims.OC.eval_bind, Claims.OC.eval]
  cases hr : (d.run.eval H).1 with
  | ok u =>
    cases u
    exact ⟨fun _ => rfl, fun _ => rfl⟩
  | error e => exact ⟨fun h => by first | cases h | simp at h, fun h => by cases h⟩

/-- `decisionSF` run against `H`: the setup at `hOf H`, then `decisionAtS` at its session. -/
theorem decisionSF_eval (H : Claims.Msg → Claims.Msg) (s : Ins) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : FinalP) :
    ((decisionSF s x link c₁ mids f).run.eval H).1 =
      (setupWith (Claims.hOf H) s.read >>= fun v => ((decisionAtS s v.1 v.2 x link c₁ mids f).run.eval H).1) := by
  unfold decisionSF
  rw [evRun_bind, (setupM_eval H s.read).1]

end FlockDeploy
