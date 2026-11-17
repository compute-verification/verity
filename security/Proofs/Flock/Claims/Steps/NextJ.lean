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
public import Proofs.Flock.Claims.Steps.NextCs
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

/-! # What the plain KS proofs read of `Claims.Recursion.NextJ`, without its recursion imports. -/

@[expose] public section

namespace FlockDeploy

open Flock

open Lean (Json)

/-- **The prover's last message at the parsed value**: the statement, the record's parsed value and the proof files. -/
structure FinalJ where
  stmt : Claims.Msg
  j : Json
  proofs : Array ByteArray

/-- The record's sample at its parsed value, rebuilt (`recordDraw` after the parse). -/
def sampleOfJ (j : Json) : Option Json :=
  match j.getObjVal? "unit_draw" with
  | .ok Json.null => none
  | .ok d => some (Flock.norm d)
  | .error _ => none

/-- The checks after the verdict at the parsed value: the statement, the sample at `f.j`, custody at `j'`. -/
def checksAtJ (s : Ins) (spec : SessionSpec) (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (f : FinalJ) (j' : Json) : Flock.V Unit := do
  Record.check (f.stmt == x) "live: the last message's statement is not the statement"
  Record.check (sampleEq (sampleOfJ f.j) s.read.draw) "live: the record's draw is not the statement's"
  custody spec (← liveOfJ spec j') link c₁ mids
  s.read.check
  if s.read.draw.isNone then ProvedScope.sampleDesign false s.read.tags false none

/-- **The decision at a session at the parsed value**: the plain verdict and custody's reading at the coin-substituted
value. -/
def decisionAtJ (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : FinalJ) :
    Claims.OCV Unit := do
  verifyJWithM Claims.Hashes.oc spec (tabsOf sts) (coinSubJ spec c₁ mids f.j) f.proofs
  monadLift (checksAtJ s spec x link c₁ mids f (coinSubJ spec c₁ mids f.j))

end FlockDeploy
