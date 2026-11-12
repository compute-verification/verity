module

public import Proofs.Flock.Claims.Recursion.Assumptions.HonestFixture
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
import Proofs.Flock.Claims.Recursion.HonestHalfTops

/-!
# The honest half at the honest run (`honest_half_fixture`)

`honest_half_tops` at the one run `HonestFixture` names: some deployed recursive run at some statement `x`, with the
deployed inner prover `IrDep x`, has every part `s < nS x` accepting (`PartAccepts` at `RecursiveKSAt`'s record and
layout), the honest witness of `RecursiveKS`'s non-vacuity, with every premise in the one closed assumption.
-/

@[expose] public section

namespace RecDeploy.HonestHalfFixture

open _root_.Flock
open Claims (Msg Out)
open FlockDeploy (Coin)
open RecDeploy.VStarBuild
open RecDeploy.VStarHonest
open RecDeploy.HonestHalf

/-- **At the honest run, every V\* part accepts**: `HonestFixture`'s run, at the deployed brings at `x` with the
deployed inner prover `IrDep x` under `deployedB` and V\*'s layout of record `layR elemR`, has `PartAccepts` at every
part `s < nS x` (`honest_half_tops`). -/
theorem honest_half_fixture (h : HonestFixture) :
    ∃ (mask : ℕ → ℕ → Bool) (b0 : Brings) (x : Msg) (aux : Msg → ℕ → Flock.RecRecord.Round)
      (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
      (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace)
      (w : Claims.Flock.Wit) (ω : Tape (bLAt mask (IrDep x) b0 x aux nOf hd recRest))
      (r : Fin ((DAt mask (IrDep x) b0 x aux nOf hd recRest).V.k x) → Coin),
      ∀ s < nS x,
        Claims.Recursion.VStarB.PartAccepts FlockDeploy.sha512 (DAt mask (IrDep x) b0 x aux nOf hd recRest)
          (deployZkRec _root_.Flock.Readable.relation) (DeployedR.layR OpenRows.elemR) Claims.Recursion.Form.Dense
          (showsPart recRead
            ({ bringsAt mask (IrDep x) (deployedB b0) x with tr := recTr aux nOf hd recRest } : Brings))
          (cmtTr recRead) s x (Claims.coinSeq r)
          ((Claims.run FlockDeploy.sha512 (refP (IAt decI).V (bLAt mask (IrDep x) b0 x aux nOf hd recRest) w)
            (DAt mask (IrDep x) b0 x aux nOf hd recRest).V x ω r).msgs.map Prod.fst) := by
  obtain ⟨mask, b0, x, aux, nOf, hd, recRest, decI, w, st, ω, r, hR, hW, hN, hT⟩ := h
  exact ⟨mask, b0, x, aux, nOf, hd, recRest, decI, w, ω, r,
    HonestHalfTops.honest_half_tops mask b0 x aux nOf hd recRest decI ω r hR hW hN hT⟩

end RecDeploy.HonestHalfFixture
