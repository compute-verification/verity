module

public import Proofs.Flock.Claims.Recursion.DeployedR
public import Proofs.Flock.Claims.Recursion.HonestPartsLayR
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
# V\*'s outer witnesses at the layout of record (`layR`): what `honest_partAccepts_layR` asks of them

`honest_partAccepts_layR` takes four facts about the outer witness the reference prover hands part `s`'s session
(`RefParts.wit`): the calls `layR` records beyond `layoutBTops`' (`hTm`), the rounds it holds whole (`hh`), its pieces
(`hp`) and the shared rows it reads (`hr`).

* `hr` holds at every witness and messages (`rowsRead_layR`): `layoutBTops` places no shared row (`rowAt` is
  `LayoutB`'s default) and `layR` changes only the calls, so no part holds one.
-/

@[expose] public section

namespace RecDeploy

open Flock
open Claims (Msg)

/-- **`layR` places no shared row**, so every part reads the shared rows where the messages carry them, at every
witness and messages (`LayoutB.RowsRead`): `layoutBTops` leaves `rowAt` at `LayoutB`'s default and `layR` changes only
the calls. -/
theorem rowsRead_layR (elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))) (x : Msg)
    (s : ℕ) (w : Claims.Flock.Wit) (ms : List Msg) : (DeployedR.layR elem).RowsRead x s w ms := by
  intro r v i o h _
  have h0 : (DeployedR.layR elem).rowHeld x s w r = none := rfl
  rw [h0] at h
  cases h

end RecDeploy
