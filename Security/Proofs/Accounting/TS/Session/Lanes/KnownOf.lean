module

public import Proofs.Accounting.TS.Session.Lanes.Decode
public import PouwWindow.KnownRows
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
# The known content a leaf's sessions are handed

`knownOf P salts k`: statement `k`'s known rows at the public inputs `P` (`PouwWindow.Lanes.knownRows`, held to the
Python that writes them by its difftest): lane `l`'s `admit` row (`P.admit`), its `pub` row (`P.salt`, `P.position`,
`P.roots` and the index `l`, each least significant bit first at its field's width) and its `o0` row 1, and the
combine's `o0`, each under its salt `salts`. `Accounting.Guarantees.TS.LeafLanesSound` hands each session this content,
so its known-content check (`Flock.Registered.checkKnownAll`) binds the path's public inputs.
-/

@[expose] public section

namespace Accounting.TS.Lanes

/-- **Statement `k`'s known content at the public inputs `P`**: `PouwWindow.Lanes.knownRows` at `P`'s fields, lane `l`'s
(`some l`) or the combine's (`none`), each row under its salt `salts`. -/
def knownOf (P : LanePub) (salts : String → ByteArray) (k : Option (Fin 32)) : Known :=
  PouwWindow.Lanes.knownRows (List.ofFn P.admit) P.salt P.position P.roots salts (k.map Fin.val)

end Accounting.TS.Lanes
