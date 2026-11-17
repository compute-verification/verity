module

public import Proofs.Pouw.TS.Session.Held.Layer
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

/-! `OkRealizes`, the premise that a session's key program computes, on its `ok` wires, the checks of one unit of the
registered circuit `Cd`. -/

@[expose] public section

namespace ResourceAccounting.Assumptions.TS

open ResourceAccounting.TS ResourceAccounting.TS.Held Flock.SoundAtBridge

/-- `accounting-ts-ok-realizes` (the checked unit's circuit is its declared unit's). **The key
program's `ok` wires carry `Cd`'s checks**: at the session `s`, for every wire assignment `vals` that the key
program's evaluation (`Circuit.eval`) agrees with on every wire of unit `u`, and whose `ok` wires of `u` (`OkWire`)
each carry the bit of a value that is the `ok` row at its position (`OkVal`, the one word `1`), every check of the
registered circuit `Cd`'s unit `unitOf u` holds at `dec vals` (`ResourceAccounting.Served.Held`).

It is a fact of one compiled circuit and one registered circuit, with `unitOf` and `dec` the map between them: that the
key program of `PouwH1TileZy_v1`'s Flock circuit computes its `ok` word as the conjunction of the checks `Cd`
declares of the unit (the Σ check and the dequant check), on the wires `dec` reads. Nothing in Lean derives a Flock
circuit from a `Declared` one or checks one against the other; `circuit-check` binds the Definition to its Boolean
lowering, not to `Cd`. It stays a premise until that binding is stated. -/
def OkRealizes {ι : Type} (s : RowsSession) (ok : String) (Cd : ResourceAccounting.Compose.Declared ι)
    (cls : ℕ → ResourceAccounting.Served.Cls) (unitOf : Fin s.n → ℕ) (dec : (Fin s.Circ.N → Bool) → ι → ℤ) : Prop :=
  ∀ (vals : Fin s.Circ.N → Bool) (u : Fin s.n), (∀ g ∈ s.P.io u, s.Circ.eval vals g = vals g) →
    (∀ g, OkWire s ok u g → ∃ w, OkVal s (s.vb.pos g) w ∧ vals g = s.vb.bit w g) →
    ResourceAccounting.Served.Held Cd cls (unitOf u) (dec vals)

end ResourceAccounting.Assumptions.TS
