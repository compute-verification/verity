module

public import Proofs.Flock.Claims.ZK.SplitInner
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
# `InnerSlot` and `InnerTriple`

Two of `SplitInner`'s statements, each from its `SimInner` lemma at `InnerAt`'s run:

* **`innerSlot_holds`**: the shared phases end at the rep's `ε` slot (`innerPre_slot`).
* **`innerTriple_holds`**: the cut's `ρ_in`, `σ_in` are the session's at the honest pads (`innerTripleE`).
-/

@[expose] public section

namespace Claims.ZKMal

open Claims FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

namespace Masked

theorem innerSlot_holds : InnerSlot := by
  rintro x w τ c n j r cons L ⟨-, -, -, s, spec, hs, dom, cm0, ic, extras, he, l, hl, o, t', l', ho, -, rfl⟩
  exact innerPre_slot hs w τ c n j r dom cm0 ic extras he hl ho

theorem innerTriple_holds : InnerTriple := fun x w τ c _ j r _ _ _ _ => innerTripleE x c w τ j r rfl

end Masked

end Claims.ZKMal
