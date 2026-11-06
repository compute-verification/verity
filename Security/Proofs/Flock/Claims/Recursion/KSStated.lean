module

public import Proofs.Flock.Claims.Recursion.KS
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
# The recursion's KS at its stated errors

The recursion's claims state `γ` and `γ_e` at the errors `KS_of` proves (`γ = γ_A`, `γ_e = γ_eA`), so the KS that
`KS_of` proves is the claim: `KSA` is `KS` (`KSA_iff_KS`), and `KSFlockA` is `KSFlock` (`KSFlockA_iff_KSFlock`).
-/

@[expose] public section

namespace Claims.Recursion

theorem KSA_iff_KS {Coin W Col T : Type} [Fintype Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]
    (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (DO : Flock.Deploy Coin Col T) :
    KSA H D I DO ↔ KS H D I DO :=
  Iff.rfl

theorem KSFlockA_iff_KSFlock {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    [DecidableEq Col] [DecidableEq Col'] [Zero Col] [Zero Col'] (H : Msg → Msg) (D : Deploy Coin)
    (DI : Flock.Deploy Coin Col T) (DO : Flock.Deploy Coin Col' T') :
    KSFlockA H D DI DO ↔ KSFlock H D DI DO :=
  Iff.rfl

end Claims.Recursion
