module

public import Proofs.Flock.Claims.ZK.FinExe
public import Proofs.Flock.Claims.ZK.FinPads
public import Proofs.Flock.Claims.ZK.FinTau
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
# `FinExe` from the hybrids, the live end and `InnerTauVal`

`repOpenRest` proves `RepOpenRest` from its two halves: level 0's opened rows and salts (`repOpenRest_l0`) and the
pads' opened columns and salts (`repOpenRest_pads`). With `RepTau` from `InnerTauVal` (`repTau_of`), `finExe_of_rest`
gives `finExe_of_inner`.
-/

@[expose] public section

namespace Claims.ZKMal

namespace Masked

/-- **The final-only data at opened positions read from the views is the session's** (`RepOpenRest`): level 0's
opened rows and salts (`repOpenRest_l0`) and the pads' opened columns and salts (`repOpenRest_pads`). -/
theorem repOpenRest : RepOpenRest := by
  intro x w τ c spec i hi r cm0 extras ic l hC p q t l' hp hq hz
  obtain ⟨h1, h2⟩ := repOpenRest_l0 x w τ c spec i hi r cm0 extras ic l hC p q t l' hp hq hz
  obtain ⟨h3, h4⟩ := repOpenRest_pads x w τ c spec i hi r cm0 extras ic l hC p q t l' hp hq hz
  exact ⟨h1, h2, h3, h4⟩

/-- **`FinExe` from the hybrids, the live end and `InnerTauVal`**: `finExe_of_rest` with `RepTau` from `InnerTauVal`
(`repTau_of`) and `RepOpenRest` proved (`repOpenRest`). -/
theorem finExe_of_inner (hL : LiveEnd) (hSC : SimCommit) (h0 : SimPiop) (h1 : SimLig) (h2 : SimInner)
    (hT : InnerTauVal) : FinExe :=
  finExe_of_rest hL hSC h0 h1 h2 (repTau_of hT) repOpenRest

end Masked

end Claims.ZKMal
