module

public import Mathlib.Algebra.Order.Archimedean.Real.Basic
public import Mathlib.Data.Finset.Card
public import Mathlib.Data.Fintype.Basic
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
# Slot writes, heads and a steered consumer (trusted)

The counting objects of EPS1-TENSOR's T1 (§3), EPS1-SPEC's Lemma 2 (§1.3) and
SAMPLING's LIN-STEER:

* a tensor run's slot writes: an instruction of form `i` writes `w i` slots and holds its pipe `c i` cycles, and the
  family's best slots per pipe cycle is `ρ_T = max_i w i / c i` (`rhoT`);
* a straight-line program's head writes: instance `t` of class `cls t` writes at most `ω (cls t)` words, and each
  committed position's value traces back through copies to one written word, its head write (`HeadWrite`);
* a consumer dot product of a lever row, with its noise-active coordinates and its least work once some of them are
  zeroed (`Consumer`).
-/

@[expose] public section

namespace Accounting.TS

open Finset

/-- The slot writes of a tensor run with `n i` instructions of form `i`, each writing `w i` slots. -/
abbrev SlotWrite (ι : Type) (w n : ι → ℕ) : Type := Σ i : ι, Fin (n i) × Fin (w i)

/-- `ρ_T = max_i w(i)/c(i)`: the family's best slots per tensor-pipe cycle. -/
noncomputable def rhoT {ι : Type} [Fintype ι] (w c : ι → ℕ) : ℝ := ⨆ i, (w i : ℝ) / c i

/-- The head writes of a straight-line program of `G` instances, instance `t` of class `cls t` writing at most
`ω (cls t)` words. -/
abbrev HeadWrite {K : Type} (ω : K → ℕ) (G : ℕ) (cls : Fin G → K) : Type := Σ t : Fin G, Fin (ω (cls t))

/-- `n_k`: the program's instances of class `k`. -/
def classCount {K : Type} [DecidableEq K] {G : ℕ} (cls : Fin G → K) (k : K) : ℕ :=
  (univ.filter fun t => cls t = k).card

/-- **A consumer dot product** of a lever row of `m` coordinates: the row's noise-active coordinates `act`, and
`cost Z`, the consumer's least work (MACs, at its cheapest route) once the coordinates in `Z` are zeroed;
`cost ∅` is its whole work `C`. -/
structure Consumer (m : ℕ) where
  act : Finset (Fin m)
  cost : Finset (Fin m) → ℝ

end Accounting.TS
