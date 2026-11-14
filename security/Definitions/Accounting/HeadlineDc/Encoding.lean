module

public import Definitions.Accounting.Headline.Encoding
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
# Exact data for a data-centre headline instance (trusted)

`Definitions.Accounting.Headline.Encoding`'s encoding, with the two things the H200's and B200's instances
(`benchmarks/pouw/lean_headline_dc.py`) need beside it:

* `DesignData.modelFill`: `DesignData.model` with a filler, its menu the one route the floor program's filler runs
  on, whose demand on each facet is the load of its counts a unit of honest work (the H200's ffill, which FILL-OPT
  prices at `χ*`);
* `cert`: a certificate over `Fin m` from sparse weights, for a certificate on more than one facet;
* `SheetDataQ`: `SheetData` with a rational pipe width, for a pipe no document rates whose width is the census's
  measured line (the uniform datapath's UP line, `DS_uniform`: 128 over 1.96110716 SMSP-cycles a uniform instruction).
-/

@[expose] public section

namespace Accounting.TS.Headline

open Accounting.TS

/-- **The design's TS model on machine `G` with a filler**: `DesignData.model`, its filler menu one route whose demand
is the load on `G` of `fill`, the filler's counts a unit of honest work. -/
noncomputable def DesignData.modelFill {R : Type} (D : DesignData) (n nt nf : ℕ) [NeZero n] (G : Machine (Fin n) R)
    (fill : Sparse) : Model (Fin n) R (Fin nt) Unit Empty (Fin nf) Unit :=
  { D.model n nt nf G with fill := fun _ => G.load (Sparse.counts fill) }

/-- A certificate over `Fin m` from `(facet index, numerator, denominator)` weights. -/
noncomputable def cert (m : ℕ) (v : Sparse) : Fin m → ℝ := Sparse.counts v

/-- **A data sheet as lists by class index, its pipe widths rational**: `SheetData`'s fields with `width : P → ℚ`. -/
structure SheetDataQ (P : Type) where
  fmax : ℕ
  macs : List ℕ
  peak : List ℕ
  pipe : List (Option P)
  width : P → ℚ
  smem : List ℕ

/-- The sheet over `Fin n`. -/
noncomputable def SheetDataQ.sheet {P : Type} (D : SheetDataQ P) (n : ℕ) : Sheet (Fin n) P where
  fmax := D.fmax
  macs k := D.macs.getD k.val 0
  peak k := D.peak.getD k.val 0
  pipe k := D.pipe.getD k.val none
  width p := D.width p
  smem k := D.smem.getD k.val 0

end Accounting.TS.Headline
