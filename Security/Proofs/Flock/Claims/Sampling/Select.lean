module

public import Proofs.Flock.Claims.Sampling.Select.Seq
public import Proofs.Flock.Claims.Sampling.Select.Take
public import Proofs.Flock.Claims.Sampling.Select.Layout
public import Proofs.Flock.Claims.Sampling.Select.Steps
public import Proofs.Flock.Claims.Sampling.Select.Bounds
public import Proofs.Flock.Claims.Sampling.Select.Misses
public import Proofs.Flock.Claims.Sampling.Select.DrawLaw
public import Proofs.Flock.Claims.Sampling.Select.Design
public import Proofs.Flock.Claims.Sampling.Select.Scope
public import Proofs.Flock.Claims.Sampling.Select.Refusals
public import Proofs.Flock.Claims.Sampling.Select.Code
public import Proofs.Flock.Claims.Sampling.Select.OS
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

/-! The deployed sampled verifier's draw (`SampledProofs.Slots.select`) against the claims' `P_a`: a sequential draw's
stopping-time bound (`Select.Seq`); the draw's coins read in order and its law's stages (`Select.Take`); a subset draw
as the live draw `Law.runOS` on the same bytes, and its bound under A3 (`Select.OS`); the
verifier's partition as the claims' slots (`Select.Layout`); `Select.select`'s steps and the design it draws
(`Select.Steps`); each law's miss bound on its coins (`Select.Bounds`); `drawMisses`, which proves
`Properties.SampledProofs.DrawMisses` (`Select.Misses`); `DrawLaw` at a deployment whose draw is `Slots.select`'s
(`Select.DrawLaw`); `designWellFormed`, which proves `Properties.SampledProofs.DesignWellFormed` (`Select.Design`); and
`ScopeMatches` and `Refuses` from it, where the scope check is the readable scope's program part (`Select.Scope`); and that
program part as five refusals of the verifier's evaluation, each needed (`Select.Refusals`). -/

@[expose] public section
