module

public import Proofs.Flock.Claims.Sampling.Reference.Prover
public import Proofs.Flock.Claims.Sampling.Reference.Complete
public import Proofs.Flock.Claims.Sampling.Reference.Deployed
public import Proofs.Flock.Claims.Sampling.Reference.DeployedCode
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

/-! The sampled reference prover and its completeness: the prover, which commits every unit's values under hm96
leaves, waits for the selection coin and runs Flock's masked reference prover in each checked unit's session on what it
opens (`Reference.Prover`); `Claims.Sampling.Complete` from Flock's completeness session by session, by the union bound
over the checked slots (`Reference.Complete`); at the deployed record, with what remains as hypotheses
(`Reference.Deployed`); and at the code whose checked units are the Lean sampled verifier's draw of record, one session
per registration (`Reference.DeployedCode`). -/

@[expose] public section
