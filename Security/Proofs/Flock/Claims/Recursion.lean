module

public import Proofs.Flock.Claims.Recursion.Serve
public import Proofs.Flock.Claims.Recursion.Wald
public import Proofs.Flock.Claims.Recursion.Compose
public import Proofs.Flock.Claims.Recursion.Transport
public import Proofs.Flock.Claims.Recursion.Cost
public import Proofs.Flock.Claims.Recursion.CostExp
public import Proofs.Flock.Claims.Recursion.Finders
public import Proofs.Flock.Claims.Recursion.Hybrid
public import Proofs.Flock.Claims.Recursion.KS
public import Proofs.Flock.Claims.Recursion.OuterKS
public import Proofs.Flock.Claims.Recursion.Inner
public import Proofs.Flock.Claims.Recursion.Abort
public import Proofs.Flock.Claims.Recursion.InnerAbort
public import Proofs.Flock.Claims.Recursion.PlurLin
public import Proofs.Flock.Claims.Recursion.BadLin
public import Proofs.Flock.Claims.Recursion.KSLin
public import Proofs.Flock.Claims.Recursion.Gap
public import Proofs.Flock.Claims.Recursion.Registered
public import Proofs.Flock.Claims.Recursion.FlockAbort
public import Proofs.Flock.Claims.Recursion.AbortWait
public import Proofs.Flock.Claims.Recursion.AbortFork
public import Proofs.Flock.Claims.Recursion.FlockKSAbort
public import Proofs.Flock.Claims.Recursion.FlockKSAcc
public import Proofs.Flock.Claims.Recursion.FlockKSDeploy
public import Proofs.Flock.Claims.Recursion.FlockKSRecord
public import Proofs.Flock.Claims.Recursion.FlockKSOnAcc
public import Proofs.Flock.Claims.Recursion.KSStated
public import Proofs.Flock.Claims.Recursion.Scope
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

/-! The ZK-recursion compiler's claims proofs: serving one rerun program's reruns by another's (`Serve`), `E^rep`'s
weighted reruns by Wald's identity (`Wald`), the composed extractor, `E^rep` served by the outer extractor, with its
cost and its knowledge soundness over `KS`/`KSLin` (`Compose`), where the recursive audit's fork-form disjuncts land
among `KS`'s error terms (`Transport`), the recursion's cost claim from the outer extractor's (`Cost`), the outer
extractor's expected reruns by Wald's argument over forks, which proves the cost claim (`CostExp`), its
extraction-based finder `Be` as a fork finder (`Finders`), the composed extractor's failure with the outer extractor
charged at every rerun (`Hybrid`), and the recursion's `KS` from the outer Flock's KS at each session and the inner
protocol's KS at the served runs (`KS`), and that outer KS at each session from Flock's own KS at the outer deployment,
the session's prover simulating the rest of the compiled verifier around it (`OuterKS`); and `ScopeMatches` at the
compiled verifier's parser and scope check (`Scope`). -/

@[expose] public section
