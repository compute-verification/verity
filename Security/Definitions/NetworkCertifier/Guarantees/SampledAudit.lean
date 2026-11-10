module

public import Definitions.NetworkCertifier.SampledAudit
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
# The sampled audit's guarantee: it implies `Accepted`, at `ε` at `δ` over the selection

`SampledAuditAccepted` is the statement, proved in `Proofs.NetworkCertifier.SampledAuditAccepted`. Wherever the compiled
sampled verifier is knowledge sound (`Claims.SamplingGeneric.Compiled.KS`), the sampled audit of a program with a
public boundary layout implies `Accepted` by the circuit's Program (`KSAccepted`,
`Definitions.NetworkCertifier.SampledAudit`). The free advice is held at the incorrect proof blocks' committed gates
only, and the incorrect set is in `M_δ` and meets no class sampled `all`. Its one hypothesis is `Compiled.KS` at the
same verifier, the sampling claim it rests on; every condition on the run is either a verifier refusal (the scope,
`ScopeAcc`) or a condition, inside the conclusion, that picks out which observations the conclusion is about
(`Carries`, `Covered`).
-/

@[expose] public section

namespace NetTiming.Guarantees

open Claims Claims.Sampling
open Claims.SamplingGeneric (Arg Compiled)

/-- **The sampled audit implies `Accepted`, at `ε` at `δ` over the selection.** At every compiled sampled verifier `D`,
over any proof system for proof blocks, at which `Compiled.KS` holds, and every public layout `lay` and horizon `k` of
its statements, `KSAccepted` holds. On the statements in scope (the sampling scope, and the layout fits the program),
at every `δ ≥ 0`: `Pr[acc ∧ ¬RAcc δ] ≤ δ + ε_ks + ε_bind`, at `Compiled.KS`'s errors. The relation `RAcc δ` says three
things:

* the incorrect set is in `M_δ`;
* the incorrect set holds no proof block of a class sampled `all` (the transfer-gate proof blocks and N2's head checks:
  error `0`);
* the advice whose inputs are the trace's committed values, holding values exactly at the incorrect proof blocks' committed
  gates, makes `Accepted` by the circuit's Program every observation sequence and operator that the trace's transfer gates
  carry with every status complete.

With no proof block faulty, the advice holds no gate, and `Accepted` is exact. -/
def SampledAuditAccepted : Prop :=
  ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    (H : Msg → Msg) (D : Compiled Coin A) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ),
    D.KS H → KSAccepted H D L K S lay k

end NetTiming.Guarantees
