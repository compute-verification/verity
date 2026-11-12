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
# The incorrect proof blocks over a monitoring period of sampled audits

A monitoring period is `m` windows, each audited on its own live coins: the period's coins are a product `Fin m → Ω`,
window `i` sampling `k i` of its `n i` units at `ω i`, a uniform `k i`-subset. The units wrong at window `i`'s
registration (`W i ω`) are fixed before its sample: they read only the earlier windows' coins. Every incorrect proof block risks
detection in its window, and the windows' samples are independent, so the escapes multiply: with `q ≤ k i / n i` in every
window, every window's audit accepts while the monitoring period holds at least `F` incorrect proof blocks in total with
probability at most `(1 - q)^F` plus the windows' soundness errors `ε i`, whatever the number of windows. At
`F ≥ ln (1/δ) / q` that is at most `δ + Σ_i ε_i`.

The statement is a fact of probability over every sample, acceptance event and error: the sample's law and each window's
soundness are its hypotheses. A deployment's instances of them (A3 for the sample, `Claims.SamplingGeneric.Compiled.KS`
for each window) come in where it is applied to the sampled verifier.
-/

@[expose] public section

namespace NetTiming.Guarantees

open Claims

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The incorrect proof blocks over a monitoring period of accepted sampled audits.** Over `m` windows on independent live
coins, window `i` sampling a uniform `k i`-subset of its `n i` units (every coin samples `k i` units, and each
`k i`-subset is sampled by the share `1 / C(n i, k i)` of the coins), with the units wrong at each registration fixed
before that window's sample, and each window's audit accepting while its sample meets an incorrect proof block with probability at most
`ε i`: for `0 ≤ q ≤ 1` with `q · n i ≤ k i` in every window,

* every window's audit accepts and the monitoring period holds at least `F` incorrect proof blocks with probability at most
  `(1 - q)^F + Σ_i ε i`;
* `(1 - q)^F ≤ δ` once `ln (1/δ) ≤ q · F`, so at most `ln (1/δ) / q` incorrect proof blocks pass the whole monitoring period
  except with probability `δ + Σ_i ε i`.

The share `q` is a hypothesis: the count a `rate:NUM/DEN` rule names meets `q · n i ≤ k i` at `q = NUM/DEN` over
whatever `n i` a window registers (`NetTiming.pooledIncorrectBlocks_rate` is this statement at that rate), while under
a fixed count the largest `q` that holds falls as the prover pads `n i`. -/
def PooledIncorrectBlocksLe : Prop :=
  ∀ {m : ℕ} {Ω : Type} [Fintype Ω] [Nonempty Ω] (n k : Fin m → ℕ) (draw : (i : Fin m) → Ω → Finset (Fin (n i)))
    (hcard : ∀ i x, (draw i x).card = k i)
    (hunif : ∀ i (S : Finset (Fin (n i))), S.card = k i →
      (Finset.univ.filter fun x => draw i x = S).card * (n i).choose (k i) = Fintype.card Ω)
    (W : (i : Fin m) → (Fin m → Ω) → Finset (Fin (n i)))
    (hW : ∀ i ω ω', (∀ j, j < i → ω j = ω' j) → W i ω = W i ω')
    (acc : Fin m → (Fin m → Ω) → Prop) (ε : Fin m → ℝ)
    (hsound : ∀ i, Pr (fun ω => acc i ω ∧ ¬Disjoint (draw i (ω i)) (W i ω)) ≤ ε i)
    (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (hq : ∀ i, q * n i ≤ k i) (F : ℕ),
    Pr (fun ω => (∀ i, acc i ω) ∧ F ≤ ∑ i, (W i ω).card) ≤ (1 - q) ^ F + ∑ i, ε i ∧
      ∀ δ : ℝ, 0 < δ → Real.log (1 / δ) ≤ q * F → (1 - q) ^ F ≤ δ

end NetTiming.Guarantees
