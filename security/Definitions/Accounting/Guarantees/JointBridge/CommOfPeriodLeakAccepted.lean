module

public import Definitions.Accounting.JointBridge.CommOfPeriodLeakAccepted
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
# The network certifier's per-period guarantee, on a run that realizes its period of sampled audits (trusted)

`CommOfPeriodLeakAccepted` is the network audit's `PeriodRun.Sound`, the coins model's per-period guarantee,
for a run whose communication audit is the network certifier's period of sampled audits
(`Accounting.JointBridge.RealizesCommOfPeriodLeakAccepted`), at the error of
`NetTiming.Guarantees.PeriodLeakAccepted`: `n · (δ + ε_ks + ε_bind + ε_tap)` over the period's `n` windows.
-/

@[expose] public section

namespace Accounting.Guarantees.Bridge

open Claims Claims.Sampling NetTiming Accounting.JointBridge
open Claims.SamplingGeneric (Arg Compiled)

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The network certifier's per-period guarantee.** On a run that realizes the network certifier's period of `n`
sampled audits at the claim `c.K`, whose deployed network certifier runs its code on its host (`trusted-host`, the
realization's `trusted`), at a compiled sampled verifier `D` where `Claims.SamplingGeneric.Compiled.KS` holds and the
trace's commitment binds (`CommitmentScheme.Binds`), at `δ ≥ 0`,
where each window's prover has `κ − δ + √(γ·Pr[B]) ≤ ε_ks`, `Pr[B_ext] ≤ ε_bind` and `Pr[B_tap] ≤ ε_tap` (`ε_ks` and
`ε_bind` as in `Compiled.KS`'s errors, `B_tap` the collisions against the network certifier's openings), the
network audit's `PeriodRun.Sound` holds at `n · (δ + ε_ks + ε_bind + ε_tap)`. -/
def CommOfPeriodLeakAccepted : Prop :=
  ∀ {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}
    [DecidableEq I] [∀ i, Fintype (Ω i)] {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    {A : Arg Coin} {r : PeriodRun d α C Ω} (c : Certificates) (p : Price) {H : Msg → Msg}
    {D : Compiled Coin A} {L Kₒ S : ℕ} {lay : AuditStatement → BoundaryGates L S} {k : AuditStatement → ℕ} {δ : ℝ}
    {reg : AuditStatement → Finset ℕ} {Dj n : ℕ}
    (R : RealizesCommOfPeriodLeakAccepted r c.K H D L Kₒ S lay k δ reg Dj n) (hks : D.KS H)
    (hb : D.commit.Binds H) (hδ : 0 ≤ δ) (εks εbind εtap : ℝ)
    (herr : ∀ ξ i, D.κ δ (R.stmt ξ i) - δ +
      Real.sqrt (D.γ (R.stmt ξ i) * PrB H D.V (R.prover ξ i) (R.stmt ξ i)) ≤ εks)
    (hbind : ∀ ξ i, (Bext H D.V D.E (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤ εbind)
    (htap : ∀ ξ i, (Btap H D.V D.E D.commit D.parse R.tops (R.Tape ξ i) (R.prover ξ i) (R.stmt ξ i)).Pr H ≤
      εtap),
    r.Sound c p .network (n * (δ + εks + εbind + εtap))

end Accounting.Guarantees.Bridge
