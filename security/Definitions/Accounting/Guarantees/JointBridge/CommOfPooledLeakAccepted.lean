module

public import Definitions.Accounting.JointBridge.CommOfPooledLeakAccepted
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
# The network certifier's per-period guarantee, pooled over a monitoring period of sampled audits (trusted)

`CommOfPooledLeakAccepted` is the network audit's `Sound`, the coins model's per-period guarantee, for a run whose
communication audit is the network certifier's monitoring period of `m` sampled audits
(`Accounting.JointBridge.RealizesCommOfPooledLeakAccepted`), in the pooled reading: at the error of
`NetTiming.Guarantees.PooledLeakAccepted`, `e^(−q·Fmax) + Σ_i e_i + Σ_i e'_i`, whose first term is paid once over the
monitoring period, where the reading per phase, the guarantee beside this one, pays its `δ` in every execution phase.
It is conditional on `PooledLeakAccepted`'s two inputs at each execution phase (`hesc`, `hleak`), which no proof
discharges yet, so it bounds no deployed run until one does.
-/

@[expose] public section

namespace Accounting.Guarantees.Bridge

open Claims Claims.Sampling NetTiming Accounting.JointBridge

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The network certifier's per-period guarantee, pooled** (conditional on its two inputs at each execution phase).
On a run that realizes the network certifier's monitoring period of `m` sampled audits at the claim `c.K`
(`RealizesCommOfPooledLeakAccepted`: the network audit's own coins are the `m` verification phases' coins, the deployed
network certifier runs its code on its host, `trusted-host`, the realization's `trusted`, and the claim is the
period's good event at the bound `Fmax` on its incorrect proof blocks), given

* `hesc`, the miss probability given the earlier phases' coins, at the realization's `acc` and `W` and at every
  value of the other audits' coins: over its own coins, execution phase `i`'s audit accepts with probability at most
  `(1 - q)^|W_i| + e i`, `W_i` the incorrect set of the witness computed from its registration and `q` the sampling
  design's detection probability of one incorrect proof block in an execution phase. SPOT's guarantee at each
  verification phase, given what came before it, is what gives it;
* `hleak`, the leak relation at that incorrect set (`PhaseLeaks` at the realization's `op`, `acc`, `W` and `record`):
  over the period's coins, execution phase `i`'s audit accepts while its observations in the network certifier's
  record fail `LeakAt` at `W_i` with probability at most `e' i`. `LeakAt` is read at the phase's operator at the
  record's earlier phases: the period's operator `R.op` is fixed before the period, and execution phase `i`'s reads
  only the execution phases closed before its own. The leak guarantee of one sampled audit, on live coins and at the
  incorrect set of the witness computed before the sample, is what gives it;

for `e ≥ 0` and `0 ≤ q ≤ 1`, the network audit's `Sound` holds at `e^(−q·Fmax) + Σ_i e i + Σ_i e' i`: at any value of
the other audits' coins, the network audit accepts while the period's good event fails with at most that probability
over its own coins. The first term is at most `δ` once `Fmax ≥ ln(1/δ)/q`, and the monitoring period pays it once.
No proof discharges either input, here or in `PooledLeakAccepted`. What the claim at `c.K` counts is the
realization's to say, and no field of it ties `K` to `PooledLeakAccepted`'s count yet. -/
def CommOfPooledLeakAccepted : Prop :=
  ∀ {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {I : Type} {Ω : I → Type}
    [DecidableEq I] [∀ i, Fintype (Ω i)] {r : PeriodRun d α C Ω} (c : Certificates) (p : Price) {L Kₒ S : ℕ}
    {lay : AuditStatement → BoundaryGates L S} {k : AuditStatement → ℕ} {reg : AuditStatement → Finset ℕ}
    {Dj m Fmax : ℕ} (R : RealizesCommOfPooledLeakAccepted r c.K L Kₒ S lay k reg Dj m Fmax)
    (e e' : Fin m → ℝ) (he : ∀ i, 0 ≤ e i) (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hesc : PeriodEscapes Set.univ R.acc R.W q e)
    (hleak : PhaseLeaks lay k reg Dj R.sw R.Pw R.ρ R.op Set.univ R.acc R.W R.record e'),
    r.Sound c p .network (Real.exp (-(q * Fmax)) + ∑ i, e i + ∑ i, e' i)

end Accounting.Guarantees.Bridge
