module

public import Definitions.NetworkCertifier.TapRows
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
# A monitoring period of sampled audits: each window's good event

Definitions only. A monitoring period is a list of windows, each audited by the sampled verifier at a statement that its
registration fixes (`SameRegistration`). A window's good event (`WindowGood`) is the leak relation at the network
certifier's rows (`RLeakCertifier`) at the audit's extracted witness, with the registered inputs pinned (`AsReg`), the trace
holding the registration's values at the registered inputs, and the observations complete. Which observations the
relation counts is picked out by the network certifier's registration of its rows (`CertifierRegisters`), not by the trace:
`RLeakCertifier` fails, on an accepted run, only where the extraction fails or the trace's openings collide with the network
certifier's (`Btap`). The registration's values at the prover's registered inputs are the one condition the event holds
that no statement here bounds.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling

section Period

open Claims.SamplingGeneric (Arg Compiled)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **A statement registered as `s₀`**: the same program, design, layout, horizon, registered inputs and partition as
`s₀`; only the trace's commitments (`roots`) and the description may differ. -/
def SameRegistration {L S : ℕ} (D : Compiled Coin A) (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (s₀ s : AuditStatement) : Prop :=
  s.prog = s₀.prog ∧ s.design = s₀.design ∧ lay s = lay s₀ ∧ k s = k s₀ ∧ reg s = reg s₀ ∧ D.partition s = D.partition s₀

/-- **A window's good event**: at the window's statement `x` and the audit's extracted witness `w`, the leak relation at
the network certifier's rows holds (the network certifier's openings `tops s`), `x` is in the audit's scope and
registered as `s₀`, the trace holds `ρ` at the registered inputs, and the run's observations `o` have the horizon's
length, jitter below `Dj` and every status complete. -/
def WindowGood (H : Msg → Msg) (D : Compiled Coin A) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (δ : ℝ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ) (s₀ : AuditStatement) (ρ : ℕ → Bool)
    (tops : AuditStatement → List (Opening Bool)) (x : Msg) (w : Wit) (o : List (Obs L K S)) : Prop :=
  RLeakCertifier H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj tops x w ∧
    ScopeAcc D lay k x ∧
    ∃ s, D.parse x = some s ∧ SameRegistration D lay k reg s₀ s ∧
      (∀ g ∈ reg s, traceVal s.prog (w.val H D.commit s) g = ρ g) ∧ Covered (k s) o ∧ o.length = k s ∧
      ∀ y ∈ o, y.jitter < Dj

end Period

end NetTiming
