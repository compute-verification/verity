module

public import Definitions.NetworkCertifier.Assumptions.RegisteredInputs
public import Definitions.NetworkCertifier.Checked
public import Definitions.NetworkCertifier.Period
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
# A monitoring period of sampled audits with checked inputs: the run the check is held to, and each window's good event

Definitions only. `Definitions.NetworkCertifier.Period`'s good event with each statement's checked inputs left out of
the count (`Definitions.NetworkCertifier.Checked`): the leak relation at the network certifier's rows, with a valid run.

**The run.** The relation at the network certifier's rows accepts the network certifier's records on the run that
reads the trace's inputs, takes its ingress from the operator's moves on those records, and holds an incorrect proof
block's committed gates at the trace filled in with the network certifier's values (`fillTrace`): the trace need open
no transfer gate inside an incorrect proof block, and the network certifier's opening there is what its record shows.
That run is `certifierRun`, and the check is held to it (`CheckedWitness` at `certifierRun`). It is not the trace's
run (`traceRun`), which holds those gates at the trace's own values, `false` where the trace opens nothing: a check
that reads a transfer gate inside an incorrect proof block can hold on one and fail on the other.

* `certifierRun`: the run at the network certifier's rows at a witness, where the network certifier registers records
  and the trace agrees with its openings wherever both open a gate. Elsewhere there is none: the relation at the
  network certifier's rows asks nothing where it registers no record, and a disagreement is a collision (`Btap`).
* `RLeakCertifierChecked`: `RLeakCertifier` with the sequences `AccSeqsChecked` counts.
* `WindowGoodChecked`: `WindowGood` with it, the window's check the registration's, and the registered inputs' values
  read as the named premise `RegisteredInputsHeld`.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling

/-- **The run at the network certifier's rows** at the witness `w`: where the trace agrees with the network
certifier's openings `tops s` wherever both open a gate, and the network certifier registers records `o` at an operator
`op` (`CertifierRegisters`), `C` over the horizon on the trace's inputs, the ingress the operator's moves on `o` give,
and the incorrect proof blocks' committed gates held at the trace filled in with the network certifier's values
(`heldAdv` of `fillTrace`). The registration fixes the operator's moves at the ingress its openings hold, so this is
one run, or none where the network certifier registers nothing or the trace disagrees with it. It is the run
`RLeakCertifierChecked` accepts at. -/
def certifierRun (H : Msg → Msg) (Cm : CommitmentScheme Bool) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (tops : AuditStatement → List (Opening Bool)) (s : AuditStatement) (Pt : Partition)
    (w : Wit) : Set (ℕ → Bool) :=
  {r | (∀ g, (∃ q ∈ tops s, q.pos = g) → ∀ b, w.val H Cm s g = some b → b = certifierVal (tops s) g) ∧
    ∃ (op : ℕ → List (Obs L K S) → List Bool) (o : List (Obs L K S)),
      CertifierRegisters H Cm (lay s) (k s) op s (tops s) o ∧
        r = valOver s.prog
          ((lay s).inputOf (k s) ((List.range (k s)).map fun i => op i (o.take i)) (traceVal s.prog (w.val H Cm s)))
          (heldAdv s.prog Pt (faulty H Cm s Pt w) (fillTrace (w.val H Cm s) (certifierVal (tops s))))}

/-- **The leak relation at the network certifier's rows with a valid run** at `δ`: `RLeakCertifier`, with the sequences
`AccSeqsChecked` counts at each statement's check `valid s`. -/
def RLeakCertifierChecked (H : Msg → Msg) (Cm : CommitmentScheme Bool) (parse : Msg → Option AuditStatement)
    (partition : AuditStatement → Option Partition) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (δ : ℝ) (As : AuditStatement → (ℕ → Bool) → Set (ℕ → Bool))
    (valid : AuditStatement → (ℕ → Bool) → Prop) (Dj : ℕ) (tops : AuditStatement → List (Opening Bool)) : Rel Wit :=
  fun x w => ∃ s P, parse x = some s ∧ partition s = some P ∧
    ∀ (op : ℕ → List (Obs L K S) → List Bool) (o : List (Obs L K S)),
      CertifierRegisters H Cm (lay s) (k s) op s (tops s) o → Covered (k s) o → o.length = k s →
        (∀ y ∈ o, y.jitter < Dj) →
        o ∈ AccSeqsChecked s.prog P s.design (lay s) (k s) (As s (traceVal s.prog (w.val H Cm s))) (valid s) δ op Dj

section Period

open Claims.SamplingGeneric (Arg Compiled)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **A window's good event with a valid run**: at the window's statement `x` and the audit's extracted witness `w`,
the leak relation at the network certifier's rows holds with a valid run (the network certifier's openings `tops s`),
`x` is in the audit's scope and registered as `s₀`, with `s₀`'s check, the trace holds `ρ` at the registered inputs
(`RegisteredInputsHeld`), and the run's observations `o` have the horizon's length, jitter below `Dj` and every status
complete. -/
def WindowGoodChecked (H : Msg → Msg) (D : Compiled Coin A) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (δ : ℝ) (reg : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop)
    (Dj : ℕ) (s₀ : AuditStatement) (ρ : ℕ → Bool) (tops : AuditStatement → List (Opening Bool)) (x : Msg) (w : Wit)
    (o : List (Obs L K S)) : Prop :=
  RLeakCertifierChecked H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s))
      valid Dj tops x w ∧
    ScopeAcc D lay k x ∧
    ∃ s, D.parse x = some s ∧ SameRegistration D lay k reg s₀ s ∧ valid s = valid s₀ ∧
      RegisteredInputsHeld H D.commit (reg s) ρ s w ∧ Covered (k s) o ∧ o.length = k s ∧ ∀ y ∈ o, y.jitter < Dj

end Period

end NetTiming
