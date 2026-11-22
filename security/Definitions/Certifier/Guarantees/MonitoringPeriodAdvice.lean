module

public import Definitions.Core.Compiler
public import Definitions.Certifier.Guarantees.MonitoringPeriod
public import Definitions.Certifier.MonitoringPeriodConditions
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
# The leak through the network attester over a monitoring period, each epoch's declared circuit compiled from advice

`MonitoringPeriodLeakAccepted` (`Definitions.Certifier.Guarantees.MonitoringPeriod`) fixes each epoch's statement, so
its count never ranges over the declared circuit. Here epoch `i`'s declared circuit is the registered compiler's
output on the advice of epochs `1, …, i` (`Verity.Compiler.run`), each advice string at most the compiler's cap `ℓa`
bits, registered information fixed before the first epoch. The count is over the secrets for which some such advice
makes every epoch meet its good event, and it charges `ℓa + 1` bits an epoch beyond the per-circuit count: there are
fewer than `2^(ℓa + 1)` strings of at most `ℓa` bits, whatever length the prover commits in each epoch.
-/

@[expose] public section

namespace NetTiming.Guarantees

open Claims.Sampling Finset
open Claims.SamplingGeneric (Arg Compiled)

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak through the network attester over a monitoring period, with the declared circuits compiled from
advice.** As `MonitoringPeriodLeakAccepted`, where epoch `i`'s statement `sw i α` depends on the advice `α`, one
string an epoch: its Program is `progOf` of the registered compiler `B`'s output on the registered values `θ` and the
advice of epochs `0, …, i` (`hsw`), with its partition `Pw i α` and `β i α` bounding the committed bits of each of its
possible incorrect sets, and `M` bounds the per-circuit count `executionPhasesCount` at every advice of at most
`B.ℓa` bits an epoch (`hM`). The registered values `ρ i` do not depend on the advice. Then:

* each epoch's audit, at a statement in its scope, has `Pr[acc ∧ ¬RLeakCertifier δ] ≤ δ + ε_ks + ε_bind + Pr[B_tap]`;
* where at every epoch's statement the network attester registers its observations, the covert sender's secrets for
  which some advice of at most `B.ℓa` bits an epoch makes every epoch meet `ExecutionPhaseGood` number at most
  `2^((ℓa + 1)·n) · #Ss · ∏_c (cap c + 1) · M`.

It bounds each epoch on its own, with SPOT's knowledge soundness (`Claims.SamplingGeneric.Compiled.KS`) a hypothesis;
the union of the epochs' failures over the monitoring period's challenges is not stated. No Lean definition maps the
compiler's circuit to the audit's Program, so the encoding `progOf` is an argument and the bound holds at every one. -/
def MonitoringPeriodLeakAdvice : Prop :=
  ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    (H : Hash.Msg → Hash.Msg) (D : Compiled Coin A) (hks : D.KS H) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (δ : ℝ) (hδ : 0 ≤ δ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ)
    (B : Verity.Compiler) (θ : List Verity.Registration.Theta) (progOf : Verity.Compiler.Output → Prog)
    {n : ℕ} (sw : Fin n → (Fin n → List Bool) → AuditStatement)
    (hsw : ∀ i α, (sw i α).prog = progOf (B.run θ ((List.ofFn α).take (i.val + 1))))
    (Pw : Fin n → (Fin n → List Bool) → Partition) (hPw : ∀ i α, D.partition (sw i α) = some (Pw i α))
    (ρ : Fin n → ℕ → Bool) (op : Fin n → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool)
    (β : Fin n → (Fin n → List Bool) → ℕ)
    (hβ : ∀ i α, ∀ F ∈ incorrectSets (sw i α).design (Pw i α) δ, faultyBits (sw i α).prog (Pw i α) F ≤ β i α)
    (M : ℕ) (hM : ∀ α : Fin n → List Bool, (∀ i, (α i).length ≤ B.ℓa) →
      executionPhasesCount lay k δ reg Dj (fun i => sw i α) (fun i => Pw i α) (fun i => β i α) ≤ M)
    {Conn Cell Srv Dev Z : Type} [Fintype Conn] (arr : Srv → Conn → Arrivals Cell)
    (f : RecordFormat Conn Cell (Fin n → List (Obs L K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedDevice tap run fun p => recordingCode f p.1 (arr p.2))
    (cap : Conn → ℕ) (Ss : Set (CertifierSchedule Conn)) (hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c)
    (hfit : ∀ s ∈ Ss, f.fits s)
    (Zs : Set Z) (schedOf : Z → CertifierSchedule Conn) (srvOf : Z → Srv) (hs : ∀ z ∈ Zs, schedOf z ∈ Ss)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z) (hdz : ∀ z ∈ Zs, dec (run tap (schedOf z, srvOf z)).1 = z)
    (x : Z → Fin n → Hash.Msg) (wit : Z → Fin n → Wit) (tops : AuditStatement → List (Opening Bool)) (hb : D.commit.Binds H),
    (∀ m, ScopeAcc D lay k m → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
      Claims.PrBad H D.V
          (RLeakCertifier H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj tops)
          D.E m P ≤
        D.κ δ m + Real.sqrt (D.γ m * Claims.PrB H D.V P m) + (Bext H D.V D.E Ω P m).Pr H +
          (Btap H D.V D.E D.commit D.parse tops Ω P m).Pr H) ∧
      ((∀ z ∈ Zs, ∀ i s, D.parse (x z i) = some s → CertifierRegisters H D.commit (lay s) (k s)
          (op i (runsBefore (run tap (schedOf z, srvOf z)).2 i)) s (tops s) ((run tap (schedOf z, srvOf z)).2 i)) →
        {z | z ∈ Zs ∧ ∃ α : Fin n → List Bool, (∀ i, (α i).length ≤ B.ℓa) ∧
            ∀ i, ExecutionPhaseGood H D L K S lay k δ reg Dj (sw i α) (ρ i) tops (x z i) (wit z i)
              ((run tap (schedOf z, srvOf z)).2 i)}.encard ≤
          ((2 ^ ((B.ℓa + 1) * n) : ℕ) : ℕ∞) * (Ss.encard * (((∏ c, (cap c + 1)) * M : ℕ) : ℕ∞)))

end NetTiming.Guarantees
