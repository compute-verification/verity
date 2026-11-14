module

public import Definitions.NetworkCertifier.Guarantees.MonitoringPeriodChecked
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
# The leak over a monitoring period with each failure charged as it happens

A connection the network certifier fails (`failCell`) is the server's move, neither the operator's nor advice, so
`MonitoringPeriodLeakChecked`'s operator and registered values, fixed before the period, cannot hold where it fell. These
statements let both follow the run's fail pattern, each connection's fail cell or none, and pay for it by counting the
patterns:

* `failCount`, `FailsBelow`, `failPatterns`: the failed connections of a pattern, a bound on each fail cell, and the
  number of patterns of exactly `j` failures below their bounds;
* `MonitoringPeriodLeakFails`: exactly `j` failures; `MonitoringPeriodLeakFailsAtMost`: at most `j`, with the held conclusion on every run;
  `MonitoringPeriodLeakCharged`: a charge that adds over periods, under weights on the number of failures (`KraftWeights`).

**The pattern covers the operator and nothing else.** `op` and `ρ` are bound before the secrets and read the fail
pattern alone, so a window's operator and registered values are fixed for each pattern before the run. A failure is
paid for once, in the patterns' factor: the count has no factor for a connection's cells.

**One factor for the period.** `Conn` is every connection slot of the monitoring period, over all its links, and the
number of failures is the period's, so the patterns' factor is one number for the period, under one logarithm.
-/

@[expose] public section

namespace NetTiming

open scoped NNRat

/-- The connections of a fail pattern that failed. -/
def failCount {κ : Type} [Fintype κ] (φ : κ → Option ℕ) : ℕ :=
  (Finset.univ.filter fun c => (φ c).isSome).card

/-- Every fail cell of the pattern is below its connection's bound. -/
def FailsBelow {κ : Type} (bound : κ → ℕ) (φ : κ → Option ℕ) : Prop :=
  ∀ c i, φ c = some i → i < bound c

/-- **The fail patterns of exactly `j` connections**: the choices of `j` connections and, for each, a fail cell below
its bound, `Σ_{J, #J = j} ∏_{c ∈ J} bound c`; `C(N, j) · bound^j` at one bound. -/
def failPatterns {κ : Type} [Fintype κ] (bound : κ → ℕ) (j : ℕ) : ℕ :=
  ∑ J ∈ (Finset.univ : Finset κ).powersetCard j, ∏ c ∈ J, bound c

/-- **Weights on the number of failures** `0 … N` that sum to at most one (Kraft's inequality, without rounding a
length up to a whole bit). -/
def KraftWeights (w : ℕ → ℚ≥0) (N : ℕ) : Prop :=
  ∑ j ∈ Finset.range (N + 1), w j ≤ 1

end NetTiming

namespace NetTiming.Guarantees

open Claims.Sampling Finset
open Claims.SamplingGeneric (Arg Compiled)
open scoped NNRat

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak over a monitoring period, each failure charged as it happens: exactly `j` failures.**
`MonitoringPeriodLeakChecked`, where each window's operator and registered values may follow the network certifier's fail cells
over the period (`op φ i` and `ρ φ i`, `φ` the fail cell of each connection, `failCell`): the secrets whose run
fails exactly `j` connections, each below its bound, and whose every window meets `ExecutionPhaseGoodChecked`, number
at most `failPatterns bound j` times `MonitoringPeriodLeakChecked`'s count. At `j = 0` the factor is one.

`op` and `ρ` are bound before the secrets and read the fail pattern alone: a window's operator and registered values
are fixed for each pattern before the run, so they carry nothing of the secret beyond the pattern, which the factor
counts. Nothing else of the statement reads the pattern. -/
def MonitoringPeriodLeakFails : Prop :=
  ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    (H : Claims.Msg → Claims.Msg) (D : Compiled Coin A) (hks : D.KS H) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (δ : ℝ) (hδ : 0 ≤ δ) (reg chk : AuditStatement → Finset ℕ)
    (valid₁ valid₂ : AuditStatement → (ℕ → Bool) → Prop) (ε₁ ε₂ : Claims.Msg → ℝ) (Dj : ℕ)
    {n : ℕ} (sw : Fin n → AuditStatement) (Pw : Fin n → Partition) (β : Fin n → ℕ)
    (hplaced : WindowsPlaced D.partition δ sw Pw β)
    (hchk : ∀ i, CheckedUnique (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i))
      (fun r => valid₁ (sw i) r ∧ valid₂ (sw i) r))
    {Conn : Type} [Fintype Conn] (ρ : (Conn → Option ℕ) → Fin n → ℕ → Bool)
    (op : (Conn → Option ℕ) → Fin n → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool)
    {Cell Srv Dev Z Par : Type} (arr : Srv → Conn → Arrivals Cell)
    (f : RecordFormat Conn Cell (Fin n → List (Obs L K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => recordingCode f p.1 (arr p.2))
    (rd : Par → (Fin n → List (Obs L K S)) → Conn → List (ℕ × CellView Cell))
    (allows : Par → CertifierSchedule Conn → Prop) (hread : f.ReadsBy rd allows)
    (Θ : Set Par) (Zs : Set Z) (schedOf : Z → CertifierSchedule Conn) (srvOf : Z → Srv)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z)
    (hch : MonitoringPeriodChannelReg f allows Θ Zs schedOf srvOf dec fun p => (run tap p).1)
    (x : Z → Fin n → Claims.Msg) (wit : Z → Fin n → Wit) (tops : AuditStatement → List (Opening Bool))
    (hb : D.commit.Binds H) (bound : Conn → ℕ)
    (j : ℕ),
    (∀ m, ScopeAcc D lay k m →
        CheckedWitness H D (certifierRun H D.commit L K S lay k tops) valid₁ ε₁ m →
        CheckedWitness H D (certifierRun H D.commit L K S lay k tops) valid₂ ε₂ m →
        ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
      Claims.PrBad H D.V
          (RLeakCertifierChecked H D.commit D.parse D.partition L K S lay k δ
            (fun s => AsReg s.prog (lay s) (k s) (reg s)) (fun s r => valid₁ s r ∧ valid₂ s r) Dj tops)
          D.E m P ≤
        D.κ δ m + ε₁ m + ε₂ m + Real.sqrt (D.γ m * Claims.PrB H D.V P m) + (Bext H D.V D.E Ω P m).Pr H +
          (Btap H D.V D.E D.commit D.parse tops Ω P m).Pr H) ∧
      ((∀ z ∈ Zs, ∀ i s, D.parse (x z i) = some s → CertifierRegisters H D.commit (lay s) (k s)
          (op (failCell (schedOf z) (arr (srvOf z))) i (runsBefore (run tap (schedOf z, srvOf z)).2 i)) s (tops s)
          ((run tap (schedOf z, srvOf z)).2 i)) →
        {z | z ∈ Zs ∧ failCount (failCell (schedOf z) (arr (srvOf z))) = j ∧
            FailsBelow bound (failCell (schedOf z) (arr (srvOf z))) ∧
            ∀ i, ExecutionPhaseGoodChecked H D L K S lay k δ reg (fun s r => valid₁ s r ∧ valid₂ s r) Dj (sw i)
              (ρ (failCell (schedOf z) (arr (srvOf z))) i) tops (x z i) (wit z i)
              ((run tap (schedOf z, srvOf z)).2 i)}.encard ≤
          ((failPatterns bound j : ℕ) : ℕ∞) *
            (Θ.encard * ((windowsCount lay k δ (fun s => reg s ∪ chk s) Dj sw Pw β : ℕ) : ℕ∞)))

/-- **The leak over a monitoring period, each failure charged as it happens.** `MonitoringPeriodLeakChecked`, where each
window's operator and registered values may follow the network certifier's fail cells over the period (`op φ i` and
`ρ φ i`, `φ` the fail cell of each connection, `failCell`), bound before the secrets and reading the pattern alone:

* each window's audit has `MonitoringPeriodLeakChecked`'s bound;
* where at every window's statement the network certifier registers its observations at that window's operator at the
  run's fail pattern (`CertifierRegisters`), for the secrets whose every window meets `ExecutionPhaseGoodChecked` at the
  registered values of the run's fail pattern:
  * **held**: the cells the client receives are read from the windows' records under a registered parameter of `Θ`,
    and each window's record is a sequence its registered Program accepts with a valid run (`AccSeqsChecked`), on a
    run with failed connections as on one with none;
  * **counted**: those whose run fails at most `j` connections, each below its bound, number at most
    `(Σ_{i ≤ j} failPatterns bound i) · #Θ · windowsCount`. The factor is one at `j = 0` and `∏_c (bound c + 1)`,
    `MonitoringPeriodLeakAccepted`'s factor for the cells, at every connection.

`Conn` is every connection slot of the period, over all its links, and `j` the period's failures: one factor for the
period, under one logarithm. -/
def MonitoringPeriodLeakFailsAtMost : Prop :=
  ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    (H : Claims.Msg → Claims.Msg) (D : Compiled Coin A) (hks : D.KS H) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (δ : ℝ) (hδ : 0 ≤ δ) (reg chk : AuditStatement → Finset ℕ)
    (valid₁ valid₂ : AuditStatement → (ℕ → Bool) → Prop) (ε₁ ε₂ : Claims.Msg → ℝ) (Dj : ℕ)
    {n : ℕ} (sw : Fin n → AuditStatement) (Pw : Fin n → Partition) (β : Fin n → ℕ)
    (hplaced : WindowsPlaced D.partition δ sw Pw β)
    (hchk : ∀ i, CheckedUnique (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i))
      (fun r => valid₁ (sw i) r ∧ valid₂ (sw i) r))
    {Conn : Type} [Fintype Conn] (ρ : (Conn → Option ℕ) → Fin n → ℕ → Bool)
    (op : (Conn → Option ℕ) → Fin n → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool)
    {Cell Srv Dev Z Par : Type} (arr : Srv → Conn → Arrivals Cell)
    (f : RecordFormat Conn Cell (Fin n → List (Obs L K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => recordingCode f p.1 (arr p.2))
    (rd : Par → (Fin n → List (Obs L K S)) → Conn → List (ℕ × CellView Cell))
    (allows : Par → CertifierSchedule Conn → Prop) (hread : f.ReadsBy rd allows)
    (Θ : Set Par) (Zs : Set Z) (schedOf : Z → CertifierSchedule Conn) (srvOf : Z → Srv)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z)
    (hch : MonitoringPeriodChannelReg f allows Θ Zs schedOf srvOf dec fun p => (run tap p).1)
    (x : Z → Fin n → Claims.Msg) (wit : Z → Fin n → Wit) (tops : AuditStatement → List (Opening Bool))
    (hb : D.commit.Binds H) (bound : Conn → ℕ)
    (j : ℕ),
    (∀ m, ScopeAcc D lay k m →
        CheckedWitness H D (certifierRun H D.commit L K S lay k tops) valid₁ ε₁ m →
        CheckedWitness H D (certifierRun H D.commit L K S lay k tops) valid₂ ε₂ m →
        ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω),
      Claims.PrBad H D.V
          (RLeakCertifierChecked H D.commit D.parse D.partition L K S lay k δ
            (fun s => AsReg s.prog (lay s) (k s) (reg s)) (fun s r => valid₁ s r ∧ valid₂ s r) Dj tops)
          D.E m P ≤
        D.κ δ m + ε₁ m + ε₂ m + Real.sqrt (D.γ m * Claims.PrB H D.V P m) + (Bext H D.V D.E Ω P m).Pr H +
          (Btap H D.V D.E D.commit D.parse tops Ω P m).Pr H) ∧
      ((∀ z ∈ Zs, ∀ i s, D.parse (x z i) = some s → CertifierRegisters H D.commit (lay s) (k s)
          (op (failCell (schedOf z) (arr (srvOf z))) i (runsBefore (run tap (schedOf z, srvOf z)).2 i)) s (tops s)
          ((run tap (schedOf z, srvOf z)).2 i)) →
        (∀ z ∈ Zs, (∀ i, ExecutionPhaseGoodChecked H D L K S lay k δ reg (fun s r => valid₁ s r ∧ valid₂ s r) Dj (sw i)
              (ρ (failCell (schedOf z) (arr (srvOf z))) i) tops (x z i) (wit z i)
              ((run tap (schedOf z, srvOf z)).2 i)) →
            ∃ θ ∈ Θ, (run tap (schedOf z, srvOf z)).1 = rd θ (run tap (schedOf z, srvOf z)).2 ∧
              ∀ i, (run tap (schedOf z, srvOf z)).2 i ∈
                AccSeqsChecked (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i))
                  (AsReg (sw i).prog (lay (sw i)) (k (sw i)) (reg (sw i))
                    (ρ (failCell (schedOf z) (arr (srvOf z))) i))
                  (fun r => valid₁ (sw i) r ∧ valid₂ (sw i) r) δ
                  (op (failCell (schedOf z) (arr (srvOf z))) i (runsBefore (run tap (schedOf z, srvOf z)).2 i)) Dj) ∧
          {z | z ∈ Zs ∧ failCount (failCell (schedOf z) (arr (srvOf z))) ≤ j ∧
              FailsBelow bound (failCell (schedOf z) (arr (srvOf z))) ∧
              ∀ i, ExecutionPhaseGoodChecked H D L K S lay k δ reg (fun s r => valid₁ s r ∧ valid₂ s r) Dj (sw i)
                (ρ (failCell (schedOf z) (arr (srvOf z))) i) tops (x z i) (wit z i)
                ((run tap (schedOf z, srvOf z)).2 i)}.encard ≤
            ((∑ i ∈ Finset.range (j + 1), failPatterns bound i : ℕ) : ℕ∞) *
              (Θ.encard * ((windowsCount lay k δ (fun s => reg s ∪ chk s) Dj sw Pw β : ℕ) : ℕ∞)))

/-- **The charge a period's failures pay, as one that adds over periods.** Under weights `w` on the number of failed
connections that sum to at most one (`KraftWeights`) and finitely many registered parameters `Θ`, a run that fails `j`
connections is charged `log2 (failPatterns bound j · #Θ · windowsCount / w j)` bits. For every `T`, the secrets whose
every window meets `ExecutionPhaseGoodChecked` and whose run is charged at most `log2 T` bits number at most `T`: the number of
failures is the sender's choice, and the weight charges for it, so the charges of two periods add. -/
def MonitoringPeriodLeakCharged : Prop :=
  ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    (H : Claims.Msg → Claims.Msg) (D : Compiled Coin A) (hks : D.KS H) (L K S : ℕ)
    (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (δ : ℝ) (hδ : 0 ≤ δ) (reg chk : AuditStatement → Finset ℕ)
    (valid₁ valid₂ : AuditStatement → (ℕ → Bool) → Prop) (ε₁ ε₂ : Claims.Msg → ℝ) (Dj : ℕ)
    {n : ℕ} (sw : Fin n → AuditStatement) (Pw : Fin n → Partition) (β : Fin n → ℕ)
    (hplaced : WindowsPlaced D.partition δ sw Pw β)
    (hchk : ∀ i, CheckedUnique (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i))
      (fun r => valid₁ (sw i) r ∧ valid₂ (sw i) r))
    {Conn : Type} [Fintype Conn] (ρ : (Conn → Option ℕ) → Fin n → ℕ → Bool)
    (op : (Conn → Option ℕ) → Fin n → List (List (Obs L K S)) → ℕ → List (Obs L K S) → List Bool)
    {Cell Srv Dev Z Par : Type} (arr : Srv → Conn → Arrivals Cell)
    (f : RecordFormat Conn Cell (Fin n → List (Obs L K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin n → List (Obs L K S)))
    (htrust : Verity.Assumptions.TrustedHost tap run fun p => recordingCode f p.1 (arr p.2))
    (rd : Par → (Fin n → List (Obs L K S)) → Conn → List (ℕ × CellView Cell))
    (allows : Par → CertifierSchedule Conn → Prop) (hread : f.ReadsBy rd allows)
    (Θ : Finset Par) (Zs : Set Z) (schedOf : Z → CertifierSchedule Conn) (srvOf : Z → Srv)
    (dec : (Conn → List (ℕ × CellView Cell)) → Z)
    (hch : MonitoringPeriodChannelReg f allows (Θ : Set Par) Zs schedOf srvOf dec fun p => (run tap p).1)
    (x : Z → Fin n → Claims.Msg) (wit : Z → Fin n → Wit) (tops : AuditStatement → List (Opening Bool))
    (hb : D.commit.Binds H) (bound : Conn → ℕ)
    (w : ℕ → ℚ≥0) (hw : KraftWeights w (Fintype.card Conn)) (T : ℕ),
    (∀ z ∈ Zs, ∀ i s, D.parse (x z i) = some s → CertifierRegisters H D.commit (lay s) (k s)
        (op (failCell (schedOf z) (arr (srvOf z))) i (runsBefore (run tap (schedOf z, srvOf z)).2 i)) s (tops s)
        ((run tap (schedOf z, srvOf z)).2 i)) →
      {z | z ∈ Zs ∧ FailsBelow bound (failCell (schedOf z) (arr (srvOf z))) ∧
          ((failPatterns bound (failCount (failCell (schedOf z) (arr (srvOf z)))) *
              (Θ.card * windowsCount lay k δ (fun s => reg s ∪ chk s) Dj sw Pw β) : ℕ) : ℚ≥0) ≤
            w (failCount (failCell (schedOf z) (arr (srvOf z)))) * (T : ℚ≥0) ∧
          ∀ i, ExecutionPhaseGoodChecked H D L K S lay k δ reg (fun s r => valid₁ s r ∧ valid₂ s r) Dj (sw i)
            (ρ (failCell (schedOf z) (arr (srvOf z))) i) tops (x z i) (wit z i)
            ((run tap (schedOf z, srvOf z)).2 i)}.encard ≤ (T : ℕ∞)

end NetTiming.Guarantees
