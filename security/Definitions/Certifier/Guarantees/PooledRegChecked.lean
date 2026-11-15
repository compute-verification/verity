module

public import Definitions.Core.Assumptions
public import Definitions.Certifier.Checked
public import Definitions.Certifier.Guarantees.MonitoringPeriodChecked
public import Definitions.Certifier.SampledReg
public import Definitions.Certifier.MonitoringPeriodConditions
public import Definitions.Certifier.PooledReg
public import Definitions.Certifier.Tap

/-!
# DRAFT (scratch branch, in no root and no lock): the checked bound pooled over a monitoring period

`PooledLeakRegistered` with each phase's checked inputs `chk` left out of the count. The good event is
`PooledGoodChecked` at the incorrect sets `WregR` (the trace read at the statement with the registered openings), each
phase's operator reading the phases before it (`runsBefore`). The one input the checked form adds per phase is that
its check holds on the run the leak relation at `WregR` is accepted at (`regRunR`), except with probability `εchk i`
(`PhaseChecks`). That run is fixed before the phase's coins: `CertifierRegisters` fixes the operator's moves at the
ingress the registered openings hold.

The two definitions `regRunR`, `PrCheckMissR` and the premise `PhaseChecks` would go in the vocabulary module
(`Definitions/Certifier/PooledReg.lean` or a `PooledRegChecked.lean` beside it); they are here so the draft is one file.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling Finset
open Claims.SamplingGeneric (Arg Compiled)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The run at the trace with the registered openings**: `C` over the horizon on the trace read at the statement with
the registered openings, filled at the network certifier's values (`fillTrace` of `regValR`), its ingress the values the
registered openings hold at each step's ingress gates, and the incorrect proof blocks of `FregR` held at that trace
(`heldAdv`). Where the network certifier registers a record among the registered openings (`CertifierRegisters`), it is
the run `leakAt_regR`'s accepted run is: the operator's moves there are those ingress values. It reads the statement,
the registered openings and the prover's strategy, and no tape. -/
noncomputable def regRunR (H : Msg → Msg) (D : Compiled Coin A) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg)
    (ops : AuditStatement → List (Opening Bool)) {L S : ℕ} (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (s : AuditStatement) (Pt : Partition) : ℕ → Bool :=
  valOver s.prog
    ((lay s).inputOf (k s) ((List.range (k s)).map fun t => ((lay s).ingress t).map (certifierVal (ops s)))
      (traceVal s.prog (fillTrace (regValR H D P x (ops s)) (certifierVal (ops s)))))
    (heldAdv s.prog Pt (FregR H D P x ops) (fillTrace (regValR H D P x (ops s)) (certifierVal (ops s))))

/-- **The check's miss at the trace with the registered openings**: the run accepts at a statement with a partition
whose `FregR` holds no proof block of a class sampled `all`, while the run at that trace (`regRunR`) is not `valid`.
The run is fixed before the coins, so this is the acceptance probability where that run is not valid, and `0` where it
is. -/
noncomputable def PrCheckMissR (H : Msg → Msg) (D : Compiled Coin A) {L S : ℕ}
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop)
    (x : Msg) {Ω : Type} [Fintype Ω] (P : Prover Ω) (ops : AuditStatement → List (Opening Bool)) : ℝ :=
  Pr fun p : Ω × (Fin (D.V.k x) → Coin) => (run H P D.V x p.1 p.2).out = .acc ∧
    ∃ s Pt, D.parse x = some s ∧ D.partition s = some Pt ∧ Disjoint (FregR H D P x ops) (allSlots s.design Pt) ∧
      ¬valid s (regRunR H D P x ops lay k s Pt)

section Phases

variable {m kV : ℕ} {Tp Sec : Type}

/-- **`εchk i` bounds execution phase `i`'s check miss** at the trace with the registered openings, at every secret and
every history of the period's coins (`PrCheckMissR` at the phase's statement, prover's strategy and registered
openings), the run judged by the check registered for the phase, `valid (sw i)`: the registration (`SameRegistration`)
fixes the phase's program, layout and steps but not its check. -/
def PhaseChecks [Fintype Tp] (H : Msg → Msg) (D : Msg → Compiled Coin A) {L S : ℕ}
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop)
    (sw : Fin m → AuditStatement) (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp)
    (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool))
    (εchk : Fin m → ℝ) : Prop :=
  ∀ z ω i, PrCheckMissR H (D (x z ω i)) lay k (fun _ => valid (sw i)) (x z ω i) (Pv z ω i) (ops z ω i) ≤ εchk i

end Phases

end NetTiming

namespace NetTiming.Guarantees

open Claims Claims.Sampling Finset
open Claims.SamplingGeneric (Arg Compiled)

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak through the network certifier over a monitoring period, at the trace with the registered openings, with
the checked inputs left out.** `PooledLeakRegistered`'s setting, with the network certifier's one run giving the cells
the client receives and the phases' observations (`trusted-device`), reading back under a registered parameter of `Θ`
(`RecordFormat.ReadsBy`, `MonitoringPeriodChannelReg` at each sample of the coins), and each phase's check pinning its
checked inputs (`CheckedUnique`) and holding on the run at its trace with the registered openings except with
probability `εchk i` (`PhaseChecks`). For every secret:

* every phase is accepted, registered as the period's and with a complete record while the period's good event with
  valid runs fails (`PooledGoodChecked` at `WregR`) with probability at most `(1 - q)^Fmax + 2 Σ_i e i + Σ_i εchk i`;
* at every sample of the coins, the secrets whose every phase is registered as the period's and that meet that good
  event number at most `#Θ · (N + 1)^Fmax · 2^(b · Fmax) · ∏_i 2^#(freeInputs_i \ (reg_i ∪ chk_i)) · Dj^k_i`,
  `N = Σ_i n_i` the period's proof blocks and `b` the most committed bits of one of them. -/
def PooledLeakRegisteredChecked : Prop :=
  ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} (H : Claims.Msg → Claims.Msg)
    (D : Claims.Msg → Compiled Coin A) (hA : A.KS H) (hb : ∀ y, (D y).commit.Binds H) (hS : ∀ y, (D y).Sessions H)
    {Lk K S : ℕ} (lay : AuditStatement → BoundaryGates Lk S) (k : AuditStatement → ℕ)
    (reg chk : AuditStatement → Finset ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop) (Dj : ℕ) {m kV : ℕ}
    {Tp Sec : Type} [Fintype Tp] [Nonempty Tp]
    (sw : Fin m → AuditStatement) (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool)
    (hchk : ∀ i, CheckedUnique (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i)) (valid (sw i)))
    (op : Fin m → List (List (Obs Lk K S)) → ℕ → List (Obs Lk K S) → List Bool)
    (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Claims.Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Claims.Prover Tp) (hk : PhaseRounds D kV x)
    (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool))
    (had : PhasesAdaptive x Pv ops)
    (hops : ∀ z ω i s, (D (x z ω i)).parse (x z ω i) = some s →
      RegOpens H (D (x z ω i)).commit s (reg s) (ops z ω i s))
    (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hrate : ∀ z ω i s Pt, (D (x z ω i)).parse (x z ω i) = some s →
      (D (x z ω i)).partition s = some Pt → RateAt s Pt q)
    (e : Fin m → ℝ) (he : PhaseErrors H D x Pv ops e)
    (εchk : Fin m → ℝ) (hck : PhaseChecks H D lay k valid sw x Pv ops εchk)
    (hfit : ∀ i, (lay (sw i)).Fits (sw i).prog (k (sw i)))
    {Conn Cell Srv Dev Par : Type} (arr : Srv → Conn → Arrivals Cell)
    (f : RecordFormat Conn Cell (Fin m → List (Obs Lk K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin m → List (Obs Lk K S)))
    (htrust : Verity.Assumptions.TrustedDevice tap run fun p => recordingCode f p.1 (arr p.2))
    (rd : Par → (Fin m → List (Obs Lk K S)) → Conn → List (ℕ × CellView Cell))
    (allows : Par → CertifierSchedule Conn → Prop) (hread : f.ReadsBy rd allows) (Θ : Set Par) (Zs : Set Sec)
    (schedOf : Sec → (Fin m → Tp × (Fin kV → Coin)) → CertifierSchedule Conn)
    (srvOf : Sec → (Fin m → Tp × (Fin kV → Coin)) → Srv)
    (dec : (Fin m → Tp × (Fin kV → Coin)) → (Conn → List (ℕ × CellView Cell)) → Sec)
    (hch : ∀ ω, MonitoringPeriodChannelReg f allows Θ Zs (fun z => schedOf z ω) (fun z => srvOf z ω) (dec ω)
      fun p => (run tap p).1)
    (hcert : ∀ z ω i s, (D (x z ω i)).parse (x z ω i) = some s → phaseAcc H D x Pv hk z i ω (ω i) →
      CertifierRegisters H (D (x z ω i)).commit (lay s) (k s)
        (op i (runsBefore (run tap (schedOf z ω, srvOf z ω)).2 i)) s (ops z ω i s)
        ((run tap (schedOf z ω, srvOf z ω)).2 i))
    (Fmax : ℕ),
    (∀ z ∈ Zs, Pr (fun ω => (∀ i, phaseAcc H D x Pv hk z i ω (ω i) ∧
          PhaseRegistered D lay k reg sw Pw ρ x ops z ω i ∧
          RecordComplete k Dj sw (fun z ω => (run tap (schedOf z ω, srvOf z ω)).2) z ω i) ∧
        ¬PooledGoodChecked lay k reg valid Dj sw Pw ρ
          (fun i => op i (runsBefore (run tap (schedOf z ω, srvOf z ω)).2 i)) Fmax (WregR H D x Pv ops z ω)
          (run tap (schedOf z ω, srvOf z ω)).2) ≤
      (1 - q) ^ Fmax + 2 * ∑ i, e i + ∑ i, εchk i) ∧
    ∀ ω, {z | z ∈ Zs ∧ (∀ i, PhaseRegistered D lay k reg sw Pw ρ x ops z ω i) ∧
        PooledGoodChecked lay k reg valid Dj sw Pw ρ
          (fun i => op i (runsBefore (run tap (schedOf z ω, srvOf z ω)).2 i)) Fmax (WregR H D x Pv ops z ω)
          (run tap (schedOf z ω, srvOf z ω)).2}.encard ≤
      Θ.encard * ((((∑ i, (Pw i).blocks.length) + 1) ^ Fmax *
        2 ^ ((univ.sup fun i => (range (Pw i).blocks.length).sup fun v => faultyBits (sw i).prog (Pw i) {v}) * Fmax) *
        ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ (reg (sw i) ∪ chk (sw i))).card *
          Dj ^ k (sw i)) : ℕ) : ℕ∞)

/-- **Candidate A: `MonitoringPeriodLeakChecked`'s good event, its windows' failures united over the period's live
coins.** The period's setting as in `PooledLeakRegisteredChecked`, at one compiled verifier `D`, the network
certifier's openings at phase `i` being its registered openings `ops z ω i` (so they read only the phases before `i`).
For every secret, every phase is accepted, registered and with a complete record while some phase has no witness meeting
`ExecutionPhaseGoodChecked` with probability at most `Σ_i e' i`, `e' i` at least `MonitoringPeriodLeakChecked`'s six
terms plus the registered openings' binding term (`Breg`, for `RegisteredInputsHeld` at the extracted witness); and,
where the network certifier registers every phase's record, the secrets whose every phase has such a witness number at
most `#Θ · executionPhasesCount(reg ∪ chk)`. Every phase pays `κ δ`: nothing is pooled but the union. -/
def PooledLeakCheckedPhases : Prop :=
  ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} (H : Claims.Msg → Claims.Msg)
    (D : Compiled Coin A) (hks : D.KS H) (hb : D.commit.Binds H) (Lk K S : ℕ)
    (lay : AuditStatement → BoundaryGates Lk S) (k : AuditStatement → ℕ) (δ : ℝ) (hδ : 0 ≤ δ)
    (reg chk : AuditStatement → Finset ℕ) (valid₁ valid₂ : AuditStatement → (ℕ → Bool) → Prop)
    (ε₁ ε₂ : Claims.Msg → ℝ) (Dj : ℕ) {m kV : ℕ} {Tp Sec : Type} [Fintype Tp] [Nonempty Tp]
    (sw : Fin m → AuditStatement) (Pw : Fin m → Partition) (β : Fin m → ℕ)
    (hplaced : ExecutionPhasesPlaced D.partition δ sw Pw β)
    (hchk : ∀ i, CheckedUnique (sw i).prog (Pw i) (sw i).design (lay (sw i)) (k (sw i)) (chk (sw i))
      (fun r => valid₁ (sw i) r ∧ valid₂ (sw i) r))
    (ρ : Fin m → ℕ → Bool) (op : Fin m → List (List (Obs Lk K S)) → ℕ → List (Obs Lk K S) → List Bool)
    (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Claims.Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Claims.Prover Tp) (hk : PhaseRounds (fun _ => D) kV x)
    (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool))
    (had : PhasesAdaptive x Pv ops)
    (hw : ∀ z ω i, ScopeAcc D lay k (x z ω i) →
      CheckedWitness H D (certifierRun H D.commit Lk K S lay k (ops z ω i)) valid₁ ε₁ (x z ω i) ∧
        CheckedWitness H D (certifierRun H D.commit Lk K S lay k (ops z ω i)) valid₂ ε₂ (x z ω i))
    (e' : Fin m → ℝ)
    (he' : ∀ z ω i, D.κ δ (x z ω i) + ε₁ (x z ω i) + ε₂ (x z ω i) +
        Real.sqrt (D.γ (x z ω i) * Claims.PrB H D.V (Pv z ω i) (x z ω i)) +
        (Bext H D.V D.E Tp (Pv z ω i) (x z ω i)).Pr H +
        (Btap H D.V D.E D.commit D.parse (ops z ω i) Tp (Pv z ω i) (x z ω i)).Pr H +
        (Breg H D.V D.E D.commit D.parse (ops z ω i) Tp (Pv z ω i) (x z ω i)).Pr H ≤ e' i)
    (hfit : ∀ i, (lay (sw i)).Fits (sw i).prog (k (sw i)))
    {Conn Cell Srv Dev Par : Type} (arr : Srv → Conn → Arrivals Cell)
    (f : RecordFormat Conn Cell (Fin m → List (Obs Lk K S))) (tap : Dev)
    (run : Dev → CertifierSchedule Conn × Srv → (Conn → List (ℕ × CellView Cell)) × (Fin m → List (Obs Lk K S)))
    (htrust : Verity.Assumptions.TrustedDevice tap run fun p => recordingCode f p.1 (arr p.2))
    (rd : Par → (Fin m → List (Obs Lk K S)) → Conn → List (ℕ × CellView Cell))
    (allows : Par → CertifierSchedule Conn → Prop) (hread : f.ReadsBy rd allows) (Θ : Set Par) (Zs : Set Sec)
    (schedOf : Sec → (Fin m → Tp × (Fin kV → Coin)) → CertifierSchedule Conn)
    (srvOf : Sec → (Fin m → Tp × (Fin kV → Coin)) → Srv)
    (dec : (Fin m → Tp × (Fin kV → Coin)) → (Conn → List (ℕ × CellView Cell)) → Sec)
    (hch : ∀ ω, MonitoringPeriodChannelReg f allows Θ Zs (fun z => schedOf z ω) (fun z => srvOf z ω) (dec ω)
      fun p => (run tap p).1),
    (∀ z ∈ Zs, Pr (fun ω => (∀ i, phaseAcc H (fun _ => D) x Pv hk z i ω (ω i) ∧
          PhaseRegistered (fun _ => D) lay k reg sw Pw ρ x ops z ω i ∧
          RecordComplete k Dj sw (fun z ω => (run tap (schedOf z ω, srvOf z ω)).2) z ω i) ∧
        ¬∀ i, ∃ w, ExecutionPhaseGoodChecked H D Lk K S lay k δ reg (fun s r => valid₁ s r ∧ valid₂ s r) Dj (sw i)
          (ρ i) (ops z ω i) (x z ω i) w ((run tap (schedOf z ω, srvOf z ω)).2 i)) ≤ ∑ i, e' i) ∧
    ∀ ω, (∀ z ∈ Zs, ∀ i s, D.parse (x z ω i) = some s → CertifierRegisters H D.commit (lay s) (k s)
        (op i (runsBefore (run tap (schedOf z ω, srvOf z ω)).2 i)) s (ops z ω i s)
        ((run tap (schedOf z ω, srvOf z ω)).2 i)) →
      {z | z ∈ Zs ∧ ∀ i, ∃ w, ExecutionPhaseGoodChecked H D Lk K S lay k δ reg
          (fun s r => valid₁ s r ∧ valid₂ s r) Dj (sw i) (ρ i) (ops z ω i) (x z ω i) w
          ((run tap (schedOf z ω, srvOf z ω)).2 i)}.encard ≤
        Θ.encard * ((executionPhasesCount lay k δ (fun s => reg s ∪ chk s) Dj sw Pw β : ℕ) : ℕ∞)

end NetTiming.Guarantees
