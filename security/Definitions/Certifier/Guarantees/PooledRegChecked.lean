module

public import Definitions.Core.Assumptions
public import Definitions.Certifier.MonitoringPeriodConditions
public import Definitions.Certifier.PooledRegChecked
public import Definitions.Certifier.Tap

/-!
# The checked bound pooled over a monitoring period, at the trace with the registered openings

`PooledLeakRegistered` with each phase's checked inputs `chk` left out of the count. The good event is
`PooledGoodChecked` at the incorrect sets `WregR` (the trace read at the statement with the registered openings), each
phase's operator reading the phases before it (`runsBefore`). The one premise the checked form adds per phase is that
its check holds on the run the leak relation at `WregR` is accepted at (`regRunR`), except with probability `εchk i`
(`PhaseChecks`). That run is fixed before the phase's coins: `CertifierRegisters` fixes the operator's moves at the
ingress the registered openings hold.
-/

@[expose] public section

namespace NetTiming.Guarantees

open Claims Claims.Sampling Finset
open Claims.SamplingGeneric (Arg Compiled)

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak through the network attester over a monitoring period, at the trace with the registered openings, with
the checked inputs left out, given `PhaseChecks`**: a conditional bound, whose premise `PhaseChecks` is open and
discharged by no check today. `valid` is any `Prop` on a run, so the statement alone would let a reader pick a `valid`
no verifier evaluates and move the whole count into `εchk`. The check meant is a deployed one that pins its checked
inputs (`CheckedUnique`) with a proved bound on its miss, `MemCheck_v1`'s offline check run every run on its own live
coins; until that check's miss bound is proved at `regRunR`, the number to quote is `PooledLeakRegistered`'s, without
`chk`. `PooledLeakRegistered`'s setting, with the network attester's one run giving the cells the client receives and
the phases' observations (`trusted-device`), reading back under a registered parameter of `Θ` (`RecordFormat.ReadsBy`,
`MonitoringPeriodChannelReg` at each sample of the coins), and each phase's check pinning its checked inputs
(`CheckedUnique`) and holding on the run at its trace with the registered openings except with
probability `εchk i` (`PhaseChecks`). For every secret:

* every phase is accepted, registered as the period's and with a complete record while the period's good event with
  valid runs fails (`PooledGoodChecked` at `WregR`) with probability at most `(1 - q)^Fmax + Σ_i e i + Σ_i εchk i`;
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
      (1 - q) ^ Fmax + ∑ i, e i + ∑ i, εchk i) ∧
    ∀ ω, {z | z ∈ Zs ∧ (∀ i, PhaseRegistered D lay k reg sw Pw ρ x ops z ω i) ∧
        PooledGoodChecked lay k reg valid Dj sw Pw ρ
          (fun i => op i (runsBefore (run tap (schedOf z ω, srvOf z ω)).2 i)) Fmax (WregR H D x Pv ops z ω)
          (run tap (schedOf z ω, srvOf z ω)).2}.encard ≤
      Θ.encard * ((((∑ i, (Pw i).blocks.length) + 1) ^ Fmax *
        2 ^ ((univ.sup fun i => (range (Pw i).blocks.length).sup fun v => faultyBits (sw i).prog (Pw i) {v}) * Fmax) *
        ∏ i, (2 ^ (freeInputs (sw i).prog (lay (sw i)) (k (sw i)) \ (reg (sw i) ∪ chk (sw i))).card *
          Dj ^ k (sw i)) : ℕ) : ℕ∞)

end NetTiming.Guarantees
