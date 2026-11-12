module

public import Definitions.NetworkCertifier.Guarantees.PooledReg
public import Proofs.Flock.SamplingGeneric.Session
public import Proofs.NetworkCertifier.PooledLeak
public import Proofs.NetworkCertifier.PooledRegEvent
public import Proofs.NetworkCertifier.PooledRegLeak
public import Proofs.NetworkCertifier.PooledRegSample
import Proofs.Flock.Soundness.Audit.PeriodsCharge

/-!
# The pooled bound at the trace with the registered openings

`NetTiming.pr_pooled_ex_le` bounds a monitoring period's good event from two inputs at each execution phase, the
first with a charge the period pays once. At the incorrect sets `WregR` both are proved: the bound on each phase's
verdict from the sample and the event at `FregR` (`phasesMiss_of_event`), and each accepted phase's leak relation with
no error (`phaseLeaks_regR`).

* `pooled_of_event` is the bound from any bound on each phase's event, the probability that its run accepts with a
  sampled proof block of its `FregR`: `e i` plus the average, over the phase's own coins, of a charge `cl z ω i` that
  is not negative and reads only the earlier phases' coins. It takes no knowledge soundness of the proof system and no
  binding: a caller meets the event with its own bound, the square-root form (`Compiled.ksRegR_of_sessionKS`) or one
  from costs (`Compiled.ksRegR_of_costs`). Its event holds, beside each phase's verdict, whatever else `R` says of the
  phase.
* `pooledLeakRegistered` is `Guarantees.PooledLeakRegistered`: `pooled_of_event` at the square-root form, the sessions'
  bound from the proof system's knowledge soundness (`Claims.SamplingGeneric.sessionKSOf`), with `R` the phase
  registered as the period's and its record complete.
* What follows from the guarantee. `pooledLeakRegistered_zero`: at the charge zero the bound is
  `(1 - q)^Fmax + Σ e_i`. `pooledLeakRegistered_budget`: where the charges' sum along the period's coins is at most
  `Q` at every coins, the bound is `(1 - q)^Fmax + Σ e_i + Q`. `chargeAdaptive_of_phases`: a charge that reads the
  period's coins only through the phase's statement, its prover's strategy and its registered openings reads only the
  earlier phases' coins, where those do.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling Finset
open Claims.SamplingGeneric (Arg Compiled)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} {m kV : ℕ} {Tp Sec : Type}

/-- **The pooled bound from any bound on each execution phase's event**, the sample's term as `(1 - q)^Fmax`: where
`e i` plus the average over phase `i`'s own coins of the charge `cl z ω i` bounds the probability that the phase's run
accepts with a sampled proof block of its `FregR`, every phase is accepted with `R` outside the period's good event
with probability at most `(1 - q)^Fmax + Σ e_i` plus the average, over the period's coins, of the charges' sum. `R` is
what else the event says of a phase, under which `hyp` gives the phase's leak relation its inputs. It takes no
knowledge soundness of the proof system and no binding: a caller meets `hev` with its own bound on the event. -/
theorem pooled_of_event [Fintype Tp] [Nonempty Tp] {H : Msg → Msg} {D : Msg → Compiled Coin A}
    (hS : ∀ y, (D y).Sessions H) {Lk K S : ℕ} (lay : AuditStatement → BoundaryGates Lk S)
    (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ) (sw : Fin m → AuditStatement)
    (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool)
    (op : Fin m → List (List (Obs Lk K S)) → ℕ → List (Obs Lk K S) → List Bool) (Zs : Set Sec)
    (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp) (hk : PhaseRounds D kV x)
    (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool))
    (obs : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → List (Obs Lk K S))
    (R : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prop) (had : PhasesAdaptive x Pv ops)
    {q : ℝ} (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hrate : ∀ z ∈ Zs, ∀ ω i s Pt, (D (x z ω i)).parse (x z ω i) = some s →
      (D (x z ω i)).partition s = some Pt → RateAt s Pt q)
    {e : Fin m → ℝ} (he0 : ∀ i, 0 ≤ e i)
    (cl : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Tp × (Fin kV → Coin) → ℝ) (hcl : ChargeAdaptive cl)
    (hcl0 : ∀ z ω i c, 0 ≤ cl z ω i c)
    (hev : ∀ z ∈ Zs, ∀ ω i, (D (x z ω i)).Scope (x z ω i) →
      Pr (fun p : Tp × (Fin ((D (x z ω i)).V.k (x z ω i)) → Coin) =>
        (run H (Pv z ω i) (D (x z ω i)).V (x z ω i) p.1 p.2).out = .acc ∧
          ∃ u ∈ (D (x z ω i)).checked (x z ω i) p.2,
            u ∈ FregR H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i)) ≤ e i + Ex (cl z ω i))
    (hyp : ∀ z ∈ Zs, ∀ ω i s, (D (x z ω i)).parse (x z ω i) = some s → phaseAcc H D x Pv hk z i ω (ω i) →
      R z ω i →
      SameRegistration (D (x z ω i)) lay k reg (sw i) s ∧ (D (x z ω i)).partition s = some (Pw i) ∧
        (lay s).Fits s.prog (k s) ∧
        CertifierRegisters H (D (x z ω i)).commit (lay s) (k s) (op i (runsBefore (obs z ω) i)) s (ops z ω i s)
          (obs z ω i) ∧
        Covered (k s) (obs z ω i) ∧ (obs z ω i).length = k s ∧ (∀ y ∈ obs z ω i, y.jitter < Dj) ∧
        ∀ g ∈ reg s, (∃ o ∈ ops z ω i s, o.pos = g) ∧ certifierVal (ops z ω i s) g = ρ i g)
    (Fmax : ℕ) :
    ∀ z ∈ Zs, Pr (fun ω => (∀ i, phaseAcc H D x Pv hk z i ω (ω i) ∧ R z ω i) ∧
        ¬PooledGood lay k reg Dj sw Pw ρ op Fmax (WregR H D x Pv ops z ω) (obs z ω)) ≤
      (1 - q) ^ Fmax + ∑ i, e i + Ex fun ω => ∑ i, cl z ω i (ω i) := by
  intro z hz
  have hmiss := phasesMiss_of_event hS Zs x Pv hk ops hq1 hrate (fun z ω i => e i + Ex (cl z ω i))
    (fun z ω i => add_nonneg (he0 i) (Ex_nonneg fun c => hcl0 z ω i c)) hev z hz
  have h := NetTiming.pr_pooled_ex_le (phaseAcc H D x Pv hk z) (phaseAcc_prefix H D x Pv hk had.1 had.2.1 z)
    (WregR H D x Pv ops z) (fun i ω ω' hω => wreg_prefix H D had z i ω ω' hω) e (fun _ => 0) he0
    (fun i ω => cl z ω i) (fun i ω ω' hω => hcl z i ω ω' hω) (fun i ω c => hcl0 z ω i c) q hq0 hq1
    (fun i ω => (hmiss i ω).trans_eq (add_assoc _ _ _).symm)
    (fun i ω => R z ω i → LeakAt lay k reg Dj (sw i) (Pw i) (ρ i) (op i (runsBefore (obs z ω) i))
      (WregR H D x Pv ops z ω i) (obs z ω i))
    (phaseLeaks_regR hS lay k reg Dj sw Pw ρ op Zs x Pv hk ops obs R hyp z hz) Fmax
  simp only [Finset.sum_const_zero, add_zero] at h
  refine le_trans (Claims.Pr_mono fun ω hω => ?_) h
  exact ⟨fun i => (hω.1 i).1, fun hg => hω.2 (And.intro hg.1 fun i => hg.2 i (hω.1 i).2)⟩

/-- **The leak through the network certifier over a monitoring period, at the trace with the registered openings**:
`Guarantees.PooledLeakRegistered`. Each phase's event is bounded by the square-root form
(`Compiled.ksRegR_of_sessionKS`), the sessions' bound from the proof system's knowledge soundness, which is at most
`e i` plus the charge's average (`PhaseErrors`); the registered openings give that they pass under each statement's
roots and that every registered input has one (`RegOpens`); and a phase's layout fits, and its record is complete over
its statement's steps, because its statement is registered as the period's. -/
theorem pooledLeakRegistered : Guarantees.PooledLeakRegistered := by
  intro Coin _ _ _ A H D hA hb hS Lk K S lay k reg Dj m kV Tp Sec _ _ sw Pw ρ op x Pv hk ops obs had hops q hq0 hq1
    hrate e he0 cl hcl hcl0 he hfit hcert Fmax z
  refine pooled_of_event hS lay k reg Dj sw Pw ρ op Set.univ x Pv hk ops obs
    (fun z ω i => PhaseRegistered D lay k reg sw Pw ρ x ops z ω i ∧ RecordComplete k Dj sw obs z ω i) had hq0 hq1
    (fun z _ => hrate z) he0 cl hcl hcl0
    (fun z _ ω i hsc =>
      (Compiled.ksRegR_of_sessionKS (hb (x z ω i)) (hS (x z ω i))
        (Claims.SamplingGeneric.sessionKSOf Coin A (D (x z ω i)) H hA (hS (x z ω i))) (ops z ω i) hsc
        (fun s hs => (hops z ω i s hs).1) Tp (Pv z ω i)).trans (he z ω i))
    (fun z _ ω i s hs hacc hR => ?_) Fmax z (Set.mem_univ z)
  obtain ⟨⟨s', hs', hsame, hPw, hval⟩, hcov, hlen, hj⟩ := hR
  have hss : s' = s := Option.some.inj (hs'.symm.trans hs)
  rw [hss] at hsame hPw hval
  have hks : k s = k (sw i) := hsame.2.2.2.1
  have hfit' : (lay s).Fits s.prog (k s) := by
    rw [hsame.1, hsame.2.2.1, hks]
    exact hfit i
  have hcov' : Covered (k s) (obs z ω i) := by
    rw [hks]
    exact hcov
  exact ⟨hsame, hPw, hfit', hcert z ω i s hs hacc, hcov', hlen.trans hks.symm, hj,
    fun g hg => ⟨(hops z ω i s hs).2 g hg, hval g hg⟩⟩

/-! ## What follows from the guarantee -/

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **A charge that reads the period's coins only through the phase reads only the earlier phases' coins**: where
each phase's statement, its prover's strategy and its registered openings read only the earlier phases' coins
(`PhasesAdaptive`), so does any charge `g z i` of those three and the phase's own coins. -/
theorem chargeAdaptive_of_phases {x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg}
    {Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp}
    {ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool)}
    (had : PhasesAdaptive x Pv ops)
    (g : Sec → Fin m → Msg → Prover Tp → (AuditStatement → List (Opening Bool)) → Tp × (Fin kV → Coin) → ℝ) :
    ChargeAdaptive fun z ω i => g z i (x z ω i) (Pv z ω i) (ops z ω i) := by
  intro z i ω ω' h
  show g z i (x z ω i) (Pv z ω i) (ops z ω i) = g z i (x z ω' i) (Pv z ω' i) (ops z ω' i)
  rw [had.1 z i ω ω' h, had.2.1 z i ω ω' h, had.2.2 z i ω ω' h]

set_option linter.unusedVariables false in
/-- **`PooledLeakRegistered` at the charge zero**: its statement where `e i` alone bounds each phase's error, with the
bound `(1 - q)^Fmax + Σ_i e i`. Each `e i` is then not negative, since it bounds an error that is not. -/
theorem pooledLeakRegistered_zero (h : Guarantees.PooledLeakRegistered) :
    ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} (H : Msg → Msg)
      (D : Msg → Compiled Coin A) (hA : A.KS H) (hb : ∀ y, (D y).commit.Binds H) (hS : ∀ y, (D y).Sessions H)
      {Lk K S : ℕ} (lay : AuditStatement → BoundaryGates Lk S) (k : AuditStatement → ℕ)
      (reg : AuditStatement → Finset ℕ) (Dj : ℕ) {m kV : ℕ} {Tp Sec : Type} [Fintype Tp] [Nonempty Tp]
      (sw : Fin m → AuditStatement) (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool)
      (op : Fin m → List (List (Obs Lk K S)) → ℕ → List (Obs Lk K S) → List Bool)
      (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
      (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp) (hk : PhaseRounds D kV x)
      (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool))
      (obs : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → List (Obs Lk K S)) (had : PhasesAdaptive x Pv ops)
      (hops : ∀ z ω i s, (D (x z ω i)).parse (x z ω i) = some s →
        RegOpens H (D (x z ω i)).commit s (reg s) (ops z ω i s))
      (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
      (hrate : ∀ z ω i s Pt, (D (x z ω i)).parse (x z ω i) = some s →
        (D (x z ω i)).partition s = some Pt → RateAt s Pt q)
      (e : Fin m → ℝ) (he : PhaseErrors H D x Pv ops e fun _ _ _ _ => 0)
      (hfit : ∀ i, (lay (sw i)).Fits (sw i).prog (k (sw i)))
      (hcert : ∀ z ω i s, (D (x z ω i)).parse (x z ω i) = some s → phaseAcc H D x Pv hk z i ω (ω i) →
        CertifierRegisters H (D (x z ω i)).commit (lay s) (k s) (op i (runsBefore (obs z ω) i)) s (ops z ω i s)
          (obs z ω i))
      (Fmax : ℕ) (z : Sec),
      Pr (fun ω => (∀ i, phaseAcc H D x Pv hk z i ω (ω i) ∧
            PhaseRegistered D lay k reg sw Pw ρ x ops z ω i ∧ RecordComplete k Dj sw obs z ω i) ∧
          ¬PooledGood lay k reg Dj sw Pw ρ op Fmax (WregR H D x Pv ops z ω) (obs z ω)) ≤
        (1 - q) ^ Fmax + ∑ i, e i := by
  intro Coin _ _ _ A H D hA hb hS Lk K S lay k reg Dj m kV Tp Sec _ _ sw Pw ρ op x Pv hk ops obs had hops q hq0 hq1
    hrate e he hfit hcert Fmax z
  have he0 : ∀ i, 0 ≤ e i := fun i => by
    have h1 := he z (fun _ => (Classical.arbitrary Tp, fun _ => default)) i
    simp only [Ex_const, add_zero] at h1
    exact le_trans (add_nonneg (Compiled.εregX_nonneg H _ _ _) (Claims.Pr_nonneg _)) h1
  have h0 := h H D hA hb hS lay k reg Dj sw Pw ρ op x Pv hk ops obs had hops q hq0 hq1 hrate e he0
    (fun _ _ _ _ => 0) (fun _ _ _ _ _ => rfl) (fun _ _ _ _ => le_rfl) he hfit hcert Fmax z
  simp only [Finset.sum_const_zero, Ex_const, add_zero] at h0
  exact h0

set_option linter.unusedVariables false in
/-- **`PooledLeakRegistered` at a budget**: where the charges' sum along the period's coins is at most `Q` at every
coins, the bound is `(1 - q)^Fmax + Σ_i e i + Q`. The budget is paid once over the period, however the phases share
it. -/
theorem pooledLeakRegistered_budget (h : Guarantees.PooledLeakRegistered) :
    ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} (H : Msg → Msg)
      (D : Msg → Compiled Coin A) (hA : A.KS H) (hb : ∀ y, (D y).commit.Binds H) (hS : ∀ y, (D y).Sessions H)
      {Lk K S : ℕ} (lay : AuditStatement → BoundaryGates Lk S) (k : AuditStatement → ℕ)
      (reg : AuditStatement → Finset ℕ) (Dj : ℕ) {m kV : ℕ} {Tp Sec : Type} [Fintype Tp] [Nonempty Tp]
      (sw : Fin m → AuditStatement) (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool)
      (op : Fin m → List (List (Obs Lk K S)) → ℕ → List (Obs Lk K S) → List Bool)
      (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
      (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp) (hk : PhaseRounds D kV x)
      (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool))
      (obs : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → List (Obs Lk K S)) (had : PhasesAdaptive x Pv ops)
      (hops : ∀ z ω i s, (D (x z ω i)).parse (x z ω i) = some s →
        RegOpens H (D (x z ω i)).commit s (reg s) (ops z ω i s))
      (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
      (hrate : ∀ z ω i s Pt, (D (x z ω i)).parse (x z ω i) = some s →
        (D (x z ω i)).partition s = some Pt → RateAt s Pt q)
      (e : Fin m → ℝ) (he0 : ∀ i, 0 ≤ e i)
      (cl : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Tp × (Fin kV → Coin) → ℝ) (hcl : ChargeAdaptive cl)
      (hcl0 : ∀ z ω i c, 0 ≤ cl z ω i c) (he : PhaseErrors H D x Pv ops e cl)
      (hfit : ∀ i, (lay (sw i)).Fits (sw i).prog (k (sw i)))
      (hcert : ∀ z ω i s, (D (x z ω i)).parse (x z ω i) = some s → phaseAcc H D x Pv hk z i ω (ω i) →
        CertifierRegisters H (D (x z ω i)).commit (lay s) (k s) (op i (runsBefore (obs z ω) i)) s (ops z ω i s)
          (obs z ω i))
      (Fmax : ℕ) (z : Sec) (Q : ℝ) (hQ : ∀ ω, ∑ i, cl z ω i (ω i) ≤ Q),
      Pr (fun ω => (∀ i, phaseAcc H D x Pv hk z i ω (ω i) ∧
            PhaseRegistered D lay k reg sw Pw ρ x ops z ω i ∧ RecordComplete k Dj sw obs z ω i) ∧
          ¬PooledGood lay k reg Dj sw Pw ρ op Fmax (WregR H D x Pv ops z ω) (obs z ω)) ≤
        (1 - q) ^ Fmax + ∑ i, e i + Q := by
  intro Coin _ _ _ A H D hA hb hS Lk K S lay k reg Dj m kV Tp Sec _ _ sw Pw ρ op x Pv hk ops obs had hops q hq0 hq1
    hrate e he0 cl hcl hcl0 he hfit hcert Fmax z Q hQ
  exact (h H D hA hb hS lay k reg Dj sw Pw ρ op x Pv hk ops obs had hops q hq0 hq1 hrate e he0 cl hcl hcl0 he hfit
    hcert Fmax z).trans (add_le_add le_rfl (FlockSoundness.Audit.Periods.Ex_le_of_forall _ Q hQ))

end NetTiming
