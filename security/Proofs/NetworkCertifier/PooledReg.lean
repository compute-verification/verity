module

public import Definitions.NetworkCertifier.Guarantees.PooledReg
public import Proofs.Flock.SamplingGeneric.Session
public import Proofs.NetworkCertifier.PooledLeak
public import Proofs.NetworkCertifier.PooledRegEvent
public import Proofs.NetworkCertifier.PooledRegLeak
public import Proofs.NetworkCertifier.PooledRegSample

/-!
# The pooled bound at the trace with the registered openings

`NetTiming.pr_pooled_le` bounds a monitoring period's good event from two inputs at each execution phase. At the
incorrect sets `WregR` both are proved: the bound on each phase's verdict from the sample and the event at `FregR`
(`phasesMiss_of_event`), and each accepted phase's leak relation with no error (`phaseLeaks_regR`).

* `pooled_of_event` is the bound from any bound `e i` on each phase's event, the probability that its run accepts with
  a sampled proof block of its `FregR`. It takes no knowledge soundness of the proof system and no binding: a caller
  meets the event with its own bound, the square-root form (`Compiled.ksRegR_of_sessionKS`) or one from costs
  (`Compiled.ksRegR_of_costs`). Its event holds, beside each phase's verdict, whatever else `R` says of the phase.
* `pooledLeakRegistered` is `Guarantees.PooledLeakRegistered`: `pooled_of_event` at the square-root form, the sessions'
  bound from the proof system's knowledge soundness (`Claims.SamplingGeneric.sessionKSOf`), with `R` the phase
  registered as the period's and its record complete.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling Finset
open Claims.SamplingGeneric (Arg Compiled)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} {m kV : ℕ} {Tp Sec : Type}

/-- **The pooled bound from any bound on each execution phase's event**, the sample's term as `(1 - q)^Fmax`: where
`e i` bounds the probability that phase `i`'s run accepts with a sampled proof block of its `FregR`, every phase is
accepted with `R` outside the period's good event with probability at most `(1 - q)^Fmax + Σ e_i`. `R` is what else
the event says of a phase, under which `hyp` gives the phase's leak relation its inputs. It takes no knowledge
soundness of the proof system and no binding: a caller meets `hev` with its own bound on the event. -/
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
    (hev : ∀ z ∈ Zs, ∀ ω i, (D (x z ω i)).Scope (x z ω i) →
      Pr (fun p : Tp × (Fin ((D (x z ω i)).V.k (x z ω i)) → Coin) =>
        (run H (Pv z ω i) (D (x z ω i)).V (x z ω i) p.1 p.2).out = .acc ∧
          ∃ u ∈ (D (x z ω i)).checked (x z ω i) p.2,
            u ∈ FregR H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i)) ≤ e i)
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
      (1 - q) ^ Fmax + ∑ i, e i := by
  intro z hz
  have h := NetTiming.pr_pooled_le (phaseAcc H D x Pv hk z) (phaseAcc_prefix H D x Pv hk had.1 had.2.1 z)
    (WregR H D x Pv ops z) (fun i ω ω' hω => wreg_prefix H D had z i ω ω' hω) e (fun _ => 0) he0 q hq0 hq1
    (phasesMiss_of_event hS Zs x Pv hk ops hq1 hrate he0 hev z hz)
    (fun i ω => R z ω i → LeakAt lay k reg Dj (sw i) (Pw i) (ρ i) (op i (runsBefore (obs z ω) i))
      (WregR H D x Pv ops z ω i) (obs z ω i))
    (phaseLeaks_regR hS lay k reg Dj sw Pw ρ op Zs x Pv hk ops obs R hyp z hz) Fmax
  simp only [Finset.sum_const_zero, add_zero] at h
  refine le_trans (Claims.Pr_mono fun ω hω => ?_) h
  exact ⟨fun i => (hω.1 i).1, fun hg => hω.2 (And.intro hg.1 fun i => hg.2 i (hω.1 i).2)⟩

/-- **The leak through the network certifier over a monitoring period, at the trace with the registered openings**:
`Guarantees.PooledLeakRegistered`. Each phase's event is bounded by the square-root form
(`Compiled.ksRegR_of_sessionKS`), the sessions' bound from the proof system's knowledge soundness; the registered
openings give that they pass under each statement's roots and that every registered input has one (`RegOpens`); and a
phase's layout fits, and its record is complete over its statement's steps, because its statement is registered as the
period's. Each `e i` is not negative, since it bounds an error that is not. -/
theorem pooledLeakRegistered : Guarantees.PooledLeakRegistered := by
  intro Coin _ _ _ A H D hA hb hS Lk K S lay k reg Dj m kV Tp Sec _ _ sw Pw ρ op x Pv hk ops obs had hops q hq0 hq1
    hrate e he hfit hcert Fmax z
  have he0 : ∀ i, 0 ≤ e i := fun i =>
    le_trans' (he z (fun _ => (Classical.arbitrary Tp, fun _ => default)) i)
      (add_nonneg (Compiled.εregX_nonneg H _ _ _) (Claims.Pr_nonneg _))
  refine pooled_of_event hS lay k reg Dj sw Pw ρ op Set.univ x Pv hk ops obs
    (fun z ω i => PhaseRegistered D lay k reg sw Pw ρ x ops z ω i ∧ RecordComplete k Dj sw obs z ω i) had hq0 hq1
    (fun z _ => hrate z) he0
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

end NetTiming
