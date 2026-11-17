module

public import Definitions.Certifier.PooledReg

/-! The pooled leak at the trace with the registered openings: the rate `q` and `PhasesAdaptive` are hypotheses nothing
here derives, and it counts no secrets. -/

@[expose] public section

namespace NetTiming.Guarantees

open Claims Claims.Sampling Finset
open Claims.SamplingGeneric (Arg Compiled)

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak through the network attester over a monitoring period of accepted sampled audits, at the trace with
the registered openings.** Over `m` execution phases on independent live coins, phase `i`'s a tape of its prover and
`kV` coins, of which its verifier reads the first it needs (`hk`), each statement `y` judged by the compiled verifier
`D y` over the proof system `A`: for each secret `z : Sec` and coins `ω`, phase `i`'s statement `x z ω i`, its prover's
strategy `Pv z ω i` and its registered openings `ops z ω i` read only the phases before `i` (`had`), and `obs z ω i`
are the phase's observations in the network attester's record. Given

* `A`'s knowledge soundness, and the commitment's binding and the sessions' contract at every statement's verifier
  (`hA`, `hb`, `hS`);
* registered openings that pass under each statement's roots and open every registered input (`hops`);
* `q`, the rate every execution phase's sample met (`hrate`), a hypothesis here, with `0 ≤ q ≤ 1`;
* `e i` at least phase `i`'s error at the trace read at the statement plus the registered openings' binding term
  (`he`);
* layouts that fit the registered statements (`hfit`), and at each accepted phase, the network attester registering
  the phase's record among the registered openings (`hcert`);

for every secret, the probability that every phase is accepted, its statement registered as `sw i` with the partition
`Pw i` and the openings holding `ρ i` at the registered inputs (`PhaseRegistered`) and its record complete
(`RecordComplete`), while the period's good event fails (`PooledGood` at the incorrect sets `WregR`: they hold under
`Fmax` proof blocks in total, and every phase's record meets its leak relation at its own) is at most
`(1 - q)^Fmax + Σ_i e i`. A period in which an accepted phase is not registered as the period's, or has an incomplete
record, is outside the bound. -/
def PooledLeakRegistered : Prop :=
  ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} (H : Claims.Msg → Claims.Msg)
    (D : Claims.Msg → Compiled Coin A) (hA : A.KS H) (hb : ∀ y, (D y).commit.Binds H) (hS : ∀ y, (D y).Sessions H)
    {Lk K S : ℕ} (lay : AuditStatement → BoundaryGates Lk S) (k : AuditStatement → ℕ)
    (reg : AuditStatement → Finset ℕ) (Dj : ℕ) {m kV : ℕ} {Tp Sec : Type} [Fintype Tp] [Nonempty Tp]
    (sw : Fin m → AuditStatement) (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool)
    (op : Fin m → List (List (Obs Lk K S)) → ℕ → List (Obs Lk K S) → List Bool)
    (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Claims.Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Claims.Prover Tp) (hk : PhaseRounds D kV x)
    (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool))
    (obs : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → List (Obs Lk K S)) (had : PhasesAdaptive x Pv ops)
    (hops : ∀ z ω i s, (D (x z ω i)).parse (x z ω i) = some s →
      RegOpens H (D (x z ω i)).commit s (reg s) (ops z ω i s))
    (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hrate : ∀ z ω i s Pt, (D (x z ω i)).parse (x z ω i) = some s →
      (D (x z ω i)).partition s = some Pt → RateAt s Pt q)
    (e : Fin m → ℝ) (he : PhaseErrors H D x Pv ops e) (hfit : ∀ i, (lay (sw i)).Fits (sw i).prog (k (sw i)))
    (hcert : ∀ z ω i s, (D (x z ω i)).parse (x z ω i) = some s → phaseAcc H D x Pv hk z i ω (ω i) →
      CertifierRegisters H (D (x z ω i)).commit (lay s) (k s) (op i (runsBefore (obs z ω) i)) s (ops z ω i s)
        (obs z ω i))
    (Fmax : ℕ) (z : Sec),
    Pr (fun ω => (∀ i, phaseAcc H D x Pv hk z i ω (ω i) ∧
          PhaseRegistered D lay k reg sw Pw ρ x ops z ω i ∧ RecordComplete k Dj sw obs z ω i) ∧
        ¬PooledGood lay k reg Dj sw Pw ρ op Fmax (WregR H D x Pv ops z ω) (obs z ω)) ≤
      (1 - q) ^ Fmax + ∑ i, e i

end NetTiming.Guarantees
