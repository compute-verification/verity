module

public import Definitions.NetworkCertifier.PooledReg

/-!
# The leak through the network certifier over a monitoring period, at the trace with the registered openings

`PooledLeakAccepted` bounds a monitoring period's good event from two inputs at each execution phase, which it takes
as hypotheses: the phase's audit accepts with probability at most `(1 - q)^{|W_i|} + e_i`, and it accepts outside its
leak relation at `W_i` with probability at most `e'_i`. `PooledLeakRegistered` is that bound with both inputs proved,
at the incorrect sets of one trace: at each execution phase, the trace read at the statement with the registered
openings (`regValR`), whose incorrect set (`WregR`) the phase's statement, its prover's strategy and its registered
openings fix before the phase's sample.

* The first input holds there with `e_i` at least the committed form's error at the trace read at the statement
  (`Compiled.εregX`) plus the registered openings' binding term (`BopsX`'s probability): an accepted run that samples
  a proof block of the set has a session's extraction fail, or its extractions disagree with themselves, with the
  trace read at the statement or with a registered opening.
* The second holds there with no error: where the network certifier registers an accepted phase's record among the
  registered openings (`CertifierRegisters`), the record meets the leak relation at that set, with no probability.

So every phase is accepted outside the period's good event (`PooledGood` at `WregR`) with probability at most
`(1 - q)^Fmax + Σ_i e_i`.

What it takes. The sampled proof system's premises (`Arg.KS`, `CommitmentScheme.Binds`, `Compiled.Sessions`). A rate
`q` that every execution phase's sample met (`RateAt`), a hypothesis here: nothing in this statement derives it from
what the verifier accepts. That each phase reads only the phases before it (`PhasesAdaptive`), a property of the
period the statement is read at, which it proves of no check. That the registered openings pass under each statement's
roots and open every registered input (`RegOpens`), and that at each accepted phase the network certifier registers
its record among them (`CertifierRegisters`), the record is complete (`RecordsComplete`) and the statement is
registered as the period's (`PhasesRegistered`).

What it does not say. It counts no secrets: `PooledLeakAccepted`'s second conclusion does, at any incorrect sets. It
fixes no sample's form and states no error's size.
-/

@[expose] public section

namespace NetTiming.Guarantees

open Claims Claims.Sampling Finset
open Claims.SamplingGeneric (Arg Compiled)

-- A guarantee keeps its theorem's hypothesis names for the reader, though no statement refers to them.
set_option linter.unusedVariables false

/-- **The leak through the network certifier over a monitoring period of accepted sampled audits, at the trace with
the registered openings.** Over `m` execution phases on independent live coins, phase `i`'s a tape of its prover and
the verifier's `kV` coins (`hk`), audited by the compiled verifier `D` over the proof system `A`: for each secret
`z ∈ Zs` and coins `ω`, phase `i`'s statement `x z ω i`, its prover's strategy `Pv z ω i` and its registered openings
`ops z ω i` read only the phases before `i` (`had`), and `obs z ω i` are the phase's observations in the network
certifier's record. Given

* `A`'s knowledge soundness, the commitment's binding and the sessions' contract (`hA`, `hb`, `hS`);
* registered openings that pass under each statement's roots and open every registered input (`hops`);
* `q`, the rate every execution phase's sample met (`hrate`), a hypothesis here, with `0 ≤ q ≤ 1`;
* `e i` at least phase `i`'s error at the trace read at the statement plus the registered openings' binding term
  (`he`);
* layouts that fit the registered statements (`hfit`); and at each accepted phase, the network certifier registering
  the phase's record among the registered openings (`hcert`), the statement registered as `sw i` with the partition
  `Pw i` and the openings holding `ρ i` at the registered inputs (`hreg`), and a complete record (`hrec`);

for every secret, every phase is accepted while the period's good event fails (`PooledGood` at the incorrect sets
`WregR`: they hold under `Fmax` proof blocks in total, and every phase's record meets its leak relation at its own)
with probability at most `(1 - q)^Fmax + Σ_i e i`. -/
def PooledLeakRegistered : Prop :=
  ∀ {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin} (H : Claims.Msg → Claims.Msg)
    (D : Compiled Coin A) (hA : A.KS H) (hb : D.commit.Binds H) (hS : D.Sessions H) {Lk K S : ℕ}
    (lay : AuditStatement → BoundaryGates Lk S) (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ)
    {m kV : ℕ} {Tp Sec : Type} [Fintype Tp] [Nonempty Tp] (sw : Fin m → AuditStatement) (Pw : Fin m → Partition)
    (ρ : Fin m → ℕ → Bool) (op : Fin m → List (List (Obs Lk K S)) → ℕ → List (Obs Lk K S) → List Bool)
    (Zs : Set Sec) (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Claims.Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Claims.Prover Tp) (hk : PhaseRounds D kV x)
    (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool))
    (obs : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → List (Obs Lk K S)) (had : PhasesAdaptive x Pv ops)
    (hops : ∀ z ∈ Zs, ∀ ω i s, D.parse (x z ω i) = some s → RegOpens H D.commit s (reg s) (ops z ω i s))
    (q : ℝ) (hq0 : 0 ≤ q) (hq1 : q ≤ 1)
    (hrate : ∀ z ∈ Zs, ∀ ω i s Pt, D.parse (x z ω i) = some s → D.partition s = some Pt → RateAt s Pt q)
    (e : Fin m → ℝ) (he0 : ∀ i, 0 ≤ e i) (he : PhaseErrors H D Zs x Pv ops e)
    (hfit : ∀ i, (lay (sw i)).Fits (sw i).prog (k (sw i)))
    (hcert : ∀ z ∈ Zs, ∀ ω i s, D.parse (x z ω i) = some s → phaseAcc H D x Pv hk z i ω (ω i) →
      CertifierRegisters H D.commit (lay s) (k s) (op i (runsBefore (obs z ω) i)) s (ops z ω i s) (obs z ω i))
    (hreg : PhasesRegistered H D lay k reg sw Pw ρ Zs x Pv hk ops) (hrec : RecordsComplete H D k Dj Zs x Pv hk obs)
    (Fmax : ℕ),
    ∀ z ∈ Zs, Pr (fun ω => (∀ i, phaseAcc H D x Pv hk z i ω (ω i)) ∧
        ¬PooledGood lay k reg Dj sw Pw ρ op Fmax (WregR H D x Pv ops z ω) (obs z ω)) ≤
      (1 - q) ^ Fmax + ∑ i, e i

end NetTiming.Guarantees
