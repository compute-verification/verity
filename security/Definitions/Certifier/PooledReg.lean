module

public import Definitions.Claims.SamplingGenericX
public import Definitions.Certifier.MonitoringPeriod
public import Definitions.Certifier.PooledLeak
public import Definitions.Certifier.SampledReg
public import Definitions.Certifier.TapRows

/-! The pooled bound at the trace with the registered openings: its vocabulary and its named premises. -/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling Finset
open Claims.SamplingGeneric (Arg Compiled)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The design catches each proof block at rate `q`**: its sample misses every set `B` of the partition's proof
blocks with probability at most `(1 - q)^{|B|}`. A uniform sample of `kd` of `n` proof blocks meets it at every `q`
with `q · n ≤ kd`, and a sample of every proof block at `q = 1`. -/
def RateAt (s : AuditStatement) (Pt : Partition) (q : ℝ) : Prop :=
  ∀ B ⊆ range Pt.blocks.length, s.design.Pa Pt B ≤ (1 - q) ^ B.card

/-- **The trace read at the statement, with the registered openings** `ops`: the registered value at every gate a
registered opening names (the network attester's at the transfer gates, the registration's at the registered inputs),
the trace read at the statement (`Compiled.regValX`) at every other gate. It reads the statement, the registered
openings and the prover's strategy, and no tape. -/
noncomputable def regValR (H : Msg → Msg) (D : Compiled Coin A) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg)
    (ops : List (Opening Bool)) (g : ℕ) : Option Bool :=
  if ops.any (fun o => o.pos == g) then some (certifierVal ops g) else D.regValX H P x g

open Classical in
/-- **Its incorrect set**: the proof blocks that no values agreeing with that trace on the program's gates, and no
interior values, make correct (`Compiled.FregX`'s form at `regValR`); empty on a statement that doesn't parse. -/
noncomputable def FregR (H : Msg → Msg) (D : Compiled Coin A) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg)
    (ops : AuditStatement → List (Opening Bool)) : Finset ℕ :=
  match D.parse x with
  | none => ∅
  | some s => (Finset.range (D.part s).blocks.length).filter fun v =>
      ¬ ∃ b, (D.part s).blocks[v]? = some b ∧ ∃ (val : ℕ → Option Bool) (ι : ℕ → Bool),
        (∀ g < s.prog.gates.length, val g = regValR H D P x (ops s) g) ∧ s.prog.Correct (s.prog.refIn b val ι) b

/-- **`B_ops,X`**, the collision finder against the registered openings at the sessions' extractions: the judged run at
its tape and coins with the extractor's coins there, and the first collision among the calls its extractions make
themselves and the calls that check the registered openings under the statement's roots. Its cost is the run's SHA-512
calls, both parties', its extractions', their reruns' included, and the checks'. -/
noncomputable def BopsX (H : Msg → Msg) (D : Compiled Coin A) (ops : AuditStatement → List (Opening Bool)) :
    FinderFamily := fun Ω _ _ P x =>
  match D.parse x with
  | none => { S := Unit, out := fun _ => none, cost := fun _ => 0 }
  | some s =>
    { S := Ω × ((Fin (D.V.k x) → Coin) × D.SlotEC x)
      ne := ⟨(Classical.arbitrary _, (fun _ => default, Classical.arbitrary _))⟩
      out := fun q =>
        findColl (ownLog H (run H P D.V x q.1) (D.runExt x s q.2.1 (run H P D.V x q.1 q.2.1).seen q.2.2) ++
          certifierLog H D.commit s.roots (ops s))
      cost := fun q =>
        (run H P D.V x q.1 q.2.1).pLog.length + (run H P D.V x q.1 q.2.1).vLog.length +
          ((D.runExt x s q.2.1 (run H P D.V x q.1 q.2.1).seen q.2.2).eval H (run H P D.V x q.1)).2.1 +
          (certifierLog H D.commit s.roots (ops s)).length }

section Phases

variable {m kV : ℕ} {Tp Sec : Type}

/-- **Every execution phase's verifier has at most `kV` rounds at its statement**, at every secret and every history
of the period's coins, so the period's coins are one type: a phase's verifier reads the first of its `kV` coins and
ignores the rest. The verifier is a function of the statement, `D y` the compiled verifier that judges `y`. -/
def PhaseRounds (D : Msg → Compiled Coin A) (kV : ℕ) (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg) :
    Prop :=
  ∀ z ω i, (D (x z ω i)).V.k (x z ω i) ≤ kV

/-- **Execution phase `i`'s verdict at its own coins**, the prover's tape and `kV` coins, at the secret `z` and the
period's coins `ω`: the run of phase `i`'s prover on phase `i`'s statement, against the statement's verifier at the
first of those coins, accepts. -/
def phaseAcc (H : Msg → Msg) (D : Msg → Compiled Coin A) (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp) (hk : PhaseRounds D kV x) :
    Sec → Fin m → (Fin m → Tp × (Fin kV → Coin)) → Tp × (Fin kV → Coin) → Prop :=
  fun z i ω c =>
    (run H (Pv z ω i) (D (x z ω i)).V (x z ω i) c.1 (fun j => c.2 (Fin.castLE (hk z ω i) j))).out = .acc

/-- **Execution phase `i`'s incorrect set at the trace with the registered openings**, the openings those registered
at phase `i` (`ops z ω i`), which may read the secret and the earlier phases' coins. -/
noncomputable def WregR [Fintype Tp] (H : Msg → Msg) (D : Msg → Compiled Coin A)
    (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp)
    (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool)) :
    Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Finset ℕ :=
  fun z ω i => FregR H (D (x z ω i)) (Pv z ω i) (x z ω i) (ops z ω i)

/-- **Each execution phase reads only the earlier phases' coins**: phase `i`'s statement, its prover's strategy and
its registered openings are one at any two histories of the period's coins that agree before `i`. -/
def PhasesAdaptive (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp)
    (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool)) : Prop :=
  (∀ z i ω ω', (∀ j, j < i → ω j = ω' j) → x z ω i = x z ω' i) ∧
    (∀ z i ω ω', (∀ j, j < i → ω j = ω' j) → Pv z ω i = Pv z ω' i) ∧
    ∀ z i ω ω', (∀ j, j < i → ω j = ω' j) → ops z ω i = ops z ω' i

/-- **`e i` bounds execution phase `i`'s error**: at every secret and every history, the committed form's
error at the trace read at the statement (`Compiled.εregX`) plus the registered openings' binding term (`BopsX`'s
probability), both at the statement's verifier, is at most `e i`. -/
def PhaseErrors [Fintype Tp] [Nonempty Tp] (H : Msg → Msg) (D : Msg → Compiled Coin A)
    (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp)
    (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool)) (e : Fin m → ℝ) :
    Prop :=
  ∀ z ω i, (D (x z ω i)).εregX H (x z ω i) (Pv z ω i) +
    (BopsX H (D (x z ω i)) (ops z ω i) Tp (Pv z ω i) (x z ω i)).Pr H ≤ e i

/-- **Execution phase `i` is registered as the period's**, at the secret `z` and the period's coins `ω`: its
statement parses to one registered as `sw i` (`SameRegistration`) with the partition `Pw i`, and the phase's registered
openings hold `ρ i` at every registered input. -/
def PhaseRegistered {Lk S : ℕ} (D : Msg → Compiled Coin A) (lay : AuditStatement → BoundaryGates Lk S)
    (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (sw : Fin m → AuditStatement)
    (Pw : Fin m → Partition) (ρ : Fin m → ℕ → Bool) (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
    (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool)) (z : Sec)
    (ω : Fin m → Tp × (Fin kV → Coin)) (i : Fin m) : Prop :=
  ∃ s, (D (x z ω i)).parse (x z ω i) = some s ∧ SameRegistration (D (x z ω i)) lay k reg (sw i) s ∧
    (D (x z ω i)).partition s = some (Pw i) ∧ ∀ g ∈ reg s, certifierVal (ops z ω i s) g = ρ i g

/-- **Execution phase `i`'s record is complete**, at the secret `z` and the period's coins `ω`: the phase's
observations in the network attester's record have every status complete over the registered statement's steps
(`Covered`), one observation a step, and jitter below `Dj`. -/
def RecordComplete {Lk K S : ℕ} (k : AuditStatement → ℕ) (Dj : ℕ) (sw : Fin m → AuditStatement)
    (obs : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → List (Obs Lk K S)) (z : Sec)
    (ω : Fin m → Tp × (Fin kV → Coin)) (i : Fin m) : Prop :=
  Covered (k (sw i)) (obs z ω i) ∧ (obs z ω i).length = k (sw i) ∧ ∀ y ∈ obs z ω i, y.jitter < Dj

end Phases

end NetTiming
