module

public import Definitions.Certifier.Checked
public import Definitions.Certifier.PooledReg

/-!
# The checked pooled bound at the trace with the registered openings: its vocabulary and its premise

Definitions only. `PooledReg.lean`'s period, with each phase's check judging the run its leak relation at `WregR` is
accepted at.

* `regRunR` is that run: the statement's program over the horizon on the trace with the registered openings, its
  ingress the values those openings hold at each step's ingress gates. Where the network attester registers a record
  among the registered openings (`CertifierRegisters`), the operator's moves are those values, so the run is fixed
  before the phase's coins.
* `PrCheckMissR` is the probability that the phase accepts at a statement and partition whose `FregR` holds no proof
  block of a class sampled `all` while `regRunR` is not valid.
* `PhaseChecks` is the premise a guarantee over these takes by name: each phase's `PrCheckMissR` is at most `εchk i`.
-/

@[expose] public section

namespace NetTiming

open Claims Claims.Sampling Finset
open Claims.SamplingGeneric (Arg Compiled)

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The run at the trace with the registered openings**: `C` over the horizon on the trace read at the statement with
the registered openings, filled at the network attester's values (`fillTrace` of `regValR`), its ingress the values the
registered openings hold at each step's ingress gates, and the incorrect proof blocks of `FregR` held at that trace
(`heldAdv`). Where the network attester registers a record among the registered openings (`CertifierRegisters`), it is
the run the leak relation at `FregR` is accepted at: the operator's moves there are those ingress values. It reads the
statement, the registered openings and the prover's strategy, and no tape. -/
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
fixes the phase's program, layout and steps but not its check.

It is a hypothesis about the check, proved for no check today: that a phase which accepts rarely does so while the run
at its registered trace fails the check. `PrCheckMissR` counts only the runs whose `FregR` holds no proof block of a
class sampled `all`; a run whose `FregR` meets such a class is charged to the phase's error `e i` instead. -/
def PhaseChecks [Fintype Tp] (H : Msg → Msg) (D : Msg → Compiled Coin A) {L S : ℕ}
    (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ) (valid : AuditStatement → (ℕ → Bool) → Prop)
    (sw : Fin m → AuditStatement) (x : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Msg)
    (Pv : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → Prover Tp)
    (ops : Sec → (Fin m → Tp × (Fin kV → Coin)) → Fin m → AuditStatement → List (Opening Bool))
    (εchk : Fin m → ℝ) : Prop :=
  ∀ z ω i, PrCheckMissR H (D (x z ω i)) lay k (fun _ => valid (sw i)) (x z ω i) (Pv z ω i) (ops z ω i) ≤ εchk i

end Phases

end NetTiming
