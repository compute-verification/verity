import Definitions.NetworkCertifier.Tied
import Definitions.NetworkCertifier.Period

/-!
# A rows-form phase's leak relation at the network certifier's rows, and its good event

Definitions only. A phase is counted as its joint Program (`Phase.prog`, `Definitions.NetworkCertifier.Tied`), so its
leak relation (`RLeakTied`) is `RLeakTap` for the joint Program: every observation sequence the network certifier
registers at the scan's transfer gates is one the joint Program accepts at the phase's layout (`AccSeqsTied`), its
inputs the traces' committed values (`Phase.val`) and values held only at the committed gates of one possible incorrect
set of each statement. Each statement's incorrect set is its own, at its own wires and its own `δ`, so the charge is
theirs together (`Phase.charge`).

A phase's good event (`PhaseGood`) is `WindowGood`'s, with the joint Program's registered inputs, the values the
registration fixes before the run, in place of each statement's. A tied gate is no registered input: it is wired, and
its value is its writer's.
-/

namespace NetTiming

open Claims Claims.Sampling

/-- **The phase's committed values**: the joint Program's gate `g`, gate `h` of the phase's `j`th statement, holds what
that statement's trace `ws[j]` opens there (`Wit.val`). -/
def Phase.val (H : Msg → Msg) (Cm : CommitmentScheme Bool) (ph : Phase) (ws : List Wit) (g : ℕ) : Option Bool :=
  (ph.locate g).bind fun (j, h) => (ph.stmts[j]?).bind fun s => (ws[j]?).bind fun w => w.val H Cm s h

/-- **The phase's possible incorrect sets** at `δ`: one set `Fs[j]` for each statement, a possible incorrect set of
statement `j` at its partition `Ps[j]` and at `δ j` (`faultySets`). -/
def Phase.IncorrectSets (ph : Phase) (Ps : List Partition) (δ : ℕ → ℝ) (Fs : List (Finset ℕ)) : Prop :=
  Fs.length = ph.stmts.length ∧
    ∀ j s P F, ph.stmts[j]? = some s → Ps[j]? = some P → Fs[j]? = some F → F ∈ faultySets s.design P (δ j)

/-- The joint Program's gate `g` is a committed gate of an incorrect proof block: gate `h` of the phase's `j`th
statement, on the boundary of a proof unit at a slot of `Fs[j]` of that statement's partition `Ps[j]` (`FaultyGate`). -/
def Phase.IncorrectGate (ph : Phase) (Ps : List Partition) (Fs : List (Finset ℕ)) (g : ℕ) : Prop :=
  ∃ j h s P F, ph.locate g = some (j, h) ∧ ph.stmts[j]? = some s ∧ Ps[j]? = some P ∧ Fs[j]? = some F ∧
    FaultyGate s.prog P F h

/-- **The phase's charge for its incorrect sets** at `δ`, at its statements' partitions `Ps` and committed-bit bounds
`β`: over its statements, each one's number of possible incorrect sets times `2^β` (`faultySets`, `faultyBits`). -/
noncomputable def Phase.charge (ph : Phase) (Ps : List Partition) (δ : ℕ → ℝ) (β : ℕ → ℕ) : ℕ :=
  ((ph.stmts.zip Ps).mapIdx fun j sP => (faultySets sP.1.design sP.2 (δ j)).card * 2 ^ β j).prod

/-- **The sequences a phase's audits accept** at `δ`: the observation sequences of length `k`, with jitter below `Dj`,
that the joint Program accepts at the phase's layout (`Phase.layout`), with its inputs other than the ingress and the
wired gates among `As`, and values held only at the committed gates of one possible incorrect set of each statement
(`Phase.IncorrectSets`, `Phase.IncorrectGate`). It is `AccSeqsSampled` with the joint Program, its layout and its
statements' incorrect sets. -/
def AccSeqsTied {L K S : ℕ} (ph : Phase) (Ps : List Partition) (lay : AuditStatement → BoundaryGates L S) (t : Ties)
    (k : ℕ) (As : Set (ℕ → Bool)) (δ : ℕ → ℝ) (op : ℕ → List (Obs L K S) → List Bool) (Dj : ℕ) :
    Set (List (Obs L K S)) :=
  {o | o.length = k ∧ (∀ x ∈ o, x.jitter < Dj) ∧ ∃ Fs, ph.IncorrectSets Ps δ Fs ∧ ∃ a ∈ As,
    ∃ free : ℕ → Option Bool, (∀ g b, free g = some b → ph.IncorrectGate Ps Fs g) ∧
      Accepted (circuitProgram ph.prog (ph.layout lay t) k) op k o (a, free)}

/-- **The phase's leak relation at the network certifier's rows** at `δ`, on its statements' bytes `xs`, the scan's
first, and their traces `ws`: the statements parse to the phase's, each with its partition `Ps[j]`, and every
observation sequence of length `k` with jitter below `Dj` that the network certifier registers at the scan's statement
(`TapOpens`, its openings `tops`), with every status complete, is among the sequences the phase's audits accept
(`AccSeqsTied`) at the inputs `As ρ`, `ρ` the traces' committed values as the joint Program reads them. -/
def RLeakTied (H : Msg → Msg) (Cm : CommitmentScheme Bool) (parse : Msg → Option AuditStatement)
    (partition : AuditStatement → Option Partition) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S) (k : ℕ)
    (ph : Phase) (t : Ties) (δ : ℕ → ℝ) (As : (ℕ → Bool) → Set (ℕ → Bool)) (Dj : ℕ) (tops : List (Opening Bool))
    (xs : List Msg) (ws : List Wit) : Prop :=
  List.Forall₂ (fun x s => parse x = some s) xs ph.stmts ∧
    ∃ Ps, List.Forall₂ (fun s P => partition s = some P) ph.stmts Ps ∧
      ∀ (op : ℕ → List (Obs L K S) → List Bool) (o : List (Obs L K S)),
        TapOpens H Cm (lay ph.scan) k op ph.scan tops o → Covered k o → o.length = k → (∀ y ∈ o, y.jitter < Dj) →
          o ∈ AccSeqsTied ph Ps lay t k (As (traceVal ph.prog (ph.val H Cm ws))) δ op Dj

/-- The `H` calls that check the phase's traces `ws`: each statement's openings under its roots. -/
def phaseLog (H : Msg → Msg) (Cm : CommitmentScheme Bool) (ph : Phase) (ws : List Wit) : List (Msg × Msg) :=
  (ph.stmts.zip ws).flatMap fun sw => tapLog H Cm sw.1.roots sw.2.vals

section Good

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **A phase registered as `ph₀`**: statement by statement, each registered as `ph₀`'s (`SameRegistration`). The
phase's registered inputs are the joint Program's (`PhaseGood`'s `reg`), so no statement's own are compared. -/
def SamePhase {L S : ℕ} (D : Deploy Coin Col T) (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ)
    (ph₀ ph : Phase) : Prop :=
  List.Forall₂ (SameRegistration D lay k fun _ => ∅) ph₀.stmts ph.stmts

/-- **A phase's good event**: at the phase's statements `xs`, the scan's first, and the audits' extracted witnesses
`ws`, each statement is in the audit's scope, and they parse to a phase registered as `ph₀` at which the phase's leak
relation at the network certifier's rows holds (`RLeakTied`, the joint Program's registered inputs `reg` pinned:
`AsReg`), the traces hold `ρ` at `reg`, and the run's observations `o` have the horizon's length, jitter below `Dj` and
every status complete. -/
def PhaseGood (H : Msg → Msg) (D : Deploy Coin Col T) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (δ : ℕ → ℝ) (Dj : ℕ) (ph₀ : Phase) (t : Ties) (reg : Finset ℕ) (ρ : ℕ → Bool)
    (tops : AuditStatement → List (Opening Bool)) (xs : List Msg) (ws : List Wit) (o : List (Obs L K S)) : Prop :=
  (∀ x ∈ xs, ScopeAcc D lay k x) ∧
    ∃ ph, SamePhase D lay k ph₀ ph ∧
      RLeakTied H D.commit D.parse D.partition L K S lay (k ph.scan) ph t δ
        (AsReg ph.prog (ph.layout lay t) (k ph.scan) reg) Dj (tops ph.scan) xs ws ∧
      (∀ g ∈ reg, traceVal ph.prog (ph.val H D.commit ws) g = ρ g) ∧ Covered (k ph.scan) o ∧ o.length = k ph.scan ∧
      ∀ y ∈ o, y.jitter < Dj

end Good

end NetTiming
