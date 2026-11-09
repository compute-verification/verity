import Definitions.NetworkCertifier.SampledCharge
import Definitions.NetworkCertifier.SampledReg
import Definitions.NetworkCertifier.Tap
import Definitions.NetworkCertifier.TapRows
import Definitions.NetworkCertifier.PeriodConditions

/-!
# A monitoring period of sampled audits: each window's good event

Definitions only. A monitoring period is a list of windows, each audited by the sampled verifier at a statement that its
registration fixes (`SameRegistration`). A window's good event (`WindowGood`) is the leak relation at the network
certifier's rows (`RLeakTap`) at the audit's extracted witness, with the registered inputs pinned (`AsReg`), the trace
holding the registration's values at the registered inputs, and the observations complete. Which observations the
relation counts is picked out by the network certifier's registration of its rows (`TapOpens`), not by the trace:
`RLeakTap` fails, on an accepted run, only where the extraction fails or the trace's openings collide with the network
certifier's (`Btap`). The registration's values at the prover's registered inputs are the one condition the event holds
that no statement here bounds.

`WindowGoodReg` is the event with that condition moved off the trace: the registered inputs are pinned at the values
the registration opens (`regVal`, its openings `ops s`), which are the window's registered values `ρ`. It reads the
registration and never the extracted trace at the registered inputs, so a trace that disagrees with the registration
fails it only through the leak relation, which fails, on an accepted run, only where the extraction fails or the trace's
openings collide with the network certifier's (`Btap`) or the registration's (`Breg`).

The registration opens its registered inputs at a statement (`RegOpensAt`). The other conditions a monitoring period's
statement takes are named in `Definitions.NetworkCertifier.PeriodConditions`.
-/

namespace NetTiming

open Claims Claims.Sampling

section Period

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **A statement registered as `s₀`**: the same program, design, layout, horizon, registered inputs and partition as
`s₀`; only the trace's commitments (`roots`) and the description may differ. -/
def SameRegistration {L S : ℕ} (D : Deploy Coin Col T) (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (reg : AuditStatement → Finset ℕ) (s₀ s : AuditStatement) : Prop :=
  s.prog = s₀.prog ∧ s.design = s₀.design ∧ lay s = lay s₀ ∧ k s = k s₀ ∧ reg s = reg s₀ ∧ D.partition s = D.partition s₀

/-- **A window's good event**: at the window's statement `x` and the audit's extracted witness `w`, the leak relation at
the network certifier's rows holds (the network certifier's openings `tops s`), `x` is in the audit's scope and
registered as `s₀`, the trace holds `ρ` at the registered inputs, and the run's observations `o` have the horizon's
length, jitter below `Dj` and every status complete. -/
def WindowGood (H : Msg → Msg) (D : Deploy Coin Col T) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S) (k : AuditStatement → ℕ)
    (δ : ℝ) (reg : AuditStatement → Finset ℕ) (Dj : ℕ) (s₀ : AuditStatement) (ρ : ℕ → Bool) (tops : AuditStatement → List (Opening Bool))
    (x : Msg) (w : Wit) (o : List (Obs L K S)) : Prop :=
  RLeakTap H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj tops x w ∧
    ScopeAcc D lay k x ∧
    ∃ s, D.parse x = some s ∧ SameRegistration D lay k reg s₀ s ∧
      (∀ g ∈ reg s, traceVal s.prog (w.val H D.commit s) g = ρ g) ∧ Covered (k s) o ∧ o.length = k s ∧
      ∀ y ∈ o, y.jitter < Dj

/-- **A window's good event at the registration's values**: at the window's statement `x` and the audit's extracted
witness `w`, the leak relation at the network certifier's rows holds with the registered inputs pinned at the values
the registration opens (`AsReg` at `regVal (ops s)`), `x` is in the audit's scope and registered as `s₀`, the
registration opens `ρ` at the registered inputs, and the run's observations `o` have the horizon's length, jitter below
`Dj` and every status complete. -/
def WindowGoodReg (H : Msg → Msg) (D : Deploy Coin Col T) (L K S : ℕ) (lay : AuditStatement → BoundaryGates L S)
    (k : AuditStatement → ℕ) (δ : ℝ) (reg : AuditStatement → Finset ℕ) (ops : AuditStatement → List (Opening Bool))
    (Dj : ℕ) (s₀ : AuditStatement) (ρ : ℕ → Bool) (tops : AuditStatement → List (Opening Bool)) (x : Msg) (w : Wit)
    (o : List (Obs L K S)) : Prop :=
  RLeakTap H D.commit D.parse D.partition L K S lay k δ
      (fun s _ => AsReg s.prog (lay s) (k s) (reg s) (regVal (ops s))) Dj tops x w ∧
    ScopeAcc D lay k x ∧
    ∃ s, D.parse x = some s ∧ SameRegistration D lay k reg s₀ s ∧ (∀ g ∈ reg s, regVal (ops s) g = ρ g) ∧
      Covered (k s) o ∧ o.length = k s ∧ ∀ y ∈ o, y.jitter < Dj

/-- **The registration opens the registered inputs at the statement `x`**: at the statement `x` parses to, its
openings `ops s` open the registered inputs `reg s` under the statement's roots (`RegOpens`). -/
def RegOpensAt (H : Msg → Msg) (D : Deploy Coin Col T) (reg : AuditStatement → Finset ℕ)
    (ops : AuditStatement → List (Opening Bool)) (x : Msg) : Prop :=
  ∀ s, D.parse x = some s → RegOpens H D.commit s (reg s) (ops s)

end Period

end NetTiming
