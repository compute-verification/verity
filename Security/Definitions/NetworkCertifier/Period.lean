import Definitions.NetworkCertifier.SampledCharge
import Definitions.NetworkCertifier.TapRows

/-!
# A monitoring period of sampled audits: each sampling block's good event

Definitions only. A monitoring period is a list of sampling blocks, each audited by the sampled verifier at a statement
that its registration fixes (`SameRegistration`). A sampling block's good event (`WindowGood`) is the leak relation at
the network certifier's rows (`RLeakTap`) at the audit's extracted witness, with the registered inputs pinned (`AsReg`),
the trace holding the registration's values at the registered inputs, and the observations complete. Which observations
the relation counts is picked out by the network certifier's registration of its rows (`TapOpens`), not by the trace:
`RLeakTap` fails, on an accepted run, only where the extraction fails or the trace's openings collide with the network
certifier's (`Btap`). The registration's values at the prover's registered inputs are the one condition the event holds
that no statement here bounds.
-/

namespace NetTiming

open Claims Claims.Sampling

section Period

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **A statement registered as `s₀`**: the same program, design, layout, horizon, registered inputs and partition as
`s₀`; only the trace's commitments (`roots`) and the description may differ. -/
def SameRegistration {L S : ℕ} (D : Deploy Coin Col T) (lay : Stmt → Layout L S) (k : Stmt → ℕ)
    (reg : Stmt → Finset ℕ) (s₀ s : Stmt) : Prop :=
  s.prog = s₀.prog ∧ s.design = s₀.design ∧ lay s = lay s₀ ∧ k s = k s₀ ∧ reg s = reg s₀ ∧ D.partition s = D.partition s₀

/-- **A window's good event**: at the window's statement `x` and the audit's extracted witness `w`, the leak relation at
the network certifier's rows holds (the network certifier's openings `tops s`), `x` is in the audit's scope and
registered as `s₀`, the trace holds `ρ` at the registered inputs, and the run's observations `o` have the horizon's
length, jitter below `Dj` and every status complete. -/
def WindowGood (H : Msg → Msg) (D : Deploy Coin Col T) (L K S : ℕ) (lay : Stmt → Layout L S) (k : Stmt → ℕ)
    (δ : ℝ) (reg : Stmt → Finset ℕ) (Dj : ℕ) (s₀ : Stmt) (ρ : ℕ → Bool) (tops : Stmt → List (Opening Bool))
    (x : Msg) (w : Wit) (o : List (Obs L K S)) : Prop :=
  RLeakTap H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj tops x w ∧
    ScopeAcc D lay k x ∧
    ∃ s, D.parse x = some s ∧ SameRegistration D lay k reg s₀ s ∧
      (∀ g ∈ reg s, traceVal s.prog (w.val H D.commit s) g = ρ g) ∧ Covered (k s) o ∧ o.length = k s ∧
      ∀ y ∈ o, y.jitter < Dj

end Period

end NetTiming
