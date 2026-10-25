import Definitions.NetworkCertifier.SampledCharge

/-!
# A period of sampled audits: each window's good event

Definitions only. A period is a list of windows, each audited by the sampled verifier at a statement that its
registration fixes (`SameRegistration`). A window's good event (`WindowGood`) is the leak relation `RLeak` at the audit's
extracted witness, with the registered inputs pinned (`AsReg`), the trace holding the registration's values at the
registered inputs, its transfer gates carrying the run's observations (`Carries`), and the observations complete.
`Carries` and `Covered` are the conditions inside `RAcc`'s conclusion that pick out which observations it is about, so
the event holds them as it holds the registration's values: what ties the transfer gates to the certifier's rows is the
audit's registered reads of those rows, opened against the certifier's signed root, and no statement here bounds the
event's failure.
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

/-- **A window's good event**: at the window's statement `x` and the audit's extracted witness `w`, the leak relation
holds, `x` is in the audit's scope and registered as `s₀`, the trace holds `ρ` at the registered inputs, its transfer
gates carry the run's observations `o` at the operator `op` (`Carries`), and `o` has the horizon's length, jitter below
`Dj` and every status complete. -/
def WindowGood (H : Msg → Msg) (D : Deploy Coin Col T) (L K S : ℕ) (lay : Stmt → Layout L S) (k : Stmt → ℕ)
    (δ : ℝ) (reg : Stmt → Finset ℕ) (Dj : ℕ) (s₀ : Stmt) (ρ : ℕ → Bool) (op : ℕ → List (Obs L K S) → List Bool)
    (x : Msg) (w : Wit) (o : List (Obs L K S)) : Prop :=
  RLeak H D.commit D.parse D.partition L K S lay k δ (fun s => AsReg s.prog (lay s) (k s) (reg s)) Dj x w ∧
    ScopeAcc D lay k x ∧
    ∃ s, D.parse x = some s ∧ SameRegistration D lay k reg s₀ s ∧
      (∀ g ∈ reg s, traceVal s.prog (w.val H D.commit s) g = ρ g) ∧
      Carries (lay s) s.prog (w.val H D.commit s) op (k s) o ∧ Covered (k s) o ∧ o.length = k s ∧
      ∀ y ∈ o, y.jitter < Dj

end Period

end NetTiming
