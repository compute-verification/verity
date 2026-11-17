module

public import Definitions.Certifier.Run

/-! Ingress: the interaction read inward, the secret the outside's message, and acceptance. -/

@[expose] public section

namespace NetTiming

/-- The real frames a row carries, in slot order. -/
def rowFrames {S : ℕ} {F : Type} (r : Row S F) : List F := (List.ofFn r).filterMap id

/-- The frames link `ℓ` carried over an observation sequence, step by step and slot by slot: its content, with the
timing erased. Under W4 the frames leave in queue order, so this is the order they were queued in. -/
def linkFrames {L K S : ℕ} (o : List (Obs L K S)) (ℓ : Fin L) : List (List Bool) :=
  o.flatMap fun x => rowFrames (x.rows ℓ)

/-- **Acceptance, as it bears on ingress:** at every step before `k`, every boundary link's status is complete. The
ingress rows are the Program's input, which it reads rather than predicts, so acceptance compares no ingress row. The
verifier's other check, the egress rows against the Program (`Accepted`, read outward), only removes runs, so a bound
over these runs covers every accepted one. -/
def StatusComplete {L K S : ℕ} (k : ℕ) (o : List (Obs L K S)) : Prop :=
  ∀ t (ht : t < k), ∃ h : t < o.length, ∀ κ, o[t].status κ = Status.complete

end NetTiming
