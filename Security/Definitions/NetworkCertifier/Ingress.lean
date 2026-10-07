import Definitions.NetworkCertifier.Run

/-!
# Ingress: the interaction read inward, content, and acceptance

Definitions only. `docs/network-transparency/timing-channel.md` §1.1, §1.3, §2.5 ("Ingress content") and §3.2.

**The mirror image.** The egress statements put the secret inside X and count what the observations carry out of it.
Ingress is the same interaction with the boundary read inward, so its statement reuses `Interaction`, `Certifier` and
`Obs` with the two parties exchanged:
* the secret is the message `m` of the outside (the operator and every colluding endpoint), which it wants to deliver
  into X. Its party is `snd m`: it chooses when ingress frames reach the certifiers, and their content, order and faults;
* `op` is X's devices, which don't know `m`;
* the certifiers' record `obs` is what the ingress links release into X: each of the `L` ingress links' row (`rows`), the
  step's jitter class, and every boundary link's status;
* `wire` is everything that enters X: what the certifiers emit inward, which W2 (`Regenerates`, read inward) makes a
  function of the record.

Both parties see every record. X's moves are a fixed function of the records, so the outside sender can recompute them,
and X's egress with them: it reacts to feedback, as the operator does in the egress statements.

**Content and timing.** The ingress frames' content and order are the Program's input, which the Program reads and the
inference-only policy charges as reads. What is left is when they arrive: the occupancy of the ingress grid.
`linkFrames` erases the timing from a record and keeps each link's frames in the order they were released.
-/

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
