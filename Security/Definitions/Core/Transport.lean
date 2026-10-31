import Definitions.Core.Game

/-!
# The transport's vocabulary: roles, channels, and what a coalition sees of them

The channels between an auditor party and the prover (`verity.ml.crypto.transport`), as the zero-knowledge statement
reads them. `Roles` says which nodes are the prover's own machines and which auditor parties are under `Confined p`, the
per-party premise beside `Faithful p` and `Secret p`; a `Channel` names its auditor end and who can read its bytes in
transit. `ViewIsRecord` is what zero knowledge needs of the transport, for one test, with its probabilities `Game.value`
over uniform draws (`Game.draw`). The premises on a channel are in `Assumptions/Transport`: `WitnessPathPrivate c` (the
per-channel premise), `ImprintPathPrivate c`, `ChannelHides` and `ChannelAuthentic`.
-/

namespace Verity.Transport

open scoped ENNReal

/-- Two probabilities are within `η` of each other. -/
def Close (a b η : ℝ≥0∞) : Prop :=
  a ≤ b + η ∧ b ≤ a + η

/-- Who is who among a deployment's nodes: the prover's own machines (`developer`), and the auditor parties under
`Confined p`, each of which sends its outputs only on its declared channels. -/
structure Roles (Node : Type) where
  developer : Node → Prop
  Confined : Node → Prop

/-- A channel between the auditor party `party` and the prover, with who can read its bytes in transit:
`readsPrivate o` when `o` can read the bytes of any of its private frames over its life, `readsImprint o` when `o` can
read the bytes of its imprint exchange. -/
structure Channel (Node : Type) where
  party : Node
  readsPrivate : Node → Prop
  readsImprint : Node → Prop

/-- Every reader is the auditor end `party`, one of the prover's own machines, or an auditor party under `Confined p`. -/
def PathPrivate {Node : Type} (s : Roles Node) (party : Node) (reads : Node → Prop) : Prop :=
  ∀ o, reads o → o = party ∨ s.developer o ∨ s.Confined o

/-- The nodes of a coalition `C` that read a channel with auditor end `party` as outsiders: none of `party`, the prover's
own machines and the `Confined` parties, so third parties on the path and the auditor's other components (its verifier,
which publishes). What the rest give `C` is `Confined p`'s to bound: declared outputs, already in the record. -/
def outside {Node : Type} (s : Roles Node) (party : Node) (C : Node → Prop) : Node → Prop :=
  fun o => C o ∧ o ≠ party ∧ ¬ s.developer o ∧ ¬ s.Confined o

open Classical in
/-- What the coalition `C` sees of bytes `w` that the nodes under `reads` can read: all of them when one of its nodes is
such a reader, and nothing otherwise. -/
noncomputable def seen {Node W : Type} (C reads : Node → Prop) (w : W) : Option W :=
  if ∃ o, C o ∧ reads o then some w else none

/-- **`view-is-record`**, for one test `E`: the coalition's view (`view`, of the run's randomness `ω`, drawn uniformly) is
within `η` of a simulator's output (`sim`, on coins `ω'` of its own) from the public record alone (`record`). -/
def ViewIsRecord {Ω Ω' Rec V : Type} [Fintype Ω] [Nonempty Ω] [Fintype Ω'] [Nonempty Ω'] (view : Ω → V)
    (record : Ω → Rec) (sim : Rec → Ω' → V) (E : V → Prop) (η : ℝ≥0∞) : Prop :=
  Close (Game.value E ((Game.draw Ω).map view))
    (Game.value E ((Game.draw Ω).bind fun ω => (Game.draw Ω').map (sim (record ω)))) η

end Verity.Transport
