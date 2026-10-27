import Properties.Verity.Game

/-!
# Soundness at the verifier of record, composed from its subprotocols' targets

The verdicts are not free fields: in the causal game after the parameters, the committed files and the commitment
(`Vocab.game`), subprotocol `a` accepts when the files pass the commit phase's check, NCI accepted the registration
sampled proofs committed to, and `a`'s compiled component accepts its own case in what the prover sent
(`Vocab.Verdict`). The event is the verifier of record's acceptance (`Accepts`) while one claim fails (`Vocab.Holds`,
sampled proofs' on the fixed extractor's transcript), with no break in it: each subprotocol's error at a strategy is its
published `δ` plus its collision term at the probability that its fixed finder outputs a SHA-512 collision
(`Vocab.error`). `Soundness` states that each subprotocol's target (`Vocab.SampledProofsSound`, `PouwSound`,
`PousSound`, `BandwidthAuditSound`, `NetworkCertifierSound`) gives `Vocab.Soundness`, with the errors summed. It is
proved by `Properties.Composition` (onsite's acceptance is NCI's on the registration and every component's) and the union
bound over one strategy in one game.

Each target is a hypothesis until its subprotocol's piece proves it at its component, and `Vocab`'s fields are given
until each piece defines its own: v2's reading of the blob, the positions, the rounds and the releases each component
read, the extractor and the finders, the gates computed, the world and the published certificates, errors and collision
terms. The wipe, the sixth subprotocol, joins in `Properties.Verity.Wipe`.
-/

namespace Verity

open Accounting

/-- **Soundness at the verifier of record, from each subprotocol's target.** On any device, for any names `V` the
subprotocols define, if sampled proofs, PoUW, PoUS, the bandwidth audit and the network certifier each meet their target
in the causal game, then at all parameters, committed files and commitments and for every strategy, the probability that
onsite's compiled verifier accepts files that passed the commit phase's check while one of the period's claims fails
(more than an `ε` share of the units unbound in the extractor's transcript, fewer than `W` gates computed after their
coin, fewer than `S` bits held on average, fewer than `Q` bits read, or more than `K` bits through the network
certifiers) is at most the subprotocols' errors at that strategy summed, each its published `δ` plus its collision term
at its finder's probability of a SHA-512 collision. -/
def Soundness : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (V : Vocab d α),
    V.SampledProofsSound → V.PouwSound → V.PousSound → V.BandwidthAuditSound → V.NetworkCertifierSound →
      V.Soundness

end Verity
