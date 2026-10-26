import Properties.Verity.Game

/-!
# Soundness at the verifier of record, composed from its subprotocols' targets

The verdicts are not free fields: in the causal game after the parameters, the circuit file and the commitment
(`Vocab.game`), subprotocol
`a` accepts when its compiled component accepts its own case in what the prover sent and every release its check read
is in the device's log (`Vocab.Verdict`), and the event is the verifier of record's acceptance (`Accepts`) while one
claim fails (`Vocab.Holds`, sampled proofs' read off the commitment), outside the break. `Soundness` states that each
subprotocol's target (`Vocab.SampledProofsSound`, `PouwSound`, `PousSound`, `BandwidthAuditSound`,
`NetworkCertifierSound`) gives `Vocab.Soundness`, with the published errors summed. It is proved by
`Properties.Composition` (onsite's acceptance is every component's) and the union bound over one strategy in one game.

Each target is a hypothesis until its subprotocol's piece proves it at its component (the one-theorem plan's ONE-8,
ONE-10, ONE-11, ONE-12 and ONE-16, in `note:proofs/20261026T0332Z-report-onethm`), and `Vocab`'s fields are given
until each piece defines its own: the circuit, partition and positions (ONE-7), the rounds and the releases each
component read (ONE-4), the world (ONE-10, ONE-11, ONE-12, ONE-16) and the published certificates and errors (ONE-19).
The wipe, the sixth subprotocol, joins in `Properties.Verity.Wipe`.
-/

namespace Verity

open Accounting

/-- **Soundness at the verifier of record, from each subprotocol's target.** On any device, for any names `V` the
subprotocols define, if sampled proofs, PoUW, PoUS, the bandwidth audit and the network certifier each meet their target
in the causal game, then at all parameters, circuit files and commitments and for every strategy, the probability that
onsite's compiled verifier accepts a file that passed the commit phase's check while one of the period's claims fails
(more than an `ε` share of the file's units unbound in some transcript the commitment opens to, less than `W` of work, fewer than `S` bits held on average, fewer than `Q`
bits read, or more than `K` bits through the network certifiers), with no break, is at most the published errors
summed. -/
def Soundness : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (V : Vocab d α),
    V.SampledProofsSound → V.PouwSound → V.PousSound → V.BandwidthAuditSound → V.NetworkCertifierSound →
      V.Soundness

end Verity
