import Properties.Verity.Game

/-!
# Soundness at the verifier of record, composed from its subprotocols' targets

The verdicts are not free fields: in the causal game after the parameters, the committed files and the commitment
(`Vocab.game`), subprotocol `a` accepts when the files pass the commit phase's check, onsite's cases are an accepted
composition whose inference-only policy and sampled proofs commit to one record, and `a`'s compiled component accepts
its own case in what the prover sent (`Vocab.Verdict`). The event is the verifier of record's acceptance (`Accepts`)
while one claim fails (`Vocab.Holds`, sampled proofs' on the fixed extractor's transcript), with no break in it: each
subprotocol's error at a strategy is its published `δ` plus its collision term at the probability that its fixed finder
outputs a SHA-512 collision (`Vocab.error`). `Soundness` states that each subprotocol's target
(`Vocab.SampledProofsSound`, `PouwSound`, `PousSound`, `BandwidthAuditSound`, `NetworkCertifierSound`) gives
`Vocab.Soundness`, with the errors summed. It is proved by `Properties.Composition` (onsite's acceptance is its
composition's, with the policy's and sampled proofs' record one, and every component's) and the union bound over one
strategy in one game.

Each target is a hypothesis until its subprotocol's piece proves it at its component (the one-theorem plan's ONE-8,
ONE-10, ONE-11, ONE-12 and ONE-16, in `note:proofs/20261026T0332Z-report-onethm`), and `Vocab`'s fields are given
until each piece defines its own: v2's reading of the blob (ONE-3b), the positions (ONE-7), the rounds and the releases
each component read (ONE-4), the extractor and the finders (ONE-8 and each target's), the gates computed (ONE-16), the
world (ONE-10, ONE-11, ONE-12) and the published certificates, errors and collision terms (ONE-19). The wipe, onsite's
seventh component, has no target here.
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
