import Properties.Verity.Game

/-!
# Soundness at the verifier of record, composed from its subprotocols' targets

The verdicts are not free fields: in the causal game after the parameters, the committed files and the commitment
(`Vocab.game`), subprotocol `a` accepts at composite `k` when the files pass the commit phase's check, `k` accepted what
the prover sent as a `same` composition, and `a`'s compiled component accepts its own case in it (`Vocab.Verdict`). The
event is the verifier of record's acceptance (`Accepts k`) while the claim of one of `k`'s subprotocols fails
(`Vocab.Holds`, sampled proofs' on the fixed extractor's transcript), with no break in it: each subprotocol's error at a
strategy is its published `δ` plus its collision term at the probability that its fixed finder outputs a SHA-512
collision (`Vocab.error`).

`SoundnessOf` is the one statement, at every composite: each of its subprotocols' targets gives `Vocab.SoundnessAt`,
with the errors summed over its subprotocols. `Soundness` is it at onsite, from its seven targets
(`Vocab.SampledProofsSound`, `PouwSound`, `InferenceOnlySound`, `PousSound`, `BandwidthAuditSound`,
`NetworkCertifierSound`, `WipeSound`), and `RemoteSoundness` the same statement at remote, whose subprotocols are a subset
of onsite's, from its three (sampled proofs, PoUW and the inference-only policy). They are proved by
`Properties.Composition` (a composite's acceptance is its components' and its record's) and the union bound over one
strategy in one game.

Each target is a hypothesis until its subprotocol's piece proves it at its component, and `Vocab`'s fields are given
until each piece defines its own: v2's reading of the blob, the positions, the rounds and the releases each component
read, the extractor and the finders, the gates computed, the world, the declaration's registered inputs, the wipe and the
play read at its boundary, and the published certificates, thresholds, errors and collision terms.
-/

namespace Verity

open Accounting

/-- **Soundness at every composite's verifier of record, from its subprotocols' targets.** On any device, for any names
`V` the subprotocols define and any composite `k`, if each of `k`'s subprotocols meets its target at `k` in the causal
game, then at all parameters, committed files and commitments and for every strategy, the probability that `k`'s
compiled verifier accepts files that passed the commit phase's check while the claim of one of `k`'s subprotocols fails
is at most their errors at that strategy summed, each its published `δ` plus its collision term at its finder's
probability of a SHA-512 collision. -/
def SoundnessOf : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (V : Vocab d α) (k : Verity.Registration.Composite),
    (∀ a ∈ subprotocols k, V.Target k a) → V.SoundnessAt k

/-- **Soundness at onsite's verifier of record, from its seven subprotocols' targets.** If sampled proofs, PoUW, the
inference-only policy, PoUS, PoUB-P2, the network certifier and the wipe each meet their target at onsite, then the
probability that onsite's compiled verifier accepts files that passed the commit phase's check while one of the
period's claims fails (more than an `ε` share of the units unbound in the extractor's transcript, fewer than `W` gates
computed after their coin, a declaration that doesn't comply with the policy, fewer than `S` bits held on average,
fewer than `Q` bits read, more than `K` bits through the network certifiers, or foreign content kept across the wipe past
its survivor threshold) is at most the seven errors at that strategy summed. -/
def Soundness : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (V : Vocab d α),
    V.SampledProofsSound .onsite → V.PouwSound .onsite → V.InferenceOnlySound .onsite → V.PousSound .onsite →
      V.BandwidthAuditSound .onsite → V.NetworkCertifierSound .onsite → V.WipeSound .onsite → V.SoundnessAt .onsite

/-- **Soundness at remote's verifier of record: onsite's statement restricted to remote's subprotocols.** If sampled
proofs, PoUW and the inference-only policy each meet their target at remote, then the probability that remote's
compiled verifier accepts files that passed the commit phase's check while one of their claims fails is at most their
three errors at that strategy summed. -/
def RemoteSoundness : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (V : Vocab d α),
    V.SampledProofsSound .remote → V.PouwSound .remote → V.InferenceOnlySound .remote → V.SoundnessAt .remote

end Verity
