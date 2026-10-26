import Properties.Verity.Game

/-!
# Soundness at the verifier of record, composed from its subprotocols' targets

`Accounting.Guarantees.JointPeriod` composes the per-period guarantees of the subprotocols that decide a period over a
period run whose verdicts are a free field (`PeriodRun.accepts`). Here the verdicts are not free: in the play `τ`, the
subprotocol `a` accepts at the coins `ξ` when its compiled component accepts its own case in what the prover sent and
the coins it read are the ones the device logged for it (`Vocab.run`), and the event is the verifier of record's
acceptance (`Accepts`) outside the break. `Soundness` states that each subprotocol's target at that run
(`Vocab.SampledProofsSound`, `PouwSound`, `PousSound`, `BandwidthAuditSound`, `NetworkCertifierSound`) gives
`Vocab.Soundness`: for every strategy, a transcript fixed by the commitment at which onsite accepts while one of the
period's claims fails, with no break, with probability at most the published errors summed. It is proved by
`Properties.Composition` (onsite's acceptance is every component's) and `JointPeriod`.

Each target is a hypothesis until its subprotocol's piece proves it at its component (the one-theorem plan's ONE-8,
ONE-10, ONE-11, ONE-12 and ONE-16, in `note:proofs/20261026T0332Z-report-onethm`), and `Vocab`'s fields are given
until each piece defines its own: the circuit and partition (ONE-7), the coins each component read (ONE-4), the world
(ONE-10, ONE-11, ONE-12, ONE-16) and the published certificates and errors (ONE-19). The wipe, the sixth subprotocol,
joins in `Properties.Verity.Wipe`.
-/

namespace Verity

open Accounting

/-- **Soundness at the verifier of record, from each subprotocol's target.** On any device, for any names `V` the
subprotocols define on one coin space, if sampled proofs, PoUW, PoUS, the bandwidth audit and the network certifier
each have their per-period guarantee at the run their play realizes, on the device's logged coins, then for every
prover strategy there is a transcript fixed by the commitment such that the probability that onsite's compiled
verifier accepts while one of the period's claims fails (more than an `ε` share of the circuit's units wrong, less
than `W` of work, fewer than `S` bits held on average, fewer than `Q` bits read, or more than `K` bits through the
network certifiers), with no break, is at most the published errors summed. -/
def Soundness : Prop :=
  ∀ (ι α I : Type) [Fintype α] [Fintype I] [DecidableEq I] (Ω : I → Type) [∀ i, Fintype (Ω i)]
    [∀ i, Nonempty (Ω i)] (d : Device ι) (V : Vocab d α Ω),
    V.SampledProofsSound → V.PouwSound → V.PousSound → V.BandwidthAuditSound → V.NetworkCertifierSound →
      V.Soundness

end Verity
