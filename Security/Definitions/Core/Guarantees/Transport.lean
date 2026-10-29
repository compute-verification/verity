import Definitions.Core.Assumptions.Transport

/-!
# What the transport establishes

Per channel, under its premises (`Assumptions/Transport`): a passive coalition sees, from outside the channel's ends,
what a simulator makes from the public record (`TransportPassiveZk`); the imprint leaves the coalition's view independent
of the root it derives (`ImprintRootHidden`); and a receiver delivers only what its peer sent, in order
(`TransportDelivers`).
-/

namespace Verity.Guarantees

open Verity.Transport Verity.Assumptions
open scoped ENNReal

/-- **Passive zero knowledge of one channel**: what a coalition `C` sees of the channel's private traffic `m` from
outside its ends (`outside`) is within `η` of what a simulator makes from the public record alone (`record m`, the
declared sizes), when no node outside the ends can read the channel's private frames (`WitnessPathPrivate s c`) or the
channel's keys hide the traffic against every test (`ChannelHides`, with `fill (record m)` the blank traffic of those
sizes). The simulator runs the channel on that blank traffic under fresh randomness of its own. -/
def TransportPassiveZk : Prop :=
  ∀ {Node Ω M W Rec : Type} [Fintype Ω] [Nonempty Ω] (s : Roles Node) (c : Channel Node) (C : Node → Prop)
    (wire : M → Ω → W) (record : M → Rec) (fill : Rec → M) (η : ℝ≥0∞) (m : M),
    (WitnessPathPrivate s c ∨ ∀ E, ChannelHides wire (fun m => fill (record m)) E η m) →
    ∀ E : Option W → Prop,
      ViewIsRecord (Ω' := Ω) (fun ω => seen (outside s c.party C) c.readsPrivate (wire m ω)) (fun _ => record m)
        (fun r ω' => seen (outside s c.party C) c.readsPrivate (wire (fill r) ω')) E η

/-- **The imprint root is hidden**: when every node that can read the cable during the imprint exchange is an end of it,
one of the prover's own machines or a `Confined` party (`ImprintPathPrivate s c`), what a coalition sees from outside the
ends of the exchange's random bytes `(R_D, R_S)`, jointly with the root `root R_D R_S` the two ends derive, is exactly a
simulator's from no record: no bytes, beside the root of a fresh exchange. -/
def ImprintRootHidden : Prop :=
  ∀ {Node R K : Type} [Fintype R] [Nonempty R] (s : Roles Node) (c : Channel Node) (C : Node → Prop)
    (root : R → R → K), ImprintPathPrivate s c →
    ∀ E : Option (R × R) × K → Prop,
      ViewIsRecord (Ω' := R × R) (fun x : R × R => (seen (outside s c.party C) c.readsImprint x, root x.1 x.2))
        (fun _ => ())
        (fun _ (x' : R × R) => ((none : Option (R × R)), root x'.1 x'.2)) E 0

/-- **A receiver delivers what its peer sent, in order**: under `ChannelAuthentic`, the bodies a receiver delivers (the
`body` of each frame it accepts) are a prefix of the bodies its peer sent, or the run holds a forgery, so no frame is
injected, replayed, reordered or reflected without one. -/
def TransportDelivers : Prop :=
  ∀ {Run Frame Body : Type} (sent accepted : Run → List Frame) (forgery : Run → Prop) (body : Frame → Body),
    ChannelAuthentic sent accepted forgery →
    ∀ r, (accepted r).map body <+: (sent r).map body ∨ forgery r

end Verity.Guarantees
