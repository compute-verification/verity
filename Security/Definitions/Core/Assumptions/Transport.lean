import Definitions.Core.Transport

/-!
# The transport's named assumptions

The premises on a channel between an auditor party and the prover (`verity.ml.crypto.transport`). The two placement
premises say who can read a channel's bytes in transit, and nothing proves them; the two cryptographic ones are what the
key schedule and the record layer give, and are theorem targets from the claims their docstrings name. They sit apart
from `Assumptions` so that the other named assumptions elaborate without the transport's vocabulary.
-/

namespace Verity.Assumptions

open Verity.Transport
open scoped ENNReal

/-- **`witness-path-private`**, the per-channel premise, beside the per-party `Faithful p`, `Secret p` and `Confined p`
and the site's `Covered`: every node that can read a private frame of `c` in transit, over the
channel's life, is its auditor end, one of the prover's own machines, or an auditor party under `Confined p`. The
certifier on the server's own uplink meets it, and so does the memory challenger reached in-region; a channel across a
network the prover doesn't own meets it only with keys (`ChannelHides`). -/
def WitnessPathPrivate {Node : Type} (s : Roles Node) (c : Channel Node) : Prop :=
  PathPrivate s c.party c.readsPrivate

/-- **`imprint-path-private`**: the same, only for the bytes of `c`'s one imprint exchange, over the cable between an
onsite auditor device and its server once per device (the imprint key). The premise shrinks from every
frame of the channel's life to that one exchange. -/
def ImprintPathPrivate {Node : Type} (s : Roles Node) (c : Channel Node) : Prop :=
  PathPrivate s c.party c.readsImprint

/-- **`channel-hides`**, for one test `E` on what a reader outside the channel's ends sees: the wire bytes of the private
traffic `m` (sealed under the channel's randomness `ω`: its keys and the session's) and of `blank m` (traffic of the same
declared sizes, as the public record has it) are within `η`. Under a secret chain key it is a target from `prf/sha-512`
(the session, traffic and AEAD keys are HMAC-SHA512 outputs, and `aead` is counter mode then HMAC); for a remote party's
session, whose key XORs in an ML-KEM-1024 secret, from `ind-cca/ml-kem-1024` in the chain key's place. -/
def ChannelHides {Ω M W : Type} [Fintype Ω] [Nonempty Ω] (wire : M → Ω → W) (blank : M → M) (E : W → Prop)
    (η : ℝ≥0∞) (m : M) : Prop :=
  Close (Game.value E ((Game.draw Ω).map (wire m))) (Game.value E ((Game.draw Ω).map (wire (blank m)))) η

/-- **`channel-authentic`**: in every run, the frames a receiver accepts are a prefix of the frames its peer sent, or the
run holds a forgery: an LMS signature on a new handshake message (`euf-cma/lms-sha512-n64`), or a tag under the
session's keys on a new unit (`prf/sha-512`). A receiver opens each unit only at its own position and refuses every unit
after one it refused (`transport.Receiver`). -/
def ChannelAuthentic {Run Frame : Type} (sent accepted : Run → List Frame) (forgery : Run → Prop) : Prop :=
  ∀ r, accepted r <+: sent r ∨ forgery r

end Verity.Assumptions
