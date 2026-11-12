module

public import Definitions.Core.Transport

/-!
# The transport's named assumptions

Two premises on a channel between an auditor party and the prover (`verity.commitments.crypto.transport`), named
assumptions under `Definitions.Assumptions`: `ImprintPathPrivate`, a placement premise that says who can read the bytes
of a channel's one imprint exchange in transit, which nothing proves; and `ChannelAuthentic`, what the key schedule and
the record layer give, a theorem target from `euf-cma/lms-sha512-n64` and `prf/sha-512`. The transport's other premises
(`WitnessPathPrivate`, `ChannelHides` and `WitnessPathPrivateOrChannelHides`) are in `Definitions.Core.Assumptions.Transport`,
which re-exports this module. Both sit apart from the rest of the named assumptions, which `Definitions.Assumptions`
re-exports, so that those elaborate without the transport's vocabulary.
-/

@[expose] public section

namespace Verity.Assumptions

open Verity.Transport

/-- **`imprint-path-private`**: the same as `witness-path-private`, only for the bytes of `c`'s one imprint exchange,
over the cable between an onsite auditor device and its server once per device (the imprint key). The premise shrinks
from every frame of the channel's life to that one exchange. -/
def ImprintPathPrivate {Node : Type} (s : Roles Node) (c : Channel Node) : Prop :=
  PathPrivate s c.party c.readsImprint

/-- **`channel-authentic`**: in every run, the frames a receiver accepts are a prefix of the frames its peer sent, or the
run holds a forgery: an LMS signature on a new handshake message (`euf-cma/lms-sha512-n64`), or a tag under the
session's keys on a new unit (`prf/sha-512`). A receiver opens each unit only at its own position and refuses every unit
after one it refused (`transport.Receiver`). -/
def ChannelAuthentic {Run Frame : Type} (sent accepted : Run → List Frame) (forgery : Run → Prop) : Prop :=
  ∀ r, accepted r <+: sent r ∨ forgery r

end Verity.Assumptions
