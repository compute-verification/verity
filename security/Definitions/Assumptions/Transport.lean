module

public import Definitions.Core.Transport

/-! The transport's named assumptions: `ImprintPathPrivate`, a placement premise nothing proves, and `ChannelAuthentic`, a
theorem target from `euf-cma/lms-sha512-n64` and `prf/sha-512`. -/

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
