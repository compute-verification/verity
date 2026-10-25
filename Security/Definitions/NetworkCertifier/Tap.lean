import Definitions.Core.AuditorDevice
import Definitions.NetworkCertifier.Warden

/-!
# The warden and the certifier: one tap on the server's uplink

Definitions only. The active warden and the network certifier are one auditor device, a tap on the server's uplink
(`Verity.TapCode`). Its one code, on the warden's schedule and on what reaches it from the server, each connection's
`Arrivals`, relays the warden's cells (`wardenCode`) and records the certifier's records (`Rec`: for a period of
sampled audits, each window's observations), from the same arrivals. The records carry the egress data that arrived
(`WardenTap.read_record`), as the warden's records are the cells it sends (`verity/core/protocols/certifier/warden.py`,
"The records").
-/

namespace NetTiming

/-- **The tap's certifier half**: on the warden's schedule and the arrivals of every connection, the records the tap
keeps (`record`), and the egress data read back from them (`read`), which is the data that arrived on every connection
(`read_record`). -/
structure WardenTap (Conn Cell Rec : Type) where
  record : WardenSchedule Conn → (Conn → Arrivals Cell) → Rec
  read : Rec → Conn → List Cell
  read_record : ∀ s a, read (record s a) = fun c => (a c).stream

/-- **The tap's one code**: the warden's cells (`wardenCode`) and the certifier's records, from one schedule and one set
of arrivals. -/
def WardenTap.code {Conn Cell Rec : Type} (T : WardenTap Conn Cell Rec) :
    Verity.TapCode (WardenSchedule Conn) (Conn → Arrivals Cell) (Conn → List (ℕ × CellView Cell)) Rec
      (Conn → List Cell) where
  relay := wardenCode
  record := T.record
  data a c := (a c).stream
  read := T.read
  read_record := T.read_record

end NetTiming
