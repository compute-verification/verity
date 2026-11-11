module

public import Definitions.NetworkCertifier.Warden

/-!
# The tap: one auditor device on each server's uplink, the warden and the certifier at once

Definitions only. The warden (`Definitions.NetworkCertifier.Warden`) and the network certifier are one device, a tap
on each AI server's uplink. Its one code run on what reaches it from the server (the `Arrivals` of each
connection) gives both what the client receives, the warden's cells (`certifierCode`), and the certifier's record of
them, the observations the audits read (`recordingCode`). Its trust is one `trusted-host`
(`Verity.Assumptions.TrustedHost`) over that code, with no second device and no second trust, so that what the
certifier saw is what the warden sent is a fact of the code, not a premise.

**The record is the wire.** The deployed tap (`benchmarks/network_traces/warden_serve.py`, one relay on the auditor's
machine) records every cell it sends, padding and fail cells included, a row a bucket, each cell a frame that names its
connection (`verity/core/protocols/certifier/warden_rows.py`; `Definitions.NetworkCertifier.RecordRows` is that
format). So on every schedule the format holds (`RecordFormat.fits`: a record of the monitoring period's buckets, each of
`S` slots, holds a schedule whose cells fall in them, at most `S` a bucket), the cells read back from the record under
the schedule (`RecordFormat.read_record`). What else the record holds, each step's jitter class and statuses, may depend on
anything the tap saw. The server's bytes after a connection's fail cell, or past its lane's end, reach the tap but
neither the client nor the record: the record fixes what the client received, not everything the server sent.

The tap is stated here, over `TrustedHost`'s shape (a device, its run and its code), so that core's device vocabulary
can lift it as it is: `RecordFormat` and `recordingCode` read nothing of network accounting but the warden's code.
-/

@[expose] public section

namespace NetTiming

/-- **The tap's record format**: what the tap writes to its record on schedule `s`, from the arrivals `a` it saw and
the cells `v` it sent on each connection, each with its bucket (`record`), how the cells read back from a record under
the schedule (`read`), and the schedules the format holds (`fits`). On a schedule it holds, the record holds every cell
the tap's code sends, in its bucket (`read_record`). -/
structure RecordFormat (κ Cell Ob : Type) where
  record : CertifierSchedule κ → (κ → Arrivals Cell) → (κ → List (ℕ × CellView Cell)) → Ob
  read : CertifierSchedule κ → Ob → κ → List (ℕ × CellView Cell)
  fits : CertifierSchedule κ → Prop
  read_record : ∀ s, fits s → ∀ a, read s (record s a (certifierCode s a)) = certifierCode s a

/-- **The tap's code**: on schedule `s` and the arrivals `arr c` of each connection `c` on the server's uplink, the
cells the client receives (`certifierCode`) and the record of them (`f.record`), from one run. -/
def recordingCode {κ Cell Ob : Type} (f : RecordFormat κ Cell Ob) (s : CertifierSchedule κ) (arr : κ → Arrivals Cell) :
    (κ → List (ℕ × CellView Cell)) × Ob :=
  (certifierCode s arr, f.record s arr (certifierCode s arr))

end NetTiming
