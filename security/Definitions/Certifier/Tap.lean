module

public import Definitions.Certifier.Cells

/-! The tap: one auditor device on each server's uplink, the relay and the certifier at once, under one `trusted-device`. -/

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
