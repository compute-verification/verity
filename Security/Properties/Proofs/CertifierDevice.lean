import Mathlib
import Properties.CertifierDevice
import Properties.Proofs.CertifierDevice.Trace

/-!
# The certifier device's properties, proved

Each statement of `Properties.CertifierDevice`, proved as the lower-case theorem of its name: the trace properties from
one step's (`Proofs.CertifierDevice.Step`, `Proofs.CertifierDevice.Trace`).
-/

namespace Properties.CertifierDevice

/-- **(1) The schedule.** -/
theorem linkSchedule : LinkSchedule := Proofs.linkSchedule_proof

/-- **(1) Content independence.** -/
theorem contentIndependent : ContentIndependent := Proofs.contentIndependent_proof

/-- **(2) The records are the rows written.** -/
theorem records : Records := Proofs.records_proof

/-- **(3) Fail-closed.** -/
theorem failClosed : FailClosed := Proofs.failClosed_proof

end Properties.CertifierDevice
