import Mathlib
import Properties.CertifierDevice
import Proofs.Properties.CertifierDevice.Complete
import Proofs.Properties.CertifierDevice.Issuer

/-!
# The certifier device's properties, proved

Each statement of `Properties.CertifierDevice`, proved as the lower-case theorem of its name: the trace properties from
one step's (`Proofs.CertifierDevice.Step`, `Proofs.CertifierDevice.Trace`), complete records from the egress FIFO's
(`Proofs.CertifierDevice.Fifo`, `Proofs.CertifierDevice.Complete`, with a complete window's clean ticks from
`Proofs.CertifierDevice.Late`),
the issuer's from LMS's correctness (`Proofs.CertifierDevice.Lms`, `Proofs.CertifierDevice.Issuer`), which is
completeness and no guarantee (`Proofs.CertifierDevice.Lms.lmsCorrect`).
-/

namespace Properties.CertifierDevice

/-- **(1) The schedule.** -/
theorem linkSchedule : LinkSchedule := Proofs.linkSchedule_proof

/-- **(1) Content independence.** -/
theorem contentIndependent : ContentIndependent := Proofs.contentIndependent_proof

/-- **(2) The records are the rows written.** -/
theorem records : Records := Proofs.records_proof

/-- **(2) A complete record is the FIFO's release of what was handed.** -/
theorem completeRows : CompleteRows := Proofs.completeRows_proof

/-- **(3) Fail-closed.** -/
theorem failClosed : FailClosed := Proofs.failClosed_proof

/-- **(4) The issuer's certificates verify.** -/
theorem issuerVerifies : IssuerVerifies := Proofs.Issuer.issuerVerifies_proof

end Properties.CertifierDevice
