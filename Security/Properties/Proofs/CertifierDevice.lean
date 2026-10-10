import Mathlib
import Properties.CertifierDevice
import Properties.Proofs.CertifierDevice.Complete
import Properties.Proofs.CertifierDevice.Issuer

/-!
# The certifier device's properties, proved

Each statement of `Properties.CertifierDevice`, proved as the lower-case theorem of its name: the trace properties from
one step's (`Proofs.CertifierDevice.Step`, `Proofs.CertifierDevice.Trace`), complete records from the egress FIFO's and
the handing buckets' (`Proofs.CertifierDevice.Fifo`, `Proofs.CertifierDevice.Late`, `Proofs.CertifierDevice.Complete`),
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

/-- **(2) A complete record is the Program's.** -/
theorem completeRows : CompleteRows := Proofs.completeRows_proof

/-- **(3) Fail-closed.** -/
theorem failClosed : FailClosed := Proofs.failClosed_proof

/-- **(4) The issuer's certificates verify.** -/
theorem issuerVerifies : IssuerVerifies := Proofs.Issuer.issuerVerifies_proof

end Properties.CertifierDevice
