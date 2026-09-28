import Properties.CertifierNetwork
import Properties.Proofs.CertifierNetwork.Accepted

/-!
# The certifier device's output is accepted, proved

`Properties.CertifierNetwork`'s statement, proved as the lower-case theorem of its name: the records a closed run
publishes, one per link-window, each of its grid (`Proofs.CertifierNetwork.Bucketer`, `Proofs.CertifierNetwork.Records`);
the frames the shaper hands, `8·B` bits each, read back as their bytes (`Proofs.CertifierNetwork.Frames`); each record's
head and certificate (`Proofs.CertifierNetwork.Encode`, `Proofs.CertifierNetwork.Certify`); and the egress grids the
Program's, so the check's honest transcript (`Proofs.CertifierNetwork.Accepted`).
-/

namespace Properties.CertifierNetwork

/-- **The device's output is accepted.** -/
theorem deviceOutputAccepted : DeviceOutputAccepted := Proofs.deviceOutputAccepted_proof

end Properties.CertifierNetwork
