import Proofs.Properties.CertifierNetwork.Defs
import Proofs.Properties.CertifierNetwork.Accepted
import Proofs.Properties.CertifierNetwork.Instance

/-!
# The certifier device's output is accepted: completeness, proved

`DeviceOutputAccepted` (`Proofs.CertifierNetwork.Defs`), proved as the lower-case theorem of its name: the records a
closed run publishes, one per sampling block, each of its grid (`Proofs.CertifierNetwork.Bucketer`,
`Proofs.CertifierNetwork.Records`); the frames the shaper hands, `8·B` bits each, read back as their bytes
(`Proofs.CertifierNetwork.Frames`); each record's head and certificate (`Proofs.CertifierNetwork.Encode`,
`Proofs.CertifierNetwork.Certify`); and the egress grids the Program's, so the check's honest transcript
(`Proofs.CertifierNetwork.Accepted`). It is completeness, not a guarantee: `Proofs.CertifierNetwork.Instance` gives one
configuration, issuer and run where its hypotheses hold.
-/

namespace Properties.CertifierNetwork

/-- **The device's output is accepted** (completeness). -/
theorem deviceOutputAccepted : DeviceOutputAccepted := Proofs.deviceOutputAccepted_proof

end Properties.CertifierNetwork
