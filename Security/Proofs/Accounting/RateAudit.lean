import Proofs.Accounting.RateAudit.Audit
import Proofs.Accounting.RateAudit.Honest
import Proofs.Accounting.RateAudit.Windows
import Proofs.Accounting.RateAudit.Joint
import Proofs.Accounting.RateAudit.Sketch
import Proofs.Accounting.RateAudit.SketchHonest

/-! The rate audit's proofs: the guarantee in the random-oracle model (`Accounting.SecurityProofs.RateAudit`), its
completeness (`Accounting.SecurityProofs.RateHonest`), its time average over windows
(`Accounting.SecurityProofs.RateAuditWindows`), its reading in a joint run with the rebuilds charged as compute
(`Accounting.SecurityProofs.RateOfAudit`), and the sketch-then-hash variant for a device that reads sectors verbatim
(`Accounting.SecurityProofs.RateAuditSketch`, its completeness `Accounting.SecurityProofs.RateSketchHonest`,
`Accounting.SecurityProofs.SketchBinds`, `Accounting.SecurityProofs.RateAuditTensorSketch`). -/
