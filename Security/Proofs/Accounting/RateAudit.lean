import Proofs.Accounting.RateAudit.Audit
import Proofs.Accounting.RateAudit.Honest
import Proofs.Accounting.RateAudit.Windows
import Proofs.Accounting.RateAudit.Joint
import Proofs.Accounting.RateAudit.Relay
import Proofs.Accounting.RateAudit.Interactive
import Proofs.Accounting.RateAudit.Sketch
import Proofs.Accounting.RateAudit.SketchHonest
import Proofs.Accounting.RateAudit.SketchInteractive
import Proofs.Accounting.RateAudit.SketchTable

/-! The rate audit's proofs: the guarantee in the random-oracle model (`Accounting.SecurityProofs.RateAudit`), its
completeness (`Accounting.SecurityProofs.RateHonest`), its time average over windows
(`Accounting.SecurityProofs.RateAuditWindows`), its reading in a joint run with the rebuilds charged as compute
(`Accounting.SecurityProofs.RateOfAudit`), the guarantee with side sources that answer what the chip sent
(`Accounting.SecurityProofs.RateAuditInteractiveRO` when they ignore the oracle, `RateAuditInteractive` with relayed
tags, and `RateOfAuditInteractive` and `RateOfAuditInteractiveRO` in a joint run), and the sketch-then-hash variant for
a device that reads sectors verbatim (`Accounting.SecurityProofs.RateAuditSketch`, its completeness
`Accounting.SecurityProofs.RateSketchHonest`, with interactive side sources that ignore the oracle
`Accounting.SecurityProofs.RateAuditSketchInteractiveRO`,
`Accounting.SecurityProofs.SketchBinds`, `Accounting.SecurityProofs.RateAuditTensorSketch`), and outside that class
the table that reads only the digest (`Accounting.SecurityProofs.RateSketchTable`,
`Accounting.SecurityProofs.RateTensorSketchTable`). -/
