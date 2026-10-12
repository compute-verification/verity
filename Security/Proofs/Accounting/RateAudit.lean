import Proofs.Accounting.RateAudit.Audit
import Proofs.Accounting.RateAudit.Honest
import Proofs.Accounting.RateAudit.Windows
import Proofs.Accounting.RateAudit.Joint
import Proofs.Accounting.RateAudit.Relay
import Proofs.Accounting.RateAudit.Interactive

/-! The rate audit's proofs: the guarantee in the random-oracle model (`Accounting.SecurityProofs.RateAudit`), its
completeness (`Accounting.SecurityProofs.RateHonest`), its time average over windows
(`Accounting.SecurityProofs.RateAuditWindows`), its reading in a joint run with the rebuilds charged as compute
(`Accounting.SecurityProofs.RateOfAudit`), and the guarantee with side sources that answer what the chip sent
(`Accounting.SecurityProofs.RateAuditInteractiveRO` when they ignore the oracle, `RateAuditInteractive` with relayed
tags, and `RateOfAuditInteractive` and `RateOfAuditInteractiveRO` in a joint run). -/
