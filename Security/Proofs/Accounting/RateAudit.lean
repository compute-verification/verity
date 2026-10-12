import Proofs.Accounting.RateAudit.Audit
import Proofs.Accounting.RateAudit.Honest
import Proofs.Accounting.RateAudit.Windows
import Proofs.Accounting.RateAudit.Joint

/-! The rate audit's proofs: the guarantee in the random-oracle model (`Accounting.SecurityProofs.RateAudit`), its
completeness (`Accounting.SecurityProofs.RateHonest`), its time average over windows
(`Accounting.SecurityProofs.RateAuditWindows`), and its reading in a joint run with the rebuilds charged as compute
(`Accounting.SecurityProofs.RateOfAudit`). -/
