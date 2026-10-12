import Proofs.Accounting.Bandwidth.AuditRO
import Proofs.Accounting.Bandwidth.Honest
import Proofs.Accounting.Bandwidth.Relay
import Proofs.Accounting.Bandwidth.Set

/-! The bandwidth audit v0's proofs: the guarantee in the random-oracle model (`Bandwidth.SecurityProofs.AuditRO`), its
completeness (`Bandwidth.SecurityProofs.HonestPasses`), the guarantee with relayed tags
(`Bandwidth.SecurityProofs.AuditRelay`), and the guarantee over a co-audited set (`Bandwidth.SecurityProofs.AuditSet`)
with its completeness (`Bandwidth.SecurityProofs.SetHonestPasses`). -/
