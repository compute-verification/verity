import Proofs.Accounting.Bandwidth.AuditRO
import Proofs.Accounting.Bandwidth.Honest
import Proofs.Accounting.Bandwidth.Relay
import Proofs.Accounting.Bandwidth.Interactive

/-! The bandwidth audit v0's proofs: the guarantee in the random-oracle model (`Bandwidth.SecurityProofs.AuditRO`), its
completeness (`Bandwidth.SecurityProofs.HonestPasses`), the guarantee with relayed tags
(`Bandwidth.SecurityProofs.AuditRelay`), and with side sources that answer what the chip sent
(`Bandwidth.SecurityProofs.AuditInteractive`, and `AuditInteractiveRO` when they ignore the oracle). -/
