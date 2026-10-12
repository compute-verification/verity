import Proofs.Accounting.Bandwidth.AuditRO
import Proofs.Accounting.Bandwidth.Honest
import Proofs.Accounting.Bandwidth.Relay
import Proofs.Accounting.Bandwidth.Interactive
import Proofs.Accounting.Bandwidth.Actual
import Proofs.Accounting.Bandwidth.Ticked

/-! The bandwidth audit v0's proofs: the guarantee in the random-oracle model (`Bandwidth.SecurityProofs.AuditRO`), its
completeness (`Bandwidth.SecurityProofs.HonestPasses`), the guarantee with relayed tags
(`Bandwidth.SecurityProofs.AuditRelay`), with side sources that answer what the chip sent
(`Bandwidth.SecurityProofs.AuditInteractive`, and `AuditInteractiveRO` when they ignore the oracle), and on the actual
circuit given a simulation (`Bandwidth.SecurityProofs.Actual.audit`, B3), and for the cores as a tick transducer over
the links' alphabets (`Bandwidth.SecurityProofs.Ticked.audit`, B3 in the capacity form). -/
