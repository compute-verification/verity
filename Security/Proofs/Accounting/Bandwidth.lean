import Proofs.Accounting.Bandwidth.AuditRO
import Proofs.Accounting.Bandwidth.Honest
import Proofs.Accounting.Bandwidth.Relay
import Proofs.Accounting.Bandwidth.Interactive
import Proofs.Accounting.Bandwidth.Actual
import Proofs.Accounting.Bandwidth.Ticked
import Proofs.Accounting.Bandwidth.Causal
import Proofs.Accounting.Bandwidth.Joint
import Proofs.Accounting.Bandwidth.TickedJoint
import Proofs.Accounting.Bandwidth.CausalJoint

/-! The bandwidth audit v0's proofs: the guarantee in the random-oracle model (`Bandwidth.SecurityProofs.AuditRO`), its
completeness (`Bandwidth.SecurityProofs.HonestPasses`), the guarantee with relayed tags
(`Bandwidth.SecurityProofs.AuditRelay`), with side sources that answer what the chip sent
(`Bandwidth.SecurityProofs.AuditInteractive`, and `AuditInteractiveRO` when they ignore the oracle), and on the actual
circuit given a simulation (`Bandwidth.SecurityProofs.Actual.audit`, B3), and for the chip as a tick transducer over
the links' signals (B3: `Bandwidth.SecurityProofs.Ticked.audit_reads` in the reads form, given passive memory, and
`Ticked.audit` in the capacity form), and for the chip of a strategy causal from every tick whose links split what
crosses into it (`Bandwidth.SecurityProofs.Causal.audit_reads`, with the chip's state `M + 1` bits), and from the links'
alphabets, passive memory and the oracle as a location (`Causal.audit_links`; `Causal.audit_capacity`, the capacity
form); and bandwidth additivity by counting, for a device that also holds a secret its chip's view determines
(`Bandwidth.SecurityProofs.AuditJoint`, Mbw), and for the chip as a tick transducer whose view determines it
(`Bandwidth.SecurityProofs.Ticked.audit_joint`; `Ticked.audit_joint_capacity`, the capacity form), and for the chip of a
strategy causal from every tick over the store and that secret (`Bandwidth.SecurityProofs.Causal.audit_joint`), and
from its links (`Causal.audit_links_joint`; `Causal.audit_capacity_joint`, the capacity form). -/
