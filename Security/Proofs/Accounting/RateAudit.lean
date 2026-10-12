import Proofs.Accounting.RateAudit.Audit
import Proofs.Accounting.RateAudit.Honest
import Proofs.Accounting.RateAudit.Windows
import Proofs.Accounting.RateAudit.Joint
import Proofs.Accounting.RateAudit.Relay
import Proofs.Accounting.RateAudit.Interactive
import Proofs.Accounting.RateAudit.Sketch
import Proofs.Accounting.RateAudit.Conservation
import Proofs.Accounting.RateAudit.Chain

/-! The rate audit's proofs: the guarantee in the random-oracle model (`Accounting.SecurityProofs.RateAudit`), its
completeness (`Accounting.SecurityProofs.RateHonest`), its time average over windows
(`Accounting.SecurityProofs.RateAuditWindows`), its reading in a joint run with the rebuilds charged as compute
(`Accounting.SecurityProofs.RateOfAudit`), the guarantee with side sources that answer what the chip sent
(`Accounting.SecurityProofs.RateAuditInteractiveRO` when they ignore the oracle, `RateAuditInteractive` with relayed
tags, and `RateOfAuditInteractive` and `RateOfAuditInteractiveRO` in a joint run), and the sketch-then-hash variant for
a device that reads sectors verbatim (`Accounting.SecurityProofs.RateAuditSketch`,
`Accounting.SecurityProofs.SketchBinds`, `Accounting.SecurityProofs.RateAuditTensorSketch`), and the remote protocol's
conservation over tiers (`Accounting.SecurityProofs.RateConservationRO`, `RateConservation`, their capacity forms
`RateConservationCapRO` and `RateConservationCap`, and `RateDisplacement`), chained passes
(`Accounting.SecurityProofs.RateChainPassRO`, `RateChainPass`, `RateChainRO`, `RateChain`), and conservation at every
count, for a pass and for a window (`Accounting.SecurityProofs.RatePaidRO`, `RatePaid`, `RateChainPaidRO`,
`RateChainPaid`), and for a window whose roots are ground against the Fiat–Shamir seed
(`Accounting.SecurityProofs.RateGrindPaidRO`, `RateGrindPaid`). -/
