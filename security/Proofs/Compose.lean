module

public import Proofs.Compose.Joint
public import Proofs.Compose.JointLive
public import Proofs.Compose.Compose
public import Proofs.Compose.ComposeMonitoringPeriod
public import Proofs.Compose.ComposeMonitoringPeriod.Honest
public import Proofs.Compose.Additive

/-!
# The proofs of the composed audits' guarantees

The audits on one device together (`Joint*`), the circuit credit over a monitoring period (`Compose`,
`ComposeMonitoringPeriod`), and the additive and period-binding bounds. `Proofs.Accounting` imports the same modules.
-/
