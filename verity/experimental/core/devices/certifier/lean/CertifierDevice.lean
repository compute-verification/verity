import CertifierDevice.Frame
import CertifierDevice.Certifier
import CertifierDevice.Online
import CertifierDevice.Shaper
import CertifierDevice.Device

/-!
# The active network certifier device

Executable Lean of `verity/experimental/protocols/certifier/active.py`'s proxy as a pure state machine
(`CertifierDevice.step`, in `CertifierDevice.Device`), over the reference's certifiers (`Certifier`, `Shaper`, `Bucketer`)
and online prover (`Online`). It reuses `certifier_program` (`NetworkCertifier`: statuses, `ClockSync`, `SyncSet`,
`eligible`) and `network_check` (`NetworkCheck`: grid parameters). Code: no Mathlib and nothing of `verity/Security/`; it
lives under `verity/experimental/core/` until its properties are proved.
-/
