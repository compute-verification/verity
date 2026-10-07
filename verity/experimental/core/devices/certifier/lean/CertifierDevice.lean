import CertifierDevice.Frame
import CertifierDevice.Certifier
import CertifierDevice.Online
import CertifierDevice.Shaper
import CertifierDevice.Device
import CertifierDevice.World
import CertifierDevice.Difftest

/-!
# The active network certifier device

Executable Lean of `verity/experimental/protocols/certifier/active.py`'s proxy as a pure state machine
(`CertifierDevice.step`, in `CertifierDevice.Device`), over the reference's certifiers (`Certifier`, `Shaper`, `Bucketer`)
and online prover (`Online`), and a minimal host (`World`) with the driver that runs the device in it (`run`). It reuses
`certifier_program` (`NetworkCertifier`: statuses, `ClockSync`, `SyncSet`, `eligible`) and `network_check`
(`NetworkCheck`: grid parameters). Code: no Mathlib and nothing of `verity/Security/`; it lives under
`verity/experimental/core/` until its properties are proved. `Difftest` holds `step` to `active.Proxy` on generated
traces.
-/
