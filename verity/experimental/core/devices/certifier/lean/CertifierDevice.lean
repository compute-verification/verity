import CertifierDevice.Frame
import CertifierDevice.Certifier
import CertifierDevice.Online
import CertifierDevice.Shaper
import CertifierDevice.Device
import CertifierDevice.World
import CertifierDevice.Issuer
import CertifierDevice.Difftest
import CertifierDevice.IssuerDifftest

/-!
# The active network certifier device

Executable Lean of `verity/experimental/protocols/certifier/active.py`'s proxy as a pure state machine
(`CertifierDevice.step`, in `CertifierDevice.Device`), over the reference's certifiers (`Certifier`, `Shaper`, `Bucketer`)
and online prover (`Online`), a minimal host (`World`) with the driver that runs the device in it (`run`), and the issuer
that signs each closed link-window's record at a leaf the host reserves (`Issuer`, `certificate.py`'s `Issuer`). It
reuses `certifier_program` (`NetworkCertifier`: statuses, `ClockSync`, `SyncSet`, `eligible`), `network_check`
(`NetworkCheck`: grid parameters, the record head and certificate encodings) and `lms` (`Lms`: signing and
verification). Code: no Mathlib and nothing of `verity/Security/`; it lives under `verity/experimental/core/` until its
properties are proved. `Difftest` holds `step` to `active.Proxy` on generated traces, and `IssuerDifftest` holds
`Issuer.certify` to `certificate.Issuer` on generated records.
-/
