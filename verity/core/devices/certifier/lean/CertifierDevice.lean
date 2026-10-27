import CertifierDevice.Frame
import CertifierDevice.Certifier
import CertifierDevice.Online
import CertifierDevice.Served
import CertifierDevice.Shaper
import CertifierDevice.Device
import CertifierDevice.Driver
import CertifierDevice.Issuer
import CertifierDevice.Difftest
import CertifierDevice.IssuerDifftest

/-!
# The active network certifier device

Executable Lean of `verity/core/protocols/certifier/active.py`'s proxy as a pure state machine
(`CertifierDevice.step`, in `CertifierDevice.Device`), over the reference's certifiers (`Certifier`, `Shaper`, `Bucketer`),
online prover (`Online`) and a served link's admission rule (`Served`), the driver that runs it in `core_world`'s `Core.World` (`Driver`, `run`), and the issuer
that signs each closed link-window's record, hidden under a salt of the certifier's own coins, at a leaf the host
reserves, and each egress window's clock sync, hidden under a salt of its own, in a clock certificate (`Issuer`,
`certificate.py`'s `Issuer`). It
reuses `certifier_program` (`NetworkCertifier`: statuses, `ClockSync`, `SyncSet`, `eligible`), `network_check`
(`NetworkCheck`: grid parameters, the record head and certificate encodings), `lms` (`Lms`: signing and verification)
and `core_world` (`Core.World`, `Core.Conn`). Code: no Mathlib and nothing of `verity/Security/`. `Difftest` holds `step` to `active.Proxy` on generated traces, and `IssuerDifftest` holds
`Issuer.certify` and `Issuer.clock` to `certificate.Issuer` on generated records and syncs.
-/
