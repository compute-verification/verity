import NetworkCheck.Certificate
import NetworkCheck.Difftest

/-!
# The network certificate's checks

Executable Lean that reads the network certificate's canonical formats (`NetworkCheck.Certificate`: the frame tree, the
parameters digest, the record head and its digest, the message a certificate signs), and its difftest against
`verity/experimental/protocols/certifier/certificate.py` (`NetworkCheck.Difftest`). Code: it imports only Lean and C-Flock's
verifier (`Flock.Hash`, `Flock.Bytes`, `Flock.Canon`), and lives under `verity/experimental/core/` until its properties are
proved.
-/
