import Lms.Verify
import Lms.Sign
import Lms.Hss
import Lms.Difftest
import Lms.SignDifftest

/-!
# LMS verification and signing

Executable Lean that reads LMS's canonical formats, the signature and the public key, and verifies, for every parameter
set `verity/ml/crypto/lms.py` admits (`Lms.Verify`); that derives a key from its seed and signs at a given leaf
(`Lms.Sign`); that verifies a device's two-level HSS signatures and its keys' role digests (`Lms.Hss`, held to the Python
by `MemoryChallenger.Difftest`); and their difftests against that file (`Lms.Difftest`, `Lms.SignDifftest`). Code: it
imports only Lean and C-Flock's verifier (`Flock.Hash`, `Flock.Bytes`).
-/
