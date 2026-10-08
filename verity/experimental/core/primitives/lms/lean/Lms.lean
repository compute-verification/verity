import Lms.Verify
import Lms.Difftest

/-!
# LMS verification

Executable Lean that reads LMS's canonical formats, the signature and the public key, and verifies, for every parameter
set `verity/ml/crypto/lms.py` admits (`Lms.Verify`), and its difftest against that file (`Lms.Difftest`). Code: it
imports only Lean and C-Flock's verifier (`Flock.Hash`, `Flock.Bytes`), and lives under `verity/experimental/core/` until
its properties are proved.
-/
