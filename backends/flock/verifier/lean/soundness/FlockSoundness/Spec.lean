import FlockSoundness.Spec.Protocol
import FlockSoundness.Spec.Assumptions

/-!
# C-Flock's spec

The trusted text of C-Flock's claims, in closed form over the executable verifier (`Flock.*`) and Mathlib: what
`flock-verify verify` accepts (`FlockSpec.Accepts`), the interaction model, and the named assumptions. Nothing here
imports a proof (the `layers` rules in `lean-audit.json`), and `FlockSoundness.SpecBridge` proves each definition the
proofs' own.

The guarantees are not stated here yet. `Discharge.FailClosed.flock_verify_sound`'s eight conclusions read the
proofs' extractor, collision finders and zero-knowledge view (`Exec.XplurL` over `Exec.simAuditOn`, `Exec.LinkCRL`,
`Exec.TableCRL`, `ZkSession.TableCRZ`, `Composed.coinedOf`, `Composed.ZkViewAt`): a claim's meaning is the reduction it
names, so stating one here means stating that model here.
-/
