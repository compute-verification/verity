import FlockSoundness.Discharge.FailClosed.Scope
import FlockSoundness.Discharge.FailClosed.Composed
import FlockSoundness.Discharge.FailClosed.ZkHidden
import FlockSoundness.Discharge.FailClosed.Exec

/-!
# The headlines over what the verifier accepts

`flock-verify verify` refuses every statement whose form the soundness proof does not cover (`Flock.ProvedScope.check`),
so its acceptance (`Verifies`, `VerifiesZK`) carries the facts about the statement that the headlines took as
hypotheses (`FailClosed/Scope.lean`). The headlines restated over it, with no hypothesis on the statement's form:
`FailClosed/Composed.lean` (public outputs, `--zk`), `FailClosed/ZkHidden.lean` (hidden outputs, `--zk`) and
`FailClosed/Exec.lean` (one table, no `--zk`, public or hidden outputs).
-/
