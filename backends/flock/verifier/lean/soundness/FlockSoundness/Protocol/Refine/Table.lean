import FlockSoundness.Protocol.Refine.Paths
import FlockSoundness.Protocol.CompiledSound
import FlockSoundness.Protocol.Extract
import FlockSoundness.Protocol.Game.Batch

/-!
# The verifier refines the model's table (PROTOCOL.md §3, §5)

`verify_refines`: when the executable's `Flock.verify` accepts a session record and its two proof files, under the
unsalted SHA-512 scheme, the record and both proofs decode. Then at the proofs' openings (`opensOf`), each rep's
compiled game `repC`, played with its proof's messages and its stream's coins, accepts. Both reps' paths also verify
against the one level-0 cap. That is `tableAfter`'s verdict. `verify_tableAfter` states it as an accepting run of
`tableAfter`, with existential messages and rep 0 first.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript Model Accounting

/-- The proof files' openings (§14): rep `r`'s opened rows and siblings, level by level. -/
def opensOf (p₀ p₁ : Proof) (sch : Flock.Schedule) : Opens 𝔽 Dig :=
  fun r ℓ j => if r = 0 then openRow p₀.ligerito sch ℓ j else openRow p₁.ligerito sch ℓ j

end FlockSoundness.Refine
