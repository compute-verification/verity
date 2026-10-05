import FlockSoundness.Discharge.ZkLink.Compiled
import FlockSoundness.Discharge.Exec.Live
import FlockSoundness.Protocol.Discharge.ZkLink.One

/-!
# One compiled `--zk` table as `Refine.modelTable`

`sessionZC_one`: the compiled session on one table is `modelTable`'s shape, with the reps `repZCAt` and the output
`[outZC …]`, as `Exec.sessionBL_one` at M0. So a one-table `modelTable` refinement (`G := repZCAt A Hs T`,
`Cp := Fin 1 → Cap D`, `out := fun caps pend os => [outZC Hs T (caps 0) pend os]`, `acc := acceptedZC [T]`) lands on
`batchedSessionZC` at the one-table plan (`batchedSessionZC_one`), where the compiled count curves apply.
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession Refine

local notation "𝔽" => FlockLevel3.GF128

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}

variable [Inhabited D] [Inhabited Sl]

end FlockSoundness.Discharge.ZkLink
