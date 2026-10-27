import Proofs.Flock.Claims.KS.PlainKSLinR
import Proofs.Flock.Claims.KS.PlainRConflict

/-!
# The table bound and `KSFlockLin` at the plain record at the deployed reader `VR rd`

`tableBoundAbortLin_VR` at `deployPlainR rel rd`, with `VR rd`'s `AccLast` and `DecideFree` (`accLast_VR`,
`decideFree_VR`), leaves the reader's three facts at the level columns `lc` (`RdAllAt`, `LevelTopAt`, `CapHitAt`) as
the table bound's premises (`plainR_tableBoundAbortLin`); with conflicts tied to collisions (`plainR_conflictTied`)
they give `KSFlockLin` at the trimmed record (`plainR_KSFlockLin_of_reader`).
-/

namespace FlockDeploy.KS

open Flock

local notation "𝔽" => FlockLevel3.GF128

/-- **The table bound at the plain record at `VR rd`**, at SHA-512, linearly at `CpDep`, from the reader's bound, top
and cap facts at `lc` (`tableBoundAbortLin_VR`, `accLast_VR`, `decideFree_VR`). -/
theorem plainR_tableBoundAbortLin (rel : Readable.Parsed → Readable.Relation) (rd : Claims.Msg → Option FinalP)
    (lc : Claims.Msg → Claims.Seen → ℕ → ℕ → List (ℕ × List 𝔽)) (hall : RdAllAt (deployPlainR rel rd) lc)
    (htopL : LevelTopAt (VR rd) lc) (hcap : CapHitAt (VR rd) lc) :
    Claims.Recursion.TableBoundAbortLin sha512 (deployPlainR rel rd) (wordLen false) (wide false) CpDep :=
  tableBoundAbortLin_VR rd (deployPlainR rel rd) rfl (fun x => εtable_nonneg false x) lc (accLast_VR rd)
    (decideFree_VR rd) hall htopL hcap

/-- **`KSFlockLin` at the trimmed plain record at `VR rd`**, at SHA-512 and `CpDep`, from the reader's three facts at
`lc` (`plainR_tableBoundAbortLin`, `plainR_KSFlockLin_of_table`). -/
theorem plainR_KSFlockLin_of_reader (rel : Readable.Parsed → Readable.Relation) (rd : Claims.Msg → Option FinalP)
    (lc : Claims.Msg → Claims.Seen → ℕ → ℕ → List (ℕ × List 𝔽)) (hall : RdAllAt (deployPlainR rel rd) lc)
    (htopL : LevelTopAt (VR rd) lc) (hcap : CapHitAt (VR rd) lc) :
    Claims.Recursion.KSFlockLin sha512 (deployPlainTrimR rel rd) CpDep :=
  plainR_KSFlockLin_of_table rel rd (plainR_tableBoundAbortLin rel rd lc hall htopL hcap)

end FlockDeploy.KS
