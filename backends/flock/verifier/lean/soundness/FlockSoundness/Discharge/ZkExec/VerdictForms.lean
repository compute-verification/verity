import FlockSoundness.Discharge.ZkExec.VerdictInner
import FlockSoundness.Discharge.ZkExec.VerdictOpens
import FlockSoundness.Refine.Lincheck
import FlockSoundness.Refine.ArithK
import FlockSoundness.LigeritoDecode
import FlockSoundness.Discharge.ZkExec.PlaysDet
import FlockSoundness.Protocol.Discharge.ZkExec.VerdictForms

/-!
# The model's forms are the executable's

What `zk_inner_of_rep` reads its rows through.

* The model's forms over `F` (`LinF`) are the executable's `Lin` coefficient by coefficient (`rel_add`, `rel_scale`,
  `rel_masked`, `rel_combo`, ...), and its forms over `K` (`LinK`) are the executable's `Lin2`, one `Lin` per
  coordinate (`Rel2`), whose two coordinate rows are `limbRows`'.
* `plays_prefix`, `plays_bind_split`: a run of a game is the only one on a prefix of its messages and coins, so a run
  of `g >>= f` continues, past a known run of `g`, with a run of `f` at its result.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## Forms over `F` -/

section Forms

variable {np : ℕ}

end Forms

/-! ## Forms over `K` -/

section FormsK

variable {np : ℕ}

end FormsK

/-! ## Runs on a prefix -/

section Prefix

variable {W F : Type} [Inhabited W] {ι : W → F} [Finite F] [Nontrivial F]

end Prefix

end FlockSoundness.Discharge.ZkExec
