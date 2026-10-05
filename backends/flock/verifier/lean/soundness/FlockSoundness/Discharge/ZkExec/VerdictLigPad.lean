import FlockSoundness.Discharge.ZkExec.RunsLig
import FlockSoundness.NovelBasis
import FlockSoundness.Discharge.ZkLink.Compiled
import FlockSoundness.Protocol.Discharge.ZkExec.VerdictLigPad
import FlockSoundness.Discharge.ZkExec.RunsStatements

/-!
# Level 0's padded column, as `Zk.ligerito` computes it

The model's padded level-0 column (`ZkLink.padColumnO`) at an opened row is three terms: the witness lanes folded by the
lane challenges, rep `r`'s extra pair weighted by `ρ`, and the padding's codeword `Model.padCorr` at `ȳ`. The executable
computes the same three at each query `j` of level 0 (`Zk.ligerito`): the fold over the first `2^lanes` entries in
`Zk.queryLevel0`, then `ρ₁·row[2^lanes + 2r] + ρ₂·row[2^lanes + 2r + 1] + padAt(…)` in its extra loop.

* `padAt_eq`: the executable's `Zk.padAt` is `Model.padCorr` at `c₀ = PAD_C`: its `xs` are the novel-basis monomials
  `X̂_i(ω_p)` (`padFold_inv`, from `X̂_i = X̂_{i − 2^⌊log₂ i⌋}·Ŵ_{⌊log₂ i⌋}`), and `X̂_{2^cols + j} = X̂_j·Ŵ_cols` for
  `j < 2^cols`.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine FlockLevel3 Model

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## The novel basis -/

/-! ## The padding's codeword -/

/-! ## Level 0's consistency value -/

end FlockSoundness.Discharge.ZkExec
