import FlockSoundness.Refine.Tapes
import FlockSoundness.Protocol.Refine.Realizes

/-!
# How the executable's statement denotes a model statement

The model's verifier is written for a `Model.Statement`: a block R1CS with matrices `A₀`, `B₀`, a pinned column and
public regions (PROTOCOL.md §16.1–§16.2). The executable's verifier never holds the matrices. The lincheck calls the
statement's circuit fold, which computes `α·A₀ᵀe + B₀ᵀe` from the statement's sparse layout (`Stmt.fold`).

`FoldRealizes c S` says the executable's fold `c` computes the model statement `S`'s. It is a fact about the statement
builder, not a verifier check. The refinement takes it as a hypothesis, and the statement decoding discharges it for
`Setup.ofCircuit` (level 3's `fold_get` describes the fold entry by entry).
-/

namespace FlockSoundness.Refine

open Flock

local notation "𝔽" => FlockLevel3.GF128

end FlockSoundness.Refine
