import Pouw.TileBound.Defs
import Pouw.TileBound.Proofs
import Pouw.TileBound.Fp8
import Pouw.TileBound.Fp8Depth
import Pouw.TileBound.AlignedExact
import Pouw.TileBound.Witness
import Pouw.TileBound.WinogradStrassen

/-!
# The FP4-tile model (`docs/pouw/fp4-tile-model.md`, bc-d9842080)

`Defs`: the model over `BLACKWELL_SM120_NVF4`'s tile gate, with per-slot pricing and the rewrite class (trusted
definitions). `Proofs`: the block-support count and the cost bounds it gives.
-/
