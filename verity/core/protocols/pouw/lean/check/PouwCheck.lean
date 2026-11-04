import PouwCheck.Json
import PouwCheck.Tile
import PouwCheck.Inputs
import PouwCheck.Check
import PouwCheck.Protocol
import PouwCheck.Opening
import PouwCheck.Pass
import PouwCheck.Rounds
import PouwCheck.Credit
import PouwCheck.Chunks
import PouwCheck.LeafOwn
import PouwCheck.Verified
import PouwCheck.LiveVerified
import PouwCheck.CommittedFiles

/-!
# PoUW's window check as a core protocol

The port of `verity/core/protocols/pouw/core_check.py`: JSON as it reads it (`PouwCheck.Json`), the tile templates
and circuits the verifier pins (`PouwCheck.Tile`), the three inputs and their decoders (`PouwCheck.Inputs`), PoUW's rule (`PouwCheck.Check`), and the `Core.Protocol` (`PouwCheck.Protocol`),
which `tests/test_difftest.py` holds to the Python through `pouw-check serve`; and the rule on one unit's opening beside it
(`PouwCheck.Opening`), held to `core_check.opening` through `pouw-check-difftest`, as are the rules on the window's pass and
draw (`PouwCheck.Pass`) and on a live window's salt and round coins (`PouwCheck.Rounds`). Code: Lean core only, and never
`Security/`. Its properties are stated in `Security/Properties/Pouw/Check.lean`.

The rules on a served window's credited set, its drawn units (the draw coin's over its layout) and their outcome, its
period (counted only closed and complete) and its A rows (`PouwCheck.Credit`, whose one entry is `window` and
`periodCredit`, every rule composed on one window and its round log) are held to `credit.py` the same way, through
`pouw-check-difftest` (`PouwCheck.CreditDifftest`). A window unit's leaf statements read their values at the unit's
placement in the window's tables, from its descriptor and round log (`PouwCheck.LeafOwn`, `pouw-leaf-own-window`). An H1
lane's chunks along K, and so the chunk rows a table at K commits a row as (`PouwCheck.Chunks`, `weights.chunks`), are held
to the Python through `pouw-check-difftest` too.
-/
