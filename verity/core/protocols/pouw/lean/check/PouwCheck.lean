module

public import PouwCheck.Json
public import PouwCheck.Tile
public import PouwCheck.Inputs
public import PouwCheck.Check
public import PouwCheck.Protocol
public import PouwCheck.Opening
public import PouwCheck.Pass
public import PouwCheck.Rounds
public import PouwCheck.Credit
public import PouwCheck.Period
public import PouwCheck.Chunks
public import PouwCheck.LeafOwn
public import PouwCheck.Lanes
public import PouwCheck.Verified
public import PouwCheck.LiveVerified
public import PouwCheck.CommittedFiles

/-!
# PoUW's execution phase check as a core protocol

The port of `verity/core/protocols/pouw/core_check.py`: JSON as it reads it (`PouwCheck.Json`), the tile templates
and circuits the verifier pins (`PouwCheck.Tile`), the three inputs and their decoders (`PouwCheck.Inputs`), PoUW's rule (`PouwCheck.Check`), and the `Core.Protocol` (`PouwCheck.Protocol`),
which `tests/test_difftest.py` holds to the Python through `pouw-check serve`; and the rule on one unit's opening beside it
(`PouwCheck.Opening`), held to `core_check.opening` through `pouw-check-difftest`, as are the rules on the execution phase's pass and
sample (`PouwCheck.Pass`) and on a live execution phase's salt and round coins (`PouwCheck.Rounds`). Code: Lean core only, and never
`Security/`. Its properties are stated in `Security/Properties/Pouw/Check.lean`.

The rules on a served execution phase's credited set, its sampled proof blocks (the sampling coin's over its layout) and their outcome, its
credit period (counted only closed and complete) and its A rows (`PouwCheck.Credit`, whose one entry is `window` and
`periodCredit`, every rule composed on one execution phase and its round log) are held to `credit.py` the same way, through
`pouw-check-difftest` (`PouwCheck.CreditDifftest`). An execution phase unit's leaf statements read their values at the unit's
placement in the execution phase's tables, from its descriptor and round log (`PouwCheck.LeafOwn`, `pouw-leaf-own-window`). An H1
lane's chunks along K, and so the chunk rows a table at K commits a row as (`PouwCheck.Chunks`, `weights.chunks`), are held
to the Python through `pouw-check-difftest` too.
-/

@[expose] public section
