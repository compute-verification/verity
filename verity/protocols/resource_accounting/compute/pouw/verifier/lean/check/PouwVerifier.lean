module

public import PouwVerifier.Json
public import PouwVerifier.Tile
public import PouwVerifier.Inputs
public import PouwVerifier.Check
public import PouwVerifier.Protocol
public import PouwVerifier.Opening
public import PouwVerifier.Pass
public import PouwVerifier.Rounds
public import PouwVerifier.Credit
public import PouwVerifier.CreditUnwritten
public import PouwVerifier.Period
public import PouwVerifier.Chunks
public import PouwVerifier.LeafOwn
public import PouwVerifier.Lanes
public import PouwVerifier.Verified
public import PouwVerifier.LiveVerified
public import PouwVerifier.CommittedFiles

/-!
# PoUW's execution phase check as a core protocol

The port of `verity/protocols/accounting/pouw/verifier/core_check.py`: JSON as it reads it (`PouwVerifier.Json`), the tile templates
and circuits the verifier pins (`PouwVerifier.Tile`), the three inputs and their decoders (`PouwVerifier.Inputs`), PoUW's rule (`PouwVerifier.Check`), and the `Core.Protocol` (`PouwVerifier.Protocol`),
which `tests/test_difftest.py` holds to the Python through `pouw-verifier serve`; and the rule on one unit's opening beside it
(`PouwVerifier.Opening`), held to `core_check.opening` through `pouw-verifier-difftest`, as are the rules on the execution phase's pass and
sample (`PouwVerifier.Pass`) and on a live execution phase's salt and round coins (`PouwVerifier.Rounds`). Code: Lean core only, and never
`security/`. Its properties are stated in `security/Properties/Pouw/Check.lean`.

The rules on a served execution phase's credited set, its sampled proof blocks (the sampling coin's over its layout) and their outcome, its
credit period (counted only closed and complete) and its A rows (`PouwVerifier.Credit`, whose one entry is `window` and
`periodCredit`, every rule composed on one execution phase and its round log) are held to `credit.py` the same way, through
`pouw-verifier-difftest` (`PouwVerifier.CreditDifftest`). An execution phase unit's leaf statements read their values at the unit's
placement in the execution phase's tables, from its descriptor and round log (`PouwVerifier.LeafOwn`, `pouw-leaf-own-window`). An H1
lane's chunks along K, and so the chunk rows a table at K commits a row as (`PouwVerifier.Chunks`, `weights.chunks`), are held
to the Python through `pouw-verifier-difftest` too.
-/

@[expose] public section
