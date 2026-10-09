import PouwCheck.Json
import PouwCheck.Inputs
import PouwCheck.Check
import PouwCheck.Protocol
import PouwCheck.Opening
import PouwCheck.Pass

/-!
# PoUW's window check as a core protocol

The port of `verity/core/protocols/pouw/core_check.py`: JSON as it reads it (`PouwCheck.Json`), the three inputs
and their decoders (`PouwCheck.Inputs`), PoUW's rule (`PouwCheck.Check`), and the `Core.Protocol` (`PouwCheck.Protocol`),
which `tests/test_difftest.py` holds to the Python through `pouw-check serve`; and the rule on one unit's opening beside it
(`PouwCheck.Opening`), held to `core_check.opening` through `pouw-check-difftest`. Code: Lean core only, and never
`Security/`. Its properties are stated in `Security/Properties/Pouw/Check.lean`.
-/
