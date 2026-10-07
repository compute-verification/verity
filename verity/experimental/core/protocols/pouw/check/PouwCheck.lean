import PouwCheck.Json
import PouwCheck.Inputs
import PouwCheck.Check
import PouwCheck.Protocol
import PouwCheck.Difftest

/-!
# PoUW's window check as a core protocol

The port of `verity/experimental/protocols/pouw/core_check.py`: JSON as it reads it (`PouwCheck.Json`), the three inputs
and their decoders (`PouwCheck.Inputs`), PoUW's rule (`PouwCheck.Check`), the `Core.Protocol` (`PouwCheck.Protocol`) and
the difftest against the Python (`PouwCheck.Difftest`). Code: Lean core only, and never `Security/`. Its properties are
stated in `Security/Properties/Pouw/Check.lean`.
-/
