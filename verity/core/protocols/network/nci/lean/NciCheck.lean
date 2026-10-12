import NciCheck.Check
import NciCheck.Difftest

/-!
# The NCI policy's check

`NciCheck.check`: the verifier's check that a declared division of a declaration into isolation units meets
`Nci.Complies Γ α` (`Security/Definitions/Nci.lean`), with the input format and its mapping to the definition's circuit
in `NciCheck.Check`'s docstring, and the difftest against the Python reference (`NciCheck.Difftest`). Code: it reads the
Program with C-Flock's decoder (`Flock.Program`) and imports nothing else but Lean core. It stays experimental until
its agreement with the definition is proved.
-/
