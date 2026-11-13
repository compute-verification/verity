module

public import PousCheck.Params
public import PousCheck.Prim.Bytes
public import PousCheck.Prim.PyJson
public import PousCheck.P2
public import PousCheck.Keys
public import PousCheck.Commit
public import PousCheck.Setup
public import PousCheck.SetupProtocol
public import PousCheck.SetupDifftest
public import PousCheck.Merkle
public import PousCheck.Challenge
public import PousCheck.Wire.Codec
public import PousCheck.Wire
public import PousCheck.Certificate
public import PousCheck.Audit
public import PousCheck.Difftest
public import PousCheck.TimedAudit
public import PousCheck.Run
public import PousCheck.RunProtocol
public import PousCheck.Verdict
public import PousCheck.VerdictDifftest

/-!
# The PoUS check, as executable Lean

The library of the package `pous_check`: what PoUS's verifier checks at setup and in the timed audit, and a run composed
of the setup and a memory challenger's session, reading the formats the Python reference writes.

* `Params`, the deployed P2 parameters; `Prim.Bytes` and `Prim.PyJson`, the byte encodings and Python's reading of a
  JSON text; `P2`, `Keys` and `Wire.Codec`, P2's maps and keys;
* `Commit`, the commitment to `W`; `Merkle`, the verification tree; `Challenge`, the audit's challenges;
* `Setup`, the public-encoder setup check, and `SetupProtocol`, that check as a core protocol;
* `Wire` and `Audit`, a setup and one timed audit, verified to its `FINAL` frame; `Certificate`, the certificate gate
  and the StorageProfile; `TimedAudit`, the timed audit's verdict on its record;
* `Run`, the setup's public outputs and the memory challenger's signed session, joined, and `RunProtocol`, that check
  as a core protocol;
* `Verdict`, the proof block whose one output is a memory challenger's session verdict, as code;
* `Difftest` and `SetupDifftest`, the functions the difftests hold to the Python, and `VerdictDifftest`, those held to
  the Definition `verdict_circuit.PousVerdict`.

Its properties are stated in `security/Properties/Pous.lean` and the modules under it.
-/

@[expose] public section
