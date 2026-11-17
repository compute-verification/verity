module

public import PousVerifier.Params
public import PousVerifier.Prim.Bytes
public import PousVerifier.Prim.PyJson
public import PousVerifier.P2
public import PousVerifier.Keys
public import PousVerifier.Commit
public import PousVerifier.Setup
public import PousVerifier.SetupProtocol
public import PousVerifier.SetupInProofs
public import PousVerifier.SetupDifftest
public import PousVerifier.Merkle
public import PousVerifier.Challenge
public import PousVerifier.Wire.Codec
public import PousVerifier.Wire
public import PousVerifier.Certificate
public import PousVerifier.Audit
public import PousVerifier.Difftest
public import PousVerifier.TimedAudit
public import PousVerifier.Run
public import PousVerifier.RunProtocol
public import PousVerifier.SignedRun
public import PousVerifier.Verdict
public import PousVerifier.VerdictDifftest

/-!
# The PoUS check, as executable Lean

The library of the package `pous_check`: what PoUS's verifier checks at setup and in the timed audit, and a run composed
of the setup and a memory challenger's session, reading the formats the Python reference writes.

* `Params`, the deployed P2 parameters; `Prim.Bytes` and `Prim.PyJson`, the byte encodings and Python's reading of a
  JSON text; `P2`, `Keys` and `Wire.Codec`, P2's maps and keys;
* `Commit`, the commitment to `W`; `Merkle`, the verification tree; `Challenge`, the audit's challenges;
* `Setup`, the public-encoder setup check, and `SetupProtocol`, that check as a core protocol; `SetupInProofs`, the
  setup checked on a sampled proof about the store in place of the store;
* `Wire` and `Audit`, a setup and one timed audit, verified to its `FINAL` frame; `Certificate`, the certificate gate
  and the StorageProfile; `TimedAudit`, the timed audit's verdict on its record;
* `Run`, the setup's public outputs and the memory challenger's signed session, joined, and `RunProtocol`, that check
  as a core protocol;
* `SignedRun`, the run on the memory challenger's signed setup record and one Merkle path per round in place of its
  table, as a core protocol;
* `Verdict`, the proof block whose one output is a memory challenger's session verdict, as code;
* `Difftest` and `SetupDifftest`, the functions the difftests hold to the Python, and `VerdictDifftest`, those held to
  the Definition `verdict_circuit.PousVerdict`.

Its properties are stated in `security/Properties/Pous.lean` and the modules under it.
-/

@[expose] public section
