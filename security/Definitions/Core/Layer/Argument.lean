module

public import Definitions.Core.ProofSystem

/-!
# The argument system: the bottom layer's interface

The argument system is `Verity.ProofSystem` (`Definitions/Core/ProofSystem.lean`), with `Verity.ZkProofSystem` for one
that also hides; this module restates neither, and names them as the layer every other layer reads.

**What it takes.** A relation `R` on statements and witnesses, a verifier `V` over public coins `Coin`, and the
random oracle `H : Msg → Msg`.

**What it gives.** Knowledge soundness, `ProofSystem.KS H` (or `KSAbort H` against provers that may abort after the
last round, which a recursion's inner owes): an extractor `E H` whose failure, once the verifier accepts, is at most
`κ` plus the collision terms `terms`, with `N` expected and `M` most reruns (`Wald`). `ZkProofSystem.ZK H` adds the
simulators `Sim t` within `εzk t δ₁ + γzk t · Pr[collision]` wherever its commitments hide up to `δ₁`.

**Its error.** The bottom layer's own: `κ` and `terms` per statement, read in the trust base's SHA-512 collision
resistance through `H`.

**Instances.**
* Flock, `Claims.Flock.system` and `Claims.Flock.zkSystem` (`Proofs/Flock/Claims/System.lean`): `system_ks_iff` and
  `system_ksAbort_iff` put its knowledge soundness in this form, `system_wald` its reruns; deployed, proved.

**Duplicates, for next week.** `Claims.SamplingGeneric.Arg` and `ZkArg` (`Definitions/Claims/SamplingGeneric.lean`),
`FlockSoundness.SampledGeneric.ProofSystem` (`Proofs/Flock/SampledGeneric/Defs.lean`) and `Claims.RecGeneric.Proto`'s
`KSAbortLin` (`Definitions/Claims/RecursionGeneric.lean`) each state an argument system again.
-/
