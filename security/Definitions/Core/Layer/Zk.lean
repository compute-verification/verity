module

public import Definitions.Core.ProofSystem

/-!
# The ZK layer: what zero knowledge is, and that each layer keeps it

Zero knowledge is `Verity.ZkProofSystem.ZK` (`Definitions/Core/ProofSystem.lean`): each statement's masked prover is
simulated, against malicious verifiers, within `εzk t δ₁` plus `γzk t` times a collision's probability, wherever the
commitments hide up to `δ₁`.

**The compiler** (`ZkCompiler`, below). It takes an argument system and a hiding commitment, and gives a ZK argument
for the same relation, still knowledge sound.

**Its error.** The simulation error is the commitment's hiding `δ₁` summed over the commitments a run opens, as each
instance's `εzk` states; the knowledge error is the argument's plus the commitment's binding term.

**Kept through the layers above.**
* Sampling: `Claims.SamplingGeneric.CompilerZK` (`Definitions/Claims/SamplingGeneric.lean`): the sampled compiler is ZK
  when each proof block's argument is; stated.
* Recursion: `Flock.Guarantees.GenericRecursiveZK` (`Proofs/Flock/RecGeneric/ZK.lean`), proved
  (`Proofs/Flock/RecGeneric/ZKProof.lean`); its statement is a proofs-tree definition, which `trees` excepts.

**Instances of `ZkCompiler`.** None stated. Flock's masking, the VEIL-style mask of its polynomials under a hiding
commitment (Rust in `verity_flock`; Lean in `Flock.ZkServer` and `Claims.Flock.zkSystem`), is its instance, and
`Flock.zk_server_lean_loop` is that instance's ZK theorem today, read through hypotheses on the live path.

**Duplicates, for next week.** `Claims.SamplingGeneric.ZkArg.ZK` states zero knowledge again over its own `ZkArg`.
-/

@[expose] public section

namespace Verity

open Claims
open scoped NNReal

/-- **A zero-knowledge compiler** (stated; no instance proves it). `compile` takes any knowledge-sound argument system
and a commitment `c`, and gives a ZK argument for the same relation that is knowledge sound, zero knowledge, and whose
commitments hide up to `δ₁` wherever `c` does (`hides`). Flock's VEIL-style masking is to prove it. -/
def ZkCompiler {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (Com : Type)
    (hides : (Msg → Msg) → Com → ℝ≥0 → Prop) (compile : ProofSystem Coin W → Com → ZkProofSystem Coin W) : Prop :=
  ∀ (H : Msg → Msg) (S : ProofSystem Coin W) (c : Com), S.KS H →
    (compile S c).R = S.R ∧ (compile S c).KS H ∧ (compile S c).ZK H ∧
      ∀ δ₁, hides H c δ₁ → (compile S c).Hides H δ₁

end Verity
