module

public import Definitions.Core.ProofSystem

/-! A zero-knowledge compiler: an argument system and a hiding commitment give a ZK argument for the same relation. -/

@[expose] public section

namespace Verity

open Claims
open scoped NNReal

/-- **A zero-knowledge compiler** (stated; nothing instantiates it). `compile` takes any knowledge-sound argument
system `S` and a commitment `c`, and gives a ZK argument for `S`'s relation that is knowledge sound (its knowledge error
`S`'s plus the commitment's binding term, as its `κ` and `terms` state), zero knowledge (`εzk t δ₁`, the hiding `δ₁`
summed over what a run opens), and whose commitments hide up to `δ₁` wherever `c` does (`hides`). Flock's VEIL-style
masking under a hiding commitment, in Rust (`verity_flock`) and in Lean (`Flock.ZkServer`, `Claims.Flock.zkSystem`),
is to be its instance. -/
def ZkCompiler {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (Com : Type)
    (hides : (Msg → Msg) → Com → ℝ≥0 → Prop) (compile : ProofSystem Coin W → Com → ZkProofSystem Coin W) : Prop :=
  ∀ (H : Msg → Msg) (S : ProofSystem Coin W) (c : Com), S.KS H →
    (compile S c).R = S.R ∧ (compile S c).KS H ∧ (compile S c).ZK H ∧
      ∀ δ₁, hides H c δ₁ → (compile S c).Hides H δ₁

end Verity
