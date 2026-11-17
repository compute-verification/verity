module

public import Properties.Pous
public import Definitions.Pous.P2.Cost
public import Mathlib.Data.ZMod.Basic
import Mathlib.Basic.ENNReal.Real

/-! `deployed`, the registered point's numbers, by reduction; `AcceptsComplete` and `DecodeIsModel` are stated here,
open, and claimed by no one. -/

@[expose] public section

namespace PousCheck.Properties

def HonestSetup (q : P2.Params) (c : SetupProtocol.Commitment) (C : Array (Array ByteArray)) : Prop :=
  q.validate = .ok q ∧ DecodesToCommitment q c C

def AcceptsComplete : Prop :=
  ∀ (q : P2.Params) (c : SetupProtocol.Commitment) (C : Array (Array ByteArray)),
    HonestSetup q c C → SetupProtocol.protocol.Accepts q c C

/-- The identity labeling `φ = id` of `Z/p` (the model's `nonemptyΩ` witness), at which `p2ring` is
`decode_segment` for given keys. -/
def idLabeling (p : ℕ) [NeZero p] : Pous.OddLabeling p :=
  ⟨(ZMod.finEquiv p).symm.toEquiv, fun x => map_neg (ZMod.finEquiv p).symm x⟩

def DecodeIsModelAt (B w p b : ℕ) [NeZero p] : Prop :=
  ∀ (keys : Fin B → Pous.Bits w × Fin p) (j : Fin B) (K c : ℕ),
    K < 2 ^ w → c < p → (keys j).1 = Pous.bitsOfNat w K →
      Pous.bitsVal ((Pous.decBlock B w p b j (Pous.bitsOfNat w c)).run
          ((Pous.ringRO B w p).answer (idLabeling p, keys)))
        = P2.decodeValue p K (keys j).2 c % 2 ^ b

def DecodeIsModel : Prop := ∀ (B w p b : ℕ) [NeZero p], p ≤ 2 ^ w → DecodeIsModelAt B w p b

end PousCheck.Properties

namespace PousCheck.Properties.SecurityProofs

set_option exponentiation.threshold 20000 in
theorem deployed : Deployed := by
  unfold Deployed
  exact ⟨rfl, rfl, rfl, by decide, by decide, by decide, rfl, rfl, by decide⟩

end PousCheck.Properties.SecurityProofs
