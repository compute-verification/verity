module

public import PousCheck.SetupInProofs

/-! What the setup check in proofs reads (`CheckReads`), stated against the Lean code: stated, not proved, and no
guarantee of the lock. -/

@[expose] public section

namespace PousCheck.Properties.SetupInProofs

open PousCheck.SetupInProofs

/-- **An acceptance reads the claim's sampling design and publishes its store root**: for every input, if the check
accepts with `pub`, the parameters are valid, the salt is 24 bytes, the claim's sampling design is `samplingDesign q`, and `pub`
is `q.blocks`, the server's commitment, the claim's store root and the salt. -/
def CheckReads : Prop :=
  ∀ (q : P2.Params) (c : SetupProtocol.Commitment) (cl : Claim) (pub : SetupProtocol.Public),
    check q c cl = .accept pub () →
      (∃ q', q.validate = .ok q') ∧ c.salt.size = 24 ∧ cl.samplingDesign = samplingDesign q ∧
        pub = ⟨q.blocks, c.w, cl.store, c.salt⟩

end PousCheck.Properties.SetupInProofs
