module

public import PousCheck.SetupInProofs

/-!
# What the setup check in proofs reads, stated against the Lean code

`verity/pous/p2/v4/setup-in-proofs` (`PousCheck.SetupInProofs.protocol`) takes, in place of `C*`, the store's registered
root, the setup unit's Program digest, the sampling design and SPOT's outcome. `CheckReads` says what an acceptance
means of those inputs: valid parameters, a 24-byte salt, the sampling design `samplingDesign q`, and `SetupProtocol.Public` with the
store's root as `root`. With SPOT's acceptance of the sampled audit over the unit at `samplingDesign q` it is the hypothesis side of
`SetupGives` (`PousCheck.Properties.InProofs`), at `right` the unit's circuit and `W` the registered rows padded to whole
blocks; that composition is not stated here. It is stated, not proved, and no guarantee of the lock.
-/

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
