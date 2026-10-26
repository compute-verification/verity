import Proofs.Flock.Claims.Recursion.NextRHonest
import Proofs.Flock.Soundness.Discharge.DecodeZL.Sizes
import Proofs.Flock.VBridge.Family.Encode

/-!
# An honest rep from the completeness model's shape

`HonestRep`'s lengths and levels are what `FlockComplete`'s `repShaped` gives the honest rep's proof: `ProofShaped` at
the shape the verifier decodes it at. A shaped proof decodes from its file (`decodeProof_encodeProof`), so it has the
lengths a decoded proof has (`decodeProof_sizes`), and its schedule has at least two levels. With zero nonces and the
unabsorbed fields as V\* takes them it is an honest rep (`honestRep_of_shaped`).
-/

namespace FlockDeploy

open Flock

/-- **A shaped proof with zero nonces is an honest rep**, at `2 + e` claims, where its `params`, `initialCap` and
`finalC` are V\*'s. -/
theorem honestRep_of_shaped {c : RepCtx} {p : Flock.Proof} {e : ℕ} (hn : c.n = 2 + e)
    (hp : FlockVBridge.Family.ProofShaped (FlockSoundness.Refine.shapeOf c.st c.sch e) p)
    (hz : (∀ n ∈ p.ligerito.queryNonces, n = 0) ∧ (∀ n ∈ p.ligerito.claimNonces, n = 0) ∧
      ∀ n ∈ p.ligerito.consistencyNonces, n = 0)
    (hpa : p.params = pcsParams c.st.merkle c.sch) (hic : p.ligerito.initialCap = p.cap)
    (hfc : p.zerocheck.finalC = dot (lagrange nodesL den6Inv c.z) p.zerocheck.round1c) : HonestRep c p := by
  have hL : 1 ≤ c.sch.levels.size - 1 := hp.2.1
  exact ⟨⟨e, hn, FlockSoundness.Discharge.DecodeZL.decodeProof_sizes _ _ _
    (FlockVBridge.Family.decodeProof_encodeProof _ _ hp)⟩, by omega, hz, hpa, hic, hfc⟩

end FlockDeploy
