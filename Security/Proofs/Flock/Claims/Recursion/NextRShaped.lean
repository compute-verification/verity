import Proofs.Flock.Claims.Recursion.NextRHonest
import Proofs.Flock.Soundness.Discharge.DecodeZL.Sizes
import Proofs.Flock.VBridge.Family.Encode

/-!
# An honest rep from the completeness model's shape

`HonestRep`'s lengths and levels are what `FlockComplete`'s `repShaped` gives the honest rep's proof: `ProofShaped` at
the shape the verifier decodes it at. A shaped proof decodes from its file (`decodeProof_encodeProof`), so it has the
lengths a decoded proof has (`decodeProof_sizes`), and its schedule has at least two levels. Its nonces are zero, as
`Lengths` has them since the verifier refuses any other (R4), so with the unabsorbed fields as V\* takes them it is an
honest rep (`honestRep_of_shaped`).
-/

namespace FlockDeploy

open Flock

/-- **A shaped proof is an honest rep**, at `2 + e` claims, where its `params`, `initialCap` and `finalC` are V\*'s: its
nonces are zero (`ZkView.NoncesZero`, the last of its `Lengths`), the only ones the verifier reads (R4). -/
theorem honestRep_of_shaped {c : RepCtx} {p : Flock.Proof} {e : ℕ} (hn : c.n = 2 + e)
    (hp : FlockVBridge.Family.ProofShaped (FlockSoundness.Refine.shapeOf c.st c.sch e) p)
    (hpa : p.params = pcsParams c.st.merkle c.sch) (hic : p.ligerito.initialCap = p.cap)
    (hfc : p.zerocheck.finalC = dot (lagrange nodesL den6Inv c.z) p.zerocheck.round1c) : HonestRep c p := by
  have hL : 1 ≤ c.sch.levels.size - 1 := hp.2.1
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, hz⟩ := hp.2.2.2
  exact ⟨⟨e, hn, FlockSoundness.Discharge.DecodeZL.decodeProof_sizes _ _ _
    (FlockVBridge.Family.decodeProof_encodeProof _ _ hp)⟩, by omega, hz, hpa, hic, hfc⟩

end FlockDeploy
