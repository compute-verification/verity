module

public import Proofs.Flock.Claims.Recursion.CallsOfProofs

/-! (P4) at the deployed bound record, at SHA-512 with no premise, since V*'s proof-file decoder is canonical. -/

@[expose] public section

namespace RecDeploy

open Claims (Msg)

theorem canonicalProofFiles : CanonicalProofFiles := fun _ _ _ _ => _root_.Flock.decodeProof_canonical

theorem lastBindsEC_deployed_sha512_closed
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    (bdDeployed rel (FlockDeploy.V false).next (FlockDeploy.deployPlainTrim rel).cols wE join rowInDeployed final
      w).LastBindsEC FlockDeploy.sha512 :=
  lastBindsEC_deployed_sha512_of_canonical rel wE join final w canonicalProofFiles

end RecDeploy
