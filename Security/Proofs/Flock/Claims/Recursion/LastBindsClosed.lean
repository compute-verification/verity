import Proofs.Flock.Claims.Recursion.CallsOfProofs

/-!
# (P4) at the deployed bound record, at SHA-512, with no premise

V*'s proof-file decoder accepts a plain proof's one file alone (`Flock.decodeProof` compares the file with
`Flock.Wire.proofFile`), so two files it decodes alike at one shape are one file (`Flock.decodeProof_canonical`). That is
`RecDeploy.CanonicalProofFiles` (`RecDeploy.canonicalProofFiles`). With the columns half and the calls half proved at
SHA-512 (`CallsOfProofs`), (P4) holds at the deployed bound record (`RecDeploy.lastBindsEC_deployed_sha512_closed`).
-/

namespace RecDeploy

open Claims (Msg)

/-- **V*'s proof-file decoder is canonical**: `Flock.decodeProof_canonical` at every shape. -/
theorem canonicalProofFiles : CanonicalProofFiles := fun _ _ _ _ => _root_.Flock.decodeProof_canonical

/-- **`PlainLastBinds` at SHA-512.** -/
theorem plainLastBinds_sha512 : PlainLastBinds FlockDeploy.sha512 :=
  plainLastBinds_sha512_of_canonical canonicalProofFiles

/-- **(P4) modulo what `E*` reads at the deployed bound record, at SHA-512, with no premise.** -/
theorem lastBindsEC_deployed_sha512_closed
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (wE : Msg → ℝ) (join : List Msg → Msg)
    (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) :
    (bdDeployed rel (FlockDeploy.V false).next (FlockDeploy.deployPlainTrim rel).cols wE join rowInDeployed final
      w).LastBindsEC FlockDeploy.sha512 :=
  lastBindsEC_deployed_sha512_of_canonical rel wE join final w canonicalProofFiles

end RecDeploy
