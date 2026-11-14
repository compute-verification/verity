module

public import Proofs.Properties.NetworkCertifier.Device.Lms
public import Proofs.Properties.Accounting.Remote.Schedule
public import Spot.ChallengerSession

/-!
# The auditor's key signs at each leaf

The challenger's reader takes a signature as bytes: it decodes them (`Lms.Signature.decode`), verifies the signature
under the key (`verifies`) and reads the leaf it was made at from its first four bytes (`Spot.Challenger.leafOf`). For a
key `Lms.key` derives at the deployed parameters, LMS's correctness
(`Properties.CertifierDevice.Proofs.Lms.verify_signAt`, `lmsCorrect`) gives, for every message and every leaf, bytes
that decode to a signature that verifies, of the signature's size, whose first four bytes name the leaf (`signs`).
-/

@[expose] public section

set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

open private verifies from Spot.ChallengerSession

namespace Properties.Challenger.Signer

open _root_.Spot _root_.Spot.Challenger

/-- **A signature's leaf is its first four bytes**, read as `Lms.Signature.decode` reads them. -/
theorem leafOf_eq (data : ByteArray) (h : 4 ≤ data.size) :
    leafOf data = (List.range 4).foldl (fun acc i => acc * 256 + (data.get! i).toNat) 0 := by
  obtain ⟨⟨l⟩⟩ := data
  match l, h with
  | a :: b :: c :: d :: rest, _ =>
    unfold leafOf
    rw [Properties.RemoteAudit.bytes_toList, ByteArray.data_extract, Array.toList_extract,
      List.extract_eq_take_drop]
    simp [ByteArray.get!, List.range_succ]

/-- **What decodes as a signature is of the signature's size and names its leaf.** -/
theorem leafOf_of_decode {p : Lms.Params} {data : ByteArray} {sig : Lms.Signature}
    (h : Lms.Signature.decode p data = .ok sig) : data.size = p.signatureBytes ∧ leafOf data = sig.q := by
  unfold Lms.Signature.decode at h
  split at h
  · cases h
  · rename_i hs
    have hsz : data.size = p.signatureBytes := by simpa using hs
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact ⟨hsz, leafOf_eq data (by rw [hsz, Lms.Params.signatureBytes]; omega)⟩

/-- **The key signs at each leaf**: for a key `Lms.key` derives at the deployed parameters, every message and every
leaf below `2^20`, a signature that verifies under the key, whose bytes decode to it, are of the signature's size and
name the leaf. -/
theorem signs {I seed : ByteArray} {k : Lms.PrivateKey} (hk : Lms.key Lms.deployed I seed = .ok k) (m : ByteArray)
    {q : Nat} (hq : q < 2 ^ 20) :
    ∃ s : Lms.Signature, Lms.Signature.decode Lms.deployed s.encode = .ok s ∧
      Lms.verify Lms.deployed k.publicKey m s = .ok true ∧ verifies ⟨Lms.deployed, k.publicKey⟩ m s.encode = true ∧
      leafOf s.encode = q ∧ s.encode.size = Lms.deployed.signatureBytes := by
  have hp : Lms.Params.make 64 1 20 false = .ok Lms.deployed := rfl
  obtain ⟨s, hs, hv, hsq, -⟩ := Properties.CertifierDevice.Proofs.Lms.verify_signAt m hp hk hq
  obtain ⟨-, hd⟩ :=
    Properties.CertifierDevice.Proofs.Lms.lmsCorrect 64 1 20 false Lms.deployed I seed k q m s hp hk hs
  obtain ⟨hsz, hl⟩ := leafOf_of_decode hd
  refine ⟨s, hd, hv, ?_, hl.trans hsq, hsz⟩
  unfold verifies
  simp only [hd, hv, Except.toOption, beq_self_eq_true]

end Properties.Challenger.Signer
