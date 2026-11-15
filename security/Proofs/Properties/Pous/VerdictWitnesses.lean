module

public import Proofs.Properties.Pous.RunAccepts
public import Proofs.Properties.Pous.WipeSigned

/-!
# Witnesses of the memory challenger's verdict properties

`verdict_honest`, the honest witness of `VerdictAccepts` and `VerdictSigned`: `verdict` accepts a session signed by a key
`Lms.keygen` derives at the deployed parameters, the verifier's one trusted key. The session is `RunAccepts`' complete
session of 105 rounds, bound to the setup over the table of zero digests, at that table's expected digests; its signature
is the key's signature at leaf 0 over the session's message (`challengePhaseMessage`). It is proved over a free setup
root, table and key (`verdict_honest_at`), so no SHA-512, signature or store is evaluated.

`verdictAccepts_nontrivial`: `Accepted` fails for that session when the verifier trusts no key.
`verdictSigned_nontrivial`: `SignedByDevice` fails when no memory challenger ran a session, for a session one of whose
rounds `Round.value` refuses (a digest that isn't 64 bytes), even with its key trusted: no run of the device is it, it has
no message for a forgery to sign, and it hashes one string, so it holds no collision (`not_signedByDevice_idle`).
-/

@[expose] public section

namespace MemoryChallenger.Properties.SecurityProofs

open MemoryChallenger Pous.EndToEnd.SecurityProofs.Honest
open MemoryChallenger.Attest.Properties.SecurityProofs (pub_spec signs)

/-- `ByteArray`'s `==` is its arrays', which is lawful. -/
local instance instLawfulBEqByteArray_verdictWitnesses : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

/-! ## The honest session -/

theorem session_honest (root : ByteArray) : (honestCommitment root).session = honestSid := rfl

/-- On any setup root `root` and table `digests` that holds row 0 whole, and for any key `Lms.key` derives at the
deployed parameters, the session of `honestRounds digests` under `honestCommitment root`, signed at leaf 0 of that key
over its message, is accepted by `verdict` with that key trusted, at `digests`' expected digests. -/
theorem verdict_honest_at (root digests : ByteArray) (hd : 64 ≤ digests.size) {I seed : ByteArray} {k : Lms.PrivateKey}
    (hk : Lms.key Lms.deployed I seed = .ok k) :
    ∃ (M : ByteArray) (σ : Lms.Signature),
      challengePhaseMessage
          ⟨honestCommitment root, COMPLETE, honestRounds digests, LMS, k.publicKey, σ.encode⟩ = some M ∧
        Lms.signAt k 0 M = .ok σ ∧
        (verdict SCHEMES [k.publicKey] (PousCheck.Run.expected PousCheck.P2.Params.deployed digests)
          ⟨honestCommitment root, COMPLETE, honestRounds digests, LMS, k.publicKey, σ.encode⟩
          (honestCommitment root) honestOpenings).accepted = true := by
  obtain ⟨hIs, hps, hpI⟩ := pub_spec hk
  obtain ⟨m, hm⟩ := message_honest root digests hd
  obtain ⟨σ, hσs, hσv⟩ := signs hk
    (Flock.Sha512.hash (field MEMORY_CHALLENGER_TAG.toUTF8 ++ I ++ field (field honestSid) ++ field m))
  have hdec := (Properties.CertifierDevice.Proofs.Lms.lmsCorrect 64 1 20 false Lms.deployed _ _ k 0 _ σ rfl hk hσs).2
  have hM : memoryChallengerMessage I honestSid m =
      .ok (Flock.Sha512.hash (field MEMORY_CHALLENGER_TAG.toUTF8 ++ I ++ field (field honestSid) ++ field m)) := by
    unfold memoryChallengerMessage
    simp only [hIs, ne_eq, not_true_eq_false, ↓reduceIte, pure, Except.pure]
  have hv : lmsVerify Lms.deployed k.publicKey honestSid m σ.encode = .ok true := by
    unfold lmsVerify
    simp only [hps, hpI, hdec, hM, ne_eq, not_true_eq_false, ↓reduceIte, bind, Except.bind]
    exact hσv
  refine ⟨_, σ, ?_, hσs, ?_⟩
  · unfold challengePhaseMessage SignedChallengePhase.message
    simp only [hm, hpI, session_honest, hM, bind, Except.bind, Except.toOption]
  · exact verdict_accepted_of (p := Lms.deployed) (m := m) (by simp [SCHEMES]) (by simp) hm hv
      (commitment_beq_self _) rfl (by simp [honestRounds, honestCommitment]) (rounds_honest root digests)

/-- **`verdict` accepts an LMS-signed session**: a key `Lms.keygen` derives at the deployed parameters signs, at leaf 0,
the message of the complete session of 105 rounds over the table of zero digests, bound to its setup's root, and
`verdict` accepts that session with the key its one trusted key, at the table's expected digests. -/
theorem verdict_honest :
    ∃ (seed : ByteArray) (k : Lms.PrivateKey) (M : ByteArray) (σ : Lms.Signature),
      Lms.keygen Lms.deployed seed = .ok k ∧
        challengePhaseMessage
            ⟨honestCommitment honestPub.root, COMPLETE, honestRounds honestDigests, LMS, k.publicKey, σ.encode⟩ =
          some M ∧
        Lms.signAt k 0 M = .ok σ ∧
        (verdict SCHEMES [k.publicKey] (PousCheck.Run.expected PousCheck.P2.Params.deployed honestDigests)
          ⟨honestCommitment honestPub.root, COMPLETE, honestRounds honestDigests, LMS, k.publicKey, σ.encode⟩
          (honestCommitment honestPub.root) honestOpenings).accepted = true := by
  obtain ⟨k, hkg, _, hk⟩ := Wipe.honest_keygen
  obtain ⟨M, σ, hM, hσ, hv⟩ :=
    verdict_honest_at honestPub.root honestDigests (size_le_of_table (zeros_size _)) hk
  exact ⟨_, k, M, σ, hkg, hM, hσ, hv⟩

/-! ## Nontrivial -/

/-- **A session is not `Accepted` when the verifier trusts no key**: the honest session, unsigned. -/
theorem verdictAccepts_nontrivial :
    ¬ Accepted SCHEMES [] (PousCheck.Run.expected PousCheck.P2.Params.deployed honestDigests)
      ⟨honestCommitment honestPub.root, COMPLETE, honestRounds honestDigests, LMS, ByteArray.empty, ByteArray.empty⟩
      (honestCommitment honestPub.root) honestOpenings := by
  rintro ⟨⟨_, _, h, _⟩, _⟩
  simp at h

theorem stream_short {r : Round} (h : r.digest.size ≠ 64) : ∃ e, recordsStream #[r] = .error e := by
  obtain ⟨e, he⟩ : ∃ e, r.value = .error e := by
    unfold Round.value
    simp only [h, ne_eq, not_false_eq_true, ↓reduceIte]
    exact ⟨_, rfl⟩
  refine ⟨e, ?_⟩
  unfold recordsStream
  rw [Array.mapM_eq_mapM_toList]
  simp only [List.mapM_cons, he, bind, Except.bind, Functor.map, Except.map]

/-- A session whose records stream `recordsStream` refuses is no session of challengers that ran none: no run is it,
`message` refuses it, so it has no message to forge, and the only string it hashes is its coin key. -/
theorem not_signedByDevice_idle {schemes : Schemes} {trusted : List ByteArray} {s : SignedChallengePhase}
    (hs : ∃ e, recordsStream s.rounds = .error e) : ¬ SignedByDevice schemes trusted s ⟨fun _ => []⟩ := by
  obtain ⟨e, he⟩ := hs
  have hd : recordsDigest s.rounds = .error e := by
    unfold recordsDigest
    simp only [he, bind, Except.bind]
  have hm : message s.commitment s.ended s.rounds = .error e := by
    unfold message
    simp only [hd, bind, Except.bind]
  have hstr : runStrings s ⟨fun _ => []⟩ = [s.commitment.coinKey] := by
    unfold runStrings challengePhaseStrings
    simp only [he, hd, hm, List.flatMap_nil, List.append_nil]
  rintro (⟨r, hr, -⟩ | ⟨x, hx, y, hy, hxy, -⟩ | ⟨p, m, -, -, hmsg, -⟩)
  · simp at hr
  · rw [hstr, List.mem_singleton] at hx hy
    exact hxy (hx.trans hy.symm)
  · unfold challengePhaseMessage SignedChallengePhase.message at hmsg
    simp only [hm, bind, Except.bind, Except.toOption, reduceCtorEq] at hmsg

/-- **A session is not `SignedByDevice` of challengers that ran none**: one round whose digest is empty, under the honest
commitment, its key trusted. -/
theorem verdictSigned_nontrivial :
    ¬ SignedByDevice SCHEMES [ByteArray.empty]
      ⟨honestCommitment honestPub.root, COMPLETE, #[⟨0, 0, 0, 0, ByteArray.empty⟩], LMS, ByteArray.empty,
        ByteArray.empty⟩
      ⟨fun _ => []⟩ :=
  not_signedByDevice_idle (stream_short (by simp))

end MemoryChallenger.Properties.SecurityProofs
