import Proofs.Properties.CertifierDevice.Issuer

/-!
# The channel's signing step, proved: its certificates verify, and the link's is the issuer's

`Issuer.signOn` signs its channel's message (`channelMessage`) at the reserved leaf, and `ChannelCertificate.verify`
recomputes that message from the certificate's fields, decodes the signature (decode∘encode) and verifies it under the
key's public key (`Lms.verify_signAt`), so it accepts (`verify_signOn`), on a channel of any kind. On a link the channel's
message is the network certificate's (`channelMessage_link`), so `Issuer.certify`'s certificate is `signOn`'s on the
record's link (`certify_signOn`), and `CertifiedRecord.verify` is that certificate's `ChannelCertificate.verify` with the
head's link-window and its digest under the salt (`verify_eq`).
-/

namespace Properties.CertifierDevice.Proofs.Channel

open _root_.CertifierDevice
open Properties.CertifierDevice.Proofs.Lms
open Properties.CertifierDevice.Proofs.Issuer (except_bind_ok prepare_spec)

/-- A link's channel message is the network certificate's message. -/
theorem channelMessage_link (link : String) (direction : NetworkCheck.Direction) (keyId session : ByteArray)
    (window : Nat) (digest : ByteArray) :
    channelMessage (.link link direction) keyId session window digest =
      NetworkCheck.certificateMessage keyId session link direction window digest := by
  unfold channelMessage
  by_cases h : digest.size = NetworkCheck.digestBytes
  · simp only [h, ne_eq, not_true_eq_false, ↓reduceIte]
    rfl
  · simp only [h, ne_eq, not_false_eq_true, ↓reduceIte, NetworkCheck.certificateMessage, NetworkCheck.messageInput]
    rfl

/-- The certificate `signOn` signs at a leaf below `2^H`, on a channel whose message is `message`, verifies under the
key's public key. -/
theorem verify_signOn {i : Issuer} (hk : ValidKey i.key) {ch : Channel} {window : Nat} {digest message : ByteArray}
    (hm : channelMessage ch i.key.I i.session window digest = .ok message) {q : Nat} (hq : q < 2 ^ i.key.params.H) :
    ∃ sig, Lms.signAt i.key q message = .ok sig ∧
      ChannelCertificate.verify ⟨i.key.I, ch, window, digest, sig.encode⟩ i.key.params i.key.publicKey i.session =
        true := by
  obtain ⟨N, W, H, toy, I, seed, hp, hkey⟩ := hk
  obtain ⟨-, hH, -, hH30⟩ := make_spec hp
  subst hH
  obtain ⟨-, hI, -, hIs, -⟩ := key_spec hkey
  obtain ⟨sig, hs, hv, hsq, hy, hpath, hC, hys, hps⟩ := verify_signAt message hp hkey hq
  have : 2 ^ i.key.params.H ≤ 2 ^ 30 := Nat.pow_le_pow_right (by omega) hH30
  have hdec := decode_encode i.key.params sig (by omega) hC hy hpath hys hps
  have hpub : i.key.publicKey.extract 0 16 = i.key.I := ByteArray.extract_append_eq_left (hI ▸ hIs).symm
  refine ⟨sig, hs, ?_⟩
  unfold ChannelCertificate.verify
  dsimp only
  rw [hm, hdec]
  dsimp only
  rw [hv, hpub]
  simp [beq_self, Except.toOption]

/-- What `signOn` returns when it signs: the channel's message, and its signature at a leaf below `2^H`. -/
theorem signOn_spec {i : Issuer} {ch : Channel} {leaf : Except String Nat} {window : Nat} {digest : ByteArray}
    {c : ChannelCertificate} (h : i.signOn ch leaf window digest = .ok (some c)) :
    ∃ message q sig, channelMessage ch i.key.I i.session window digest = .ok message ∧ leaf = .ok q ∧
      q < 2 ^ i.key.params.H ∧ Lms.signAt i.key q message = .ok sig ∧
      c = ⟨i.key.I, ch, window, digest, sig.encode⟩ := by
  unfold Issuer.signOn at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i message hm
    cases leaf with
    | error e => simp [pure, Except.pure] at h
    | ok q =>
      simp only at h
      split at h
      · cases h
      · rename_i sig hs
        simp only [pure, Except.pure, Except.ok.injEq, Option.some.injEq] at h
        have hq : q < 2 ^ i.key.params.H := by
          unfold Lms.signAt at hs
          by_contra hq
          simp [show q ≥ 2 ^ i.key.params.H by omega] at hs
          cases hs
        exact ⟨message, q, sig, hm, rfl, hq, hs, h.symm⟩

/-- Every certificate `signOn` returns with a valid key verifies, on a channel of any kind. -/
theorem signOn_verifies {i : Issuer} (hk : ValidKey i.key) {ch : Channel} {leaf : Except String Nat} {window : Nat}
    {digest : ByteArray} {c : ChannelCertificate} (h : i.signOn ch leaf window digest = .ok (some c)) :
    c.verify i.key.params i.key.publicKey i.session = true := by
  obtain ⟨message, q, sig, hm, -, hq, hs, rfl⟩ := signOn_spec h
  obtain ⟨sig', hs', hv⟩ := verify_signOn hk hm hq
  rw [hs] at hs'
  cases hs'
  exact hv

theorem channelVerifies_proof : ChannelVerifies := by
  intro N W H toy p I seed k session refused ch leaf window digest c hp hk h
  obtain ⟨hpk, -⟩ := Lms.key_spec hk
  have hv : ValidKey k := ⟨N, W, H, toy, I, seed, by rw [hpk]; exact hp, by rw [hpk]; exact hk⟩
  have := signOn_verifies (i := { key := k, session, refused }) hv h
  rw [hpk] at this
  exact this

/-- **The link's certificate is the channel step's.** The certificate `Issuer.certify` returns is the one `signOn` returns
on the record's link, for its window and the digest it signs. -/
theorem certify_signOn {i i' : Issuer} {salt : ByteArray} {leaf : Except String Nat} {record : Record}
    {direction : NetworkCheck.Direction} {params : Params} {parameters : ByteArray} {deadline : Bool}
    {cr : CertifiedRecord} (h : i.certify salt leaf record direction params parameters deadline = .ok (i', some cr)) :
    i.signOn (.link record.link direction) leaf record.window cr.certificate.digest =
      .ok (some cr.certificate.onChannel) := by
  unfold Issuer.certify at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i prepared hprep
    obtain ⟨head, digest, message⟩ := prepared
    obtain ⟨-, -, -, -, -, hm⟩ := prepare_spec hprep
    unfold Issuer.finish at h
    cases leaf with
    | error e => simp [pure, Except.pure] at h
    | ok q =>
      simp only [bind, Except.bind] at h
      split at h
      · cases h
      · rename_i sig hs
        simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq, Option.some.injEq] at h
        obtain ⟨-, rfl⟩ := h
        unfold Issuer.signOn
        rw [channelMessage_link, hm]
        simp only [bind, Except.bind, hs, pure, Except.pure, Certificate.onChannel]

/-- **The link's record check is the channel's certificate check with the record's opening.** `CertifiedRecord.verify`
is the head's opening (the head has a digest, the certificate names the head's link-window and signs that digest under
the salt) and the certificate's check on its link (`ChannelCertificate.verify`). -/
theorem verify_eq (head : NetworkCheck.RecordHead) (salt : ByteArray) (x : Certificate) (params : Lms.Params)
    (publicKey session : ByteArray) :
    CertifiedRecord.verify ⟨head, salt, x⟩ params publicKey session =
      ((match head.digest with
        | .ok d => x.link == head.link && x.direction == head.direction && (x.window : Int) == head.window &&
            x.digest == NetworkCheck.certificateDigest salt d
        | .error _ => false) && x.onChannel.verify params publicKey session) := by
  unfold CertifiedRecord.verify ChannelCertificate.verify Certificate.onChannel
  dsimp only
  rw [channelMessage_link]
  cases head.digest <;>
    cases NetworkCheck.certificateMessage x.keyId session x.link x.direction (x.window : Int) x.digest <;>
    cases Lms.Signature.decode params x.signature
  all_goals simp only [Bool.false_and, Bool.and_false]
  ac_rfl

/-- `IssuerVerifies` from the channel's: `certify`'s certificate is `signOn`'s on the record's link, which verifies
(`signOn_verifies`), and its head and salt open the digest it signs (`prepare_spec`). -/
theorem issuerVerifies_of_channel {i i' : Issuer} {salt : ByteArray} {leaf : Except String Nat} {record : Record}
    {direction : NetworkCheck.Direction} {params : Params} {parameters : ByteArray} {deadline : Bool}
    {cr : CertifiedRecord} (hk : ValidKey i.key)
    (h : i.certify salt leaf record direction params parameters deadline = .ok (i', some cr)) :
    cr.verify i.key.params i.key.publicKey i.session = true := by
  have hc := signOn_verifies hk (certify_signOn h)
  unfold Issuer.certify at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i prepared hprep
    obtain ⟨head, digest, message⟩ := prepared
    obtain ⟨-, hl, hd, hw, ⟨R, hR, hdg⟩, -⟩ := prepare_spec hprep
    unfold Issuer.finish at h
    cases leaf with
    | error e => simp [pure, Except.pure] at h
    | ok q =>
      simp only [bind, Except.bind] at h
      split at h
      · cases h
      · rename_i sig hs
        simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq, Option.some.injEq] at h
        obtain ⟨-, rfl⟩ := h
        rw [verify_eq, hR]
        dsimp only at hc ⊢
        rw [hc, hl, hd, hw, ← hdg]
        simp [beq_self]

/-- `IssuerVerifies`, the link's, as `ChannelVerifies`' instance on the record's link (`issuerVerifies_of_channel`). -/
theorem issuerVerifies_proof : IssuerVerifies := by
  intro N W H toy p I seed k session refused i' salt leaf record direction params parameters deadline cr hp hk h
  obtain ⟨hpk, -⟩ := Lms.key_spec hk
  have hv : ValidKey k := ⟨N, W, H, toy, I, seed, by rw [hpk]; exact hp, by rw [hpk]; exact hk⟩
  have := issuerVerifies_of_channel (i := { key := k, session, refused }) hv h
  rw [hpk] at this
  exact this

end Properties.CertifierDevice.Proofs.Channel
