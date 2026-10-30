import Proofs.Properties.CertifierDevice.Lms

/-!
# The issuer, proved: its certificates verify

`Issuer.prepare` fixes the record head (its link, direction and window are the record's), the head's digest under the
salt and the certificate message; `Issuer.finish` signs the message at the reserved leaf. `CertifiedRecord.verify` recomputes the
digest and the message from the certificate's fields, which are the ones signed, decodes the signature (decode∘encode)
and verifies it under the key's public key (`Lms.verify_signAt`), so it accepts (`verify_cert`).
-/

namespace Properties.CertifierDevice

/-- An issuer's key: one `Lms.key` derives under parameters `Lms.Params.make` admits. An honest-case hypothesis of the
completeness statements (`Proofs.Issuer.IssuerSigns`, `CertifierNetwork.DeviceOutputAccepted`); the guarantees take the
two calls' acceptances instead. -/
def ValidKey (k : Lms.PrivateKey) : Prop :=
  ∃ (N W H : Nat) (toy : Bool) (I seed : ByteArray), Lms.Params.make N W H toy = .ok k.params ∧
    Lms.key k.params I seed = .ok k

end Properties.CertifierDevice

namespace Properties.CertifierDevice.Proofs.Issuer

open _root_.CertifierDevice
open Properties.CertifierDevice.Proofs.Lms

theorem recordHead_fields {link : String} {direction : NetworkCheck.Direction} {window : Int}
    {grid : Array (Array (Option ByteArray))} {status : Option String} {p : NetworkCheck.Params} {parameters : ByteArray}
    {deadline : Bool} {head : NetworkCheck.RecordHead}
    (h : NetworkCheck.recordHead link direction window grid status p parameters deadline = .ok head) :
    head.link = link ∧ head.direction = direction ∧ head.window = window := by
  unfold NetworkCheck.recordHead at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · split at h
    · cases h
    · simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      exact ⟨rfl, rfl, rfl⟩

theorem direction_beq (d : NetworkCheck.Direction) : (d == d) = true := by
  cases d <;> rfl

theorem except_bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem except_ok_bind {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a >>= f) = f a := rfl

/-- What `prepare` returns: the record's head, the head's digest under the salt, which is 64 bytes, and the message
under the issuer's identifier. -/
theorem prepare_spec {i : Issuer} {salt : ByteArray} {record : Record} {direction : NetworkCheck.Direction}
    {params : Params} {parameters : ByteArray} {deadline : Bool} {head : NetworkCheck.RecordHead}
    {digest message : ByteArray}
    (h : i.prepare salt record direction params parameters deadline = .ok (head, digest, message)) :
    salt.size = NetworkCheck.saltBytes ∧ head.link = record.link ∧ head.direction = direction ∧
      head.window = record.window ∧ (∃ R, head.digest = .ok R ∧ digest = NetworkCheck.certificateDigest salt R) ∧
      NetworkCheck.certificateMessage i.key.I i.session record.link direction record.window digest = .ok message := by
  unfold Issuer.prepare at h
  -- the signed digest stays opaque: unfolding SHA-512 of the tag's bytes overflows the kernel
  generalize hD : NetworkCheck.certificateDigest salt = D at h
  by_cases hs : salt.size = NetworkCheck.saltBytes
  · simp only [hs, ne_eq, not_true_eq_false, ↓reduceIte] at h
    obtain ⟨head', hh, h⟩ := except_bind_ok h
    obtain ⟨R, hd, h⟩ := except_bind_ok h
    obtain ⟨message', hm, h⟩ := except_bind_ok h
    have h' := Except.ok.inj h
    simp only [Prod.mk.injEq] at h'
    obtain ⟨rfl, rfl, rfl⟩ := h'
    subst hD
    exact ⟨hs, (recordHead_fields hh).1, (recordHead_fields hh).2.1, (recordHead_fields hh).2.2, ⟨R, hd, rfl⟩, hm⟩
  · simp only [hs, ne_eq, not_false_eq_true, ↓reduceIte] at h
    obtain ⟨_, h, -⟩ := except_bind_ok h
    cases h

/-- The head `prepare` returns is `recordHead` of the record. -/
theorem prepare_head {i : Issuer} {salt : ByteArray} {record : Record} {direction : NetworkCheck.Direction}
    {params : Params} {parameters : ByteArray} {deadline : Bool} {head : NetworkCheck.RecordHead}
    {digest message : ByteArray}
    (h : i.prepare salt record direction params parameters deadline = .ok (head, digest, message)) :
    NetworkCheck.recordHead record.link direction record.window record.grid
      ((Status.reason? record.status).map reasonValue) params parameters deadline = .ok head := by
  unfold Issuer.prepare at h
  generalize NetworkCheck.certificateDigest salt = D at h
  by_cases hs : salt.size = NetworkCheck.saltBytes
  · simp only [hs, ne_eq, not_true_eq_false, ↓reduceIte] at h
    obtain ⟨head', hh, h⟩ := except_bind_ok h
    obtain ⟨_, -, h⟩ := except_bind_ok h
    obtain ⟨_, -, h⟩ := except_bind_ok h
    have h' := Except.ok.inj h
    simp only [Prod.mk.injEq] at h'
    obtain ⟨rfl, -, -⟩ := h'
    exact hh
  · simp only [hs, ne_eq, not_false_eq_true, ↓reduceIte] at h
    obtain ⟨_, h, -⟩ := except_bind_ok h
    cases h

/-- The certificate `finish` builds at a leaf below `2^H` from a prepared record verifies under the key's public key. -/
theorem verify_cert {i : Issuer} (hk : ValidKey i.key) {salt : ByteArray} {record : Record}
    {direction : NetworkCheck.Direction} {params : Params} {parameters : ByteArray} {deadline : Bool}
    {head : NetworkCheck.RecordHead} {digest message : ByteArray}
    (hprep : i.prepare salt record direction params parameters deadline = .ok (head, digest, message)) {q : Nat}
    (hq : q < 2 ^ i.key.params.H) :
    ∃ sig, Lms.signAt i.key q message = .ok sig ∧
      CertifiedRecord.verify ⟨head, salt, ⟨i.key.I, record.link, direction, record.window, digest, sig.encode⟩⟩
        i.key.params i.key.publicKey i.session = true := by
  obtain ⟨N, W, H, toy, I, seed, hp, hkey⟩ := hk
  obtain ⟨-, hH, -, hH30⟩ := make_spec hp
  subst hH
  obtain ⟨-, hI, -, hIs, -⟩ := key_spec hkey
  obtain ⟨sig, hs, hv, hsq, hy, hpath, hC, hys, hps⟩ := verify_signAt message hp hkey hq
  have : 2 ^ i.key.params.H ≤ 2 ^ 30 := Nat.pow_le_pow_right (by omega) hH30
  have hdec := decode_encode i.key.params sig (by omega) hC hy hpath hys hps
  obtain ⟨-, hl, hd, hw, ⟨R, hdig, hdg⟩, hm⟩ := prepare_spec hprep
  have hpub : i.key.publicKey.extract 0 16 = i.key.I := ByteArray.extract_append_eq_left (hI ▸ hIs).symm
  refine ⟨sig, hs, ?_⟩
  unfold CertifiedRecord.verify
  dsimp only
  rw [hdig, hm, hdec]
  dsimp only
  rw [hv, hpub, hl, hd, hw, ← hdg]
  simp [beq_self, Except.toOption]

theorem issuerVerifies_valid {i i' : Issuer} {salt : ByteArray} {leaf : Except String Nat} {record : Record}
    {direction : NetworkCheck.Direction} {params : Params} {parameters : ByteArray} {deadline : Bool}
    {cr : CertifiedRecord} (hk : ValidKey i.key)
    (h : i.certify salt leaf record direction params parameters deadline = .ok (i', some cr)) :
    cr.verify i.key.params i.key.publicKey i.session = true := by
  unfold Issuer.certify at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i prepared hprep
    obtain ⟨head, digest, message⟩ := prepared
    unfold Issuer.finish at h
    cases leaf with
    | error e => simp [pure, Except.pure] at h
    | ok q =>
      simp only [bind, Except.bind] at h
      split at h
      · cases h
      · rename_i sig hs
        simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq, Option.some.injEq] at h
        obtain ⟨rfl, rfl⟩ := h
        have hq : q < 2 ^ i.key.params.H := by
          unfold Lms.signAt at hs
          by_contra hq
          simp [show q ≥ 2 ^ i.key.params.H by omega] at hs
          cases hs
        obtain ⟨sig', hs', hv⟩ := verify_cert hk hprep hq
        rw [hs] at hs'
        cases hs'
        exact hv

theorem issuerVerifies_proof : IssuerVerifies := by
  intro N W H toy p I seed k session refused i' salt leaf record direction params parameters deadline cr hp hk h
  obtain ⟨hpk, -⟩ := Lms.key_spec hk
  have hv : ValidKey k := ⟨N, W, H, toy, I, seed, by rw [hpk]; exact hp, by rw [hpk]; exact hk⟩
  have := issuerVerifies_valid (i := { key := k, session, refused }) hv h
  rw [hpk] at this
  exact this

/-- **The issuer certifies at every leaf** (completeness, not a guarantee): with a key `ValidKey`, given a leaf below
`2^H` and a record with a head and a 64-byte salt (`Issuer.prepare` succeeds), `Issuer.certify` returns a certificate,
which verifies. -/
def IssuerSigns : Prop :=
  ∀ (i : Issuer) (salt : ByteArray) (q : Nat) (record : Record) (direction : NetworkCheck.Direction) (params : Params)
    (parameters : ByteArray) (deadline : Bool) (prepared : NetworkCheck.RecordHead × ByteArray × ByteArray),
    ValidKey i.key → q < 2 ^ i.key.params.H →
    i.prepare salt record direction params parameters deadline = .ok prepared →
    ∃ cr, i.certify salt (.ok q) record direction params parameters deadline = .ok (i, some cr) ∧
      cr.verify i.key.params i.key.publicKey i.session = true

theorem issuerSigns : IssuerSigns := by
  intro i salt q record direction params parameters deadline prepared hk hq hprep
  obtain ⟨head, digest, message⟩ := prepared
  obtain ⟨sig, hs, hv⟩ := verify_cert hk hprep hq
  refine ⟨_, ?_, hv⟩
  unfold Issuer.certify Issuer.finish
  simp only [hprep, hs, bind, Except.bind, pure, Except.pure]

end Properties.CertifierDevice.Proofs.Issuer
