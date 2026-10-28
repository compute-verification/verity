import Proofs.Properties.CertifierNetwork.Encode

set_option maxRecDepth 20000

/-!
# The host's assembly

At a leaf below `2^H`, `Issuer.certify` signs a complete record whose head `prepare` makes, and its certificate carries
what the check reads: the key's identifier, the record's link-window, the head's digest under the record's salt, and a
signature `Lms.verify` accepts over the certificate message (`certify_ok`, `verify_facts`). So with a leaf for each
record, a head with no status for each (so each is complete, `complete_of_head`) and 64-byte salts, `certifyAll`
issues one certificate per record, in order (`certifyAll_ok`).
-/

namespace Properties.CertifierNetwork.Proofs

open _root_.CertifierDevice
open Properties.CertifierDevice (ValidKey)
open Properties.CertifierDevice.Proofs.Issuer (verify_cert prepare_spec prepare_head)

/-- What a certificate that verifies carries: the key's identifier, and a signature over its message. -/
theorem verify_facts {head : NetworkCheck.RecordHead} {salt kid : ByteArray} {l : String} {d : NetworkCheck.Direction}
    {w : Nat} {dg σ : ByteArray} {params : Lms.Params} {pk session : ByteArray}
    (h : CertifiedRecord.verify ⟨head, salt, ⟨kid, l, d, w, dg, σ⟩⟩ params pk session = true) :
    kid = pk.extract 0 16 ∧ ∃ m, NetworkCheck.certificateMessage kid session l d w dg = .ok m ∧
      ∃ s, Lms.Signature.decode params σ = .ok s ∧ Lms.verify params pk m s = .ok true := by
  unfold CertifiedRecord.verify at h
  dsimp only at h
  split at h
  · rename_i d' m s _ hm hs
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    obtain ⟨⟨⟨⟨⟨h1, -⟩, -⟩, -⟩, -⟩, h6⟩ := h
    have h1' : kid.data = (pk.extract 0 16).data := by
      have : (kid.data == (pk.extract 0 16).data) = true := h1
      simpa using this
    refine ⟨ByteArray.ext h1', m, hm, s, hs, ?_⟩
    cases hv : Lms.verify params pk m s with
    | error e => rw [hv] at h6; cases h6
    | ok b =>
      rw [hv] at h6
      simp only [Except.toOption, Option.some.injEq] at h6
      rw [h6]
  · cases h

/-- **At a leaf below `2^H`**, `certify` signs a prepared complete record, and the certificate verifies. -/
theorem certify_ok {i : Issuer} (hk : ValidKey i.key) {salt : ByteArray} {r : Record} {d : NetworkCheck.Direction}
    {g : Params} {pd : ByteArray} {head : NetworkCheck.RecordHead} {dg m : ByteArray}
    (hprep : i.prepare salt r d g pd true = .ok (head, dg, m)) (hst : r.status = .complete) {q : Nat}
    (hq : q < 2 ^ i.key.params.H) :
    ∃ sig : Lms.Signature,
      i.certify salt (.ok q) r d g pd = .ok (i, some ⟨head, salt, ⟨i.key.I, r.link, d, r.window, dg, sig.encode⟩⟩) ∧
      CertifiedRecord.verify ⟨head, salt, ⟨i.key.I, r.link, d, r.window, dg, sig.encode⟩⟩ i.key.params
        i.key.publicKey i.session = true := by
  obtain ⟨sig, hs, hv⟩ := verify_cert hk hprep hq
  refine ⟨sig, ?_, hv⟩
  unfold Issuer.certify Issuer.finish
  rw [hprep]
  simp only [bind, Except.bind, hs, hst, Status.reason?, Option.isSome_none, Bool.false_eq_true, ↓reduceIte, pure,
    Except.pure]

/-- A record `prepare` makes the head of, where that head is `record_head`'s with no status, is complete. -/
theorem complete_of_head {i : Issuer} {salt : ByteArray} {r : Record} {d : NetworkCheck.Direction} {g : Params}
    {pd : ByteArray} {head : NetworkCheck.RecordHead} {dg m : ByteArray}
    (hh : NetworkCheck.recordHead r.link d r.window r.grid none g pd true = .ok head)
    (hprep : i.prepare salt r d g pd true = .ok (head, dg, m)) : r.status = .complete := by
  have h2 := (recordHead_fields (prepare_head hprep)).2.2.2.2.2.2.2.1
  rw [(recordHead_fields hh).2.2.2.2.2.2.2.1] at h2
  cases hs : r.status with
  | complete => rfl
  | notEstablished x => simp [hs, Status.reason?] at h2

/-- **A refused start signs nothing.** `certifyWith` hands a start of the session that the signer's state refuses (one it
has already started, `Core.World.start`) to `finish` as a refused reservation: no certificate, and the link-window is
listed in `refused`. -/
theorem finish_refused (i : Issuer) (salt : ByteArray) (e : String) (r : Record) (d : NetworkCheck.Direction)
    (prepared : NetworkCheck.RecordHead × ByteArray × ByteArray) :
    i.finish salt (.error e) r d prepared = .ok ({ i with refused := i.refused.push (r.link, d, r.window) }, none) := by
  obtain ⟨_, _, _⟩ := prepared
  rfl

/-- What the assembly issues for record `r`: a certificate of the head `record_head` makes of `r` under its grid's
parameters digest, naming `r`'s link-window, of the key's identifier, signing the head's digest under the record's salt,
and verifying under the key over its message. -/
structure Issued (c : Config) (i : Issuer) (r : Record) (cr : CertifiedRecord) : Prop where
  head : ∃ pd, parameters (gridOf c r) (syncSetOf c r) = .ok pd ∧
    NetworkCheck.recordHead r.link (directionOf c r) r.window r.grid none (gridOf c r) pd true = .ok cr.head
  link : cr.certificate.link = r.link
  dir : cr.certificate.direction = directionOf c r
  window : cr.certificate.window = r.window
  keyId : cr.certificate.keyId = i.key.publicKey.extract 0 16
  digest : ∃ dg, cr.head.digest = .ok dg ∧ cr.certificate.digest = NetworkCheck.certificateDigest cr.salt dg
  signs : ∃ m, NetworkCheck.certificateMessage (i.key.publicKey.extract 0 16) i.session r.link (directionOf c r)
      r.window cr.certificate.digest = .ok m ∧
    ∃ s, Lms.Signature.decode i.key.params cr.certificate.signature = .ok s ∧
      Lms.verify i.key.params i.key.publicKey m s = .ok true

theorem certifyAll_cons {c : Config} {i i' : Issuer} {next : Nat} {salts : Nat → ByteArray} {r : Record}
    {rs : List Record} {pd : ByteArray} {cr? : Option CertifiedRecord}
    (hpd : parameters (gridOf c r) (syncSetOf c r) = .ok pd)
    (hc : i.certify (salts 0) (reserve i next) r (directionOf c r) (gridOf c r) pd = .ok (i', cr?)) :
    certifyAll c i next salts (r :: rs) =
      (certifyAll c i' (advance i next) (fun n => salts (n + 1)) rs).map fun x => (x.1, cr?.toList ++ x.2) := by
  rw [certifyAll, hpd]
  simp only [bind, Except.bind, hc]
  cases certifyAll c i' (advance i next) (fun n => salts (n + 1)) rs <;> rfl

/-- **The assembly issues a certificate for every record**, in order, given a leaf for each, a head for each and its
salt (`prepare` succeeds under it). -/
theorem certifyAll_ok {c : Config} {i : Issuer} (hk : ValidKey i.key) :
    ∀ (rs : List Record) (next : Nat) (salts : Nat → ByteArray), next + rs.length ≤ 2 ^ i.key.params.H →
      (∀ n < rs.length, ∀ r, rs[n]? = some r → ∃ pd, parameters (gridOf c r) (syncSetOf c r) = .ok pd ∧
        ∃ head dg m,
        NetworkCheck.recordHead r.link (directionOf c r) r.window r.grid none (gridOf c r) pd true = .ok head ∧
        i.prepare (salts n) r (directionOf c r) (gridOf c r) pd true = .ok (head, dg, m)) →
      ∃ crs, certifyAll c i next salts rs = .ok (i, crs) ∧ List.Forall₂ (Issued c i) rs crs
  | [], _, _, _, _ => ⟨[], rfl, List.Forall₂.nil⟩
  | r :: rs, next, salts, hn, hp => by
    obtain ⟨pd, hpd, head, dg, m, hh, hprep⟩ := hp 0 (by simp) r rfl
    have hq : next < 2 ^ i.key.params.H := by simp only [List.length_cons] at hn; omega
    obtain ⟨sig, hc, hv⟩ := certify_ok hk hprep (complete_of_head hh hprep) hq
    have hres : reserve i next = .ok next := by simp [reserve, hq]
    have hadv : advance i next = next + 1 := by simp [advance, hq]
    obtain ⟨crs, hcrs, hall⟩ := certifyAll_ok hk rs (next + 1) (fun n => salts (n + 1))
      (by simp only [List.length_cons] at hn; omega)
      (fun n hn' r' hr' => hp (n + 1) (by simp only [List.length_cons]; omega) r' hr')
    rw [← hres] at hc
    refine ⟨⟨head, salts 0, ⟨i.key.I, r.link, directionOf c r, r.window, dg, sig.encode⟩⟩ :: crs, ?_,
      List.Forall₂.cons ?_ hall⟩
    · rw [certifyAll_cons hpd hc, hadv, hcrs]
      rfl
    · obtain ⟨-, -, -, -, hdig, -⟩ := prepare_spec hprep
      obtain ⟨hkid, m', hm', s, hs, hv'⟩ := verify_facts hv
      exact ⟨⟨pd, hpd, hh⟩, rfl, rfl, rfl, hkid, hdig, ⟨m', by rw [← hkid]; exact hm', s, hs, hv'⟩⟩

theorem forall₂_filter {α β : Type _} {R : α → β → Prop} {p : α → Bool} {q : β → Bool}
    (hpq : ∀ a b, R a b → p a = q b) : ∀ {l₁ : List α} {l₂ : List β}, List.Forall₂ R l₁ l₂ →
      List.Forall₂ R (l₁.filter p) (l₂.filter q)
  | [], [], .nil => .nil
  | a :: l₁, b :: l₂, .cons h hs => by
    rw [List.filter_cons, List.filter_cons, hpq a b h]
    split
    · exact .cons h (forall₂_filter hpq hs)
    · exact forall₂_filter hpq hs

end Properties.CertifierNetwork.Proofs
