import Properties.NetworkCertifier.LinkWindow

/-!
# N2's link-window verdict: what an acceptance establishes

The proof of `Properties.NetworkCertifier.LinkWindowAcceptMeans`: `verdict`'s accept bit is `acceptedOf` of its four
checks (`verdict_shape`), and each check that holds is its predicate (`proved_spec`, `opened_spec`, `certified_spec`,
`synced_spec`).
-/

open NetworkCheck NetworkCheck.LinkWindow

namespace Properties.NetworkCertifier

/-- `ByteArray`'s `==` is its arrays', which is lawful. -/
local instance instLawfulBEqByteArray_linkWindow : LawfulBEq ByteArray where
  eq_of_beq h := ByteArray.ext (beq_iff_eq.1 h)
  rfl {a} := show (a.data == a.data) = true from beq_iff_eq.2 rfl

theorem proved_spec {c : Case} (h : proved c = true) : WindowProved c := by
  unfold proved at h
  split at h
  · cases h
  · rename_i fs hf
    simp only [Bool.and_eq_true, Array.all_eq_true', Array.isEmpty_eq_false_iff, Bool.not_eq_true', beq_iff_eq] at h
    exact ⟨h.1.1, fs, hf, h.1.2, fun f hm => ⟨(h.2 f hm).1.1, (h.2 f hm).1.2, (h.2 f hm).2⟩⟩

/-- What `verdict` returns: the public row computed, its accept bit `acceptedOf` its checks, its outputs of that bit and
the opened word, and its digest of them. -/
theorem verdict_shape {c : Case} {v : Verdict} (hv : verdict c = .ok v) :
    ∃ row, pubRow c = .ok row ∧
      v.accepted = acceptedOf (proved c) (if proved c then opened c row else none) (certified c) (synced c) ∧
      v.outputs = outputsOf c v.accepted ((if proved c then opened c row else none).getD 0) ∧
      verdictDigest c v.accepted v.outputs = .ok v.digest := by
  unfold verdict at hv
  cases hr : pubRow c with
  | error e => simp [hr, bind, Except.bind] at hv
  | ok row =>
    simp only [hr, bind, Except.bind, pure, Except.pure] at hv
    split at hv
    · cases hv
    · rename_i d hd
      cases hv
      exact ⟨row, rfl, rfl, rfl, hd⟩

theorem opens_spec {row : ByteArray} {bits : Nat} {salt : Option ByteArray} {bc : ByteArray}
    (h : opens row bits salt bc = true) : CommitsRow row bits salt bc := by
  unfold opens at h
  cases salt with
  | none => cases h
  | some s =>
    simp only [Bool.and_eq_true, beq_iff_eq] at h
    exact ⟨s, rfl, h.1, h.2⟩

theorem opened_spec {c : Case} {row : Option ByteArray} {w : Nat} (h : opened c row = some w) :
    c.output = some w ∧ ∃ r file u bcPub bcWit bcOut, row = some r ∧ c.statement = some file ∧
      c.units[0]? = some u ∧ commitments file u = some [bcPub, bcWit, bcOut] ∧
      CommitsRow r (8 * r.size) c.saltPub bcPub ∧ CommitsRow (le 2 w) outBits c.saltOut bcOut := by
  simp only [opened, bind, Option.bind_eq_some_iff] at h
  obtain ⟨r, hr, file, hf, u, hu, bcs, hb, word, ho, h⟩ := h
  split at h
  · simp at h
  · split at h
    · rename_i bcPub bcWit bcOut
      split at h
      · rename_i hop
        cases h
        simp only [Bool.and_eq_true] at hop
        exact ⟨ho, r, file, u, bcPub, bcWit, bcOut, hr, hf, hu, hb, opens_spec hop.1, opens_spec hop.2⟩
      · cases h
    · cases h

theorem verifies_spec {c : Case} {x : Cert} (h : verifies c x = true) :
    x.keyId = c.key.extract 0 16 ∧ ∃ m sig,
      certificateMessage x.keyId c.session x.link x.direction x.window x.digest = .ok m ∧
      Lms.Signature.decode c.lms x.signature = .ok sig ∧ Lms.verify c.lms c.key m sig = .ok true := by
  unfold verifies at h
  simp only [Bool.and_eq_true, beq_iff_eq] at h
  refine ⟨h.1, ?_⟩
  obtain ⟨-, h⟩ := h
  split at h
  · rename_i m sig hm hs
    refine ⟨m, sig, hm, hs, ?_⟩
    cases hv : Lms.verify c.lms c.key m sig with
    | error e => rw [hv] at h; cases h
    | ok b => rw [hv] at h; cases b <;> first | rfl | cases h
  · cases h

theorem certified_spec {c : Case} (h : certified c = true) : WindowCertified c := by
  unfold certified at h
  simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range] at h
  obtain ⟨hs, hall⟩ := h
  refine ⟨hs, fun d hd => ?_⟩
  have hd' : d < c.certificates.size := hs ▸ hd
  have hx := hall d hd
  rw [getElem!_pos c.certificates d hd'] at hx
  split at hx
  · rename_i x hcx
    simp only [Bool.and_eq_true, beq_iff_eq] at hx
    obtain ⟨⟨⟨hl, hdir⟩, hw⟩, hv⟩ := hx
    exact ⟨x, by rw [Array.getElem?_eq_getElem hd', hcx], hl, hdir, hw, verifies_spec hv⟩
  · cases hx

theorem synced_spec {c : Case} (h : synced c = true) : WindowSynced c := by
  unfold synced at h
  split at h
  · rename_i s hs
    simp only [Bool.and_eq_true, Array.contains_iff_mem, Array.any_eq_true', beq_iff_eq] at h
    obtain ⟨hd, r, hr, he⟩ := h
    exact ⟨s, hs, hd, r, hr, he⟩
  · cases h

theorem linkWindowAcceptMeans : LinkWindowAcceptMeans := by
  intro c h
  cases hv : verdict c with
  | error e => rw [hv] at h; cases h
  | ok v =>
    rw [hv] at h
    have hacc : v.accepted = true := by injection h
    obtain ⟨row, hr, ha, hout, hdig⟩ := verdict_shape hv
    rw [hacc] at ha hout hdig
    have ha := ha.symm
    simp only [acceptedOf, Bool.and_eq_true, beq_iff_eq] at ha
    obtain ⟨⟨⟨hp, ho⟩, hc⟩, hs⟩ := ha
    simp only [hp, ↓reduceIte] at ho
    obtain ⟨hw, r, file, u, bcPub, bcWit, bcOut, hrow, hf, hu, hb, h1, h2⟩ := opened_spec ho
    subst hrow
    refine ⟨⟨proved_spec hp, ⟨hw, r, file, u, bcPub, bcWit, bcOut, hr, hf, hu, hb, h1, h2⟩, certified_spec hc,
      synced_spec hs⟩, v, hv, ?_, hdig⟩
    simp only [hout, hp, ↓reduceIte, ho, Option.getD_some]
    rfl

end Properties.NetworkCertifier
