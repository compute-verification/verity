import Properties.NetworkCertifier.LinkWindow
import Proofs.Properties.CertifierDevice.Lms
import Std.Data.TreeMap.Raw.WF

/-!
# N2's sampling block verdict: what an acceptance establishes

The proof of `Properties.NetworkCertifier.LinkWindowAcceptMeans`: `verdict`'s accept bit is `acceptedOf` of its four
checks (`verdict_shape`), and each check that holds is its predicate (`proved_spec`, `opened_spec`, `certified_spec`,
`synced_spec`).

Its two lemmas:

* honest: `linkWindowAcceptMeans_honest`, the converse, that a sampling block proved, opened, certified and synced is
  accepted with the property's outputs and digest (each predicate is its check, `proved_of` … `synced_of`, and the
  verdict's digest always computes, `verdictDigest_ok`); and `linkWindowAcceptMeans_honest_warden`, that on a warden
  link a window proved and opened whose two certificates the certifier's device signed (`CertifierSigned`, under a key
  `Lms.key` derives at parameters `Lms.Params.make` admits) is accepted whatever syncs it declares, by LMS's
  completeness (`Properties.CertifierDevice.Proofs.Lms.lmsCorrect`). What stays a hypothesis is the proof service's:
  the findings of its run of C-Flock's verifier (`WindowProved`) and the statement's public file holding the drawn
  unit's row commitments (`WindowOpened`, whose header line `commitments` reads through `Json.parse`, which the kernel
  doesn't evaluate, so no concrete file is shown to open here; the replays over recorded runs exhibit them);
* nontrivial: `linkWindowAcceptMeans_nontrivial`, that a verifier accepting every sampling block doesn't have the
  property (`LinkWindowAcceptMeansBy`, which is the property at the verdict, `linkWindowAcceptMeans_by`): it accepts a
  warden sampling block with no findings, which isn't proved.
-/

open NetworkCheck NetworkCheck.LinkWindow
open Lean (Json toJson)

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
  cases hk : c.kind with
  | warden => simp only [WindowSynced, hk]
  | plain | served | advised =>
    simp only [synced, hk] at h
    simp only [WindowSynced, hk]
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

/-! ## The honest lemma: each predicate is its check -/

theorem proved_of {c : Case} (h : WindowProved c) : proved c = true := by
  obtain ⟨hg, fs, hf, hne, hall⟩ := h
  unfold proved
  rw [hf]
  simp only [Bool.and_eq_true, Array.all_eq_true', Array.isEmpty_eq_false_iff, Bool.not_eq_true', beq_iff_eq]
  exact ⟨⟨hg, hne⟩, fun f hm => ⟨⟨(hall f hm).1, (hall f hm).2.1⟩, (hall f hm).2.2⟩⟩

theorem opens_of {row : ByteArray} {bits : Nat} {salt : Option ByteArray} {bc : ByteArray}
    (h : CommitsRow row bits salt bc) : opens row bits salt bc = true := by
  obtain ⟨s, rfl, hs, he⟩ := h
  simp only [opens, Bool.and_eq_true, beq_iff_eq]
  exact ⟨hs, he⟩

theorem opened_of {c : Case} (h : WindowOpened c) :
    ∃ row, pubRow c = .ok (some row) ∧ opened c (some row) = some 3 := by
  obtain ⟨hw, row, file, u, bcPub, bcWit, bcOut, hr, hf, hu, hb, h1, h2⟩ := h
  refine ⟨row, hr, ?_⟩
  simp [opened, hf, hu, hb, hw, opens_of h1, opens_of h2]

theorem verifies_of {c : Case} {x : Cert} {m : ByteArray} {sig : Lms.Signature} (hk : x.keyId = c.key.extract 0 16)
    (hm : certificateMessage x.keyId c.session x.link x.direction x.window x.digest = .ok m)
    (hs : Lms.Signature.decode c.lms x.signature = .ok sig) (hv : Lms.verify c.lms c.key m sig = .ok true) :
    verifies c x = true := by
  unfold verifies
  rw [hm, hs]
  simp [hk, hv, Except.toOption]

theorem certified_of {c : Case} (h : WindowCertified c) : certified c = true := by
  obtain ⟨hs, hall⟩ := h
  unfold certified
  simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range]
  refine ⟨hs, fun d hd => ?_⟩
  obtain ⟨x, hx, hl, hdir, hw, hk, m, sig, hm, hsig, hv⟩ := hall d hd
  have hd' : d < c.certificates.size := hs ▸ hd
  have hcx : c.certificates[d] = some x := by simpa [Array.getElem?_eq_getElem hd'] using hx
  rw [getElem!_pos c.certificates d hd', hcx]
  simp only [Bool.and_eq_true, beq_iff_eq]
  exact ⟨⟨⟨hl, hdir⟩, hw⟩, verifies_of hk hm hsig hv⟩

theorem synced_of {c : Case} (h : WindowSynced c) : synced c = true := by
  cases hk : c.kind with
  | warden => simp only [synced, hk]
  | plain | served | advised =>
    simp only [WindowSynced, hk] at h
    obtain ⟨s, hs, hd, r, hr, he⟩ := h
    simp only [synced, hk, hs, Bool.and_eq_true, Array.contains_iff_mem, Array.any_eq_true', beq_iff_eq]
    exact ⟨hd, r, hr, he⟩

/-! ## The verdict's digest always computes

`Flock.canon` refuses only a number that isn't an integer, and the verdict's digest body holds strings, Booleans and
naturals. -/

theorem listMapM_ok {α β ε : Type} {f : α → Except ε β} :
    ∀ l : List α, (∀ x ∈ l, ∃ y, f x = .ok y) → ∃ ys, l.mapM f = .ok ys
  | [], _ => ⟨[], rfl⟩
  | x :: l, h => by
    obtain ⟨y, hy⟩ := h x (by simp)
    obtain ⟨ys, hys⟩ := listMapM_ok l fun z hz => h z (by simp [hz])
    refine ⟨y :: ys, ?_⟩
    rw [List.mapM_cons, hy, hys]
    rfl

theorem arrayMapM_ok {α β ε : Type} {f : α → Except ε β} {a : Array α} (h : ∀ x ∈ a, ∃ y, f x = .ok y) :
    ∃ ys, a.mapM f = .ok ys := by
  obtain ⟨ys, hys⟩ := listMapM_ok a.toList fun x hx => h x (Array.mem_toList_iff.1 hx)
  exact ⟨ys.toArray, by rw [Array.mapM_eq_mapM_toList, hys]; rfl⟩

theorem canon_str (s : String) : ∃ t, Flock.canon (.str s) = .ok t := by
  simp only [Flock.canon]
  exact ⟨_, rfl⟩

theorem canon_bool (b : Bool) : ∃ t, Flock.canon (.bool b) = .ok t := by
  simp only [Flock.canon]
  exact ⟨_, rfl⟩

theorem canon_nat (n : Nat) : ∃ t, Flock.canon (toJson n) = .ok t := by
  simp only [toJson, Flock.canon]
  exact ⟨_, rfl⟩

theorem canon_arr {a : Array Json} (h : ∀ x ∈ a, ∃ t, Flock.canon x = .ok t) : ∃ t, Flock.canon (.arr a) = .ok t := by
  obtain ⟨ps, hps⟩ := arrayMapM_ok (f := fun x : {x // x ∈ a} => Flock.canon x.1) (a := a.attach)
    fun x _ => h x.1 x.2
  simp only [Flock.canon]
  rw [hps]
  exact ⟨_, rfl⟩

/-- Every pair of an object `Json.mkObj` builds holds a value of its list. -/
theorem mkObj_value {l : List (String × Json)} {k : String} {v : Json}
    (h : (k, v) ∈ (Std.TreeMap.Raw.ofList l compare).toArray) : ∃ k', (k', v) ∈ l := by
  rw [Std.TreeMap.Raw.mem_toArray_iff_getElem?_eq_some Std.TreeMap.Raw.WF.ofList,
    Std.TreeMap.Raw.ofList_eq_insertMany_empty, Std.TreeMap.Raw.getElem?_insertMany_list Std.TreeMap.Raw.WF.emptyc,
    Std.TreeMap.Raw.getElem?_emptyc, Option.or_none, List.findSomeRev?_eq_findSome?_reverse] at h
  obtain ⟨⟨k', v'⟩, hm, he⟩ := List.exists_of_findSome?_eq_some h
  by_cases hc : compare k' k = .eq
  · simp only [hc, ↓reduceIte, Option.some.injEq] at he
    exact ⟨k', he ▸ List.mem_reverse.1 hm⟩
  · simp [hc] at he

theorem canon_mkObj {l : List (String × Json)} (h : ∀ p ∈ l, ∃ t, Flock.canon p.2 = .ok t) :
    ∃ t, Flock.canon (Json.mkObj l) = .ok t := by
  obtain ⟨ps, hps⟩ := arrayMapM_ok
    (f := fun p : {x // x ∈ (Std.TreeMap.Raw.ofList l compare).toArray} =>
      (do pure (p.1.1, ← Flock.canon p.1.2) : Flock.V (String × String)))
    (a := (Std.TreeMap.Raw.ofList l compare).toArray.attach) fun ⟨⟨k, v⟩, hp⟩ _ => by
      obtain ⟨k', hk'⟩ := mkObj_value hp
      obtain ⟨t, ht⟩ := h _ hk'
      exact ⟨(k, t), by simp only [ht]; rfl⟩
  simp only [Json.mkObj, Flock.canon]
  rw [hps]
  exact ⟨_, rfl⟩

theorem acceptedOutputs_canon (c : Case) : ∃ t, Flock.canon (acceptedOutputs c) = .ok t := by
  unfold acceptedOutputs
  apply canon_mkObj
  intro p hp
  cases hk : c.kind with
  | advised view advice =>
    simp only [hk, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with (rfl | rfl) | rfl
    · exact canon_nat 1
    · exact canon_nat 1
    · cases advice with
      | some a => exact canon_str a
      | none => simp only [Flock.canon]; exact ⟨_, rfl⟩
  | plain | served | warden =>
    simp only [hk, List.append_nil, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl
    · exact canon_nat 1
    · exact canon_nat 1

theorem verdictDigest_ok (c : Case) : ∃ d, verdictDigest c true (acceptedOutputs c) = .ok d := by
  obtain ⟨t, ht⟩ : ∃ t, Flock.canon (Json.mkObj [("format", .str verdictFormat), ("registration", .str c.registration),
      ("sessions", .arr (c.sessions.map Json.str)), ("accepted", .bool true), ("public", acceptedOutputs c)]) =
      .ok t := by
    apply canon_mkObj
    intro p hp
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with rfl | rfl | rfl | rfl | rfl
    · exact canon_str _
    · exact canon_str _
    · exact canon_arr fun x hx => by
        obtain ⟨s, -, rfl⟩ := Array.mem_map.1 hx
        exact canon_str s
    · exact canon_bool true
    · exact acceptedOutputs_canon c
  unfold verdictDigest
  simp only [ht, bind, Except.bind, pure, Except.pure]
  exact ⟨_, rfl⟩

/-- **A link-window proved, opened, certified and synced is accepted** (the honest lemma of `LinkWindowAcceptMeans`,
its converse): the verdict accepts it, with the outputs and digest the property states. -/
theorem linkWindowAcceptMeans_honest :
    ∀ c : Case, WindowAccepted c → (verdict c).map Verdict.accepted = .ok true ∧ WindowVerdict c := by
  intro c ⟨hp, ho, hc, hs⟩
  obtain ⟨row, hr, hrow⟩ := opened_of ho
  obtain ⟨d, hd⟩ := verdictDigest_ok c
  have hout : outputsOf c true 3 = acceptedOutputs c := rfl
  have hv : verdict c = .ok ⟨true, acceptedOutputs c, d, true, some 3, true, true⟩ := by
    unfold verdict
    simp only [hr, proved_of hp, ↓reduceIte, hrow, certified_of hc, synced_of hs, acceptedOf, beq_self_eq_true,
      Bool.and_self, Option.getD_some, hout, hd, bind, Except.bind, pure, Except.pure]
  exact ⟨by rw [hv]; rfl, _, hv, rfl, hd⟩

/-- **An honest warden sampling block is accepted** (the honest lemma of `LinkWindowAcceptMeans` on a warden link): one
proved and opened whose two certificates the certifier's device signed under its key (a key `Lms.key` derives at
parameters `Lms.Params.make` admits, the link's), whatever syncs it declares. -/
theorem linkWindowAcceptMeans_honest_warden :
    ∀ (c : Case) (w : ByteArray) (N W H : Nat) (toy : Bool) (I seed : ByteArray) (k : Lms.PrivateKey),
      c.kind = .warden w → Lms.Params.make N W H toy = .ok c.lms → Lms.key c.lms I seed = .ok k →
      CertifierSigned c k → WindowProved c → WindowOpened c →
      (verdict c).map Verdict.accepted = .ok true ∧ WindowVerdict c := by
  intro c w N W H toy I seed k hk hmake hkey ⟨hpub, hsize, hall⟩ hp ho
  obtain ⟨-, hI, -, hIs, -⟩ := Properties.CertifierDevice.Proofs.Lms.key_spec hkey
  have hid : c.key.extract 0 16 = k.I := by
    rw [hpub, Lms.PrivateKey.publicKey]
    exact ByteArray.extract_append_eq_left (hI ▸ hIs).symm
  refine linkWindowAcceptMeans_honest c ⟨hp, ho, ⟨hsize, fun d hd => ?_⟩, by simp only [WindowSynced, hk]⟩
  obtain ⟨x, m, q, sig, hx, hl, hdir, hw, hxk, hm, hsig, hsx⟩ := hall d hd
  obtain ⟨hv, hdec⟩ := Properties.CertifierDevice.Proofs.Lms.lmsCorrect N W H toy c.lms I seed k q m sig hmake hkey hsig
  exact ⟨x, hx, hl, hdir, hw, hxk.trans hid.symm, m, sig, hm, hsx ▸ hdec, hpub ▸ hv⟩

/-! ## The nontrivial lemma -/

theorem linkWindowAcceptMeans_by : LinkWindowAcceptMeans ↔ LinkWindowAcceptMeansBy verdict := Iff.rfl

/-- A warden sampling block with no findings, on an empty grid: no verifier's acceptance of it means it was proved. -/
def unproved (w : ByteArray) : Case where
  link := ""
  run := ""
  window := 0
  root := .empty
  registration := ""
  units := #[]
  program := ""
  partition := ""
  gates := #[]
  findings := none
  statement := none
  saltPub := none
  saltOut := none
  output := none
  sessions := #[]
  certificates := #[]
  digests := #[]
  syncs := #[]
  deltas := #[]
  rhos := #[]
  lms := Lms.deployed
  key := .empty
  session := .empty
  egress := ⟨⟨0, 0, 0, 0, 0⟩, none⟩
  ingress := ⟨⟨0, 0, 0, 0, 0⟩, none⟩
  kind := .warden w

/-- **No verifier that accepts every sampling block has `LinkWindowAcceptMeans`' property** (its nontrivial lemma): such
a verifier accepts a warden sampling block with no findings (`unproved`), which isn't `WindowProved`. -/
theorem linkWindowAcceptMeans_nontrivial : ∀ v, AcceptsAll v → ¬ LinkWindowAcceptMeansBy v := by
  intro v hall h
  obtain ⟨-, fs, hf, -⟩ := (h (unproved .empty) (hall _)).1.1
  cases hf

end Properties.NetworkCertifier
