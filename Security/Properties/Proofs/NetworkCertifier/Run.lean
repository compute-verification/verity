import Properties.Proofs.NetworkCertifier.Outcomes

/-!
# An accepted run's rows are the Program's (5a)

Under LMS's unforgeability every accepted certificate's message is one the device signed, and under SHA-512's collision
resistance on the run's strings that message names the same link-window and digest (`messageInput_inj`), and the digest
the same head (`head_inj`). On egress that head is both the one `record_head` makes of the Program's grid and the one
the device made of its recorded grid, so the grids are one (`recordHead_grid_inj`), and so are their rows
(`bytesGrid_inj`).
-/

open NetworkCheck hiding Params
open NetworkCheck.Check
open Flock
open NetworkCertifier (ClockSync LogicalFrame expectedRow)

set_option maxRecDepth 20000

namespace Properties.NetworkCertifier

open Bytes Tree GridProofs CheckProofs

theorem digest_ok {h : RecordHead} {dg : ByteArray} (e : h.digest = .ok dg) :
    ∃ enc, h.encode = .ok enc ∧ dg = Sha512.hash (taggedInput recordTag enc) := by
  unfold RecordHead.digest at e
  cases he : h.encode with
  | error _ => rw [he] at e; exact (error_ne_ok e).elim
  | ok enc => rw [he] at e; exact ⟨enc, rfl, (Except.ok.inj e).symm⟩

theorem certificateMessage_ok {k : ByteArray} {l : String} {d : Direction} {w : Int} {g m : ByteArray}
    (e : certificateMessage k l d w g = .ok m) : ∃ x, messageInput k l d w g = .ok x ∧ m = Sha512.hash x := by
  unfold certificateMessage at e
  cases hx : messageInput k l d w g with
  | error _ => rw [hx] at e; exact (error_ne_ok e).elim
  | ok x => rw [hx] at e; exact ⟨x, rfl, (Except.ok.inj e).symm⟩

/-- Two heads with one digest, where SHA-512 has no collision on the strings both hash, are one. -/
theorem head_inj {X : ByteArray → Prop} (hX : CF X) {h h' : RecordHead} {dg : ByteArray}
    (hs : ∀ x, HeadString h x → X x) (hs' : ∀ x, HeadString h' x → X x) (e : h.digest = .ok dg)
    (e' : h'.digest = .ok dg) : h = h' := by
  obtain ⟨enc, he, rfl⟩ := digest_ok e
  obtain ⟨enc', he', hd⟩ := digest_ok e'
  have := taggedInput_inj (hX _ _ (hs _ ⟨enc, he, rfl⟩) (hs' _ ⟨enc', he', rfl⟩) hd)
  subst this
  exact encode_inj he he'

theorem lmsAccepts_of {p : Params} {m sig : ByteArray} (h : LmsVerifies p m sig) : lmsAccepts p m sig = true := by
  obtain ⟨s, hs, hv⟩ := h
  simp [lmsAccepts, hs, hv, Except.toOption]

theorem link_short {h : RecordHead} {dg : ByteArray} (e : h.digest = .ok dg) : h.link.toUTF8.size < 2 ^ 16 := by
  obtain ⟨enc, he, -⟩ := digest_ok e
  exact encode_link he

theorem zip_index {p : Params} {e : Evidence} (hfit : EvidenceFits p e) (ℓ : Fin p.egress.size) {link : Link}
    (hl : e.links[ℓ.1]? = some link) : (p.egress[ℓ], link) ∈ p.egress.toList.zip e.links.toList := by
  have h1 : ℓ.1 < e.links.size := by rw [hfit.1]; exact ℓ.2
  have hz : ℓ.1 < (p.egress.toList.zip e.links.toList).length := by
    rw [List.length_zip, Array.length_toList, Array.length_toList]; omega
  have := List.getElem_mem hz
  rw [List.getElem_zip] at this
  rw [Array.getElem?_eq_getElem h1, Option.some.injEq] at hl
  simpa [← hl] using this

theorem acceptedRowsAreProgramRows : AcceptedRowsAreProgramRows := by
  intro p c e dev eg ig hacc hpd hhon hcr hlms
  obtain ⟨hfit, eg', ig', hpd', he, hi⟩ := (acceptsIff p c e).mp hacc
  have heg : eg' = eg := Except.ok.inj (hpd'.1.symm.trans hpd.1)
  subst eg'
  obtain ⟨hh1, hh2, hh3⟩ := hhon
  -- every accepted certificate is of a head the device issued for its link-window
  have issued : ∀ d l w r dg, Signs p c d l w r dg →
      ∃ i ∈ dev.issued, i.direction = d ∧ i.link = l ∧ i.window = w ∧ i.head.digest = .ok dg := by
    intro d l w r dg ⟨hrec, hdig, hcdg, hk, hcl, hcd, hcw, m, hm, hσ⟩
    have hmem : r ∈ recordsOf c d l w := by rw [hrec]; exact List.mem_singleton_self r
    have hr : r ∈ c.records.toList := (List.mem_filter.mp hmem).1
    obtain ⟨i, hi, dg', hdg', hm'⟩ := hlms m r.certificate.signature (by
      simp only [runAccepts, lmsAccepts_of hσ, Bool.true_and, ← Array.any_toList, List.any_eq_true]
      exact ⟨r, hr, ba_beq.mpr rfl⟩)
    obtain ⟨x, hx, rfl⟩ := certificateMessage_ok hm
    obtain ⟨x', hx', hxx⟩ := certificateMessage_ok hm'
    have hX : RunStrings p c e dev eg x := Or.inl ⟨r, hr, Or.inr (by rw [hk, hcl, hcd, hcw, hcdg]; exact hx)⟩
    have hX' : RunStrings p c e dev eg x' := Or.inr (Or.inr (Or.inl ⟨i, hi, Or.inr ⟨dg', hdg', hx'⟩⟩))
    have := hcr _ _ hX hX' hxx
    subst this
    obtain ⟨hil, -, -, -⟩ := hh1 i hi
    obtain ⟨-, rfl, rfl, hw, rfl⟩ := messageInput_inj hx hx'
      (by rw [← (recordsOf_mem hmem).2.1]; exact link_short hdig)
      (by rw [← hil]; exact link_short hdg')
    exact ⟨i, hi, rfl, rfl, by exact_mod_cast hw.symm, hdg'⟩
  -- and the device gave it the record's head
  have same : ∀ d l w r dg, Signs p c d l w r dg →
      ∃ i ∈ dev.issued, i.direction = d ∧ i.link = l ∧ i.window = w ∧ i.head = r.head := by
    intro d l w r dg hs
    obtain ⟨i, hi, hd, hl, hw, hidg⟩ := issued d l w r dg hs
    have hmem : r ∈ recordsOf c d l w := by rw [hs.1]; exact List.mem_singleton_self r
    have hr : r ∈ c.records.toList := (List.mem_filter.mp hmem).1
    refine ⟨i, hi, hd, hl, hw, head_inj hcr (fun x hx => Or.inr (Or.inr (Or.inl ⟨i, hi, Or.inl hx⟩)))
      (fun x hx => Or.inl ⟨r, hr, Or.inl hx⟩) hidg hs.2.1⟩
  refine ⟨fun ℓ link hlink syncs hdecl t ht => ?_, fun d l hl w hw => ?_⟩
  · have hT : 0 < p.grid.buckets := Nat.pos_of_ne_zero fun h0 => by simp [h0] at ht
    have hw : t / p.grid.buckets < p.windows := (Nat.div_lt_iff_lt_mul hT).mpr ht
    have hm := zip_index hfit ℓ hlink
    have hlm : link ∈ e.links.toList := (List.of_mem_zip hm).2
    obtain ⟨r, dg, hset, -, hrec⟩ := he _ link hm _ hw
    obtain ⟨h, hh, hdg⟩ := hrec syncs hdecl
    obtain ⟨i, hi, hid, hil, hiw, hidg⟩ := issued _ _ _ r dg hset.1
    have hhi : h = i.head := head_inj hcr
      (fun x hx => Or.inr (Or.inl ⟨_, link, hm, _, hw, syncs, hdecl, Or.inr ⟨h, hh, hx⟩⟩))
      (fun x hx => Or.inr (Or.inr (Or.inl ⟨i, hi, Or.inl hx⟩))) hdg hidg
    obtain ⟨dl, hdev⟩ := hh2 i hi hid ℓ hil.symm
    rw [hil, hiw, ← hhi] at hdev
    obtain ⟨hgrid, -⟩ := recordHead_grid_inj hcr hh hdev
      (fun x hx => Or.inr (Or.inl ⟨_, link, hm, _, hw, syncs, hdecl, Or.inl hx⟩))
      (fun x hx => Or.inr (Or.inr (Or.inr ⟨ℓ, _, hw, hx⟩)))
    have hrows := bytesGrid_inj hgrid
      (fun t s f hf => by
        obtain ⟨k, rfl⟩ := expectedRow_slot hf
        rcases payloads_length (hfit.2.2 link hlm) k with h8 | h8
        · rw [h8]; omega
        · rw [h8]; rfl)
      (fun t s f hf => by rw [hh3 ℓ t s f hf]; omega)
      (t % p.grid.buckets) (Nat.mod_lt _ hT)
    rw [Nat.mul_comm, Nat.div_add_mod] at hrows
    exact hrows.symm
  · have hsettled : ∃ r dg, Settled p c d l w r dg := by
      cases d with
      | egress =>
        obtain ⟨link, hm⟩ := zip_left hfit hl
        obtain ⟨r, dg, hset, -⟩ := he l link hm w hw
        exact ⟨r, dg, hset⟩
      | ingress =>
        obtain ⟨r, dg, hset, -⟩ := hi l hl w hw
        exact ⟨r, dg, hset⟩
    obtain ⟨r, dg, hset⟩ := hsettled
    obtain ⟨i, hi, rfl, rfl, rfl, hih⟩ := same _ _ _ r dg hset.1
    rw [← (hh1 i hi).2.2.2, hih, hset.2.1]

end Properties.NetworkCertifier
